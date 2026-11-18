"""Recognition service for the app: loads every model once (CPU) and runs the benchmarked ladder per request.

Box images go through data.word_pipeline (boxes_to_tensors, batch_to_probs), the same helpers the benchmark
uses, and the correction rows come from models.ladder.run_ladder, so the app runs the measured pipeline.
"""

from __future__ import annotations

import base64
import io
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from PIL import Image
from transformers import BertForMaskedLM

from data.prepare_data import read_data
from data.preprocess import CANONICAL_36_CHARS
from data.word_pipeline import batch_to_probs, boxes_to_tensors
from models.assets import BERT_INT8_FILE
from models.bert_reranker import BertMLMReranker
from models.char_models import load_char_cnn_fp32, load_dense_mlp
from models.classical_nlp import ClassicalNLPCorrector
from models.errors import UserInputError
from models.gradcam import generate_gradcam_b64
from models.ladder import run_ladder, timed
from models.quantization import (
    load_quantized_bert_dynamic,
    load_quantized_cnn,
    save_quantized_state_dict,
)

_ROWS = (("raw_fp32_cnn", "raw_fp32", "FP32 CNN, no correction"), ("classical_ngram", "classical", "FP32 CNN + n-gram (FP32)"),
         ("bert_fp32", "bert_fp32", "FP32 CNN + FP32 BERT"), ("quantized_app", "quantized_app", "INT8 CNN + INT8 BERT"))
PRIMARY_MODEL = "quantized_app"

__all__ = ["PRIMARY_MODEL", "RecognitionService"]


def _png_b64(arr_0_1: np.ndarray) -> str:
    img = Image.fromarray(np.clip(np.rint(arr_0_1 * 255.0), 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _prediction(row: np.ndarray) -> dict[str, Any]:
    """Top-1 character, its probability and the top-2 list for one folded probability row."""
    order = np.argsort(-row, kind="stable")[:2]
    top2 = [{"char": CANONICAL_36_CHARS[int(i)], "prob": float(row[int(i)])} for i in order]
    return {"char": top2[0]["char"], "prob": top2[0]["prob"], "top2": top2}


class RecognitionService:
    """Holds Dense MLP, FP32/INT8 CNNs, the n-gram corrector and FP32/INT8 BERT; all on CPU, loaded once."""

    def __init__(self, weights_dir: Path, results_dir: Path) -> None:
        self.dense = load_dense_mlp(weights_dir / "dense_mlp.pt")
        self.cnn_fp32 = load_char_cnn_fp32(weights_dir / "char_cnn_fp32.pt")
        self.cnn_int8 = load_quantized_cnn(weights_dir / "char_cnn_int8.pt")
        self.ngram = ClassicalNLPCorrector.load(weights_dir / "ngram_lm.pkl")
        self.params = read_data(results_dir / "bert_fusion_params.json", as_json=True)["params"]
        self.bert_fp32 = BertMLMReranker()
        self.bert_int8 = self._load_int8_bert(weights_dir / BERT_INT8_FILE)

    def _load_int8_bert(self, path: Path) -> BertMLMReranker:
        if path.is_file():
            config = self.bert_fp32.model.config
            model = load_quantized_bert_dynamic(lambda: BertForMaskedLM(config), path)
            return BertMLMReranker(self.bert_fp32.model_name, model=model, tokenizer=self.bert_fp32.tokenizer)
        int8 = self.bert_fp32.to_int8()
        save_quantized_state_dict(int8.model, path, metadata={"base_model": self.bert_fp32.model_name, "quantization": "dynamic_int8_linear"})
        return int8

    def _box_payload(self, i: int, tensor: torch.Tensor, empty: bool, probs: dict[str, np.ndarray]) -> dict[str, Any]:
        preds = {name: _prediction(probs[name][i]) for name in probs}
        gradcam = None if empty else generate_gradcam_b64(self.cnn_fp32, tensor.unsqueeze(0))
        return {"index": i, "empty": empty, "preview_png_b64": _png_b64(tensor[0].numpy()), "top2": preds["cnn_fp32"]["top2"],
                "dense_pred": preds["dense"], "cnn_fp32_pred": preds["cnn_fp32"], "cnn_int8_pred": preds["cnn_int8"], "gradcam_png_b64": gradcam}

    def recognize(self, sentence: str, box_images: Sequence[Image.Image]) -> dict[str, Any]:
        """Preprocess boxes, run all character models, then the shared correction ladder on the non-empty boxes."""
        batch, empty = boxes_to_tensors(box_images)
        keep = [i for i, flag in enumerate(empty) if not flag]
        if not keep:
            raise UserInputError("All boxes are empty; write at least one character.")
        probs32, t32 = timed(lambda: batch_to_probs(batch[keep], self.cnn_fp32))
        probs8, t8 = timed(lambda: batch_to_probs(batch[keep], self.cnn_int8))
        display = {"dense": batch_to_probs(batch, self.dense), "cnn_fp32": batch_to_probs(batch, self.cnn_fp32), "cnn_int8": batch_to_probs(batch, self.cnn_int8)}
        models = {"ngram": self.ngram, "bert_fp32": self.bert_fp32, "bert_int8": self.bert_int8, "params": self.params}
        ladder = run_ladder(sentence, probs32, probs8, models, {"fp32": t32, "int8": t8})
        rows = [{"name": name, "precision": precision, **ladder["rows"][key]} for name, key, precision in _ROWS]
        boxes = [self._box_payload(i, batch[i], empty[i], display) for i in range(len(empty))]
        return {"request_id": uuid.uuid4().hex, "boxes": boxes, "models": rows, "primary_model": PRIMARY_MODEL}
