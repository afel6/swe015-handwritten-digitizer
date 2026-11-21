"""Re-run digits-only recognition on the generated numeric benchmark strips.

From the repository root, run python scripts/evaluate_digits.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if str(ROOT := Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data._io_errors import describe_cli_os_error
from data.prepare_data import EMNIST_ROOT, TEST_WORKSHEETS_PATH, load_emnist_dataset, load_worksheet_items, read_data, write_data
from data.word_pipeline import item_to_box_probs
from models.quantization import load_quantized_cnn
from models.recognizer import read_digits

DEFAULT_PREDICTIONS = Path("results/test_predictions.json")
CNN_CHECKPOINT = Path("weights/char_cnn_int8.pt")


def evaluate() -> dict[str, Any]:
    required_keys = ("id", "target_word", "difficulty", "emnist_split", "emnist_indices")
    items = load_worksheet_items(TEST_WORKSHEETS_PATH, required=required_keys)
    numeric_items = [item for item in items if item["difficulty"] == "numeric"]
    if not numeric_items:
        raise ValueError("No numeric items in the worksheet fixture.")

    default_rows = read_data(DEFAULT_PREDICTIONS, as_json=True)["rows"]["quantized_app"]
    default_by_id = {row["item_id"]: row for row in default_rows}
    if len(default_by_id) != len(default_rows):
        raise ValueError("The committed default predictions contain duplicate worksheet IDs.")
    if len({item["id"] for item in numeric_items}) != len(numeric_items):
        raise ValueError("The numeric worksheet fixture contains duplicate IDs.")

    torch.set_num_threads(4)
    model = load_quantized_cnn(CNN_CHECKPOINT)
    dataset = load_emnist_dataset(EMNIST_ROOT, train=False)
    records = []
    for item in numeric_items:
        item_id = item["id"]
        target = item["target_word"]
        if item["emnist_split"] != "test" or not (isinstance(target, str) and target.isascii() and target.isdigit()):
            raise ValueError(f"{item_id} must contain only digits from the EMNIST test split.")
        default = default_by_id.get(item_id)
        if default is None or default["target_word"] != target:
            raise ValueError(f"The committed default predictions have no matching target for {item_id}.")
        records.append({
            "item_id": item_id,
            "target_word": target,
            "num_boxes": len(target),
            "default_prediction": default["predicted_word"],
            "digits_only_prediction": read_digits(item_to_box_probs(item, dataset, model)),
        })

    return {
        "scope": (
            "Post-fix controlled regression on generated EMNIST test strips. "
            "The numeric field type and character-box count are supplied by the worksheet fixture. "
            "This supplemental check leaves the original default benchmark unchanged."
        ),
        "worksheets": TEST_WORKSHEETS_PATH.as_posix(),
        "default_prediction_source": DEFAULT_PREDICTIONS.as_posix(),
        "cnn_checkpoint": CNN_CHECKPOINT.as_posix(),
        "items": len(records),
        "default_correct": sum(row["default_prediction"] == row["target_word"] for row in records),
        "digits_only_correct": sum(row["digits_only_prediction"] == row["target_word"] for row in records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout.")
    args = parser.parse_args()
    try:
        result = evaluate()
        if args.output is None:
            print(json.dumps(result, indent=2))
        else:
            write_data(args.output, result, as_json=True)
    except OSError as exc:
        parser.exit(1, f"{describe_cli_os_error(exc)}\n")
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, f"Error: {' '.join(str(exc).split())}\n")


if __name__ == "__main__":
    main()
