# SWE015 repository walkthrough

The [submission video is on YouTube](https://youtu.be/QUfX1MxBz0o).

The timeline below describes the supplemental [repository walkthrough](SWE015_Demo.mp4).
It runs for under two minutes. The worksheet segment uses captions, and the final segment shows
Full-page OCR through local Ollama.

| Time | Screen | What the demo shows |
|---|---|---|
| 0:00–0:12 | Answer reader result | The model uses a sentence to recover HIT from an ambiguous handwritten character. |
| 0:12–0:24 | Draw tab | Enter a sentence with a blank. Draw one character per box and run recognition. |
| 0:24–0:32 | Strip image tab | Upload a cropped answer strip and enter its box count, or load a sample. |
| 0:32–0:49 | Per-box evidence and model results | Inspect the raw and corrected words, per-character guesses, and Grad-CAM overlays. |
| 0:49–1:10 | Model pipeline | Slice and normalize the boxes. Score characters with a CNN. Generate candidates with SymSpell and rank them with the n-gram or BERT model. |
| 1:10–1:25 | Benchmark | The raw FP32 CNN scored 61.67% and the default INT8 pipeline scored 87.67% on 300 generated held-out strips. The INT8 pipeline averaged 49.9 ms/word in the offline CPU benchmark. |
| 1:25–1:32 | Worksheet upload and demo page | Load the four-question demo or upload an aligned image of the supplied template. |
| 1:32–1:43 | Review table and CSV export | Inspect each crop and suggestion. The template uses digits-only mode for numeric fields. Correct 1076 to 1876 and export the reviewed answers. Classroom accuracy and time savings remain unmeasured. |
| 1:43–2:00 | Full-page OCR | Upload or paste a page image, run local Qwen3-VL through Ollama, and copy the returned text. |

Recompute the saved benchmark statistics with `python scripts/recompute_metrics.py`. Run the local
app with `python app.py` and open [the worksheet reader](http://127.0.0.1:8000/worksheet).
