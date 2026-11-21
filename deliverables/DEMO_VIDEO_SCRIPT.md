# SWE015 demo video

[Official YouTube video](https://youtu.be/m2d_cP5pWOk) · [Offline MP4](SWE015_Demo.mp4)

The two-minute walkthrough uses female narration and shows the v1.0.1 app.

| Time | Screen | What the demo shows |
|---|---|---|
| 0:00–0:08 | Recurrence photo | An unboxed math image belongs in Full-page OCR. |
| 0:08–0:27 | Answer reader | Sample `test_162` yields EXPELLED. The app shows the raw read, character evidence and corrected word. |
| 0:27–0:43 | Digits only | The numeric example yields 2000. |
| 0:43–1:05 | Worksheet review | Review four answer crops, correct a suggestion and export a CSV. The pilot supports timed comparison with manual entry. |
| 1:05–1:18 | Strip-image warning | The app flags unusually narrow slices and transfers the original photo to Full-page OCR. |
| 1:18–1:32 | Full-page OCR | Local Qwen3-VL through Ollama transcribes the equation, including theta. |
| 1:32–1:47 | Course benchmark | Exact word accuracy on 300 generated held-out strips rises from 61.67% with the raw CNN to 87.67% with the default INT8 pipeline. |
| 1:47–2:00 | Public download | Get the release ZIP and follow the Windows or Linux setup instructions. |

Recompute the saved benchmark statistics with `python scripts/recompute_metrics.py`. Whole-page
OCR accuracy and classroom review time remain unmeasured.
