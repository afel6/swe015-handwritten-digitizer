# Demo video script (2:00 or less)

Setup before recording: `python app.py` is running; the browser shows http://127.0.0.1:8000; the
README architecture diagram and `results/benchmark_results.md` are open in a second tab.

| Time | Screen | Narration |
|---|---|---|
| 0:00–0:12 | App page, empty | "This is the Handwritten Answer Digitizer. A student writes the missing word of a sentence into boxes; the app reads each box and uses the sentence to correct the word. Everything runs locally on the CPU." |
| 0:12–0:35 | Draw tab: sentence "She [MASK] the ball hard.", 3 boxes, draw H, I, T; click Recognize | "I type a cloze sentence and draw the answer, one character per box." |
| 0:35–0:55 | Result: CNN read **H1T**, corrected **HIT**; point at box 2 badges and Grad-CAM | "The CNN reads the second box as a one, not an I: a classic 1/I confusion. The badges show its top-2 guesses, and Grad-CAM shows the strokes it looked at. The n-gram model and BERT both use the sentence and correct the word to HIT." |
| 0:55–1:10 | Four-model table, app row highlighted | "Side by side: the raw CNN, the n-gram corrector, full-precision BERT, and the quantized app model: an INT8 CNN plus INT8 BERT, which is the one the app uses." |
| 1:10–1:22 | Sample selector: pick a numeric sample, Recognize | "Samples come from the 300 held-out worksheets. Numbers are the hard case: when the CNN mixes letters into digits, the correction cannot recover them; the slides show this failure." |
| 1:22–1:40 | README architecture diagram | "The pipeline: validation, box slicing and 28x28 preprocessing, three character models, the 47-to-36 class fold, a SymSpell candidate pool, then the n-gram or BERT re-ranker with dev-tuned weights and a numeric guard." |
| 1:40–2:00 | `results/benchmark_results.md` Table 2 | "On the 300 held-out worksheets, correction lifts word accuracy from about 62 to 88 percent, and the INT8 app model stays within a point of full-precision BERT at under half its size. Every number can be recomputed with python verify.py." |
