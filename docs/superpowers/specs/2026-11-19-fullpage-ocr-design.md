# Full-page OCR tab and Qwen reference benchmark: design

Date: 2026-11-19. Deadline for the graded submission: 2026-11-22. Status: draft for the user's approval, then the planner's slicing.

## Intent

SWE015 today proves the course fundamentals on one narrow job: it reads a single answer word written one letter per box, using a CNN we trained, then corrects it with an n-gram model and BERT, and it quantizes the models. That is graded and stays as it is.

The user also wants SWE015 to be useful as a real OCR tool: anyone can run it on their own laptop, give it a photo or scan of handwriting or a document, and get the text back. Nothing should leave their machine. The pitch to the grader is the split: "we understand the fundamentals (graded pipeline), and we can also solve a real problem companies have (full-page OCR)". A demo video will be recorded, so the UI must feel smooth on screen.

## Goals

1. A second tab in the existing app, "Full-page OCR": upload, drop or paste an image, and get the transcription from Qwen3-VL 8B running locally through Ollama.
2. A "Qwen3-VL reference" result: the same 300 test worksheets read by Qwen, so our trained models can be compared with a large off-the-shelf model.
3. A smoother UI across both tabs, good enough to record.
4. README, slides and grader notes explain the split.

## Non-goals

- Picking out specific fields (invoice number, totals). Plain text only for now.
- Multi-page PDFs, Tesseract fallback, a hosted website, accounts.
- Any change to the graded pipeline, its tuning or its reported numbers.

## Anti-goals (failure even if it works)

- The grader's path breaks or slows down when Ollama is not installed (the grader will not have it, nor CI).
- Any existing number in Table 1, Table 2, the checks or `recompute_metrics.py` changes.
- An image or text leaves the laptop. Ollama is reached only at `http://127.0.0.1:11434`.
- The Qwen row reads as "our model"; it must read as an outside reference.

## Constraints

- Repo, framework and pinned versions stay (FastAPI app, `app.py`, `static/`, `requirements.txt`). No new Python dependency; the Ollama client uses the standard library.
- Existing app guards apply to the new endpoint: Host allow-list, JSON only, size cap, no Transfer-Encoding, one-line 400s via `UserInputError`.
- The deck stays exactly 10 slides (a test enforces it); new content goes into existing slides.
- The README and AGENTS.md grader sections stay identical (a test enforces it).
- CI cells have no Ollama: every Qwen-dependent test is skipped there with a reason, and asset-free tests still pass.
- Three days: finish code by 2026-11-21 so the video and the final checks fit before 2026-11-22.

## Approaches considered

1. Grow SWE015 into the OCR tool after submission: safest for the grade, but nothing to show in the video.
2. **Chosen by the user:** add an optional full-page OCR tab now, beside the untouched graded pipeline, plus a Qwen reference benchmark.
3. Replace the CNN reader with Qwen now: rejected, it removes what the grade is based on.

## Design

### Model adapter (`models/vlm_ocr.py`, new)

- Fixed endpoint `http://127.0.0.1:11434`, model `qwen3-vl:8b-instruct-q4_K_M`, temperature 0, seed 42, logprobs on, the transcription prompt used in the folio-ocr smoke test.
- `status()`: is Ollama reachable, and is the model pulled (via `/api/tags`)? Returns a small dict; never raises.
- `transcribe(image_bytes)`: validates the image with the app's existing image rules, calls `/v1/chat/completions`, returns text, latency, token count and the lowest token probability.
- Errors: bad image becomes `UserInputError` (400); Ollama missing or failing becomes a new `OcrUnavailable` (503) with a one-line message telling the user how to install it.

### App (`app.py`)

- `GET /api/ocr/status` and `POST /api/ocr` (body: `{"image_base64": ...}`), with the same guards as `/api/recognize`.
- The models for the graded tab load exactly as now; the OCR tab loads nothing at startup.

### UI (`static/`)

- Two tabs at the top: "Answer reader (course models)" and "Full-page OCR (Qwen3-VL, local)".
- The OCR tab: a drop zone (click, drop or paste), an image preview, a Transcribe button, and the result with time, a confidence hint, and a Copy button. Right-to-left text displays correctly.
- If Ollama is missing, the tab shows a friendly setup card (install Ollama, then `ollama pull ...`) instead of failing.
- "Smoother" for the video (builder's proposal, the user can correct): no layout jumps while loading; a progress state with a short message ("Loading the model, about a minute the first time"); short fade-in of results; consistent spacing and buttons across both tabs; keyboard and screen-reader friendly. The strip-image mode gets a clearer hint that it reads one word in equal boxes, which is what confused the first manual test.

### Qwen reference benchmark (`scripts/benchmark_vlm.py`, new)

- Reads the same 300 test worksheets as the ladder: for each, the boxed strip image and the sentence with its blank. Qwen is asked for the one answer word. It sees no labels, so the test set gets no tuning.
- Scores word accuracy and character accuracy with the ladder's own scoring functions.
- Writes `results/vlm_reference.json`: per-item predictions, the two accuracies, mean GPU ms per word, model name and digest, Ollama version, GPU name, settings.
- `recompute_metrics.py` recomputes the two accuracies from the per-item predictions (no Ollama needed). `verify.py` is unchanged.
- Reported apart from the ladder: a separate "Reference (not trained by us, GPU)" line under Table 2 in the README and on slide 8, with no pass/fail check.

### Docs and slides

- README: a short "Two parts" section near the top, the OCR tab's setup steps (optional), and the reference row.
- Slides (still 10): slide 1 or 2 names the split; slide 8 carries the reference row; slide 9 mentions the real-world tab as a win and the plain-text limit as a next step.

## Tests

- Adapter: against a tiny fake Ollama served on a local port in the test (no real model): request shape, parsing, lowest-probability maths, 503 when unreachable, 400 for bad images.
- App: `/api/ocr` guards (host, content type, size, bad base64) return one-line 400s without touching Ollama; status endpoint works with Ollama absent.
- Benchmark: scoring on a fixed small prediction file; recompute matches the committed JSON.
- Live tests (real Ollama and model) are skipped unless the model is present.
- Existing suite, `verify.py` and CI stay green; all reported numbers unchanged.

## Decisions after the builder check (builder's calls, the user can correct)

1. **Benchmark input, two runs.** Qwen reads the 300 strips twice: (a) the strip image alone, compared with row (1) as a pure vision reference; (b) the strip plus the sentence with its blank, compared with rows (3) and (4) as a context-aware reference. Both are labelled "reference, not trained by us, GPU". Each run is about 300 calls at about 1 s on this laptop's RTX 5070 Ti, so both fit in about 15 minutes.
2. **Real results, committed before the deadline.** The runs happen on this laptop and `results/vlm_reference.json` is committed with real numbers. The code tests use a small fixture file; the recompute check uses the committed JSON once it exists.
3. **The OCR tab still needs the graded weights to start the app.** `build_service()` and startup stay untouched near the deadline. The README notes that OCR-only start is a next step.
4. **Large photos are shrunk in the browser.** Before upload, the page downsizes the image to at most 2048 px on the longest side (JPEG, quality 0.9), so the existing 8 MB body cap and image rules hold. The server keeps the same limits, and any photo a phone takes works.
5. **Model tag and logprobs are confirmed.** `qwen3-vl:8b-instruct-q4_K_M` (digest 70e58e37beb1) is pulled, and `/v1/chat/completions` returned `choices[0].logprobs.content` in the 2026-11-19 smoke test. If a reply lacks logprobs, the confidence hint is hidden.
6. **Concurrency and timeouts.** `/api/ocr` has its own lock (one OCR at a time, independent of `/api/recognize`). Ollama gets a 300 s timeout, enough for the first model load; the UI shows a progress state the whole time.
7. **CSP.** `img-src` adds `blob:` for the preview. The Copy button uses `navigator.clipboard.writeText`, which needs no CSP change. Nothing else loosens.

## Left to the builder

File-level layout inside `static/`, exact wording of UI text, CSS details, and the exact Qwen prompt for the benchmark (it must ask for one word and forbid explanations).

## Coordination

The planner session owns SWE015's scope and slices. This design goes to it for slicing and review before code. The folio-ocr session is told that the full-page OCR tool now lives in SWE015.
