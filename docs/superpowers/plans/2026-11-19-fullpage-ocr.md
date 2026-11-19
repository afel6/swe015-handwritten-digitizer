# Full-page OCR Tab and Qwen Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional full-page OCR tab (Qwen3-VL through local Ollama) to the SWE015 app, a smoother two-tab UI for the demo video, and a Qwen reference benchmark on the 300 test worksheets, without changing any graded number.

**Architecture:** A standard-library Ollama adapter (`models/vlm_ocr.py`) is called by two new FastAPI routes behind the existing guards and its own lock. The static UI gains a second tab that shrinks photos in the browser before upload. A separate script (`scripts/benchmark_vlm.py`) runs Qwen on the same test strips in two modes and writes `results/vlm_reference.json`, which `recompute_metrics.py` checks and the README/slides report apart from the ladder.

**Tech Stack:** Python 3.12/3.13, FastAPI (existing pins), urllib, Pillow, vanilla JS/CSS, pytest, Ollama 0.40.16 with `qwen3-vl:8b-instruct-q4_K_M`.

**Spec:** `docs/superpowers/specs/2026-11-19-fullpage-ocr-design.md`

## Global Constraints

- No new Python dependency; `requirements.txt` unchanged.
- Ollama base URL is fixed: `http://127.0.0.1:11434`. Model `qwen3-vl:8b-instruct-q4_K_M`. temperature 0, seed 42, logprobs true, stream false.
- Ollama timeout 300 s; `/api/ocr` has its own lock, independent of `_RECOGNIZE_LOCK`.
- Existing guards apply unchanged (Host allow-list, JSON only, 8 MB body, no Transfer-Encoding); bad input gives a one-line 400; Ollama missing or failing gives a one-line 503.
- Browser downscales to at most 2048 px on the longest side, JPEG quality 0.9.
- CSP changes only `img-src` to `'self' data: blob:`.
- No change to Table 1, Table 2, the checks, `verify.py`, or any existing results file.
- Deck stays exactly 10 slides; README and AGENTS.md grader sections stay identical.
- Every test needing a real Ollama/model is skipped with a reason when it is absent (CI has none).
- Lint: `uvx ruff check . --select E4,E7,E9,F,I,RUF,ISC,UP --ignore E701,E702` adds no new finding.
- Conventional commits, no attribution line, push to `master` after each green task (standing OK).

## Review Focus

1. Ollama running but the model not pulled: status says `model_ready: false`; `/api/ocr` returns 503 naming `ollama pull qwen3-vl:8b-instruct-q4_K_M`. (Task 1 test `test_status_reports_missing_model`, Task 2 test `test_ocr_503_when_model_missing`.)
2. Ollama answers with HTTP 500, non-JSON, or JSON without `choices`: one-line 503, never a 500 traceback. (Task 1 test `test_transcribe_bad_reply_raises_unavailable`.)
3. A huge or non-image upload that bypasses the browser downscale: one-line 400 before Ollama is called. (Task 2 test `test_ocr_rejects_bad_images_without_calling_ollama`.)
4. A blank image: empty text comes back and the UI shows "(no text found)". (Task 1 test `test_summarise_empty_reply`; Task 3 browser check.)
5. An OCR request in flight must not block `/api/recognize`. (Task 2 test `test_recognize_not_blocked_by_ocr`.)

---

### Task 1: Ollama adapter

**Files:**
- Create: `models/vlm_ocr.py`
- Create: `tests/fake_ollama.py` (test helper: a local fake Ollama server)
- Test: `tests/test_vlm_ocr.py`

**Interfaces:**
- Produces:
  - `OLLAMA_URL: str = "http://127.0.0.1:11434"`, `MODEL: str`, `OCR_PROMPT: str` (the smoke-test prompt verbatim), `TIMEOUT_S = 300.0`, `MAX_TOKENS = 1024`, `PULL_COMMAND = "ollama pull qwen3-vl:8b-instruct-q4_K_M"`.
  - `class OcrUnavailable(RuntimeError)`: one-line, user-facing message.
  - `status(base_url: str | None = None, timeout_s: float = 3.0) -> dict` with keys `available: bool`, `model_ready: bool`, `detail: str`. Never raises.
  - `chat(image: bytes, prompt: str, *, base_url: str | None = None, timeout_s: float = TIMEOUT_S, max_tokens: int = MAX_TOKENS) -> dict`: the raw parsed reply; raises `OcrUnavailable` on connection error, timeout, non-200, non-JSON, or missing `choices[0].message`.
  - `summarise(reply: dict, elapsed_ms: float) -> dict` with keys `text: str`, `latency_ms: int`, `tokens: int`, `min_prob: float | None` (exp of the lowest logprob, 4 dp), `finish_reason: str | None`.
  - `transcribe(image: bytes, prompt: str = OCR_PROMPT, *, base_url: str | None = None) -> dict`: `summarise(chat(...))`.
  - `base_url=None` means read the module's `OLLAMA_URL` at call time (tests monkeypatch it).
  - `tests/fake_ollama.py`: `@contextmanager fake_ollama(reply: dict | bytes | None = REPLY_A, status: int = 200, models: list[str] = [MODEL]) -> Iterator[FakeOllama]`, where `FakeOllama.url: str` and `FakeOllama.requests: list[dict]` (parsed JSON bodies received). It serves `GET /api/tags` and `POST /v1/chat/completions`. `REPLY_A` is the raw line-a body from the 2026-11-19 smoke test.

- [ ] **Step 1: Write the failing tests** in `tests/test_vlm_ocr.py`:
  - `test_chat_sends_fixed_decoding_and_image`: one request; its `model == MODEL`, `temperature == 0`, `seed == 42`, `logprobs is True`, `stream is False`; content[0] url starts with `data:image/png;base64,iVBORw0KGgo`; content[1] text == `OCR_PROMPT`.
  - `test_transcribe_returns_text_and_confidence`: with `REPLY_A`, `text == "Invoice number: INV-2026-3706"`, `tokens == 14`, `min_prob == round(math.exp(-0.000002861027041944908), 4)`.
  - `test_summarise_empty_reply`: `{"choices":[{"message":{"content":""}}]}` gives `text == ""`, `tokens == 0`, `min_prob is None`.
  - `test_status_reports_missing_model`: fake with `models=[]` gives `{"available": True, "model_ready": False}` and `PULL_COMMAND in detail`.
  - `test_status_when_ollama_absent`: base_url pointing at a closed port gives `available is False`, and does not raise.
  - `test_transcribe_bad_reply_raises_unavailable`: parametrised over `status=500`, `reply=b"not json"`, `reply={}`; each raises `OcrUnavailable`.
  - `test_live_ollama_reads_invoice_line` (skip unless `status()["model_ready"]`): render "Invoice number: INV-2026-3706" with Pillow's default font at size 48 and assert "INV-2026-3706" in the text.
- [ ] **Step 2: Run** `.venv\Scripts\python -m pytest -q tests/test_vlm_ocr.py`. Expected: fails with an ImportError on `models.vlm_ocr`.
- [ ] **Step 3: Implement** `models/vlm_ocr.py` (urllib.request, JSON, base64; MIME from magic bytes, PNG or JPEG only, otherwise `UserInputError`) and `tests/fake_ollama.py` (a `ThreadingHTTPServer` on port 0 in a daemon thread).
- [ ] **Step 4: Run** the same command. Expected: all pass; the live test passes on this laptop.
- [ ] **Step 5: Commit** `feat(ocr): local Ollama adapter for Qwen3-VL transcription`.

### Task 2: OCR API routes

**Files:**
- Modify: `app.py` (CSP line 58, new request model, new lock, two routes)
- Test: `tests/test_app_ocr.py`

**Interfaces:**
- Consumes: `vlm_ocr.status`, `vlm_ocr.transcribe`, `vlm_ocr.OcrUnavailable`, `fake_ollama`.
- Produces:
  - `class OcrRequest(BaseModel)`: strict, `extra="forbid"`, field `image_base64: str`.
  - `_OCR_LOCK = threading.Lock()`.
  - `_decode_raw(b64: str) -> bytes`: same checks as `_decode_image` (length, plain base64, `load_validated_image_bytes`), but returns the validated raw bytes.
  - `GET /api/ocr/status` returns `vlm_ocr.status()`.
  - `POST /api/ocr` returns `vlm_ocr.transcribe(raw)`; `OcrUnavailable` becomes 503 `{"detail": str}`.
  - The validation error handler gives `/api/ocr` its own one-line message: "Send JSON with a text 'image_base64' (base64 PNG or JPEG)."

- [ ] **Step 1: Write the failing tests** in `tests/test_app_ocr.py`. Reuse the `client` fixture pattern from `tests/test_app.py` (module skip when weights are absent, as that file does).
  - `test_ocr_status_without_ollama`: monkeypatch `vlm_ocr.OLLAMA_URL` to a closed port; status 200 and `available is False`.
  - `test_ocr_returns_transcription`: with the fake serving `REPLY_A` and a tiny valid PNG, status 200 and `text == "Invoice number: INV-2026-3706"`.
  - `test_ocr_rejects_bad_images_without_calling_ollama`: parametrised over a `data:` prefix, not-base64, a GIF, an over-limit string (`"A" * (MAX_B64_CHARS + 4)`), and an extra field. Each gives 400, a one-line detail, and `fake.requests == []`.
  - `test_ocr_503_when_model_missing`: fake with `models=[]` returning status 404 on chat gives 503 with a one-line detail.
  - `test_recognize_not_blocked_by_ocr`: monkeypatch `vlm_ocr.transcribe` to wait on an Event; start `/api/ocr` in a thread; `/api/recognize` on a sample still answers 200 before the Event is set.
  - `test_csp_allows_blob_previews`: the CSP header contains `img-src 'self' data: blob:`.
- [ ] **Step 2: Run** `.venv\Scripts\python -m pytest -q tests/test_app_ocr.py`. Expected: fails with 404/405 on the new routes.
- [ ] **Step 3: Implement** the routes in `app.py` as plain `def` (threadpool), with `_OCR_LOCK` around `transcribe`.
- [ ] **Step 4: Run** the new file and `tests/test_app.py`. Expected: all pass; `python verify.py` still gives 4/4 PASS.
- [ ] **Step 5: Commit** `feat(app): /api/ocr and /api/ocr/status behind the existing guards`.

### Task 3: Two-tab UI, smoother for the video

**Files:**
- Modify: `static/index.html`, `static/app.css`, `static/app.js` (split the OCR code into `static/ocr.js` if `app.js` would pass 400 lines; then add a `/static/ocr.js` route in `app.py`)
- Test: `tests/test_static_ui.py`

**Interfaces:**
- Consumes: `GET /api/ocr/status`, `POST /api/ocr` (`{"image_base64"}`), and the existing `/api/recognize` and `/api/samples`.
- Produces:
  - Tabs with ids `tab-reader` and `tab-ocr`, and panels `panel-reader` and `panel-ocr` (ARIA `tablist`/`tab`/`tabpanel`; arrow keys switch tabs).
  - OCR panel: drop zone (click, drop, Ctrl+V), preview, Transcribe button, result box with `dir="auto"`, a meta line (seconds, lowest-token confidence %), Copy button, and a setup card shown when the status is not ready (install Ollama, then `ollama pull qwen3-vl:8b-instruct-q4_K_M`).
  - `downscale(file) -> Promise<string>` (base64 JPEG, longest side at most 2048 px, quality 0.9).

- [ ] **Step 1: Write the failing tests** (static content only):
  - `test_index_has_both_tabs`: `index.html` contains `id="tab-reader"`, `id="tab-ocr"`, `role="tablist"`.
  - `test_ocr_script_targets_api_and_downscales`: the served JS contains `/api/ocr`, `2048` and `0.9`, and has no `innerHTML =` (results use `textContent`).
  - `test_strip_mode_hint_explains_equal_boxes`: `index.html` contains "one word written in equal boxes".
- [ ] **Step 2: Run** `.venv\Scripts\python -m pytest -q tests/test_static_ui.py`. Expected: FAIL.
- [ ] **Step 3: Implement.** Smoothness rules from the spec:
  - Reserve the result area's height so nothing jumps.
  - Disable the button and show progress text while waiting ("Loading the model, about a minute the first time" when the first OCR call is still pending after 3 s).
  - Fade results in over 150 ms, and respect `prefers-reduced-motion`.
  - Use the same card, button and spacing styles on both tabs.
  - Keep visible focus rings.
  - Copy flashes "Copied".
- [ ] **Step 4: Run** the tests (PASS). Then check in the browser pane at http://localhost:8000:
  - Switch tabs.
  - Transcribe a canvas-made test line through the page (expect the exact text).
  - Upload a blank image (expect "(no text found)").
  - Stop Ollama, or point the status at a closed port with a test-only monkeypatch, and expect the setup card.
  - Recognize a sample on the reader tab (unchanged).
  - Take screenshots at desktop and 375 px widths.
- [ ] **Step 5: Commit** `feat(ui): full-page OCR tab and smoother two-tab layout`.

### Task 4: Qwen reference benchmark

**Files:**
- Create: `scripts/benchmark_vlm.py`
- Modify: `scripts/recompute_metrics.py` (reference checks)
- Create: `results/vlm_reference.json` (real run, this laptop)
- Test: `tests/test_benchmark_vlm.py`

**Interfaces:**
- Consumes: `vlm_ocr.chat`/`summarise`, `data.word_pipeline.item_to_strip`, `data.prepare_data.load_emnist_dataset` and `TEST_WORKSHEETS_PATH` (via `scripts.evaluate_all` as `retime_benchmark.py` does), and `recompute_metrics.row_metrics`/`normalize`.
- Produces:
  - `MODES = ("strip_only", "strip_sentence")`.
  - `build_prompt(mode: str, sentence: str) -> str`. Both prompts ask for the single word written in the boxes, in capitals, with no explanation. `strip_sentence` also shows the sentence with its blank and says the word fills it.
  - `run_mode(items: Sequence[Mapping], strips: Sequence[bytes], mode: str, ask: Callable[[bytes, str], dict]) -> list[dict]`. Each record has `item_id`, `target_word`, `predicted_word` (`normalize` of the reply text), `raw_reply`, `latency_ms`.
  - `build_report(runs: Mapping[str, list[dict]], meta: Mapping) -> dict`, shaped as `{"model", "digest", "ollama_version", "gpu", "settings", "n", "label": "Reference, not trained by us, GPU", "runs": {mode: {"word_acc_pct", "char_acc_pct", "latency_mean_ms", "records"}}}`.
  - `recompute_metrics.compare_reference(reference: Mapping) -> list[tuple[str, float, float, bool]]`. `main` gains `--reference` (default `results/vlm_reference.json`) and adds those checks only when the file exists.

- [ ] **Step 1: Write the failing tests:**
  - `test_prompts_differ_by_mode`: the `strip_sentence` prompt contains the sentence and the `strip_only` prompt does not.
  - `test_run_mode_normalises_and_scores`: a fake `ask` returns replies "hot", " Hot.", "HAT" for targets HOT, HOT, HOT. The predicted words are HOT, HOT, HAT; `build_report` gives `word_acc_pct == 200/3` and `char_acc_pct == 800/9`.
  - `test_compare_reference_matches_report`: build a report from those records; every `compare_reference` check is ok, and a tampered `word_acc_pct` fails.
  - `test_recompute_skips_missing_reference`: `main(["--reference", "missing.json"])` on the committed results gives exit 0 and the same check count as before.
  - `test_committed_reference_recomputes` (skip if the file is absent): every check is ok, `n == 300` for both modes, and `label` is as above.
- [ ] **Step 2: Run** `.venv\Scripts\python -m pytest -q tests/test_benchmark_vlm.py`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** the script with `--modes`, `--limit` (smoke) and `--out`, and an exit code of 1 with a one-line error when Ollama is unavailable. Record the Ollama version from `/api/version`, the model digest from `/api/tags`, and the GPU name from `nvidia-smi --query-gpu=name --format=csv,noheader`.
- [ ] **Step 4: Run** the tests (PASS except the committed-file test, which skips). Then:
  - Run the smoke `python scripts/benchmark_vlm.py --limit 5`.
  - Then run the full `python scripts/benchmark_vlm.py` in the background (about 15 minutes). Run nothing heavy alongside it.
  - Run `python scripts/recompute_metrics.py`. Expected: all PASS, with the old 35 plus 6 new.
  - Run the full test file again. Expected: PASS.
- [ ] **Step 5: Commit** `feat(bench): Qwen3-VL reference on the 300 test worksheets (two modes)` with the JSON.

### Task 5: README, slides, final checks, push

**Files:**
- Modify: `README.md`, `AGENTS.md` (grader-section sync only if it changes), `scripts/slide_content.py` (`load_results` adds `vlm_reference` when present; new `reference_line(res) -> str`), `scripts/generate_slides.py` (slides 2, 8, 9), `deliverables/SWE015_Presentation.{pdf,pptx}`
- Test: `tests/test_grader_tools.py`, `tests/test_slides.py`

**Interfaces:**
- Consumes: `results/vlm_reference.json` (Task 4).
- Produces: `slide_content.reference_line(res) -> str`, shaped as "Reference (not trained by us, <gpu> GPU): Qwen3-VL 8B <a> % strip only, <b> % with sentence, <ms> ms/word". Every value is read from the JSON.

- [ ] **Step 1: Write the failing tests:**
  - `test_readme_explains_two_parts_and_reference`: the README contains "## Two parts", "Full-page OCR", "not trained by us", and both reference word accuracies formatted `.2f` from the JSON.
  - `test_reference_line_reads_json`: the line contains both `.2f` accuracies and "not trained by us".
  - `test_slide_8_shows_reference` (in `tests/test_slides.py`, under its existing skip): page index 7 contains the reference line's accuracies; the deck still has 10 pages.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement** the following:
  - README "Two parts" section near the top.
  - Optional OCR setup (install Ollama, `ollama pull ...`); the README notes OCR-only start as a next step.
  - The reference line under Table 2.
  - Slide 2: the split, in one line. Slide 8: the reference line. Slide 9: the real-world tab as a win, and plain text only / no field extraction as a next step.
  - Regenerate the deck with `--name "Abdalraouf Alkabir" --student-id 2309015482` (video still TBD).
- [ ] **Step 4: Verify:**
  - Full `pytest -q` (all pass, or skip with reasons).
  - `python verify.py` (4/4).
  - `python scripts/recompute_metrics.py` (all PASS).
  - Lint adds no new finding.
  - `git diff results/benchmark_results.json results/test_predictions.json` is empty.
  - A code-review pass over the whole branch diff.
- [ ] **Step 5: Commit** `docs: two-part README, reference row and regenerated slides`. Push, watch CI until all 6 cells are green, and report to the planner (commit hashes, CI run id). Tell the folio-ocr session that the full-page OCR tool now lives in SWE015 (spec "Coordination").
