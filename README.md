# SLM Document Analysis (Proof of Concept)

M.Tech project: *"A Study of Parameter-Efficient Fine-Tuning of Small Language Models (SLMs) for Document Analysis."*

The user chooses a text-based PDF and types a question. **One request** carries both to this project's backend.
The backend extracts the PDF text **keeping its structure** (headings, lists, tables as Markdown) and sends that text
(`context`) and the question to the **model server's `POST /predict`**. The model server (a separate project that
hosts the fine-tuned Qwen1.5-0.5B + LoRA model) builds the prompt itself, runs the model and returns the answer. This
project only passes it on to the UI. There is **no chunking, no retrieval (no RAG) and no server-side storage**, and
nothing is trained here.

```
 Browser (Chrome)                 This project's backend                 Model project (separate)
 ┌──────────────────┐  PDF +      ┌──────────────────────────┐  JSON      ┌─────────────────────────┐
 │ React + Vite UI  │  question   │ FastAPI 127.0.0.1:8000   │ {context,  │ FastAPI 127.0.0.1:8765  │
 │ localhost:5173   │ ──────────► │ POST /answer             │  question} │ POST /predict           │
 │                  │             │ PDF → Markdown (PyMuPDF) │ ─────────► │ builds the prompt,      │
 │                  │ ◄────────── │                          │ ◄───────── │ runs Qwen + LoRA        │
 └──────────────────┘  answer     └──────────────────────────┘  answer    └─────────────────────────┘
```

## What is in this folder

| Path | Purpose |
|---|---|
| `frontend/` | React + TypeScript + Vite web UI |
| `slm-document-analysis-backend/` | The backend (PDF extraction + client for the model server), pytest tests, its own README with the API reference |
| `colab/` | **Legacy.** A Colab notebook that loads a model inside the backend. It is not used by the flow above and cannot reach a model server running on your computer |
| `notebooks/` | Your training notebook, kept as reference only. Nothing here depends on it |

## Run it (three terminals, all in WSL/Ubuntu)

**1. The model server** (from the model project, once cloned, see its `docs/run_locally.md`):

```bash
cd ~/my-project
bash scripts/start_server.sh
```

Wait for `Uvicorn running on http://127.0.0.1:8765`. If that port is busy the script picks the next one and prints it; then set `MODEL_SERVER_URL` below to match. Check: `curl http://127.0.0.1:8765/health`.

**2. This project's backend:**

```bash
cd ~/ARC/Project/slm-document-analysis-backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # MODEL_SERVER_URL defaults to http://127.0.0.1:8765
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check: `curl http://127.0.0.1:8000/health` should show `"model_ready": true` and the model name.
If the model server was started with `SERVING_API_KEY`, put the same value in `.env` as `MODEL_SERVER_API_KEY`.

**3. The frontend** (Node.js 20.19+ or 22.12+):

```bash
cd ~/ARC/Project/frontend
npm install
cp .env.example .env        # VITE_API_BASE_URL defaults to http://127.0.0.1:8000
npm run dev
```

Open <http://localhost:5173> in Chrome. The port is fixed (`strictPort`) because the backend's CORS allows exactly that origin.

Use the UI: **Step 1** choose a PDF, **Step 2** type a question and press **Get Answer**, then read the **Generated Answer**. The PDF is re-sent with every question because the server keeps nothing.

## What the model server does (so you know what this project does NOT do)

* This project does **not** build a prompt. It sends only `{"context": "<document as Markdown>", "question": "<question>"}`.
* The model server wraps them in its own prompt (`build_qa_prompt` in the model project; do not change it, the model was trained on that wording), cuts the context to what the model can read, generates greedily and returns `{"answer", "latency_ms", "context_truncated"}`.
* The model reads only about **430 tokens, roughly 1-2 pages**. For a longer document it reads the start only and says `context_truncated: true`; this project then shows a warning. If the answer is in the cut-off part the model may wrongly answer `unanswerable`.
* The model server does not report token counts or how much of the text it used, so this project does not show them.

## API summary

All errors: `{"error": {"code": "<machine_code>", "message": "<text>"}}`.

* `GET /health` → `{"status","model_ready","model_name","adapter_loaded","device","model_error"}`. `model_ready` is true only when the model server's `/health` answers.
* `POST /answer` (multipart: `file` = PDF, `question` = text) → `answer`, `unanswerable`, `message`, `model_name`, `inference_time_seconds`, `document` (page/heading/table/word counts), `document_truncated`, `structured_text` (the Markdown that was sent), `answer_locations`, `disclaimer`. `prompt_tokens`, `generated_tokens`, `document_chars_used` and `document_chars_total` are `null` because the model server does not report them.

Error codes: 400 `empty_file`; 413 `file_too_large` / `too_many_pages`; 415 `unsupported_file_type`; 422 `malformed_pdf` / `no_extractable_text` / `empty_question` / `question_too_long` / `validation_error`; 503 `model_not_ready` (model server not running); 504 `inference_timeout`; 500 `inference_failed` / `model_auth_failed` / `internal_error`.

### About `answer_locations`

After the answer returns, the backend searches the extracted document for the answer text (case/punctuation-insensitive, whole words) and reports the page, heading path and, for tables, the row and column labels. **This is plain string matching, not model output**. It is empty when the answer does not appear verbatim and always empty for `unanswerable`. (The UI does not display it any more; the API still returns it.)

## What was actually tested (and what was not)

Tested in Linux/WSL:

* Backend: 77 pytest tests pass, 3 are skipped (`requires_model`, for the legacy in-process model). 17 of the tests cover the model-server client against a fake server: the request body, the `X-API-Key` header, `unanswerable`, `context_truncated`, server down, timeout, bad responses and URL validation.
* **Against the real model server** (the model project's own `src.serving.server`, adapter `experiments/fin10k/adapter`, running on CPU), through this backend with a generated PDF:
  * "What was total revenue in fiscal 2023?" → `$4.2 billion` (about 2.6 s).
  * "Who is the chief executive officer?" → `unanswerable`.
  * A Markdown table PDF: "revenue of North" → `120 million dollars`; "profit of South" → `15 million` (the numbers are right, the unit wording was added by the model).
  * A 9-page PDF with the fact on the last page → `unanswerable` with `document_truncated: true`. This is the 430-token limit in action, not a bug.
  * Stopping the model server → `/health` shows `model_ready: false` and `/answer` returns 503 with a clear message.
* Frontend: 17 unit tests pass; type-check and production build pass.

**Not tested:** the rendered UI in a browser against this backend after the latest changes, a real multi-page report, large PDFs, `SERVING_API_KEY` against the real model server (the header is tested only against a fake server), answer quality beyond the few cases above, and macOS.

## Known limitations

* No OCR: scanned PDFs are rejected.
* Structure detection is heuristic (font size, bold, ruling lines). Tables without ruling lines are read as plain text; multi-column pages may be read in the wrong order.
* The model reads about 1-2 pages. Longer documents are cut at the end (see above).
* The model can invent details and can wrongly say `unanswerable`. No accuracy or confidence metrics are produced.
* No authentication on this backend; it listens on `127.0.0.1` only. One generation at a time on the model server.
