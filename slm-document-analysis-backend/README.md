# SLM Document Analysis Backend

FastAPI backend for the M.Tech project *"A Study of Parameter-Efficient Fine-Tuning of Small
Language Models (SLMs) for Document Analysis"*.

The browser sends **one request containing a PDF and a question**. The backend extracts the PDF's text
**with its structure kept** (headings, lists, tables as Markdown) and sends that text (`context`) and the
question to the **model server's `POST /predict`**. The model server (a separate project hosting the
fine-tuned Qwen1.5-0.5B + LoRA model) builds the prompt itself, runs the model and returns the answer,
which this backend passes on. There is **no chunking, no retrieval/RAG, no server-side storage**, and
nothing is trained here.

```
Frontend --(PDF + question, one multipart request)--> POST /answer
   -> structure-preserving extraction (PyMuPDF)  -> Markdown
   -> POST {"context": <Markdown>, "question": <question>}  to  MODEL_SERVER_URL/predict
   -> {"answer", "latency_ms", "context_truncated"}  from the model server
   -> JSON response
```

## Project structure

```
slm-document-analysis-backend/
├── app/
│   ├── main.py                  # FastAPI app: GET /health, POST /answer, error handlers, CORS
│   ├── config.py                # Settings read from environment variables / .env
│   ├── errors.py                # ApiError (uniform error responses)
│   ├── schemas.py               # Pydantic response models
│   ├── structure.py             # DocumentElement model, Markdown rendering, locate_answer()
│   ├── document_processor.py    # PDF validation + structure extraction (PyMuPDF heuristics)
│   ├── remote_model_service.py  # Client for the model server (/health, /predict)  <- used by the app
│   ├── model_service.py         # Result type + error types; also a legacy in-process model (unused)
│   └── prompting.py             # Legacy prompt template (unused: the model server builds the prompt)
├── tests/
│   ├── conftest.py              # PDF generators + fake model service
│   ├── test_document_processor.py
│   ├── test_structure.py        # Markdown, locate_answer
│   ├── test_api.py
│   ├── test_remote_model_service.py   # model-server client vs a fake model server
│   └── test_model_integration.py      # legacy in-process model; needs a real model; skipped by default
├── scripts/smoke_test.sh        # manual end-to-end check with curl
├── .env.example  .gitignore  pytest.ini  requirements.txt  README.md
```

## 1. Set up

Requires Python 3.11 or newer.

```bash
cd slm-document-analysis-backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## 2. Configure the model server

Edit `.env`:

| Variable | Meaning |
|---|---|
| `MODEL_SERVER_URL` | Where the model server listens. Default `http://127.0.0.1:8765`. Plain `http` is accepted only for localhost; any other host must be `https`. |
| `MODEL_SERVER_API_KEY` | Only if the model server was started with `SERVING_API_KEY`. Sent as the `X-API-Key` header, never logged. Leave empty otherwise. |
| `GENERATION_TIMEOUT_SECONDS` | How long to wait for an answer (default 120). |

Start the model server first (in the model project: `bash scripts/start_server.sh`, see its `docs/run_locally.md`).
If it is not running, `/health` returns `model_ready: false` with the reason and `/answer` returns 503
`model_not_ready`. This backend never substitutes another model.

### What this backend sends

Exactly `{"context": "<document as Markdown>", "question": "<question>"}`. It does **not** build a prompt:
the model server wraps the text in the prompt the model was trained on. Do not add instructions to the context.

### What "structure-preserving" means

| PDF content | Markdown sent as `context` |
|---|---|
| Title / headings | `#`, `##`, `###`, `####` (from font size/bold and section numbers like `1.1`) |
| Ruled tables (`page.find_tables()`) | `\| col \| col \|` rows with a `\| --- \|` separator |
| Bullets / numbered items | `- item` / `1. item` |
| Body text | paragraphs separated by blank lines (wrapped lines joined) |
| Page numbers, repeated headers/footers | removed |

Heuristics, not magic: headings are recognised by size/bold, so documents that style headings
differently may be mis-classified; tables **without ruling lines** are read as ordinary text; multi-column
pages may be read in the wrong order; **no OCR** (scanned PDFs are rejected).

### Long documents

The whole extracted document is sent (up to the model server's limit of 200,000 characters). The model
itself reads only about **430 tokens, roughly 1-2 pages**; the model server cuts the rest and answers
`context_truncated: true`, which this backend returns as `document_truncated: true` with a warning message.
If the answer is in the cut-off part, the model may wrongly answer `unanswerable`.

## 3. Run

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive docs: <http://127.0.0.1:8000/docs>. Quick manual check:

```bash
scripts/smoke_test.sh path/to/file.pdf "What is this document about?"
```

## 4. Run the tests

```bash
pytest        # unit + API tests; no model needed
```

The API tests use a **fake model service**, and `test_remote_model_service.py` uses a **fake model server**
on `127.0.0.1`. They check HTTP behaviour and the request sent to `/predict`, not real model output.
`pytest -m requires_model` (with `MODEL_NAME_OR_PATH` set) exercises the legacy in-process model only.

## 5. API reference

All errors use one shape: `{"error": {"code": "<machine_code>", "message": "<text>"}}`.

### `GET /health`
```json
{"status":"ok","model_ready":true,"model_name":"Qwen/Qwen1.5-0.5B",
 "adapter_loaded":false,"device":null,"model_error":null}
```
`model_ready` is true when the model server's `GET /health` answers. `adapter_loaded` and `device` are not
reported by the model server, so they are `false` / `null` (not claimed).

### `POST /answer` (multipart/form-data: `file` = PDF, `question` = text)

```bash
curl -F "file=@report.pdf;type=application/pdf" -F "question=What was the profit of the North region?" \
  http://127.0.0.1:8000/answer
```

Response 200 (illustrative values; every field is computed per request):
```json
{
  "question": "What was the profit of the North region?",
  "answer": "30",
  "unanswerable": false,
  "message": null,
  "model_name": "Qwen/Qwen1.5-0.5B",
  "inference_time_seconds": 1.25,
  "prompt_tokens": null,
  "generated_tokens": null,
  "document": {"filename":"report.pdf","page_count":2,"heading_count":4,"table_count":1,
               "character_count":388,"word_count":70,"empty_page_numbers":[]},
  "document_truncated": false,
  "document_chars_used": null,
  "document_chars_total": null,
  "structured_text": "# Annual Report 2024\n\n## 1. Introduction\n\n…| Region | Revenue | Profit |\n| --- | --- | --- |\n| North | 120 | 30 |…",
  "structured_text_truncated": false,
  "answer_locations": [
    {"page_number":2,"element_type":"table","section_path":["Annual Report 2024","2. Results"],
     "snippet":"North | 120 | 30","table_cell":{"value":"30","row_label":"North","column_label":"Profit"}}
  ],
  "disclaimer": "Answers are generated by a small language model and may be incorrect. Check them against the document."
}
```

* `answer` is what the model server returned. `unanswerable` is true when it is `unanswerable`; `message` then explains it.
* `inference_time_seconds` is the time this backend waited for the model server (it includes the model's own time).
* `prompt_tokens`, `generated_tokens`, `document_chars_used` and `document_chars_total` are `null`: the model server does not report them and they are not estimated.
* `structured_text` is the exact Markdown (bounded by `MAX_MARKDOWN_RESPONSE_CHARS`) that was sent as `context`.
* `answer_locations` is **not produced by the model**. After the answer returns, the backend searches the extracted
  document for the answer text (case/punctuation-insensitive, whole words). It is `[]` when the answer does not
  appear verbatim, and always `[]` for `unanswerable`. Up to 5 hits are returned. The UI does not show it.
* No confidence score is returned because the model does not provide a calibrated one.

| Status | code | Cause |
|---|---|---|
| 400 | `empty_file` | zero-byte upload |
| 413 | `file_too_large` / `too_many_pages` | over `MAX_UPLOAD_MB` / `MAX_PAGES` |
| 415 | `unsupported_file_type` | not a `.pdf` or no PDF signature |
| 422 | `malformed_pdf` | corrupt / password-protected PDF |
| 422 | `no_extractable_text` | no meaningful text (e.g. scanned PDF; OCR not supported) |
| 422 | `empty_question` / `question_too_long` / `validation_error` | blank, >`MAX_QUESTION_CHARS`, or a missing field |
| 503 | `model_not_ready` | the model server is not reachable (checked before the PDF is processed) |
| 504 | `inference_timeout` | the model server did not answer within `GENERATION_TIMEOUT_SECONDS` |
| 500 | `model_auth_failed` | the model server rejected `MODEL_SERVER_API_KEY` |
| 500 | `inference_failed` / `internal_error` / `processing_failed` | other failures (details only in server logs) |

## 6. Frontend example (React / Vite, origin `http://localhost:5173`)

```js
const API = "http://127.0.0.1:8000";

export async function health() {
  return (await fetch(`${API}/health`)).json();
}

export async function ask(file, question) {
  const form = new FormData();
  form.append("file", file);          // field names must be "file" and "question"
  form.append("question", question);  // do not set Content-Type manually
  const res = await fetch(`${API}/answer`, { method: "POST", body: form });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error?.message ?? "Request failed");
  return body;
}
```

Other origins must be added to `CORS_ORIGINS`. Each question re-sends the PDF, because the server
keeps nothing between requests.

## 7. Design notes

* **Stateless:** nothing is stored. The PDF is read into memory for the request and discarded. The
  filename is only a display label and is never used as a path.
* **One generation at a time:** the model server serialises generation; concurrent requests queue.
* **Security:** CORS is restricted to `CORS_ORIGINS`; the server binds `127.0.0.1`; calls to the model
  server use a timeout, verify TLS certificates for `https` and refuse plain `http` to non-local hosts;
  the API key is read from the environment, sent only in a header and never logged; errors never include
  stack traces, filesystem paths or the model server's error bodies; document text is never logged
  (only counts).
* **Legacy code:** `model_service.py` still contains the in-process Transformers/PEFT loader and
  `prompting.py` the old prompt template. The app does not use them, and `torch`, `transformers`,
  `peft` and `accelerate` in `requirements.txt` are needed only for them.

## 8. Known limitations

* No OCR; structure detection is heuristic (see above).
* The model reads about 1-2 pages; longer documents are cut at the end.
* No authentication on this backend, no persistence, no streaming responses.
