# SLM Document Analysis Backend

FastAPI backend for the M.Tech project *"A Study of Parameter-Efficient Fine-Tuning of Small
Language Models (SLMs) for Document Analysis"*.

It performs **inference only**. The browser sends **one request containing a PDF and a question**.
The backend extracts the PDF's text **with its structure kept** (headings, lists, tables), puts the
whole document and the question into the **same prompt format used for fine-tuning**, runs the
Small Language Model (base model + LoRA adapter), and returns the answer. It does **not** train
anything, and there is **no chunking, no retrieval/RAG, and no server-side storage**.

```
Frontend --(PDF + question, one multipart request)--> POST /answer
   -> structure-preserving extraction (PyMuPDF)  -> Markdown
   -> analyst instruction prompt (Document Content / Question / Instructions / Answer)
   -> Qwen base model + LoRA adapter (greedy decoding)  -> first line = answer
   -> JSON response
```

## Project structure

```
slm-document-analysis-backend/
├── app/
│   ├── main.py                # FastAPI app: GET /health, POST /answer, error handlers, CORS
│   ├── config.py              # Settings read from environment variables / .env
│   ├── errors.py              # ApiError (uniform error responses)
│   ├── schemas.py             # Pydantic response models
│   ├── structure.py           # DocumentElement model, Markdown rendering, locate_answer()
│   ├── document_processor.py  # PDF validation + structure extraction (PyMuPDF heuristics)
│   ├── prompting.py           # The prompt template
│   └── model_service.py       # Model + LoRA loading, token budget, generation
├── tests/
│   ├── conftest.py            # PDF generators + fake model service
│   ├── test_document_processor.py
│   ├── test_structure.py      # Markdown, locate_answer, prompt template
│   ├── test_api.py
│   └── test_model_integration.py   # needs a real model; skipped by default
├── scripts/smoke_test.sh      # manual end-to-end check with curl
├── .env.example  .gitignore  pytest.ini  requirements.txt  README.md
```

## 1. Set up (macOS)

Requires Python 3.11–3.13 (any version supported by your PyTorch/Transformers install).

```bash
cd slm-document-analysis-backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## 2. Configure your model (REQUIRED)

No model location is assumed. Edit `.env`:

| Variable | What to put there |
|---|---|
| `MODEL_NAME_OR_PATH` | The **same base model** you fine-tuned: a Hugging Face id or a local folder. Your notebook `simple_pipeline.ipynb` used `Qwen/Qwen1.5-0.5B`; set it only if that is what produced your adapter. |
| `ADAPTER_PATH` | The folder the notebook saved with `model.save_pretrained(f"{OUT}/adapter")` (contains `adapter_config.json`). If empty, the **un-fine-tuned base model** runs. |
| `MODEL_DEVICE` | `auto` (cuda → mps → cpu), `cpu`, `mps`, `cuda`. |
| `MODEL_DTYPE` | `auto` (float32 on CPU, float16 otherwise), or `float32` / `float16` / `bfloat16`. |
| `MAX_NEW_TOKENS` | Default 64 (the notebook evaluated with 32). |
| `MAX_INPUT_TOKENS` | Default 8192; upper limit for document + question tokens. |

If `MODEL_NAME_OR_PATH` is empty the server still starts, `/health` returns `model_ready: false` with
the reason, and `/answer` returns HTTP 503 `model_not_ready`. It never substitutes another model.

### Prompt format

`app/prompting.py` builds this prompt:

```
You are a senior business and finance analyst at a reputed organization. Your task is to analyze the provided document content and answer the user's question accurately and concisely.

**Document Content:**
{document as Markdown}

**Question:**
{question}

**Instructions:**
- Answer the question using only the information provided in the document.
- Provide a precise, crisp, and professional answer that directly addresses the question.
- Avoid unnecessary explanations, background information, or repetition.
- Do not make assumptions or introduce information that is not supported by the document.
- If the answer is not available in the provided content, respond with: "The requested information is not available in the provided document."

**Answer:**
```

No chat template is added. The model's output is decoded from the **new tokens only**, with
**greedy decoding**, and the **first line** is the answer. An answer equal to the "not available"
sentence (or the bare word `unanswerable`) is reported as `unanswerable: true`.

Note: this instruction prompt differs from the `### Context / ### Question / ### Answer` layout used in
`notebooks/simple_pipeline.ipynb`. Confirm that your adapter was trained with this prompt; a different
training layout can reduce answer quality.

### What "structure-preserving" means

| PDF content | Markdown given to the model |
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

The whole document is sent. If document + question exceed `MAX_INPUT_TOKENS` (or the model's context
window), the **end of the document** is cut (never the question, the instructions or `**Answer:**`). The response then has
`document_truncated: true`, `document_chars_used` and `document_chars_total`. Note that your adapter was trained
with `MAX_LEN = 384` tokens, so quality on documents of several pages is **not established by the training
set** — evaluate it before relying on it.

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
pytest                          # unit + API tests; no model download needed
MODEL_NAME_OR_PATH=<model> [ADAPTER_PATH=<dir>] pytest -m requires_model   # mechanics with a real model
```

The API tests use a **fake model service**; they verify HTTP behaviour and the prompt/Markdown plumbing,
not real model output. The `requires_model` tests check mechanics (prompt token ids, first-line answer,
document-only truncation); they do not measure answer quality.

## 5. API reference

All errors use one shape: `{"error": {"code": "<machine_code>", "message": "<text>"}}`.

### `GET /health`
```json
{"status":"ok","model_ready":true,"model_name":"Qwen/Qwen1.5-0.5B + LoRA adapter 'adapter'",
 "adapter_loaded":true,"device":"cuda","model_error":null}
```

### `POST /answer` (multipart/form-data: `file` = PDF, `question` = text)

```bash
curl -F "file=@report.pdf;type=application/pdf" -F "question=What was the profit of the North region?" \
  http://127.0.0.1:8000/answer
```

Response 200:
```json
{
  "question": "What was the profit of the North region?",
  "answer": "30",
  "unanswerable": false,
  "message": null,
  "model_name": "Qwen/Qwen1.5-0.5B + LoRA adapter 'adapter'",
  "inference_time_seconds": 1.25,
  "prompt_tokens": 122,
  "generated_tokens": 3,
  "document": {"filename":"report.pdf","page_count":2,"heading_count":4,"table_count":1,
               "character_count":388,"word_count":70,"empty_page_numbers":[]},
  "document_truncated": false,
  "document_chars_used": 388,
  "document_chars_total": 388,
  "structured_text": "# Annual Report 2024\n\n## 1. Introduction\n\n…| Region | Revenue | Profit |\n| --- | --- | --- |\n| North | 120 | 30 |…",
  "structured_text_truncated": false,
  "answer_locations": [
    {"page_number":2,"element_type":"table","section_path":["Annual Report 2024","2. Results"],
     "snippet":"North | 120 | 30","table_cell":{"value":"30","row_label":"North","column_label":"Profit"}}
  ],
  "disclaimer": "Answers are generated by a small language model and may be incorrect. Check them against the document."
}
```
(Illustrative values; every field is computed per request.)

* `answer` is the first line of the model's output. `unanswerable` is true when it equals the training label
  `unanswerable`; `message` then explains it.
* `structured_text` is the exact Markdown (bounded by `MAX_MARKDOWN_RESPONSE_CHARS`) that was placed in the
  prompt, so the user can see what the model was given.
* `answer_locations` is **not produced by the model**. After generation the backend searches the extracted
  document for the answer text (case/punctuation-insensitive, whole words). Each hit reports the page,
  the heading path, and for tables the row and column labels. It is `[]` when the answer text does not
  appear verbatim (for example when the model rephrases or is wrong), and always `[]` for `unanswerable`.
  Short answers such as `2` can match several places; up to 5 are returned.
* No confidence score is returned because the model does not provide a calibrated one.

| Status | code | Cause |
|---|---|---|
| 400 | `empty_file` | zero-byte upload |
| 413 | `file_too_large` / `too_many_pages` | over `MAX_UPLOAD_MB` / `MAX_PAGES` |
| 415 | `unsupported_file_type` | not a `.pdf` or no PDF signature |
| 422 | `malformed_pdf` | corrupt / password-protected PDF |
| 422 | `no_extractable_text` | no meaningful text (e.g. scanned PDF; OCR not supported) |
| 422 | `empty_question` / `question_too_long` / `validation_error` | blank, >`MAX_QUESTION_CHARS`, or a missing field |
| 503 | `model_not_ready` / `model_not_configured` | model not loaded (checked before the PDF is processed) |
| 504 | `inference_timeout` | exceeded `GENERATION_TIMEOUT_SECONDS` |
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
  filename is only a display label and is never used as a path. Run with any number of workers; each
  keeps its own copy of the model, so one worker is advisable for large models.
* **One generation at a time:** a lock serialises generation; concurrent requests queue.
* **Security:** CORS is restricted to `CORS_ORIGINS`; the server binds `127.0.0.1`; errors never include
  stack traces or filesystem paths; document text is never logged (only counts).

## 8. Can the model run locally?

I could not inspect your adapter, but the notebook shows a **0.5B-parameter** base, which is small.

* `Qwen/Qwen1.5-0.5B` in float32 on **CPU** was run here (without your adapter) at about 3.8 s for a
  122-token prompt and 16 new tokens. Long (several-page) prompts are slower on CPU.
* **MacBook Air (Apple Silicon):** `MODEL_DEVICE=mps` or `cpu` should work for a 0.5B model; memory is
  shared, so a very long prompt (`MAX_INPUT_TOKENS`) needs more RAM. **This was not tested on a Mac.**
* **Google Colab (GPU):** see `../colab/` — the same backend runs there and is reached through a tunnel.
* **Remote inference from a local backend (NOT implemented).** A class with the same interface as
  `ModelService` (`is_ready`, `status`, `generate_answer`) could be passed to `create_app(model_service=...)`.

## 9. Known limitations

* No OCR; structure detection is heuristic (see above).
* Quality on multi-page documents has not been measured; the adapter was trained on ≤384-token examples.
* No authentication, no persistence, no streaming responses.
