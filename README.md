# SLM Document Analysis (Proof of Concept)

M.Tech project: *"A Study of Parameter-Efficient Fine-Tuning of Small Language Models (SLMs) for Document Analysis."*

The user chooses a text-based PDF and types a question. **One request** carries both to the backend.
The backend extracts the PDF text **keeping its structure** (headings, lists, tables as Markdown), builds a
prompt in the **exact format the model was fine-tuned with**, runs your Small Language Model
(Qwen base + LoRA adapter) and returns the answer. This is **inference only**: nothing is trained. There is
**no chunking, no retrieval (no RAG) and no server-side storage**.

```
 MacBook / WSL (Chrome)                           Google Colab (GPU runtime)
 ┌──────────────────┐  one multipart request   ┌─────────────────────────────────────────┐
 │ React + Vite UI  │ ───────────────────────► │ cloudflared quick tunnel                │
 │ localhost:5173   │   POST /answer           │   └► uvicorn / FastAPI (127.0.0.1)      │
 │ (file + question)│ ◄─────────────────────── │        PDF → Markdown (PyMuPDF)         │
 └──────────────────┘   JSON answer            │        → analyst instruction prompt     │
   VITE_API_BASE_URL                           │           → SLM + LoRA                  │
                                               └─────────────────────────────────────────┘
```

## What is in this folder

| Path | Purpose |
|---|---|
| `frontend/` | React + TypeScript + Vite web UI |
| `colab/slm_document_analysis_backend.ipynb` | **The Colab notebook** (backend + model + tunnel) |
| `colab/requirements.txt` | Packages the notebook installs (Colab already provides PyTorch) |
| `colab/build_notebook.py` | Regenerates the notebook from the tested backend modules (do not edit the notebook's backend code by hand) |
| `colab/verify_notebook.py` | Runs the notebook's cells locally and exercises the API (developer check) |
| `slm-document-analysis-backend/` | The same backend as a normal Python project (modules, pytest tests, `requirements.txt`, its own README with the full API reference) |
| `notebooks/` | Your training notebook, kept as reference only. Nothing here depends on it and it was not modified |

## Information you must supply

Nothing about your trained model is pre-filled:

1. `MODEL_NAME_OR_PATH`: the base model your adapter was trained on (Hugging Face id or path). Your training notebook used `Qwen/Qwen1.5-0.5B`; enter it only if that is what produced the adapter you load.
2. `ADAPTER_PATH`: the folder saved with `model.save_pretrained(...)` (contains `adapter_config.json`). If left empty, the **un-fine-tuned base model** runs, so its answers are not representative of your project.
3. Where they live: Hugging Face (add `HF_TOKEN` to Colab Secrets if gated) or Google Drive (`MOUNT_GOOGLE_DRIVE = True`).

If the model is not configured the server still starts, `/health` reports `model_ready: false` with the reason, and `/answer` returns 503. It never substitutes another model.

## Prompt format

`slm-document-analysis-backend/app/prompting.py` builds this prompt (the one text the model receives):

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

No chat template is added. Decoding is greedy, only the newly generated tokens are decoded, and the **first line** is the answer. An answer equal to the "not available" sentence (or the bare word `unanswerable`) is reported as unanswerable.

## Part A: Run the backend in Google Colab

1. Open <https://colab.research.google.com>, choose **Upload**, select `colab/slm_document_analysis_backend.ipynb`.
2. `Runtime → Change runtime type → T4 GPU`.
3. If the model is private/gated: key icon → *Secrets* → add `HF_TOKEN`, enable notebook access. Never paste tokens into cells.
4. **Cell 1** installs dependencies. (If a later cell shows an import error: `Runtime → Restart session`, run again from Cell 1.)
5. **Cell 2**: fill in `MODEL_NAME_OR_PATH`, `ADAPTER_PATH`, `MOUNT_GOOGLE_DRIVE` in the `EDIT THESE VALUES` block.
6. **Cell 3** loads the model (mounts Drive if requested). It stops with a clear error if the model/adapter is missing.
7. Run **Cells 4–7** in order. **Cell 6** runs the real model once on a built-in test document; read its output to confirm inference works.
8. **Cell 8** starts the API and a Cloudflare quick tunnel, then prints a line like

   ```
   VITE_API_BASE_URL=https://something-random.trycloudflare.com
   ```
   Copy it. The cell also prints a `/health` check and a CORS self-check.
9. **Stop:** run Cell 8b. **Restart:** re-run Cell 8 (the URL changes each time → update the frontend `.env`).

**Tunnel and security notes**

* Colab's `localhost` is not reachable from your computer, hence the tunnel. A quick tunnel needs no account and **no token**. The API only listens on `127.0.0.1` inside Colab.
* Anyone who knows the random URL can call the API (CORS only restricts browsers). Keep it private and stop the tunnel when done. There is no authentication in this proof of concept.
* CORS lists the **frontend's** origin (default `http://localhost:5173` and `http://127.0.0.1:5173`). A wildcard is refused.
* Cloudflare cuts requests at ~100 s; `GENERATION_TIMEOUT_SECONDS` defaults to 90 in the notebook.

## Part B: Run the frontend

Requires Node.js 20.19+ or 22.12+.

```bash
cd frontend
npm install
cp .env.example .env
# edit .env: set VITE_API_BASE_URL to the URL printed by Colab Cell 8 (no trailing slash)
npm run dev
```

Open <http://localhost:5173> in Chrome. The port is fixed (`strictPort`) because the backend's CORS allows exactly that origin. After editing `.env`, stop and re-run `npm run dev`.

Other commands: `npm test` (17 unit tests), `npm run build` (type-check + production build).

The UI has four steps: **1** choose a PDF, **2** type a question and press **Get Answer**, **3** the answer with the model, timing and token counts plus *where the answer text appears in the document*, **4** the extracted Markdown structure that was given to the model. The PDF is re-sent with every question because the server keeps nothing.

## Optional: run the backend locally instead of Colab

```bash
cd slm-document-analysis-backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # set MODEL_NAME_OR_PATH, ADAPTER_PATH, MODEL_DEVICE
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then use `VITE_API_BASE_URL=http://127.0.0.1:8000`. See `slm-document-analysis-backend/README.md` for settings, the API reference and error codes. A 0.5B model should be feasible on CPU or Apple-Silicon (`MODEL_DEVICE=mps`), but this was not tested on a Mac.

## API summary

All errors: `{"error": {"code": "<machine_code>", "message": "<text>"}}`.

* `GET /health` → `{"status","model_ready","model_name","adapter_loaded","device","model_error"}`
* `POST /answer` (multipart: `file` = PDF, `question` = text) → `answer`, `unanswerable`, `model_name`, `inference_time_seconds`, `prompt_tokens`, `generated_tokens`, `document` (page/heading/table/word counts), `document_truncated` (+ chars used/total), `structured_text` (the Markdown given to the model), `answer_locations`, `disclaimer`.

Error codes: 400 `empty_file`; 413 `file_too_large` / `too_many_pages`; 415 `unsupported_file_type`; 422 `malformed_pdf` / `no_extractable_text` / `empty_question` / `question_too_long` / `validation_error`; 503 `model_not_ready` / `model_not_configured`; 504 `inference_timeout`; 500 `inference_failed` / `internal_error`.

### About `answer_locations`

After generation, the backend searches the extracted document for the model's answer text (case/punctuation-insensitive, whole words) and reports the page, heading path and, for tables, the row and column labels. **This is plain string matching, not model output**: it is not an explanation of the model's reasoning and not a confidence score. It is empty when the answer does not appear verbatim (for example when the model rephrases or is wrong) and always empty for `unanswerable`. Short answers such as `2` can match several places.

## End-to-end test procedure (for you to run)

1. Start the Colab notebook through Cell 8; note `VITE_API_BASE_URL`.
2. Open `<that URL>/health` in a browser: expect `"model_ready": true` and your model name.
3. Put the URL in `frontend/.env`, run `npm run dev`, open <http://localhost:5173>. The green banner should show the model.
4. Choose a small text-based PDF, ask a question whose answer is in it, press **Get Answer**. Check the answer against the PDF and look at Step 4 to see how headings and tables were read.
5. Ask a second question (the PDF is re-sent automatically).
6. Ask something absent from the document; judge whether the model says `unanswerable`.
7. Choose a `.txt` file (UI error) and a renamed non-PDF `fake.pdf` (backend `malformed_pdf` error).
8. Stop the tunnel (Cell 8b) and press **Get Answer**/reload: the red "Backend unavailable" state should appear.

## What was actually tested (and what was not)

Tested during development, in Linux/WSL:

* Backend: 59 pytest tests pass, 3 are marked `requires_model`. Those 3 were run separately with a tiny random model (`hf-internal-testing/tiny-random-LlamaForCausalLM`, gibberish output) and passed. The API tests use a fake model service, so they check HTTP behaviour and prompt/Markdown plumbing, not answer quality.
* The real `Qwen/Qwen1.5-0.5B` **base model without your adapter** was run on CPU through the API: `POST /answer` returned 200 (122-token prompt, 16 new tokens, ~3.8 s). Its answer was wrong, as expected without the adapter; this shows the pipeline works, not that answers are good.
* The notebook's code cells were executed in order locally (`colab/verify_notebook.py`, real Qwen1.5-0.5B on CPU, tunnel disabled): health, a single request, invalid PDF, blank question, missing question, old `/documents` routes gone, CORS preflight: all passed.
* Structure extraction was checked on a generated PDF (title, headings, bullets, a ruled table, paragraphs).
* Frontend: 17 unit tests pass; type-check and production build pass.

**Not tested:** Google Colab itself (the notebook was never opened there), the cloudflared tunnel with this version of the notebook (an earlier version of the same tunnel code worked), your actual LoRA adapter, GPU inference, Google Drive mounting, Colab Secrets, answer quality (especially on multi-page documents), the rendered UI in a browser against the new backend (no browser end-to-end test was run for the single-request flow), and macOS / Apple-Silicon (MPS).

## Known limitations

* No OCR: scanned PDFs are rejected.
* Structure detection is heuristic (font size, bold, ruling lines). Tables without ruling lines are read as plain text; multi-column pages may be read in the wrong order.
* Your adapter was trained on examples of at most 384 tokens (`MAX_LEN=384`). Whole multi-page documents are much longer, so quality on them is **not established by the training set**; evaluate it.
* If the prompt exceeds `MAX_INPUT_TOKENS` (or the model's context window), the **end of the document** is cut and the response says so.
* No authentication; keep the tunnel URL private. One generation at a time. Answers can be wrong; no accuracy or confidence metrics are produced.
