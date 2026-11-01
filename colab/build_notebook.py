"""Build ``slm_document_analysis_backend.ipynb`` (Google Colab) and ``requirements.txt``.

The notebook's Python is NOT written separately: it is assembled from the tested modules in
``../slm-document-analysis-backend/app/`` (with the ``from app.x import ...`` lines removed, since
everything lives in one notebook namespace). That keeps the notebook and the unit-tested
backend identical.

Usage (from this folder):  python build_notebook.py
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "slm-document-analysis-backend" / "app"

PIP_PACKAGES = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "python-multipart>=0.0.9",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "pymupdf>=1.24",
    "transformers>=4.56",
    "peft>=0.13",
]

_APP_IMPORT = re.compile(r"^from app\.\w+ import (?:\([^)]*\)|[^\n]*)\n", re.MULTILINE)
_FUTURE = re.compile(r"^from __future__ import annotations\n", re.MULTILINE)


def module_source(name: str) -> str:
    """Return a module's source, minus intra-package imports and the __future__ import."""
    src = (APP / f"{name}.py").read_text(encoding="utf-8")
    src = _APP_IMPORT.sub("", src)
    src = _FUTURE.sub("", src)
    return src.strip() + "\n"


def join_modules(*parts: str) -> str:
    return "from __future__ import annotations\n\n" + "\n\n".join(parts)


# --------------------------------------------------------------------------------------
# Cell contents
# --------------------------------------------------------------------------------------
INTRO_MD = """\
# SLM Document Analysis: Backend (Google Colab)

This notebook runs the **backend** for the *SLM Document Analysis* proof of concept. The React
frontend sends **one request containing a PDF and a question**. The backend extracts the PDF text
**with its structure kept** (headings, lists, tables as Markdown), builds a prompt in the **same format
the model was fine-tuned with**, runs your **Small Language Model (base model + LoRA adapter)**, and
returns the answer. The frontend on your MacBook reaches it over HTTPS through a temporary tunnel.

**Inference only.** Nothing is trained here. The LoRA adapter is an already-trained artefact that is
only *loaded*. There is **no chunking, no retrieval (no RAG) and no server-side storage**: the whole
document goes into the prompt and nothing is kept after the response is sent.

**How to run**
1. `Runtime -> Change runtime type -> GPU` (a T4 is plenty for a 0.5B model).
2. Fill in **Cell 2** (model and adapter). No model is assumed: you must supply it.
3. Run the cells **in order**, 1 to 8. Cell 8 prints the URL that goes into the frontend's `.env`.

**Limitations (read this)**
* Text-based PDFs only. **There is no OCR**: scanned PDFs are rejected.
* Structure detection is heuristic (font size, bold, ruling lines). Tables without ruling lines are
  read as plain text.
* The adapter was trained on examples of at most 384 tokens. Whole multi-page documents are far
  longer, so answer quality on them must be evaluated; the training set does not establish it.
* If a document exceeds `MAX_INPUT_TOKENS`, the END of the document is cut and the response says so.
* The tunnel URL changes every time you start it. Update the frontend `.env` and restart `npm run dev`.
* Answers come from a small model and can be wrong.

> The Python in this notebook is generated from the unit-tested backend modules in
> `slm-document-analysis-backend/app/` by `colab/build_notebook.py`. If you change the backend
> source, rebuild the notebook instead of editing both.
"""

CELL1_MD = """\
## Cell 1: Install dependencies

Only what the code imports is installed. **PyTorch is not reinstalled**: Colab already ships a
CUDA build of it, and replacing it would be slow and could break the GPU runtime.

* `fastapi`, `uvicorn`, `python-multipart`: the API server and multipart (file upload) parsing
* `pydantic`, `pydantic-settings`: request validation and the settings object
* `pymupdf`: PDF text extraction
* `transformers`, `peft`: loading the base model and the LoRA adapter

If pip upgrades `transformers` or `pydantic` and a later cell raises an import error, use
`Runtime -> Restart session` and run the cells again from Cell 1.
"""

CELL1_CODE = r'''%pip install -q __PIP_ARGS__

import importlib.metadata as _md

for _pkg in ["fastapi", "uvicorn", "pydantic", "pydantic-settings", "pymupdf", "torch", "transformers", "peft"]:
    print(f"{_pkg:<18} {_md.version(_pkg)}")
'''

CELL2_MD = """\
## Cell 2: Configuration

Edit the values in the **"EDIT THESE VALUES"** block. Everything below it is the settings class
(taken from the tested backend) and normally needs no changes.

**Values you must supply**

| Variable | Meaning |
|---|---|
| `MODEL_NAME_OR_PATH` | The **same base model** you fine-tuned: a Hugging Face id (for example `"org/model-name"`) or a path such as `"/content/drive/MyDrive/.../base_model"`. |
| `ADAPTER_PATH` | Folder containing the saved LoRA adapter (it holds `adapter_config.json`), i.e. the folder the training notebook wrote with `model.save_pretrained(f"{OUT}/adapter")`. If left `""`, the **un-fine-tuned** base model runs. |
| `MOUNT_GOOGLE_DRIVE` | Set to `True` if the model or adapter lives in Google Drive. Cell 3 then mounts Drive. |

Nothing is pre-filled. If `MODEL_NAME_OR_PATH` is empty, Cell 3 stops with an explanatory error
instead of picking a model for you.

**Private or gated Hugging Face models:** store your token in Colab's **Secrets** panel (key icon in
the left sidebar) under the name `HF_TOKEN` and enable notebook access. Cell 3 reads it from there.
Never paste tokens into a cell.
"""

CELL2_USER_BLOCK = '''\
# ======================================================================================
#                                 EDIT THESE VALUES
# ======================================================================================

# --- Model (REQUIRED: nothing is assumed) -----------------------------------------------
MODEL_NAME_OR_PATH = ""      # <<< FILL IN: base model id or path (the one your adapter was trained on)
ADAPTER_PATH = ""            # <<< FILL IN if you use a LoRA adapter, e.g. "/content/drive/MyDrive/.../adapter"
TOKENIZER_NAME_OR_PATH = ""  # optional; leave empty to use the base model's tokenizer
MOUNT_GOOGLE_DRIVE = False   # True if the model/adapter is stored in Google Drive

# --- Device and precision -----------------------------------------------------------------
MODEL_DEVICE = "auto"        # "auto" (cuda if available), "cuda" or "cpu"
MODEL_DTYPE = "auto"         # "auto" (float16 on GPU, float32 on CPU), "float16", "bfloat16", "float32"
MERGE_ADAPTER = False        # True merges the LoRA weights into the base model (slightly faster inference)
TRUST_REMOTE_CODE = False    # only set True for models that require custom code you trust

# --- Generation (greedy decoding, as in the training notebook's evaluation) -------------------
MAX_NEW_TOKENS = 64          # the training notebook evaluated with 32; the answer is the first line
MAX_INPUT_TOKENS = 8192      # document + question tokens; the END of a longer document is cut
# Keep this BELOW ~100 s: Cloudflare tunnels cut requests off at about 100 s with an HTML error page.
GENERATION_TIMEOUT_SECONDS = 90

# --- Uploads -------------------------------------------------------------------------------
MAX_UPLOAD_MB = 20
MAX_PAGES = 100

# --- API server and CORS ---------------------------------------------------------------------
API_HOST = "127.0.0.1"       # keep as is: the tunnel connects locally; never expose 0.0.0.0
API_PORT = 8000
# Origins of the FRONTEND (where the browser page is served from), NOT the tunnel URL.
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
START_TUNNEL = True          # False = start only the local server (no public URL)

# ======================================================================================
#                      END OF EDITABLE VALUES (settings class below)
# ======================================================================================
'''

CELL2_BUILD_SETTINGS = '''

settings = Settings(
    _env_file=None,  # ignore any .env file; the values above are the single source of truth
    model_name_or_path=MODEL_NAME_OR_PATH,
    adapter_path=ADAPTER_PATH,
    tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH,
    model_device=MODEL_DEVICE,
    model_dtype=MODEL_DTYPE,
    merge_adapter=MERGE_ADAPTER,
    trust_remote_code=TRUST_REMOTE_CODE,
    max_new_tokens=MAX_NEW_TOKENS,
    max_input_tokens=MAX_INPUT_TOKENS,
    generation_timeout_seconds=GENERATION_TIMEOUT_SECONDS,
    max_upload_mb=MAX_UPLOAD_MB,
    max_pages=MAX_PAGES,
    cors_origins=",".join(ALLOWED_ORIGINS),
    load_model_on_startup=False,  # the model is loaded explicitly in Cell 3
)


def get_settings() -> Settings:
    """Used by create_app() (Cell 9) when no settings object is passed in."""
    return settings


print("Configuration OK")
print("  model   :", settings.model_name_or_path or "<<< NOT SET: fill in Cell 2")
print("  adapter :", settings.adapter_path or "(none: base model only)")
print("  limits  :", settings.max_input_tokens, "prompt tokens,", settings.max_new_tokens, "new tokens,", settings.max_pages, "pages")
print("  origins :", settings.cors_origin_list)
'''

CELL3_MD = """\
## Cell 3: Load the model

`ModelService` loads the tokenizer, the **base model**, and (if `ADAPTER_PATH` is set) the **LoRA adapter**
with `peft.PeftModel.from_pretrained`, then switches the model to evaluation (inference) mode.
It is loaded **once** and reused for every question.

* If `MOUNT_GOOGLE_DRIVE = True`, Drive is mounted first (Colab will ask you to authorise it).
* A Hugging Face token, if needed, is read from Colab Secrets (`HF_TOKEN`).
* Model loading is deliberately separate from the PDF extraction code below.
* It never substitutes another model: a missing or wrong path stops with an error.
* Like the training notebook, it sets `pad_token = eos_token` when the tokenizer has no pad token.

This class also contains the generation method (`generate_answer`) used by Cell 6 and the API.
"""

CELL3_LOAD_CODE = '''\


# --------------------------------------------------------------------------------------
# Load the configured model (runs once)
# --------------------------------------------------------------------------------------
if MOUNT_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")

# A Hugging Face token is only needed for gated/private models. Read it from Colab Secrets.
try:
    from google.colab import userdata

    _hf_token = userdata.get("HF_TOKEN")
    if _hf_token:
        os.environ["HF_TOKEN"] = _hf_token
except Exception:  # not running on Colab, or no HF_TOKEN secret / access not granted
    pass

model_service = ModelService(settings)
try:
    model_service.load()
except ModelServiceError as exc:
    print("MODEL NOT LOADED:", exc.message)
    raise
print("Model ready:", model_service.status)
'''



CELL8_MD = """\
## Cell 8: Start the API server and the tunnel

Colab's `localhost` is **not reachable from your MacBook**, so two things happen here:

1. **uvicorn** runs the API in a background thread on `127.0.0.1:8000` (inside Colab only). It stays up
   while the Colab session is alive, and the cell returns immediately so you can keep using the notebook.
2. **cloudflared** (Cloudflare "quick tunnel") publishes that local port at a temporary
   `https://<random>.trycloudflare.com` address. No account and **no token** are needed, so there is no
   secret to protect or leak. The binary is downloaded from Cloudflare's official GitHub releases.

**Using it**
* Copy the `VITE_API_BASE_URL=...` line printed at the end into the frontend's `.env`, then restart
  `npm run dev`.
* Anyone who knows the random URL can call the API (CORS only restricts *browsers*). Treat the URL as
  private and stop the tunnel when you are done (Cell 8b).
* Cloudflare cuts requests at about 100 s, so keep `GENERATION_TIMEOUT_SECONDS` at 90 or below.
* **Restart:** re-run this cell (it stops the previous server and tunnel first). The URL changes, so
  update the frontend `.env`. If you changed Cells 2 to 7, re-run those first. If the Colab runtime
  was restarted, run everything from Cell 1.

Alternatives such as ngrok need an auth token; if you use one, keep it in Colab Secrets
(`userdata.get(...)`), never in a cell or in the frontend. They are not implemented here.
"""

CELL8_CODE = r'''import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

import uvicorn

WORK_DIR = "/content" if os.path.isdir("/content") else tempfile.gettempdir()
CLOUDFLARED_PATH = os.path.join(WORK_DIR, "cloudflared")
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
_TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

_server = None
_server_thread = None
_tunnel_process = None


def stop_server() -> None:
    global _server, _server_thread
    if _server is not None:
        _server.should_exit = True
    if _server_thread is not None:
        _server_thread.join(timeout=10)
    _server = _server_thread = None


def stop_tunnel() -> None:
    global _tunnel_process
    if _tunnel_process is not None and _tunnel_process.poll() is None:
        _tunnel_process.terminate()
        try:
            _tunnel_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _tunnel_process.kill()
    _tunnel_process = None


def start_server() -> None:
    """Run uvicorn in a background thread (a notebook cell cannot block)."""
    global _server, _server_thread
    stop_server()
    _server = uvicorn.Server(uvicorn.Config(app, host=API_HOST, port=API_PORT, log_level="info"))
    _server_thread = threading.Thread(target=_server.run, name="uvicorn", daemon=True)
    _server_thread.start()
    for _ in range(100):
        if _server.started:
            return
        if not _server_thread.is_alive():
            raise RuntimeError(f"The server failed to start. Is port {API_PORT} already in use? Re-run this cell.")
        time.sleep(0.1)
    raise RuntimeError("The server did not start within 10 seconds.")


def start_tunnel(timeout: float = 60.0) -> str:
    """Start a cloudflared quick tunnel to the local server and return its public https URL."""
    global _tunnel_process
    stop_tunnel()
    if not os.path.exists(CLOUDFLARED_PATH):
        print("Downloading cloudflared ...")
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED_PATH)
        os.chmod(CLOUDFLARED_PATH, 0o755)

    proc = subprocess.Popen(
        [CLOUDFLARED_PATH, "tunnel", "--no-autoupdate", "--url", f"http://{API_HOST}:{API_PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _tunnel_process = proc
    lines: "queue.Queue[str]" = queue.Queue()

    def _pump() -> None:  # keep draining output so the pipe never blocks cloudflared
        for line in proc.stdout:
            lines.put(line)

    threading.Thread(target=_pump, daemon=True).start()

    recent: list[str] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            line = lines.get(timeout=1)
        except queue.Empty:
            if proc.poll() is not None:
                break
            continue
        recent.append(line.rstrip())
        match = _TUNNEL_URL.search(line)
        if match and match.group(0) != "https://api.trycloudflare.com":
            return match.group(0)
    stop_tunnel()
    raise RuntimeError("cloudflared did not report a tunnel URL. Last output:\n" + "\n".join(recent[-10:]))


def http_request(url: str, method: str = "GET", headers: dict | None = None, timeout: float = 15.0):
    """Tiny HTTP helper for self-checks. Returns (status, response_headers, body_text)."""
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "slm-backend-selfcheck/1.0", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read().decode("utf-8", "replace")


# ---- start the local server and check it ------------------------------------------------
start_server()
LOCAL_URL = f"http://{API_HOST}:{API_PORT}"
status, _, body = http_request(f"{LOCAL_URL}/health")
print(f"\nLocal check  GET {LOCAL_URL}/health -> {status}\n{body}")

# ---- verify CORS from the server's point of view ------------------------------------------
for origin in [*ALLOWED_ORIGINS[:1], "https://not-allowed.example"]:
    status, hdrs, _ = http_request(
        f"{LOCAL_URL}/answer",
        method="OPTIONS",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )
    allowed = hdrs.get("access-control-allow-origin")
    print(f"CORS preflight from {origin}: HTTP {status}, allow-origin = {allowed!r}")

# ---- optional public tunnel -------------------------------------------------------------------
if START_TUNNEL:
    PUBLIC_URL = start_tunnel()
    print("\nTunnel started:", PUBLIC_URL)
    for attempt in range(15):  # a brand-new tunnel hostname can take a few seconds to resolve
        try:
            status, _, body = http_request(f"{PUBLIC_URL}/health")
            if status == 200:
                print(f"Tunnel check GET {PUBLIC_URL}/health -> {status}\n{body}")
                break
        except OSError:
            pass
        time.sleep(2)
    else:
        print("WARNING: the tunnel did not answer /health yet. Wait a few seconds and open the URL in a browser.")
    print("\n" + "=" * 78)
    print("Put this line in frontend/.env, then restart `npm run dev`:\n")
    print(f"VITE_API_BASE_URL={PUBLIC_URL}")
    print("=" * 78)
else:
    print("\nSTART_TUNNEL is False: the API is reachable only inside Colab at", LOCAL_URL)
'''

CELL8B_MD = """\
### Cell 8b: Stop the server and the tunnel (optional)

Run this when you are finished. It stops the public URL and the API. (The server never stores
uploaded documents, so there is nothing to clean up.) To start again, re-run Cell 8.
"""

CELL8B_CODE = """stop_tunnel()
stop_server()
print("Tunnel and server stopped.")
"""


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------
def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(True),
    }




# --------------------------------------------------------------------------------------
# Cells 4 to 7 (new single-request design)
# --------------------------------------------------------------------------------------
CELL4_MD = """\
## Cell 4: Structure-preserving PDF extraction

`process_pdf(filename, data, max_bytes=...)` validates the upload and reads the PDF with PyMuPDF
**keeping its structure**, then `StructuredDocument.markdown` renders it as Markdown:

| PDF content | Markdown |
|---|---|
| title / headings | `#`, `##`, `###`, `####` (font size relative to the body text, bold, section numbers such as `1.1`) |
| ruled tables | `\\| col \\| col \\|` rows with a `\\| --- \\|` separator |
| bullets / numbered items | `- item` / `1. item` |
| body text | paragraphs separated by blank lines |

Page numbers, repeated headers and footers are removed. Every element remembers its page so the answer
can later be located.

It raises a specific error for a wrong file type (extension **and** `%PDF-` signature), an empty file,
a file over `MAX_UPLOAD_MB`, too many pages, a corrupt or password-protected PDF, and a PDF with **no
meaningful text**. **No OCR:** scanned PDFs are rejected, not guessed at.

These are heuristics: unusual layouts (multi-column pages, tables without ruling lines, headings styled
like body text) may be read imperfectly.
"""

CELL5_MD = """\
## Cell 5: Prompt

`build_prompt(context, question)` produces the text the model receives:

```
You are a senior business and finance analyst at a reputed organization. Your task is to analyze
the provided document content and answer the user's question accurately and concisely.

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

No chat template is added. The model continues after `**Answer:**` and its **first line** is the
answer. The sentence "The requested information is not available in the provided document." (or the
bare word `unanswerable`) is reported as an unanswerable result.

`structure.py` (included in this cell) also holds `locate_answer`: after generation it **searches the
extracted document for the answer text** and reports the page, the heading path and, for tables, the
row and column labels. This is plain string matching, **not model output**; it returns nothing when the
answer text does not appear verbatim.
"""

CELL6_MD = """\
## Cell 6: Model inference (smoke test)

`model_service.generate_answer(document_markdown, question)` builds the prompt, runs **greedy**
generation and decodes **only the newly generated tokens**. If the prompt would exceed
`MAX_INPUT_TOKENS` (or the model's context window), the END of the document is cut; the question and
`**Answer:**` are never cut, and the result says `document_truncated`.

This cell calls the **real loaded model** once on a tiny test document so you can confirm that
inference works before starting the API. The document below is test input only; it is not used by the
application. Judge the answer yourself: this is a functional check, not an evaluation.
"""

CELL6_CODE = r'''import time

TEST_DOCUMENT = (
    "# Meridian Report\n\n## 1. Summary\n\nThe Meridian Report was published in 2021. "
    "Its main recommendation is that regional libraries extend their weekend opening hours.\n\n"
    "| Library | Opening hours |\n| --- | --- |\n| North | 9-17 |\n| South | 10-16 |"
)
TEST_QUESTION = "What is the main recommendation of the Meridian Report?"

_started = time.monotonic()
try:
    _result = model_service.generate_answer(TEST_DOCUMENT, TEST_QUESTION)
except ModelServiceError as exc:
    print("INFERENCE FAILED:", exc.message)
    raise
_elapsed = time.monotonic() - _started

print("Question :", TEST_QUESTION)
print("Answer   :", _result.answer, "(unanswerable)" if _result.unanswerable else "")
print(f"Model    : {_result.model_name}")
print(f"Time     : {_elapsed:.2f} s  (prompt tokens: {_result.prompt_tokens}, generated tokens: {_result.generated_tokens})")
'''

CELL7_MD = """\
## Cell 7: FastAPI application and CORS

`create_app()` builds the API:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | server status and real model readiness |
| POST | `/answer` | multipart form with `file` (PDF) and `question`: extract, prompt, run the SLM, return the answer |

The response contains the answer, the model name, the measured inference time, the extraction facts
(pages, headings, tables, words), the structured Markdown that was given to the model, and
`answer_locations` (string matches of the answer in the document). All errors share one JSON shape,
`{"error": {"code": "...", "message": "..."}}`, and never include stack traces or file paths.
Document contents are not logged and nothing is stored.

**CORS.** A browser only lets the page at `http://localhost:5173` call this API if the API names that
origin. `ALLOWED_ORIGINS` (Cell 2) is that list; `create_app()` passes it to `CORSMiddleware`
(methods GET, POST, OPTIONS; no credentials). The browser's `Origin` is the *frontend's* origin, not
the tunnel URL, so the tunnel URL does **not** go into `ALLOWED_ORIGINS`. A wildcard `*` is refused below,
and the API is only ever bound to localhost.
"""

CELL7_CODE = r'''if "*" in ALLOWED_ORIGINS:
    raise ValueError("ALLOWED_ORIGINS must list explicit origins; a wildcard '*' is not allowed.")
if API_HOST not in ("127.0.0.1", "localhost"):
    raise ValueError("API_HOST must be 127.0.0.1 or localhost. The tunnel connects locally.")

app = create_app(settings=settings, model_service=model_service)

print("FastAPI app created.")
print("Allowed browser origins:", settings.cors_origin_list)
print("Routes:", sorted({f"{sorted(r.methods - {'HEAD', 'OPTIONS'})[0]} {r.path}" for r in app.routes if hasattr(r, "methods") and r.path.startswith(("/health", "/answer"))}))
'''


def build_cells() -> list[dict]:
    pip_args = " ".join(f'"{p}"' for p in PIP_PACKAGES)

    cell2 = join_modules(CELL2_USER_BLOCK, module_source("config").split("\n@lru_cache")[0].replace(
        "from functools import lru_cache\n", "") + CELL2_BUILD_SETTINGS)
    cell3 = join_modules(module_source("model_service"), CELL3_LOAD_CODE)
    cell5 = join_modules(module_source("structure"), module_source("prompting"))
    cell7 = join_modules(
        module_source("errors"),
        module_source("schemas"),
        module_source("main").rsplit("\napp = create_app()", 1)[0],
    )

    # NOTE: model_service imports app.prompting, and process_pdf imports app.structure; in the
    # notebook these names are plain globals, so the cell order below only has to define them
    # before they are *called*. Cell 3 defines classes only; Cells 4/5 define the rest.
    return [
        md(INTRO_MD),
        md(CELL1_MD), code(CELL1_CODE.replace("__PIP_ARGS__", pip_args)),
        md(CELL2_MD), code(cell2),
        md(CELL3_MD), code(cell3),
        md(CELL4_MD), code(join_modules(module_source("document_processor"))),
        md(CELL5_MD), code(cell5),
        md(CELL6_MD), code(CELL6_CODE),
        md(CELL7_MD), code(cell7 + "\n\n" + CELL7_CODE),
        md(CELL8_MD), code(CELL8_CODE),
        md(CELL8B_MD), code(CELL8B_CODE),
    ]


def main() -> None:
    notebook = {
        "cells": build_cells(),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "slm_document_analysis_backend.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = HERE / "slm_document_analysis_backend.ipynb"
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "requirements.txt").write_text(
        "# Packages installed by Cell 1 of the notebook (keep in sync via build_notebook.py).\n"
        "# torch is NOT listed: Colab provides a CUDA build.\n" + "\n".join(PIP_PACKAGES) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out.name} ({len(notebook['cells'])} cells) and requirements.txt")


if __name__ == "__main__":
    main()
