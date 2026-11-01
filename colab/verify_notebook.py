"""Execute the notebook's code cells locally to verify them (no Colab needed).

* Cell 1 (``%pip``) is skipped; dependencies must already be installed in the current environment.
* Cell 2 is patched to use ``MODEL_NAME_OR_PATH`` (+ optional ``ADAPTER_PATH``) from the environment,
  CPU, and ``START_TUNNEL = False``.
* Cell 8 then starts the real uvicorn server on localhost and the script exercises the API over HTTP.

Usage:  MODEL_NAME_OR_PATH=<model> [ADAPTER_PATH=<dir>] python verify_notebook.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = json.loads((HERE / "slm_document_analysis_backend.ipynb").read_text(encoding="utf-8"))
codes = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
assert len(codes) == 9, len(codes)  # cells 1-8 plus 8b

model = os.environ["MODEL_NAME_OR_PATH"]
adapter = os.environ.get("ADAPTER_PATH", "")
codes[1] = (
    codes[1]
    .replace('MODEL_NAME_OR_PATH = ""', f"MODEL_NAME_OR_PATH = {model!r}")
    .replace('ADAPTER_PATH = ""', f"ADAPTER_PATH = {adapter!r}")
    .replace('MODEL_DEVICE = "auto"', 'MODEL_DEVICE = "cpu"')
    .replace("MAX_NEW_TOKENS = 64", "MAX_NEW_TOKENS = 8")
    .replace("START_TUNNEL = True", "START_TUNNEL = False")
    .replace("API_PORT = 8000", "API_PORT = 8771")
)

# Colab executes cells in the real __main__ module; pydantic resolves type hints through it,
# so run the cells in this script's own globals (which ARE __main__) rather than a private dict.
ns: dict = globals()
for number, source in enumerate(codes):
    if number == 0:
        continue  # %pip install
    if number == 8:
        continue  # cell 8b stops the server; do not run it yet
    label = str(number + 1)
    print(f"\n######## executing notebook cell {label}", flush=True)
    exec(compile(source, f"<notebook cell {label}>", "exec"), ns)

# ---- exercise the running server exactly like the frontend would ------------------------------
base = "http://127.0.0.1:8771"


def call(method: str, path: str, body: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(base + path, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def multipart(filename: str, data: bytes, question: str | None):
    boundary = uuid.uuid4().hex
    parts = [
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            "Content-Type: application/pdf\r\n\r\n"
        ).encode()
        + data
        + b"\r\n"
    ]
    if question is not None:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"question\"\r\n\r\n{question}\r\n".encode()
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


import pymupdf

pdf_doc = pymupdf.open()
page = pdf_doc.new_page()
page.insert_text((72, 80), "Meridian Report", fontsize=24, fontname="hebo")
page.insert_text((72, 120), "1. Summary", fontsize=16, fontname="hebo")
page.insert_text((72, 145), "The Meridian Report was published in 2021. Its main recommendation is longer library hours.", fontsize=11)
pdf = pdf_doc.tobytes()

failures: list[str] = []


def check(label: str, condition: bool, detail: object = "") -> None:
    print(("PASS " if condition else "FAIL ") + label, detail if not condition else "")
    if not condition:
        failures.append(label)


s, h = call("GET", "/health")
check("health ready", s == 200 and h["model_ready"] is True, h)

body, hdr = multipart("report.pdf", pdf, "What does the Meridian Report recommend?")
s, a = call("POST", "/answer", body, hdr)
check(
    "single request returns answer + structure",
    s == 200
    and a["answer"]
    and a["document"]["heading_count"] == 2
    and a["structured_text"].startswith("# Meridian Report")
    and "## 1. Summary" in a["structured_text"],
    a,
)
print("   answer:", a.get("answer"), "| model:", a.get("model_name"), "| time:", a.get("inference_time_seconds"))

body, hdr = multipart("bad.pdf", b"%PDF-1.4 not really a pdf", "x?")
s, e = call("POST", "/answer", body, hdr)
check("invalid pdf rejected", s == 422 and e["error"]["code"] == "malformed_pdf", e)

body, hdr = multipart("report.pdf", pdf, "   ")
s, e = call("POST", "/answer", body, hdr)
check("blank question rejected", s == 422 and e["error"]["code"] == "empty_question", e)

body, hdr = multipart("report.pdf", pdf, None)
s, e = call("POST", "/answer", body, hdr)
check("missing question rejected", s == 422, e)

s, e = call("POST", "/documents", b"", {})
check("old /documents endpoint is gone", s == 404, e)

ns["stop_server"]()
print("\nNotebook verification:", "FAILED " + str(failures) if failures else "all checks passed")
sys.exit(1 if failures else 0)
