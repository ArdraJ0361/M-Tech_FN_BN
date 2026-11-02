"""Tests for RemoteModelService against a fake model server (real HTTP on 127.0.0.1)."""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model_service import InferenceError, InferenceTimeoutError, ModelNotReadyError
from app.remote_model_service import ModelAuthError, RemoteModelService
from tests.conftest import make_structured_pdf


class FakeModelServer:
    """Mimics the model repo's server: GET /health and POST /predict (X-API-Key optional)."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.requests: list[dict] = []
        self.headers_seen: list[dict] = []
        self.reply: dict | str = {"answer": "15%", "latency_ms": 5, "context_truncated": False}
        self.status = 200
        self.delay = 0.0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence test output
                pass

            def _send(self, status: int, payload) -> None:
                raw = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, {"status": "ok", "model": "Qwen/Qwen1.5-0.5B", "task": "qa"})
                else:
                    self._send(404, {"detail": "Not Found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                outer.headers_seen.append(dict(self.headers))
                if outer.api_key and self.headers.get("X-API-Key") != outer.api_key:
                    self._send(401, {"detail": "Missing or invalid X-API-Key header."})
                    return
                time.sleep(outer.delay)
                self._send(outer.status, outer.reply)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def server():
    srv = FakeModelServer()
    yield srv
    srv.close()


def make_service(url: str, **kwargs) -> RemoteModelService:
    return RemoteModelService(Settings(_env_file=None, model_server_url=url, **kwargs))


def closed_port_url() -> str:
    """A localhost URL that refuses connections immediately (the port was just released)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def header(headers: dict, name: str) -> str | None:
    return next((v for k, v in headers.items() if k.lower() == name.lower()), None)


def test_health_reports_ready_and_model_name(server):
    svc = make_service(server.url)
    assert svc.status["model_ready"] is True
    assert svc.model_name == "Qwen/Qwen1.5-0.5B"
    assert svc.is_ready is True


def test_answer_sends_only_context_and_question(server):
    svc = make_service(server.url)
    result = svc.generate_answer("  # Title\n\nRevenue is 12 crore.  ", "  How much?  ")
    assert server.requests == [{"context": "# Title\n\nRevenue is 12 crore.", "question": "How much?"}]
    assert result.answer == "15%"
    assert result.unanswerable is False
    assert result.document_truncated is False
    # Numbers the model server does not report are None, never invented.
    assert result.prompt_tokens is None and result.generated_tokens is None
    assert result.document_chars_used is None and result.document_chars_total is None


def test_no_api_key_header_when_none_configured(server):
    make_service(server.url).generate_answer("doc", "q")
    assert header(server.headers_seen[0], "X-API-Key") is None


def test_api_key_is_sent_in_header():
    srv = FakeModelServer(api_key="secret-key")
    try:
        make_service(srv.url, model_server_api_key="secret-key").generate_answer("doc", "q")
        assert header(srv.headers_seen[0], "X-API-Key") == "secret-key"
    finally:
        srv.close()


def test_wrong_api_key_is_a_clear_auth_error():
    srv = FakeModelServer(api_key="secret-key")
    try:
        with pytest.raises(ModelAuthError) as err:
            make_service(srv.url, model_server_api_key="wrong").generate_answer("doc", "q")
        assert "secret-key" not in err.value.message and "wrong" not in err.value.message
    finally:
        srv.close()


def test_unanswerable_answer_is_flagged(server):
    server.reply = {"answer": "unanswerable", "latency_ms": 3, "context_truncated": False}
    assert make_service(server.url).generate_answer("doc", "q").unanswerable is True


def test_context_truncated_is_passed_on(server):
    server.reply = {"answer": "x", "latency_ms": 3, "context_truncated": True}
    assert make_service(server.url).generate_answer("doc", "q").document_truncated is True


def test_server_down_is_not_ready_and_gives_503_style_error():
    svc = make_service(closed_port_url())
    assert svc.is_ready is False
    status = svc.status
    assert status["model_ready"] is False and "not reachable" in status["load_error"]
    with pytest.raises(ModelNotReadyError):
        svc.generate_answer("doc", "q")


def test_timeout_raises_timeout_error(server):
    server.delay = 1.5
    svc = make_service(server.url, generation_timeout_seconds=0.3)
    with pytest.raises(InferenceTimeoutError):
        svc.generate_answer("doc", "q")


@pytest.mark.parametrize("reply", ["not json at all", {"latency_ms": 1}, {"answer": "   "}, ["x"]])
def test_bad_responses_are_inference_errors(server, reply):
    server.reply = reply
    with pytest.raises(InferenceError):
        make_service(server.url).generate_answer("doc", "q")


def test_server_error_status_is_an_inference_error(server):
    server.status = 500
    server.reply = {"detail": "boom with secrets"}
    with pytest.raises(InferenceError) as err:
        make_service(server.url).generate_answer("doc", "q")
    assert "secrets" not in err.value.message


def test_url_validation_rejects_plain_http_to_other_hosts():
    with pytest.raises(ValueError):
        Settings(_env_file=None, model_server_url="http://example.com:8765")
    with pytest.raises(ValueError):
        Settings(_env_file=None, model_server_url="ftp://127.0.0.1")
    assert Settings(_env_file=None, model_server_url="https://example.com").model_server_url == "https://example.com"
    assert Settings(_env_file=None, model_server_url="http://localhost:8765/").model_server_url == "http://localhost:8765"


# --- end to end through the real FastAPI app: PDF + question -> /predict -> response ----------
def test_full_flow_pdf_to_predict_to_response(server):
    server.reply = {"answer": "Revenue is 120 for North.", "latency_ms": 7, "context_truncated": False}
    app = create_app(settings=Settings(_env_file=None, model_server_url=server.url))
    client = TestClient(app)

    assert client.get("/health").json()["model_ready"] is True

    response = client.post(
        "/answer",
        files={"file": ("report.pdf", make_structured_pdf(), "application/pdf")},
        data={"question": "What is the revenue of North?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Revenue is 120 for North."
    assert body["model_name"] == "Qwen/Qwen1.5-0.5B"
    assert body["prompt_tokens"] is None and body["document_chars_used"] is None

    sent = server.requests[0]
    assert set(sent) == {"context", "question"}
    assert sent["question"] == "What is the revenue of North?"
    assert "# Annual Report 2024" in sent["context"] and "| North | 120 | 30 |" in sent["context"]
    assert "You are a senior" not in sent["context"]  # no prompt is built on our side


def test_full_flow_with_model_server_down_returns_503():
    app = create_app(settings=Settings(_env_file=None, model_server_url=closed_port_url()))
    client = TestClient(app)
    assert client.get("/health").json()["model_ready"] is False
    response = client.post(
        "/answer",
        files={"file": ("report.pdf", make_structured_pdf(), "application/pdf")},
        data={"question": "What is the revenue of North?"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_not_ready"
