"""API tests with a fake model (no real model is loaded)."""

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model_service import InferenceError, InferenceTimeoutError
from tests.conftest import FakeModelService, make_pdf, make_structured_pdf


def post(client, data=None, filename="report.pdf", question="What was the profit?", content_type="application/pdf"):
    files = {"file": (filename, data if data is not None else make_structured_pdf(), content_type)}
    return client.post("/answer", files=files, data={"question": question})


# --- health ----------------------------------------------------------------------------
def test_health_reflects_model_state(client, fake_model):
    assert client.get("/health").json()["model_ready"] is True
    fake_model.ready = False
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_ready"] is False
    assert body["model_error"]


def test_removed_two_step_endpoints_are_gone(client):
    assert client.post("/documents").status_code == 404
    assert client.get("/documents/abc/text").status_code == 404


# --- success -----------------------------------------------------------------------------
def test_single_request_returns_answer_and_structure(client, fake_model):
    fake_model.answer = "30"
    res = post(client)
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "30"
    assert body["question"] == "What was the profit?"
    assert body["unanswerable"] is False
    assert body["model_name"] == "fake-model"
    assert body["inference_time_seconds"] >= 0
    doc = body["document"]
    assert doc["filename"] == "report.pdf"
    assert doc["page_count"] == 2
    assert doc["table_count"] == 1
    assert doc["heading_count"] == 4
    assert body["structured_text"].startswith("# Annual Report 2024")
    assert "| North | 120 | 30 |" in body["structured_text"]
    assert body["structured_text_truncated"] is False
    assert body["disclaimer"]


def test_model_receives_markdown_and_trimmed_question(client, fake_model):
    post(client, question="  What was the profit?  ")
    [(markdown, question)] = fake_model.calls
    assert question == "What was the profit?"
    assert "## 2. Results" in markdown and "| South | 90 | 15 |" in markdown


def test_answer_locations_found_by_string_match(client, fake_model):
    fake_model.answer = "120"
    body = post(client).json()
    [loc] = body["answer_locations"]
    assert loc["page_number"] == 2
    assert loc["element_type"] == "table"
    assert loc["section_path"] == ["Annual Report 2024", "2. Results"]
    assert loc["table_cell"] == {"value": "120", "row_label": "North", "column_label": "Revenue"}


def test_no_location_when_answer_text_not_in_document(client, fake_model):
    fake_model.answer = "completely unrelated words"
    assert post(client).json()["answer_locations"] == []


def test_unanswerable_is_reported_without_locations(client, fake_model):
    fake_model.answer = "unanswerable"
    fake_model.unanswerable = True
    body = post(client).json()
    assert body["unanswerable"] is True
    assert body["answer_locations"] == []
    assert "does not contain" in body["message"]


def test_each_request_is_independent(client, fake_model):
    post(client, data=make_pdf(["Document A talks about apples and oranges."]))
    post(client, data=make_pdf(["Document B talks about trains and planes."]))
    (a, _), (b, _) = fake_model.calls
    assert "apples" in a and "trains" not in a
    assert "trains" in b and "apples" not in b


def test_structured_text_is_bounded(fake_model):
    small = Settings(_env_file=None, max_upload_mb=1, max_markdown_response_chars=1000)
    c = TestClient(create_app(settings=small, model_service=fake_model))
    body = post(c, data=make_pdf(["word " * 800])).json()
    assert len(body["structured_text"]) == 1000
    assert body["structured_text_truncated"] is True


# --- validation errors -----------------------------------------------------------------------
def test_empty_question_rejected(client, fake_model):
    for q in ("", "   "):
        res = post(client, question=q)
        assert res.status_code == 422
        assert res.json()["error"]["code"] in {"empty_question", "validation_error"}
    assert fake_model.calls == []


def test_missing_question_or_file_rejected(client):
    assert client.post("/answer", data={"question": "x"}).status_code == 422
    files = {"file": ("a.pdf", make_structured_pdf(), "application/pdf")}
    assert client.post("/answer", files=files).status_code == 422


def test_overlong_question_rejected(client):
    res = post(client, question="q" * 2001)
    assert res.status_code == 422 and res.json()["error"]["code"] == "question_too_long"


def test_wrong_file_type_rejected(client):
    res = post(client, data=b"hello", filename="notes.txt", content_type="text/plain")
    assert res.status_code == 415
    assert res.json()["error"]["code"] == "unsupported_file_type"


def test_empty_upload_rejected(client):
    res = post(client, data=b"")
    assert res.status_code == 400 and res.json()["error"]["code"] == "empty_file"


def test_oversized_upload_rejected(client):
    res = post(client, data=b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10))
    assert res.status_code == 413 and res.json()["error"]["code"] == "file_too_large"


def test_malformed_pdf_rejected(client):
    res = post(client, data=b"%PDF-1.4\nthis is garbage, not a pdf")
    assert res.status_code == 422
    assert res.json()["error"]["code"] in {"malformed_pdf", "no_extractable_text"}


def test_pdf_without_text_rejected(client):
    res = post(client, data=make_pdf(["", ""]))
    assert res.status_code == 422 and res.json()["error"]["code"] == "no_extractable_text"


# --- model state / failures --------------------------------------------------------------------
def test_model_not_ready_returns_503_without_processing(client, fake_model):
    fake_model.ready = False
    res = post(client)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "model_not_ready"
    assert fake_model.calls == []


def test_inference_failure_returns_500(settings):
    class Failing(FakeModelService):
        def generate_answer(self, document_markdown, question):
            raise InferenceError("Generation failed (RuntimeError).")

    res = post(TestClient(create_app(settings=settings, model_service=Failing())))
    assert res.status_code == 500
    assert res.json()["error"] == {"code": "inference_failed", "message": "Generation failed (RuntimeError)."}


def test_inference_timeout_returns_504(settings):
    class Slow(FakeModelService):
        def generate_answer(self, document_markdown, question):
            raise InferenceTimeoutError("Answer generation exceeded 120 seconds.")

    res = post(TestClient(create_app(settings=settings, model_service=Slow())))
    assert res.status_code == 504 and res.json()["error"]["code"] == "inference_timeout"


def test_unexpected_error_is_generic(settings):
    class Boom(FakeModelService):
        def generate_answer(self, document_markdown, question):
            raise ZeroDivisionError("secret /home/user/path")

    c = TestClient(create_app(settings=settings, model_service=Boom()), raise_server_exceptions=False)
    r = post(c)
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "An unexpected error occurred."}}


# --- CORS ----------------------------------------------------------------------------------------
def test_cors_allows_configured_origin_only(client):
    ok = client.options(
        "/answer",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options(
        "/answer",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers
