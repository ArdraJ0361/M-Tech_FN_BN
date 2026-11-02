"""FastAPI application: upload a PDF and a question in ONE request, get an answer.

Flow of ``POST /answer``::

    PDF + question --> structure-preserving extraction (Markdown)
                   --> POST {context, question} to the model server's /predict
                       (the model server builds the prompt and runs the fine-tuned SLM)
                   --> answer

There is no retrieval, no chunking and no server-side storage: nothing is kept after the
response is sent.

Run with:  ``uvicorn app.main:app --host 127.0.0.1 --port 8000``
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.document_processor import (
    DocumentProcessingError,
    EmptyFileError,
    FileTooLargeError,
    MalformedPDFError,
    NoExtractableTextError,
    TooManyPagesError,
    UnsupportedFileTypeError,
    process_pdf,
    sanitize_filename,
)
from app.errors import ApiError
from app.model_service import (
    AnswerGenerator,
    InferenceTimeoutError,
    ModelNotConfiguredError,
    ModelNotReadyError,
    ModelService,
    ModelServiceError,
)
from app.schemas import (
    AnswerLocation,
    AnswerResponse,
    DocumentInfo,
    ErrorResponse,
    HealthResponse,
    TableCell,
)
from app.remote_model_service import RemoteModelService
from app.structure import locate_answer

logger = logging.getLogger("app")

_UPLOAD_STATUS = {
    UnsupportedFileTypeError: 415,
    EmptyFileError: 400,
    FileTooLargeError: 413,
    TooManyPagesError: 413,
    MalformedPDFError: 422,
    NoExtractableTextError: 422,
}

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


def create_app(
    settings: Optional[Settings] = None,
    model_service: Optional[AnswerGenerator] = None,
) -> FastAPI:
    """Application factory. ``settings`` and ``model_service`` can be injected (used by tests)."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service: AnswerGenerator = model_service or RemoteModelService(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Only the built-in ModelService is loaded here; injected services manage themselves.
        if isinstance(service, ModelService) and settings.load_model_on_startup:
            try:
                await run_in_threadpool(service.load)
            except ModelServiceError as exc:
                # The API still starts so /health can report the problem clearly.
                logger.error("Model not available at startup: %s", exc.message)
        yield

    app = FastAPI(
        title="SLM Document Analysis API",
        description="Send a PDF and a question in one request; a small language model answers "
        "from the document's structure-preserving text. Nothing is stored on the server.",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    # --- error handlers ------------------------------------------------------
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Report only field location and a short message; never echo the submitted input.
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}"
            for e in exc.errors()
        )
        return _error_response(422, "validation_error", f"Invalid request: {problems}")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled error", exc_info=exc)
        return _error_response(500, "internal_error", "An unexpected error occurred.")

    # --- endpoints -------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        status = service.status
        return HealthResponse(
            status="ok",
            model_ready=bool(status.get("model_ready")),
            model_name=status.get("model_name"),
            adapter_loaded=bool(status.get("adapter_loaded")),
            device=status.get("device"),
            model_error=status.get("load_error"),
        )

    @app.post(
        "/answer",
        response_model=AnswerResponse,
        responses=_ERROR_RESPONSES,
        tags=["question answering"],
    )
    async def answer_question(
        file: UploadFile = File(..., description="A text-based PDF."),
        question: str = Form(..., description="The question about the PDF."),
    ) -> AnswerResponse:
        question = question.strip()
        if not question:
            raise ApiError(422, "empty_question", "The question must not be empty.")
        if len(question) > settings.max_question_chars:
            raise ApiError(
                422,
                "question_too_long",
                f"The question must be at most {settings.max_question_chars} characters.",
            )

        # Fail fast, before spending time on extraction, if the model cannot answer.
        if not service.is_ready:
            status = service.status
            raise ApiError(
                503,
                "model_not_ready",
                status.get("load_error")
                or "The language model is not ready. Check GET /health for details.",
            )

        # Read at most max_bytes + 1 so oversized uploads are detected without buffering them all.
        data = await file.read(settings.max_upload_bytes + 1)
        filename = sanitize_filename(file.filename)
        try:
            structured = await run_in_threadpool(
                process_pdf,
                file.filename,
                data,
                max_bytes=settings.max_upload_bytes,
                min_text_chars=settings.min_text_chars,
                max_pages=settings.max_pages,
            )
        except DocumentProcessingError as exc:
            status_code = next(
                (s for cls, s in _UPLOAD_STATUS.items() if isinstance(exc, cls)), 422
            )
            logger.info("Upload rejected: %s", exc.code)
            raise ApiError(status_code, exc.code, exc.message) from exc
        except Exception as exc:
            logger.exception("Unexpected failure while processing PDF")
            raise ApiError(500, "processing_failed", "The document could not be processed.") from exc

        markdown = structured.markdown
        logger.info(
            "Document extracted pages=%d headings=%d tables=%d chars=%d",
            structured.page_count, structured.heading_count, structured.table_count, len(markdown),
        )

        try:
            started = time.monotonic()
            result = await run_in_threadpool(service.generate_answer, markdown, question)
            elapsed = time.monotonic() - started
        except InferenceTimeoutError as exc:
            raise ApiError(504, exc.code, exc.message) from exc
        except (ModelNotReadyError, ModelNotConfiguredError) as exc:
            raise ApiError(503, exc.code, exc.message) from exc
        except ModelServiceError as exc:
            raise ApiError(500, exc.code, exc.message) from exc

        # Plain string matching of the answer against the extracted elements (not model output).
        locations: list[AnswerLocation] = []
        if not result.unanswerable:
            locations = [
                AnswerLocation(
                    page_number=loc.page_number,
                    element_type=loc.element_type,
                    section_path=list(loc.section_path),
                    snippet=loc.snippet,
                    table_cell=(
                        TableCell(
                            value=loc.table_cell.value,
                            row_label=loc.table_cell.row_label,
                            column_label=loc.table_cell.column_label,
                        )
                        if loc.table_cell
                        else None
                    ),
                )
                for loc in locate_answer(structured.elements, result.answer)
            ]

        limit = settings.max_markdown_response_chars
        message: Optional[str] = None
        if result.unanswerable and result.document_truncated:
            message = (
                "The model found no answer in the beginning of the document, which is all it "
                "could read (about 1-2 pages). The answer may be in the part that was cut off."
            )
        elif result.unanswerable:
            message = "The model judged that the document does not contain the answer."
        elif result.document_truncated:
            message = (
                "The document is longer than the model can read (about 1-2 pages), so only its "
                "beginning was used. The answer may be wrong or missing information."
            )

        return AnswerResponse(
            question=question,
            answer=result.answer,
            unanswerable=result.unanswerable,
            message=message,
            model_name=result.model_name,
            inference_time_seconds=round(elapsed, 3),
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            document=DocumentInfo(
                filename=filename,
                page_count=structured.page_count,
                heading_count=structured.heading_count,
                table_count=structured.table_count,
                character_count=len(markdown),
                word_count=len(markdown.split()),
                empty_page_numbers=structured.empty_page_numbers,
            ),
            document_truncated=result.document_truncated,
            document_chars_used=result.document_chars_used,
            document_chars_total=result.document_chars_total,
            structured_text=markdown[:limit],
            structured_text_truncated=len(markdown) > limit,
            answer_locations=locations,
        )

    return app


app = create_app()
