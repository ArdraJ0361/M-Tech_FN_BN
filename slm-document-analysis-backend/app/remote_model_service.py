"""Client for the fine-tuned model server (``POST /predict`` and ``GET /health``).

The model runs as its own local server (the model repo's ``scripts/start_server.sh``). That
server builds the prompt itself, cuts the context to what the model can read, generates the
answer and cleans it up. This module therefore sends only the extracted document text
(``context``) and the user's question, and never builds a prompt.

It implements the same ``AnswerGenerator`` interface as the in-process ``ModelService``, so
``main.py`` and the frontend do not care which one is used.

Only the standard library is used (``urllib``). TLS certificates are verified for ``https``
URLs, every call has a timeout, and the API key (if any) is sent in ``X-API-Key`` and never
logged.
"""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from typing import Any, Optional

from app.config import Settings
from app.model_service import (
    GenerationResult,
    InferenceError,
    InferenceTimeoutError,
    ModelNotReadyError,
    ModelServiceError,
)
from app.prompting import is_unanswerable

logger = logging.getLogger(__name__)

# Limits of the model server's request model (src/serving/server.py).
SERVER_MAX_CONTEXT_CHARS = 200_000
SERVER_MAX_QUESTION_CHARS = 2_000

_HEALTH_TIMEOUT_SECONDS = 3.0
_MAX_RESPONSE_BYTES = 1_000_000


class ModelAuthError(ModelServiceError):
    code = "model_auth_failed"


class RemoteModelService:
    """Answers questions by calling the model server's ``/predict`` endpoint."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base = settings.model_server_url
        self._last_health: Optional[dict[str, Any]] = None
        self._last_error: Optional[str] = None

    # -- HTTP -------------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._settings.model_server_api_key:
            headers["X-API-Key"] = self._settings.model_server_api_key
        return headers

    def _call(self, method: str, path: str, body: Optional[dict], timeout: float) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{self._base}{path}", data=data, headers=self._headers(), method=method
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("response too large")
        return json.loads(raw.decode("utf-8"))

    # -- state -------------------------------------------------------------------------
    def _refresh_health(self) -> bool:
        try:
            health = self._call("GET", "/health", None, _HEALTH_TIMEOUT_SECONDS)
            if not isinstance(health, dict) or health.get("status") != "ok":
                raise ValueError("unexpected /health response")
        except Exception as exc:  # any failure means "not ready"; details go to the log only
            logger.warning("Model server health check failed: %s", type(exc).__name__)
            self._last_health = None
            self._last_error = (
                f"The model server is not reachable at {self._base}. Start it with "
                "'bash scripts/start_server.sh' in the model project and check MODEL_SERVER_URL."
            )
            return False
        self._last_health = health
        self._last_error = None
        return True

    @property
    def is_ready(self) -> bool:
        return self._refresh_health()

    @property
    def model_name(self) -> str:
        model = (self._last_health or {}).get("model")
        return str(model) if model else "model server"

    @property
    def status(self) -> dict[str, Any]:
        ready = self._refresh_health()
        return {
            "model_ready": ready,
            "model_name": self.model_name if ready else None,
            "adapter_loaded": False,  # /health does not say; not claimed
            "device": None,
            "load_error": None if ready else self._last_error,
        }

    # -- inference ---------------------------------------------------------------------
    def generate_answer(self, document_markdown: str, question: str) -> GenerationResult:
        context = document_markdown.strip()
        question = question.strip()
        if len(question) > SERVER_MAX_QUESTION_CHARS:
            raise InferenceError(
                f"The question must be at most {SERVER_MAX_QUESTION_CHARS} characters."
            )
        # The model server rejects bigger contexts; it reads far less than this anyway.
        capped = len(context) > SERVER_MAX_CONTEXT_CHARS
        sent = context[:SERVER_MAX_CONTEXT_CHARS] if capped else context

        try:
            body = self._call(
                "POST",
                "/predict",
                {"context": sent, "question": question},
                self._settings.generation_timeout_seconds,
            )
        except urllib.error.HTTPError as exc:
            logger.error("Model server returned HTTP %s", exc.code)
            if exc.code in (401, 403):
                raise ModelAuthError(
                    "The model server rejected the API key. Check MODEL_SERVER_API_KEY."
                ) from exc
            raise InferenceError(f"The model server returned an error (HTTP {exc.code}).") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise InferenceTimeoutError(
                f"The model did not answer within {self._settings.generation_timeout_seconds:.0f} seconds."
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise InferenceTimeoutError(
                    f"The model did not answer within {self._settings.generation_timeout_seconds:.0f} seconds."
                ) from exc
            logger.error("Model server unreachable: %s", type(exc.reason).__name__)
            raise ModelNotReadyError(
                f"The model server is not reachable at {self._base}. Start it with "
                "'bash scripts/start_server.sh' in the model project."
            ) from exc
        except (ValueError, OSError) as exc:  # bad JSON, oversized body, broken connection
            logger.error("Unusable response from the model server: %s", type(exc).__name__)
            raise InferenceError("The model server returned an unusable response.") from exc

        answer = body.get("answer") if isinstance(body, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise InferenceError("The model produced an empty answer.")
        answer = answer.strip()
        server_truncated = body.get("context_truncated")
        if not isinstance(server_truncated, bool):
            server_truncated = False

        return GenerationResult(
            answer=answer,
            unanswerable=is_unanswerable(answer),
            model_name=self.model_name,
            prompt_tokens=None,
            generated_tokens=None,
            document_truncated=server_truncated or capped,
            document_chars_total=None,
            document_chars_used=None,
            device="remote",
            dtype="remote",
        )
