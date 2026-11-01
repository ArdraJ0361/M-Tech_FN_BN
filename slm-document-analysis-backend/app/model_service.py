"""Model service: loads the configured SLM (+ optional LoRA adapter) once and answers questions.

Kept independent of PDF extraction. The rest of the app only depends on the small
``AnswerGenerator`` protocol below, so a remote inference implementation (e.g. an HTTP
client for a model hosted elsewhere) could replace ``ModelService`` without touching other
modules. No remote implementation exists.

Inference mirrors the training/evaluation notebook:

* the prompt is the plain-text training template (no chat template),
* decoding is greedy (``do_sample=False``),
* only newly generated tokens are decoded,
* the FIRST LINE of the generated text is the answer.

Heavy libraries (torch / transformers / peft) are imported lazily inside ``load`` so the
unit tests and the API can start without them being initialised.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from app.config import Settings
from app.prompting import build_prompt, first_line, is_unanswerable, prompt_parts

logger = logging.getLogger(__name__)


# --- Errors -------------------------------------------------------------------
class ModelServiceError(Exception):
    code = "model_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotConfiguredError(ModelServiceError):
    code = "model_not_configured"


class ModelNotReadyError(ModelServiceError):
    code = "model_not_ready"


class ModelLoadError(ModelServiceError):
    code = "model_load_failed"


class InferenceError(ModelServiceError):
    code = "inference_failed"


class InferenceTimeoutError(ModelServiceError):
    code = "inference_timeout"


# --- Public result / interface --------------------------------------------------
@dataclass(frozen=True)
class GenerationResult:
    answer: str                 # first line of the generated text (may be "unanswerable")
    unanswerable: bool          # the model produced the training target "unanswerable"
    model_name: str
    prompt_tokens: int
    generated_tokens: int
    document_truncated: bool    # the document text had to be shortened to fit the model
    document_chars_total: int
    document_chars_used: int
    device: str
    dtype: str


class AnswerGenerator(Protocol):
    """Interface the API depends on."""

    @property
    def is_ready(self) -> bool: ...

    @property
    def model_name(self) -> str: ...

    @property
    def status(self) -> dict[str, Any]: ...

    def generate_answer(self, document_markdown: str, question: str) -> GenerationResult: ...


# --- Implementation -----------------------------------------------------------
class ModelService:
    """Local Transformers (+ PEFT) inference. The model is loaded once and reused."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._tokenizer: Any = None
        self._device = "unloaded"
        self._dtype = "unloaded"
        self._adapter_loaded = False
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()  # one generation at a time

    # -- state -------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def model_name(self) -> str:
        base = self._settings.model_name_or_path or "unconfigured"
        # Local paths are reduced to their last component to avoid leaking filesystem layout.
        if os.path.isabs(base) or base.startswith(("~", ".")):
            base = os.path.basename(os.path.normpath(os.path.expanduser(base)))
        if self._adapter_loaded and self._settings.adapter_path:
            adapter = os.path.basename(os.path.normpath(self._settings.adapter_path))
            return f"{base} + LoRA adapter '{adapter}'"
        return base

    @property
    def status(self) -> dict[str, Any]:
        return {
            "model_ready": self.is_ready,
            "model_name": self.model_name if self._settings.model_name_or_path else None,
            "adapter_loaded": self._adapter_loaded,
            "device": self._device,
            "load_error": self._load_error,
        }

    # -- loading -----------------------------------------------------------------
    def load(self) -> None:
        """Load tokenizer, base model and (optionally) the LoRA adapter.

        Raises ``ModelNotConfiguredError`` or ``ModelLoadError``. Safe to call again;
        it does nothing if the model is already loaded.
        """
        with self._load_lock:
            if self.is_ready:
                return
            s = self._settings
            if not s.model_name_or_path:
                msg = (
                    "MODEL_NAME_OR_PATH is not set. Set it (and ADAPTER_PATH if you use a "
                    "LoRA adapter) in the .env file."
                )
                self._load_error = msg
                raise ModelNotConfiguredError(msg)
            if s.adapter_path and not os.path.isdir(os.path.expanduser(s.adapter_path)):
                msg = "ADAPTER_PATH does not point to an existing directory."
                self._load_error = msg
                raise ModelLoadError(msg)

            try:
                self._load_unchecked()
                self._load_error = None
            except ModelServiceError as exc:
                self._load_error = exc.message
                raise
            except Exception as exc:
                logger.exception("Model loading failed")
                self._model = None
                self._tokenizer = None
                self._load_error = f"Model loading failed ({type(exc).__name__}). See server logs."
                raise ModelLoadError(self._load_error) from exc

    def _load_unchecked(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s = self._settings
        device = self._resolve_device(torch)
        dtype_name, dtype = self._resolve_dtype(torch, device)
        logger.info("Loading model on device=%s dtype=%s", device, dtype_name)

        tokenizer = AutoTokenizer.from_pretrained(
            s.tokenizer_name_or_path or s.model_name_or_path,
            trust_remote_code=s.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            s.model_name_or_path,
            dtype=dtype,
            trust_remote_code=s.trust_remote_code,
        )

        adapter_loaded = False
        if s.adapter_path:
            from peft import PeftModel

            logger.info("Loading LoRA adapter")
            model = PeftModel.from_pretrained(model, os.path.expanduser(s.adapter_path))
            if s.merge_adapter:
                model = model.merge_and_unload()
            adapter_loaded = True

        model.to(device)
        model.eval()

        # Same as the training notebook.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._dtype = dtype_name
        self._adapter_loaded = adapter_loaded
        logger.info("Model ready")

    def _resolve_device(self, torch: Any) -> str:
        requested = self._settings.model_device
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise ModelLoadError("MODEL_DEVICE=cuda but CUDA is not available on this machine.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ModelLoadError("MODEL_DEVICE=mps but Apple MPS is not available on this machine.")
        return requested

    def _resolve_dtype(self, torch: Any, device: str) -> tuple[str, Any]:
        requested = self._settings.model_dtype
        if requested == "auto":
            requested = "float32" if device == "cpu" else "float16"
        return requested, getattr(torch, requested)

    # -- inference -----------------------------------------------------------------
    def generate_answer(self, document_markdown: str, question: str) -> GenerationResult:
        """Build the training-format prompt, generate greedily and decode only new tokens."""
        if not self.is_ready:
            raise ModelNotReadyError(
                self._load_error or "The model is not loaded yet. Check /health for details."
            )
        import torch

        s = self._settings
        try:
            input_ids, used_chars = self._encode_within_budget(document_markdown, question)
        except ModelServiceError:
            raise
        except Exception as exc:
            logger.exception("Prompt construction failed")
            raise InferenceError("Failed to build the model prompt.") from exc
        prompt_tokens = int(input_ids.shape[-1])

        started = time.monotonic()
        try:
            with self._generate_lock, torch.inference_mode():
                output = self._model.generate(
                    input_ids=input_ids.to(self._device),
                    attention_mask=torch.ones_like(input_ids).to(self._device),
                    max_new_tokens=s.max_new_tokens,
                    max_time=s.generation_timeout_seconds,
                    do_sample=False,  # greedy, as in the notebook's evaluation
                    pad_token_id=self._tokenizer.pad_token_id,
                )
        except Exception as exc:
            logger.exception("Generation failed")
            raise InferenceError(f"Generation failed ({type(exc).__name__}).") from exc
        elapsed = time.monotonic() - started

        new_tokens = output[0][prompt_tokens:]
        generated_tokens = int(new_tokens.shape[-1])
        # max_time stops generation early; an unfinished, timed-out run is a timeout.
        hit_eos = generated_tokens > 0 and int(new_tokens[-1]) == self._tokenizer.eos_token_id
        if (
            elapsed >= s.generation_timeout_seconds
            and not hit_eos
            and generated_tokens < s.max_new_tokens
        ):
            raise InferenceTimeoutError(
                f"Answer generation exceeded {s.generation_timeout_seconds:.0f} seconds."
            )

        decoded = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        answer = first_line(decoded)
        if not answer:
            raise InferenceError("The model produced an empty answer.")

        return GenerationResult(
            answer=answer,
            unanswerable=is_unanswerable(answer),
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            document_truncated=used_chars < len(document_markdown.strip()),
            document_chars_total=len(document_markdown.strip()),
            document_chars_used=used_chars,
            device=self._device,
            dtype=self._dtype,
        )

    def _token_budget(self) -> int:
        """Max prompt tokens: MAX_INPUT_TOKENS, capped by the model's context window."""
        s = self._settings
        limit = getattr(self._tokenizer, "model_max_length", None)
        if not isinstance(limit, int) or limit > 10_000_000:
            limit = getattr(getattr(self._model, "config", None), "max_position_embeddings", None)
        budget = s.max_input_tokens
        if isinstance(limit, int) and 0 < limit <= 10_000_000:
            budget = min(budget, limit - s.max_new_tokens)
        return budget

    def _encode(self, text: str) -> Any:
        # Same call as the training notebook: tok(prompt)["input_ids"] (default special tokens).
        return self._tokenizer(text, return_tensors="pt")["input_ids"]

    def _encode_within_budget(self, document: str, question: str) -> tuple[Any, int]:
        """Encode the full prompt; if too long, cut the END of the document (never the
        question or ``### Answer:``). Returns ``(input_ids, document_chars_used)``."""
        document = document.strip()
        budget = self._token_budget()
        ids = self._encode(build_prompt(document, question))
        if ids.shape[-1] <= budget:
            return ids, len(document)

        prefix, suffix = prompt_parts(question)

        def fits(n: int) -> tuple[bool, Any]:
            encoded = self._encode(prefix + document[:n].rstrip() + suffix)
            return encoded.shape[-1] <= budget, encoded

        if not fits(0)[0]:
            raise InferenceError(
                "The question is too long to fit the model's input limit. Shorten it."
            )
        low, high = 0, len(document)  # largest n with fits(n); fits(0) is True, fits(high) False
        while high - low > 1:
            mid = (low + high) // 2
            if fits(mid)[0]:
                low = mid
            else:
                high = mid
        # Avoid cutting inside a word: back up to the previous whitespace when possible.
        cut = low
        boundary = max(document.rfind(" ", 0, cut), document.rfind("\n", 0, cut))
        if boundary > cut * 0.9:
            cut = boundary
        ok, encoded = fits(cut)
        if not ok:  # extremely unlikely; fall back to the exact fitting length
            cut = low
            _, encoded = fits(cut)
        logger.info("Document truncated to %d of %d characters", cut, len(document))
        return encoded, len(document[:cut].rstrip())
