"""Application settings, read from environment variables (and an optional ``.env`` file).

Every value has a safe default except the model location: ``MODEL_NAME_OR_PATH``
and ``ADAPTER_PATH`` are intentionally empty because they are specific to your
training results and must be supplied by you. See ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunable settings. Environment variable names are the upper-case field names."""

    # ``protected_namespaces=()`` allows field names that start with ``model_``.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Model ---------------------------------------------------------------
    # Hugging Face model id or local directory of the BASE model used in training.
    model_name_or_path: Optional[str] = None
    # Directory containing the saved LoRA adapter (adapter_config.json, ...).
    adapter_path: Optional[str] = None
    # Optional tokenizer location; defaults to the base model's tokenizer.
    tokenizer_name_or_path: Optional[str] = None
    # auto | cpu | mps | cuda  ("auto" prefers cuda, then mps, then cpu)
    model_device: str = "auto"
    # auto | float32 | float16 | bfloat16
    model_dtype: str = "auto"
    # Merge the LoRA weights into the base model after loading (faster inference).
    merge_adapter: bool = False
    trust_remote_code: bool = False
    # Try to load the model when the server starts.
    load_model_on_startup: bool = True

    # --- Generation ------------------------------------------------------------
    # Decoding is always greedy (as in the training notebook's evaluation).
    # The notebook evaluated with 32; answers are short, so a small limit is enough.
    max_new_tokens: int = Field(64, ge=1, le=1024)
    # Upper bound on prompt tokens (document + question), also capped by the model's
    # context window. The whole document is sent; if it does not fit it is cut at the
    # END and the response says so. Larger values need more RAM / GPU memory.
    max_input_tokens: int = Field(8192, ge=256)
    generation_timeout_seconds: float = Field(120.0, gt=0)

    # --- Upload / extraction ---------------------------------------------------
    max_upload_mb: int = Field(20, ge=1, le=500)
    max_pages: int = Field(100, ge=1)
    # Minimum number of letters/digits that must be extracted for a PDF to count
    # as containing "meaningful text".
    min_text_chars: int = Field(20, ge=1)
    max_question_chars: int = Field(2000, ge=1)
    # Size limit of the Markdown text echoed back to the browser (the model still gets
    # the whole document, subject to MAX_INPUT_TOKENS).
    max_markdown_response_chars: int = Field(200_000, ge=1000)

    # --- API -------------------------------------------------------------------
    # Comma-separated list of allowed browser origins.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    log_level: str = "INFO"

    @field_validator(
        "model_name_or_path", "adapter_path", "tokenizer_name_or_path", mode="before"
    )
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("model_device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("MODEL_DEVICE must be one of: auto, cpu, mps, cuda")
        return value

    @field_validator("model_dtype")
    @classmethod
    def _check_dtype(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError(
                "MODEL_DTYPE must be one of: auto, float32, float16, bfloat16"
            )
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
