"""Shared test fixtures: generated PDFs and a fake model service (no real model needed)."""

from __future__ import annotations

from typing import Any

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model_service import GenerationResult, ModelNotReadyError


def make_pdf(pages: list[str]) -> bytes:
    """Plain text-based PDF where each string becomes one page (single font size)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def make_structured_pdf() -> bytes:
    """A 2-page PDF with a title, headings, a paragraph, bullets and a ruled table.

    Page 1: title (24pt), "1. Introduction" (16pt), body paragraph (11pt), "1.1 Scope"
            (13pt), two bullet items.
    Page 2: "2. Results" (16pt), a ruled 3x3 table, a closing sentence.
    """
    doc = pymupdf.open()

    page = doc.new_page()
    page.insert_text((50, 80), "Annual Report 2024", fontsize=24, fontname="hebo")
    page.insert_text((50, 130), "1. Introduction", fontsize=16, fontname="hebo")
    body = [
        "This report describes the performance of the company during the year.",
        "Revenue increased because of strong demand in the northern region.",
    ]
    for i, line in enumerate(body):
        page.insert_text((50, 155 + i * 14), line, fontsize=11)
    page.insert_text((50, 215), "1.1 Scope", fontsize=13, fontname="hebo")
    page.insert_text((50, 240), "\u2022 Financial results", fontsize=11)
    page.insert_text((50, 254), "\u2022 Customer satisfaction", fontsize=11)

    page2 = doc.new_page()
    page2.insert_text((50, 80), "2. Results", fontsize=16, fontname="hebo")
    left, top, row_h = 50, 110, 24
    col_w = [140, 100, 100]
    rows = [["Region", "Revenue", "Profit"], ["North", "120", "30"], ["South", "90", "15"]]
    right = left + sum(col_w)
    for r in range(len(rows) + 1):  # horizontal rules
        y = top + r * row_h
        page2.draw_line((left, y), (right, y))
    x = left
    for w in [0] + col_w:  # vertical rules
        x += w
        page2.draw_line((x, top), (x, top + len(rows) * row_h))
    for r, row in enumerate(rows):
        x = left
        for c, cell in enumerate(row):
            page2.insert_text((x + 6, top + r * row_h + 16), cell, fontsize=11)
            x += col_w[c]
    page2.insert_text((50, top + len(rows) * row_h + 40), "The north region was the most profitable.", fontsize=11)

    data = doc.tobytes()
    doc.close()
    return data


class FakeModelService:
    """Stands in for ModelService. Records the document and question it was given."""

    def __init__(self, ready: bool = True, answer: str = "fake answer", unanswerable: bool = False) -> None:
        self.ready = ready
        self.answer = answer
        self.unanswerable = unanswerable
        self.truncated = False
        self.calls: list[tuple[str, str]] = []

    @property
    def is_ready(self) -> bool:
        return self.ready

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def status(self) -> dict[str, Any]:
        return {
            "model_ready": self.ready,
            "model_name": "fake-model",
            "adapter_loaded": False,
            "device": "cpu",
            "load_error": None if self.ready else "Model has not been loaded.",
        }

    def generate_answer(self, document_markdown: str, question: str) -> GenerationResult:
        if not self.ready:
            raise ModelNotReadyError("not ready")
        self.calls.append((document_markdown, question))
        return GenerationResult(
            answer=self.answer,
            unanswerable=self.unanswerable,
            model_name="fake-model",
            prompt_tokens=10,
            generated_tokens=2,
            document_truncated=self.truncated,
            document_chars_total=len(document_markdown),
            document_chars_used=len(document_markdown),
            device="cpu",
            dtype="float32",
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, max_upload_mb=1)


@pytest.fixture
def fake_model() -> FakeModelService:
    return FakeModelService()


@pytest.fixture
def client(settings: Settings, fake_model: FakeModelService) -> TestClient:
    return TestClient(create_app(settings=settings, model_service=fake_model))
