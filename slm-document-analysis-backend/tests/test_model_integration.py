"""Tests that need the REAL configured model. Skipped unless MODEL_NAME_OR_PATH is set.

Run with:  MODEL_NAME_OR_PATH=<model> [ADAPTER_PATH=<dir>] pytest -m requires_model

They check the mechanics (training-format prompt ids, first-line answer, truncation of the
document only). They do NOT measure answer quality.
"""

import os

import pytest

from app.config import Settings
from app.model_service import ModelService
from app.prompting import build_prompt

pytestmark = [
    pytest.mark.requires_model,
    pytest.mark.skipif(
        not os.environ.get("MODEL_NAME_OR_PATH"), reason="MODEL_NAME_OR_PATH is not set"
    ),
]

DOC = "# Report\n\n## 1. Results\n\n| Region | Revenue |\n| --- | --- |\n| North | 120 |\n"


@pytest.fixture(scope="module")
def service():
    svc = ModelService(Settings(_env_file=None, max_new_tokens=16))
    svc.load()
    return svc


def test_prompt_ids_equal_training_style_tokenization(service):
    ids, used = service._encode_within_budget(DOC, "What is the revenue of North?")
    expected = service._tokenizer(build_prompt(DOC, "What is the revenue of North?"))["input_ids"]
    assert ids[0].tolist() == expected
    assert used == len(DOC.strip())


def test_generates_a_single_line_answer(service):
    result = service.generate_answer(DOC, "What is the revenue of North?")
    assert result.answer and "\n" not in result.answer
    assert result.generated_tokens > 0
    assert result.document_truncated is False


def test_long_document_is_cut_at_the_end_but_question_is_kept(service):
    service._settings.max_input_tokens = 300
    try:
        long_doc = "# Report\n\n" + "Filler sentence about nothing. " * 400
        ids, used = service._encode_within_budget(long_doc, "What is X?")
        text = service._tokenizer.decode(ids[0])
        assert ids.shape[-1] <= 300
        assert used < len(long_doc.strip())
        assert "**Question:**\nWhat is X?\n\n**Instructions:**" in text
        assert text.endswith("**Answer:**\n")
    finally:
        service._settings.max_input_tokens = 8192
