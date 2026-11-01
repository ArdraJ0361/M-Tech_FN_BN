"""Tests for Markdown rendering, answer location and the training-format prompt."""

from app.prompting import build_prompt, first_line, is_unanswerable, prompt_parts
from app.structure import (
    DocumentElement,
    elements_to_markdown,
    locate_answer,
    normalize,
    table_to_markdown,
)

TABLE = [["Region", "Revenue", "Profit"], ["North", "120", "30"], ["South", "90", "15"]]


def elements():
    return [
        DocumentElement("title", "Annual Report", 1),
        DocumentElement("heading_1", "1. Introduction", 1),
        DocumentElement("paragraph", "The company grew in the northern region.", 1),
        DocumentElement("heading_1", "2. Results", 2),
        DocumentElement("heading_2", "2.1 Regional table", 2),
        DocumentElement("table", TABLE, 2),
        DocumentElement("list_item", "• Hire 21 engineers", 2),
    ]


# --- Markdown -----------------------------------------------------------------------
def test_table_markdown_has_header_separator_and_rows():
    assert table_to_markdown(TABLE).splitlines() == [
        "| Region | Revenue | Profit |",
        "| --- | --- | --- |",
        "| North | 120 | 30 |",
        "| South | 90 | 15 |",
    ]


def test_table_markdown_escapes_pipes_and_pads_short_rows():
    out = table_to_markdown([["A", "B"], ["x|y"]]).splitlines()
    assert out[2] == "| x\\|y |  |"


def test_markdown_heading_levels_and_lists():
    md = elements_to_markdown(elements())
    assert "# Annual Report\n\n## 1. Introduction" in md
    assert "### 2.1 Regional table" in md
    assert md.endswith("- Hire 21 engineers")


def test_consecutive_list_items_stay_adjacent():
    md = elements_to_markdown(
        [DocumentElement("list_item", "- a", 1), DocumentElement("list_item", "- b", 1)]
    )
    assert md == "- a\n- b"


def test_numbered_list_items_keep_their_numbers():
    md = elements_to_markdown([DocumentElement("list_item", "1. first", 1)])
    assert md == "1. first"


# --- locate_answer ----------------------------------------------------------------------
def test_locate_in_paragraph_returns_page_and_section():
    [loc] = locate_answer(elements(), "northern region")
    assert loc.page_number == 1
    assert loc.element_type == "paragraph"
    assert loc.section_path == ("Annual Report", "1. Introduction")
    assert loc.table_cell is None


def test_locate_in_table_returns_row_and_column_labels():
    [loc] = locate_answer(elements(), "120")
    assert loc.page_number == 2
    assert loc.element_type == "table"
    assert loc.section_path == ("Annual Report", "2. Results", "2.1 Regional table")
    assert loc.table_cell.value == "120"
    assert loc.table_cell.row_label == "North"
    assert loc.table_cell.column_label == "Revenue"


def test_locate_matches_whole_words_only():
    # "12" is a substring of "120" but not a whole word anywhere
    assert locate_answer(elements(), "12") == []
    # "21" is a whole word in "Hire 21 engineers" (and only there)
    assert len(locate_answer(elements(), "21")) == 1
    # a partial word does not match ("nor" is inside "northern"/"North")
    assert locate_answer(elements(), "nor") == []


def test_locate_ignores_case_and_punctuation():
    assert len(locate_answer(elements(), "The Company grew!")) == 1


def test_locate_returns_nothing_when_answer_absent():
    assert locate_answer(elements(), "Zubrowka") == []
    assert locate_answer(elements(), "") == []
    assert locate_answer(elements(), "...") == []


def test_locate_respects_max_results():
    els = [DocumentElement("paragraph", "alpha beta", n) for n in range(1, 10)]
    assert len(locate_answer(els, "alpha", max_results=3)) == 3


def test_normalize():
    assert normalize("  Hello,   World! ") == "hello world"


# --- prompt format (the owner's instruction template) ---------------------------------------
EXPECTED_PROMPT = (
    "You are a senior business and finance analyst at a reputed organization. Your task is to "
    "analyze the provided document content and answer the user's question accurately and "
    "concisely.\n\n"
    "**Document Content:**\n"
    "# Title\n\nSome text.\n\n"
    "**Question:**\n"
    "What is it?\n\n"
    "**Instructions:**\n"
    "- Answer the question using only the information provided in the document.\n"
    "- Provide a precise, crisp, and professional answer that directly addresses the question.\n"
    "- Avoid unnecessary explanations, background information, or repetition.\n"
    "- Do not make assumptions or introduce information that is not supported by the document.\n"
    '- If the answer is not available in the provided content, respond with: '
    '"The requested information is not available in the provided document."\n\n'
    "**Answer:**\n"
)


def test_prompt_matches_the_template():
    assert build_prompt("  # Title\n\nSome text.  ", "  What is it?  ") == EXPECTED_PROMPT


def test_prompt_parts_recombine_to_the_full_prompt():
    prefix, suffix = prompt_parts("What?")
    assert prefix + "DOC" + suffix == build_prompt("DOC", "What?")


def test_first_line_and_unanswerable():
    assert first_line("  120\nextra text") == "120"
    assert first_line("") == ""
    assert is_unanswerable("unanswerable")
    assert is_unanswerable(" Unanswerable. ")
    assert is_unanswerable("The requested information is not available in the provided document.")
    assert not is_unanswerable("The answer is unanswerable")
