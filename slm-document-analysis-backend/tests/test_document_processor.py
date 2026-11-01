"""Tests for PDF validation and structure extraction (no model needed)."""

import pymupdf
import pytest

from app.document_processor import (
    EmptyFileError,
    FileTooLargeError,
    MalformedPDFError,
    NoExtractableTextError,
    TooManyPagesError,
    UnsupportedFileTypeError,
    process_pdf,
    sanitize_filename,
    validate_pdf_upload,
)
from tests.conftest import make_pdf, make_structured_pdf

MAX = 5 * 1024 * 1024


def _types(doc):
    return [e.type for e in doc.elements]


# --- structure ----------------------------------------------------------------------
def test_structured_pdf_headings_lists_and_table_are_recovered():
    doc = process_pdf("report.pdf", make_structured_pdf(), max_bytes=MAX)
    assert doc.page_count == 2
    assert _types(doc) == [
        "title", "heading_1", "paragraph", "heading_2", "list_item", "list_item",
        "heading_1", "table", "paragraph",
    ]
    assert doc.heading_count == 4
    assert doc.table_count == 1
    assert [e.page_number for e in doc.elements] == [1, 1, 1, 1, 1, 1, 2, 2, 2]


def test_paragraph_lines_are_joined_into_one_paragraph():
    doc = process_pdf("report.pdf", make_structured_pdf(), max_bytes=MAX)
    paragraph = next(e for e in doc.elements if e.type == "paragraph")
    assert paragraph.content.startswith("This report describes")
    assert "Revenue increased" in paragraph.content
    assert "\n" not in paragraph.content


def test_markdown_has_heading_markers_and_markdown_table():
    md = process_pdf("report.pdf", make_structured_pdf(), max_bytes=MAX).markdown
    assert md.startswith("# Annual Report 2024")
    assert "\n## 1. Introduction\n" in md
    assert "\n### 1.1 Scope\n" in md
    assert "- Financial results\n- Customer satisfaction" in md
    assert "| Region | Revenue | Profit |\n| --- | --- | --- |\n| North | 120 | 30 |" in md


def test_table_text_is_not_repeated_as_paragraph():
    doc = process_pdf("report.pdf", make_structured_pdf(), max_bytes=MAX)
    paragraphs = " ".join(str(e.content) for e in doc.elements if e.type == "paragraph")
    assert "Revenue" not in paragraphs.replace("Revenue increased", "")
    assert "120" not in paragraphs


def test_plain_pdf_without_styling_yields_paragraphs_only():
    doc = process_pdf("plain.pdf", make_pdf(["Just a plain sentence about nothing special."]), max_bytes=MAX)
    assert _types(doc) == ["paragraph"]
    assert doc.heading_count == 0 and doc.table_count == 0


def test_page_numbers_follow_the_pdf():
    doc = process_pdf(
        "two.pdf",
        make_pdf(["First page has enough words here.", "Second page has other words here."]),
        max_bytes=MAX,
    )
    assert [e.page_number for e in doc.elements] == [1, 2]


def test_repeated_page_numbers_in_footer_are_dropped():
    pdf = pymupdf.open()
    for n in range(1, 4):
        page = pdf.new_page()
        page.insert_text((50, 100), f"Body text of page number {n} with enough words.", fontsize=11)
        page.insert_text((290, 820), str(n), fontsize=9)
    data = pdf.tobytes()
    doc = process_pdf("n.pdf", data, max_bytes=MAX)
    assert all(not str(e.content).strip().isdigit() for e in doc.elements)


def test_blank_page_is_reported_but_document_is_accepted():
    pdf = make_pdf(["Content on the first page, long enough.", ""])
    doc = process_pdf("blank.pdf", pdf, max_bytes=MAX)
    assert doc.empty_page_numbers == [2]


# --- rejections ------------------------------------------------------------------------
def test_wrong_extension_is_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        process_pdf("notes.txt", b"%PDF-1.4 hello", max_bytes=MAX)


def test_fake_pdf_without_signature_is_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        process_pdf("fake.pdf", b"this is not a pdf at all", max_bytes=MAX)


def test_empty_file_is_rejected():
    with pytest.raises(EmptyFileError):
        process_pdf("empty.pdf", b"", max_bytes=MAX)


def test_oversized_file_is_rejected():
    with pytest.raises(FileTooLargeError):
        process_pdf("big.pdf", make_pdf(["Some text that is long enough."]), max_bytes=100)


def test_truncated_pdf_is_malformed_or_unreadable():
    data = make_pdf(["Some text that is long enough to matter."])[:60]
    with pytest.raises((MalformedPDFError, NoExtractableTextError)):
        process_pdf("broken.pdf", data, max_bytes=MAX)


def test_pdf_without_text_is_rejected():
    with pytest.raises(NoExtractableTextError):
        process_pdf("blank.pdf", make_pdf(["", ""]), max_bytes=MAX)


def test_too_many_pages_is_rejected():
    with pytest.raises(TooManyPagesError):
        process_pdf("long.pdf", make_pdf(["Some text on this page."] * 3), max_bytes=MAX, max_pages=2)


def test_password_protected_pdf_is_rejected():
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 100), "Secret text that is long enough.")
    data = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user"
    )
    with pytest.raises(MalformedPDFError):
        process_pdf("locked.pdf", data, max_bytes=MAX)


# --- helpers ---------------------------------------------------------------------------
def test_validate_accepts_valid_pdf():
    validate_pdf_upload("a.pdf", make_pdf(["hello world"]), MAX)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\me\\report.pdf", "report.pdf"),
        ("", "document.pdf"),
        (None, "document.pdf"),
        ("a\x00b.pdf", "ab.pdf"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected
