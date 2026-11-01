"""PDF validation and structure-preserving text extraction (text-based PDFs only).

``process_pdf`` validates an upload and returns a ``StructuredDocument``: an ordered list
of headings, paragraphs, list items and tables, each tagged with its page number. The
structure is recovered with simple, explainable heuristics on top of PyMuPDF:

* **Body font size** = the most common font size (weighted by characters).
* **Headings** = short lines that are noticeably larger than the body text
  (>= 1.5x title, >= 1.25x heading 1, >= 1.10x heading 2) or fully bold
  (heading 3). Section numbers such as ``1.``, ``1.1``, ``1.1.1`` give the level.
* **Lists** = lines that start with a bullet glyph or a number such as ``1.`` / ``(a)``.
* **Tables** = ruled tables found by ``page.find_tables()``; text inside a table's
  bounding box is not repeated as paragraphs.
* **Paragraphs** = consecutive lines with small vertical gaps.
* Repeated page headers/footers and bare page numbers are dropped.

Limitations: scanned/image-only PDFs are NOT supported (no OCR); tables without ruling
lines are read as ordinary text; multi-column layouts may be read in the wrong order.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

import pymupdf

from app.structure import (
    BULLET_MARKER,
    NUMBERED_MARKER,
    DocumentElement,
    StructuredDocument,
)

logger = logging.getLogger(__name__)


# --- Errors -----------------------------------------------------------------
class DocumentProcessingError(Exception):
    """Base class for PDF processing problems. ``code`` is machine readable."""

    code = "document_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(DocumentProcessingError):
    code = "unsupported_file_type"


class EmptyFileError(DocumentProcessingError):
    code = "empty_file"


class FileTooLargeError(DocumentProcessingError):
    code = "file_too_large"


class TooManyPagesError(DocumentProcessingError):
    code = "too_many_pages"


class MalformedPDFError(DocumentProcessingError):
    code = "malformed_pdf"


class NoExtractableTextError(DocumentProcessingError):
    code = "no_extractable_text"


# --- Validation helpers -------------------------------------------------------
_PDF_MAGIC = b"%PDF-"
_ALNUM = re.compile(r"[^\W_]", re.UNICODE)  # letters and digits


def sanitize_filename(filename: str | None) -> str:
    """Return a display-safe filename. It is only a label and is never used as a path."""
    name = os.path.basename((filename or "").replace("\\", "/")).strip()
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    return name[:255] or "document.pdf"


def validate_pdf_upload(filename: str | None, data: bytes, max_bytes: int) -> None:
    """Validate extension, size and PDF signature. Raises ``DocumentProcessingError``.

    The client-supplied Content-Type header is deliberately ignored because it is
    trivially spoofed; the file signature is checked instead.
    """
    if not filename or not filename.lower().endswith(".pdf"):
        raise UnsupportedFileTypeError("Only files with a .pdf extension are supported.")
    if len(data) == 0:
        raise EmptyFileError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"The uploaded file exceeds the maximum size of {max_bytes // (1024 * 1024)} MB."
        )
    if _PDF_MAGIC not in data[:1024]:
        raise UnsupportedFileTypeError("The uploaded file is not a valid PDF document.")


# --- Heuristic thresholds (relative to the body font size) ----------------------
TITLE_RATIO = 1.5
H1_RATIO = 1.25
H2_RATIO = 1.10
BOLD_HEADING_MIN_RATIO = 0.95
MAX_HEADING_CHARS = 120
PARAGRAPH_GAP_FACTOR = 0.5  # vertical gap (x font size) that starts a new paragraph
MARGIN_FRACTION = 0.08      # top/bottom page fraction searched for headers/footers

_SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)[\.\)]?\s+\S")
_PAGE_NUMBER_ONLY = re.compile(r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE)


@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    x0: float
    y0: float
    y1: float


# --- Low-level page reading -----------------------------------------------------
def _span_is_bold(span: dict[str, Any]) -> bool:
    font = str(span.get("font", "")).lower()
    return bool(span.get("flags", 0) & 16) or any(w in font for w in ("bold", "black", "heavy"))


def _inside(bbox: tuple[float, float, float, float], x: float, y: float, pad: float = 2.0) -> bool:
    return bbox[0] - pad <= x <= bbox[2] + pad and bbox[1] - pad <= y <= bbox[3] + pad


def _clean_cell(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _extract_tables(page: Any) -> list[tuple[tuple[float, float, float, float], list[list[str]]]]:
    """Return ``[(bbox, rows)]`` for ruled tables on the page (empty on any failure)."""
    found: list[tuple[tuple[float, float, float, float], list[list[str]]]] = []
    try:
        finder = page.find_tables()
        for table in finder.tables:
            rows = [[_clean_cell(c) for c in row] for row in table.extract()]
            rows = [row for row in rows if any(row)]  # drop empty rows
            if not rows:
                continue
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            keep = [i for i in range(width) if any(r[i] for r in rows)]  # drop empty columns
            rows = [[r[i] for i in keep] for r in rows]
            if len(rows) >= 2 and rows and len(rows[0]) >= 2:  # ignore 1-row/1-column "tables"
                found.append((tuple(table.bbox), rows))
    except Exception as exc:  # table detection is best-effort
        logger.warning("Table detection failed on a page: %s", type(exc).__name__)
    return found


def _read_lines(page: Any, table_boxes: list[tuple[float, float, float, float]]) -> list[_Line]:
    """Text lines in reading order, excluding lines that lie inside detected tables."""
    lines: list[_Line] = []
    page_dict = page.get_text("dict", sort=True)
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block (1 = image)
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if str(s.get("text", "")).strip()]
            if not spans:
                continue
            x0, y0, x1, y1 = line["bbox"]
            if any(_inside(box, (x0 + x1) / 2, (y0 + y1) / 2) for box in table_boxes):
                continue
            text = " ".join("".join(str(s["text"]) for s in line["spans"]).split())
            text = text.replace("\x00", "")
            if not text:
                continue
            dominant = max(spans, key=lambda s: len(str(s["text"]).strip()))
            lines.append(
                _Line(
                    text=text,
                    size=round(float(dominant.get("size", 0.0)) * 2) / 2,  # nearest 0.5pt
                    bold=all(_span_is_bold(s) for s in spans),
                    x0=float(x0),
                    y0=float(y0),
                    y1=float(y1),
                )
            )
    return lines


def _drop_headers_footers(
    pages_lines: list[tuple[list[_Line], float]],
) -> list[list[_Line]]:
    """Remove bare page numbers and (for >=3 pages) text repeated in the page margins."""
    page_count = len(pages_lines)
    seen: Counter[str] = Counter()

    def in_margin(line: _Line, height: float) -> bool:
        return line.y1 <= height * MARGIN_FRACTION or line.y0 >= height * (1 - MARGIN_FRACTION)

    def signature(text: str) -> str:
        return re.sub(r"\d+", "#", text.lower())

    for lines, height in pages_lines:
        for sig in {signature(l.text) for l in lines if in_margin(l, height)}:
            seen[sig] += 1

    repeated_min = max(2, (page_count + 1) // 2)
    cleaned: list[list[_Line]] = []
    for lines, height in pages_lines:
        kept = []
        for line in lines:
            if in_margin(line, height):
                if _PAGE_NUMBER_ONLY.match(line.text):
                    continue
                if page_count >= 3 and seen[signature(line.text)] >= repeated_min:
                    continue
            kept.append(line)
        cleaned.append(kept)
    return cleaned


# --- Classification and assembly --------------------------------------------------
def _body_font_size(all_lines: list[_Line]) -> float:
    weights: Counter[float] = Counter()
    for line in all_lines:
        weights[line.size] += len(line.text)
    return weights.most_common(1)[0][0] if weights else 0.0


def _heading_type(line: _Line, body: float, first_title_allowed: bool) -> Optional[str]:
    """Return heading_1/2/3 or title for a heading-like line, otherwise ``None``."""
    text = line.text
    if len(text) > MAX_HEADING_CHARS:
        return None
    number = _SECTION_NUMBER.match(text)
    if text.endswith((",", ";")) or (text.endswith(".") and not number):
        return None
    ratio = line.size / body if body else 1.0

    size_based: Optional[str] = None
    if ratio >= TITLE_RATIO:
        size_based = "title" if first_title_allowed else "heading_1"
    elif ratio >= H1_RATIO:
        size_based = "heading_1"
    elif ratio >= H2_RATIO:
        size_based = "heading_2"

    bold_based = line.bold and ratio >= BOLD_HEADING_MIN_RATIO and len(text) <= 100
    if size_based is None and not bold_based:
        return None
    if number and (size_based or bold_based) and size_based != "title":
        depth = number.group(1).count(".") + 1
        return f"heading_{min(depth, 3)}"
    return size_based or "heading_3"


def _is_list_start(text: str) -> bool:
    return bool(BULLET_MARKER.match(text) or NUMBERED_MARKER.match(text))


def _join(previous: str, nxt: str) -> str:
    """Join two lines of the same paragraph, repairing end-of-line hyphenation."""
    if previous.endswith("-") and len(previous) > 1 and previous[-2].isalpha() and nxt[:1].islower():
        return previous[:-1] + nxt
    return f"{previous} {nxt}"


def _build_text_elements(
    lines: list[_Line], page_number: int, body: float, state: dict[str, bool]
) -> list[tuple[float, DocumentElement]]:
    """Group classified lines into elements. Returns ``(y0, element)`` in reading order."""
    out: list[tuple[float, DocumentElement]] = []
    current_type: Optional[str] = None
    current_text = ""
    current_y0 = 0.0
    previous: Optional[_Line] = None

    def flush() -> None:
        nonlocal current_type, current_text
        if current_type and current_text.strip():
            out.append(
                (current_y0, DocumentElement(current_type, current_text.strip(), page_number))
            )
        current_type, current_text = None, ""

    for line in lines:
        heading = _heading_type(line, body, first_title_allowed=not state["title_seen"])
        if heading == "title":
            state["title_seen"] = True
        if heading:
            kind = heading
        elif _is_list_start(line.text):
            kind = "list_item"
        else:
            kind = "paragraph"

        gap = (line.y0 - previous.y1) if previous else 0.0
        tight = previous is not None and gap < PARAGRAPH_GAP_FACTOR * max(line.size, 1.0)
        starts_numbered_heading = bool(heading and _SECTION_NUMBER.match(line.text))

        continues = False
        if current_type is not None and tight:
            if kind == "paragraph" and current_type in ("paragraph", "list_item"):
                continues = True  # wrapped paragraph line / list-item continuation
            elif kind == current_type and heading and not starts_numbered_heading:
                continues = line.size == (previous.size if previous else line.size)

        if continues:
            current_text = _join(current_text, line.text)
        else:
            flush()
            current_type, current_text, current_y0 = kind, line.text, line.y0
        previous = line
    flush()
    return out


def _merge_tables(
    text_items: list[tuple[float, DocumentElement]],
    tables: list[tuple[tuple[float, float, float, float], list[list[str]]]],
    page_number: int,
) -> list[DocumentElement]:
    """Place each table before the first text element that lies below its top edge."""
    pending = sorted(tables, key=lambda t: t[0][1])
    merged: list[DocumentElement] = []
    for y0, element in text_items:
        while pending and pending[0][0][1] <= y0:
            merged.append(DocumentElement("table", pending.pop(0)[1], page_number))
        merged.append(element)
    for _, rows in pending:
        merged.append(DocumentElement("table", rows, page_number))
    return merged


# --- Public entry point ------------------------------------------------------------
def process_pdf(
    filename: str | None,
    data: bytes,
    *,
    max_bytes: int,
    min_text_chars: int = 20,
    max_pages: int = 100,
) -> StructuredDocument:
    """Validate a PDF and extract its structure.

    Raises:
        UnsupportedFileTypeError, EmptyFileError, FileTooLargeError: invalid upload.
        TooManyPagesError: more than ``max_pages`` pages.
        MalformedPDFError: corrupt, encrypted or otherwise unreadable PDF.
        NoExtractableTextError: no meaningful text (e.g. scanned PDF without OCR).
    """
    validate_pdf_upload(filename, data, max_bytes)

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types
        logger.warning("PDF could not be opened: %s", type(exc).__name__)
        raise MalformedPDFError("The PDF is corrupt or could not be read.") from exc

    try:
        if doc.needs_pass:
            raise MalformedPDFError("Password-protected PDFs are not supported.")
        page_count = doc.page_count
        if page_count == 0:
            raise MalformedPDFError("The PDF contains no pages.")
        if page_count > max_pages:
            raise TooManyPagesError(
                f"The PDF has {page_count} pages; the maximum supported is {max_pages}."
            )

        page_tables: list[list[tuple[tuple[float, float, float, float], list[list[str]]]]] = []
        page_lines: list[tuple[list[_Line], float]] = []
        for index in range(page_count):
            try:
                page = doc.load_page(index)
                tables = _extract_tables(page)
                lines = _read_lines(page, [bbox for bbox, _ in tables])
                height = float(page.rect.height)
            except DocumentProcessingError:
                raise
            except Exception as exc:
                logger.warning(
                    "Extraction failed on page %d: %s", index + 1, type(exc).__name__
                )
                raise MalformedPDFError(f"Text extraction failed on page {index + 1}.") from exc
            page_tables.append(tables)
            page_lines.append((lines, height))
    finally:
        doc.close()

    cleaned = _drop_headers_footers(page_lines)
    body = _body_font_size([line for lines in cleaned for line in lines])

    elements: list[DocumentElement] = []
    empty_pages: list[int] = []
    state = {"title_seen": False}
    for index, lines in enumerate(cleaned):
        page_number = index + 1
        items = _build_text_elements(lines, page_number, body, state)
        page_elements = _merge_tables(items, page_tables[index], page_number)
        if not page_elements:
            empty_pages.append(page_number)
        elements.extend(page_elements)

    def element_text(e: DocumentElement) -> str:
        return " ".join(c for row in e.content for c in row) if e.type == "table" else str(e.content)

    meaningful = sum(len(_ALNUM.findall(element_text(e))) for e in elements)
    if meaningful < min_text_chars:
        raise NoExtractableTextError(
            "No meaningful text could be extracted. The PDF may be scanned or "
            "image-only; OCR is not supported."
        )

    return StructuredDocument(
        elements=elements, page_count=page_count, empty_page_numbers=empty_pages
    )
