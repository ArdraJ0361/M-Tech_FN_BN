"""Structured document model: headings, paragraphs, lists and tables -> Markdown.

The extractor (``document_processor.py``) produces a list of ``DocumentElement``. This module
turns that list into the Markdown text that is placed in the model prompt:

* title / headings -> ``#`` / ``##`` / ``###`` / ``####`` so the model sees the hierarchy
* tables           -> Markdown tables (``| a | b |`` rows with a ``| --- | --- |`` separator)
* list items       -> ``- item`` lines
* paragraphs       -> plain text separated by blank lines

It also contains ``locate_answer``: a plain string match of the model's answer against the
extracted elements, used to tell the user *where* in the document the answer text appears.
This is NOT produced by the model and is only reported when an actual match is found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# element.type -> Markdown heading marker
HEADING_MARKERS = {"title": "#", "heading_1": "##", "heading_2": "###", "heading_3": "####"}
# element.type -> nesting depth used to build the "section path"
HEADING_LEVELS = {"title": 0, "heading_1": 1, "heading_2": 2, "heading_3": 3}

# Bullet glyphs (including U+00B7 and the Symbol-font private-use bullets U+F0B7/U+F0A7) may be
# followed directly by the text; ASCII/dash markers need a following space.
BULLET_MARKER = re.compile(r"^(?:[•▪◦‣●○■□·\uf0b7\uf0a7]\s*|[\-\*–—]\s+)")
NUMBERED_MARKER = re.compile(r"^(?:\d+[\.\)]|\([a-zA-Z0-9]+\))\s+")
LIST_MARKER = re.compile(f"(?:{BULLET_MARKER.pattern})|(?:{NUMBERED_MARKER.pattern})")


@dataclass(frozen=True)
class DocumentElement:
    """One structural element in reading order.

    ``content`` is a ``str`` for text elements and a ``list[list[str]]`` (rows of cells,
    first row = header) for tables. ``page_number`` is 1-based.
    """

    type: str  # title | heading_1 | heading_2 | heading_3 | paragraph | list_item | table
    content: Any
    page_number: int


@dataclass
class StructuredDocument:
    elements: list[DocumentElement]
    page_count: int
    empty_page_numbers: list[int] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        return elements_to_markdown(self.elements)

    @property
    def heading_count(self) -> int:
        return sum(1 for e in self.elements if e.type in HEADING_MARKERS)

    @property
    def table_count(self) -> int:
        return sum(1 for e in self.elements if e.type == "table")


# --------------------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------------------
def _escape_cell(cell: str) -> str:
    return cell.replace("|", "\\|")


def table_to_markdown(grid: list[list[str]]) -> str:
    """Render rows of cells as a Markdown table (first row is the header)."""
    if not grid:
        return ""
    columns = max(len(row) for row in grid)
    rows = [[_escape_cell(c) for c in row] + [""] * (columns - len(row)) for row in grid]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * columns) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def element_to_markdown(element: DocumentElement) -> str:
    if element.type in HEADING_MARKERS:
        return f"{HEADING_MARKERS[element.type]} {element.content}"
    if element.type == "list_item":
        text = str(element.content).strip()
        if NUMBERED_MARKER.match(text):
            return text  # keep "1. item" / "(a) item" as written
        return "- " + BULLET_MARKER.sub("", text, count=1)
    if element.type == "table":
        return table_to_markdown(element.content)
    return str(element.content)


def elements_to_markdown(elements: list[DocumentElement]) -> str:
    """Join elements with blank lines; consecutive list items stay on adjacent lines."""
    parts: list[str] = []
    previous: Optional[str] = None
    for element in elements:
        rendered = element_to_markdown(element)
        if not rendered:
            continue
        if parts and previous == "list_item" and element.type == "list_item":
            parts[-1] += "\n" + rendered
        else:
            parts.append(rendered)
        previous = element.type
    return "\n\n".join(parts).strip()


# --------------------------------------------------------------------------------------
# Answer location (string matching, not model output)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TableCellRef:
    value: str
    row_label: Optional[str]     # first-column value of the row (None for header/first column)
    column_label: Optional[str]  # header text of the column (None when the match is in the header)


@dataclass(frozen=True)
class AnswerMatch:
    page_number: int
    element_type: str
    section_path: tuple[str, ...]  # headings above the match, outermost first
    snippet: str
    table_cell: Optional[TableCellRef] = None


_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    """Lower-case, replace punctuation by spaces, collapse whitespace."""
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Whole-word phrase match on normalized text (so '2' does not match '21')."""
    return f" {needle} " in f" {haystack} "


def _snippet(text: str, answer: str, width: int = 300) -> str:
    text = " ".join(text.split())
    index = text.lower().find(answer.lower().strip(" .")) if answer.strip(" .") else -1
    start = max(0, index - width // 3) if index > 0 else 0
    piece = text[start : start + width]
    return ("..." if start > 0 else "") + piece + ("..." if start + width < len(text) else "")


def locate_answer(
    elements: list[DocumentElement], answer: str, max_results: int = 5
) -> list[AnswerMatch]:
    """Find where the answer text occurs in the extracted document.

    Returns up to ``max_results`` locations in reading order, each with the page, the heading
    path (section) and, for tables, the row/column labels. Returns ``[]`` when the answer text
    does not occur verbatim (after normalisation) in any element.
    """
    needle = normalize(answer)
    if not needle:
        return []

    results: list[AnswerMatch] = []
    path: list[tuple[int, str]] = []  # (heading level, heading text)

    for element in elements:
        if element.type in HEADING_LEVELS:
            level = HEADING_LEVELS[element.type]
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, str(element.content)))

        if len(results) >= max_results:
            break
        section = tuple(text for _, text in path)

        if element.type == "table":
            grid: list[list[str]] = element.content
            for r, row in enumerate(grid):
                for c, cell in enumerate(row):
                    if len(results) < max_results and _contains_phrase(normalize(cell), needle):
                        ref = TableCellRef(
                            value=cell,
                            row_label=row[0] if r > 0 and c > 0 else None,
                            column_label=grid[0][c] if r > 0 and c < len(grid[0]) else None,
                        )
                        results.append(
                            AnswerMatch(
                                page_number=element.page_number,
                                element_type="table",
                                section_path=section,
                                snippet=_snippet(" | ".join(row), answer),
                                table_cell=ref,
                            )
                        )
        elif _contains_phrase(normalize(str(element.content)), needle):
            results.append(
                AnswerMatch(
                    page_number=element.page_number,
                    element_type=element.type,
                    section_path=section,
                    snippet=_snippet(str(element.content), answer),
                )
            )
    return results
