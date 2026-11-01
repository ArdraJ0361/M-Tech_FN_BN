"""
notebooks/build_extractor_notebook.py
-------------------------------------
Generates notebooks/document_structure_extractor.ipynb
A self-contained, Google Colab-ready notebook for structured document text extraction.
"""

import json
import pathlib

CELLS = []


def md(text: str):
    CELLS.append(("markdown", text.strip()))


def code(text: str):
    CELLS.append(("code", text.strip()))


# ---------------------------------------------------------------------------
# Notebook Definition
# ---------------------------------------------------------------------------

md("""
# 📄 Hierarchical Document Structure & Text Extraction Pipeline

**Extract clean, structured text from documents while preserving reading order and logical hierarchy.**

### What this pipeline does:
- 🏷️ **Identifies Hierarchy**: Title, Headings ($H1, H2, H3$), Sections, and Subsections in logical sequence.
- 📖 **Preserves Reading Order**: No jumbled or fragmented words — text flows top-to-bottom naturally.
- 📊 **Extracts Tables**: Automatically detects tables and formats them as clean **Markdown tables** or **Pandas DataFrames**.
- 📝 **Preserves Paragraphs & Lists**: Retains bullet points, numbered lists, and paragraph boundaries.
- 📦 **Multi-Format Support**: Works seamlessly on **PDF**, **DOCX**, and **TXT / Markdown** documents.
- 💾 **Export Ready**: Exports directly to **Markdown (`.md`)**, **Structured JSON (`.json`)**, and **Plain Text (`.txt`)**.

---
### 🚀 How to Run in Google Colab:
1. Click **Runtime** → **Run all** (or run cells one by one).
2. Use **Cell 3** to test with auto-generated sample documents, or **Cell 4** to upload your own document.
3. View the rendered document structure in **Cell 5** and download the outputs in **Cell 6**.
""")

code("""# Cell 1: Install required dependencies in Google Colab / Local Environment
import sys
import subprocess

packages = ["pymupdf>=1.24.0", "python-docx>=1.1.0", "pandas>=2.0.0", "tabulate>=0.9.0"]

if "google.colab" in sys.modules:
    print("📦 Installing dependencies in Google Colab...")
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("✅ All dependencies installed successfully in Colab!")
else:
    print("💻 Local environment detected. Verifying dependencies...")
    try:
        import pymupdf, docx, pandas
        print("✅ Core dependencies already installed!")
    except ImportError:
        print("Installing missing dependencies...")
        for pkg in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
            except Exception as e:
                print(f"Note: {e}")
""")

code("""# Cell 2: Core Hierarchical Document Extractor Engine (Self-contained)
import os
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional, Union
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("DocExtractor")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DocumentElement:
    '''Represents a single structural element within a document.'''
    type: str  # 'title', 'heading_1', 'heading_2', 'heading_3', 'paragraph', 'list_item', 'table'
    content: Any  # str for text, List[List[str]] for tables
    level: int = 0  # 0 for root/body, 1 for H1, 2 for H2, 3 for H3
    page_num: int = 1
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        '''Convert element to clean GitHub-flavored Markdown.'''
        if self.type == "title":
            return f"# {self.content}\\n"
        elif self.type == "heading_1":
            return f"## {self.content}\\n"
        elif self.type == "heading_2":
            return f"### {self.content}\\n"
        elif self.type == "heading_3":
            return f"#### {self.content}\\n"
        elif self.type == "list_item":
            item = str(self.content).strip()
            if not re.match(r"^([*•\\-]|\\d+\\.)\\s+", item):
                return f"- {item}\\n"
            return f"{item}\\n"
        elif self.type == "table":
            return format_table_as_markdown(self.content) + "\\n"
        elif self.type == "code_block":
            return f"```\\n{self.content}\\n```\\n"
        else:
            return f"{self.content}\\n"


@dataclass
class StructuredDocument:
    '''Represents a fully parsed document with preserved hierarchy and reading order.'''
    filename: str
    file_type: str
    elements: List[DocumentElement] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def get_title(self) -> Optional[str]:
        for el in self.elements:
            if el.type in ("title", "heading_1"):
                return str(el.content)
        return None

    def to_markdown(self) -> str:
        '''Renders the entire document in ordered Markdown.'''
        parts = []
        for el in self.elements:
            md_text = el.to_markdown()
            if md_text:
                parts.append(md_text)
        return "\\n".join(parts).strip() + "\\n"

    def to_text(self) -> str:
        '''Renders the document in formatted human-readable plain text.'''
        parts = []
        for el in self.elements:
            if el.type == "title":
                parts.append(f"{el.content}\\n{'=' * len(str(el.content))}\\n")
            elif el.type == "heading_1":
                parts.append(f"\\n{el.content}\\n{'-' * len(str(el.content))}\\n")
            elif el.type == "heading_2":
                parts.append(f"\\n[Section] {el.content}\\n")
            elif el.type == "heading_3":
                parts.append(f"\\n[Sub-section] {el.content}\\n")
            elif el.type == "table":
                parts.append("\\n" + format_table_as_markdown(el.content) + "\\n")
            elif el.type == "list_item":
                parts.append(f"  * {el.content}")
            else:
                parts.append(str(el.content))
        return "\\n".join(parts).strip() + "\\n"

    def to_json(self, indent: int = 2) -> str:
        '''Exports the document structure to a JSON string.'''
        data = {
            "filename": self.filename,
            "file_type": self.file_type,
            "metadata": self.metadata,
            "total_elements": len(self.elements),
            "elements": [el.to_dict() for el in self.elements],
        }
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def get_outline(self) -> str:
        '''Generates an ASCII outline tree of document headings and tables.'''
        lines = [f"📂 Document: {self.filename}"]
        for el in self.elements:
            indent = "  " * max(el.level, 1)
            if el.type == "title":
                lines.append(f" └── 🏷️ [Title] {el.content}")
            elif el.type == "heading_1":
                lines.append(f"  ├── 📌 [H1] {el.content}")
            elif el.type == "heading_2":
                lines.append(f"  │   ├── 🔹 [H2] {el.content}")
            elif el.type == "heading_3":
                lines.append(f"  │   │   ├── ▪️ [H3] {el.content}")
            elif el.type == "table":
                rows = len(el.content) if isinstance(el.content, list) else 0
                cols = len(el.content[0]) if rows > 0 and isinstance(el.content[0], list) else 0
                lines.append(f"{indent}├── 📊 [Table: {rows}x{cols}] (Page {el.page_num})")
        return "\\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting Helper
# ---------------------------------------------------------------------------

def format_table_as_markdown(table_data: Any) -> str:
    '''Converts a 2D list of cells into a clean Markdown table.'''
    if not table_data or not isinstance(table_data, list):
        return ""
    clean_rows = []
    for row in table_data:
        if isinstance(row, list):
            clean_row = [str(cell).strip().replace("\\n", " ") if cell is not None else "" for cell in row]
            if any(clean_row):
                clean_rows.append(clean_row)
        else:
            clean_rows.append([str(row).strip()])

    if not clean_rows:
        return ""

    num_cols = max(len(r) for r in clean_rows)
    normalized_rows = [r + [""] * (num_cols - len(r)) for r in clean_rows]
    col_widths = [3] * num_cols
    for row in normalized_rows:
        for c_idx, val in enumerate(row):
            col_widths[c_idx] = max(col_widths[c_idx], len(val))

    header = normalized_rows[0]
    header_line = "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(header)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(num_cols)) + " |"

    body_lines = []
    for row in normalized_rows[1:]:
        body_lines.append("| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |")

    return "\\n".join([header_line, sep_line] + body_lines)


# ---------------------------------------------------------------------------
# Format Extractors: DOCX, PDF, Plain Text
# ---------------------------------------------------------------------------

class DocxExtractor:
    '''Extracts sequential paragraphs and tables from DOCX files.'''

    @classmethod
    def extract(cls, file_path: Path) -> StructuredDocument:
        import docx
        from docx.oxml.text.paragraph import CT_P
        from docx.oxml.table import CT_Tbl
        from docx.text.paragraph import Paragraph
        from docx.table import Table

        doc = docx.Document(str(file_path))
        elements: List[DocumentElement] = []
        title_found = False

        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                p = Paragraph(child, doc)
                text = p.text.strip()
                if not text:
                    continue

                style_name = p.style.name.lower() if p.style and p.style.name else ""

                if "title" in style_name and not title_found:
                    elements.append(DocumentElement(type="title", content=text, level=0, page_num=1))
                    title_found = True
                elif "heading 1" in style_name:
                    elements.append(DocumentElement(type="heading_1", content=text, level=1, page_num=1))
                elif "heading 2" in style_name:
                    elements.append(DocumentElement(type="heading_2", content=text, level=2, page_num=1))
                elif "heading 3" in style_name:
                    elements.append(DocumentElement(type="heading_3", content=text, level=3, page_num=1))
                elif "bullet" in style_name or "list" in style_name or text.startswith(("•", "-", "*")):
                    elements.append(DocumentElement(type="list_item", content=text, level=0, page_num=1))
                else:
                    is_all_bold = p.runs and all(r.bold for r in p.runs if r.text.strip())
                    if is_all_bold and len(text) < 80 and not text.endswith("."):
                        elements.append(DocumentElement(type="heading_2", content=text, level=2, page_num=1))
                    else:
                        elements.append(DocumentElement(type="paragraph", content=text, level=0, page_num=1))

            elif isinstance(child, CT_Tbl):
                t = Table(child, doc)
                grid = []
                for row in t.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    cleaned_row = []
                    for val in row_data:
                        if not cleaned_row or val != cleaned_row[-1] or val != "":
                            cleaned_row.append(val)
                    if any(cleaned_row):
                        grid.append(cleaned_row)
                if grid:
                    elements.append(DocumentElement(type="table", content=grid, level=0, page_num=1))

        return StructuredDocument(filename=file_path.name, file_type="docx", elements=elements)


class PDFExtractor:
    '''Extracts layout-ordered text, headings, and tables from PDFs.'''

    @classmethod
    def extract(cls, file_path: Path) -> StructuredDocument:
        import pymupdf

        doc = pymupdf.open(str(file_path))
        elements: List[DocumentElement] = []

        all_spans = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("text", "").strip():
                                all_spans.append(span)

        font_sizes = [round(s.get("size", 10), 1) for s in all_spans]
        if font_sizes:
            from collections import Counter
            body_font_size = Counter(font_sizes).most_common(1)[0][0]
        else:
            body_font_size = 10.0

        title_threshold = body_font_size * 1.5
        h1_threshold = body_font_size * 1.25
        h2_threshold = body_font_size * 1.10

        title_found = False

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            tables, table_bboxes = [], []
            try:
                found_tabs = page.find_tables()
                if found_tabs and len(found_tabs.tables) > 0:
                    for tab in found_tabs:
                        extracted = tab.extract()
                        if extracted:
                            tables.append((tab.bbox, extracted))
                            table_bboxes.append(pymupdf.Rect(tab.bbox))
            except Exception:
                pass

            page_dict = page.get_text("dict")
            raw_blocks = page_dict.get("blocks", [])

            items_to_sort = []
            for b in raw_blocks:
                if b.get("type") == 0:
                    bbox = pymupdf.Rect(b.get("bbox"))
                    inside = any(bbox.intersects(tb) and (bbox & tb).get_area() > 0.5 * bbox.get_area() for tb in table_bboxes)
                    if not inside:
                        items_to_sort.append(("text", bbox.y0, bbox.x0, b))

            for tb_bbox, grid in tables:
                items_to_sort.append(("table", tb_bbox[1], tb_bbox[0], grid))

            items_to_sort.sort(key=lambda x: (x[1], x[2]))

            for item in items_to_sort:
                if item[0] == "table":
                    elements.append(DocumentElement(type="table", content=item[3], level=0, page_num=page_num))
                elif item[0] == "text":
                    b = item[3]
                    block_lines = []
                    max_span_size = 0.0
                    is_bold = False

                    for line in b.get("lines", []):
                        lt = "".join(span.get("text", "") for span in line.get("spans", "")).strip()
                        if lt:
                            block_lines.append(lt)
                        for span in line.get("spans", []):
                            sz = span.get("size", body_font_size)
                            if sz > max_span_size:
                                max_span_size = sz
                            flags = span.get("flags", 0)
                            fname = span.get("font", "").lower()
                            if (flags & 2 != 0) or (flags & 16 != 0) or "bold" in fname or "black" in fname:
                                is_bold = True

                    block_text = " ".join(block_lines).strip()
                    if not block_text:
                        continue

                    is_sec = bool(re.match(r"^(\\d+\\.|\\d+\\.\\d+|\\d+\\.\\d+\\.\\d+|Section\\s+\\d+|Chapter\\s+\\d+)\\s+[A-Z]", block_text))
                    is_bullet = block_text.startswith(("•", "-", "*", "–", "—")) or bool(re.match(r"^(\\([a-zA-Z0-9]+\\)|\\d+\\))\\s+", block_text))

                    if not title_found and page_idx == 0 and (max_span_size >= title_threshold or (is_bold and max_span_size > h1_threshold)):
                        elements.append(DocumentElement(type="title", content=block_text, level=0, page_num=page_num))
                        title_found = True
                    elif max_span_size >= h1_threshold or (is_sec and max_span_size >= body_font_size and len(block_text) < 100):
                        elements.append(DocumentElement(type="heading_1", content=block_text, level=1, page_num=page_num))
                    elif max_span_size >= h2_threshold or (is_bold and len(block_text) < 100 and not block_text.endswith(".")):
                        elements.append(DocumentElement(type="heading_2", content=block_text, level=2, page_num=page_num))
                    elif is_sec and len(block_text) < 120:
                        elements.append(DocumentElement(type="heading_3", content=block_text, level=3, page_num=page_num))
                    elif is_bullet:
                        elements.append(DocumentElement(type="list_item", content=block_text, level=0, page_num=page_num))
                    else:
                        elements.append(DocumentElement(type="paragraph", content=block_text, level=0, page_num=page_num))

        return StructuredDocument(filename=file_path.name, file_type="pdf", elements=elements)


class TextExtractor:
    '''Extracts structured hierarchy from plain text / Markdown files.'''

    @classmethod
    def extract(cls, file_path: Path) -> StructuredDocument:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        elements: List[DocumentElement] = []
        table_buffer: List[List[str]] = []
        title_found = False

        def flush_table():
            nonlocal table_buffer
            if table_buffer:
                elements.append(DocumentElement(type="table", content=table_buffer, level=0, page_num=1))
                table_buffer = []

        i = 0
        while i < len(lines):
            line = lines[i]
            s_line = line.strip()

            if s_line.startswith("|") and s_line.endswith("|") and "|" in s_line[1:-1]:
                if not re.match(r"^\\|(\\s*:?-+:?\\s*\\|)+$", s_line):
                    cells = [c.strip() for c in s_line.split("|")[1:-1]]
                    table_buffer.append(cells)
                i += 1
                continue
            else:
                flush_table()

            if not s_line:
                i += 1
                continue

            if i + 1 < len(lines):
                next_l = lines[i + 1].strip()
                if next_l and re.match(r"^={3,}$", next_l):
                    elements.append(DocumentElement(type="title" if not title_found else "heading_1", content=s_line, level=1, page_num=1))
                    title_found = True
                    i += 2
                    continue
                elif next_l and re.match(r"^-{3,}$", next_l) and not s_line.startswith(("-", "*", "#")):
                    elements.append(DocumentElement(type="heading_2", content=s_line, level=2, page_num=1))
                    i += 2
                    continue

            if s_line.startswith("# "):
                elements.append(DocumentElement(type="title" if not title_found else "heading_1", content=s_line[2:].strip(), level=1, page_num=1))
                title_found = True
            elif s_line.startswith("## "):
                elements.append(DocumentElement(type="heading_1", content=s_line[3:].strip(), level=1, page_num=1))
            elif s_line.startswith("### "):
                elements.append(DocumentElement(type="heading_2", content=s_line[4:].strip(), level=2, page_num=1))
            elif s_line.startswith("#### "):
                elements.append(DocumentElement(type="heading_3", content=s_line[5:].strip(), level=3, page_num=1))
            elif re.match(r"^(\\d+\\.|\\d+\\.\\d+|\\d+\\.\\d+\\.\\d+|[I|V|X]+\\.|\\bSECTION\\b|\\bCHAPTER\\b)\\s+[A-Z0-9]", s_line) and len(s_line) < 100:
                lvl = 1 if re.match(r"^(\\d+\\.|[I|V|X]+\\.|\\bSECTION\\b)", s_line) else 2
                elements.append(DocumentElement(type=f"heading_{lvl}", content=s_line, level=lvl, page_num=1))
            elif re.match(r"^([*•\\-]|\\d+\\.|\\([a-z0-9]+\\))\\s+", s_line):
                elements.append(DocumentElement(type="list_item", content=s_line, level=0, page_num=1))
            else:
                para_lines = [s_line]
                while i + 1 < len(lines):
                    peek = lines[i + 1].strip()
                    if not peek or peek.startswith(("#", "*", "•", "-", "|")) or re.match(r"^(\\d+\\.|\\bSECTION\\b)", peek):
                        break
                    para_lines.append(peek)
                    i += 1
                elements.append(DocumentElement(type="paragraph", content=" ".join(para_lines), level=0, page_num=1))

            i += 1

        flush_table()
        return StructuredDocument(filename=file_path.name, file_type="txt", elements=elements)


def extract_document(file_path: Union[str, Path]) -> StructuredDocument:
    '''Unified pipeline entrance point for extracting structured document text.'''
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return PDFExtractor.extract(path)
    elif ext in (".docx", ".doc"):
        return DocxExtractor.extract(path)
    else:
        return TextExtractor.extract(path)

print("⚡ Extraction Engine Initialized Successfully!")
""")

md("""
## 🧪 Create Sample Documents (1-Click Test)
Run this cell to generate test documents (`.docx`, `.txt`) containing:
- Multi-level headings ($H1, H2, H3$)
- Sequential paragraphs
- Structured tables (with columns and values)
- Bullet points and lists
""")

code("""# Cell 3: Generate Test Documents for immediate demonstration
import docx

os.makedirs("sample_docs", exist_ok=True)

# 1. Create Sample DOCX
docx_path = "sample_docs/enterprise_ai_strategy.docx"
doc = docx.Document()
doc.add_heading("Enterprise AI Adoption & Governance Report", level=0)
doc.add_paragraph("This strategic document outlines the deployment roadmap, architecture considerations, and performance benchmarks for enterprise large language models.")

doc.add_heading("1. Executive Summary", level=1)
doc.add_paragraph("Modern natural language systems demand structured evaluation across accuracy, latency, and operational cost. Our evaluation pipeline extracts structured document hierarchies.")

doc.add_heading("1.1 Model Performance Benchmarks", level=2)
doc.add_paragraph("The table below summarizes performance across model sizes and tuning strategies:")

tbl = doc.add_table(rows=4, cols=4)
headers = ["Architecture", "Parameters", "Exact Match (%)", "Latency (ms)"]
for col_idx, h in enumerate(headers):
    tbl.cell(0, col_idx).text = h

data = [
    ["SLM Base", "0.5B", "44.2%", "18 ms"],
    ["SLM + LoRA", "0.5B", "68.5%", "19 ms"],
    ["LLM Base", "7.0B", "71.3%", "142 ms"],
]
for row_idx, row_vals in enumerate(data, start=1):
    for col_idx, val in enumerate(row_vals):
        tbl.cell(row_idx, col_idx).text = val

doc.add_heading("2. Key Implementation Principles", level=1)
doc.add_paragraph("• Preserve reading order from top to bottom across all pages.")
doc.add_paragraph("• Structure tables into clean 2D grids rather than raw text strings.")
doc.add_paragraph("• Extract headings hierarchically so downstream RAG systems retain context.")

doc.save(docx_path)
print(f"✅ Generated sample DOCX: {docx_path}")

# 2. Create Sample Plain Text / Markdown
txt_path = "sample_docs/project_architecture.txt"
txt_content = '''# System Architecture Design Document

Author: System Engineering Team
Version: 2.1 | Status: Production Ready

1. Architectural Overview
The document parsing subsystem ingests heterogeneous digital formats and standardizes them into structured semantic trees.

1.1 Data Ingestion Pipeline
The pipeline supports native PDFs, Word documents (.docx), and plain text.

| Component | Responsibility | Latency Budget |
| DocxExtractor | XML node traversal & table parsing | < 50ms |
| PDFExtractor | Bounding-box sorting & table extraction | < 200ms |
| TextExtractor | Regex pattern header recognition | < 10ms |

2. Recommendations
• Standardize document exports to Markdown for LLM ingestion.
• Use structured JSON for vector database indexing.
'''
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(txt_content)
print(f"✅ Generated sample TXT: {txt_path}")
""")

md("""
## 📤 Upload or Select a Document
You can either:
1. **Upload your own file** (PDF, DOCX, TXT, MD) using the file uploader.
2. **Or pick one of the sample files** generated above.
""")

code("""# Cell 4: Upload and Process Document
import sys

TARGET_FILE = "sample_docs/enterprise_ai_strategy.docx"

if "google.colab" in sys.modules:
    from google.colab import files
    print("📤 Upload your document (PDF, DOCX, TXT, or MD):")
    uploaded = files.upload()
    if uploaded:
        TARGET_FILE = list(uploaded.keys())[0]
        print(f"\\n🎯 Processing uploaded file: {TARGET_FILE}")
    else:
        print(f"\\nℹ️ No file uploaded. Processing default sample: {TARGET_FILE}")
else:
    print(f"🎯 Local execution: Processing '{TARGET_FILE}'")
    print("(You can change TARGET_FILE to any path on your machine)")

# Run the Extraction Pipeline
doc_result = extract_document(TARGET_FILE)
print(f"\\n✨ Successfully extracted '{doc_result.filename}' with {len(doc_result.elements)} structural elements!")
""")

md("""
## 👁️ Visual Results & Hierarchy Exploration
Below you can examine the extracted document across 4 complementary views:
1. **Outline Hierarchy Tree**
2. **Rendered Markdown (Visual Document View)**
3. **Extracted Tables (Pandas DataFrames)**
4. **Structured JSON (For downstream RAG / LLM Pipelines)**
""")

code("""# Cell 5: Display Extracted Document in Multiple Views
from IPython.display import display, Markdown

print("=" * 70)
print("1️⃣ DOCUMENT HIERARCHY OUTLINE")
print("=" * 70)
print(doc_result.get_outline())

print("\\n" + "=" * 70)
print("2️⃣ VISUALLY RENDERED MARKDOWN (How the document flows)")
print("=" * 70)
display(Markdown(doc_result.to_markdown()))

# 3. Explore Extracted Tables
tables = [el for el in doc_result.elements if el.type == "table"]
if tables:
    print("\\n" + "=" * 70)
    print(f"3️⃣ EXTRACTED TABLES ({len(tables)} found)")
    print("=" * 70)
    for idx, tbl in enumerate(tables, start=1):
        print(f"\\n📊 Table #{idx} (Page {tbl.page_num}):")
        if isinstance(tbl.content, list) and len(tbl.content) > 1:
            df = pd.DataFrame(tbl.content[1:], columns=tbl.content[0])
            display(df)
        else:
            print(tbl.content)

# 4. JSON Structure Sample
print("\\n" + "=" * 70)
print("4️⃣ STRUCTURED JSON SAMPLE (First 4 elements)")
print("=" * 70)
sample_json = {
    "filename": doc_result.filename,
    "file_type": doc_result.file_type,
    "total_elements": len(doc_result.elements),
    "elements_preview": [el.to_dict() for el in doc_result.elements[:4]]
}
print(json.dumps(sample_json, indent=2, ensure_ascii=False))
""")

md("""
## 💾 Export & Download
Save and download your extracted document in **Markdown (`.md`)**, **Structured JSON (`.json`)**, and **Plain Text (`.txt`)**.
""")

code("""# Cell 6: Export Results and Trigger Download
os.makedirs("extracted_outputs", exist_ok=True)
stem = Path(doc_result.filename).stem

md_file = f"extracted_outputs/{stem}_extracted.md"
json_file = f"extracted_outputs/{stem}_extracted.json"
txt_file = f"extracted_outputs/{stem}_extracted.txt"

# Write files
with open(md_file, "w", encoding="utf-8") as f:
    f.write(doc_result.to_markdown())

with open(json_file, "w", encoding="utf-8") as f:
    f.write(doc_result.to_json(indent=2))

with open(txt_file, "w", encoding="utf-8") as f:
    f.write(doc_result.to_text())

print("✅ Files generated successfully:")
print(f"  📝 Markdown: {md_file} ({os.path.getsize(md_file)} bytes)")
print(f"  📊 JSON:     {json_file} ({os.path.getsize(json_file)} bytes)")
print(f"  📄 Text:     {txt_file} ({os.path.getsize(txt_file)} bytes)")

# Download in Colab
if "google.colab" in sys.modules:
    from google.colab import files
    print("\\n⬇️ Triggering downloads...")
    for f in [md_file, json_file, txt_file]:
        files.download(f)
""")


def cell(i, kind, text):
    c = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": text}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        },
        "colab": {
            "provenance": []
        }
    },
    "cells": [cell(i, k, t) for i, (k, t) in enumerate(CELLS)],
}

out_path = pathlib.Path(__file__).parent / "document_structure_extractor.ipynb"
out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"✅ Generated Notebook: {out_path} ({len(CELLS)} cells)")
