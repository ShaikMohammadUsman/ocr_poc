"""
OCR Service using Azure Document Intelligence
Supports: typed text, handwriting, tables, forms, mixed-layout PDFs
"""

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption
import io
import os
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException

load_dotenv()

endpoint = os.getenv("OCR_ENDPOINT")
key = os.getenv("OCR_KEY")

MAX_PAGES = 50  # Liberal page limit for general documents


def get_client() -> DocumentIntelligenceClient:
    if not endpoint or not key:
        raise RuntimeError("OCR_ENDPOINT and OCR_KEY must be set in .env")
    return DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def _extract_tables_as_text(result) -> dict:
    """
    Convert table data into a readable ASCII text block, keyed by page number (1-indexed).
    Returns a dict: { page_number: [table_text_block, ...] }
    """
    page_tables: dict = {}

    if not result.tables:
        return page_tables

    for table_idx, table in enumerate(result.tables):
        # Determine which page this table is on (use first bounding region)
        page_num = 1
        if table.bounding_regions:
            page_num = table.bounding_regions[0].page_number

        # Build a 2-D grid
        rows = table.row_count
        cols = table.column_count
        grid = [["" for _ in range(cols)] for _ in range(rows)]

        for cell in table.cells:
            r, c = cell.row_index, cell.column_index
            grid[r][c] = (cell.content or "").replace("\n", " ").strip()

        # Format as ASCII-style table
        col_widths = [
            max((len(grid[r][c]) for r in range(rows)), default=4)
            for c in range(cols)
        ]
        separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        lines = [f"\n[Table {table_idx + 1}]", separator]
        for r in range(rows):
            row_str = "| " + " | ".join(
                grid[r][c].ljust(col_widths[c]) for c in range(cols)
            ) + " |"
            lines.append(row_str)
            lines.append(separator)

        table_text = "\n".join(lines)
        page_tables.setdefault(page_num, []).append(table_text)

    return page_tables


def _extract_tables_as_grids(result) -> dict:
    """
    Return raw 2-D grids (list of rows, each row a list of cell strings)
    keyed by page number. Used for CSV export in the frontend.
    Returns: { page_number: [ [[r0c0, r0c1, ...], [r1c0, ...]], ... ] }
    """
    page_grids: dict = {}

    if not result.tables:
        return page_grids

    for table_idx, table in enumerate(result.tables):
        page_num = 1
        if table.bounding_regions:
            page_num = table.bounding_regions[0].page_number

        rows = table.row_count
        cols = table.column_count
        grid = [["" for _ in range(cols)] for _ in range(rows)]

        for cell in table.cells:
            r, c = cell.row_index, cell.column_index
            grid[r][c] = (cell.content or "").replace("\n", " ").strip()

        page_grids.setdefault(page_num, []).append(grid)

    return page_grids


async def extract_text_with_ocr(pdf_file: UploadFile) -> dict:
    """
    Extract all content from a PDF using Azure Document Intelligence.

    Returns a dict with:
      - full_text: str  (plain concatenated text, good for downstream NLP)
      - pages: list of {page_num, text, tables, tables_data}
      - page_count: int
      - has_handwriting: bool
      - has_tables: bool
    """
    try:
        content = await pdf_file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        client = get_client()

        # Use 'prebuilt-layout' — best for mixed PDFs (typed + handwriting + tables)
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=io.BytesIO(content),
            content_type="application/octet-stream",
        )
        result = poller.result()

        if not result.pages:
            raise HTTPException(status_code=422, detail="No pages found in document.")

        if len(result.pages) > MAX_PAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Document has {len(result.pages)} pages; limit is {MAX_PAGES}.",
            )

        # Detect if any handwriting is present
        has_handwriting = False
        if result.styles:
            has_handwriting = any(
                getattr(style, "is_handwritten", False) for style in result.styles
            )

        has_tables = bool(result.tables)

        # Build per-page text using paragraph spans mapped to pages
        page_tables = _extract_tables_as_text(result)
        page_grids  = _extract_tables_as_grids(result)

        # Collect paragraphs per page
        page_paragraphs: dict = {}
        if result.paragraphs:
            for para in result.paragraphs:
                pg = 1
                if para.bounding_regions:
                    pg = para.bounding_regions[0].page_number
                page_paragraphs.setdefault(pg, []).append(para.content or "")

        pages_output = []
        full_text_parts = []

        for page in result.pages:
            pg = page.page_number
            para_texts = page_paragraphs.get(pg, [])
            tables_text = page_tables.get(pg, [])

            page_text = "\n".join(para_texts)
            tables_combined = "\n".join(tables_text)

            combined = page_text
            if tables_combined:
                combined = page_text + "\n" + tables_combined

            pages_output.append({
                "page_num": pg,
                "text": page_text.strip(),
                "tables": tables_combined.strip(),
                "tables_data": page_grids.get(pg, []),   # list of 2-D grids for CSV export
            })
            full_text_parts.append(combined.strip())

        full_text = "\n\n--- Page Break ---\n\n".join(full_text_parts)

        return {
            "full_text": full_text.strip(),
            "pages": pages_output,
            "page_count": len(result.pages),
            "has_handwriting": has_handwriting,
            "has_tables": has_tables,
            "filename": pdf_file.filename,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
