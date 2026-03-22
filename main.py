"""
FastAPI application — PDF OCR extraction service
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from ocr_service import extract_text_with_ocr
from ai_extractor import extract_key_values, chat_with_document

app = FastAPI(
    title="PDF OCR Extraction API",
    description="Extract text, tables, and handwriting from any PDF using Azure Document Intelligence",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


from typing import List

@app.post("/extract", summary="Extract text + structured data from a PDF")
async def extract(files: List[UploadFile] = File(...)):
    """
    Upload up to 5 PDF files and receive:
    - Extracted text (per page)
    - Table content
    - AI-generated key-value pairs (JSON) via GPT-4o-mini
    """
    if len(files) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 files can be uploaded."
        )

    allowed_types = {"application/pdf", "application/octet-stream"}
    results = []

    for file in files:
        fname = file.filename or f"Document_{len(results)+1}.pdf"
        if file.content_type not in allowed_types and not fname.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported. {fname} is not a valid PDF.",
            )

        # Step 1: OCR extraction
        ocr_result = await extract_text_with_ocr(file)

        # Step 2: AI structured key-value extraction
        try:
            kv_data = extract_key_values(ocr_result["full_text"])
        except Exception as e:
            kv_data = {"error": f"AI extraction failed: {str(e)}"}

        ocr_result["key_values"] = kv_data
        ocr_result["filename"] = fname
        import csv
        known = set()
        categories = {"Metadata", "Income", "Expense", "Other"}
        csv_path = os.path.join(os.path.dirname(__file__), "housing_category_mapping.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    norm = row.get("Normalized Name", "").strip()
                    cat = row.get("Category", "").strip()
                    if norm: known.add(norm)
                    if cat: categories.add(cat)
        ocr_result["known_fields"] = list(known)
        ocr_result["categories"] = list(categories)

        results.append(ocr_result)

    # Detect Overlapping Months
    MONTH_MAP = {
        "january": "January", "jan": "January",
        "february": "February", "feb": "February",
        "march": "March", "mar": "March",
        "april": "April", "apr": "April",
        "may": "May",
        "june": "June", "jun": "June",
        "july": "July", "jul": "July",
        "august": "August", "aug": "August",
        "september": "September", "sep": "September",
        "october": "October", "oct": "October",
        "november": "November", "nov": "November",
        "december": "December", "dec": "December"
    }

    # Enhanced Overlapping Month detection
    import re
    from datetime import datetime
    
    def extract_periods(text, ai_data=None):
        text_lower = text.lower()
        found_periods = set()
        
        # 0. Try to use AI-extracted metadata first (highly reliable)
        if ai_data and "Metadata" in ai_data:
            meta = ai_data["Metadata"]
            m = meta.get("statement_month", "").strip()
            y = meta.get("statement_year", "").strip()
            if m and y:
                # Normalize month name
                m_norm = MONTH_MAP.get(m.lower(), m.capitalize())
                found_periods.add(f"{m_norm} {y}")
                return found_periods # If AI found it, we trust it

        # 1. Look for definitive Month Year patterns (e.g. "January 2024", "Jan 2024", "2024 January")
        month_names = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        month_shorts = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        all_months = month_names + month_shorts
        
        month_regex = "|".join(all_months)
        year_regex = r"(20\d{2})"
        
        # Pattern: Month Year or Month, Year
        pattern1 = rf"\b({month_regex})\b[\s,.-]*\b{year_regex}\b"
        for m in re.finditer(pattern1, text_lower):
            m_str = m.group(1)
            y_str = m.group(2)
            full_m = MONTH_MAP.get(m_str, m_str.capitalize())
            found_periods.add(f"{full_m} {y_str}")
            
        # Pattern: Year Month
        pattern2 = rf"\b{year_regex}\b[\s,.-]*\b({month_regex})\b"
        for m in re.finditer(pattern2, text_lower):
            y_str = m.group(1)
            m_str = m.group(2)
            full_m = MONTH_MAP.get(m_str, m_str.capitalize())
            found_periods.add(f"{full_m} {y_str}")
            
        # 2. If no "Month Year" pair found, look for ANY year and ANY month
        if not found_periods:
            detected_months = set()
            for m_key, full_m in MONTH_MAP.items():
                if re.search(rf"\b{m_key}\b", text_lower):
                    detected_months.add(full_m)
            
            detected_years = set(re.findall(r"\b20\d{2}\b", text_lower))
            
            if detected_months:
                if len(detected_years) == 1:
                    year = list(detected_years)[0]
                    for m in detected_months:
                        found_periods.add(f"{m} {year}")
                else:
                    # Fallback to month only
                    for m in detected_months:
                        found_periods.add(m)
                        
        return found_periods

    # Use a list of (filename, periods) to handle duplicate filenames
    file_info_list = []
    for res in results:
        periods = extract_periods(res.get("full_text", ""), res.get("key_values"))
        file_info_list.append((res["filename"], periods))
        res["detected_months"] = list(periods)
        
    overlaps = {}
    for filename, periods in file_info_list:
        for p in periods:
            overlaps.setdefault(p, []).append(filename)
            
    warnings = []
    for period, fnames in overlaps.items():
        if len(fnames) > 1:
            joined_fnames = " and ".join(fnames) if len(fnames) == 2 else ", ".join(fnames[:-1]) + ", and " + fnames[-1]
            warnings.append(f"The files {joined_fnames} appear to overlap for the period: {period}")

    return JSONResponse(content={"results": results, "warnings": warnings})


@app.get("/health")
async def health():
    return {"status": "ok", "service": "PDF OCR Extraction", "version": "2.0.0"}


class ChatRequest(BaseModel):
    question: str
    document_text: str
    history: list = []  # list of {"role": "user"|"assistant", "content": "..."}


@app.post("/chat", summary="Chat with the extracted document")
async def chat(req: ChatRequest):
    """
    Ask any question about the uploaded document.
    Provide the full_text from the /extract response, plus the user's question.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not req.document_text.strip():
        raise HTTPException(status_code=400, detail="No document text provided.")
    try:
        answer = chat_with_document(
            question=req.question,
            document_text=req.document_text,
            history=req.history,
        )
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

