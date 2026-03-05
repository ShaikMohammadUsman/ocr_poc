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


@app.post("/extract", summary="Extract text + structured data from a PDF")
async def extract(file: UploadFile = File(...)):
    """
    Upload a PDF file and receive:
    - Extracted text (per page)
    - Table content
    - AI-generated key-value pairs (JSON) via GPT-4o-mini
    """
    allowed_types = {"application/pdf", "application/octet-stream"}
    if file.content_type not in allowed_types and not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    # Step 1: OCR extraction
    ocr_result = await extract_text_with_ocr(file)

    # Step 2: AI structured key-value extraction
    try:
        kv_data = extract_key_values(ocr_result["full_text"])
    except Exception as e:
        kv_data = {"error": f"AI extraction failed: {str(e)}"}

    ocr_result["key_values"] = kv_data
    return JSONResponse(content=ocr_result)


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

