"""
Azure OpenAI structured extraction service.
Takes raw OCR text and asks GPT-4o-mini to return ALL key-value pairs
it can identify from the document, as a flat or nested JSON object.
"""

import os
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_API_VERSION", "2025-01-01-preview"),
        )
    return _client


SYSTEM_PROMPT = """You are a highly accurate document data extractor.
Your job is to read the raw text extracted from a PDF document and identify
every meaningful key-value pair, field, or piece of structured information present.

Rules:
- Return ONLY a valid JSON object. No markdown fences, no explanation.
- Use clean, human-readable key names (e.g. "Full Name", "Date of Birth", "Invoice Total").
- If a section has multiple items (e.g. a list of products, education history),
  represent them as an array of objects under an appropriate key.
- If a value is a table, represent it as a list of row objects.
- Group related fields under a nested object if it makes semantic sense
  (e.g. "Address": {"Street": "...", "City": "...", "PIN": "..."}).
- Do NOT skip any field you can identify — be as exhaustive as possible.
- If something is unclear, include it with a best-guess key.
- Never return null values; use empty string "" if value is missing/illegible.
"""

USER_PROMPT_TEMPLATE = """Here is the full text extracted from a PDF document:

---
{text}
---

Extract every possible key-value pair from this document and return a single JSON object.
"""


def extract_key_values(ocr_text: str) -> dict:
    """
    Send OCR text to GPT-4o-mini and return a structured dict of all
    key-value pairs found in the document.
    """
    if not ocr_text or not ocr_text.strip():
        return {"error": "No text provided for AI extraction"}

    # Truncate if too long (GPT-4o-mini context ~128k tokens, ~500k chars safe)
    text = ocr_text[:60_000] if len(ocr_text) > 60_000 else ocr_text

    client = _get_client()

    response = client.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
        temperature=0,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"

    # Strip any accidental markdown fences
    raw = re.sub(r"^```json\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())

    return json.loads(raw)


CHAT_SYSTEM_PROMPT = """You are a helpful document assistant. The user has uploaded a PDF document and you have
been given its full extracted text below. Answer the user's questions accurately and concisely based ONLY on
the information contained in the document. If the answer cannot be found in the document, say so clearly.

=== DOCUMENT TEXT START ===
{document_text}
=== DOCUMENT TEXT END ===
"""


def chat_with_document(question: str, document_text: str, history: list = []) -> str:
    """
    Answer a user question based on the provided document text.
    `history` is a list of previous messages: [{"role": "user"|"assistant", "content": "..."}]
    """
    # Truncate doc text to stay within context limits
    doc_text = document_text[:50_000] if len(document_text) > 50_000 else document_text

    client = _get_client()

    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT.format(document_text=doc_text)},
    ]
    # Append conversation history (last 10 turns max)
    for msg in history[-20:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Append the new user question
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini"),
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content or "I couldn't generate a response."

