"""
Azure OpenAI structured extraction service.
Takes raw OCR text and asks GPT-4o-mini to return ALL key-value pairs
it can identify from the document, as a flat or nested JSON object.
"""

import os
import json
import re
import csv
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


def get_system_prompt() -> str:
    categories_set = set()
    mapping_text = ""
    csv_path = os.path.join(os.path.dirname(__file__), "housing_category_mapping.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                norm_name = row.get("Normalized Name", "")
                cat = row.get("Category", "")
                fields = row.get("Possible Field Names", "")
                if cat:
                    categories_set.add(cat)
                if norm_name:
                    mapping_text += f"- Normalize [{fields}] to '{norm_name}' under category '{cat}'\n"
                else:
                    mapping_text += f"- {cat}: Use this for fields like {fields}\n"
    
    categories = list(categories_set)
    if not categories:
        categories = ["Income", "Expense", "Other"]
        mapping_text = "- No explicit mapping provided.\n"

    # Always add 'Other' just in case
    if "Other" not in categories:
        categories.append("Other")

    top_level_keys = ", ".join(f'"{c}"' for c in categories)

    return f"""You are a highly accurate document data extractor.
Your job is to read the raw text extracted from a PDF document and identify
every meaningful key-value pair, field, or piece of structured information present.

Crucially, you must group these fields under specific top-level categories: {top_level_keys}.

Use the following mapping as a strict source of truth to normalize and categorize fields.
If a field in the document matches or means the same as any of the "Possible Field Names" in a rule below, 
you must change its field name exactly to the unquoted Normalized name and place it under its Category.

{mapping_text}

Rules:
- Return ONLY a valid JSON object. No markdown fences, no explanation.
- The JSON object must consist of the top-level keys: {top_level_keys}.
- Normalize the extracted key-value pairs to the specified names and categories based on the mapping above.
- If a section has multiple items (e.g. a list of products, education history),
  represent them as an array of objects under an appropriate key within its category.
- If a value is a table, represent it as a list of row objects.
- Group related fields under a nested object if it makes semantic sense.
- Do NOT skip any field you can identify — be as exhaustive as possible.
- If a field is found that doesn't fit the mapping, include it with a clean descriptive key in the "Other" category.
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

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
            ],
            temperature=0,
            max_tokens=15000, # Increased from 4096 to prevent early cutoff
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"

        # Strip any accidental markdown fences
        raw = re.sub(r"^```json\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())

        try:
            return json.loads(raw)
        except json.JSONDecodeError as err:
            if response.choices[0].finish_reason == "length":
                return {
                    "error": "Extraction incomplete because the document generated too much data and reached output limits.",
                    "partial_data_length": len(raw)
                }
            return {"error": f"AI extraction failed to generate valid JSON: {str(err)}", "partial_data_length": len(raw)}
            
    except Exception as e:
        return {"error": f"AI extraction failed: {str(e)}"}


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

