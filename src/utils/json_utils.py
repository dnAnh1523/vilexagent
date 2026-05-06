# src/utils/json_utils.py
"""
Robust JSON extraction from LLM outputs.
Handles all common failure cases:
  - markdown fences with or without language tag (```json ... ```)
  - conversational filler before/after the JSON block
  - empty or whitespace-only responses
  - nested braces (extracts the outermost valid JSON object)
"""

import re
import json

def extract_json_from_llm_output(raw: str) -> dict:
    """
    Extract and parse the first valid JSON object from an LLM response string.

    Raises:
        ValueError: if no valid JSON object can be found or parsed.
    """
    if not raw or not raw.strip():
        raise ValueError("LLM returned an empty response")

    # Step 1: strip markdown fences if present
    # Handles ```json ... ```, ``` ... ```, and variations with whitespace
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        raw = fence_match.group(1).strip()

    # Step 2: find the outermost {...} block in whatever remains
    # This handles cases where the LLM prepends a sentence before the JSON
    brace_match = re.search(r"\{[\s\S]*\}", raw)
    if not brace_match:
        raise ValueError(f"No JSON object found in LLM output: {raw[:200]!r}")

    candidate = brace_match.group(0).strip()

    # Step 3: parse
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e} | candidate was: {candidate[:300]!r}")