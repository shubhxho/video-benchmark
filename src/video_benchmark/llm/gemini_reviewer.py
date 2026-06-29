"""Optional Tier 2 LLM-based video quality review using Gemini."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Default Gemini model. Override via VB_GEMINI_MODEL or by editing this constant.
GEMINI_MODEL = "gemini-2.5-flash"

_INSTALL_HINT = (
    "google-genai not installed. Install with: uv add --group llm google-genai"
)


def review_with_gemini(
    scores: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    api_key: str | None = None,
) -> str | None:
    """Send scoring results to Gemini for qualitative review.

    Requires the google-genai package (install with: uv add --group llm google-genai).
    Returns None when the package or API key is unavailable, or on any API error.
    """
    try:
        from google import genai
    except ImportError:
        logger.warning(_INSTALL_HINT)
        return None

    if not api_key:
        logger.warning("No Gemini API key provided. Skipping LLM review.")
        return None

    prompt = f"""Analyze these video quality benchmark results for operator headband cameras.

Rankings (top 10):
{json.dumps(rankings[:10], indent=2)}

Provide a brief assessment:
1. Overall quality of the operator camera fleet
2. Key patterns in the data
3. Recommendations for operators with low scores
4. Any concerning trends

Keep the response concise (under 200 words).
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = response.text
        return text if isinstance(text, str) else None
    except Exception:
        logger.exception("Gemini review failed")
        return None
