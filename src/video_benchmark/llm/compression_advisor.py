"""LLM helper to fine-tune compression parameters."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from video_benchmark.compression import CompressionPlan, VideoProbe

logger = logging.getLogger(__name__)


def advise_compression(
    probe: VideoProbe,
    base_plan: CompressionPlan,
    api_key: str | None = None,
) -> CompressionPlan | None:
    """Ask Gemini to tweak compression settings for quality/size balance.

    Returns a possibly adjusted CompressionPlan; on any failure returns None.
    """
    if not api_key:
        logger.warning("No LLM API key provided; using base compression plan.")
        return None

    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning(
            "google-generativeai not installed. "
            "Install with: uv add --group llm google-generativeai"
        )
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
You are optimizing ffmpeg compression for wearable operator camera footage.
Input video probe:
{json.dumps(asdict(probe), indent=2)}

Current plan:
{json.dumps(asdict(base_plan), indent=2)}

Goals:
- Minimize file size while keeping action readability and faces legible.
- Prefer settings that are fast to encode on laptops; avoid exotic filters.
- Return ONLY a JSON object with keys: codec, preset, crf (int or null),
  video_bitrate (string or null), scale (string or null), audio_bitrate (string).
- Do not include any other text.
"""
    try:
        response = model.generate_content(prompt)
        text = response.text.strip() if response and response.text else ""
        data: dict[str, Any] = json.loads(text)
        return CompressionPlan(
            codec=data.get("codec", base_plan.codec),
            preset=data.get("preset", base_plan.preset),
            crf=data.get("crf", base_plan.crf),
            video_bitrate=data.get("video_bitrate", base_plan.video_bitrate),
            scale=data.get("scale", base_plan.scale),
            audio_bitrate=data.get("audio_bitrate", base_plan.audio_bitrate),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM compression advice failed: %s", exc)
        return None
