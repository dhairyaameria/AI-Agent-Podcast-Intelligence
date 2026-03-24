"""Synthesis via OpenAI-compatible Chat Completions (OpenAI, Groq, OpenRouter, etc.)."""

from __future__ import annotations

from podcast_intel_agent.config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from podcast_intel_agent.synthesis_prompt import SYNTHESIS_INSTRUCTION


def synthesize_briefing_openai_compat(user_message: str, *, backend: str) -> str:
    """One chat completion: system = synthesis rules, user = pipeline JSON + header."""
    from openai import OpenAI

    if backend == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError("SYNTHESIS_BACKEND=groq requires GROQ_API_KEY in .env")
        client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
        model = GROQ_MODEL
    elif backend == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("SYNTHESIS_BACKEND=openai requires OPENAI_API_KEY in .env")
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        model = OPENAI_MODEL
    else:
        raise RuntimeError(f"compat_synthesis: backend must be groq or openai, got {backend!r}")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYNTHESIS_INSTRUCTION},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    choice = resp.choices[0].message
    text = (choice.content or "").strip()
    if not text:
        raise RuntimeError("OpenAI-compatible API returned empty content.")
    return text
