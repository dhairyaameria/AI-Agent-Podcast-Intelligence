"""All tunable defaults live in `.env` — no duplicated defaults here (only validation/clamping)."""

from __future__ import annotations

import os
from pathlib import Path

from podcast_intel_agent import env_bootstrap  # noqa: F401 — side effect: load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _req(key: str) -> str:
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        raise RuntimeError(
            f"Missing {key} in .env (set it explicitly — see .env.example).",
        )
    return str(raw).strip()


def _opt(key: str) -> str:
    return (os.environ.get(key) or "").strip()


def _bool_env(key: str) -> bool:
    v = _req(key).lower()
    return v in ("1", "true", "yes", "on")


def _int_env(key: str) -> int:
    return int(_req(key))


# --- Models & transcription ---
GEMINI_MODEL = _req("GEMINI_MODEL")
WHISPER_MODEL = _req("WHISPER_MODEL").lower()
if WHISPER_MODEL not in ("tiny", "base", "small"):
    raise RuntimeError("WHISPER_MODEL must be one of: tiny, base, small")

TRANSCRIBE_MAX_SECONDS = max(60, min(_int_env("TRANSCRIBE_MAX_SECONDS"), 600))

_raw_feeds = _req("PODCAST_RSS_URLS")
FEED_URLS = [u.strip() for u in _raw_feeds.split(",") if u.strip()]
if len(FEED_URLS) != 3 or len(set(FEED_URLS)) != 3:
    raise RuntimeError("PODCAST_RSS_URLS must be exactly three distinct comma-separated URLs")

ADK_TOOLS_ONLY = _bool_env("ADK_TOOLS_ONLY")

ORCHESTRATOR_MIN_SUCCESS = max(1, min(3, _int_env("ORCHESTRATOR_MIN_SUCCESS")))

GROQ_MODEL = _req("GROQ_MODEL")
GROQ_BASE_URL = _req("GROQ_BASE_URL")

OPENAI_MODEL = _req("OPENAI_MODEL")
OPENAI_BASE_URL = _opt("OPENAI_BASE_URL") or None

SYNTHESIS_BACKEND = _req("SYNTHESIS_BACKEND").lower()
if SYNTHESIS_BACKEND not in ("gemini", "groq", "openai"):
    raise RuntimeError("SYNTHESIS_BACKEND must be gemini, groq, or openai")

PODCAST_RSS_RETRIES = max(1, _int_env("PODCAST_RSS_RETRIES"))
PODCAST_TRANSCRIBE_RETRIES = max(1, _int_env("PODCAST_TRANSCRIBE_RETRIES"))

CHECKPOINT_DIR = _req("CHECKPOINT_DIR")
DEAD_LETTER_PATH = _req("DEAD_LETTER_PATH")

LLM_MIN_INTERVAL_SEC = float(_req("LLM_MIN_INTERVAL_SEC"))
_refill = _opt("LLM_TOKEN_BUCKET_REFILL_PER_SEC")
LLM_TOKEN_BUCKET_REFILL_PER_SEC: float | None = float(_refill) if _refill else None
LLM_TOKEN_BUCKET_CAPACITY = float(_req("LLM_TOKEN_BUCKET_CAPACITY"))

BUILD_SAMPLE_TRANSCRIBE_SECONDS = max(60, min(_int_env("BUILD_SAMPLE_TRANSCRIBE_SECONDS"), 600))

# API keys (optional — whichever backend you use must be non-empty at runtime)
GOOGLE_API_KEY = _opt("GOOGLE_API_KEY")
GEMINI_API_KEY = _opt("GEMINI_API_KEY")
GROQ_API_KEY = _opt("GROQ_API_KEY")
OPENAI_API_KEY = _opt("OPENAI_API_KEY")


def resolved_checkpoint_dir() -> Path:
    p = Path(CHECKPOINT_DIR)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolved_dead_letter_path() -> Path:
    p = Path(DEAD_LETTER_PATH)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p
