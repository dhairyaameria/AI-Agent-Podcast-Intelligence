"""ADK agent: RSS ingestion + intro transcription tools and briefing LlmAgent."""

from __future__ import annotations

import email.utils
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser

from google.adk.agents import LlmAgent

from podcast_intel_agent.config import (
    GEMINI_MODEL,
    PODCAST_RSS_RETRIES,
    PODCAST_RSS_RETRY_BASE_DELAY_SEC,
    PODCAST_TRANSCRIBE_RETRIES,
    WHISPER_MODEL,
    resolved_checkpoint_dir,
    resolved_dead_letter_path,
)
from podcast_intel_agent.resilience import retry_sync
from podcast_intel_agent.synthesis_prompt import SYNTHESIS_INSTRUCTION

_WHISPER_LOCK = threading.Lock()
_WHISPER_MODEL = None
_WHISPER_MODEL_NAME: str | None = None


def _get_whisper_model(model_name: str):
    """Load Whisper once; thread-safe for parallel transcribers."""
    global _WHISPER_MODEL, _WHISPER_MODEL_NAME
    with _WHISPER_LOCK:
        if _WHISPER_MODEL is None or _WHISPER_MODEL_NAME != model_name:
            import whisper  # lazy: heavy import

            _WHISPER_MODEL = whisper.load_model(model_name)
            _WHISPER_MODEL_NAME = model_name
        return _WHISPER_MODEL


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _dead_letter_append(record: dict[str, Any]) -> None:
    record = {**record, "ts": datetime.now(timezone.utc).isoformat()}
    path = resolved_dead_letter_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _rss_retries() -> int:
    return PODCAST_RSS_RETRIES


def _transcribe_retries() -> int:
    return PODCAST_TRANSCRIBE_RETRIES


def _parse_rss_datetime(entry: feedparser.FeedParserDict) -> str | None:
    t = entry.get("published_parsed") or entry.get("updated_parsed")
    if not t:
        raw = entry.get("published") or entry.get("updated")
        if raw:
            try:
                dt = email.utils.parsedate_to_datetime(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except (TypeError, ValueError):
                return raw
        return None
    try:
        dt = datetime(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, tzinfo=timezone.utc)
        return dt.isoformat()
    except (TypeError, ValueError):
        return None


def _audio_url_from_entry(entry: feedparser.FeedParserDict) -> str | None:
    for enc in entry.get("enclosures") or []:
        href = enc.get("href")
        if not href:
            continue
        typ = (enc.get("type") or "").lower()
        if typ.startswith("audio") or href.endswith((".mp3", ".m4a", ".mp4", ".ogg", ".wav")):
            return href
    if entry.get("enclosures"):
        href = entry["enclosures"][0].get("href")
        if href:
            return href
    for link in entry.get("links") or []:
        typ = (link.get("type") or "").lower()
        if typ.startswith("audio"):
            return link.get("href")
    media = entry.get("media_content") or []
    if media:
        url = media[0].get("url")
        if url:
            return url
    return None


def _author_for_entry(entry: feedparser.FeedParserDict, feed: feedparser.FeedParserDict) -> str:
    return (
        entry.get("author")
        or entry.get("itunes_author")
        or entry.get("dc_creator")
        or feed.feed.get("author")
        or feed.feed.get("itunes_author")
        or feed.feed.get("title")
        or "Unknown"
    )


def _ingest_single_feed(url: str) -> dict[str, Any]:
    """Parse one feed with retries; isolated failures do not raise."""

    def attempt() -> dict[str, Any]:
        parsed = feedparser.parse(
            url,
            agent="PodcastIntelAgent/1.0 (+https://github.com/dhairyaameria/AI-Agent-Podcast-Intelligence)",
        )
        if parsed.bozo and not parsed.entries:
            bozo = getattr(parsed, "bozo_exception", "unknown")
            raise RuntimeError(f"Failed to parse feed: {bozo}")
        entries = list(parsed.entries)
        if not entries:
            raise RuntimeError("No entries in feed.")

        dated = [
            e
            for e in entries
            if e.get("published_parsed") or e.get("updated_parsed")
        ]
        if dated:
            latest = max(
                dated,
                key=lambda e: e.get("published_parsed") or e.get("updated_parsed"),
            )
        else:
            latest = entries[-1]
        audio = _audio_url_from_entry(latest)
        item: dict[str, Any] = {
            "feed_url": url,
            "podcast_title": parsed.feed.get("title") or parsed.feed.get("subtitle") or "Unknown show",
            "episode_title": latest.get("title") or "Untitled",
            "author": _author_for_entry(latest, parsed),
            "published": _parse_rss_datetime(latest),
            "audio_url": audio,
        }
        if not audio:
            item["error"] = "No audio enclosure found for latest episode."
        return item

    try:
        return retry_sync(
            attempt,
            max_attempts=_rss_retries(),
            base_delay=PODCAST_RSS_RETRY_BASE_DELAY_SEC,
        )
    except Exception as exc:  # noqa: BLE001
        _dead_letter_append({"phase": "rss", "feed_url": url, "error": str(exc)})
        return {"feed_url": url, "error": str(exc)}


def ingest_latest_episodes(feed_urls: list[str]) -> dict[str, Any]:
    """Parse 3 podcast RSS feeds in parallel and return metadata plus audio URL for the latest episode of each.

    Args:
        feed_urls: Exactly three distinct public RSS or Atom feed URLs.

    Returns:
        JSON-serializable dict with key "episodes", each with feed_url, episode_title, author,
        published, audio_url, and optional error.
    """
    if not isinstance(feed_urls, list) or len(feed_urls) != 3:
        return {
            "status": "error",
            "error": "feed_urls must be a list of exactly 3 RSS feed URLs.",
        }
    if len(set(feed_urls)) != 3:
        return {"status": "error", "error": "feed_urls must contain 3 distinct URLs."}

    episodes: list[dict[str, Any] | None] = [None] * 3

    def job(idx: int, u: str) -> None:
        episodes[idx] = _ingest_single_feed(u)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(job, i, feed_urls[i]) for i in range(3)]
        for fut in as_completed(futs):
            fut.result()

    return {"status": "ok", "episodes": [e for e in episodes if e is not None]}


def _transcribe_intro_impl(audio_url: str, max_seconds: int) -> dict[str, Any]:
    """Single attempt: ffmpeg crop + Whisper (no checkpoint, no retry)."""
    if not audio_url or not isinstance(audio_url, str):
        return {"status": "error", "error": "audio_url must be a non-empty string."}
    cap = max(60, min(int(max_seconds), 600))

    tmp_dir = tempfile.mkdtemp(prefix="podcast_intel_")
    try:
        out_wav = os.path.join(tmp_dir, "clip.wav")
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return {"status": "error", "error": "ffmpeg not found on PATH; install ffmpeg."}

        # -t before -i: input duration limit so HTTP read stops after ~cap seconds of media.
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-user_agent",
            "PodcastIntelAgent/1.0",
            "-t",
            str(cap),
            "-i",
            audio_url,
            "-y",
            "-ar",
            "16000",
            "-ac",
            "1",
            out_wav,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)

        model_name = WHISPER_MODEL
        wmodel = _get_whisper_model(model_name)
        result = wmodel.transcribe(out_wav, fp16=False)
        text = (result.get("text") or "").strip()
        return {
            "status": "ok",
            "audio_url": audio_url,
            "seconds_processed": cap,
            "transcript": text,
            "whisper_model": model_name,
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "ffmpeg timed out reading audio.", "audio_url": audio_url}
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc))[:500]
        return {"status": "error", "error": f"ffmpeg failed: {err}", "audio_url": audio_url}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc), "audio_url": audio_url}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _checkpoint_file(audio_url: str, cap: int) -> Path:
    key = hashlib.sha256(f"{audio_url}\n{cap}".encode()).hexdigest()[:32]
    return resolved_checkpoint_dir() / f"{key}.json"


def transcribe_intro_snippet(audio_url: str, max_seconds: int = 300) -> dict[str, Any]:
    """Stream the first max_seconds of audio (ffmpeg input duration limit), transcribe with Whisper tiny/base.

    Uses checkpoint files and exponential-backoff retries; failures append to dead_letter.jsonl.

    Args:
        audio_url: Direct HTTP(S) URL to an audio file or stream.
        max_seconds: Length of the snippet (default 300 = 5 minutes). Assignment: 3–5 minutes.

    Returns:
        Dict with transcript text, seconds_processed, audio_url, and status.
    """
    if not audio_url or not isinstance(audio_url, str):
        return {"status": "error", "error": "audio_url must be a non-empty string."}
    cap = max(60, min(int(max_seconds), 600))
    ck = _checkpoint_file(audio_url, cap)
    if ck.is_file():
        try:
            data = json.loads(ck.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("status") == "ok" and data.get("transcript") is not None:
                return data
        except (OSError, json.JSONDecodeError):
            pass

    def attempt() -> dict[str, Any]:
        out = _transcribe_intro_impl(audio_url, cap)
        if out.get("status") != "ok":
            raise RuntimeError(out.get("error", "transcription failed"))
        return out

    try:
        result = retry_sync(attempt, max_attempts=_transcribe_retries())
        try:
            ck.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return result
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        _dead_letter_append({"phase": "transcription", "audio_url": audio_url, "error": err})
        return {"status": "error", "error": err, "audio_url": audio_url}


_BRIEFING_INSTRUCTION = """You are a podcast intelligence analyst. You MUST use tools — never guess RSS or audio content.

Workflow:
1) Call `ingest_latest_episodes` once with the three feed URLs the user provides (ingestion runs in parallel per feed with retries).
2) For each episode that has an `audio_url` and no `error`, call `transcribe_intro_snippet` with that URL (use max_seconds=300 unless the user specifies otherwise).
3) **Orchestration rule:** Only produce the full Markdown briefing if at least **2 of 3** podcast pipelines yield a usable intro transcript (`transcribe_intro_snippet` status ok). If fewer than 2 succeed, respond with a short Markdown note titled `# Podcast Intelligence Briefing (aborted)` explaining that the orchestrator requires 2/3 successes, list which feeds failed or lacked audio, and do **not** invent episode content.
4) When you have ≥2 successful transcripts, write the final answer ONLY as Markdown (no JSON wrapper).

Markdown format (use these headings exactly):
- Start with title: `# Podcast Intelligence Briefing`
- For each of the three feeds in order, a section `##` with the `podcast_title` from ingestion (fallback: episode title).
  Under it include lines:
  - **Episode title:** (exact string from tools)
  - **Author:** (from ingestion)
  - **Date:** (from ingestion `published` field — if it looks like ISO ``2026-03-20T16:00:55+00:00``, rewrite for readers as a plain calendar date, e.g. **March 20, 2026**, without changing the actual day)
  - **Intro summary:** then exactly two Markdown bullet lines (`- `) summarizing only the intro, grounded strictly in the transcript. If transcription failed, use one bullet stating the error and do not invent content.
- Then `## Cross-Pollination` — one short paragraph comparing themes, tone, or contrasts across the three shows using only grounded facts from the transcripts and metadata.

Rules: Do not invent episode titles, authors, or dates. Do not fabricate transcript content. If a tool returns an error, state that briefly and continue."""


root_agent = LlmAgent(
    name="podcast_intelligence_agent",
    model=GEMINI_MODEL,
    description="Ingests three podcast RSS feeds, transcribes intro audio snippets, and writes a Markdown intelligence briefing.",
    instruction=_BRIEFING_INSTRUCTION,
    tools=[ingest_latest_episodes, transcribe_intro_snippet],
)

synthesis_agent = LlmAgent(
    name="podcast_synthesis_agent",
    model=GEMINI_MODEL,
    description="Turns checkpointed ingestion+transcription JSON into the Markdown intelligence briefing.",
    instruction=SYNTHESIS_INSTRUCTION,
    tools=[],
)
