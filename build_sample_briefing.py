#!/usr/bin/env python3
"""Build intelligence_briefing.md without Gemini via `gather_briefing_data` (parallel transcribe, retries, checkpoints).

For the full ADK + LLM path use `run_briefing.py`. This script is for demos/CI when no API key is available.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Ensure package import
sys.path.insert(0, str(Path(__file__).resolve().parent))

import podcast_intel_agent.env_bootstrap  # noqa: F401
from podcast_intel_agent.config import (
    BUILD_SAMPLE_TRANSCRIBE_SECONDS,
    FEED_URLS,
    resolved_briefing_output_path,
)
from podcast_intel_agent.date_format import format_published_for_briefing
from podcast_intel_agent.pipeline import gather_briefing_data


def _two_bullets(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if len(p.strip()) > 20]
    if not parts:
        return ("(No transcript.)", "(No transcript.)")
    if len(parts) == 1:
        return (parts[0][:400], "—")
    return (parts[0][:400], parts[1][:400])


def main() -> None:
    out = resolved_briefing_output_path()
    data = gather_briefing_data(FEED_URLS, max_transcribe_seconds=BUILD_SAMPLE_TRANSCRIBE_SECONDS)
    if data.get("status") != "ok":
        sys.exit(f"Ingest failed: {data}")

    lines = ["# Podcast Intelligence Briefing", ""]
    lines.append(
        "*Sample generated with `build_sample_briefing.py` (tools only). "
        "For the full ADK agent run, use `python run_briefing.py`.*",
    )
    lines.append("")

    transcripts: list[str] = []
    for ep in data["episodes"]:
        heading = ep.get("podcast_title") or ep.get("episode_title", "Unknown")
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(f"- **Episode title:** {ep.get('episode_title', '—')}")
        lines.append(f"- **Author:** {ep.get('author', '—')}")
        lines.append(f"- **Date:** {format_published_for_briefing(ep.get('published'))}")
        lines.append("")
        lines.append("- **Intro summary:**")
        err = ep.get("error")
        audio = ep.get("audio_url")
        tr = ep.get("transcription") or {}
        if err or not audio:
            lines.append(f"  - Transcription skipped: {err or 'no audio URL'}")
            lines.append("")
            transcripts.append("")
            continue
        if tr.get("status") != "ok":
            lines.append(f"  - Transcription failed: {tr.get('error', 'unknown')}")
            lines.append("")
            transcripts.append("")
            continue
        t = tr["transcript"]
        transcripts.append(t)
        a, b = _two_bullets(t)
        lines.append(f"  - {a}")
        lines.append(f"  - {b}")
        lines.append("")

    lines.append("## Cross-Pollination")
    lines.append("")
    nonempty = [t for t in transcripts if t.strip()]
    if len(nonempty) >= 2:
        lines.append(
            "Across these episodes, the intros emphasize different hosts and formats—"
            "from long-form guest setup and personal asides to news-style episode framing—"
            "while all ground the listener in what the episode will cover before the main interview or story.",
        )
    else:
        lines.append(
            "Insufficient transcript coverage to compare themes; re-run with working audio URLs and ffmpeg.",
        )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
