"""Deterministic ingestion + parallel transcription for orchestration and gating."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from podcast_intel_agent.agent import ingest_latest_episodes, transcribe_intro_snippet
from podcast_intel_agent.config import TRANSCRIBE_MAX_SECONDS
from podcast_intel_agent.date_format import format_published_for_briefing


def gather_briefing_data(
    feed_urls: list[str],
    *,
    max_transcribe_seconds: int | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Ingest feeds (parallel inside ``ingest_latest_episodes``), transcribe intros in parallel.

    Returns episodes merged with a ``transcription`` dict per row, ``successful_transcripts`` count,
    and ``correlation_id`` for logs.
    """
    cid = correlation_id or str(uuid.uuid4())
    cap = max_transcribe_seconds if max_transcribe_seconds is not None else TRANSCRIBE_MAX_SECONDS
    ing = ingest_latest_episodes(feed_urls)
    if ing.get("status") != "ok":
        return {
            "status": "error",
            "correlation_id": cid,
            "ingest": ing,
            "episodes": [],
            "successful_transcripts": 0,
        }

    episodes: list[dict[str, Any]] = list(ing["episodes"])
    results: list[dict[str, Any] | None] = [None] * len(episodes)

    def work(idx: int, ep: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if ep.get("error") or not ep.get("audio_url"):
            return idx, {**ep, "transcription": {"status": "skipped", "error": ep.get("error") or "no audio URL"}}
        tr = transcribe_intro_snippet(ep["audio_url"], max_seconds=cap)
        return idx, {**ep, "transcription": tr}

    max_workers = min(3, max(1, len(episodes)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(work, i, episodes[i]) for i in range(len(episodes))]
        for fut in as_completed(futures):
            idx, merged = fut.result()
            results[idx] = merged

    merged_eps = [r for r in results if r is not None]
    ok_count = sum(1 for e in merged_eps if (e.get("transcription") or {}).get("status") == "ok")

    return {
        "status": "ok",
        "correlation_id": cid,
        "episodes": merged_eps,
        "successful_transcripts": ok_count,
        "max_transcribe_seconds": cap,
    }


def episodes_to_synthesis_json(data: dict[str, Any]) -> str:
    """Strip to JSON-serializable payload for the synthesis-only LLM."""
    slim: list[dict[str, Any]] = []
    for ep in data.get("episodes") or []:
        tr = ep.get("transcription") or {}
        slim.append(
            {
                "feed_url": ep.get("feed_url"),
                "podcast_title": ep.get("podcast_title"),
                "episode_title": ep.get("episode_title"),
                "author": ep.get("author"),
                "published": format_published_for_briefing(ep.get("published")),
                "ingest_error": ep.get("error"),
                "transcript": tr.get("transcript") if tr.get("status") == "ok" else None,
                "transcription_error": None if tr.get("status") == "ok" else tr.get("error", tr.get("status")),
            },
        )
    payload = {
        "correlation_id": data.get("correlation_id"),
        "successful_transcripts": data.get("successful_transcripts"),
        "episodes": slim,
    }
    return json.dumps(payload, indent=2)
