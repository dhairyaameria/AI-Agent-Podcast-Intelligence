#!/usr/bin/env python3
"""Run the briefing job: pipeline + synthesis via Gemini (ADK), Groq, or OpenAI (OpenAI-compatible SDK)."""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

from google.genai import types

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

APP_NAME = "podcast_intel_agent"
USER_ID = "briefing_user"
SESSION_ID = "briefing_session"
OUT_FILE = Path(__file__).resolve().parent / "intelligence_briefing.md"


def _require_api_keys(*, tools_only: bool, backend: str) -> None:
    from podcast_intel_agent import config

    if tools_only or backend == "gemini":
        if not (config.GOOGLE_API_KEY or config.GEMINI_API_KEY):
            print(
                "Missing GOOGLE_API_KEY or GEMINI_API_KEY in .env (required for Gemini / ADK_TOOLS_ONLY).",
                file=sys.stderr,
            )
            raise RuntimeError("No Gemini API key in environment.")
    if not tools_only and backend == "groq":
        if not config.GROQ_API_KEY:
            print("SYNTHESIS_BACKEND=groq requires GROQ_API_KEY in .env", file=sys.stderr)
            raise RuntimeError("No GROQ_API_KEY in environment.")
    if not tools_only and backend == "openai":
        if not config.OPENAI_API_KEY:
            print("SYNTHESIS_BACKEND=openai requires OPENAI_API_KEY in .env", file=sys.stderr)
            raise RuntimeError("No OPENAI_API_KEY in environment.")


def _llm_rate_limit_wait() -> None:
    from podcast_intel_agent import config
    from podcast_intel_agent.resilience import TokenBucket

    if config.LLM_MIN_INTERVAL_SEC > 0:
        time.sleep(config.LLM_MIN_INTERVAL_SEC)

    refill = config.LLM_TOKEN_BUCKET_REFILL_PER_SEC
    if refill is not None and refill > 0:
        cap = config.LLM_TOKEN_BUCKET_CAPACITY
        if cap > 0:
            TokenBucket(capacity=cap, refill_per_second=refill).acquire(1.0)


async def _run_adk_agent(agent, user_text: str) -> str:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    content = types.Content(role="user", parts=[types.Part(text=user_text)])
    final_text = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_text += part.text
    return final_text.strip()


async def run_briefing() -> str:
    import podcast_intel_agent.env_bootstrap  # noqa: F401
    from podcast_intel_agent import config

    tools_only = config.ADK_TOOLS_ONLY
    backend = config.SYNTHESIS_BACKEND
    if tools_only and backend != "gemini":
        print(
            "ADK_TOOLS_ONLY=1 always uses Gemini (root_agent). Set SYNTHESIS_BACKEND=gemini in .env.",
            file=sys.stderr,
        )
        backend = "gemini"

    _require_api_keys(tools_only=tools_only, backend=backend)

    feeds = config.FEED_URLS
    correlation_id = str(uuid.uuid4())
    print(
        f'{{"event":"orchestrator_start","correlation_id":"{correlation_id}","feeds":3,"synthesis_backend":"{backend}"}}',
        file=sys.stderr,
    )

    if tools_only:
        from podcast_intel_agent.agent import root_agent

        user_text = (
            "Generate today's podcast intelligence briefing using exactly these three RSS feeds "
            f"(in order): {feeds[0]!r}, {feeds[1]!r}, {feeds[2]!r}. "
            "Call ingest_latest_episodes with this list, then transcribe each episode that has an audio URL."
        )
        final_text = await _run_adk_agent(root_agent, user_text)
    else:
        from podcast_intel_agent.pipeline import episodes_to_synthesis_json, gather_briefing_data

        data = gather_briefing_data(feeds, correlation_id=correlation_id)
        if data.get("status") != "ok":
            print(
                f'{{"event":"ingest_fatal","correlation_id":"{correlation_id}","detail":{data!r}}}',
                file=sys.stderr,
            )
            raise RuntimeError(f"Ingestion pipeline failed: {data}")

        need = config.ORCHESTRATOR_MIN_SUCCESS
        ok_n = int(data.get("successful_transcripts") or 0)
        if ok_n < need:
            msg = (
                f"ORCHESTRATOR_ABORT: need>={need} successful transcripts, got {ok_n}. "
                f"correlation_id={correlation_id}"
            )
            print(msg, file=sys.stderr)
            raise RuntimeError(msg)

        _llm_rate_limit_wait()
        user_text = (
            "Produce the briefing from this pipeline JSON only (no tools).\n\n"
            + episodes_to_synthesis_json(data)
        )

        if backend in ("groq", "openai"):
            from podcast_intel_agent.compat_synthesis import synthesize_briefing_openai_compat

            final_text = synthesize_briefing_openai_compat(user_text, backend=backend)
        else:
            from podcast_intel_agent.agent import synthesis_agent

            final_text = await _run_adk_agent(synthesis_agent, user_text)

    if not final_text.strip():
        raise RuntimeError(
            "Empty briefing output; check API keys, model names, and SYNTHESIS_BACKEND in .env.",
        )
    OUT_FILE.write_text(final_text.strip() + "\n", encoding="utf-8")
    print(
        f'{{"event":"briefing_written","correlation_id":"{correlation_id}","path":{str(OUT_FILE)!r}}}',
        file=sys.stderr,
    )
    return str(OUT_FILE)


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import podcast_intel_agent.env_bootstrap  # noqa: F401 — ensure .env before asyncio imports config

    try:
        out = asyncio.run(run_briefing())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
