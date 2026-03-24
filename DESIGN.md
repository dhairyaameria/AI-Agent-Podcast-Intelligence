# Podcast Intelligence Pipeline — Design Document

This document explains the end-to-end flow (as in the architecture diagram), how components map to this repository, and the **resource-conscious techniques** used to stay within CPU, network, and API limits.

## Purpose

The system runs on a schedule (conceptually: **cron / Cloud Scheduler** at e.g. 6 AM). It pulls the **latest episode** from **three** RSS feeds, transcribes only a **short intro** from each episode, optionally **gates** on partial success, rate-limits the **LLM** call, and writes a single Markdown file: **`intelligence_briefing.md`**.

In code, the default path is **`python run_briefing.py`**: deterministic ingest + parallel transcription → **≥ 2 / 3** successful transcripts → **one** synthesis call (Gemini via ADK, or Groq / OpenAI-compatible). An alternate path uses the ADK **`root_agent`** with tools (`ADK_TOOLS_ONLY=1`), which trades **more LLM turns** for conversational orchestration.

---

## Layered architecture (diagram → implementation)

| Layer | Diagram concept | Implementation |
|--------|-----------------|----------------|
| **1 — Trigger & orchestration** | Cron / Cloud Scheduler; root orchestrator with timeouts | **You** schedule `run_briefing.py`. Orchestration logic: `gather_briefing_data` + `run_briefing.py` gates; optional **`root_agent`** for tool-driven runs. Structured **stderr** JSON includes **`correlation_id`** for tracing. |
| **2 — Extraction (podcast agents)** | Three parallel agents: RSS, extract, retry | **`ingest_latest_episodes`** processes **three** feeds (isolated per URL) with **`retry_sync`** (exponential backoff + jitter). Failures are recorded; healthy feeds continue. |
| **3 — Transcription workers** | Stream crop 3–5 min; Whisper-base; async | **`transcribe_intro_snippet`**: **ffmpeg** reads the audio URL and decodes only the first **N** seconds (`TRANSCRIBE_MAX_SECONDS`, default ~5 minutes). **Whisper** model tier from **`WHISPER_MODEL`** (`tiny` / **`base`** / `small`). Parallel fan-out via **`ThreadPoolExecutor`** in **`gather_briefing_data`**. |
| **4 — Checkpoint store** | GCS / local disk; skip on re-run | **`.checkpoints/`** (path from **`CHECKPOINT_DIR`**): one JSON file per **(audio URL, crop length)** after a successful transcript. Re-runs **skip** re-transcription when the checkpoint exists. *GCS* is listed in the diagram as a scaling option; this repo uses **local disk** by default. |
| **5 — Rate limit & aggregation** | Token bucket, exp. backoff; collect 2/3+ | Before synthesis: **`LLM_MIN_INTERVAL_SEC`** and optional **`TokenBucket`** (`LLM_TOKEN_BUCKET_*`). **`ORCHESTRATOR_MIN_SUCCESS`** (default **2**): if fewer transcripts succeed, the run **aborts** before the LLM (no briefing file). |
| **6 — Synthesis** | Gemini (or equivalent) | **`synthesis_agent`** (ADK) or **`synthesize_briefing_openai_compat`** (Groq/OpenAI). Input is **JSON only** from **`episodes_to_synthesis_json`** — no tools on this path. |
| **7 — Output & monitoring** | `intelligence_briefing.md`; dead letter; Slack/email | **`intelligence_briefing.md`**. **`dead_letter.jsonl`**: append-only log for RSS/transcription failures after retries. **Slack/email on partial failure** is **not** built in: wrap the job in your scheduler and alert on **non-zero exit** or on stderr patterns (e.g. `ORCHESTRATOR_ABORT`). |

---

## End-to-end flow (happy path)

1. **Start** — Load **`.env`** via **`env_bootstrap`**; validate **`config`** (three distinct feeds, model names, paths).
2. **Ingest** — For each RSS URL, fetch/parse with retries; produce episode metadata and **enclosure** audio URL (or per-feed error).
3. **Transcribe (parallel)** — Up to **three** workers; each successful run writes a **checkpoint**; exhausted retries → **dead letter** line.
4. **Aggregate** — Count **`successful_transcripts`**. If **&lt; `ORCHESTRATOR_MIN_SUCCESS`**, stop with a clear stderr message and **correlation_id**.
5. **Throttle** — Sleep / token bucket so the **single** synthesis request respects provider limits.
6. **Synthesize** — LLM produces Markdown: per-show blocks (**title, author, date, two-bullet intro**) and a **Cross-Pollination** paragraph (see **`SYNTHESIS_INSTRUCTION`** in **`synthesis_prompt.py`**).
7. **Write** — Persist **`intelligence_briefing.md`**.

---

## Artifacts

| Artifact | Role |
|----------|------|
| **`intelligence_briefing.md`** | Final user-facing briefing. |
| **`.checkpoints/*.json`** | Cached transcripts; idempotent re-runs. |
| **`dead_letter.jsonl`** | Failed feeds/transcriptions for later inspection or retry. |

---

## Resource limitations and how we addressed them (“tricks”)

These are the main levers that keep **compute**, **bandwidth**, and **LLM quota** under control.

| Technique | Why it matters | Where it lives |
|-----------|----------------|----------------|
| **Intro-only transcription (ffmpeg `-t`)** | Full episodes are long; transcribing everything is slow and expensive. We only decode/transcribe the **first N seconds** from the stream URL. | **`transcribe_intro_snippet`** in **`agent.py`**; **`TRANSCRIBE_MAX_SECONDS`** in **`.env`**. |
| **Smaller Whisper tier (`base`, or `tiny`)** | Larger Whisper models improve quality but multiply CPU/GPU time. **`WHISPER_MODEL`** defaults favor **speed and footprint** over maximum accuracy for this use case. | **`config.py`**, **`agent.py`**. |
| **Checkpointing** | Retries or partial failures should not redo successful ASR. Checkpoints key off URL + crop length. | **`.checkpoints/`**, **`_checkpoint_file`** in **`agent.py`**. |
| **Per-feed isolation + RSS retries** | One bad feed must not kill the others. **`retry_sync`** adds **exponential backoff + jitter** to avoid hammering flaky hosts. | **`ingest_latest_episodes`**, **`resilience.retry_sync`**. |
| **Dead-letter queue** | After retries, failures are **logged** instead of failing silently or crashing the whole job. | **`dead_letter.jsonl`**, **`_dead_letter_append`**. |
| **Partial success gate (≥ 2 of 3)** | The product can still be useful if **one** stream fails; strict “all three or nothing” would waste successful work and API budget. | **`ORCHESTRATOR_MIN_SUCCESS`**, **`run_briefing.py`**. |
| **Single synthesis LLM call (default path)** | Each LLM round trip costs **tokens** and **rate-limit budget**. The default pipeline does **one** structured call with trimmed JSON payload. | **`run_briefing.py`** + **`episodes_to_synthesis_json`**. |
| **LLM rate limiting (token bucket + minimum interval)** | Stays under **RPM/TPM** style limits; combines with provider-side retries where applicable. | **`TokenBucket`**, **`_llm_rate_limit_wait`** in **`run_briefing.py`**. |
| **Parallel transcription with bounded workers** | Overlap I/O and ASR without unbounded threads (capped at **three** for three feeds). | **`gather_briefing_data`** in **`pipeline.py`**. |

**Operational note:** For **free-tier Gemini** limits, the README recommends shortening **`TRANSCRIBE_MAX_SECONDS`** (fewer **input tokens**), choosing a quota-friendly **`GEMINI_MODEL`**, or switching **`SYNTHESIS_BACKEND`** to Groq/OpenAI — same pipeline, different chat provider.

---

## Configuration surface

All tunables are driven from **`.env`** (see **`.env.example`**). Notable keys for this design:

- **`PODCAST_RSS_URLS`** — exactly three comma-separated feeds.  
- **`TRANSCRIBE_MAX_SECONDS`**, **`WHISPER_MODEL`** — ASR cost vs. coverage.  
- **`PODCAST_RSS_RETRIES`**, **`PODCAST_TRANSCRIBE_RETRIES`** — resilience vs. latency.  
- **`CHECKPOINT_DIR`**, **`DEAD_LETTER_PATH`** — persistence locations.  
- **`ORCHESTRATOR_MIN_SUCCESS`** — usually **2**.  
- **`LLM_MIN_INTERVAL_SEC`**, **`LLM_TOKEN_BUCKET_CAPACITY`**, **`LLM_TOKEN_BUCKET_REFILL_PER_SEC`** — synthesis throttle.  
- **`SYNTHESIS_BACKEND`**, model names, API keys — provider selection.

---

## Scaling beyond three feeds (diagram → production)

The diagram generalizes to **N** shows: keep **parallel workers** with **timeouts**, **checkpoints** (local or **object storage** with content-addressed keys), **queues** for ASR, and **one rate-limited synthesis** step (or batched calls if the API allows). The README section **“Scaling to ~50 podcasts every morning”** expands on coordinator pattern, GPU/managed ASR, and token-bucket boundaries.

---

## Summary

The pipeline matches the diagram’s intent: **scheduled trigger → parallel RSS → cropped async transcription → checkpoints → partial-success aggregation → rate-limited LLM → Markdown**, with **retries**, **dead-letter logging**, and **minimal LLM usage** on the default path so the system remains practical under **CPU, network, and API** constraints.
