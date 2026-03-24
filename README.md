# Podcast Intelligence Agent · Google ADK

A code-first AI agent using [Google ADK](https://google.github.io/adk-docs/) (`google-adk`) that ingests **three RSS feeds in parallel**, transcribes the first ~5 minutes of each latest episode with Whisper (**ffmpeg** applies **`-t` before `-i`** so the HTTP input stops after ~N seconds of media — not a full-episode download), and produces a structured Markdown intelligence briefing via **Gemini**, **Groq**, or **OpenAI**.

**Default feeds:** Dwarkesh Podcast · Acquired · Darknet Diaries (override with `PODCAST_RSS_URLS`)

---

## How it works (end-to-end flow)

```
Scheduled trigger (cron / Cloud Scheduler)
        │
        ▼
Root orchestrator
        │
        ├── Podcast agent 1 ──┐
        ├── Podcast agent 2 ──┼── parallel RSS ingest + retry ×3
        └── Podcast agent 3 ──┘
                │
                ▼
        Transcription workers (ThreadPoolExecutor)
        ffmpeg stream-crop → Whisper (base/tiny)
                │
                ▼
        Checkpoint store (.checkpoints/)
        Skip re-transcription on re-run
                │
                ▼
        ≥ 2/3 gate (ORCHESTRATOR_MIN_SUCCESS)
        Abort if too few succeeded
                │
                ▼
        Rate limit manager
        Token bucket + LLM_MIN_INTERVAL_SEC
                │
                ▼
        Synthesis agent (Gemini / Groq / OpenAI)
        One structured LLM call
                │
                ▼
        intelligence_briefing.md
```

---

## Architecture

| Component | Role |
|---|---|
| `podcast_intel_agent/agent.py` | ADK **`root_agent`** (tools: `ingest_latest_episodes`, `transcribe_intro_snippet`) for ADK Web / `adk run`. ADK **`synthesis_agent`** (no tools) turns pipeline JSON into Markdown. |
| `podcast_intel_agent/pipeline.py` | **`gather_briefing_data`**: deterministic ingest + parallel transcription. **`episodes_to_synthesis_json`** for the synthesis prompt. |
| `podcast_intel_agent/resilience.py` | **`retry_sync`** (exponential backoff + jitter), **`TokenBucket`** (optional LLM throttle). |
| `podcast_intel_agent/env_bootstrap.py` | Loads project `.env` once at import time. |
| `podcast_intel_agent/config.py` | All tunables — no duplicate defaults in Python. |
| `podcast_intel_agent/synthesis_prompt.py` | Shared `SYNTHESIS_INSTRUCTION` for Gemini ADK and OpenAI-compatible APIs. |
| `podcast_intel_agent/compat_synthesis.py` | **`synthesize_briefing_openai_compat`** — one Chat Completions call for `SYNTHESIS_BACKEND=groq` or `openai`. |
| `run_briefing.py` | Pipeline → ≥ 2/3 gate → one synthesis call. `ADK_TOOLS_ONLY=1` forces Gemini (same path; use **`adk web` / `adk run`** for `root_agent` with tools). |
| `build_sample_briefing.py` | Same ingest/transcription path with no LLM — writes `intelligence_briefing.md` for demos/CI. |

**Artifacts:** `intelligence_briefing.md` (output) · `.checkpoints/` (transcript cache) · `dead_letter.jsonl` (hard failures after retries)

---

## Fault tolerance

| Failure point | How it is handled |
|---|---|
| RSS down / parse error | Per-feed isolation + exponential-backoff retries (`PODCAST_RSS_RETRIES`); failures appended to `dead_letter.jsonl`; other feeds continue. |
| Audio / ffmpeg failure | `transcribe_intro_snippet` retries (`PODCAST_TRANSCRIBE_RETRIES`); successes checkpointed; final failure → dead letter. |
| Transcription crash mid-run | Re-run skips completed episodes via checkpoint files (keyed by URL + crop length). |
| LLM rate limit | Token bucket (`LLM_TOKEN_BUCKET_*`) or `LLM_MIN_INTERVAL_SEC` before synthesis call. |
| < 2 of 3 successful transcripts | Orchestrator aborts before synthesis; no `intelligence_briefing.md` written; stderr alert includes `correlation_id`. |
| Full run failure | `correlation_id` in stderr; wrap `run_briefing.py` in your scheduler and alert on non-zero exit. |

---

## Resource efficiency techniques

| Technique | Why | Where |
|---|---|---|
| **Intro-only crop (`-t` before `-i`)** | Limits **input** duration so the network read stops after ~N seconds of audio; then decode/transcribe that clip only | `transcribe_intro_snippet`, `TRANSCRIBE_MAX_SECONDS` |
| **Smaller Whisper tier** | `base` or `tiny` prioritises speed and footprint for this use case | `config.py`, `WHISPER_MODEL` |
| **Checkpointing** | Retries or partial failures don't redo successful ASR | `.checkpoints/`, `_checkpoint_file` |
| **Per-feed isolation + retries** | One bad feed doesn't kill the others; jitter avoids hammering flaky hosts | `ingest_latest_episodes`, `resilience.retry_sync` |
| **Dead-letter queue** | Failures logged instead of crashing the job | `dead_letter.jsonl`, `_dead_letter_append` |
| **Partial success gate (≥ 2/3)** | Still useful if one stream fails; avoids wasting successful work | `ORCHESTRATOR_MIN_SUCCESS`, `run_briefing.py` |
| **Single synthesis LLM call** | One structured call with trimmed JSON payload on the default path | `run_briefing.py`, `episodes_to_synthesis_json` |
| **Token bucket + min interval** | Stays under RPM/TPM limits | `TokenBucket`, `_llm_rate_limit_wait` |
| **Parallel transcription** | Overlaps I/O and ASR; capped at 3 workers for 3 feeds | `gather_briefing_data` in `pipeline.py` |

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on your `PATH`
  - macOS: `brew install ffmpeg`
  - Colab: `apt-get update && apt-get install -y ffmpeg`
- **`GOOGLE_API_KEY`** from [Google AI Studio](https://aistudio.google.com/app/apikey) when using `SYNTHESIS_BACKEND=gemini` (default) or `ADK_TOOLS_ONLY=1`
- **`GROQ_API_KEY`** or **`OPENAI_API_KEY`** when using `SYNTHESIS_BACKEND=groq` or `openai`

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Set API keys and SYNTHESIS_BACKEND (gemini / groq / openai)
```

All tunables are in `.env.example` as real assignments. Notable keys:

| Key | Purpose |
|---|---|
| `SYNTHESIS_BACKEND` | `gemini` / `groq` / `openai` |
| `PODCAST_RSS_URLS` | Exactly three comma-separated RSS feed URLs |
| `TRANSCRIBE_MAX_SECONDS` | Audio crop length (default ~300s / 5 min) |
| `WHISPER_MODEL` | `tiny` / `base` / `small` |
| `ORCHESTRATOR_MIN_SUCCESS` | Minimum successful transcripts before synthesis (default `2`) |
| `CHECKPOINT_DIR` | Path for transcript cache |
| `DEAD_LETTER_PATH` | Path for failure log |
| `LLM_MIN_INTERVAL_SEC` | Minimum gap before LLM call |
| `LLM_TOKEN_BUCKET_CAPACITY` | Token bucket size for LLM throttle |
| `PODCAST_RSS_RETRIES` | Per-feed retry attempts |
| `PODCAST_RSS_RETRY_BASE_DELAY_SEC` | First backoff delay for RSS retries (seconds; podcast CDNs often need longer gaps) |
| `PODCAST_TRANSCRIBE_RETRIES` | Per-transcription retry attempts |
| `BRIEFING_OUTPUT_PATH` | Markdown output path (relative to project root unless absolute) |

If the app raises `Missing … in .env`, add that variable from `.env.example`.

---

## Run

```bash
source .venv/bin/activate
python run_briefing.py
```

Runs deterministic ingest + parallel transcription → ≥ 2/3 gate → one synthesis call → `intelligence_briefing.md`.

### No LLM (tools-only sample)

```bash
python build_sample_briefing.py
```

Real RSS + Whisper output, no LLM call. Good for demos or CI.

### ADK dev UI (optional)

```bash
adk web --port 8000   # from the parent directory of podcast_intel_agent/
# or
adk run podcast_intel_agent
```

---

## Free-tier Gemini notes

Google enforces separate caps (RPM, input TPM, requests/day) per project. Hitting any one returns 429 even on first use if the daily budget is exhausted. [Check your live limits.](https://aistudio.google.com/rate-limit) Daily quotas reset at midnight Pacific.

**If you get 429:**
1. Switch `GEMINI_MODEL` to a Flash or Flash-Lite model with remaining quota (e.g. `gemini-2.5-flash-lite`)
2. Lower `TRANSCRIBE_MAX_SECONDS` to shorten the synthesis prompt
3. Set `SYNTHESIS_BACKEND=groq` — same pipeline, different provider
4. Run `build_sample_briefing.py` for a tools-only submission with no Gemini usage

---

## Scaling to 50 podcasts (assignment answer)

At 50 shows a single monolithic loop is slow and brittle. The architecture generalises as follows:

**Agent topology:** 1 root orchestrator fans out to N podcast agents (ADK `ParallelAgent`). Each agent is isolated — one failure doesn't affect others.

**Transcription compute:** Move Whisper off the main process into a Celery worker pool (backed by Redis). Each transcription job is a queued task — scale horizontally by adding workers or using Cloud Run Jobs. Always crop with ffmpeg before decode (as in this project).

**Pipelining:** `gather_briefing_data` runs RSS ingest (parallel) then transcription (parallel) as **two sequential phases**. For three feeds the cost is negligible; at **N ≫ 3** you would pipeline (start transcribing episode 1 while others still ingest) via a queue or async producer–consumer pattern.

**LLM throughput:** Token bucket + queue at the synthesis boundary. Batch episodes where the API allows. Exponential backoff + jitter on 429/5xx. Model tiering: small model for per-episode summaries, larger model once for cross-show analysis.

**Idempotency:** Key work by `(feed_id, episode_guid, date)`. Use GCS with content-hash keys for zero-cost cache hits on re-runs — same checkpoint pattern used here, generalised to object storage.

**Observability:** Structured logs per agent with `correlation_id` (already in this codebase) so you can trace exactly which podcast failed and why.

---

## Deliverables

1. **Code** — this repository
2. **`intelligence_briefing.md`** — from `python run_briefing.py` (with API key), or `build_sample_briefing.py` for a tools-only sample
3. **Scaling answer** — section above
