# Podcast Intelligence Agent (Google ADK)

Code-first agent using [Google ADK](https://google.github.io/adk-docs/) (`google-adk`) for tooling and optional Gemini synthesis: ingests **three** RSS feeds in parallel, transcribes the first ~5 minutes of each latest episode with Whisper (ffmpeg crop, no full download), and produces a Markdown intelligence briefing via **Gemini**, **Groq**, or **OpenAI** (same pipeline; synthesis backend is configurable).

**Default feeds:** Dwarkesh Podcast, Acquired, Darknet Diaries (override with `PODCAST_RSS_URLS`).

## Architecture (this repo)

| Piece | Role |
|------|------|
| `podcast_intel_agent/agent.py` | ADK **`root_agent`** (tools: `ingest_latest_episodes`, `transcribe_intro_snippet`) for **ADK Web / `adk run`**. ADK **`synthesis_agent`** (no tools) turns pipeline JSON into Markdown. |
| `podcast_intel_agent/pipeline.py` | **`gather_briefing_data`**: deterministic ingest + parallel transcription; **`episodes_to_synthesis_json`** for the synthesis prompt. |
| `podcast_intel_agent/resilience.py` | **`retry_sync`** (exponential backoff + jitter), **`TokenBucket`** (optional LLM throttle). |
| `podcast_intel_agent/env_bootstrap.py` | Loads project **`.env`** once at import time. |
| `podcast_intel_agent/config.py` | **All** tunables (`GEMINI_MODEL`, `WHISPER_MODEL`, `TRANSCRIBE_MAX_SECONDS`, `PODCAST_RSS_URLS`, `ADK_TOOLS_ONLY`, `ORCHESTRATOR_MIN_SUCCESS`, `GROQ_MODEL`, paths, etc.) — **no duplicate defaults in Python**. |
| `podcast_intel_agent/synthesis_prompt.py` | Shared **`SYNTHESIS_INSTRUCTION`** for Gemini ADK and OpenAI-compatible APIs. |
| `podcast_intel_agent/compat_synthesis.py` | **`synthesize_briefing_openai_compat`** — one Chat Completions call for **`SYNTHESIS_BACKEND=groq`** or **`openai`**. |
| `run_briefing.py` | Pipeline → **≥2/3** gate → synthesis via **`SYNTHESIS_BACKEND`** (`gemini` / **`groq`** / **`openai`**). **`ADK_TOOLS_ONLY=1`** always uses Gemini **`root_agent`**. |
| `build_sample_briefing.py` | Same ingestion/transcription path as the pipeline (**no LLM**), writes `intelligence_briefing.md` for demos/CI. |

**Artifacts:** `intelligence_briefing.md` (output), `.checkpoints/` (transcript cache), `dead_letter.jsonl` (hard failures after retries).

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (e.g. `brew install ffmpeg` on macOS; on Colab: `apt-get update && apt-get install -y ffmpeg`)
- **`GOOGLE_API_KEY`** from [Google AI Studio](https://aistudio.google.com/app/apikey) when using **`SYNTHESIS_BACKEND=gemini`** (default) or **`ADK_TOOLS_ONLY=1`**
- **`GROQ_API_KEY`** or **`OPENAI_API_KEY`** when using **`SYNTHESIS_BACKEND=groq`** or **`openai`** (see below)

## Setup

```bash
cd Felix-Assignment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create **`.env`** in this directory. **Every** tunable is listed in **`.env.example`** as real assignments (not buried in Python). Copy the file and edit:

```bash
cp .env.example .env
# Then set API keys and SYNTHESIS_BACKEND (gemini / groq / openai). Leave unused keys empty.

```

Required keys include: **`SYNTHESIS_BACKEND`**, **`GEMINI_MODEL`**, **`WHISPER_MODEL`**, **`PODCAST_RSS_URLS`**, **`TRANSCRIBE_MAX_SECONDS`**, **`BUILD_SAMPLE_TRANSCRIBE_SECONDS`**, **`ADK_TOOLS_ONLY`** (`0` or `1`), **`ORCHESTRATOR_MIN_SUCCESS`**, **`GROQ_MODEL`**, **`GROQ_BASE_URL`**, **`OPENAI_MODEL`**, retry counts, **`CHECKPOINT_DIR`**, **`DEAD_LETTER_PATH`**, **`LLM_MIN_INTERVAL_SEC`**, **`LLM_TOKEN_BUCKET_CAPACITY`**, and optionally empty **`LLM_TOKEN_BUCKET_REFILL_PER_SEC`**. See **`.env.example`** for the full list and example values.

If the app raises **`Missing … in .env`**, add that variable from **`.env.example`**.

## Run (end-to-end)

```bash
source .venv/bin/activate
python run_briefing.py
```

By default this runs **deterministic ingestion + parallel transcription** (retries, checkpoints, dead-letter on hard failures), enforces **at least 2/3 successful transcripts**, applies optional **LLM rate limiting**, then **synthesizes** Markdown from pipeline JSON via **`SYNTHESIS_BACKEND`**: **`gemini`** (ADK **`synthesis_agent`**), **`groq`**, or **`openai`** (OpenAI SDK). Output: **`intelligence_briefing.md`**.

Structured **stderr** lines include **`correlation_id`** for tracing.

### Free tier (Gemini quota)

Google enforces **separate** caps (typically **requests/minute**, **input tokens/minute**, **requests/day**) per **project**. Hitting any one returns **429** — even if you only run this script once, if your **daily** budget for that model is already zero, you must wait for the reset or switch models. [See your live limits](https://aistudio.google.com/rate-limit) and the [official rate-limit doc](https://ai.google.dev/gemini-api/docs/rate-limits). Daily quotas reset at **midnight Pacific**.

**How this repo stays cheap**

- Default **`run_briefing.py`** path makes **one** `generateContent` call (synthesis only). Whisper + RSS are **local / free**.
- Do **not** use **`ADK_TOOLS_ONLY=1`** or **`adk web`** for your “real” run if you are quota-limited — those paths can burn **multiple** LLM turns per briefing.

**If you get 429**

1. In **`.env`**, set **`GEMINI_MODEL`** to a **Flash** or **Flash-Lite** model your key still has quota for (names change over time; pick one listed in [AI Studio](https://aistudio.google.com/) for your project). Many accounts do well with **`gemini-2.5-flash-lite`** or **`gemini-2.5-flash`** when **`gemini-2.0-flash`** is exhausted.
2. Lower **`TRANSCRIBE_MAX_SECONDS`** (e.g. **`180`** or **`120`**) so the synthesis prompt is **shorter** — fewer **input tokens** per call (helps TPM limits).
3. Wait for the **retry** hint in the error (per-minute) or until the next **Pacific** day (per-day).
4. For a submission with **no Gemini usage at all**, run **`python build_sample_briefing.py`** — you still get real RSS + Whisper output in **`intelligence_briefing.md`**.
5. Or set **`SYNTHESIS_BACKEND=groq`** (or **`openai`**) and use a **Groq** / **OpenAI** key so the **same** `run_briefing.py` pipeline still runs end-to-end with a different chat model.

### Tool-grounded sample (no Gemini)

```bash
python build_sample_briefing.py
```

Uses **`gather_briefing_data`** (parallel transcribe, checkpoints, retries) with a shorter snippet window (`SAMPLE_MAX_SECONDS=90` in script). Overwrites **`intelligence_briefing.md`**. For LLM-polished briefing, use **`run_briefing.py`** with **`SYNTHESIS_BACKEND=gemini`**, **`groq`**, or **`openai`**.

## ADK dev UI (optional)

From the parent directory of the agent package:

```bash
adk web --port 8000
```

Select **`podcast_intel_agent`** and chat. For CLI: **`adk run podcast_intel_agent`** (from the directory that contains the **`podcast_intel_agent`** folder). This uses **`root_agent`** with tools (same behavior as **`ADK_TOOLS_ONLY=1`** on `run_briefing.py`).

## Tools (ADK)

| Tool | Role |
|------|------|
| `ingest_latest_episodes` | Parses 3 RSS URLs **in parallel** (isolated per feed, **retry_sync**); returns latest episode metadata and audio URL per feed. |
| `transcribe_intro_snippet` | ffmpeg reads the URL and decodes only the first N seconds (`-t`), then Whisper transcribes; **retries**, **checkpoint** on success, **dead letter** on final failure. |

## Scaling to ~50 podcasts every morning (assignment answer)

**Fault tolerance (boundaries in this codebase):**

| Failure point | How it is handled |
|---|---|
| RSS down / parse error | Per-feed isolation + exponential-backoff retries (`PODCAST_RSS_RETRIES`); failures append to **`dead_letter.jsonl`**; other feeds continue. |
| Audio / ffmpeg failure | **`transcribe_intro_snippet`** retries (`PODCAST_TRANSCRIBE_RETRIES`); successes **checkpointed** under **`.checkpoints/`**; final failure → dead letter. |
| Transcription crash mid-run | Re-run **skips** completed episodes via checkpoint files (keyed by URL + crop length). |
| LLM rate limit | Optional **token bucket** (`LLM_TOKEN_BUCKET_REFILL_PER_SEC`, `LLM_TOKEN_BUCKET_CAPACITY`) or **`LLM_MIN_INTERVAL_SEC`** before the synthesis call. |
| Fewer than 2 of 3 successful transcripts | Orchestrator **aborts** before synthesis (`ORCHESTRATOR_MIN_SUCCESS`, default **2**); **no** `intelligence_briefing.md`; stderr alert includes **`correlation_id`**. |
| Full run failure | **correlation_id** in stderr; wrap **`run_briefing.py`** in a scheduler and alert **Slack/email** on non-zero exit. |

**At 50 shows** a single monolithic LLM loop is slow and brittle. A practical pattern:

1. **Coordinator** — Validates the feed list, fans out work, aggregates the final briefing; does not run ASR itself.
2. **Parallel fan-out** — One worker (or ADK branch) per podcast: RSS → audio URL → transcription task; timeouts per task. Same idea as this repo’s **ThreadPoolExecutor** + isolated per-feed RSS, generalized to **N** feeds and a **job queue**.
3. **Synthesis agent** — One (or batched) LLM call over structured records (metadata + intro transcript). Keeps **ingestion and ASR deterministic** outside the model (as here with **`gather_briefing_data`** + **`synthesis_agent`**).
4. **Idempotency** — Key work by `(feed_id, episode_guid_or_url, date)`; skip or resume (like **checkpoints** + optional **GCS** with content-hash keys for zero-cost cache hits on re-runs).

**Transcription compute:** 50 × ~5 min of audio/day is heavy on CPU Whisper; prefer a **GPU pool**, **Kubernetes / Cloud Run Jobs**, or **managed ASR** / **faster-whisper** behind **Celery + Redis** (or similar) with horizontal workers. Always **crop in ffmpeg** before decode (as in this project).

**LLM throughput:** **Token bucket + queue** at the synthesis boundary; **batch** episodes where the API allows; **exponential backoff + jitter** on 429/5xx; optional **model tiering** (small model for per-episode notes, larger once for cross-show analysis).

Together: **queued RSS + ASR workers**, **parallel fan-out with timeouts**, and **one rate-limited synthesis step** scale cleanly to dozens of shows.

## Assignment deliverables

1. **Code** — this repository (or copy to Colab / GitHub).
2. **`intelligence_briefing.md`** — from a successful **`python run_briefing.py`** (API key + network), or **`build_sample_briefing.py`** for a tools-only sample.
3. **Scaling / fault-tolerance** — section **Scaling to ~50 podcasts every morning** above.
