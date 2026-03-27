# Voice Agent

A fully local, sub-second voice assistant built for financial transfer support. Speak a question, and the agent transcribes it, classifies your intent, and responds with synthesised speech — all in under one second after warmup.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works — End-to-End Pipeline](#how-it-works--end-to-end-pipeline)
3. [Why It's Fast (Sub-1s Response)](#why-its-fast-sub-1s-response)
4. [3-Stage Classification Pipeline](#3-stage-classification-pipeline)
   - [Stage 1 — Embedding Similarity](#stage-1--embedding-similarity)
   - [TOP_K Averaging](#top_k-averaging)
   - [Stage 2 — LLM Disambiguation](#stage-2--llm-disambiguation)
   - [Stage 3 — Final Decision & Fallback](#stage-3--final-decision--fallback)
5. [Fallback Logic](#fallback-logic)
6. [Caching Systems](#caching-systems)
   - [Embedding Cache](#embedding-cache)
   - [TTS Cache](#tts-cache)
7. [Startup Warmup](#startup-warmup)
8. [Intents](#intents)
9. [Recording Management](#recording-management)
10. [Frontend](#frontend)
11. [Project Structure](#project-structure)
12. [Requirements & Prerequisites](#requirements--prerequisites)
13. [Installation](#installation)
14. [Running the Agent](#running-the-agent)
15. [Configuration Reference](#configuration-reference)
16. [Timing Breakdown](#timing-breakdown)

---

## Overview

Voice Agent is a locally-run voice interface for a financial transfer support bot. It has no cloud dependencies for inference — everything (STT, intent classification, TTS) runs on your machine. The pipeline is designed so that after an initial warmup phase, responses are delivered in under one second.

The agent handles questions about Wise (international money transfers) such as:

- "Where is my money?"
- "When will my transfer arrive?"
- "What is a proof of payment?"
- "What is a UTR number?"

If the user's query doesn't match any known intent with sufficient confidence, the agent gracefully falls back to a human handoff response.

---

## How It Works — End-to-End Pipeline

```
User speaks
    │
    ▼
[Browser] Records audio via MediaRecorder API (WebM format)
    │
    ▼  HTTP POST /voice  (multipart form data)
[Backend — FastAPI]
    │
    ├─► [STT] faster-whisper transcribes audio → plain text
    │
    ├─► [Classifier] 3-stage intent classifier → (intent_key, confidence)
    │       │
    │       ├─ Stage 1: Cosine similarity against pre-embedded examples
    │       ├─ Stage 2: LLM disambiguation (only if Stage 1 is uncertain)
    │       └─ Stage 3: Accept or fall back to human_handoff
    │
    ├─► [Intents] Look up canned response text for the resolved intent
    │
    └─► [TTS] edge-tts synthesises audio → base64-encoded MP3
            │
            ▼  Server-Sent Events stream back to browser
[Browser] Plays audio + displays debug panel (transcript, intent, confidence, timings)
```

The entire pipeline runs as a **Server-Sent Events (SSE) stream**. This means the browser receives incremental status updates (`transcribing → classifying → generating_audio → done`) while the backend is still working, keeping the UI responsive.

---

## Why It's Fast (Sub-1s Response)

Several architectural decisions combine to produce sub-second response times after warmup:

### 1. faster-whisper with int8 quantisation
`faster-whisper` is a CTranslate2-optimised reimplementation of OpenAI Whisper. Running the `base` model with `compute_type="int8"` reduces memory and CPU inference time significantly compared to the original PyTorch Whisper. Typical transcription of a 3–5 second recording takes **50–200 ms** on CPU.

### 2. Pre-computed & cached intent embeddings
At startup, every example phrase for every intent is embedded using `mxbai-embed-large` through Ollama. These embeddings are stored as normalised numpy matrices in memory and persisted to disk. When a user speaks, **only the query** needs to be embedded — a single `ollama.embeddings()` call — and scoring is a batch matrix multiply (dot product) across all intent matrices simultaneously. This takes **~30–80 ms**.

### 3. Stage 1 fast-path: no LLM for confident queries
When the top cosine similarity score is ≥ `EMBED_HIGH` (0.80), the system accepts the intent immediately **without any LLM call**. For clear, unambiguous queries this completely eliminates the most expensive step. The total classify time in this path is roughly **50–100 ms** (just embedding + dot product).

### 4. TTS pre-warmed and disk-cached
At startup, `edge-tts` pre-generates audio for **all known response texts in parallel**, and saves them to a JSON cache file. At runtime, `speak()` is a cache lookup — no network call, no synthesis latency. This makes the TTS step essentially **0 ms** for all known intents.

### 5. Whisper JIT warmup
A silent 0.5-second WAV is transcribed during warmup to trigger Whisper's internal JIT compilation. Without this, the first real transcription would pay the JIT penalty (~1–2 seconds). After warmup, Whisper is fully compiled and ready.

### 6. LLM kept alive in memory
When the LLM (llama3.2) is used in Stage 2, it is called with `keep_alive: -1`, which tells Ollama to keep the model loaded in GPU/CPU memory indefinitely. This means there is no model-load latency between requests — the model is already resident.

### Typical timing breakdown (after warmup)

| Step | Stage 1 path | Stage 2 path |
|------|-------------|--------------|
| STT (Whisper) | ~80–200 ms | ~80–200 ms |
| Embed query | ~40–80 ms | ~40–80 ms |
| Cosine scoring | ~1 ms | ~1 ms |
| LLM inference | **0 ms** (skipped) | ~200–600 ms |
| TTS (cache hit) | ~1 ms | ~1 ms |
| **Total** | **~120–300 ms** | **~320–900 ms** |

Even in the worst case (Stage 2 LLM path), total end-to-end time stays under one second.

---

## 3-Stage Classification Pipeline

Located in `backend/services/classifier.py`, this is the core intelligence of the agent.

### Stage 1 — Embedding Similarity

Every intent has a set of example phrases (stored in `intents.json`). At startup, the label plus all examples are embedded into vectors using `mxbai-embed-large` (a high-quality 1024-dimensional embedding model). These vectors are L2-normalised and stacked into per-intent numpy matrices.

When classifying a new transcript:

1. The transcript is embedded into a query vector.
2. The query vector is L2-normalised.
3. For each intent, the dot product of the query against every example vector is computed (cosine similarity, since both are normalised).
4. The **TOP_K mean** is computed per intent (see below).
5. All intents are ranked by score.
6. The top score determines the Stage 1 outcome:

| Top score | Action |
|-----------|--------|
| ≥ `EMBED_HIGH` (0.80) | **Accept immediately** — return intent, skip LLM |
| < `EMBED_LOW` (0.55) | **Reject immediately** — return `human_handoff` |
| Between 0.55 and 0.80 | **Uncertain** — escalate to Stage 2 |

This thresholding design means the expensive LLM is only involved for genuinely ambiguous queries.

### TOP_K Averaging

```python
TOP_K = 3
```

Rather than taking the maximum similarity score or the mean of all example similarities, the classifier averages the **top K** similarities per intent. This is a deliberate design choice:

- **Why not max?** A single very similar example might be an outlier or a phrase the user happened to echo verbatim. Max scoring can be fragile.
- **Why not full mean?** Intents with many diverse examples can have their score diluted by weaker or noisier examples. A genuine match might rank lower than it should.
- **Why top-K mean?** It rewards intents that have multiple (at least K) strongly matching examples, while ignoring weaker ones. It is also resistant to outlier examples that are semantically far from the query.

The implementation uses `np.partition` instead of a full sort — this is O(n) rather than O(n log n), which matters when scoring many examples per intent at low latency.

```python
def _topk_score(sims: np.ndarray) -> float:
    k = min(TOP_K, len(sims))
    top_k = np.partition(sims, -k)[-k:]   # O(n) partial sort
    return float(np.mean(top_k))
```

### Stage 2 — LLM Disambiguation

When Stage 1 is uncertain, the top candidate intents are sent to `llama3.2` for disambiguation.

**Candidate selection:** Only intents whose score is within **0.10** of the top score are included, capped at `MAX_STAGE2_CANDIDATES` (4). This narrows the LLM's focus to genuinely competing intents rather than asking it to choose from all 8.

**Prompt design:** The LLM receives a focused system prompt listing only the candidate intents with their labels and up to 3 example phrases each. It is instructed to:
- Choose exactly one of the candidate keys (or `"unknown"`)
- Return a confidence score 0–100
- Respond with valid JSON only — no prose, no markdown

```json
{"intent": "transfer_delayed", "confidence": 88}
```

The LLM is run with `temperature: 0` (deterministic, no creativity) and `num_predict: 60` (output capped at 60 tokens — enough for a short JSON object, nothing more). Both settings reduce latency.

### Stage 3 — Final Decision & Fallback

After the LLM responds:

- If the returned intent is not in the known classifiable intents, fall back to `human_handoff`.
- If `llm_confidence < LLM_THRESHOLD` (75), fall back to `human_handoff`.
- Otherwise, accept the LLM's intent.

---

## Fallback Logic

The agent never crashes or gives a silent failure. At every decision point there is a defined fallback:

| Situation | Fallback | Reason |
|-----------|----------|--------|
| Stage 1 score < `EMBED_LOW` (0.55) | `human_handoff` | Query is too dissimilar to any known intent |
| LLM returns `"unknown"` | `human_handoff` | LLM itself isn't confident |
| LLM confidence < `LLM_THRESHOLD` (75) | `human_handoff` | Low-confidence LLM guess is worse than no guess |
| LLM returns an intent key not in the classifier's known set | `human_handoff` | Prevents hallucinated intent keys from being used |
| Embedding API error (network/Ollama down) | `human_handoff` with confidence 0.0 | Exception caught, safe fallback |
| LLM API error or malformed JSON | `human_handoff` with confidence 0.0 | Exception caught, safe fallback |
| TTS cache miss + network error during `speak()` | Exception propagates to `/voice` handler | Rare; edge-tts is covered by prewarm |

The `human_handoff` intent has a canned response:
> *"I'm connecting you with a human agent who can better assist you. Please hold."*

This gives the user a clear, graceful experience even when classification fails.

---

## Caching Systems

There are two independent disk caches, both stored in `backend/cache/`.

### Embedding Cache

**File:** `backend/cache/intent_embeddings_cache.json`

**What it stores:** For each intent, the raw embedding vectors (as JSON arrays of floats) for all its example phrases, keyed alongside an MD5 hash of the phrase list + model name.

**How cache validity is checked:** On startup, for each intent the classifier computes:
```
hash = MD5(embed_model_name + JSON(sorted(phrases)))
```
If the stored hash matches, the cached vectors are loaded. If the hash differs (phrases were edited, model changed) or the intent is new, embeddings are recomputed and the cache is updated.

**Why this matters:** Embedding ~80 phrases through Ollama takes 30–90 seconds on first run. With cache, startup takes a few milliseconds to load the JSON — the expensive computation only happens once.

### TTS Cache

**File:** `backend/cache/tts_cache.json`

**What it stores:** A mapping of `response_text → base64-encoded MP3 string` for every known intent response.

**How it works:** On startup, `prewarm_tts()` is called with the full list of response texts. Any response not already in the cache is synthesised in parallel via `asyncio.gather()` and saved. At runtime, `speak()` checks the in-memory dict first — if found, it returns the cached base64 immediately without any network call to edge-tts.

**Why this matters:** For a fixed-response bot (canned answers per intent), TTS synthesis is completely front-loaded. The runtime `speak()` call costs ~0 ms.

---

## Startup Warmup

`main.py` starts the backend and waits for it to be ready before opening the browser. The backend's `lifespan` startup hook runs three warmup steps sequentially:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_stt()           # 1. Load Whisper model + run silent transcription
    warmup()               # 2. Load/build embeddings + warm LLM
    await prewarm_tts(...)  # 3. Pre-generate TTS for all intents
    yield
```

**Step 1 — STT warmup (`warmup_stt`):**
- Loads the `faster-whisper base` model into memory.
- Creates a 0.5-second silent WAV in memory and transcribes it.
- This forces Python/CTranslate2 to JIT-compile the model execution graph.
- Without this, the first live transcription would be ~2x slower.

**Step 2 — Classifier warmup (`warmup`):**
- Ensures `mxbai-embed-large` and `llama3.2` are downloaded via Ollama (once only).
- Loads or rebuilds the embedding index from the disk cache.
- Sends a "hi" message to the LLM with `keep_alive: -1` to load it into memory.

**Step 3 — TTS prewarm (`prewarm_tts`):**
- Calls `edge-tts` for every uncached response text in parallel.
- All known responses are available as compiled audio before the first request arrives.

Total warmup time on a typical machine: **30–90 seconds on first run**, **5–15 seconds on subsequent runs** (models already downloaded, embeddings cached, TTS cached).

---

## Intents

Defined in `backend/data/intents.json`. Each intent has:
- `label` — human-readable name
- `examples` — list of representative phrases used for embedding similarity
- `response` — the canned text response the agent speaks

| Intent Key | Label |
|------------|-------|
| `transfer_location` | Where is my money? |
| `transfer_status_check` | How do I check my transfer status? |
| `transfer_arrival_time` | When will my money arrive? |
| `transfer_marked_complete` | Transfer marked complete but money not arrived |
| `transfer_delayed` | Transfer taking longer than estimated |
| `proof_of_payment` | What is a proof of payment? |
| `banking_reference_number` | What is a banking partner reference number? |
| `human_handoff` | Human handoff (fallback, no examples) |

To add a new intent: add an entry to `intents.json` with at least 4–8 diverse example phrases. Delete the embedding cache file so it is rebuilt on next startup.

---

## Recording Management

By default, every audio file received by `/voice` is saved to the `recordings/` directory with a timestamp filename (e.g. `20260328_143022.webm`). This is useful for debugging transcription accuracy.

To delete recordings after transcription (save disk space):

```
python main.py --delete-recordings
```

The `KEEP_RECORDINGS` environment variable is passed from `main.py` to the backend process and read in `backend/app.py`.

---

## Frontend

`frontend/index.html` is a single-file vanilla HTML/JS UI with no build step or dependencies.

**UI elements:**
- A large microphone button (🎤) — click to start recording, click again to stop.
- A status line with a spinner animation during processing.
- A debug panel that appears after each response showing:
  - The recognised transcript
  - The classified intent name
  - Confidence score with a visual progress bar
  - The text response
  - Per-step timings (STT / Classify / TTS / Total in ms)

**Recording:** Uses the browser's `MediaRecorder` API to record audio as WebM. When the user stops recording, the blob is POSTed to `http://localhost:8000/voice`.

**Streaming:** The response from `/voice` is a Server-Sent Events stream. The frontend reads it with a `ReadableStream` reader and updates the UI incrementally as each event arrives. Audio playback begins immediately when the `done` event arrives.

**Mic permission:** If the browser denies microphone access (e.g. user blocks it), the status shows "Microphone access denied." — no silent failures.

---

## Project Structure

```
voice_agent/
│
├── main.py                      # Entry point — Ollama setup, launches backend + frontend
├── requirements.txt             # Python dependencies
│
├── backend/
│   ├── app.py                   # FastAPI app, /voice endpoint, SSE pipeline
│   ├── __init__.py
│   │
│   ├── data/
│   │   └── intents.json         # Intent definitions: labels, examples, responses
│   │
│   ├── cache/                   # Auto-created at runtime
│   │   ├── intent_embeddings_cache.json   # Cached embedding vectors
│   │   └── tts_cache.json                 # Cached TTS audio (base64 MP3)
│   │
│   └── services/
│       ├── classifier.py        # 3-stage intent classification engine
│       ├── intents.py           # Loads intents.json, provides get_response()
│       ├── stt.py               # faster-whisper wrapper + warmup
│       ├── tts.py               # edge-tts wrapper + cache
│       └── __init__.py
│
├── frontend/
│   └── index.html               # Single-file UI (no build step)
│
└── recordings/                  # Auto-created — saved audio files (optional)
```

---

## Requirements & Prerequisites

### System Requirements

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Required for modern type hint syntax used in the code |
| Windows / macOS / Linux | All platforms supported |
| Microphone | Required for voice input |
| ~4 GB RAM minimum | For Whisper base + llama3.2 in memory |
| ~6 GB disk space | For Ollama models (llama3.2 ~2 GB, mxbai-embed-large ~0.7 GB) |

### Ollama (Required)

> **Ollama must be installed before running this project.** It provides two models used by the classifier.

Download and install Ollama from **[https://ollama.com/download](https://ollama.com/download)**.

`main.py` will check if Ollama is installed and running. If Ollama is installed but not running, it will start it automatically. If it is not installed, the launcher will exit with a message directing you to the download page.

The two models used are:

| Model | Purpose | Size |
|-------|---------|------|
| `llama3.2` | LLM disambiguation (Stage 2 classifier) | ~2.0 GB |
| `mxbai-embed-large` | Embedding similarity (Stage 1 classifier) | ~0.7 GB |

Both models are pulled automatically on first run if not already present. You do not need to run `ollama pull` manually.

### Python Dependencies

```
fastapi           — Web framework for the backend API
uvicorn[standard] — ASGI server to serve FastAPI
python-multipart  — Required for FastAPI file upload parsing
faster-whisper    — Optimised Whisper STT (CTranslate2-based)
edge-tts          — Microsoft Edge TTS (free, no API key needed)
ollama            — Python client for Ollama (embeddings + LLM chat)
numpy             — Efficient vector math for cosine similarity
```

---

## Installation

```bash
# 1. Clone or download the project
cd voice_agent

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install Ollama (if not already installed)
# https://ollama.com/download
# Ollama models are pulled automatically on first run.
```

---

## Running the Agent

```bash
python main.py
```

This will:

1. Check if Ollama is installed; exit with an error if not.
2. Start the Ollama server if it isn't running.
3. Pull `llama3.2` and `mxbai-embed-large` if not already downloaded (first run only).
4. Start the FastAPI backend on port 8000 in a background thread.
5. Start a Python HTTP server for the frontend on port 3000 in a background thread.
6. Wait for the backend warmup to complete (STT, embeddings, TTS).
7. Open `http://localhost:3000` in your default browser.

**To delete recordings after transcription:**
```bash
python main.py --delete-recordings
```

**To stop:** Press `Ctrl+C`.

---

## Configuration Reference

All tunable constants are in `backend/services/classifier.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `"mxbai-embed-large"` | Ollama embedding model for Stage 1 |
| `LLM_MODEL` | `"llama3.2"` | Ollama LLM for Stage 2 disambiguation |
| `EMBED_HIGH` | `0.80` | Cosine threshold: above this, Stage 1 accepts immediately (no LLM) |
| `EMBED_LOW` | `0.55` | Cosine threshold: below this, Stage 1 rejects immediately (human handoff) |
| `LLM_THRESHOLD` | `75` | LLM confidence % must exceed this to accept its answer; below → handoff |
| `TOP_K` | `3` | Number of top example similarities to average per intent |
| `MAX_STAGE2_CANDIDATES` | `4` | Max number of intents passed to LLM for disambiguation |

In `backend/services/tts.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `VOICE` | `"en-US-JennyNeural"` | Edge TTS voice name |

In `main.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `"llama3.2"` | The LLM model to check/pull (must match `LLM_MODEL` in classifier) |
| `OLLAMA_URL` | `"http://localhost:11434"` | Ollama API base URL |

**Tuning tips:**
- Raise `EMBED_HIGH` (e.g. to 0.85) if the agent is accepting weak matches too readily.
- Lower `EMBED_LOW` (e.g. to 0.50) if the agent is routing too many valid queries to handoff.
- Raise `TOP_K` (e.g. to 5) if your intents have many (8+) examples and you want more averaging.
- Raise `LLM_THRESHOLD` (e.g. to 80) if the LLM is making low-confidence guesses that turn out to be wrong.
- Lower `MAX_STAGE2_CANDIDATES` (e.g. to 3) to give the LLM a tighter, more focused prompt.

---

## Timing Breakdown

The backend logs per-step timings to the terminal on every request:

```
[Pipeline] Started at 14:30:22.413
[STT] Transcript: When will my money arrive?
[Timer] STT transcription: 142.3 ms
[Classifier] Embed scores: {'transfer_arrival_time': 0.847, 'transfer_delayed': 0.712, ...}
[Classifier] Stage 1 accept → transfer_arrival_time (0.847)
[Timer] Classification: 71.4 ms
[Timer] TTS synthesis: 0.8 ms
[Timer] Total pipeline: 215.1 ms  (STT 142.3 + CLF 71.4 + TTS 0.8)
```

The frontend debug panel mirrors these timings in the UI after each response.
