import json
import hashlib
import os
import time
import numpy as np
import ollama
from backend.services.intents import INTENTS, HUMAN_HANDOFF

# Path to the embeddings cache file — stored in <project_root>/cache/
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "intent_embeddings_cache.json")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2"

# ---------------------------------------------------------------------------
# Thresholds
#   EMBED_HIGH            – cosine score above this → accept immediately, skip LLM
#   EMBED_LOW             – cosine score below this → reject immediately (human handoff)
#   LLM_THRESHOLD         – LLM confidence (0-100) must exceed this to accept
#   TOP_K                 – number of top example similarities to average per intent
#   MAX_STAGE2_CANDIDATES – hard cap on intents sent to the LLM in Stage 2
# ---------------------------------------------------------------------------
EMBED_HIGH = 0.8
EMBED_LOW = 0.55
LLM_THRESHOLD = 75
TOP_K = 3
MAX_STAGE2_CANDIDATES = 4

_LLM_OPTIONS = {
    "temperature": 0,
    "num_predict": 60,
}

# Intent keys excluding the handoff sentinel
_CLASSIFIABLE_INTENTS = {k: v for k, v in INTENTS.items() if k != HUMAN_HANDOFF}

# ---------------------------------------------------------------------------
# Embedding index  (populated in warmup)
# ---------------------------------------------------------------------------
# Maps intent_key → pre-normalised float32 matrix of shape (n_examples, dim)
_intent_embeddings: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------------------
# LLM prompt  (dynamically built so it only shows the candidate intents)
# ---------------------------------------------------------------------------
def _build_llm_prompt(candidates: list[str]) -> str:
    candidate_lines = "\n".join(
        f'- "{key}": {INTENTS[key]["label"]}\n'
        f'  Examples: {" | ".join(INTENTS[key]["examples"][:3])}'
        for key in candidates
    )
    keys_str = ", ".join(f'"{k}"' for k in candidates)
    return f"""You are a precise intent classifier for a financial transfer support voice agent.

A customer has spoken to the agent. Your job is to identify EXACTLY which support topic they are asking about.

## Candidate intents
{candidate_lines}

## Rules
- Choose ONLY from the candidate keys listed above.
- If the message is ambiguous or does not clearly match any candidate, use "unknown".
- "confidence" must be an integer 0-100 reflecting how certain you are.
- Reply with VALID JSON only — no prose, no markdown fences.

## Output format
{{"intent": <one of: {keys_str}, "unknown">, "confidence": <0-100>}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _embed(text: str) -> list[float]:
    """Return raw embedding from ollama (list[float], JSON-serialisable)."""
    return ollama.embeddings(model=EMBED_MODEL, prompt=text, keep_alive=-1)["embedding"]


def _normalize_stack(vecs: list[list[float]]) -> np.ndarray:
    """Stack vectors into a (n, dim) float32 matrix with each row L2-normalised."""
    mat = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def _ensure_model_pulled(model: str) -> None:
    """Pull the model if it is not already available locally."""
    local_models = {m.model.split(":")[0] for m in ollama.list().models}
    tag_less = model.split(":")[0]
    if tag_less not in local_models:
        print(f"[Classifier] Pulling '{model}' — this only happens once...")
        for chunk in ollama.pull(model, stream=True):
            status = getattr(chunk, "status", "")
            total = getattr(chunk, "total", None)
            completed = getattr(chunk, "completed", None)
            if total and completed:
                pct = int(completed / total * 100)
                print(f"\r[Classifier] {model}: {pct}%", end="", flush=True)
            elif status:
                print(f"[Classifier] {model}: {status}")
        print(f"\n[Classifier] '{model}' ready.")
    else:
        print(f"[Classifier] '{model}' already available.")


def warmup() -> None:
    """Pull models if needed, load/build intent embeddings from cache, warm the LLM."""
    # ── 0. Ensure cache directory exists ──
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)

    # ── 1. Ensure both models exist locally ──
    _ensure_model_pulled(EMBED_MODEL)
    _ensure_model_pulled(LLM_MODEL)

    # ── 2. Load cache (if it exists) ──
    cache: dict[str, dict] = {}  # { intent_key: { "hash": str, "vectors": list[list[float]] } }
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"[Classifier] Loaded embedding cache ({len(cache)} intents cached).")
        except Exception as e:
            print(f"[Classifier] Cache unreadable, rebuilding. ({e})")
            cache = {}

    # ── 3. For each intent, check if cached embeddings are still valid ──
    cache_dirty = False
    for key, meta in _CLASSIFIABLE_INTENTS.items():
        phrases = [meta["label"]] + meta.get("examples", [])
        phrase_hash = hashlib.md5((EMBED_MODEL + json.dumps(phrases, sort_keys=True)).encode()).hexdigest()

        if key in cache and cache[key].get("hash") == phrase_hash:
            # Cache hit — reuse stored vectors (convert to normalised numpy matrix)
            _intent_embeddings[key] = _normalize_stack(cache[key]["vectors"])
            print(f"[Classifier] Cache hit  → {key}")
        else:
            # Cache miss (new intent or phrases changed) — (re)compute
            print(f"[Classifier] Cache miss → computing embeddings for '{key}'...")
            raw_vecs = [_embed(p) for p in phrases]
            _intent_embeddings[key] = _normalize_stack(raw_vecs)
            cache[key] = {"hash": phrase_hash, "vectors": raw_vecs}  # store raw lists for JSON
            cache_dirty = True

    # ── 4. Persist updated cache if anything changed ──
    if cache_dirty:
        try:
            with open(_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            print(f"[Classifier] Embedding cache saved to {_CACHE_PATH}")
        except Exception as e:
            print(f"[Classifier] Warning: could not save cache: {e}")

    print(f"[Classifier] Index ready ({len(_intent_embeddings)} intents).")

    # ── 5. Warm the LLM so first real inference is instant ──
    print("[Classifier] Warming up LLM...")
    ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        options={**_LLM_OPTIONS, "keep_alive": -1},
    )
    print("[Classifier] All models warm.")


# ---------------------------------------------------------------------------
# Classification pipeline
# ---------------------------------------------------------------------------
def classify(transcript: str) -> tuple[str, float]:
    """
    3-stage pipeline:

    Stage 1 – Embedding similarity
        Embed the transcript and score it against every intent's example bank.
        • top score ≥ EMBED_HIGH  → accept immediately (no LLM call)
        • top score <  EMBED_LOW  → human handoff immediately
        • otherwise               → pass top candidates to Stage 2

    Stage 2 – LLM disambiguation (only when Stage 1 is uncertain)
        Pass the transcript and the top-scoring candidate intents to the LLM
        with a focused, few-shot prompt.

    Stage 3 – Final decision
        Accept the LLM result if its confidence ≥ LLM_THRESHOLD, else handoff.
    """
    # ── Stage 1: Embedding similarity ──
    try:
        t0 = time.perf_counter()
        raw_vec = _embed(transcript)
        print(f"[Timer] Embed query: {(time.perf_counter() - t0)*1000:.1f} ms")
    except Exception as e:
        print(f"[Classifier] Embedding error: {e}")
        return HUMAN_HANDOFF, 0.0

    # Normalise query vector once, then score all intents via a single dot product per intent.
    # Top-k mean: average the TOP_K highest cosine similarities for each intent so that
    # weaker/noisier examples don't drag the score down for genuine matches.
    t0 = time.perf_counter()
    query_arr = np.array(raw_vec, dtype=np.float32)
    query_arr /= np.linalg.norm(query_arr) + 1e-10

    def _topk_score(sims: np.ndarray) -> float:
        k = min(TOP_K, len(sims))
        top_k = np.partition(sims, -k)[-k:]  # O(n) — no need to fully sort
        return float(np.mean(top_k))

    scores: dict[str, float] = {
        key: _topk_score(mat @ query_arr)
        for key, mat in _intent_embeddings.items()
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_intent, top_score = ranked[0]
    print(f"[Timer] Cosine scoring all intents: {(time.perf_counter() - t0)*1000:.1f} ms")

    print(f"[Classifier] Embed scores: { {k: round(v,3) for k,v in ranked} }")

    if top_score >= EMBED_HIGH:
        print(f"[Classifier] Stage 1 accept → {top_intent} ({top_score:.3f})")
        return top_intent, top_score

    if top_score < EMBED_LOW:
        print(f"[Classifier] Stage 1 reject → handoff ({top_score:.3f})")
        return HUMAN_HANDOFF, top_score

    # ── Stage 2: LLM disambiguation ──
    # Send only the intents whose score is within 0.10 of the top score, capped at MAX_STAGE2_CANDIDATES
    candidates = [k for k, s in ranked if top_score - s <= 0.10][:MAX_STAGE2_CANDIDATES]
    print(f"[Classifier] Stage 2 LLM disambiguation on: {candidates}")

    system_prompt = _build_llm_prompt(candidates)
    try:
        t_llm = time.perf_counter()
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            options={**_LLM_OPTIONS, "keep_alive": -1},
        )
        print(f"[Timer] LLM inference: {(time.perf_counter() - t_llm)*1000:.1f} ms")
        raw = response.message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        data = json.loads(raw)
        intent = data.get("intent", "unknown")
        llm_confidence = float(data.get("confidence", 0))

        print(f"[Classifier] LLM → {intent} ({llm_confidence:.0f}/100)")

        # ── Stage 3: Final decision ──
        if intent not in _CLASSIFIABLE_INTENTS or llm_confidence < LLM_THRESHOLD:
            return HUMAN_HANDOFF, llm_confidence / 100

        return intent, llm_confidence / 100

    except Exception as e:
        print(f"[Classifier] LLM error: {e}")
        return HUMAN_HANDOFF, 0.0
