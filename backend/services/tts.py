import asyncio
import base64
import io
import json
import os
import time
import edge_tts

VOICE = "en-US-JennyNeural"

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "tts_cache.json")

# Load disk cache on startup
_tts_cache: dict[str, str] = {}
if os.path.exists(_CACHE_PATH):
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as _f:
            _tts_cache = json.load(_f)
        print(f"[TTS] Loaded {len(_tts_cache)} cached responses from disk.")
    except Exception as e:
        print(f"[TTS] Cache unreadable, starting fresh. ({e})")


def _save_cache() -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_tts_cache, f)
    except Exception as e:
        print(f"[TTS] Warning: could not save cache: {e}")


async def _generate(text: str) -> str:
    """Make a live edge-tts network call and return base64-encoded MP3."""
    communicate = edge_tts.Communicate(text, VOICE)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return base64.b64encode(buf.getvalue()).decode()


async def prewarm_tts(responses: list[str]) -> None:
    """Pre-generate TTS audio for all known responses so runtime hits are instant."""
    uncached = [t for t in responses if t not in _tts_cache]
    if not uncached:
        print(f"[TTS] All {len(responses)} responses already cached.")
        return
    print(f"[TTS] Pre-warming {len(uncached)} responses in parallel...")
    t0 = time.perf_counter()
    results = await asyncio.gather(*[_generate(t) for t in uncached], return_exceptions=True)
    cache_dirty = False
    for text, result in zip(uncached, results):
        if isinstance(result, Exception):
            print(f"[TTS] Prewarm failed for '{text[:40]}': {result}")
        else:
            _tts_cache[text] = result
            cache_dirty = True
    if cache_dirty:
        _save_cache()
    print(f"[Timer] TTS prewarm total: {(time.perf_counter() - t0)*1000:.1f} ms ({len(_tts_cache)} cached)")


async def speak(text: str) -> str:
    """Return base64-encoded MP3, served from cache when available."""
    if text in _tts_cache:
        print("[TTS] Cache hit — skipping network call")
        return _tts_cache[text]
    t0 = time.perf_counter()
    result = await _generate(text)
    print(f"[Timer] edge-tts stream (uncached): {(time.perf_counter() - t0)*1000:.1f} ms")
    _tts_cache[text] = result
    _save_cache()
    return result
