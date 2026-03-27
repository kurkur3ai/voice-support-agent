import os
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from backend.services.stt import transcribe, warmup as warmup_stt
from backend.services.classifier import classify, warmup
from backend.services.tts import speak, prewarm_tts
from backend.services.intents import get_response, INTENTS


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_stt()
    warmup()
    await prewarm_tts([get_response(k) for k in INTENTS])
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "recordings"))
os.makedirs(SAVE_DIR, exist_ok=True)

KEEP_RECORDINGS: bool = os.environ.get("KEEP_RECORDINGS", "1") == "1"


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def voice_pipeline(path: str):
    pipeline_start = time.perf_counter()
    print(f"[Pipeline] Started at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

    yield sse({"status": "transcribing"})
    t0 = time.perf_counter()
    transcript = transcribe(path)
    stt_ms = (time.perf_counter() - t0) * 1000
    print(f"[STT] Transcript: {transcript}")
    print(f"[Timer] STT transcription: {stt_ms:.1f} ms")

    yield sse({"status": "classifying", "transcript": transcript})
    t0 = time.perf_counter()
    intent, confidence = classify(transcript)
    clf_ms = (time.perf_counter() - t0) * 1000
    print(f"[Classifier] Intent: {intent} ({confidence:.2f})")
    print(f"[Timer] Classification: {clf_ms:.1f} ms")

    response_text = get_response(intent)
    yield sse({"status": "generating_audio", "intent": intent,
               "confidence": confidence, "response_text": response_text})

    t0 = time.perf_counter()
    audio_b64 = await speak(response_text)
    tts_ms = (time.perf_counter() - t0) * 1000
    print(f"[Timer] TTS synthesis: {tts_ms:.1f} ms")

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    print(f"[Timer] Total pipeline: {total_ms:.1f} ms  (STT {stt_ms:.1f} + CLF {clf_ms:.1f} + TTS {tts_ms:.1f})")

    yield sse({"status": "done", "transcript": transcript, "intent": intent,
               "confidence": confidence, "response_text": response_text,
               "audio_b64": audio_b64,
               "timings": {"stt_ms": round(stt_ms), "clf_ms": round(clf_ms),
                           "tts_ms": round(tts_ms), "total_ms": round(total_ms)}})


@app.post("/voice")
async def voice(file: UploadFile = File(...)):
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
    path = os.path.join(SAVE_DIR, filename)

    with open(path, "wb") as f:
        f.write(await file.read())

    async def stream():
        async for chunk in voice_pipeline(path):
            yield chunk
        if not KEEP_RECORDINGS:
            os.remove(path)

    return StreamingResponse(stream(), media_type="text/event-stream")
