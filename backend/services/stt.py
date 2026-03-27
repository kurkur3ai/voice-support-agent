import io
import time
import wave
import tempfile
from faster_whisper import WhisperModel

_model = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        print("[STT] Loading faster-whisper model (base) on cpu (int8)...")
        _model = WhisperModel("base", device="cpu", compute_type="int8")
        print("[STT] Model ready.")
    return _model


def warmup() -> None:
    """Run a silent dummy transcription to force Whisper's internal JIT init."""
    # Build a minimal silent WAV (0.5 s, 16 kHz, mono) entirely in memory
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)          # 16-bit
        w.setframerate(16000)
        w.writeframes(b'\x00\x00' * 8000)  # 0.5 s of silence
    buf.seek(0)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(buf.read())
        tmp_path = tmp.name
    t0 = time.perf_counter()
    model = get_model()
    list(model.transcribe(tmp_path, beam_size=1, language="en")[0])  # consume generator
    import os; os.unlink(tmp_path)
    print(f"[STT] Warmup done ({(time.perf_counter() - t0)*1000:.0f} ms)")


def transcribe(audio_path: str) -> str:
    model = get_model()
    t0 = time.perf_counter()
    segments, _ = model.transcribe(audio_path, beam_size=5, language="en")
    result = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"[Timer] Whisper inference: {(time.perf_counter() - t0)*1000:.1f} ms")
    return result
