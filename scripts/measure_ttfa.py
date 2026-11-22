import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tts.factory import create_tts_engine

S = "Good morning. My name is Vishnu. Thank you very much for your help."
CASES = [
    ("piper-lessac", create_tts_engine("piper", voice_dir="assets/voices/en/en_US/lessac/medium")),
    ("piper-amy", create_tts_engine("piper", voice="en_US-amy-medium",
                                    voice_dir="assets/voices/en/en_US/amy/medium")),
    ("kokoro", create_tts_engine("kokoro")),
]
for name, eng in CASES:
    list(eng.stream_chunks("warmup."))
    t0 = time.time()
    n = 0
    first = None
    for chunk in eng.stream_chunks(S):
        if first is None:
            first = time.time() - t0
        n += 1
    total = time.time() - t0
    print(f"{name}: TTFA={first:.2f}s chunks={n} full={total:.2f}s", flush=True)
