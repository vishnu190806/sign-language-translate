import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf

from src.tts.factory import create_tts_engine

S = "Good morning. My name is Vishnu. Thank you very much for your help."
CASES = [
    ("en_US-amy-medium", "assets/voices/en/en_US/amy/medium", "results/piper_amy_sample.wav"),
    ("en_US-libritts_r-medium", "assets/voices/en/en_US/libritts_r/medium",
     "results/piper_libritts_sample.wav"),
]
for voice, vdir, out in CASES:
    eng = create_tts_engine("piper", voice=voice, voice_dir=vdir)
    eng.synthesize("warmup.")
    t0 = time.time()
    audio, sr = eng.synthesize(S)
    dt = time.time() - t0
    sf.write(out, audio, sr)
    print(f"{voice}: {len(audio) / sr:.1f}s audio in {dt:.2f}s -> {out}", flush=True)
