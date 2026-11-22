#!/usr/bin/env python
"""Run the real-time sign language translation demo."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Sign Language Translation Demo")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None, help="Override config (e.g. configs/isl.yaml)")
    parser.add_argument("--checkpoint", default="models/multihead_v2/best.pt")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS output")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode (use cached voice, no network)")
    args = parser.parse_args()

    if args.offline:
        os.environ["PIPER_OFFLINE"] = "1"
        os.environ["PIPER_DATA_DIR"] = str(Path("demo/piper_voice").resolve())

    from src.app.demo import run_demo
    run_demo(args.config, args.override, args.checkpoint, no_tts=args.no_tts, offline=args.offline)


if __name__ == "__main__":
    main()
