#!/usr/bin/env python
"""Extract MediaPipe landmarks from ISL-CSLTR sentence videos.

Sentence-disjoint splits: ~15 held-out sentences -> test, rest train/val.
Writes data/processed/isl-csltr/landmarks/*.npz {features, glosses, sentence, split}.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEST_SENTENCES = {
    "are you hiding something", "bring water for me", "congratulations",
    "could you please talk slower", "do you need help", "how old are you",
    "i love you", "see you tomorrow", "take care", "thank you very much",
    "what is your name", "where do you live", "happy birthday", "good morning",
    "nice to meet you",
}


def split_of(sentence: str) -> str:
    return "test" if sentence.lower() in TEST_SENTENCES else "train"


def main() -> None:
    from src.features.extractor import OfflineExtractor

    inv = json.loads(Path("data/processed/isl-csltr/inventory.json").read_text())
    out = Path("data/processed/isl-csltr/landmarks")
    out_word = Path("data/processed/isl-csltr/word_landmarks")
    out.mkdir(parents=True, exist_ok=True)
    out_word.mkdir(parents=True, exist_ok=True)
    sents = [i for i in inv["items"] if i["kind"] == "sentence" and i["glosses"]]
    logger.info(f"Sentence videos with glosses: {len(sents)}")

    ext = OfflineExtractor()
    ext_extract = lambda p: ext.extract_from_video(p, target_fps=30)
    ok = fail = skip = 0
    for n, item in enumerate(sents):
        stem = Path(item["file"]).stem.replace(" ", "_").replace("(", "").replace(")", "")
        npz = out / f"{stem}.npz"
        if npz.exists():
            skip += 1
            continue
        try:
            feats = ext.extract_from_video(item["file"], target_fps=30)
            if feats is None or len(feats) == 0:
                fail += 1
                continue
            split = "test" if item["sentence"].lower() in TEST_SENTENCES else "train"
            np.savez_compressed(npz, features=np.asarray(feats, dtype=np.float32),
                                glosses=np.array(item["glosses"]), sentence=item["sentence"], split=split)
            ok += 1
        except Exception as e:
            logger.warning(f"Failed {item['file']}: {e}")
            fail += 1
        if (n + 1) % 50 == 0:
            logger.info(f"Progress {n + 1}/{len(sents)} ok={ok} fail={fail} skip={skip}")
    try:
        pass
    except Exception:
        pass
    logger.info(f"Done sentences: ok={ok} fail={fail} skip={skip}")

    words = [i for i in inv["items"] if i["kind"] == "word"]
    logger.info(f"Word clips: {len(words)}")
    ok = fail = skip = 0
    for n, item in enumerate(words):
        stem = Path(item["file"]).stem.replace(" ", "_").replace("(", "").replace(")", "")
        npz = out_word / f"{stem}.npz"
        if npz.exists():
            skip += 1
            continue
        try:
            feats = ext_extract(item["file"])
            if feats is None or len(feats) == 0 or not np.any(feats):
                fail += 1
                continue
            np.savez_compressed(npz, features=np.asarray(feats, dtype=np.float32),
                                label=item["word"], split=split_of(item["sentence"]),
                                dataset="csltr-word", signer="csltr")
            ok += 1
        except Exception as e:
            logger.warning(f"Failed {item['file']}: {e}")
            fail += 1
        if (n + 1) % 100 == 0:
            logger.info(f"Word progress {n + 1}/{len(words)} ok={ok} fail={fail} skip={skip}")
    logger.info(f"Done words: ok={ok} fail={fail} skip={skip}")
    try:
        ext.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
