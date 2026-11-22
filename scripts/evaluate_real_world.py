#!/usr/bin/env python
"""Two-phase evaluation: cache landmarks first, then evaluate from cache."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.config import load_config
from src.features.extractor import OfflineExtractor
from src.features.normalize import build_frame_features
from src.features.smooth import LandmarkSmoother
from src.infer.buffer import RingBuffer
from src.infer.gate import StabilityGate
from src.models.builder import build_model

CACHE_DIR = Path("isl_40_test/.landmark_cache")


def load_label_map(path: str | Path) -> dict[int, str]:
    with open(path) as f:
        raw = json.load(f)
    return {int(v): k for k, v in raw.items()}


SYNONYMS = {
    "alright": "okay",
    "okay": "okay",
    "thank you": "thank you",
    "thankyou": "thank you",
    "big large": "big",
    "small little": "small",
    "store or shop": "store",
    "race (ethnicity)": "race",
    "you (plural)": "you plural",
}


def normalize_label(label: str) -> str:
    if "." in label:
        label = label.split(".")[-1].strip()
    label = label.lower().strip()
    return SYNONYMS.get(label, label)


def build_model_vocabulary(label_map: dict[int, str]) -> set[str]:
    return {normalize_label(name) for name in label_map.values()}


def labels_match(predicted: str, ground_truth: str) -> bool:
    pred = normalize_label(predicted)
    gt = normalize_label(ground_truth)
    return pred == gt


def cache_key(video_path: Path) -> str:
    stat = video_path.stat()
    raw = f"{video_path}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def extract_and_cache(video_path: Path, cache_dir: Path, smoother: LandmarkSmoother) -> list[np.ndarray]:
    key = cache_key(video_path)
    cached = cache_dir / f"{key}.npz"
    if cached.exists():
        data = np.load(cached)
        return [data[f"frame_{i}"] for i in range(data["n_frames"])]

    ext = OfflineExtractor(
        hand_model_path="assets/hand_landmarker.task",
        pose_model_path="assets/pose_landmarker.lite.task",
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        ext.close()
        return []

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts_ms = frame_idx * 33
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = ext._hand_landmarker.detect_for_video(mp_image, ts_ms)
        pose_result = ext._pose_landmarker.detect_for_video(mp_image, ts_ms)

        left_hand = right_hand = None
        if hand_result.hand_landmarks:
            for i, hand_lms in enumerate(hand_result.hand_landmarks):
                handedness = hand_result.handedness[i][0].category_name
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
                if handedness == "Left":
                    left_hand = coords
                else:
                    right_hand = coords

        pose_coords = None
        if pose_result.pose_landmarks:
            all_pose = pose_result.pose_landmarks[0]
            pose_coords = np.array([[lm.x, lm.y, lm.z] for lm in all_pose], dtype=np.float32)

        features = build_frame_features(left_hand, right_hand, pose_coords, ext.spec)
        if smoother:
            features = smoother.smooth(features)
        frames.append(features)
        frame_idx += 1

    cap.release()
    ext.close()

    if frames:
        save_dict = {"n_frames": len(frames)}
        for i, f in enumerate(frames):
            save_dict[f"frame_{i}"] = f
        np.savez_compressed(cached, **save_dict)

    return frames


def run_inference(frames: list[np.ndarray], model, ring_buffer: RingBuffer, gate: StabilityGate,
                  label_map: dict[int, str], config, device) -> dict:
    ring_buffer.reset()
    gate.reset()

    all_probs = []
    emitted_words = []
    INFERENCE_STRIDE = 2

    for i, features in enumerate(frames):
        ring_buffer.add(features)
        if ring_buffer.is_full and i % INFERENCE_STRIDE == 0:
            seq, mask = ring_buffer.get_sequence()
            seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(seq_tensor, mask_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            emitted_idx = gate.update(probs)
            if emitted_idx is not None and emitted_idx in label_map:
                emitted_words.append(label_map[emitted_idx])

    if not all_probs:
        return {"top1": "", "top1_conf": 0, "top3": [], "top5": [], "emitted": emitted_words}

    avg_probs = np.mean(all_probs, axis=0)
    top5_idx = np.argsort(avg_probs)[-5:][::-1]
    top5 = [(label_map.get(int(i), f"class_{i}"), float(avg_probs[i])) for i in top5_idx]
    top3 = top5[:3]

    return {
        "top1": top3[0][0],
        "top1_conf": top3[0][1],
        "top3": top3,
        "top5": top5,
        "emitted": emitted_words,
    }


def main():
    dataset_dir = Path("isl_40_test")
    metadata_path = dataset_dir / "metadata.csv"

    if not metadata_path.exists():
        print(f"Metadata not found at {metadata_path}")
        sys.exit(1)

    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        metadata = list(reader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("configs/default.yaml", "configs/isl.yaml")

    label_map = load_label_map("models/label_map.json")
    model_vocab = build_model_vocabulary(label_map)

    print(f"Model vocabulary: {len(model_vocab)} labels")

    ckpt = torch.load("models/best.pt", map_location=device, weights_only=False)
    ckpt_model = ckpt.get("config", {}).get("model", {})
    if ckpt_model.get("num_classes"):
        config.model.num_classes = ckpt_model["num_classes"]

    model = build_model(config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())} params, {config.model.num_classes} classes")

    smoother = LandmarkSmoother(feature_dim=config.model.input_dim)
    ring_buffer = RingBuffer(config.processing.sequence_length, config.model.input_dim)
    gate = StabilityGate(
        threshold=config.processing.stability_gate.threshold,
        consecutive_frames=config.processing.stability_gate.consecutive_frames,
        cooldown_frames=config.processing.stability_gate.cooldown_frames,
        agreement_ratio=config.processing.stability_gate.agreement_ratio,
        min_confidence_delta=config.processing.stability_gate.min_confidence_delta,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    videos = []
    for row in metadata:
        video_path = dataset_dir / row["video_path"]
        if video_path.exists():
            videos.append((video_path, row))

    iv_words = sorted({normalize_label(row.get("normalized_word", row.get("word", ""))) for row in metadata
                       if normalize_label(row.get("normalized_word", row.get("word", ""))) in model_vocab})
    oov_words = sorted({normalize_label(row.get("normalized_word", row.get("word", ""))) for row in metadata
                        if normalize_label(row.get("normalized_word", row.get("word", ""))) not in model_vocab})

    print(f"\nVocabulary overlap: {len(iv_words)} in-vocabulary, {len(oov_words)} out-of-vocabulary")
    print(f"In-vocabulary words: {iv_words}")
    print(f"Out-of-vocabulary words: {oov_words}")

    cached_count = sum(1 for vp, _ in videos if (CACHE_DIR / f"{cache_key(vp)}.npz").exists())
    print(f"\nLandmark cache: {cached_count}/{len(videos)} already cached")

    results_by_source = defaultdict(lambda: {"correct_top1": 0, "correct_top3": 0, "correct_top5": 0,
                                              "emitted_match": 0, "total": 0, "no_emission": 0, "predictions": []})
    iv_results = {"total": 0, "correct_top1": 0, "correct_top3": 0, "correct_top5": 0}
    oov_results = {"total": 0, "correct_top1": 0, "correct_top3": 0, "correct_top5": 0}

    print(f"\nEvaluating {len(videos)} videos from {len(set(r['dataset'] for _, r in videos))} sources\n")
    print(f"{'Source':<20} {'Word':<12} {'IV/OOV':<6} {'Top-1':<25} {'Conf':>5} {'T1':>3} {'T3':>3} {'T5':>3} {'Emit':>5}")
    print("-" * 110)

    t_start = time.time()
    for i, (video_path, row) in enumerate(videos):
        source = row.get("dataset", "unknown")
        word = row.get("normalized_word", row.get("word", ""))
        word_norm = normalize_label(word)
        is_iv = word_norm in model_vocab

        frames = extract_and_cache(video_path, CACHE_DIR, smoother)

        if not frames:
            results_by_source[source]["total"] += 1
            results_by_source[source]["predictions"].append({"word": word, "iv": is_iv, "match": False, "emitted": False})
            print(f"{source:<20} {word:<12} {'IV' if is_iv else 'OOV':<6} {'NO LANDMARKS':<25}")
            continue

        result = run_inference(frames, model, ring_buffer, gate, label_map, config, device)

        top1_match = labels_match(result["top1"], word) if result["top1"] else False
        top3_match = any(labels_match(p[0], word) for p in result["top3"])
        top5_match = any(labels_match(p[0], word) for p in result["top5"])
        emitted_match = any(labels_match(e, word) for e in result["emitted"])

        r = results_by_source[source]
        r["total"] += 1
        if top1_match: r["correct_top1"] += 1
        if top3_match: r["correct_top3"] += 1
        if top5_match: r["correct_top5"] += 1
        if emitted_match: r["emitted_match"] += 1
        if not result["emitted"]: r["no_emission"] += 1
        r["predictions"].append({
            "word": word, "iv": is_iv, "top1": result["top1"], "top1_conf": result["top1_conf"],
            "top3": [p[0] for p in result["top3"]], "emitted": result["emitted"],
            "top1_match": top1_match, "top3_match": top3_match, "emitted_match": emitted_match,
        })

        target = iv_results if is_iv else oov_results
        target["total"] += 1
        if top1_match: target["correct_top1"] += 1
        if top3_match: target["correct_top3"] += 1
        if top5_match: target["correct_top5"] += 1

        t1 = "OK" if top1_match else "  "
        t3 = "OK" if top3_match else "  "
        t5 = "OK" if top5_match else "  "
        em = "OK" if emitted_match else "  "
        pred_display = result["top1"][:25] if result["top1"] else "NONE"
        iv_tag = "IV" if is_iv else "OOV"
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(videos) - i - 1) / rate if rate > 0 else 0
        print(f"{source:<20} {word:<12} {iv_tag:<6} {pred_display:<25} {result['top1_conf']:>4.0%} {t1:>3} {t3:>3} {t5:>3} {em:>5}")

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(videos)} processed ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    print("\n" + "=" * 110)
    print("RESULTS BY SOURCE")
    print("=" * 110)
    print(f"{'Source':<20} {'Total':>5} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'Emitted':>8} {'No Emit':>8}")
    print("-" * 70)

    totals = {"total": 0, "correct_top1": 0, "correct_top3": 0, "correct_top5": 0, "emitted_match": 0, "no_emission": 0}
    for source in sorted(results_by_source.keys()):
        r = results_by_source[source]
        t = r["total"]
        if t == 0:
            continue
        top1_pct = r["correct_top1"] / t * 100
        top3_pct = r["correct_top3"] / t * 100
        top5_pct = r["correct_top5"] / t * 100
        emit_pct = r["emitted_match"] / t * 100
        no_emit = r["no_emission"]
        print(f"{source:<20} {t:>5} {r['correct_top1']:>3}/{t:<3} ({top1_pct:>5.1f}%) {r['correct_top3']:>3}/{t:<3} ({top3_pct:>5.1f}%) {r['correct_top5']:>3}/{t:<3} ({top5_pct:>5.1f}%) {r['emitted_match']:>3}/{t:<3} ({emit_pct:>5.1f}%) {no_emit:>5}")
        totals["total"] += t
        totals["correct_top1"] += r["correct_top1"]
        totals["correct_top3"] += r["correct_top3"]
        totals["correct_top5"] += r["correct_top5"]
        totals["emitted_match"] += r["emitted_match"]
        totals["no_emission"] += r["no_emission"]

    t = totals["total"]
    if t > 0:
        print("-" * 70)
        print(f"{'TOTAL':<20} {t:>5} {totals['correct_top1']:>3}/{t:<3} ({totals['correct_top1']/t*100:>5.1f}%) {totals['correct_top3']:>3}/{t:<3} ({totals['correct_top3']/t*100:>5.1f}%) {totals['correct_top5']:>3}/{t:<3} ({totals['correct_top5']/t*100:>5.1f}%) {totals['emitted_match']:>3}/{t:<3} ({totals['emitted_match']/t*100:>5.1f}%) {totals['no_emission']:>5}")

    print("\n" + "=" * 110)
    print("IN-VOCABULARY vs OUT-OF-VOCABULARY")
    print("=" * 110)
    for label, stats in [("IN-VOCABULARY (model was trained on these words)", iv_results),
                         ("OUT-OF-VOCABULARY (model was NOT trained on these words)", oov_results)]:
        t = stats["total"]
        if t > 0:
            print(f"\n{label}:")
            print(f"  Clips: {t}")
            print(f"  Top-1: {stats['correct_top1']}/{t} ({stats['correct_top1']/t*100:.1f}%)")
            print(f"  Top-3: {stats['correct_top3']}/{t} ({stats['correct_top3']/t*100:.1f}%)")
            print(f"  Top-5: {stats['correct_top5']}/{t} ({stats['correct_top5']/t*100:.1f}%)")

    print("\n" + "=" * 110)
    print("PER-WORD ACCURACY (Top-1)")
    print("=" * 110)
    word_stats = defaultdict(lambda: {"correct": 0, "total": 0, "iv": False})
    for source, r in results_by_source.items():
        for p in r["predictions"]:
            w = p["word"]
            word_stats[w]["total"] += 1
            word_stats[w]["iv"] = p["iv"]
            if p["top1_match"]:
                word_stats[w]["correct"] += 1

    for word in sorted(word_stats.keys()):
        s = word_stats[word]
        pct = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        bar = "#" * int(pct / 5)
        iv_tag = "IV" if s["iv"] else "OOV"
        print(f"  {word:<15} {iv_tag:<4} {s['correct']:>2}/{s['total']:<2} {pct:>5.1f}% {bar}")

    output_path = Path("results/real_world_accuracy.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "vocabulary": {
                "model_labels": len(model_vocab),
                "test_words": len(set(normalize_label(row.get("normalized_word", row.get("word", ""))) for row in metadata)),
                "in_vocabulary_words": iv_words,
                "out_of_vocabulary_words": oov_words,
                "in_vocabulary_clips": iv_results["total"],
                "out_of_vocabulary_clips": oov_results["total"],
            },
            "by_source": {s: {"total": r["total"], "top1": r["correct_top1"], "top3": r["correct_top3"],
                              "top5": r["correct_top5"], "emitted": r["emitted_match"], "no_emission": r["no_emission"]}
                          for s, r in results_by_source.items()},
            "totals": totals,
            "in_vocabulary": iv_results,
            "out_of_vocabulary": oov_results,
            "per_word": {w: s for w, s in word_stats.items()},
        }, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
