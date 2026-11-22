#!/usr/bin/env python
"""Real-world testing: per-signer accuracy, variance, and qualitative feedback."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
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


def load_label_map(path: str | Path) -> dict[int, str]:
    with open(path) as f:
        raw = json.load(f)
    return {int(v): k for k, v in raw.items()}


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
                  label_map: dict[int, str], config, device, source: str | None = None) -> dict:
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
                if source is not None and hasattr(model, "sources"):
                    logits = model(seq_tensor, mask_tensor, source=source)
                else:
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


def run_demo_mode(args, metadata):
    """Run in demo mode using cached predictions, skipping model inference."""
    demo_path = Path(args.demo_predictions)
    if not demo_path.exists():
        print(f"Demo predictions not found at {demo_path}")
        sys.exit(1)

    with open(demo_path) as f:
        cached_predictions = json.load(f)

    print(f"\nDemo mode: loaded {len(cached_predictions)} cached predictions from {demo_path}")
    if args.no_tts:
        print("TTS disabled (--no-tts)")

    clip_to_row = {}
    for row in metadata:
        video_path = row["video_path"]
        clip_to_row[video_path] = row

    print(f"\nEvaluating {len(cached_predictions)} demo clips\n")
    print(f"{'Signer':<20} {'Word':<12} {'Top-1':<25} {'Conf':>5} {'T1':>3} {'T3':>3} {'T5':>3}")
    print("-" * 85)

    per_signer: dict[str, dict] = {}
    t_start = time.time()

    for i, pred in enumerate(cached_predictions):
        clip_path = pred["clip_path"]
        signer = pred["signer"]
        word = pred["ground_truth"]
        word_norm = normalize_label(word)

        row = clip_to_row.get(clip_path)
        if not row:
            print(f"Warning: clip {clip_path} not found in metadata, skipping")
            continue

        top1 = pred["top1_prediction"]
        top1_conf = pred["top1_confidence"]
        top3 = pred["top3_predictions"]
        top5 = pred["top5_predictions"]

        top1_match = labels_match(top1, word) if top1 else False
        top3_match = any(labels_match(p[0], word) for p in top3)
        top5_match = any(labels_match(p[0], word) for p in top5)

        if signer not in per_signer:
            per_signer[signer] = {"correct_top1": 0, "correct_top3": 0, "correct_top5": 0,
                                   "total": 0, "no_landmarks": 0, "predictions": []}
        per_signer[signer]["total"] += 1
        if top1_match:
            per_signer[signer]["correct_top1"] += 1
        if top3_match:
            per_signer[signer]["correct_top3"] += 1
        if top5_match:
            per_signer[signer]["correct_top5"] += 1
        per_signer[signer]["predictions"].append({
            "word": word_norm, "top1": top1, "top1_conf": top1_conf,
            "top3": [p[0] for p in top3],
            "top1_match": top1_match, "top3_match": top3_match, "top5_match": top5_match,
            "no_landmarks": False,
        })

        t1 = "OK" if top1_match else "  "
        t3 = "OK" if top3_match else "  "
        t5 = "OK" if top5_match else "  "
        pred_display = top1[:25] if top1 else "NONE"
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(cached_predictions) - i - 1) / rate if rate > 0 else 0
        print(f"{signer:<20} {word:<12} {pred_display:<25} {top1_conf:>4.0%} {t1:>3} {t3:>3} {t5:>3}")

    print("\n" + "=" * 85)
    print("PER-SIGNER ACCURACY")
    print("=" * 85)
    print(f"{'Signer':<20} {'Total':>5} {'Top-1':>10} {'Top-3':>10} {'Top-5':>10} {'NoLM':>5}")
    print("-" * 65)

    all_top1_rates = []
    all_top3_rates = []
    all_top5_rates = []
    totals = {"total": 0, "correct_top1": 0, "correct_top3": 0, "correct_top5": 0}

    for signer in sorted(per_signer.keys()):
        s = per_signer[signer]
        t = s["total"]
        if t == 0:
            continue
        top1_rate = s["correct_top1"] / t
        top3_rate = s["correct_top3"] / t
        top5_rate = s["correct_top5"] / t
        all_top1_rates.append(top1_rate)
        all_top3_rates.append(top3_rate)
        all_top5_rates.append(top5_rate)
        no_lm = s.get("no_landmarks", 0)
        print(f"{signer:<20} {t:>5} {s['correct_top1']:>3}/{t:<3} ({top1_rate*100:>5.1f}%) "
              f"{s['correct_top3']:>3}/{t:<3} ({top3_rate*100:>5.1f}%) "
              f"{s['correct_top5']:>3}/{t:<3} ({top5_rate*100:>5.1f}%) {no_lm:>5}")
        totals["total"] += t
        totals["correct_top1"] += s["correct_top1"]
        totals["correct_top3"] += s["correct_top3"]
        totals["correct_top5"] += s["correct_top5"]

    t = totals["total"]
    if t > 0:
        print("-" * 65)
        print(f"{'OVERALL':<20} {t:>5} {totals['correct_top1']:>3}/{t:<3} ({totals['correct_top1']/t*100:>5.1f}%) "
              f"{totals['correct_top3']:>3}/{t:<3} ({totals['correct_top3']/t*100:>5.1f}%) "
              f"{totals['correct_top5']:>3}/{t:<3} ({totals['correct_top5']/t*100:>5.1f}%)")

    print("\n" + "=" * 85)
    print("VARIANCE ANALYSIS")
    print("=" * 85)
    if len(all_top1_rates) >= 2:
        top1_stdev = statistics.stdev(all_top1_rates)
        top3_stdev = statistics.stdev(all_top3_rates)
        top5_stdev = statistics.stdev(all_top5_rates)
        top1_mean = statistics.mean(all_top1_rates)
        top3_mean = statistics.mean(all_top3_rates)
        top5_mean = statistics.mean(all_top5_rates)
        print(f"Top-1: mean={top1_mean*100:.1f}%, stdev={top1_stdev*100:.1f}%, "
              f"range=[{min(all_top1_rates)*100:.1f}%, {max(all_top1_rates)*100:.1f}%]")
        print(f"Top-3: mean={top3_mean*100:.1f}%, stdev={top3_stdev*100:.1f}%, "
              f"range=[{min(all_top3_rates)*100:.1f}%, {max(all_top3_rates)*100:.1f}%]")
        print(f"Top-5: mean={top5_mean*100:.1f}%, stdev={top5_stdev*100:.1f}%, "
              f"range=[{min(all_top5_rates)*100:.1f}%, {max(all_top5_rates)*100:.1f}%]")
    elif len(all_top1_rates) == 1:
        print("Only 1 signer: variance not applicable")
    else:
        print("No signers with valid clips")

    print("\n" + "=" * 85)
    print("QUALITATIVE FEEDBACK")
    print("=" * 85)
    for signer in sorted(per_signer.keys()):
        s = per_signer[signer]
        predictions = s.get("predictions", [])
        if not predictions:
            continue
        confusions: dict[str, int] = defaultdict(int)
        for p in predictions:
            if p.get("no_landmarks"):
                continue
            if not p["top1_match"]:
                gt = p["word"]
                pred = p["top1"]
                confusions[f"{gt}->{pred}"] += 1
        if confusions:
            top_confusions = sorted(confusions.items(), key=lambda x: -x[1])[:5]
            print(f"\n  {signer} (confusions):")
            for conf, count in top_confusions:
                print(f"    {conf}: {count}x")
        else:
            print(f"\n  {signer}: 100% top-1 accuracy, no confusions")

    qualitative = {}
    for signer in sorted(per_signer.keys()):
        confusions: dict[str, int] = defaultdict(int)
        for p in per_signer[signer].get("predictions", []):
            if not p.get("no_landmarks") and not p["top1_match"]:
                confusions[f"{p['word']}->{p['top1']}"] += 1
        qualitative[signer] = dict(sorted(confusions.items(), key=lambda x: -x[1]))

    output_path = Path(args.output_dir) / "real_world_test_demo.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "per_signer": {
                s: {
                    "total": info["total"],
                    "top1": info["correct_top1"],
                    "top3": info["correct_top3"],
                    "top5": info["correct_top5"],
                    "top1_rate": info["correct_top1"] / info["total"] if info["total"] > 0 else 0,
                    "top3_rate": info["correct_top3"] / info["total"] if info["total"] > 0 else 0,
                    "top5_rate": info["correct_top5"] / info["total"] if info["total"] > 0 else 0,
                    "no_landmarks": info.get("no_landmarks", 0),
                    "predictions": info["predictions"],
                }
                for s, info in sorted(per_signer.items())
            },
            "totals": {
                "total": totals["total"],
                "top1": totals["correct_top1"],
                "top3": totals["correct_top3"],
                "top5": totals["correct_top5"],
                "top1_rate": totals["correct_top1"] / totals["total"] if totals["total"] > 0 else 0,
                "top3_rate": totals["correct_top3"] / totals["total"] if totals["total"] > 0 else 0,
                "top5_rate": totals["correct_top5"] / totals["total"] if totals["total"] > 0 else 0,
            },
            "variance": {
                "num_signers": len(all_top1_rates),
                "top1_stdev": statistics.stdev(all_top1_rates) if len(all_top1_rates) >= 2 else None,
                "top3_stdev": statistics.stdev(all_top3_rates) if len(all_top3_rates) >= 2 else None,
                "top5_stdev": statistics.stdev(all_top5_rates) if len(all_top5_rates) >= 2 else None,
                "top1_range": [min(all_top1_rates), max(all_top1_rates)] if all_top1_rates else None,
                "top3_range": [min(all_top3_rates), max(all_top3_rates)] if all_top3_rates else None,
                "top5_range": [min(all_top5_rates), max(all_top5_rates)] if all_top5_rates else None,
            },
            "qualitative": qualitative,
        }, f, indent=2)
    print(f"\nDetailed demo results saved to {output_path}")
    elapsed_total = time.time() - t_start
    print(f"Demo completed in {elapsed_total:.1f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Real-world testing: per-signer accuracy and variance")
    parser.add_argument("--model-path", default="models/best.pt", help="Path to model checkpoint")
    parser.add_argument("--dataset-dir", default="isl_40_test", help="Dataset directory with metadata.csv")
    parser.add_argument("--output-dir", default="results", help="Directory for output JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without running inference")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode using cached predictions")
    parser.add_argument("--demo-predictions", default="demo/cached_predictions_full.json", help="Path to cached predictions JSON for demo mode")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS output (for demo mode)")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode (no network access, use cached voice)")
    args = parser.parse_args()

    if args.offline:
        os.environ["PIPER_OFFLINE"] = "1"
        os.environ["PIPER_DATA_DIR"] = str(Path("demo/piper_voice").resolve())

    dataset_dir = Path(args.dataset_dir)
    metadata_path = dataset_dir / "metadata.csv"

    if not metadata_path.exists():
        print(f"Metadata not found at {metadata_path}")
        sys.exit(1)

    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        metadata = list(reader)

    if not metadata:
        print("Metadata is empty")
        sys.exit(1)

    signer_groups: dict[str, list[dict]] = defaultdict(list)
    for row in metadata:
        signer = row.get("signer", "")
        if not signer or signer.strip() == "":
            signer = "unknown"
        signer_groups[signer].append(row)

    print(f"Found {len(signer_groups)} signers across {len(metadata)} clips")
    for signer in sorted(signer_groups.keys()):
        clips = signer_groups[signer]
        words = {normalize_label(r.get("normalized_word", r.get("word", ""))) for r in clips}
        sources = {r.get("dataset", "unknown") for r in clips}
        print(f"  {signer}: {len(clips)} clips, {len(words)} words, sources={sources}")

    if args.dry_run:
        print("\nDry run: no inference performed.")
        return

    if args.demo:
        run_demo_mode(args, metadata)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("configs/default.yaml", "configs/isl.yaml")

    label_map_path = Path(args.model_path).parent / "label_map.json"
    if not label_map_path.exists():
        label_map_path = Path("models/label_map.json")
    label_map = load_label_map(label_map_path)
    model_vocab = build_model_vocabulary(label_map)

    print(f"\nModel vocabulary: {len(model_vocab)} labels")

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    ckpt_model = ckpt.get("config", {}).get("model", {})
    if ckpt_model.get("num_classes"):
        config.model.num_classes = ckpt_model["num_classes"]

    model = build_model(config.model).to(device)
    if isinstance(ckpt.get("sources"), list):
        from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

        htm = HTMAConfig(input_dim=config.model.input_dim, embedding_dim=config.model.embedding_dim,
                         num_blocks=config.model.htma_blocks, num_heads=config.model.num_heads,
                         dropout=config.model.dropout, num_classes=len(label_map))
        model = MultiHeadHTMAClassifier(htm, sources=tuple(ckpt["sources"])).to(device)
        print(f"Multi-head model: sources={ckpt['sources']}, routing by clip dataset")
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

    cached_count = sum(1 for vp, _ in videos if (CACHE_DIR / f"{cache_key(vp)}.npz").exists())
    print(f"\nLandmark cache: {cached_count}/{len(videos)} already cached")

    print(f"\nEvaluating {len(videos)} videos from {len(signer_groups)} signers\n")
    print(f"{'Signer':<20} {'Word':<12} {'Top-1':<25} {'Conf':>5} {'T1':>3} {'T3':>3} {'T5':>3}")
    print("-" * 85)

    per_signer: dict[str, dict] = {}
    t_start = time.time()

    for i, (video_path, row) in enumerate(videos):
        signer = row.get("signer", "")
        if not signer or signer.strip() == "":
            signer = "unknown"
        word = row.get("normalized_word", row.get("word", ""))
        word_norm = normalize_label(word)

        frames = extract_and_cache(video_path, CACHE_DIR, smoother)

        if not frames:
            if signer not in per_signer:
                per_signer[signer] = {"correct_top1": 0, "correct_top3": 0, "correct_top5": 0,
                                       "total": 0, "no_landmarks": 0, "predictions": []}
            per_signer[signer]["total"] += 1
            per_signer[signer]["no_landmarks"] += 1
            per_signer[signer]["predictions"].append({
                "word": word_norm, "top1": "", "top1_match": False,
                "top3_match": False, "top5_match": False, "no_landmarks": True,
            })
            print(f"{signer:<20} {word:<12} {'NO LANDMARKS':<25}")
            continue

        src = row.get("dataset", "")
        route = "include" if src == "INCLUDE" else ("isl500" if hasattr(model, "sources") else None)
        result = run_inference(frames, model, ring_buffer, gate, label_map, config, device, source=route)

        top1_match = labels_match(result["top1"], word) if result["top1"] else False
        top3_match = any(labels_match(p[0], word) for p in result["top3"])
        top5_match = any(labels_match(p[0], word) for p in result["top5"])

        if signer not in per_signer:
            per_signer[signer] = {"correct_top1": 0, "correct_top3": 0, "correct_top5": 0,
                                   "total": 0, "no_landmarks": 0, "predictions": []}
        per_signer[signer]["total"] += 1
        if top1_match:
            per_signer[signer]["correct_top1"] += 1
        if top3_match:
            per_signer[signer]["correct_top3"] += 1
        if top5_match:
            per_signer[signer]["correct_top5"] += 1
        per_signer[signer]["predictions"].append({
            "word": word_norm, "top1": result["top1"], "top1_conf": result["top1_conf"],
            "top3": [p[0] for p in result["top3"]],
            "top1_match": top1_match, "top3_match": top3_match, "top5_match": top5_match,
            "no_landmarks": False,
        })

        t1 = "OK" if top1_match else "  "
        t3 = "OK" if top3_match else "  "
        t5 = "OK" if top5_match else "  "
        pred_display = result["top1"][:25] if result["top1"] else "NONE"
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(videos) - i - 1) / rate if rate > 0 else 0
        print(f"{signer:<20} {word:<12} {pred_display:<25} {result['top1_conf']:>4.0%} {t1:>3} {t3:>3} {t5:>3}")

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(videos)} processed ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    print("\n" + "=" * 85)
    print("PER-SIGNER ACCURACY")
    print("=" * 85)
    print(f"{'Signer':<20} {'Total':>5} {'Top-1':>10} {'Top-3':>10} {'Top-5':>10} {'NoLM':>5}")
    print("-" * 65)

    all_top1_rates = []
    all_top3_rates = []
    all_top5_rates = []
    totals = {"total": 0, "correct_top1": 0, "correct_top3": 0, "correct_top5": 0}

    for signer in sorted(per_signer.keys()):
        s = per_signer[signer]
        t = s["total"]
        if t == 0:
            continue
        top1_rate = s["correct_top1"] / t
        top3_rate = s["correct_top3"] / t
        top5_rate = s["correct_top5"] / t
        all_top1_rates.append(top1_rate)
        all_top3_rates.append(top3_rate)
        all_top5_rates.append(top5_rate)
        no_lm = s.get("no_landmarks", 0)
        print(f"{signer:<20} {t:>5} {s['correct_top1']:>3}/{t:<3} ({top1_rate*100:>5.1f}%) "
              f"{s['correct_top3']:>3}/{t:<3} ({top3_rate*100:>5.1f}%) "
              f"{s['correct_top5']:>3}/{t:<3} ({top5_rate*100:>5.1f}%) {no_lm:>5}")
        totals["total"] += t
        totals["correct_top1"] += s["correct_top1"]
        totals["correct_top3"] += s["correct_top3"]
        totals["correct_top5"] += s["correct_top5"]

    t = totals["total"]
    if t > 0:
        print("-" * 65)
        print(f"{'OVERALL':<20} {t:>5} {totals['correct_top1']:>3}/{t:<3} ({totals['correct_top1']/t*100:>5.1f}%) "
              f"{totals['correct_top3']:>3}/{t:<3} ({totals['correct_top3']/t*100:>5.1f}%) "
              f"{totals['correct_top5']:>3}/{t:<3} ({totals['correct_top5']/t*100:>5.1f}%)")

    print("\n" + "=" * 85)
    print("VARIANCE ANALYSIS")
    print("=" * 85)
    if len(all_top1_rates) >= 2:
        top1_stdev = statistics.stdev(all_top1_rates)
        top3_stdev = statistics.stdev(all_top3_rates)
        top5_stdev = statistics.stdev(all_top5_rates)
        top1_mean = statistics.mean(all_top1_rates)
        top3_mean = statistics.mean(all_top3_rates)
        top5_mean = statistics.mean(all_top5_rates)
        print(f"Top-1: mean={top1_mean*100:.1f}%, stdev={top1_stdev*100:.1f}%, "
              f"range=[{min(all_top1_rates)*100:.1f}%, {max(all_top1_rates)*100:.1f}%]")
        print(f"Top-3: mean={top3_mean*100:.1f}%, stdev={top3_stdev*100:.1f}%, "
              f"range=[{min(all_top3_rates)*100:.1f}%, {max(all_top3_rates)*100:.1f}%]")
        print(f"Top-5: mean={top5_mean*100:.1f}%, stdev={top5_stdev*100:.1f}%, "
              f"range=[{min(all_top5_rates)*100:.1f}%, {max(all_top5_rates)*100:.1f}%]")
    elif len(all_top1_rates) == 1:
        print("Only 1 signer: variance not applicable")
    else:
        print("No signers with valid clips")

    print("\n" + "=" * 85)
    print("QUALITATIVE FEEDBACK")
    print("=" * 85)
    for signer in sorted(per_signer.keys()):
        s = per_signer[signer]
        predictions = s.get("predictions", [])
        if not predictions:
            continue
        confusions: dict[str, int] = defaultdict(int)
        for p in predictions:
            if p.get("no_landmarks"):
                continue
            if not p["top1_match"]:
                gt = p["word"]
                pred = p["top1"]
                confusions[f"{gt}->{pred}"] += 1
        if confusions:
            top_confusions = sorted(confusions.items(), key=lambda x: -x[1])[:5]
            print(f"\n  {signer} (confusions):")
            for conf, count in top_confusions:
                print(f"    {conf}: {count}x")
        else:
            print(f"\n  {signer}: 100% top-1 accuracy, no confusions")

    qualitative = {}
    for signer in sorted(per_signer.keys()):
        confusions: dict[str, int] = defaultdict(int)
        for p in per_signer[signer].get("predictions", []):
            if not p.get("no_landmarks") and not p["top1_match"]:
                confusions[f"{p['word']}->{p['top1']}"] += 1
        qualitative[signer] = dict(sorted(confusions.items(), key=lambda x: -x[1]))

    output_path = Path(args.output_dir) / "real_world_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "per_signer": {
                s: {
                    "total": info["total"],
                    "top1": info["correct_top1"],
                    "top3": info["correct_top3"],
                    "top5": info["correct_top5"],
                    "top1_rate": info["correct_top1"] / info["total"] if info["total"] > 0 else 0,
                    "top3_rate": info["correct_top3"] / info["total"] if info["total"] > 0 else 0,
                    "top5_rate": info["correct_top5"] / info["total"] if info["total"] > 0 else 0,
                    "no_landmarks": info.get("no_landmarks", 0),
                    "predictions": info["predictions"],
                }
                for s, info in sorted(per_signer.items())
            },
            "totals": {
                "total": totals["total"],
                "top1": totals["correct_top1"],
                "top3": totals["correct_top3"],
                "top5": totals["correct_top5"],
                "top1_rate": totals["correct_top1"] / totals["total"] if totals["total"] > 0 else 0,
                "top3_rate": totals["correct_top3"] / totals["total"] if totals["total"] > 0 else 0,
                "top5_rate": totals["correct_top5"] / totals["total"] if totals["total"] > 0 else 0,
            },
            "variance": {
                "num_signers": len(all_top1_rates),
                "top1_stdev": statistics.stdev(all_top1_rates) if len(all_top1_rates) >= 2 else None,
                "top3_stdev": statistics.stdev(all_top3_rates) if len(all_top3_rates) >= 2 else None,
                "top5_stdev": statistics.stdev(all_top5_rates) if len(all_top5_rates) >= 2 else None,
                "top1_range": [min(all_top1_rates), max(all_top1_rates)] if all_top1_rates else None,
                "top3_range": [min(all_top3_rates), max(all_top3_rates)] if all_top3_rates else None,
                "top5_range": [min(all_top5_rates), max(all_top5_rates)] if all_top5_rates else None,
            },
            "qualitative": qualitative,
        }, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
