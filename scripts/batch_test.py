#!/usr/bin/env python
"""Batch test on CISLR videos from the downloaded 40-word dataset."""
from __future__ import annotations

import json
import sys
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
from src.features.spec import DEFAULT_FEATURE_SPEC
from src.features.normalize import build_frame_features
from src.features.smooth import LandmarkSmoother
from src.infer.buffer import RingBuffer
from src.infer.gate import StabilityGate
from src.models.builder import build_model


def load_label_map(path: str | Path) -> dict[int, str]:
    with open(path) as f:
        raw = json.load(f)
    return {int(v): k for k, v in raw.items()}


def test_video(video_path: Path, model, extractor, ring_buffer, gate,
               label_map: dict[int, str], config, device, smoother: LandmarkSmoother | None = None) -> dict:
    INFERENCE_STRIDE = 2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path.name}"}

    ring_buffer.reset()
    gate.reset()
    if smoother:
        smoother.reset()

    frame_idx = 0
    all_probs = []
    emitted_words = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts_ms = frame_idx * 33
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = extractor._hand_landmarker.detect_for_video(mp_image, ts_ms)
        pose_result = extractor._pose_landmarker.detect_for_video(mp_image, ts_ms)

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

        features = build_frame_features(left_hand, right_hand, pose_coords, extractor.spec)
        if smoother:
            features = smoother.smooth(features)
        ring_buffer.add(features)

        if ring_buffer.is_full and frame_idx % INFERENCE_STRIDE == 0:
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

        frame_idx += 1

    cap.release()

    if not all_probs:
        return {"error": "No predictions"}

    avg_probs = np.mean(all_probs, axis=0)
    top3_idx = np.argsort(avg_probs)[-3:][::-1]
    top3 = [(label_map.get(int(i), f"class_{i}"), float(avg_probs[i])) for i in top3_idx]

    return {
        "top1": top3[0][0],
        "top1_conf": top3[0][1],
        "top3": top3,
        "emitted": emitted_words,
    }


def main():
    dataset_dir = Path("isl_40_test")
    if not dataset_dir.exists():
        print("Dataset not found. Run download first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("configs/default.yaml", "configs/isl.yaml")

    label_map_path = Path("models/label_map.json")
    if not label_map_path.exists():
        print(f"Label map not found at {label_map_path}")
        sys.exit(1)
    label_map = load_label_map(label_map_path)

    ckpt = torch.load("models/best.pt", map_location=device, weights_only=False)
    ckpt_model = ckpt.get("config", {}).get("model", {})
    if ckpt_model.get("num_classes"):
        config.model.num_classes = ckpt_model["num_classes"]

    model = build_model(config.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    smoother = LandmarkSmoother(feature_dim=config.model.input_dim)

    extractor = OfflineExtractor(
        hand_model_path="assets/hand_landmarker.task",
        pose_model_path="assets/pose_landmarker.lite.task",
    )
    ring_buffer = RingBuffer(config.processing.sequence_length, config.model.input_dim)
    gate = StabilityGate(
        threshold=config.processing.stability_gate.threshold,
        consecutive_frames=config.processing.stability_gate.consecutive_frames,
        cooldown_frames=config.processing.stability_gate.cooldown_frames,
        agreement_ratio=config.processing.stability_gate.agreement_ratio,
        min_confidence_delta=config.processing.stability_gate.min_confidence_delta,
    )

    words = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])

    print(f"{'Word':<12} {'Predicted':<30} {'Conf':>6} {'Match':>6} {'Emitted'}")
    print("-" * 90)

    correct = 0
    total = 0
    no_emission = 0

    for word in words:
        vids = list((dataset_dir / word).glob("*.mp4"))
        if not vids:
            continue

        vid = vids[0]
        ext = OfflineExtractor(
            hand_model_path="assets/hand_landmarker.task",
            pose_model_path="assets/pose_landmarker.lite.task",
        )
        result = test_video(vid, model, ext, ring_buffer, gate, label_map, config, device, smoother)
        ext.close()

        if "error" in result:
            print(f"{word:<12} ERROR: {result['error']}")
            continue

        pred_raw = result["top1"]
        pred = pred_raw.split(".")[-1].strip().lower() if "." in pred_raw else pred_raw.lower()
        gt = word.lower()
        match = pred == gt or gt in pred or pred in gt
        emitted = result["emitted"]
        emitted_match = any(gt in e.lower() or e.lower() in gt for e in emitted)

        if match or emitted_match:
            correct += 1
        if not emitted:
            no_emission += 1
        total += 1

        status = "PASS" if (match or emitted_match) else "FAIL"
        emit_str = ", ".join(e.split(".")[-1].strip() for e in emitted[:3]) if emitted else "(none)"
        print(f"{word:<12} {result['top1']:<30} {result['top1_conf']:>5.1%} {status:>6} [{emit_str}]")

    extractor.close()

    print("\n" + "=" * 90)
    print(f"RESULTS: {correct}/{total} correct ({correct/total*100:.1f}%)" if total else "No results")
    print(f"No emissions: {no_emission}/{total} videos")
    print("=" * 90)


if __name__ == "__main__":
    main()
