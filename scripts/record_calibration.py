#!/usr/bin/env python
"""Guide and validate self-calibration recording sessions for ISL domain adaptation."""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

TEST_WORDS = [
    "brother", "come", "drink", "eat", "father", "food", "friend", "go",
    "goodbye", "he", "hello", "help", "home", "hospital", "market", "me",
    "mother", "no", "okay", "please", "read", "school", "she", "sister",
    "sit", "sorry", "stand", "stop", "student", "tea", "teacher",
    "thank you", "today", "water", "what", "when", "where", "write",
    "yes", "you",
]

INCLUDE_STYLE_CONDITIONS = [
    "Recording length: 2-6 seconds per word (similar to INCLUDE's per-word clips)",
    "Camera: webcam at 30 FPS, 640x480 minimum resolution",
    "Framing: signer visible from waist up, hands clearly in frame",
    "Background: plain wall or curtain, avoid clutter and other people",
    "Lighting: front-facing, even illumination, no harsh shadows on hands",
    "Clothing: short sleeves preferred for wrist visibility, solid colors",
    "Position: seated or standing, centered in frame, consistent distance",
]

RECORDING_CHECKLIST = [
    "Lighting: front-facing, even, no backlight or overhead shadows",
    "Background: plain, solid color (white/gray/green preferred)",
    "Camera settings: 30 FPS, 640x480 or higher, auto-exposure OFF",
    "Frame: waist-up, centered, hands visible at all times",
    "Clothing: short sleeves, solid color, no patterns or jewelry on hands",
    "Environment: quiet room, no other people moving in frame",
    "Recording tool: OBS Studio or similar (lossless or high-quality codec)",
    "Consistency: same position and lighting for all 40 words",
]


def print_guide(signer_id: str, data_dir: Path) -> None:
    print("=" * 70)
    print("SELF-CALIBRATION RECORDING GUIDE")
    print("=" * 70)
    print(f"\nSigner ID:   {signer_id}")
    print(f"Output dir:  {data_dir}")
    print(f"Words:       {len(TEST_WORDS)}")
    print()

    print("INCLUDE-STYLE RECORDING CONDITIONS")
    print("-" * 50)
    for i, condition in enumerate(INCLUDE_STYLE_CONDITIONS, 1):
        print(f"  {i}. {condition}")

    print()
    print("RECORDING CHECKLIST (verify before each session)")
    print("-" * 50)
    for item in RECORDING_CHECKLIST:
        print(f"  [ ] {item}")

    print()
    print("WORD LIST (40 words)")
    print("-" * 50)
    for i, word in enumerate(TEST_WORDS, 1):
        label = word.replace(" ", "_")
        filename = f"{label}__{signer_id}__{i - 1:04d}.mp4"
        print(f"  {i:>3}. {word:<15} -> {filename}")

    print()
    print("FILE NAMING CONVENTION")
    print("-" * 50)
    print("  {word}__{signer_id}__{index}.mp4")
    print("  Example: thank_you__user01__0000.mp4")
    print("  Store in: {data_dir}/{word}/")

    print()
    print("POST-RECORDING STEPS")
    print("-" * 50)
    print("  1. Organize clips into word-named subdirectories")
    print("  2. Run: python scripts/record_calibration.py --validate --data-dir <dir>")
    print("  3. Generate metadata CSV for the extraction pipeline")


def generate_metadata_template(signer_id: str, data_dir: Path) -> Path:
    metadata_path = data_dir / "calibration_metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "word_label", "signer_id", "recording_date"])
        today_str = date.today().isoformat()
        for word in TEST_WORDS:
            label = word.replace(" ", "_")
            video_path = f"{label}/{label}__{signer_id}__0000.mp4"
            writer.writerow([video_path, word, signer_id, today_str])

    return metadata_path


def validate_clips(data_dir: Path) -> tuple[int, int, list[str]]:
    found = 0
    missing: list[str] = []
    for word in TEST_WORDS:
        label = word.replace(" ", "_")
        word_dir = data_dir / label
        if word_dir.exists():
            mp4_files = list(word_dir.glob("*.mp4"))
            if mp4_files:
                found += 1
            else:
                missing.append(word)
        else:
            missing.append(word)
    return found, len(TEST_WORDS), missing


def print_validation_report(found: int, total: int, missing: list[str], data_dir: Path) -> int:
    print("=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    print(f"\nData directory: {data_dir}")
    print(f"Words found:    {found}/{total}")
    print(f"Words missing:  {total - found}/{total}")

    if missing:
        print(f"\nMissing words ({len(missing)}):")
        for word in missing:
            print(f"  - {word}")

    print()
    if not missing:
        print("STATUS: ALL CLIPS PRESENT")
        print("\nNext steps:")
        print("  1. python scripts/extract_landmarks.py --dataset calibration")
        print("  2. Retrain or fine-tune model with calibration data")
    else:
        print(f"STATUS: {len(missing)} WORDS NEED RECORDING")

    return 0 if not missing else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-calibration recording guide and validator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--guide", action="store_true", help="Print step-by-step recording instructions")
    group.add_argument("--validate", action="store_true", help="Check that recorded clips exist for all 40 words")
    parser.add_argument("--signer-id", default="user01", help="Signer identifier (default: user01)")
    parser.add_argument("--data-dir", type=Path, default=Path("calibration_data"), help="Directory for calibration recordings")
    args = parser.parse_args()

    if args.guide:
        print_guide(args.signer_id, args.data_dir)
        metadata_path = generate_metadata_template(args.signer_id, args.data_dir)
        print(f"\nMetadata template written to: {metadata_path}")
        return 0

    if args.validate:
        found, total, missing = validate_clips(args.data_dir)
        return print_validation_report(found, total, missing, args.data_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
