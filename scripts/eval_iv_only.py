from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute IV-only metrics from real_world_accuracy.json")
    parser.add_argument("--input", default="results/real_world_accuracy.json")
    parser.add_argument("--output", default="results/iv_only_metrics.json")
    args = parser.parse_args()

    src = Path(args.input)
    data = json.loads(src.read_text())

    iv = data.get("in_vocabulary", {})
    iv_total = int(iv.get("total", 0))
    iv_top1 = int(iv.get("correct_top1", 0))
    iv_top3 = int(iv.get("correct_top3", 0))
    iv_top5 = int(iv.get("correct_top5", 0))

    totals = data.get("totals", {})
    total = int(totals.get("total", 0))

    out = {
        "source_artifact": str(src),
        "overall_clips": total,
        "overall_top1": totals.get("correct_top1", 0) / total if total else 0.0,
        "iv_clips": iv_total,
        "iv_top1": iv_top1 / iv_total if iv_total else 0.0,
        "iv_top3": iv_top3 / iv_total if iv_total else 0.0,
        "iv_top5": iv_top5 / iv_total if iv_total else 0.0,
        "iv_correct_top1": iv_top1,
        "iv_correct_top3": iv_top3,
        "iv_correct_top5": iv_top5,
        "oov_clips": int(data.get("out_of_vocabulary", {}).get("total", 0)),
        "oov_top1": 0.0,
        "by_source_overall": data.get("by_source", {}),
        "per_word_iv": {w: v for w, v in data.get("per_word", {}).items() if v.get("iv")},
        "note": "OOV is 0% by definition (model cannot predict unseen classes). Report IV-only going forward. Per-source IV split unavailable in source artifact; by_source above is overall per source.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"IV top-1: {out['iv_top1']*100:.2f}% ({iv_top1}/{iv_total})")
    print(f"Overall top-1: {out['overall_top1']*100:.2f}%")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
