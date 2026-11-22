#!/usr/bin/env python
"""Gate hyperparameter sweep: GPU prob-dump once, CPU replay per setting.

Phase 1: --dump-probs runs the ring-buffer + stride-2 forward loop per clip
(cache-hit only) and persists per-step probabilities.
Phase 2: --replay replays StabilityGate settings over stored probs (CPU-only)
and scores gate emissions (overall / IV / OOV / per-source).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.real_world_test import (  # noqa: E402
    CACHE_DIR,
    SYNONYMS,
    build_model_vocabulary,
    cache_key,
    labels_match,
    load_label_map,
    normalize_label,
)
from src.features.smooth import LandmarkSmoother  # noqa: E402
from src.infer.buffer import RingBuffer  # noqa: E402
from src.infer.gate import StabilityGate  # noqa: E402

GRID = {
    "threshold": [0.25, 0.35, 0.40, 0.50, 0.60, 0.70],
    "consecutive_frames": [6, 9, 12, 16],
    "agreement_ratio": [0.50, 0.60, 0.75, 1.00],
    "min_confidence_delta": [0.00, 0.05, 0.08, 0.12, 0.16],
}
DEFAULT = {"threshold": 0.50, "consecutive_frames": 12,
           "agreement_ratio": 0.60, "min_confidence_delta": 0.08}


def load_model(model_path: str, config, device):
    from src.models.builder import build_model

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    use_source = isinstance(ckpt.get("sources"), list)
    if use_source:
        from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

        state = ckpt["model_state_dict"]
        head_classes: dict[str, int] = dict(ckpt.get("head_classes") or {})
        for src in ckpt["sources"]:
            for suffix in ("heads." + src + ".3.weight", "heads." + src + ".weight"):
                if suffix in state:
                    head_classes[src] = state[suffix].shape[0]
                    break
        n_cls = max(head_classes.values()) if head_classes else config.model.num_classes
        htm = HTMAConfig(input_dim=168, embedding_dim=256, num_blocks=2, num_heads=8,
                         dropout=0.1, num_classes=n_cls)
        model = MultiHeadHTMAClassifier(htm, sources=tuple(ckpt["sources"]),
                                        head_classes=head_classes).to(device).eval()
        own = model.state_dict()
        pruned = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(pruned, strict=False)
    else:
        if ckpt.get("config", {}).get("model", {}).get("num_classes"):
            config.model.num_classes = ckpt["config"]["model"]["num_classes"]
        model = build_model(config.model).to(device).eval()
        model.load_state_dict(ckpt["model_state_dict"])
    return model, use_source


def build_united_map(model_path: str) -> tuple[dict[int, str], list[tuple[str, int, int]]]:
    joint: dict[str, int] = json.loads(Path("data/processed/landmarks/joint_label_map.json").read_text())
    ex_path = Path("models/multihead_extra/label_map_extra.json")
    extra: dict[str, int] = json.loads(ex_path.read_text()) if ex_path.exists() else {}
    ij = {v: k for k, v in joint.items()}
    ie = {v: k for k, v in extra.items()}
    united: dict[int, str] = {}
    spans = [("include", len(joint)), ("isl500", len(joint)), ("extra", len(extra))]
    off = 0
    for src, n in spans:
        for i in range(n):
            united[off + i] = (ij if src != "extra" else ie).get(i, f"{src}:{i}")
        off += n
    return united, spans


def dump_probs(args) -> None:
    from src.config import load_config

    config = load_config(args.config, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, use_source = load_model(args.model_path, config, device)

    smoother = LandmarkSmoother(feature_dim=config.model.input_dim)
    dataset_dir = Path(args.dataset_dir)
    with open(dataset_dir / "metadata.csv") as f:
        metadata = list(csv.DictReader(f))

    out = Path(args.probs_dir)
    out.mkdir(parents=True, exist_ok=True)
    united, spans = build_united_map(args.model_path)
    n_total = sum(n for _, n in spans)
    joint: dict[str, int] = json.loads(
        Path("data/processed/landmarks/joint_label_map.json").read_text())
    ex_path = Path("models/multihead_extra/label_map_extra.json")
    extra: dict[str, int] = json.loads(ex_path.read_text()) if ex_path.exists() else {}
    model_vocab = {normalize_label(k) for k in list(joint) + list(extra)}
    manifest = []
    t0 = time.time()
    for n, row in enumerate(metadata):
        vp = dataset_dir / row["video_path"]
        if not vp.exists():
            continue
        cached = CACHE_DIR / f"{cache_key(vp)}.npz"
        assert cached.exists(), f"Cache miss (re-extraction forbidden): {vp}"
        data = np.load(cached)
        frames = [data[f"frame_{i}"] for i in range(int(data["n_frames"]))]
        ring = RingBuffer(config.processing.sequence_length, config.model.input_dim)
        word_norm = normalize_label(row.get("normalized_word", row.get("word", "")))
        src = row.get("dataset", "")
        probs = []
        with torch.inference_mode():
            for i, feat in enumerate(frames):
                ring.add(feat)
                if ring.is_full and i % 2 == 0:
                    seq, mask = ring.get_sequence()
                    st = torch.from_numpy(seq).unsqueeze(0).to(device)
                    mt = torch.from_numpy(mask).unsqueeze(0).to(device)
                    if use_source and hasattr(model, "sources"):
                        parts = []
                        for _src in model.sources:
                            _logits = model(st, mt, source=_src)
                            parts.append(torch.softmax(_logits, 1).cpu().numpy()[0])
                        probs.append(np.concatenate(parts))
                    else:
                        logits = model(st, mt)
                        probs.append(torch.softmax(logits, 1).cpu().numpy()[0])
        probs = np.stack(probs) if probs else np.zeros((0, n_total))
        cid = cache_key(vp)
        np.savez_compressed(out / f"{cid}.npz", probs=probs)
        manifest.append({"clip_id": cid, "word": row.get("word", ""), "word_norm": word_norm,
                         "signer": row.get("signer", ""), "dataset": src,
                         "iv": word_norm in model_vocab, "n_frames": len(frames),
                         "n_steps": len(probs)})
        if (n + 1) % 100 == 0:
            print(f"dumped {n + 1} clips ({time.time()-t0:.0f}s)")
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"Dumped {len(manifest)} clips to {out}")


def replay_setting(rows: list[dict], probs_dir: Path, setting: dict,
                   idx2word: dict[int, str], spans: list[tuple[str, int, int]]) -> dict:
    iv_tot = iv_ok = iv_emit = iv_wrong = 0
    oov_tot = oov_emit = 0
    lat: list = []
    head_emit: dict[str, int] = {}
    src_stats: dict[str, dict] = {}
    for r in rows:
        iv = r["iv"] == "True" or r["iv"] is True
        probs = np.load(probs_dir / f"{r['clip_id']}.npz")["probs"]
        gate = StabilityGate(threshold=setting["threshold"],
                             consecutive_frames=setting["consecutive_frames"],
                             cooldown_frames=30,
                             agreement_ratio=setting["agreement_ratio"],
                             min_confidence_delta=setting["min_confidence_delta"])
        first = None
        for s in range(len(probs)):
            e = gate.update(probs[s])
            if e is not None:
                first = (idx2word.get(int(e), ""), s, int(e))
                break
        sname = r["dataset"]
        ss = src_stats.setdefault(sname, {"tot": 0, "ok": 0, "emit": 0})
        ss["tot"] += 1

        def _head_of(idx: int) -> str:
            off = 0
            for _src, _n in spans:
                if idx < off + _n:
                    return _src
                off += _n
            return "unknown"
        if iv:
            iv_tot += 1
            if first is None:
                continue
            iv_emit += 1
            ss["emit"] += 1
            head_emit[_head_of(first[2])] = head_emit.get(_head_of(first[2]), 0) + 1
            ok = labels_match(str(first[0]), r["word"])
            if ok:
                iv_ok += 1
                ss["ok"] += 1
                lat.append(first[1])
            else:
                iv_wrong += 1
        else:
            oov_tot += 1
            if first is not None:
                oov_emit += 1
                head_emit[_head_of(first[2])] = head_emit.get(_head_of(first[2]), 0) + 1
    return {
        "setting": setting,
        "iv_total": iv_tot, "iv_correct": iv_ok, "iv_emitted": iv_emit, "iv_wrong": iv_wrong,
        "iv_top1": iv_ok / iv_tot if iv_tot else 0.0,
        "iv_precision": iv_ok / iv_emit if iv_emit else float("nan"),
        "iv_emission_rate": iv_emit / iv_tot if iv_tot else 0.0,
        "iv_wrong_rate": iv_wrong / iv_tot if iv_tot else 0.0,
        "oov_total": oov_tot, "oov_emitted": oov_emit,
        "oov_false_rate": oov_emit / oov_tot if oov_tot else 0.0,
        "head_emissions": head_emit,
        "median_latency_steps": float(np.median(lat)) if lat else None,
        "p90_latency_steps": float(np.percentile(lat, 90)) if lat else None,
        "per_source": {k: {"top1": v["ok"] / v["tot"], "emit_rate": v["emit"] / v["tot"],
                           "total": v["tot"]} for k, v in src_stats.items()},
    }


def replay(args) -> None:
    probs_dir = Path(args.probs_dir)
    with open(probs_dir / "manifest.csv") as f:
        rows = list(csv.DictReader(f))
    united, spans = build_united_map(args.model_path)
    idx2word = united
    settings = [dict(DEFAULT)]
    for k, vals in GRID.items():
        for v in vals:
            if v == DEFAULT[k]:
                continue
            s = dict(DEFAULT)
            s[k] = v
            settings.append(s)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for s in settings:
            r = replay_setting(rows, probs_dir, s, idx2word, spans)
            f.write(json.dumps(r) + "\n")
            oov = r["oov_false_rate"]
            print(f"{s} -> IV top1={r['iv_top1']*100:.1f}% prec={r['iv_precision']*100:.0f}% "
                  f"emit={r['iv_emission_rate']*100:.0f}% OOV-false={oov*100:.1f}%")
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate sweep: dump probs, replay settings")
    parser.add_argument("--dump-probs", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--model-path", default="models/multihead_v2/best.pt")
    parser.add_argument("--dataset-dir", default="isl_40_test")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument("--probs-dir", default="results/gate_sweep_v2/probs")
    parser.add_argument("--out", default="results/gate_sweep_v2/sweep.jsonl")
    args = parser.parse_args()
    if args.dump_probs:
        dump_probs(args)
    if args.replay:
        replay(args)
    if not args.dump_probs and not args.replay:
        parser.error("Pass --dump-probs and/or --replay")


if __name__ == "__main__":
    main()
