#!/usr/bin/env python
"""DPC pretraining on OpenHands memmap clips. --smoke N times the loop for the
12h wall-time gate before any full run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def collate(batch):
    import torch

    clips = torch.stack(batch)
    assert clips.shape[1:] == (70, 168)
    return clips.view(-1, 7, 10, 168)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memmap", default="data/processed/openhands/sign_library.f32.memmap")
    parser.add_argument("--clips", default="data/processed/openhands/clips.json")
    parser.add_argument("--out-dir", default="models/dpc")
    parser.add_argument("--iters", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()

    from src.data.dpc_dataset import DPCWindowDataset
    from src.models.dpc import DPCPretrainer
    from src.models.htma import HTMAClassifier, HTMAConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DPCWindowDataset(args.memmap, args.clips)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate, drop_last=True)
    print(f"clips: {ds.num_clips}", flush=True)

    cfg = HTMAConfig(input_dim=168, embedding_dim=256, num_blocks=2,
                     num_heads=8, dropout=0.1, num_classes=263)
    dpc = DPCPretrainer(HTMAClassifier(cfg)).to(device)
    opt = torch.optim.Adam(dpc.parameters(), lr=args.lr)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "dpc_log.jsonl", "a")

    iters = args.smoke or args.iters
    t0 = time.time()
    step = 0
    dpc.train()
    use_amp = device.type == "cuda"
    while step < iters:
        for clips in loader:
            if step >= iters:
                break
            clips = clips.to(device)
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=use_amp):
                loss, acc = dpc(clips)
            loss.backward()
            nn.utils.clip_grad_norm_(dpc.parameters(), max_norm=1.0)
            opt.step()
            step += 1
            if step % 50 == 0 or step == iters:
                dt = time.time() - t0
                print(f"step {step}/{iters} loss={loss.item():.4f} "
                      f"acc={acc.item():.3f} elapsed={dt:.0f}s", flush=True)
                log.write(json.dumps({"step": step, "loss": round(loss.item(), 4),
                                      "acc": round(acc.item(), 4)}) + "\n")
                log.flush()
            if step % args.ckpt_every == 0:
                torch.save({"model_state_dict": dpc.model.state_dict(), "step": step},
                           out / f"encoder_step{step}.pt")
    if args.smoke:
        dt = time.time() - t0
        print(f"SMOKE: {iters} iters in {dt:.1f}s -> {dt / iters:.3f}s/step; "
              f"50k projection: {dt / iters * 50000 / 3600:.1f}h")


if __name__ == "__main__":
    main()
