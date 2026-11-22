#!/usr/bin/env python
"""Phase-1 Robotka step (M10): CTC pretrain on SYNTHETIC sentences with a
TRAINABLE encoder, then hand off encoder weights to real-data fine-tuning.

Vocab: <blank>=0 + joint map (lowercase keys, UPPER gloss match).
Val: last 20 synthetic sentences by id (deterministic holdout).
Saves full model state (encoder+head) for --init-encoder.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SynthCTCDataset(Dataset):
    def __init__(self, manifest: list[dict], out_dir: Path, vocab: dict[str, int]):
        self.items = []
        upper = {k.upper(): v for k, v in vocab.items()}
        for m in manifest:
            ids = []
            for g in m["glosses"]:
                key = g.strip().upper()
                if key not in upper:
                    break
                ids.append(upper[key])
            else:
                if ids:
                    self.items.append((out_dir / f"synth_{m['id']:04d}.npz", ids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, ids = self.items[i]
        f = np.asarray(np.load(p)["features"], dtype=np.float32)
        if f.shape[0] > 256:
            f = f[np.linspace(0, f.shape[0] - 1, 256).astype(int)]
        return torch.from_numpy(f.copy()), torch.tensor(ids, dtype=torch.long)


def collate(batch):
    feats, ids = zip(*batch)
    lens = torch.tensor([f.shape[0] for f in feats])
    T = int(lens.max().item())
    X = torch.zeros(len(feats), T, 168)
    M = torch.zeros(len(feats), T, dtype=torch.bool)
    for i, f in enumerate(feats):
        X[i, :f.shape[0]] = f
        M[i, :f.shape[0]] = True
    return X, M, torch.cat(ids), torch.tensor([len(t) for t in ids]), lens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/synthetic/manifest.json")
    parser.add_argument("--out-dir", default="data/processed/synthetic")
    parser.add_argument("--joint-map", default="data/processed/landmarks/joint_label_map.json")
    parser.add_argument("--base-checkpoint", default="models/multihead_v2/best.pt")
    parser.add_argument("--output-dir", default="models/ctc_synth")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--blank-penalty", type=float, default=0.1)
    parser.add_argument("--holdout", type=int, default=20)
    args = parser.parse_args()

    from src.models.ctc_head import HTMACTCModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joint = json.loads(Path(args.joint_map).read_text())
    vocab = {"<blank>": 0}
    for k, v in joint.items():
        vocab[k.upper()] = v + 1
    manifest = json.loads(Path(args.manifest).read_text())
    tr, va = manifest[:-args.holdout], manifest[-args.holdout:]
    out = Path(args.out_dir)

    model = HTMACTCModel(args.base_checkpoint, len(vocab) - 1, device=device,
                         train_encoder=True).to(device)
    logger.info(f"Trainable: {model.n_trainable:,}")

    def loader(rows, shuffle):
        ds = SynthCTCDataset(rows, out, vocab)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collate, num_workers=0), ds

    train_dl, train_ds = loader(tr, True)
    val_dl, val_ds = loader(va, False)
    logger.info(f"Synth train: {len(train_ds)}, val: {len(val_ds)}")
    if not len(train_ds) or not len(val_ds):
        raise ValueError("Empty synthetic split")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=1e-2)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs - 10, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[10])

    from scripts.train_ctc import greedy_wer
    idx2word = {v: k for k, v in vocab.items()}
    best, patience = 1e9, 0
    for epoch in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for X, M, flat, tlen, lens in train_dl:
            X, M, flat = X.to(device), M.to(device), flat.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                logp = model(X, M).log_softmax(2).permute(1, 0, 2)
                loss = criterion(logp.float(), flat, lens, tlen)
                if args.blank_penalty > 0:
                    with torch.no_grad():
                        valid = M.permute(1, 0).unsqueeze(-1).float()
                    blank_mass = (logp.float()[:, :, 0].exp().unsqueeze(-1) * valid).sum()
                    blank_mass = blank_mass / valid.sum().clamp_min(1)
                    loss = loss + args.blank_penalty * blank_mass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot += loss.item() * X.size(0)
            n += X.size(0)
        scheduler.step()
        w = greedy_wer(model, val_dl, device, idx2word)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | loss={tot/max(n,1):.4f} | synth-val WER={w*100:.1f}%")
        if w < best:
            best, patience = w, 0
            out.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "vocab": vocab,
                        "best_wer": best}, out / "best.pt")
        else:
            patience += 1
        if patience >= args.patience:
            break
    logger.info(f"Done. Best synth-val WER: {best*100:.1f}%")


if __name__ == "__main__":
    main()
