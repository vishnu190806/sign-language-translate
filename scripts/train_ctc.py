#!/usr/bin/env python
"""Train CTC head on frozen multi-head encoder (CSLTR sentences)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def greedy_wer(model, loader, device, idx2word) -> float:
    from scripts.continuous_baseline import wer

    model.eval()
    S = D = I = N = 0
    with torch.no_grad():
        for X, M, flat, tlen, lens in loader:
            logits = model(X.to(device), M.to(device))
            pred = logits.argmax(-1).cpu()
            off = 0
            for b in range(len(lens)):
                T = lens[b].item()
                seq = []
                for t in pred[b, :T].tolist():
                    if t == 0:
                        continue
                    if not seq or seq[-1] != t:
                        seq.append(t)
                hyp = [idx2word[i] for i in seq]
                ref_ids = flat[off:off + tlen[b].item()].tolist()
                off += tlen[b].item()
                ref = [idx2word[i] for i in ref_ids]
                s, d, i, n = wer(hyp, ref)
                S += s
                D += d
                I += i
                N += n
    return (S + D + I) / max(N, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output-dir", default="models/ctc")
    parser.add_argument("--blank-penalty", type=float, default=0.1)
    parser.add_argument("--init-head", default=None,
                        help="Path to CE-pretrained head checkpoint")
    parser.add_argument("--init-encoder", default=None,
                        help="Path to full CTC checkpoint (phase-1); loads encoder tensors")
    parser.add_argument("--train-encoder", action="store_true",
                        help="Unfreeze encoder blocks (norms stay frozen)")
    parser.add_argument("--npz-dir", default="data/processed/isl-csltr/landmarks")
    parser.add_argument("--word-dir", default="data/processed/isl-csltr/word_landmarks")
    parser.add_argument("--vocab", default="models/ctc/gloss_vocab.json")
    parser.add_argument("--base-checkpoint", default="models/multihead/best.pt")
    args = parser.parse_args()

    from src.data.csltr_dataset import CSLTRDataset, ctc_collate
    from src.models.ctc_head import HTMACTCModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab: dict[str, int] = json.loads(Path(args.vocab).read_text())
    idx2word = {v: k for k, v in vocab.items()}

    model = HTMACTCModel(args.base_checkpoint, len(vocab) - 1,
                         train_encoder=args.train_encoder).to(device)
    if args.init_head:
        model.head.load_state_dict(torch.load(args.init_head, map_location=device)["head_state_dict"])
        logger.info(f"Initialized head from {args.init_head}")
    if args.init_encoder:
        _full = torch.load(args.init_encoder, map_location=device, weights_only=False)
        _state = _full.get("model_state_dict", _full)
        _own = model.state_dict()
        _enc = {k: v for k, v in _state.items()
                if k.startswith("encoder.") and k in _own and _own[k].shape == v.shape}
        model.load_state_dict(_enc, strict=False)
        logger.info(f"Initialized encoder from {args.init_encoder} ({len(_enc)} tensors)")
    logger.info(f"Trainable params: {model.n_trainable:,}")
    import random as _random

    _random.seed(42)
    _train_sents = sorted({str(np.load(p, allow_pickle=True).get("sentence", "")).lower()
                           for p in Path(args.npz_dir).glob("*.npz")
                           if str(np.load(p, allow_pickle=True).get("split", "")) == "train"})
    _val_sents = set(_random.sample(_train_sents, k=min(12, len(_train_sents) // 5)))
    logger.info(f"Held-out val sentences: {len(_val_sents)}")
    train_ds = CSLTRDataset(args.npz_dir, "train", vocab, augment=True,
                            val_sentences=_val_sents,
                            word_dir=args.word_dir)
    val_ds = CSLTRDataset(args.npz_dir, "train", vocab,
                          val_sentences=_val_sents, only_val=True)
    if len(val_ds) == 0:
        raise ValueError("Empty validation set: cannot tune or early-stop without val samples")
    if len(train_ds) == 0:
        raise ValueError("Empty training set: check --npz-dir and vocab coverage")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=ctc_collate, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=ctc_collate, num_workers=0)
    logger.info(f"Train: {len(train_ds)}, val: {len(val_ds)}")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=1e-2)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
    cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs - 10, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[10])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best, patience = 1e9, 0
    for epoch in range(args.epochs):
        model.train()
        train_ds.set_epoch(epoch)
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
                    blank_mass = (logp.float()[:, :, 0].exp().unsqueeze(-1) * valid).sum() / valid.sum().clamp_min(1)
                    loss = loss + args.blank_penalty * blank_mass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tot += loss.item() * X.size(0)
            n += X.size(0)
        scheduler.step()
        w = greedy_wer(model, val_dl, device, idx2word)
        logger.info(f"Epoch {epoch+1}/{args.epochs} | loss={tot/max(n,1):.4f} | val WER={w*100:.1f}%")
        if w < best:
            best, patience = w, 0
            torch.save({"head_state_dict": model.head.state_dict(), "vocab": vocab,
                        "val_wer": w, "epoch": epoch}, out / "best_ctc.pt")
            logger.info(f"New best saved: WER {w*100:.1f}%")
        else:
            patience += 1
        if patience >= args.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    logger.info(f"Done. Best val WER: {best*100:.1f}%")


if __name__ == "__main__":
    main()
