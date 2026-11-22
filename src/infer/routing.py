from __future__ import annotations

import numpy as np
import torch


def unified_predict(
    model,
    seq: torch.Tensor,
    mask: torch.Tensor,
    sources: list[str],
    label_maps: dict[str, dict[int, str]],
    temperature: float = 1.0,
    temperatures: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[int, str]]:
    model.eval()
    with torch.inference_mode():
        parts: list[np.ndarray] = []
        united: dict[int, str] = {}
        offset = 0
        batched = None
        if hasattr(model, "predict_all"):
            try:
                batched = model.predict_all(seq, mask)
            except Exception:
                batched = None
        for src in sources:
            if batched is not None and src in batched:
                logits = batched[src]
            else:
                try:
                    logits = model(seq, mask, source=src)
                except TypeError:
                    logits = model(seq, mask)
            t = temperatures.get(src, temperature) if temperatures else temperature
            if t != 1.0:
                logits = logits / t
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            parts.append(probs)
            lm = label_maps.get(src, {})
            for i, p in enumerate(probs):
                united[offset + i] = lm.get(i, f"{src}:{i}")
            offset += len(probs)
    return np.concatenate(parts), united


def top_k(probs: np.ndarray, united: dict[int, str], k: int = 3):
    idx = np.argsort(probs)[-k:][::-1]
    return [(united.get(int(i), f"class_{i}"), float(probs[i])) for i in idx]
