from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

TEXT = (255, 255, 255)
DIM = (180, 180, 180)
SENTENCE = (255, 255, 0)
OK = (0, 210, 0)
WARN = (0, 165, 255)
LOW = (60, 60, 220)
PANEL_ALPHA = 0.55


def conf_color(conf: float) -> tuple[int, int, int]:
    if conf >= 0.7:
        return OK
    if conf >= 0.4:
        return WARN
    return LOW


def _blend_panels(img: np.ndarray, rects: list[tuple[int, int, int, int]]) -> None:
    overlay = img.copy()
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, PANEL_ALPHA, img, 1 - PANEL_ALPHA, 0, img)


def _wrap(text: str, width_chars: int = 72, max_lines: int = 2) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) <= width_chars:
            cur = trial
        else:
            lines.append(cur)
            cur = w
            if len(lines) == max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines and len(" ".join(lines)) < len(text):
        lines[-1] = lines[-1][:width_chars - 1] + "…"
    return lines or [""]


def draw_predictions(
    frame: np.ndarray,
    top3: list[tuple[str, float]],
    sentence: str,
    fps: float = 0.0,
    latency: Optional[dict[str, float]] = None,
    tts_playing: bool = False,
    tts_text: str = "",
    auto_speak_enabled: bool = True,
    debug: Optional[dict] = None,
    orb_level: float = 0.0,
    voice_name: str = "",
    draft: str = "",
    auto_speak_progress: float | None = None,
) -> np.ndarray:
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    rects = [(0, 0, 300, 118), (0, h - 140, w, h)]
    if debug:
        rects.append((w - 230, 96, w, 300))
    _blend_panels(overlay, rects)

    cv2.putText(overlay, f"FPS {fps:.0f}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 2)
    auto_txt = "AUTO-SPEAK ON" if auto_speak_enabled else "AUTO-SPEAK OFF"
    cv2.putText(overlay, auto_txt, (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200) if auto_speak_enabled else DIM, 1)

    if top3:
        word, conf = top3[0]
        color = conf_color(conf)
        cv2.putText(overlay, word, (12, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT, 2)
        bx0, bx1, by = 12, 240, 100
        cv2.rectangle(overlay, (bx0, by), (bx1, by + 8), (60, 60, 60), -1)
        cv2.rectangle(overlay, (bx0, by), (bx0 + int(conf * (bx1 - bx0)), by + 8), color, -1)
        for tick in (0.4, 0.7):
            tx = bx0 + int(tick * (bx1 - bx0))
            cv2.line(overlay, (tx, by - 2), (tx, by + 10), TEXT, 1)
        cv2.putText(overlay, f"{conf:.2f}", (bx1 + 8, by + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1)
        for i, (wd, cf) in enumerate(top3[1:3]):
            cv2.putText(overlay, f"{wd} {cf:.2f}", (250, 66 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1)

    cx, cy = w - 46, 46
    if tts_playing:
        r = int(14 + max(0.0, min(1.0, orb_level)) * 12)
        cv2.circle(overlay, (cx, cy), r + 6, (0, 100, 200), 2)
        cv2.circle(overlay, (cx, cy), r, (0, 180, 255), -1)
        if voice_name:
            cv2.putText(overlay, voice_name, (cx - 90, cy + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1)
    else:
        cv2.circle(overlay, (cx, cy), 14, (80, 80, 80), 1)

    for i, line in enumerate(_wrap(sentence)):
        cv2.putText(overlay, line, (12, h - 62 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, SENTENCE, 2)
    if draft and draft.strip().lower() != sentence.strip().lower():
        for i, line in enumerate(_wrap(draft)):
            cv2.putText(overlay, line, (12, h - 118 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1)
    cv2.putText(overlay, "[Space] Speak  [A] Auto  [C] Clear  [D] Debug  [Q] Quit",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1)
    if auto_speak_progress is not None:
        p = max(0.0, min(1.0, auto_speak_progress))
        cv2.rectangle(overlay, (12, h - 8), (12 + int(p * (w - 24)), h - 5), SENTENCE, -1)

    if debug:
        px0 = w - 230
        y = 116
        cv2.putText(overlay, "DEBUG", (px0 + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, WARN, 1)
        for key in ("heads", "gate", "latency_ms", "tts_ms", "fps_hist"):
            if key not in debug:
                continue
            y += 20
            cv2.putText(overlay, f"{key}: {debug[key]}", (px0 + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1)

    return overlay
