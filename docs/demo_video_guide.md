# Demo Video Guide (M13)

## Run the live demo
`run_live_demo.bat` (venv + webcam + Kokoro voice). NOT `run_demo.bat` (offline batch test).

## Setup (5 min)
- Plain background, face the camera, torso + both hands in frame (640×480).
- Even front lighting; avoid backlight and fast ceiling fans (motion confuses MediaPipe).
- Hold each sign ~1 second; pause between signs so the stability gate fires.
- First speech takes ~20 s (Kokoro weight load, one time, background thread).

## Scripted beats (verified end-to-end on this build)
1. Single words from 100%-recall classes: DOG, PEACE, BICYCLE, BUS, CAR, BOAT.
2. Full sentence: sign YOU → GOOD, wait for auto-speak → "you are good."
3. Press `D` to show the debug overlay (per-head tops, gate streak, latency).
4. Speak on demand with `Space`; `C` clears; `Q` quits.

## Fallbacks (if a sign misfires live)
- Fingerspell-free zone: stick to the scripted list above; OOV words cannot emit by design.
- If lighting degrades: move closer, slow down, re-hold the sign.
- If speech lags: first utterance is always slowest (~20 s); later ones ~0.13 s.
- Kill switch: `Q` quits; `run_demo.bat` (offline) reproduces numbers without a camera.

## Recording checklist
- 30–60 s, 1080p screen capture (window: "Sign Language Translation").
- Capture: (a) word emission + conf bar, (b) full sentence speech, (c) `D` debug panel.
- Keep raw footage + upload unlisted; link it in README and this file.
