from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import load_config, Config
from src.features.extractor import LandmarkExtractor
from src.features.smooth import LandmarkSmoother
from src.models.builder import build_model
from src.infer.buffer import RingBuffer
from src.infer.gate import StabilityGate
from src.infer.sentence import SentenceAssembler, SentenceConfig
from src.infer.profiler import LatencyProfiler
from src.app.camera import ThreadedCamera
from src.app.overlay import draw_predictions
from src.tts.factory import create_tts_engine
from src.tts.playback import AudioPlaybackManager

logger = logging.getLogger(__name__)

INFERENCE_STRIDE = 2
CAMERA_WARMUP_FRAMES = 10


def load_label_map(path: str | Path) -> dict[int, str]:
    with open(path) as f:
        raw = json.load(f)
    return {int(v): k for k, v in raw.items()}


def warmup_model(model: torch.nn.Module, device: torch.device, config: Config) -> None:
    dummy = torch.randn(1, config.processing.sequence_length, config.model.input_dim, device=device)
    dummy_mask = torch.ones(1, config.processing.sequence_length, dtype=torch.bool, device=device)
    with torch.inference_mode():
        _ = model(dummy, dummy_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    logger.info("Model warmup complete")


def warmup_camera(camera: ThreadedCamera, frames: int = CAMERA_WARMUP_FRAMES) -> None:
    for _ in range(frames):
        camera.read()
        time.sleep(0.01)
    logger.info(f"Camera warmup complete ({frames} frames flushed)")


def run_demo(
    config_path: str = "configs/default.yaml",
    override_path: str | None = None,
    checkpoint: str = "models/multihead_v2/best.pt",
    no_tts: bool = False,
    offline: bool = False,
) -> None:
    config = load_config(config_path, override_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    label_map_path = Path(checkpoint).parent / "label_map.json"
    if not label_map_path.exists():
        label_map_path = Path(config.paths.checkpoints) / "label_map.json"
    if not label_map_path.exists():
        label_map = {}
        for i in range(config.model.num_classes):
            label_map[i] = f"class_{i}"
    else:
        label_map = load_label_map(label_map_path)

    model = build_model(config.model).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    mh_sources: list[str] | None = None
    mh_label_maps: dict[str, dict[int, str]] = {}
    if isinstance(ckpt.get("sources"), list):
        from src.models.htma import HTMAConfig, MultiHeadHTMAClassifier

        mh_sources = list(ckpt["sources"])
        state = ckpt["model_state_dict"]
        head_classes: dict[str, int] = dict(ckpt.get("head_classes") or {})
        for src in mh_sources:
            for suffix in ("heads." + src + ".3.weight", "heads." + src + ".weight"):
                if suffix in state:
                    head_classes[src] = state[suffix].shape[0]
                    break
        htm = HTMAConfig(input_dim=config.model.input_dim, embedding_dim=config.model.embedding_dim,
                         num_blocks=config.model.htma_blocks, num_heads=config.model.num_heads,
                         dropout=config.model.dropout, num_classes=config.model.num_classes)
        model = MultiHeadHTMAClassifier(htm, sources=tuple(mh_sources),
                                        head_classes=head_classes).to(device)
        own = model.state_dict()
        pruned = {k: v for k, v in ckpt["model_state_dict"].items()
                  if k in own and own[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(pruned, strict=False)
        if missing:
            logger.info(f"Multi-head load: {len(missing)} keys initialized fresh "
                        f"({sorted(missing)[:4]})")
        model.eval()
        maps_sidecar = Path(checkpoint).parent / "label_maps.json"
        if maps_sidecar.exists():
            import json as _json

            raw_maps = _json.loads(maps_sidecar.read_text())
            for src, mp in raw_maps.items():
                mh_label_maps[src] = {int(v): k for k, v in _json.loads(Path(mp).read_text()).items()}
        if not mh_label_maps:
            mh_label_maps = {src: label_map for src in mh_sources}
        logger.info(f"Multi-head model loaded from {checkpoint} (default head: {ckpt['sources'][0]})")
    elif "full_state_dict" in ckpt or "adapter_state_dict" in ckpt:
        from src.models.dst_adapter import DSTAdapter

        adapter_dim = int(ckpt.get("adapter_dim", 64))
        model = DSTAdapter(model, adapter_dim=adapter_dim).to(device)
        state = ckpt.get("full_state_dict", ckpt.get("adapter_state_dict", {}))
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            model.model.load_state_dict(ckpt.get("model_state_dict", {}))
        logger.info(f"DSTAdapter model loaded from {checkpoint} (adapter_dim={adapter_dim})")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Model loaded from {checkpoint}")
    model.eval()

    temperature = 1.0
    temperatures: dict[str, float] = {}
    temp_path = Path(checkpoint).parent / "temperature.json"
    if temp_path.exists():
        try:
            import json as _json

            temp_cfg = _json.loads(temp_path.read_text())
            temperatures = {k: float(v) for k, v in temp_cfg.get("temperatures", {}).items()}
            src_key = temp_cfg.get("default_source", "include")
            temperature = float(temperatures.get(src_key, 1.0))
            logger.info(f"Temperature scaling enabled: {temperatures} ({temp_path})")
        except Exception as e:
            logger.warning(f"Ignoring {temp_path}: {e}")

    warmup_model(model, device, config)

    smoother = LandmarkSmoother(feature_dim=config.model.input_dim)

    tts = None
    playback = None
    if not no_tts:
        try:
            voice_dir = config.tts.voice_dir
            if offline:
                voice_dir = str(Path("demo/piper_voice").resolve())
            tts = create_tts_engine(
                config.tts.engine,
                voice=config.tts.voice,
                sample_rate=config.tts.sample_rate,
                length_scale=config.tts.length_scale,
                sentence_silence=config.tts.sentence_silence,
                voice_dir=voice_dir,
            )
            playback = AudioPlaybackManager(
                tts,
                max_queue_size=config.tts.max_queue_size,
                device=config.tts.device,
            )
            if tts.is_available:
                playback.start()
                logger.info("TTS playback started")
            else:
                logger.warning("TTS engine not available — speech disabled")
        except Exception as e:
            logger.warning(f"TTS initialization failed: {e} — speech disabled")

    ring_buffer = RingBuffer(
        config.processing.sequence_length,
        config.model.input_dim,
    )
    gate = StabilityGate(
        threshold=config.processing.stability_gate.threshold,
        consecutive_frames=config.processing.stability_gate.consecutive_frames,
        cooldown_frames=config.processing.stability_gate.cooldown_frames,
        agreement_ratio=config.processing.stability_gate.agreement_ratio,
        min_confidence_delta=config.processing.stability_gate.min_confidence_delta,
    )
    sentence_config = SentenceConfig(
        auto_speak_timeout=3.0,
        max_words=20,
    )
    assembler = SentenceAssembler(config=sentence_config)
    profiler = LatencyProfiler(ema_alpha=0.1)

    _polish_engine = None

    def speak_sentence(text: str) -> None:
        if playback is None:
            return

        def _work() -> None:
            nonlocal _polish_engine
            try:
                if _polish_engine is None:
                    from pathlib import Path as _P

                    from src.text.polish import RulesOnlyEngine, TransformersPolishEngine

                    qp = _P("models/qwen15b")
                    if qp.exists() and any(qp.glob("*.safetensors")):
                        _polish_engine = TransformersPolishEngine(str(qp))
                        logger.info("LLM polish enabled (Qwen2.5-1.5B)")
                    else:
                        _polish_engine = RulesOnlyEngine()
                        logger.info("Rules-only sentence polish (no LLM found)")
                final, how = _polish_engine.polish(text.split())
                logger.info(f"Speaking ({how}): {final}")
                playback.speak(final)
            except Exception as e:
                logger.warning(f"Polish failed, speaking raw: {e}")
                playback.speak(text)

        import threading as _threading

        _threading.Thread(target=_work, daemon=True).start()

    camera = None
    extractor = None
    camera = ThreadedCamera(source=0, backend="dshow")
    extractor = LandmarkExtractor(
        hand_model_path=f"{config.paths.assets}/hand_landmarker.task",
        pose_model_path=f"{config.paths.assets}/pose_landmarker.lite.task",
    )

    if playback is not None:
        assembler.set_ready_callback(lambda s: speak_sentence(s))

    top3: list[tuple[str, float]] = []
    head_tops: dict[str, str] = {}
    frame_idx = 0
    show_debug = False
    fps_hist: list[float] = []

    try:
        with camera:
            warmup_camera(camera)

            while True:
                profiler.begin()

                profiler.start("capture")
                ret, frame = camera.read()
                profiler.stop("capture")

                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                profiler.start("landmarks")
                extractor.detect_async(frame)
                features = extractor.get_latest_features()
                profiler.stop("landmarks")

                if features is None:
                    features = np.zeros(config.model.input_dim, dtype=np.float32)

                if np.any(features):
                    features = smoother.smooth(features)
                else:
                    smoother.reset()
                ring_buffer.add(features)

                if ring_buffer.is_full and frame_idx % INFERENCE_STRIDE == 0:
                    profiler.start("infer")
                    seq, mask = ring_buffer.get_sequence()
                    seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(device)
                    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)

                    with torch.inference_mode():
                        if mh_sources is not None:
                            from src.infer.routing import top_k as _top_k
                            from src.infer.routing import unified_predict as _predict

                            probs, united = _predict(model, seq_tensor, mask_tensor,
                                                     mh_sources, mh_label_maps, temperature,
                                                     temperatures or None)
                        else:
                            logits = model(seq_tensor, mask_tensor)
                            if temperature != 1.0:
                                logits = logits / temperature
                            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                            united = label_map
                    profiler.stop("infer")

                    emitted_idx = gate.update(probs)
                    if emitted_idx is not None and emitted_idx in united:
                        word = united[emitted_idx]
                        if assembler.add_word(word):
                            speak_sentence(assembler.get_sentence())
                            assembler.clear()
                        logger.info(f"Emitted: {word}")

                    if mh_sources is not None:
                        top3 = _top_k(probs, united, k=3)
                        head_tops = {}
                        off = 0
                        for _src in mh_sources:
                            _n = len(mh_label_maps.get(_src, {})) or (
                                len(probs) - off) // max(1, len(mh_sources))
                            _seg = probs[off:off + _n]
                            _bi = int(np.argmax(_seg))
                            head_tops[_src] = (
                                f"{mh_label_maps.get(_src, {}).get(_bi, _src)} {_seg[_bi]:.2f}")
                            off += _n
                    else:
                        head_tops = {}
                        top_indices = np.argsort(probs)[-3:][::-1]
                        top3 = [
                            (label_map.get(int(i), f"class_{i}"), float(probs[i]))
                            for i in top_indices
                        ]
                elif not ring_buffer.is_full:
                    top3 = []

                assembler.update()

                tts_playing = playback.is_playing if playback else False
                tts_text = playback.current_text if playback else ""
                tts_level = playback.level if playback else 0.0
                tts_voice = getattr(tts, "voice", "") if tts else ""
                try:
                    from src.text.rules import gloss_to_sentence as _g2s

                    _words = assembler.words
                    _live_draft = _g2s(_words) if _words else ""
                except Exception:
                    _live_draft = ""
                try:
                    _elapsed = assembler.get_time_since_last_word()
                    _speak_progress = (_elapsed / 3.0
                                       if assembler.words and assembler.auto_speak_enabled
                                       else None)
                except Exception:
                    _speak_progress = None

                profiler.end_frame()
                frame_idx += 1
                fps_hist.append(profiler.fps)
                if len(fps_hist) > 60:
                    fps_hist.pop(0)

                debug_info = None
                if show_debug:
                    gs = gate.status()
                    debug_info = {
                        "heads": head_tops,
                        "gate": (f"streak={gs['streak']} stab={gs['stability']:.2f} "
                                 f"cd={gs['cooldown']} last={gs['last_emitted']}"),
                        "latency_ms": profiler.summary(),
                        "tts_ms": round(playback.last_speak_ms, 1) if playback else 0.0,
                        "fps_hist": [round(v, 1) for v in fps_hist[-12:]],
                    }

                display = draw_predictions(
                    frame, top3, assembler.get_sentence(),
                    profiler.fps, profiler.summary(),
                    tts_playing=tts_playing,
                    tts_text=tts_text,
                    auto_speak_enabled=assembler.auto_speak_enabled,
                    debug=debug_info,
                    orb_level=tts_level,
                    voice_name=str(tts_voice),
                    draft=_live_draft,
                    auto_speak_progress=_speak_progress,
                )
                cv2.imshow("Sign Language Translation", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" ") and playback is not None:
                    sentence = assembler.get_sentence()
                    if sentence:
                        speak_sentence(sentence)
                elif key == ord("a"):
                    assembler.set_auto_speak(not assembler.auto_speak_enabled)
                    logger.info(f"Auto-speak: {'ON' if assembler.auto_speak_enabled else 'OFF'}")
                elif key == ord("c"):
                    assembler.clear()
                    gate.reset()
                    ring_buffer.reset()
                elif key == ord("d"):
                    show_debug = not show_debug
                    logger.info(f"Debug overlay: {'ON' if show_debug else 'OFF'}")
                elif key == 8:
                    assembler.backspace()

    finally:
        logger.info(f"Latency summary: {profiler.summary()}")
        if extractor is not None:
            extractor.close()
        if playback is not None:
            playback.stop(wait=True, timeout=2.0)
        cv2.destroyAllWindows()
        logger.info("Demo stopped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sign Language Translation Demo")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None, help="Override config (e.g. configs/isl.yaml)")
    parser.add_argument("--checkpoint", default="models/multihead_v2/best.pt")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS output")
    args = parser.parse_args()
    run_demo(args.config, args.override, args.checkpoint, no_tts=args.no_tts)
