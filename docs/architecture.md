# System Architecture and Data Flow

This document describes the architecture and data flow of the Modular Sign Language Translation System.
The system is ISL-only by decision (2026-10-03) and modular within ISL, keeping INCLUDE, ISL500, and CSLTR sources behind one interface.

## Overview

## High-Level Architecture
```
Webcam (30 FPS)
  → OpenCV capture (thread, always-latest frame)
  → MediaPipe Tasks: HandLandmarker (2 hands) + PoseLandmarker (lite) [CPU, async]
  → Frame feature builder (~168 dims: both hands' landmarks + hand-presence flags + upper-body pose subset)
  → Normalization (per-frame: center on hand/wrist, scale by hand size → signer/distance invariant)
  → Ring buffer of last T frames (T=32–64, mask missing hands)
  → Model: HTMA + attention → N-way softmax (N = number of signs in current dataset)
  → Stability gate (top-1 conf ≥ τ for k consecutive frames + cooldown) → emit word
  → Sentence bar (append word, dedup consecutive, backspace via gesture/keypress)
  → TTS: Piper (offline) → speakers
```

## Component Details

### 1. Video Capture
- **Technology**: OpenCV with `CAP_DSHOW` backend on Windows to avoid MSMF startup lag.
- **Design**: Threaded capture to always provide the latest frame, reducing latency.
- **Warmup**: Flushes initial frames at startup to stabilize auto-exposure and white balance.
- **Output**: Raw video frames (BGR format) at configurable resolution (default: 640x480).

### 2. Landmark Extraction (MediaPipe Tasks API)
- **Models Used**:
  - `HandLandmarker`: Detects 21 landmarks per hand (x, y, z) plus handedness.
  - `PoseLandmarker` (lite): Detects 33 pose landmarks; we use an upper-body subset (13 landmarks).
- **Mode**: LIVE_STREAM with async detection and result callbacks for monotonic timestamps.
- **CPU Performance**: ~15-25 ms per frame on i9-14900HX (no GPU delegate needed on Windows).
- **Output**: Per-frame landmark vectors (168 dimensions) as follows:
  - Left hand: 21 × 3 = 63 floats + 1 presence flag
  - Right hand: 21 × 3 = 63 floats + 1 presence flag
  - Upper-body pose: 13 × 3 = 39 floats + 1 presence flag
  - **Total**: (63+1) + (63+1) + (39+1) = 168 dims

### 3. Feature Builder and Normalization
- **Input**: Raw landmark vectors from MediaPipe (168 dims).
- **Normalization per hand**:
  - Translate so wrist = origin (0,0,0).
  - Scale by distance from wrist to middle-MCP (metacarpophalangeal joint) to achieve invariance to distance and body size.
- **Pose normalization**:
  - Translate so mid-shoulder = origin.
  - Scale by shoulder width (distance between left and right shoulders) for invariance to body size.
- **Missing Hand Handling**: If a hand is not detected, its landmarks are set to zero and the presence flag is 0 (masked in the model).

### 4. Temporal Buffering
- **Ring Buffer**: Stores the last T frames (T configurable, default 32-64) of normalized landmark features.
- **Masking**: Frames with missing hands (presence flag 0) are masked in the model to avoid using invalid data.
- **Uniform Temporal Sampling (Optional)**: As a cheaper variant, we can uniformly sample T frames from the video instead of using a ring buffer.

### 5. HTMA (Head-Token Mixing Attention) Model
Based on the CVPR 2026 Workshop paper: "Isolated Sign Language Recognition via MediaPipe Landmarks: A Case Study On Indian Sign Language".

- **Input**: Tensor of shape (B, T, 168) where B is batch size, T is sequence length.
- **Steps**:
  1. Linear projection to embedding dimension D (default 256).
  2. Add Absolute Positional Encoding (APE) and Rotary Positional Embedding (RoPE).
  3. Pass through one or more HTMA blocks:
      - Local Conv1D-based temporal mixing (kernel size=3) on the value vectors.
      - Multi-token convolution in the attention score space (simulating multi-token attention).
      - Standard multi-head attention (H=8 heads) with APE+RoPE.
      - Layer normalization and residual connections.
  4. Output sequence of features (B, T, D).
  5. Attention pooling: Learnable query to aggregate over the temporal dimension → fixed-size vector (B, D).
  6. MLP head: Two layers (D → D → N) with GELU activation and dropout, where N is the number of sign classes.
  7. N-way softmax to produce class probabilities.

- **Parameter Count**: 2,026,056 (2 HTMA blocks, 8 heads, D=256, 264 classes, `results/full_dataset/metrics.json`).
- **Measured Results**:
  - INCLUDE full test (478 samples, 264 classes): top-1 **81.80%**, top-5 95.40%, macro 83.33% (`results/full_dataset/metrics.json`).
  - INCLUDE-50 historical: top-1 100% val (early tuning checkpoint; superseded by full-dataset results above).

### 6. Latency Profiling and Warmup Strategies
- **LatencyProfiler**: Tracks per-stage timing with EMA smoothing for:
  - Capture: Time to read frame from threaded camera
  - Landmarks: Time for MediaPipe async detection + feature extraction
  - Infer: Time for model forward pass (including tensor conversion)
  - Total: End-to-end frame processing time
- **Model Warmup**: Dummy forward pass at startup to eliminate cold-start latency from CUDA context initialization and kernel compilation
- **Camera Warmup**: Flush initial frames to stabilize auto-exposure, white balance, and focus
- **Inference Stride**: Runs model inference every Nth frame (default N=2) while maintaining full-rate landmark extraction to balance accuracy and compute load

### 7. Stability Gate for Word Emission
- **Purpose**: To prevent spurious word predictions and ensure stable output.
- **Logic**:
  - Maintain a running mean of the softmax probabilities over the last k frames (k configurable, default 12).
  - Emit a word W when the running-mean probability for W exceeds a threshold τ (default 0.50, tuned by gate sweep `results/gate_sweep/sweep.jsonl`) with 60% agreement and confidence delta ≥0.08.
  - After emission, enter a cooldown period (default 30 frames) to prevent immediate re-emission of the same word.
- **Idle State**: When no hands are detected for a configurable number of frames, the buffer is reset.

### 8. Augmentation Pipeline (`src/features/augment_pipeline.py`)
- **Purpose**: Unified interface for applying geometry-aware, temporal, and landmark-specific augmentations during training.
- **Architecture**:
  - **Geometry Augmentations** (6 types, custom heuristics): CameraDepthVariation, HVShift, HandSizeVariation, ViewpointRotation, FingerArticulation, ElbowDisplacement.
  - **Temporal Augmentations** (3+ types): FramePermutation, SpeedVariation, RandomFrameDrop — applied to non-relevant frames to preserve sign meaning.
  - **Landmark Augmentations** (4 types): SpatialRotation, RandomTranslation, FrameFusion, RandomKeypointRemoval — landmark-space robustness following KD-MSLRT (Li et al., AAAI 2025, https://ojs.aaai.org/index.php/AAAI/article/view/35037).
  - **Pose Dropout** (`random_pose_dropout`, p=0.5): zeroes the pose block to unify pose-present (INCLUDE) and pose-absent (ISL500) regimes — added after measurement showed pose-manifold collapse (see `docs/RESULTS.md` joint-softmax section).
- **Features**:
  - Configurable augmentation probabilities and severity ranges
  - Sequential or parallel application modes
  - Statistics tracking (augmentation counts, severity distributions) logged to JSON
  - Visualization tools for debugging augmented samples

### 9. Test-Time Adaptation (TTA) — SAME Adapter (`src/models/same_adapter.py`)
- **Purpose**: Enable unsupervised adaptation during inference to bridge domain gap between signers.
- **Based on**: SAME framework (Yang et al., ACL 2026) — first TTA method specifically for sign language translation.
- **Architecture**:
  - **MoE Module**: Lightweight experts (LoRA-based, rank=8) inserted after HTMA encoder layers
  - **Signer-Aware Gating**: Network conditioned on signer features (hand statistics, pose statistics) to route inputs to specialized experts
  - **Unsupervised Objectives**:
    - Entropy minimization — Encourage confident predictions
    - Minimum class confusion — Reduce inter-class confusion
    - Pseudo-label supervision — Prevent drifting from source model
- **Key Metrics (paper-reported, SLT/BLEU task — not isolated top-1)**: lightweight MoE experts; unsupervised entropy + class-confusion + pseudo-label objectives. Our TTA code (`src/models/same_adapter.py`) has now been measured episodically: **per-actor deltas −0.0015 / +0.0014, mean −2.2e-05** (`results/tta_episodic.json`) — the adapter is live (α = sigmoid(0.5) ≈ 0.62, not a no-op) but gives **no reliable per-signer gain**. Note the paper targets sign language *translation* (seq2seq); we evaluate isolated classification, which is outside its reported setting.

### 10. Parameter-Efficient Fine-Tuning — DSTAdapter (`src/models/dst_adapter.py`)
- **Purpose**: Fine-tune with minimal parameters while preserving backbone knowledge.
- **Based on**: DSTAdapter (Tian et al., **IJMLC 2026**, 17:219, DOI `10.1007/s13042-026-03002-x`) — divided spatial-temporal adapter for sign language recognition on a frozen CLIP backbone. Preprint year was 2025; version of record is 2026.
- **Architecture**:
  - **Spatial Adapter**: (1,3,3) depthwise convolutions for hand shape features
  - **Temporal Adapter**: (3,1,1) depthwise convolutions for gesture dynamics
  - **Channel-Aware Fusion**: Dynamic weighting between adapter and backbone features
- **Key Metrics (measured in this repo)**: ~11.6% trainable parameters (266,240/2,292,296); +3.98pp on weak base (`best.pt` 75.73%→79.71% test); parity −0.84pp on champion (`baseline.pt` 81.80%→80.96%). Paper-reported transfer-learning figures (memory/speed) not independently verified here.

### 11. Knowledge Distiller (`src/train/distiller.py`)
- **Purpose**: Train lightweight student model from larger teacher while preserving accuracy.
- **Based on**: KD-MSLRT (Li et al., AAAI 2025) — 3D-to-1D knowledge distillation for lightweight SLR.
- **Architecture**:
  - **Teacher Model**: Larger HTMA (4 blocks, D=512) trained on augmented data
  - **Student Model**: Current HTMA (2 blocks, D=256)
  - **Distillation Loss**: KL divergence between teacher and student logits (temperature-scaled soft targets)
  - **Feature Distillation**: Match intermediate feature maps between teacher and student
- **Key Metrics (paper-reported: 12.93 MB TFLite; our repo)**: distiller implemented (`src/train/distiller.py`, T=4.0, alpha=0.7); measured ablation showed −7.11pp with the then-current teacher, so distillation is parked until a stronger teacher exists.

### 12. Multi-Scale Feature Fusion in HTMA
- **Purpose**: Combine hand and pose features at multiple temporal scales for richer representations.
- **Location**: Integrated into `src/models/htma.py`
- **Fusion Methods**:
  - **Early Fusion** (default): Concatenate hand (128 dims) and pose (40 dims) at input level → 168-dim vector
  - **Late Fusion**: Separate encoders for hand and pose, fuse at attention pooling stage
  - **Multi-Scale Fusion**: Fuse features at multiple HTMA block outputs, capturing both low-level and high-level representations
- **Hand Features**: 2×(21×3+1) = 128 dims (both hands' landmarks + presence flags)
- **Pose Features**: 13×3+1 = 40 dims (upper-body pose + presence flag)

### 13. Per-Signer Normalization (`src/features/normalize.py`)
- **Purpose**: Normalize features per signer to improve cross-signer generalization.
- **Components**:
  - **SignerStatistics**: Stores per-signer mean and std computed during training
  - **SignerEstimator**: Estimates signer identity from first few frames of feature sequence
  - **SignerNormalization**: Applies signer-specific normalization using estimated signer statistics
- **Workflow**:
  1. Compute per-signer statistics (mean, std) during training and store in model checkpoint
  2. At inference, estimate signer from first few frames using SignerEstimator
  3. Apply signer-specific normalization to subsequent frames
- **Status (targets, unmeasured)**: signer estimation accuracy and domain-gap reduction figures are design targets, not measured results. Per-signer normalization code exists; real-world per-signer variance is measured instead (`results/real_world_test*.json`).

### 16. Multi-Head Co-Training (`src/models/htma.py: MultiHeadHTMAClassifier`, `scripts/train_multihead.py`)
- **Purpose**: One checkpoint serving multiple data regimes. Shared HTMA encoder + one MLP head per source (`include`, `isl500`).
- **Based on**: Logos multi-dataset co-training (Ovodov et al., 2025, https://arxiv.org/html/2505.10481v1) — joint co-training with per-language heads beats pretrain→fine-tune and flat union softmax.
- **Why**: Flat union-softmax training collapsed INCLUDE to ~0.7% (measured; pose-manifold + source-skew mechanism in `docs/RESULTS.md`). Separate heads absorb per-source offsets while the encoder learns shared motion primitives.
- **Measured**: INCLUDE test 77.20%, ISL500 unseen-users 61.82%, real-world overall 41.0% / IV 68.0% (`results/multihead_by_source.json`, `results/real_world_test_multihead.json`). Deployment model (`models/multihead/best.pt`, 2,292,565 params).
- **Training**: Alternating equal source batches (strict 50/50), shared AdamW, best on mean per-source val (85.62%). Demo and real-world eval route per clip/dataset to the matching head.

### 17. ISL500 Data Pipeline (`scripts/download_isl500.py`, `scripts/convert_isl500.py`)
- **Dataset**: ISL500/ISL-DATA (community, https://huggingface.co/datasets/ISL500/ISL-DATA) — 500–1000 words, 15 signers, MP4 + MediaPipe `.h5` + MMPose, research/academic license, gated (HF login + terms).
- **Selection**: 180/263 INCLUDE words overlap (68%, `results/vocab_overlap.json`); 26,796 landmark files downloaded (~2h), 26,483 converted to 168-d `.npz` (`data/processed/landmarks/isl500/`), 0 failures. `.h5` schema: `(T, 2, 21, 3)` hands-only, no pose, 30fps.
- **Joint set**: `data/processed/landmarks/joint/` — 30,767 NTFS-hardlinked files (4,284 INCLUDE + 26,483 ISL500, zero extra disk), normalized 263-class map (`joint_label_map.json`, contiguous 0–262), signer-disjoint splits (ISL500 users 001–011 train / 012–013 val / 014–015 test).

### 18. Sentence Assembly
- **State Machine**:
  - On word emission: append the word to the sentence bar.
  - Deduplication: Do not append if the emitted word is the same as the last word in the sentence (configurable).
  - Backspace/Clear: Supported via keyboard (Backspace, C) or a predefined gesture (to be defined).
- **Output**: The current assembled sentence is displayed on-screen.

### 19. Text-to-Speech (Piper Engine)
- **Technology**: Piper 1.8 (`piper-tts` PyPI package) for offline, high-quality speech synthesis.
- **Design**:
  - Non-blocking: Synthesis and playback handled by a worker queue so that the recognition pipeline never stalls.
  - Voice: Default `en_US-lessac-medium` (no en_IN voice exists; verified at implementation).
  - Audio Playback: Via `sounddevice` for low-latency output.
- **Future A/B candidate — Kokoro-82M (researched 2026-10-03, not integrated):**
  Apache-2.0, 82M params, ~327MB weights (HF `hexgrad/Kokoro-82M`), 54 preset voices
  (`af_heart` most natural), 24kHz, CPU-fast; TTS Arena #1 at launch. Fits beside
  everything (torch package, not onnx — onnx variant has an espeak data-path bug).
  Requirements: `pip install kokoro soundfile` + espeak-ng `.msi` on PATH (+
  `PHONEMIZER_ESPEAK_LIBRARY` DLL workaround if phonemizer can't find it).
  Caveats: English-centric training (narrative > conversational), OOD words fall back
  to espeak, no voice cloning. Rule: implement `KokoroEngine` behind `TTSEngine`,
  A/B on identical C2 sentences, ship only on measured win. Blocker is a user step:
  espeak-ng `.msi` install (needs elevation).
- **A/B measured 2026-10-03 (`results/kokoro_ab.json`, `src/tts/kokoro_engine.py`, 5 tests):**
  `pip install kokoro soundfile` worked; bundled espeakng-loader sufficed (no system
  espeak needed for tested sentences). `af_heart`: cold first-synthesis ~19s (one-time
  weight load + kernel autotune), **steady-state 0.13s for 4.2s audio** — same league as
  Piper (lessac 0.08s, amy 0.21s). Early 2–5s readings were first-inference autotune,
  not real cost. Streaming added to both engines (`stream_chunks`; Piper TTFA 0.03s,
  Kokoro single-chunk). Sample: `results/kokoro_sample.wav` — listen and judge.
  Latency objection withdrawn: decision is voice quality only.
  **Decision 2026-10-03: Kokoro wins — demo default** (`configs/default.yaml`:
  engine kokoro, voice af_heart). Rejected amy/libritts_r voices removed (135MB);
  lessac + Piper retained as fallback.
- **Flow**:
  - When a word or sentence is ready for speaking (triggered by Space key or automatically after a pause), the text is sent to the TTS engine.
  - The engine synthesizes audio samples (int16 or float32) and places them in a playback queue.
  - A separate thread consumes the queue and plays the audio via the default output device.

## Data Flow Details

### Offline Preprocessing (Training Data Preparation)
1. **Download**: Obtain ISL dataset videos (INCLUDE) via Zenodo or HuggingFace.
2. **Landmark Extraction**: Run MediaPipe Tasks API on each video to extract landmark sequences.
   - Output: Per-video landmark arrays saved as compressed `.npz` files (float16 to save space).
   - Metadata: Save word label, split (train/val/test), and video length for each video.
3. **Storage**: The processed landmarks and metadata are stored in `data/processed/landmarks/isl/` and `data/raw/isl/` respectively.

### Training Pipeline
1. **DataLoader**: PyTorch `Dataset` loads the `.npz` files on-the-fly.
   - Supports variable-length sequences with padding and attention masking.
   - BalancedBatchSampler for per-class balanced training.
   - Uniform temporal sampling to fixed length (32 frames).
2. **Augmentations** (7 transforms, applied in landmark space):
   - Temporal jitter (±2 frames shift)
   - Temporal crop (70-100% of sequence)
   - Temporal dropout (15% frames zeroed)
   - Gaussian noise (std=0.01)
   - Random hand dropout (10% probability per hand per frame)
   - Spatial scale (0.9-1.1x)
   - Random translation (±0.05 units)
3. **Training Loop**:
   - Mixed precision (AMP with bfloat16) for RTX 4070.
   - Optimizer: AdamW (lr=5e-4, weight_decay=0.01).
   - Scheduler: Linear warmup (10%) + CosineAnnealingLR (eta_min=1e-6).
   - Label smoothing: 0.1
   - Gradient clipping: max_norm=1.0
   - Early stopping: patience=20 on validation top-1 accuracy.
   - Checkpointing: Best validation model saved as `best.pt` with `label_map.json`.

### Live Inference Pipeline
1. **Capture**: Threaded OpenCV captures frames from webcam (with warmup flush).
2. **Landmark Extraction**: MediaPipe Tasks API processes each frame asynchronously.
3. **Feature Building**: Convert raw landmarks to the 168-dimensional feature vector per frame.
4. **Normalization**: Apply per-hand and pose normalization as described.
5. **Buffering**: Add normalized frame to ring buffer; mask if hands are missing.
6. **Latency Profiling**:
   - Begin frame timing
   - Profile capture stage
   - Profile landmarks stage
   - Profile inference stage (when stride condition met)
   - End frame timing and update EMA statistics
7. **Model Inference** (every Nth frame):
   - Convert buffer to tensor batch (B=1, T, F).
   - Run through HTMA model in `torch.inference_mode()` with AMP.
   - Obtain logits → softmax → class probabilities.
8. **Stability Gate**: Update gate with probabilities; emit word if conditions met.
9. **Sentence Assembly**: Add emitted word to sentence bar (with deduplication).
10. **Display**: Render frame with FPS, latency breakdown (capture/landmarks/infer/total), top-3 predictions, and sentence.
11. **TTS Trigger**: On Space key press or after pause in speaking, send sentence to Piper engine.
12. **Audio Playback**: Non-blocking playback via sounddevice worker queue.

## Modularity and Extensibility
- **Dataset Registry**: ISL sources (INCLUDE, ISL500, CSLTR) are added as dataset handlers adhering to the abstract base class (download, metadata extraction, label mapping, split handling).
- **Configuration**: Dataset-specific overrides in `configs/isl.yaml` allow changing parameters without code modification.
- **Model Interface**: The input dimension and number of classes are determined at runtime from the dataset, so the same HTMA model architecture can be reused for any sign language dataset.
- **Training and Inference Pipelines**: Remain unchanged when adding a new dataset; only the dataset handler and config need to be provided.

## Diagram of Component Interactions
```
+----------------+     +---------------------+     +---------------------+
|   Webcam       |     | MediaPipe Tasks     |     | Feature Builder     |
| (OpenCV Thread)| ---> | (Hand+Pose)         | ---> | (168-dim vector)    |
+----------------+     +---------------------+     +---------------------+
                                                       |
                                                       v
                                             +---------------------+
                                             | Normalization         |
                                             | (wrist/shoul. center) |
                                             +---------------------+
                                                       |
                                                       v
                                             +---------------------+
                                             | Ring Buffer (T frames)|
                                             +---------------------+
                                                       |
                                                       v
                                             +---------------------+
                                             | HTMA Model            |
                                             | (seq -> logits)       |
                                             +---------------------+
                                                       |
                                                       v
                                             +---------------------+
                                             | Softmax               |
                                             +---------------------+
                                                       |
                                                       v
                                             +---------------------+
                                             | Stability Gate        |
                                             | (emit word?)          |
                                             +---------------------+
                                                       | Yes
                                                       v
                                             +---------------------+
                                             | Sentence Assembler    |
                                             | (append word)         |
                                             +---------------------+
                                                       |
                                                       v
                                             +---------------------+
                                             | On-screen Display     |
                                             | (sentence, top-3,     |
                                             |  latency breakdown)   |
                                             +---------------------+
                                                       |
                                                       v
                                             +---------------------+
                                             | TTS Engine (Piper)    |
                                             | (synthesize audio)    |
                                             +---------------------+
                                                       |
                                                       v
                                             +---------------------+
                                             | Audio Playback        |
                                             | (sounddevice)         |
                                             +---------------------+
```

## Configuration Structure
```
configs/
├─ default.yaml          # Base configuration (batch 256, 6 workers, model, processing, paths)
├─ isl.yaml              # Full INCLUDE overrides (264 classes, lr 5e-4, 200 epochs)
├─ isl_include50.yaml    # INCLUDE-50 subset config
├─ joint.yaml            # INCLUDE+ISL500 union training (263 normalized classes, balanced sampler)
├─ joint_v2.yaml         # Joint + pose-dropout/source-balance retrain
├─ v2_tuned.yaml         # 5-op tuned augmentation profile
├─ m8_adapter.yaml / m8_augmentation.yaml / m8_distill.yaml  # M8 profiles
```

### Example `default.yaml` Snippet
```yaml
dataset:
  name: isl          # Selects active dataset (isl, asl, etc.)
  root: data/raw     # Root directory for raw data
  processed: data/processed  # Root for processed landmarks

model:
  name: htma
  input_dim: 168
  embedding_dim: 256
  htma_blocks: 2
  num_heads: 8
  dropout: 0.1

processing:
  sequence_length: 32          # T frames in buffer
  stability_gate:
    threshold: 0.40
    consecutive_frames: 12
    cooldown_frames: 30
    agreement_ratio: 0.60
    min_confidence_delta: 0.08

tts:
  engine: piper
  voice: en_US-lessac-medium   # verified; no en_IN voice exists
  sample_rate: 22050

paths:
  assets: assets               # MediaPipe task models
  checkpoints: models          # Model checkpoints
  logs: logs                   # TensorBoard logs
```

## Conclusion
This architecture provides a solid foundation for a real-time sign language translation system that is accurate, efficient, and extensible. By leveraging the HTMA temporal model and a modular design, we achieve state-of-the-art performance on ISL while keeping the system adaptable to other sign languages and future enhancements. The addition of latency profiling and warmup strategies ensures predictable performance and eliminates startup transients for reliable real-time operation.