# Sign Language Translation System - Project Specification

## 1. Overview
A modular real-time sign language translation system for Indian Sign Language (ISL) that processes webcam video input, extracts MediaPipe landmarks, uses an HTMA temporal model for word prediction, assembles sentences, and synthesizes speech via offline TTS (Piper).

## 2. Goals
- **Primary**: Real-time ISL-to-text-and-speech translation with ≥90% top-1 accuracy on INCLUDE-263 test split.
- **Secondary**: Modular design for easy extension to ASL and other sign languages.
- **Target Deployment**: Local execution on Windows 11 laptop (Acer Predator Helios Neo 16: i9-14900HX, RTX 4070 8GB, 16GB RAM, 1TB NVMe).

## 3. Functional Requirements
### 3.1 Input
- Live video stream from webcam (minimum 15 FPS, target 30 FPS).
- Supported resolutions: 640x480, 1280x720 (configurable).

### 3.2 Processing Pipeline
1. **Video Capture**: Threaded OpenCV capture with `CAP_DSHOW` backend on Windows.
2. **Landmark Extraction**:
   - MediaPipe Tasks API: `HandLandmarker` (2 hands) + `PoseLandmarker` lite.
   - LIVE_STREAM mode with async detection and result callbacks.
   - Output: per-frame landmark vectors (168 dimensions: both hands × (21×3 + 1) + upper-body pose × (13×3 + 1)).
3. **Feature Normalization**:
   - Per-hand: wrist-centered translation, scaling by wrist-to-middle-MCP distance.
   - Pose: mid-shoulder centering, scaling by shoulder width.
4. **Temporal Modeling**:
   - HTMA (Head-Token Mixing Attention) encoder with absolute and rotary positional encodings.
   - Sequence processing: ring buffer of last T frames (T=32-64) with masking for missing hands.
   - HTMA architecture (per CVPR 2026 paper):
     - Input projection to embedding dimension D (default 256).
     - Addition of Absolute Positional Encoding (APE) and Rotary Positional Embedding (RoPE).
     - Stack of HTMA blocks (default 2):
       * Local Conv1D-based temporal mixing (kernel size=3) on value vectors.
       * Multi-token convolution in attention score space.
       * Standard multi-head attention (H=8 heads) with APE+RoPE.
       * Layer normalization and residual connections.
     - Attention pooling: learnable query to aggregate temporal dimension.
     - MLP head: two layers (D → D → N) with GELU activation and dropout.
   - Optional auxiliary losses (frame-level classification) can be enabled during training to improve performance.
5. **Word Emission**:
   - Stability gate: emit word when running-mean softmax probability ≥ τ (0.85) for k consecutive frames (k=8-12).
   - Cooldown period to prevent duplicates (1.5s or until class changes).
6. **Sentence Assembly**:
   - Append emitted words to sentence bar.
   - Deduplicate consecutive identical words.
   - Support for backspace/clear via keyboard or gesture.
7. **Text-to-Speech**:
   - Offline synthesis using Piper TTS engine.
   - Non-blocking audio playback via sounddevice worker queue.
   - Configurable voice (default: `en_US`).

### 3.3 Output
- On-screen display:
  - Real-time video feed with landmark overlay (optional).
  - Top-3 predictions with confidence bars.
  - Current assembled sentence.
  - FPS and latency metrics (debug mode).
- Audio output: synthesized speech of recognized words/sentences.

### 3.4 Configuration
- YAML-based configuration system:
  - Base config (`configs/default.yaml`).
  - Dataset-specific overrides (`configs/isl.yaml`).
  - CLI overrides for rapid experimentation.
- Key configurable parameters:
  - Model: embedding dimension, HTMA blocks, attention heads.
  - Processing: sequence length T, stability gate thresholds (τ, k).
  - TTS: voice model, playback device.
  - Paths: data, model, asset directories.

### 3.5 Modularity
- Dataset-agnostic design via registry pattern.
- Adding a new sign language (e.g., ASL) requires:
  1. Implementing a dataset handler (download, metadata, label mapping).
  2. Adding dataset-specific config override.
  3. No changes to core training/inference pipelines.

## 4. Non-Functional Requirements
### 4.1 Performance
- **Latency**: End-to-end latency < 200ms (target <150ms) for interactive use.
- **Throughput**: Sustain ≥25 FPS processing pipeline on target hardware.
- **Model Size**: < 10 MB disk footprint for exported model.

### 4.2 Accuracy
- **Target**: ≥90% top-1 accuracy on INCLUDE-263 test split.
- **Stretch Goal**: Match literature performance (~95%+) with augmentation and tuning.

### 4.3 Reliability
- Graceful handling of missing landmarks (carry-forward ≤3 frames, then masked).
- Robust to varying lighting and background via normalization.
- Deterministic behavior with fixed seeds for reproducibility.

### 4.4 Usability
- Minimal setup: single environment setup script (`setup.ps1`).
- Clear documentation for end-users and developers.
- Intuitive UI controls (Space to speak, C to clear).

## 5. Technical Stack
### 5.1 Core
- **Language**: Python 3.11
- **Deep Learning**: PyTorch 2.x with CUDA 12.1
- **Computer Vision**: OpenCV, MediaPipe Tasks API (0.10.x)
- **TTS**: Piper 1.8 (`piper-tts` PyPI package)
- **Audio**: sounddevice, soundfile for playback
- **Configuration**: PyYAML
- **Logging & Viz**: TensorBoard, Matplotlib (for debugging/EDA)

### 5.2 Development & Testing
- **Testing**: pytest
- **Code Quality**: Ruff, mypy (optional)
- **Version Control**: Git
- **Documentation**: Markdown

### 5.3 Assets
- MediaPipe task models: `hand_landmarker.task`, `pose_landmarker.lite.task` (downloaded from Google storage).
- Pretrained model checkpoints (generated during training).
- Piper voice models (downloaded on first use).

## 6. API Interfaces (Internal)
### 6.1 Dataset Handler (Abstract Base)
```python
class SignLanguageDataset(ABC):
    @abstractmethod
    def download(self, dest_dir: str) -> None: ...
    @abstractmethod
    def build_metadata(self) -> pd.DataFrame: ...
    @abstractmethod
    def get_label_map(self) -> Dict[str, int]: ...
    @abstractmethod
    def get_splits(self) -> Dict[str, List[str]]: ...  # train/val/test video paths
```

### 6.2 Feature Extractor
```python
class LandmarkExtractor:
    def __init__(self, feature_spec: FeatureSpec): ...
    def extract_from_frame(self, frame: np.ndarray) -> np.ndarray: ...  # (168,)
    def extract_from_video(self, video_path: str) -> np.ndarray: ...    # (T, 168)
    def extract_live(self, frame_callback: Callable[[np.ndarray], None]): ...  # LIVE_STREAM
```

### 6.3 HTMA Model
```python
class HTMAClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, config: HTMAConfig): ...
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor: ...  # logits (B, N)
```

### 6.4 Stability Gate
```python
class StabilityGate:
    def __init__(self, threshold: float = 0.85, consecutive_frames: int = 10, cooldown_frames: int = 45): ...
    def update(self, probs: np.ndarray) -> Optional[int]: ...  # returns emitted class index or None
```

### 6.5 Sentence Assembler
```python
class SentenceAssembler:
    def __init__(self, dedup: bool = True): ...
    def add_word(self, word: str) -> None: ...
    def clear(self) -> None: ...
    def backspace(self) -> None: ...
    def get_sentence(self) -> str: ...
```

### 6.6 TTS Engine (Abstract Base)
```python
class TTSEngine(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray: ...  # audio samples
    @abstractmethod
    def play(self, audio: np.ndarray, sample_rate: int) -> None: ...  # non-blocking
```

## 7. Data Flow
1. **Offline Preprocessing**:
   - Dataset videos → MediaPipe landmark extraction → `.npz` cache (float16) + metadata splits.
2. **Training**:
   - `.npz` cache → PyTorch DataLoader → HTMA model → cross-entropy loss (optional: + auxiliary frame-level loss) → AdamW optimizer → checkpoint.
3. **Live Inference**:
   - Webcam → OpenCV → MediaPipe Tasks (async) → feature builder → normalization → ring buffer → HTMA model → softmax → stability gate → sentence assembler → TTS → audio output.

## 8. Success Criteria
### 8.1 Milestone Acceptance (see IMPLEMENTATION_PLAN.md for details)
- **M0**: GPU visible; live landmark loop ≥25 FPS.
- **M1**: 4,287 ISL videos downloaded; 263 classes; EDA notebook runnable.
- **M2**: ≥95% videos extracted to landmarks; overlays correct.
- **M3**: DataLoader yields ≥10k frames/s; augmentation tests pass; overfit-one-batch → ~100% train accuracy.
- **M4**: HTMA on INCLUDE-50: val top-1 ≥90%; on INCLUDE-263: val top-1 ≥88%.
- **M5**: Live loop ≥25 FPS; stable word predictions; latency breakdown.
- **M6**: 3-5 word ISL sequence → sentence + audible speech; works offline.
- **M7**: Final metrics recorded; demo video; README complete with setup, results, limitations.

### 8.2 Final Validation
- Test-split top-1/top-5 accuracy reported.
- Per-class accuracy and confusion matrix analyzed.
  - Live-demo latency table and FPS measured.
  - Documented limitations (word-level scope, single-signer domain gap, lighting sensitivity).

## 9. Future Work (Post-M7)
- Extension to ASL (M8-M12).
- Exploration of sentence-level translation with ISH-NEWS dataset.
- Integration of non-manual features (facial expressions via FaceLandmarker).
- Optimization for mobile/web deployment (TensorRT, ONNX, WebAssembly).
- Continuous signing translation (CTC/seq2seq models).

---
*Version: 1.1 (Updated with HTMA architecture details as of 2026-09-10)*