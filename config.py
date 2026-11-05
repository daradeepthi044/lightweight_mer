"""Configuration for Lightweight Graph-Based Multimodal Emotion Recognition."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
IEMOCAP_ROOT = PROJECT_ROOT / "IEMOCAP" / "IEMOCAP_full_release"

# Standard IEMOCAP 4-class mapping (happy = hap + exc)
EMOTION_MAP = {
    "ang": 0,
    "hap": 1,
    "exc": 1,
    "sad": 2,
    "neu": 3,
}
EMOTION_NAMES = ["angry", "happy", "sad", "neutral"]
NUM_CLASSES = 4

# Feature dims after extraction (before projection)
AUDIO_FEAT_DIM = 120   # 40 MFCC mean/std/max
VISUAL_FEAT_DIM = 165  # MoCap marker stats
TEXT_FEAT_DIM = 100    # Truncated SVD on TF-IDF


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 0
    teacher_epochs: int = 18
    student_epochs: int = 22
    lr_teacher: float = 1e-3
    lr_student: float = 1.5e-3
    weight_decay: float = 1e-4
    kd_temperature: float = 3.0
    kd_alpha: float = 0.6          # weight on soft KD loss
    feature_kd_weight: float = 0.2
    prune_ratio: float = 0.5      # fraction of off-diagonal edges kept per node
    sparse_topk: int = 8           # sparse attention top-k
    device: str = "cpu"
    sessions: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    # Leave-one-session-out fold (None = run all folds)
    fold_session: int | None = None
    max_utterances: int | None = None  # optional subsample for smoke tests
    early_stop_patience: int = 8


@dataclass
class ModelConfig:
    # Shared projected dim
    d_model: int = 128
    # Teacher (heavy)
    teacher_hidden: int = 256
    teacher_layers: int = 3
    teacher_heads: int = 4
    teacher_dropout: float = 0.3
    # Student (light)
    student_hidden: int = 96
    student_layers: int = 2
    student_heads: int = 2
    student_dropout: float = 0.25
    num_modalities: int = 3  # audio, visual, text


CFG = TrainConfig()
MCFG = ModelConfig()

OUTPUT_DIR = ROOT / "outputs"
FEATURE_CACHE = OUTPUT_DIR / "features" / "iemocap_features.npz"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_JSON = OUTPUT_DIR / "results.json"
METRICS_CSV = OUTPUT_DIR / "metrics_summary.csv"
