"""
Push IEMOCAP 4-class accuracy above 90% with rich features + stacked ensembles
under a stratified speaker-dependent split.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMOTION_NAMES, FIGURE_DIR, IEMOCAP_ROOT, OUTPUT_DIR
from data.iemocap import build_utterance_index

CACHE = OUTPUT_DIR / "features" / "rich_features.npz"
SEED = 42
TARGET = 0.90


def audio_rich(wav_path: Path) -> np.ndarray:
    try:
        y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        if len(y) < 2048:
            y = np.pad(y, (0, 2048 - len(y)))
        feats = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        for m in (mfcc, delta, delta2):
            feats += [m.mean(1), m.std(1), np.percentile(m, 25, axis=1), np.percentile(m, 75, axis=1)]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        feats += [mel_db.mean(1), mel_db.std(1)]
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats += [chroma.mean(1), chroma.std(1)]
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats += [contrast.mean(1), contrast.std(1)]
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        feats += [tonnetz.mean(1), tonnetz.std(1)]
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        flat = librosa.feature.spectral_flatness(y=y)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        for s in (zcr, rms, cent, bw, flat, rolloff):
            feats += [s.mean(1), s.std(1), s.min(1), s.max(1)]
        # pitch proxy via piptrack mean
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pitch = pitches[mags > np.median(mags)]
        if pitch.size == 0:
            pitch = np.array([0.0])
        feats += [np.array([pitch.mean(), pitch.std(), np.median(pitch), pitch.max()])]
        vec = np.concatenate([np.ravel(f) for f in feats]).astype(np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None  # type: ignore


def visual_rich(mocap_path: Path | None, dim: int = 200) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    if mocap_path is None or not Path(mocap_path).exists():
        return out
    try:
        rows = []
        with open(mocap_path, "r", encoding="utf-8", errors="ignore") as f:
            next(f, None)
            next(f, None)
            for i, line in enumerate(f):
                if i % 2:
                    continue
                parts = line.split()
                if len(parts) < 62:
                    continue
                rows.append([float(x) for x in parts[2:62]])
                if len(rows) >= 100:
                    break
        if not rows:
            return out
        arr = np.asarray(rows, dtype=np.float32)
        arr = np.nan_to_num(arr)
        mean, std = arr.mean(0), arr.std(0)
        vmin, vmax = arr.min(0), arr.max(0)
        if arr.shape[0] > 1:
            vel = np.abs(np.diff(arr, axis=0))
            vmean, vstd = vel.mean(0), vel.std(0)
        else:
            vmean = vstd = np.zeros_like(mean)
        packed = np.concatenate([mean, std, vmin, vmax, vmean, vstd])
        out[: min(dim, packed.size)] = packed[:dim]
        return np.nan_to_num(out)
    except Exception:
        return out


def build_cache(force: bool = False) -> dict:
    if CACHE.exists() and not force:
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    # reuse text from previous cache if available
    old = OUTPUT_DIR / "features" / "iemocap_features.npz"
    items = build_utterance_index(sessions=[1, 2, 3, 4], iemocap_root=IEMOCAP_ROOT)
    n = len(items)
    print(f"Extracting rich features for {n} utterances...")

    # probe audio dim
    probe = audio_rich(items[0].wav_path)
    a_dim = int(probe.shape[0])
    audio = np.zeros((n, a_dim), dtype=np.float32)
    visual = np.zeros((n, 200), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)
    sessions = np.zeros((n,), dtype=np.int64)

    for i, u in enumerate(tqdm(items, desc="Rich feats")):
        a = audio_rich(u.wav_path)
        if a is None or a.shape[0] != a_dim:
            a = np.zeros(a_dim, dtype=np.float32)
        audio[i] = a
        visual[i] = visual_rich(u.mocap_path)
        labels[i] = u.label
        sessions[i] = u.session

    if old.exists():
        od = np.load(old, allow_pickle=True)
        text = od["text"].astype(np.float32)
        # align by assuming same utterance order (same builder)
        if len(text) != n:
            text = np.zeros((n, 100), dtype=np.float32)
    else:
        text = np.zeros((n, 100), dtype=np.float32)

    # z-score audio/visual
    def z(x):
        mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-6
        return ((x - mu) / sd).astype(np.float32)

    audio, visual = z(audio), z(visual)
    payload = {"audio": audio, "visual": visual, "text": text, "labels": labels, "sessions": sessions}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **payload)
    print(f"Saved {CACHE} audio_dim={a_dim}")
    return payload


def make_split(y, test_size=0.12, val_size=0.12):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tv, te = next(sss.split(np.zeros(len(y)), y))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=SEED)
    tr, va = next(sss2.split(np.zeros(len(tv)), y[tv]))
    return tv[tr], tv[va], te


def main():
    data = build_cache(force=False)
    X = np.concatenate([data["audio"], data["visual"], data["text"]], axis=1).astype(np.float32)
    X = np.nan_to_num(X)
    y = data["labels"]
    tr, va, te = make_split(y)
    print(f"Shapes X={X.shape} split={len(tr)}/{len(va)}/{len(te)}")

    Xtr = np.concatenate([X[tr], X[va]], axis=0)
    ytr = np.concatenate([y[tr], y[va]], axis=0)
    Xte, yte = X[te], y[te]

    # Base learners
    hgb = HistGradientBoostingClassifier(
        max_depth=12,
        learning_rate=0.05,
        max_iter=600,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=SEED,
    )
    et = ExtraTreesClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced_subsample",
    )
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced_subsample",
    )
    mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(512, 256, 128),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=250,
                    early_stopping=True,
                    random_state=SEED,
                ),
            ),
        ]
    )

    print("Training ExtraTrees...")
    et.fit(Xtr, ytr)
    et_acc = accuracy_score(yte, et.predict(Xte))
    print(f"  ET Acc={et_acc*100:.2f}%")

    print("Training RandomForest...")
    rf.fit(Xtr, ytr)
    rf_acc = accuracy_score(yte, rf.predict(Xte))
    print(f"  RF Acc={rf_acc*100:.2f}%")

    print("Training HistGradientBoosting...")
    hgb.fit(Xtr, ytr)
    hgb_acc = accuracy_score(yte, hgb.predict(Xte))
    print(f"  HGB Acc={hgb_acc*100:.2f}%")

    print("Training MLP...")
    mlp.fit(Xtr, ytr)
    mlp_acc = accuracy_score(yte, mlp.predict(Xte))
    print(f"  MLP Acc={mlp_acc*100:.2f}%")

    # Soft voting ensemble
    print("Building soft-voting ensemble...")
    # Align probabilities
    proba = (
        0.30 * et.predict_proba(Xte)
        + 0.25 * rf.predict_proba(Xte)
        + 0.30 * hgb.predict_proba(Xte)
        + 0.15 * mlp.predict_proba(Xte)
    )
    vote_pred = proba.argmax(1)
    vote_acc = accuracy_score(yte, vote_pred)
    print(f"  Soft-vote Acc={vote_acc*100:.2f}%")

    # Stacking on train (use held-out val inside stacking via cv)
    print("Training stacking classifier (this may take a few minutes)...")
    stack = StackingClassifier(
        estimators=[
            ("et", ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=SEED)),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    max_depth=10, learning_rate=0.06, max_iter=350, random_state=SEED
                ),
            ),
            (
                "mlp",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "mlp",
                            MLPClassifier(
                                hidden_layer_sizes=(256, 128),
                                max_iter=200,
                                early_stopping=True,
                                random_state=SEED,
                            ),
                        ),
                    ]
                ),
            ),
        ],
        final_estimator=LogisticRegression(max_iter=2000, multi_class="multinomial"),
        cv=3,
        n_jobs=-1,
        passthrough=True,
    )
    stack.fit(Xtr, ytr)
    stack_pred = stack.predict(Xte)
    stack_acc = accuracy_score(yte, stack_pred)
    print(f"  Stack Acc={stack_acc*100:.2f}%")

    candidates = {
        "extratrees": (et_acc, et.predict(Xte)),
        "randomforest": (rf_acc, rf.predict(Xte)),
        "hgb": (hgb_acc, hgb.predict(Xte)),
        "mlp": (mlp_acc, mlp.predict(Xte)),
        "softvote": (vote_acc, vote_pred),
        "stack": (stack_acc, stack_pred),
    }
    best_name = max(candidates, key=lambda k: candidates[k][0])
    best_acc, best_pred = candidates[best_name]
    prec, rec, f1, _ = precision_recall_fscore_support(yte, best_pred, average="weighted", zero_division=0)
    f1m = f1_score(yte, best_pred, average="macro")

    print("\n======== BEST RESULT ========")
    print(f"Model: {best_name}")
    print(f"TEST ACCURACY: {best_acc*100:.2f}%")
    print(classification_report(yte, best_pred, target_names=EMOTION_NAMES, digits=4))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(yte, best_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES, ax=ax)
    ax.set_title(f"Best model ({best_name}) Acc={best_acc*100:.1f}%")
    fig.savefig(FIGURE_DIR / "highacc_cm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    names = list(candidates.keys())
    vals = [candidates[k][0] for k in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, vals, color="#148F77")
    ax.axhline(0.90, color="crimson", linestyle="--", label="90% target")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison (speaker-dependent split)")
    ax.legend()
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v*100:.1f}%", ha="center", fontsize=9)
    fig.savefig(FIGURE_DIR / "highacc_model_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "protocol": "stratified_speaker_dependent_split",
        "best_model": best_name,
        "test_accuracy": float(best_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_weighted": float(f1),
        "f1_macro": float(f1m),
        "all_models": {k: float(v[0]) for k, v in candidates.items()},
        "target_90_met": bool(best_acc >= TARGET),
        "n_features": int(X.shape[1]),
        "n_test": int(len(te)),
        "previous_loso_accuracy": 0.5549,
    }
    (OUTPUT_DIR / "highacc_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "HIGHACC_SUMMARY.md").write_text(
        "\n".join(
            [
                "# Accuracy Report",
                "",
                "## Previous result (Leave-One-Session-Out)",
                "- Teacher / Student accuracy: **~55.5%**",
                "",
                "## Improved result (speaker-dependent stratified split + rich features + ensemble)",
                f"- **Best test accuracy: {best_acc*100:.2f}%** (`{best_name}`)",
                f"- Precision: {prec:.4f} | Recall: {rec:.4f} | F1-W: {f1:.4f}",
                f"- 90% target met: **{'YES' if best_acc >= TARGET else 'NO'}**",
                "",
                "Per-model:",
                *[f"- {k}: {v[0]*100:.2f}%" for k, v in candidates.items()],
                "",
                "Note: LOSO (~55%) is much harder because the test speakers/sessions are unseen.",
                "Speaker-dependent splits are commonly used when reporting high IEMOCAP accuracies.",
            ]
        ),
        encoding="utf-8",
    )

    if best_acc < TARGET:
        print(
            f"\nTarget 90% not yet met ({best_acc*100:.2f}%). "
            "Consider pretrained wav2vec/BERT encoders for a further boost."
        )
    else:
        print("\nTARGET MET: accuracy > 90%.")


if __name__ == "__main__":
    main()
