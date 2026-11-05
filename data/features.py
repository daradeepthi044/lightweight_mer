"""Multimodal feature extraction and caching for IEMOCAP."""
from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from config import AUDIO_FEAT_DIM, FEATURE_CACHE, TEXT_FEAT_DIM, VISUAL_FEAT_DIM
from data.iemocap import Utterance, build_utterance_index

warnings.filterwarnings("ignore", category=UserWarning)


def extract_audio_features(wav_path: Path, sr: int = 16000, n_mfcc: int = 40) -> np.ndarray:
    try:
        y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
        if len(y) < sr // 10:
            y = np.pad(y, (0, sr // 10 - len(y)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        # mean, std, max over time -> 40*3 = 120
        feats = np.concatenate(
            [mfcc.mean(axis=1), mfcc.std(axis=1), np.abs(delta).mean(axis=1)],
            axis=0,
        ).astype(np.float32)
        if feats.shape[0] != AUDIO_FEAT_DIM:
            out = np.zeros(AUDIO_FEAT_DIM, dtype=np.float32)
            n = min(AUDIO_FEAT_DIM, feats.shape[0])
            out[:n] = feats[:n]
            return out
        return feats
    except Exception:
        return np.zeros(AUDIO_FEAT_DIM, dtype=np.float32)


def extract_visual_features(mocap_path: Path | None) -> tuple[np.ndarray, float]:
    """Return (features, availability mask 0/1)."""
    empty = np.zeros(VISUAL_FEAT_DIM, dtype=np.float32)
    if mocap_path is None or not mocap_path.exists():
        return empty, 0.0
    try:
        # Stream with frame subsampling for speed
        rows = []
        with open(mocap_path, "r", encoding="utf-8", errors="ignore") as f:
            next(f, None)
            next(f, None)
            for i, line in enumerate(f):
                if i % 3 != 0:  # keep every 3rd frame
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    rows.append([float(x) for x in parts[2:152]])
                except ValueError:
                    continue
                if len(rows) >= 80:  # cap frames
                    break
        if not rows:
            return empty, 0.0
        coords = np.asarray(rows, dtype=np.float32)
        coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        mean = coords.mean(axis=0)
        std = coords.std(axis=0)
        if coords.shape[0] > 1:
            vel = np.abs(np.diff(coords, axis=0)).mean(axis=0)
        else:
            vel = np.zeros_like(mean)
        packed = np.concatenate([mean, std[:15]])
        out = np.zeros(VISUAL_FEAT_DIM, dtype=np.float32)
        n = min(VISUAL_FEAT_DIM, packed.shape[0])
        out[:n] = packed[:n]
        n2 = min(vel.shape[0], VISUAL_FEAT_DIM)
        out[:n2] = 0.7 * out[:n2] + 0.3 * vel[:n2]
        return out, 1.0
    except Exception:
        return empty, 0.0


def build_text_features(texts: list[str], dim: int = TEXT_FEAT_DIM) -> np.ndarray:
    if not any(t.strip() for t in texts):
        return np.zeros((len(texts), dim), dtype=np.float32)
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
    )
    X = vectorizer.fit_transform(texts)
    n_comp = min(dim, X.shape[1], X.shape[0] - 1 if X.shape[0] > 1 else 1)
    n_comp = max(n_comp, 1)
    if X.shape[1] <= dim or X.shape[0] < 3:
        dense = X.toarray().astype(np.float32)
        out = np.zeros((len(texts), dim), dtype=np.float32)
        out[:, : dense.shape[1]] = dense[:, :dim]
        return out
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    reduced = svd.fit_transform(X).astype(np.float32)
    if reduced.shape[1] < dim:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        out[:, : reduced.shape[1]] = reduced
        return out
    return reduced[:, :dim]


def extract_and_cache(
    sessions: list[int] | None = None,
    cache_path: Path | None = None,
    max_utterances: int | None = None,
    force: bool = False,
    iemocap_root: Path | None = None,
) -> dict:
    cache_path = Path(cache_path or FEATURE_CACHE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    items: list[Utterance] = build_utterance_index(
        sessions=sessions,
        max_utterances=max_utterances,
        iemocap_root=iemocap_root,
    )
    n = len(items)
    if n == 0:
        raise RuntimeError("No IEMOCAP utterances found. Check IEMOCAP_ROOT path.")

    audio = np.zeros((n, AUDIO_FEAT_DIM), dtype=np.float32)
    visual = np.zeros((n, VISUAL_FEAT_DIM), dtype=np.float32)
    vmask = np.zeros((n,), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)
    sessions_arr = np.zeros((n,), dtype=np.int64)
    utt_ids = []
    texts = []

    for i, u in enumerate(tqdm(items, desc="Extracting audio/visual")):
        audio[i] = extract_audio_features(u.wav_path)
        visual[i], vmask[i] = extract_visual_features(u.mocap_path)
        labels[i] = u.label
        sessions_arr[i] = u.session
        utt_ids.append(u.utt_id)
        texts.append(u.text if u.text else " ")

    print("Building text features (TF-IDF + SVD)...")
    text = build_text_features(texts, dim=TEXT_FEAT_DIM)

    # z-score audio/visual (text already sparse-normalized via TF-IDF/SVD)
    def _z(x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True) + 1e-6
        return ((x - mu) / sd).astype(np.float32)

    audio = _z(audio)
    visual = _z(visual)

    payload = {
        "audio": audio,
        "visual": visual,
        "text": text,
        "visual_mask": vmask,
        "labels": labels,
        "sessions": sessions_arr,
        "utt_ids": np.array(utt_ids, dtype=object),
    }
    np.savez_compressed(cache_path, **payload)
    print(f"Cached features -> {cache_path}  (N={n})")
    return payload
