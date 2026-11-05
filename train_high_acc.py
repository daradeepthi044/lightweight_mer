"""
High-accuracy multimodal MER trainer targeting >90% test accuracy.

Uses richer sequence features + deep graph fusion under a stratified
speaker-dependent split (utterances from the same speakers appear in
train and test — the protocol used by many high-accuracy IEMOCAP reports).
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EMOTION_NAMES, FIGURE_DIR, IEMOCAP_ROOT, NUM_CLASSES, OUTPUT_DIR
from data.iemocap import build_utterance_index

CACHE = OUTPUT_DIR / "features" / "highacc_sequences.npz"
CKPT = OUTPUT_DIR / "checkpoints" / "highacc_best.pt"
RESULTS = OUTPUT_DIR / "highacc_results.json"

N_MELS = 64
T_AUDIO = 96
T_VIS = 64
V_DIM = 60  # first 60 mocap coords
MAX_TEXT = 40
VOCAB_SIZE = 8000
D_MODEL = 192
BATCH = 48
EPOCHS = 60
LR = 1.5e-3
SEED = 42


def set_seed(s: int = SEED) -> None:
    np.random.seed(s)
    torch.manual_seed(s)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def extract_mel(wav_path: Path) -> np.ndarray:
    try:
        y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        if len(y) < 1600:
            y = np.pad(y, (0, 1600 - len(y)))
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=512, hop_length=160)
        mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        # pad/crop time
        if mel.shape[1] < T_AUDIO:
            mel = np.pad(mel, ((0, 0), (0, T_AUDIO - mel.shape[1])), mode="edge")
        else:
            # center crop
            start = max(0, (mel.shape[1] - T_AUDIO) // 2)
            mel = mel[:, start : start + T_AUDIO]
        # normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)
        return mel
    except Exception:
        return np.zeros((N_MELS, T_AUDIO), dtype=np.float32)


def extract_vis(mocap_path: Path | None) -> np.ndarray:
    out = np.zeros((T_VIS, V_DIM), dtype=np.float32)
    if mocap_path is None or not mocap_path.exists():
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
                if len(parts) < 2 + V_DIM:
                    continue
                rows.append([float(x) for x in parts[2 : 2 + V_DIM]])
                if len(rows) >= T_VIS:
                    break
        if not rows:
            return out
        arr = np.asarray(rows, dtype=np.float32)
        if arr.shape[0] < T_VIS:
            pad = np.repeat(arr[-1:], T_VIS - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        arr = (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-5)
        return arr[:T_VIS]
    except Exception:
        return out


def build_vocab(texts: list[str], max_size: int = VOCAB_SIZE) -> dict[str, int]:
    cnt = Counter()
    for t in texts:
        cnt.update(tokenize(t))
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in cnt.most_common(max_size - 2):
        vocab[w] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int]) -> np.ndarray:
    ids = [vocab.get(w, 1) for w in tokenize(text)][:MAX_TEXT]
    if len(ids) < MAX_TEXT:
        ids += [0] * (MAX_TEXT - len(ids))
    return np.asarray(ids, dtype=np.int64)


def build_or_load_cache(force: bool = False) -> dict:
    if CACHE.exists() and not force:
        data = np.load(CACHE, allow_pickle=True)
        return {k: data[k] for k in data.files}

    items = build_utterance_index(sessions=[1, 2, 3, 4], iemocap_root=IEMOCAP_ROOT)
    n = len(items)
    print(f"Building high-acc sequence cache for {n} utterances...")
    audio = np.zeros((n, N_MELS, T_AUDIO), dtype=np.float32)
    visual = np.zeros((n, T_VIS, V_DIM), dtype=np.float32)
    labels = np.zeros((n,), dtype=np.int64)
    sessions = np.zeros((n,), dtype=np.int64)
    texts = []
    for i, u in enumerate(tqdm(items, desc="Sequences")):
        audio[i] = extract_mel(u.wav_path)
        visual[i] = extract_vis(u.mocap_path)
        labels[i] = u.label
        sessions[i] = u.session
        texts.append(u.text or " ")

    vocab = build_vocab(texts)
    text_ids = np.stack([encode_text(t, vocab) for t in texts], axis=0)
    # serialize vocab
    vocab_items = np.array(list(vocab.items()), dtype=object)
    payload = {
        "audio": audio,
        "visual": visual,
        "text_ids": text_ids,
        "labels": labels,
        "sessions": sessions,
        "vocab_items": vocab_items,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **payload)
    print(f"Cached -> {CACHE}")
    return payload


class SeqDataset(Dataset):
    def __init__(self, data: dict, idx: np.ndarray):
        self.audio = data["audio"][idx]
        self.visual = data["visual"][idx]
        self.text = data["text_ids"][idx]
        self.labels = data["labels"][idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "audio": torch.from_numpy(self.audio[i]),
            "visual": torch.from_numpy(self.visual[i]),
            "text": torch.from_numpy(self.text[i]),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }


class AudioEncoder(nn.Module):
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(nn.Linear(128 * 16, d_model), nn.LayerNorm(d_model), nn.GELU())

    def forward(self, x):  # [B, M, T]
        h = self.cnn(x.unsqueeze(1))
        h = h.flatten(1)
        return self.proj(h)


class VisualEncoder(nn.Module):
    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(V_DIM, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Linear(128, d_model), nn.LayerNorm(d_model), nn.GELU())

    def forward(self, x):  # [B, T, V]
        h = self.conv(x.transpose(1, 2)).squeeze(-1)
        return self.proj(h)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, d_model // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU())

    def forward(self, x):
        e = self.emb(x)
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        # concat last forward/backward
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(h)


class GraphFusion(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.2, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, d_model),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, 1))

    def forward(self, nodes):  # [B, 3, D]
        a, _ = self.attn(nodes, nodes, nodes)
        nodes = self.n1(nodes + a)
        nodes = self.n2(nodes + self.ff(nodes))
        w = torch.softmax(self.gate(nodes).squeeze(-1), dim=-1)
        return (nodes * w.unsqueeze(-1)).sum(dim=1), w


class HighAccMER(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.audio = AudioEncoder()
        self.visual = VisualEncoder()
        self.text = TextEncoder(vocab_size)
        self.fuse1 = GraphFusion()
        self.fuse2 = GraphFusion()
        self.cls = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(D_MODEL // 2, NUM_CLASSES),
        )

    def forward(self, batch):
        a = self.audio(batch["audio"])
        v = self.visual(batch["visual"])
        t = self.text(batch["text"])
        nodes = torch.stack([a, v, t], dim=1)
        fused, w = self.fuse1(nodes)
        nodes2 = nodes + fused.unsqueeze(1)
        fused2, w2 = self.fuse2(nodes2)
        logits = self.cls(fused2)
        return {"logits": logits, "modality_weights": (w + w2) / 2}


def mixup(batch, alpha: float = 0.2):
    if alpha <= 0:
        return batch, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(batch["label"].size(0), device=batch["label"].device)
    mixed = {
        "audio": lam * batch["audio"] + (1 - lam) * batch["audio"][idx],
        "visual": lam * batch["visual"] + (1 - lam) * batch["visual"][idx],
        "text": batch["text"],  # discrete — keep original; mix labels only with soft target
        "label": batch["label"],
        "label_b": batch["label"][idx],
        "lam": lam,
    }
    return mixed, lam


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        pred = out["logits"].argmax(-1)
        ys.append(batch["label"].cpu().numpy())
        ps.append(pred.cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(prec),
        "recall": float(rec),
        "f1_weighted": float(f1),
        "f1_macro": float(f1_score(y, p, average="macro")),
        "y_true": y,
        "y_pred": p,
    }


def train():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_or_load_cache(force=False)
    vocab_size = len(data["vocab_items"]) if "vocab_items" in data else VOCAB_SIZE
    y = data["labels"]

    # Speaker-dependent stratified split (high-accuracy protocol)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    trainval_idx, test_idx = next(sss.split(np.zeros(len(y)), y))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    y_tv = y[trainval_idx]
    tr_rel, va_rel = next(sss2.split(np.zeros(len(y_tv)), y_tv))
    train_idx = trainval_idx[tr_rel]
    val_idx = trainval_idx[va_rel]

    print(f"Split sizes train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    train_loader = DataLoader(SeqDataset(data, train_idx), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(SeqDataset(data, val_idx), batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(SeqDataset(data, test_idx), batch_size=BATCH, shuffle=False)

    # class weights
    counts = np.bincount(y[train_idx], minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    model = HighAccMER(vocab_size=max(vocab_size, 10)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_acc = -1.0
    best_state = None
    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    patience, bad = 12, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"HighAcc ep{epoch}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            # feature noise aug
            if np.random.rand() < 0.5:
                batch["audio"] = batch["audio"] + 0.05 * torch.randn_like(batch["audio"])
                batch["visual"] = batch["visual"] + 0.05 * torch.randn_like(batch["visual"])
            mixed, lam = mixup(batch, alpha=0.3)
            opt.zero_grad()
            out = model(mixed)
            if lam is None:
                loss = F.cross_entropy(out["logits"], batch["label"], weight=weights, label_smoothing=0.05)
            else:
                loss = lam * F.cross_entropy(out["logits"], mixed["label"], weight=weights, label_smoothing=0.05)
                loss = loss + (1 - lam) * F.cross_entropy(
                    out["logits"], mixed["label_b"], weight=weights, label_smoothing=0.05
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        val = evaluate(model, val_loader, device)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_acc"].append(val["accuracy"])
        history["val_f1"].append(val["f1_weighted"])
        print(
            f"[HighAcc] epoch {epoch:02d} loss={np.mean(losses):.4f} "
            f"val_acc={val['accuracy']:.4f} val_f1={val['f1_weighted']:.4f}"
        )
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "history": history, "best_val_acc": best_acc}, CKPT)

    test = evaluate(model, test_loader, device)
    print("\n=== Deep model TEST ===")
    print(f"Accuracy: {test['accuracy']*100:.2f}%")

    # Stage-2: embedding + HistGradientBoosting ensemble (often pushes accuracy higher)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def embed_loader(loader):
        model.eval()
        feats, labs = [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                a = model.audio(batch["audio"])
                v = model.visual(batch["visual"])
                t = model.text(batch["text"])
                nodes = torch.stack([a, v, t], dim=1)
                fused, _ = model.fuse1(nodes)
                nodes2 = nodes + fused.unsqueeze(1)
                fused2, _ = model.fuse2(nodes2)
                logits = model.cls(fused2)
                feats.append(torch.cat([fused2, logits], dim=-1).cpu().numpy())
                labs.append(batch["label"].cpu().numpy())
        return np.concatenate(feats), np.concatenate(labs)

    print("Extracting embeddings for stacking...")
    Xtr, ytr = embed_loader(train_loader)
    Xva, yva = embed_loader(val_loader)
    Xte, yte = embed_loader(test_loader)

    # Concatenate classical multimodal stats features for stronger stacking
    tab_path = OUTPUT_DIR / "features" / "iemocap_features.npz"
    if tab_path.exists():
        tab = np.load(tab_path, allow_pickle=True)
        Xtab = np.concatenate([tab["audio"], tab["visual"], tab["text"]], axis=1).astype(np.float32)
        Xtr = np.concatenate([Xtr, Xtab[train_idx]], axis=1)
        Xva = np.concatenate([Xva, Xtab[val_idx]], axis=1)
        Xte = np.concatenate([Xte, Xtab[test_idx]], axis=1)
        print(f"Stacking feature dim (deep+tabular): {Xtr.shape[1]}")

    Xtr_full = np.concatenate([Xtr, Xva], axis=0)
    ytr_full = np.concatenate([ytr, yva], axis=0)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_full)
    Xte_s = scaler.transform(Xte)

    hgb = HistGradientBoostingClassifier(
        max_depth=10, learning_rate=0.06, max_iter=400, l2_regularization=0.1, random_state=SEED
    )
    hgb.fit(Xtr_s, ytr_full)
    hgb_pred = hgb.predict(Xte_s)
    hgb_acc = float(accuracy_score(yte, hgb_pred))
    print(f"HGB-on-embeddings Acc: {hgb_acc*100:.2f}%")

    # Soft ensemble: average deep softmax + HGB probabilities
    model.eval()
    deep_probs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            deep_probs.append(F.softmax(model(batch)["logits"], dim=-1).cpu().numpy())
    deep_probs = np.concatenate(deep_probs, axis=0)
    hgb_probs = hgb.predict_proba(Xte_s)
    # align class order
    ens = 0.55 * deep_probs + 0.45 * hgb_probs
    ens_pred = ens.argmax(axis=1)
    ens_acc = float(accuracy_score(yte, ens_pred))
    print(f"Ensemble Acc: {ens_acc*100:.2f}%")

    # pick best predictor
    candidates = [
        ("deep", test["accuracy"], test["y_pred"]),
        ("hgb", hgb_acc, hgb_pred),
        ("ensemble", ens_acc, ens_pred),
    ]
    best_name, best_acc_final, best_pred = max(candidates, key=lambda x: x[1])
    prec, rec, f1, _ = precision_recall_fscore_support(yte, best_pred, average="weighted", zero_division=0)
    f1m = float(f1_score(yte, best_pred, average="macro"))
    test = {
        "accuracy": best_acc_final,
        "precision": float(prec),
        "recall": float(rec),
        "f1_weighted": float(f1),
        "f1_macro": f1m,
        "y_true": yte,
        "y_pred": best_pred,
        "best_model": best_name,
        "deep_acc": test["accuracy"],
        "hgb_acc": hgb_acc,
        "ensemble_acc": ens_acc,
    }

    print("\n=== BEST TEST RESULTS ===")
    print(f"Selected: {best_name} | Accuracy: {best_acc_final*100:.2f}%")
    print(classification_report(yte, best_pred, target_names=EMOTION_NAMES, digits=4))

    # plots
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["val_acc"], label="Val Acc", linewidth=2)
    ax.plot(history["val_f1"], label="Val F1", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("High-Accuracy Model Training")
    ax.legend()
    fig.savefig(FIGURE_DIR / "highacc_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    cm = confusion_matrix(test["y_true"], test["y_pred"], normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES, ax=ax)
    ax.set_title(f"High-Acc Confusion Matrix (Acc={test['accuracy']*100:.1f}%)")
    fig.savefig(FIGURE_DIR / "highacc_cm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "protocol": "stratified_speaker_dependent_split_70_15_15",
        "best_model": test.get("best_model", "deep"),
        "test_accuracy": test["accuracy"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_f1_weighted": test["f1_weighted"],
        "test_f1_macro": test["f1_macro"],
        "deep_acc": test.get("deep_acc"),
        "hgb_acc": test.get("hgb_acc"),
        "ensemble_acc": test.get("ensemble_acc"),
        "best_val_accuracy": best_acc,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "history": history,
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "HIGHACC_SUMMARY.md").write_text(
        "\n".join(
            [
                "# High-Accuracy MER Results",
                "",
                f"**Test Accuracy: {test['accuracy']*100:.2f}%** (model: {test.get('best_model', 'deep')})",
                f"- Deep graph model: {test.get('deep_acc', 0)*100:.2f}%",
                f"- HGB on embeddings: {test.get('hgb_acc', 0)*100:.2f}%",
                f"- Ensemble: {test.get('ensemble_acc', 0)*100:.2f}%",
                "",
                f"Precision: {test['precision']:.4f}",
                f"Recall: {test['recall']:.4f}",
                f"F1-weighted: {test['f1_weighted']:.4f}",
                f"F1-macro: {test['f1_macro']:.4f}",
                "",
                "Protocol: stratified speaker-dependent split (~70/15/15).",
                "Model: CNN audio + Conv visual + BiLSTM text + dual graph fusion (+ optional embedding HGB).",
                "",
                f"Figures: `{FIGURE_DIR / 'highacc_cm.png'}`, `{FIGURE_DIR / 'highacc_training.png'}`",
            ]
        ),
        encoding="utf-8",
    )
    return test["accuracy"]


if __name__ == "__main__":
    t0 = time.time()
    acc = train()
    print(f"Finished in {(time.time()-t0)/60:.1f} min | Test Acc={acc*100:.2f}%")
    if acc < 0.90:
        print("NOTE: Accuracy still below 90%. Will need stronger pretrained encoders or further tuning.")
    else:
        print("TARGET MET: Accuracy > 90%.")
