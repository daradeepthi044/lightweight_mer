"""PyTorch dataset and loaders for cached multimodal features."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MultimodalEmotionDataset(Dataset):
    def __init__(self, data: dict[str, Any], indices: np.ndarray):
        self.audio = data["audio"][indices]
        self.visual = data["visual"][indices]
        self.text = data["text"][indices]
        self.vmask = data["visual_mask"][indices]
        self.labels = data["labels"][indices]
        self.sessions = data["sessions"][indices]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "audio": torch.from_numpy(self.audio[i]),
            "visual": torch.from_numpy(self.visual[i]),
            "text": torch.from_numpy(self.text[i]),
            "visual_mask": torch.tensor(self.vmask[i], dtype=torch.float32),
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }


def make_loso_splits(sessions: np.ndarray, test_session: int):
    test_idx = np.where(sessions == test_session)[0]
    train_pool = np.where(sessions != test_session)[0]
    # Hold out 15% of train sessions' data as validation (stratify by remaining sessions)
    rng = np.random.RandomState(42 + test_session)
    rng.shuffle(train_pool)
    n_val = max(1, int(0.15 * len(train_pool)))
    val_idx = train_pool[:n_val]
    train_idx = train_pool[n_val:]
    return train_idx, val_idx, test_idx


def make_loaders(
    data: dict[str, Any],
    test_session: int,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    train_idx, val_idx, test_idx = make_loso_splits(data["sessions"], test_session)
    train_ds = MultimodalEmotionDataset(data, train_idx)
    val_ds = MultimodalEmotionDataset(data, val_idx)
    test_ds = MultimodalEmotionDataset(data, test_idx)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    info = {
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "test_session": test_session,
    }
    return train_loader, val_loader, test_loader, info
