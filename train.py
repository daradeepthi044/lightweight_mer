"""Training loops for teacher and distilled student."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

from config import CHECKPOINT_DIR, ModelConfig, TrainConfig
from models.student import StudentMER, count_parameters, model_size_mb
from models.teacher import TeacherMER


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, device: torch.device) -> dict[str, Any]:
    model.eval()
    ys, preds = [], []
    losses = []
    mod_w = []
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        loss = criterion(out["logits"], batch["label"])
        losses.append(loss.item())
        pred = out["logits"].argmax(dim=-1)
        ys.append(batch["label"].cpu().numpy())
        preds.append(pred.cpu().numpy())
        if out.get("modality_weights") is not None:
            mod_w.append(out["modality_weights"].cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, p, average="weighted", zero_division=0
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y, p, average="macro", zero_division=0
    )
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(prec),
        "recall": float(rec),
        "f1_weighted": float(f1),
        "f1_macro": float(f1_macro),
        "y_true": y,
        "y_pred": p,
        "modality_weights": np.concatenate(mod_w) if mod_w else None,
    }


def train_teacher(
    train_loader,
    val_loader,
    cfg: TrainConfig,
    mcfg: ModelConfig,
    save_path: Path | None = None,
) -> tuple[TeacherMER, dict]:
    device = torch.device(cfg.device)
    model = TeacherMER(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_teacher, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.teacher_epochs)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_state = None
    patience = 0

    for epoch in range(1, cfg.teacher_epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Teacher ep{epoch}", leave=False):
            batch = move_batch(batch, device)
            opt.zero_grad()
            out = model(batch)
            loss = criterion(out["logits"], batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        val = evaluate_model(model, val_loader, device)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["val_f1"].append(val["f1_weighted"])
        print(
            f"[Teacher] epoch {epoch:02d}  train_loss={np.mean(losses):.4f}  "
            f"val_acc={val['accuracy']:.4f}  val_f1={val['f1_weighted']:.4f}"
        )
        if val["f1_weighted"] > best_f1:
            best_f1 = val["f1_weighted"]
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("Early stopping teacher.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    save_path = save_path or (CHECKPOINT_DIR / "teacher_best.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": history, "params": count_parameters(model)}, save_path)
    history["params"] = count_parameters(model)
    history["size_mb"] = model_size_mb(model)
    return model, history


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    temperature: float,
    alpha: float,
    feat_weight: float,
) -> torch.Tensor:
    ce = F.cross_entropy(student_logits, labels)
    T = temperature
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    # feature matching (cosine)
    s = F.normalize(student_feat, dim=-1)
    t = F.normalize(teacher_feat.detach(), dim=-1)
    feat = (1 - (s * t).sum(dim=-1)).mean()
    return alpha * soft + (1 - alpha) * ce + feat_weight * feat


def train_student(
    teacher: TeacherMER,
    train_loader,
    val_loader,
    cfg: TrainConfig,
    mcfg: ModelConfig,
    save_path: Path | None = None,
) -> tuple[StudentMER, dict]:
    device = torch.device(cfg.device)
    teacher.eval()

    student = StudentMER(mcfg, cfg).to(device)
    # Align feature projection to actual teacher out dim
    student.feat_align = nn.Linear(student.out_dim, teacher.out_dim).to(device)

    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr_student, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.student_epochs)
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0
    best_state = None
    patience = 0

    for epoch in range(1, cfg.student_epochs + 1):
        student.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Student ep{epoch}", leave=False):
            batch = move_batch(batch, device)
            opt.zero_grad()
            with torch.no_grad():
                t_out = teacher(batch)
            s_out = student(batch)
            loss = distillation_loss(
                s_out["logits"],
                t_out["logits"],
                batch["label"],
                s_out["features_aligned"],
                t_out["features"],
                cfg.kd_temperature,
                cfg.kd_alpha,
                cfg.feature_kd_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        val = evaluate_model(student, val_loader, device)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["val_f1"].append(val["f1_weighted"])
        print(
            f"[Student] epoch {epoch:02d}  train_loss={np.mean(losses):.4f}  "
            f"val_acc={val['accuracy']:.4f}  val_f1={val['f1_weighted']:.4f}"
        )
        if val["f1_weighted"] > best_f1:
            best_f1 = val["f1_weighted"]
            best_state = copy.deepcopy(student.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print("Early stopping student.")
                break

    if best_state is not None:
        student.load_state_dict(best_state)
    save_path = save_path or (CHECKPOINT_DIR / "student_best.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": student.state_dict(), "history": history, "params": count_parameters(student)},
        save_path,
    )
    history["params"] = count_parameters(student)
    history["size_mb"] = model_size_mb(student)
    return student, history


@torch.no_grad()
def measure_efficiency(model: nn.Module, loader, device: torch.device, warmup: int = 5) -> dict:
    model.eval()
    # warmup
    n = 0
    for batch in loader:
        batch = move_batch(batch, device)
        _ = model(batch)
        n += 1
        if n >= warmup:
            break

    latencies = []
    peak_mem = 0.0
    n_samples = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    for batch in loader:
        batch = move_batch(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        bs = batch["label"].shape[0]
        latencies.append((t1 - t0) / bs * 1000.0)  # ms / sample
        n_samples += bs
        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    # CPU RSS approximation via model size + activation estimate
    size_mb = model_size_mb(model)
    if device.type != "cuda":
        peak_mem = size_mb * 2.5  # rough working-set proxy

    return {
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_std": float(np.std(latencies)),
        "throughput_sps": float(1000.0 / max(np.mean(latencies), 1e-6)),
        "peak_memory_mb": float(peak_mem),
        "model_size_mb": float(size_mb),
        "num_params": int(count_parameters(model)),
        "n_samples_measured": int(n_samples),
    }
