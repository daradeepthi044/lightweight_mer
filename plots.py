"""Publication-style plots for Lightweight Graph MER experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import EMOTION_NAMES, FIGURE_DIR

sns.set_theme(style="whitegrid", context="talk")
PALETTE = ["#1B4F72", "#148F77", "#B9770E", "#922B21"]


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def plot_training_curves(results: list[dict], name: str = "training_curves.png") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for fold in results:
        sid = fold["fold"]["test_session"]
        th = fold["teacher"]["history"]
        sh = fold["student"]["history"]
        axes[0].plot(th["val_acc"], label=f"Teacher S{sid}", linewidth=2)
        axes[0].plot(sh["val_acc"], label=f"Student S{sid}", linewidth=2, linestyle="--")
        axes[1].plot(th["val_f1"], label=f"Teacher S{sid}", linewidth=2)
        axes[1].plot(sh["val_f1"], label=f"Student S{sid}", linewidth=2, linestyle="--")
    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Validation Weighted F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Training Dynamics (Leave-One-Session-Out)", fontsize=14, y=1.02)
    return _save(fig, name)


def plot_confusion(y_true, y_pred, title: str, name: str) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTION_NAMES))))
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=EMOTION_NAMES,
        yticklabels=EMOTION_NAMES,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return _save(fig, name)


def plot_metric_bars(agg: dict, name: str = "metric_comparison.png") -> Path:
    metrics = ["accuracy", "precision", "recall", "f1_weighted", "f1_macro"]
    labels = ["Accuracy", "Precision", "Recall", "F1-W", "F1-Macro"]
    t_vals = [agg["teacher"][m] for m in metrics]
    s_vals = [agg["student"][m] for m in metrics]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, t_vals, w, label="Teacher", color=PALETTE[0])
    ax.bar(x + w / 2, s_vals, w, label="Student", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Recognition Performance: Teacher vs Lightweight Student")
    ax.legend()
    for i, (tv, sv) in enumerate(zip(t_vals, s_vals)):
        ax.text(i - w / 2, tv + 0.02, f"{tv:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, sv + 0.02, f"{sv:.2f}", ha="center", fontsize=9)
    return _save(fig, name)


def plot_efficiency(agg: dict, name: str = "efficiency_comparison.png") -> Path:
    keys = [
        ("num_params", "Parameters", 1e-3, "K"),
        ("model_size_mb", "Model Size", 1.0, "MB"),
        ("latency_ms_mean", "Latency", 1.0, "ms"),
        ("peak_memory_mb", "Memory", 1.0, "MB"),
        ("flops_proxy", "FLOPs proxy", 1e-6, "M"),
    ]
    fig, axes = plt.subplots(1, len(keys), figsize=(15, 4))
    for ax, (k, title, scale, unit) in zip(axes, keys):
        t = agg["teacher_eff"][k] * scale
        s = agg["student_eff"][k] * scale
        ax.bar(["Teacher", "Student"], [t, s], color=[PALETTE[0], PALETTE[1]])
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.text(0, t, f"{t:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(1, s, f"{s:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Computational Efficiency", y=1.05)
    fig.tight_layout()
    return _save(fig, name)


def plot_pareto(agg: dict, name: str = "pareto_accuracy_latency.png") -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        [agg["teacher_eff"]["latency_ms_mean"]],
        [agg["teacher"]["accuracy"]],
        s=180,
        c=PALETTE[0],
        label="Teacher",
        zorder=3,
    )
    ax.scatter(
        [agg["student_eff"]["latency_ms_mean"]],
        [agg["student"]["accuracy"]],
        s=180,
        c=PALETTE[1],
        label="Student",
        zorder=3,
    )
    ax.annotate("Teacher", (agg["teacher_eff"]["latency_ms_mean"], agg["teacher"]["accuracy"]),
                textcoords="offset points", xytext=(8, 8))
    ax.annotate("Student", (agg["student_eff"]["latency_ms_mean"], agg["student"]["accuracy"]),
                textcoords="offset points", xytext=(8, 8))
    ax.set_xlabel("Inference latency (ms / sample)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy–Latency Trade-off")
    ax.legend()
    return _save(fig, name)


def plot_constraints(results_fold0: dict, name: str = "constraint_feasibility.png") -> Path:
    profiles = list(results_fold0["constraints"].keys())
    t_ok = [1 if results_fold0["constraints"][p]["teacher_feasible"] else 0 for p in profiles]
    s_ok = [1 if results_fold0["constraints"][p]["student_feasible"] else 0 for p in profiles]
    x = np.arange(len(profiles))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, t_ok, w, label="Teacher", color=PALETTE[0])
    ax.bar(x + w / 2, s_ok, w, label="Student", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(profiles, rotation=15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fail", "Pass"])
    ax.set_title("Feasibility under Simulated Resource Constraints")
    ax.legend()
    return _save(fig, name)


def plot_modality_weights(weights: np.ndarray, name: str = "adaptive_modality_weights.png") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    means = weights.mean(axis=0)
    axes[0].bar(["Audio", "Visual", "Text"], means, color=PALETTE[:3])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Mean Adaptive Modality Weights")
    axes[0].set_ylabel("Weight")
    for i, m in enumerate(means):
        axes[0].text(i, m + 0.02, f"{m:.2f}", ha="center")
    sns.violinplot(data=weights, ax=axes[1], palette=PALETTE[:3])
    axes[1].set_xticklabels(["Audio", "Visual", "Text"])
    axes[1].set_title("Per-sample Modality Selection Distribution")
    axes[1].set_ylabel("Weight")
    fig.tight_layout()
    return _save(fig, name)


def plot_session_heatmap(results: list[dict], name: str = "loso_session_heatmap.png") -> Path:
    sessions = [r["fold"]["test_session"] for r in results]
    metrics = ["accuracy", "f1_weighted"]
    data_t = np.array([[r["teacher"]["metrics"][m] for m in metrics] for r in results])
    data_s = np.array([[r["student"]["metrics"][m] for m in metrics] for r in results])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(
        data_t,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=["Acc", "F1-W"],
        yticklabels=[f"Test S{s}" for s in sessions],
        ax=axes[0],
        vmin=0.3,
        vmax=1.0,
    )
    axes[0].set_title("Teacher LOSO")
    sns.heatmap(
        data_s,
        annot=True,
        fmt=".3f",
        cmap="YlOrBr",
        xticklabels=["Acc", "F1-W"],
        yticklabels=[f"Test S{s}" for s in sessions],
        ax=axes[1],
        vmin=0.3,
        vmax=1.0,
    )
    axes[1].set_title("Student LOSO")
    fig.tight_layout()
    return _save(fig, name)


def plot_compression(agg: dict, name: str = "compression_gains.png") -> Path:
    c = agg["compression"]
    labels = [
        "Param\nreduction",
        "Size\nreduction",
        "Latency\nspeedup (x)",
        "Memory\nreduction",
    ]
    vals = [
        c["param_reduction"] * 100,
        c["size_reduction"] * 100,
        c["latency_speedup"],
        c["memory_reduction"] * 100,
    ]
    colors = [PALETTE[1]] * 4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_title("Student Compression Gains vs Teacher")
    ax.set_ylabel("% or speedup (x)")
    for b, v, lab in zip(bars, vals, labels):
        suffix = "x" if "speedup" in lab else "%"
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}{suffix}", ha="center", va="bottom")
    return _save(fig, name)


def plot_classwise(y_true, y_pred, title: str, name: str) -> Path:
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(EMOTION_NAMES))), zero_division=0
    )
    x = np.arange(len(EMOTION_NAMES))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w, p, w, label="Precision", color=PALETTE[0])
    ax.bar(x, r, w, label="Recall", color=PALETTE[1])
    ax.bar(x + w, f, w, label="F1", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_NAMES)
    ax.set_ylim(0, 1.15)
    ax.set_title(title)
    ax.legend()
    return _save(fig, name)


def generate_all_plots(fold_results: list[dict[str, Any]], agg: dict) -> list[Path]:
    paths = []
    paths.append(plot_training_curves(fold_results))
    paths.append(plot_metric_bars(agg))
    paths.append(plot_efficiency(agg))
    paths.append(plot_pareto(agg))
    paths.append(plot_constraints(fold_results[0]))
    paths.append(plot_session_heatmap(fold_results))
    paths.append(plot_compression(agg))

    # Aggregate predictions across folds for overall CM
    y_t = np.concatenate([r["teacher"]["y_true"] for r in fold_results])
    p_t = np.concatenate([r["teacher"]["y_pred"] for r in fold_results])
    y_s = np.concatenate([r["student"]["y_true"] for r in fold_results])
    p_s = np.concatenate([r["student"]["y_pred"] for r in fold_results])
    paths.append(plot_confusion(y_t, p_t, "Teacher Confusion Matrix (LOSO pooled)", "cm_teacher.png"))
    paths.append(plot_confusion(y_s, p_s, "Student Confusion Matrix (LOSO pooled)", "cm_student.png"))
    paths.append(plot_classwise(y_t, p_t, "Teacher Class-wise Metrics", "classwise_teacher.png"))
    paths.append(plot_classwise(y_s, p_s, "Student Class-wise Metrics", "classwise_student.png"))

    # Modality weights from last fold student
    w = fold_results[-1]["student"].get("all_modality_weights")
    if w is not None:
        paths.append(plot_modality_weights(w))

    return paths
