"""Evaluation under simulated resource constraints + reporting."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from train import evaluate_model, measure_efficiency, move_batch


CONSTRAINT_PROFILES = {
    "unconstrained": {"max_latency_ms": 1e9, "max_memory_mb": 1e9, "max_params": 1e12},
    "mobile": {"max_latency_ms": 40.0, "max_memory_mb": 80.0, "max_params": 1_500_000},
    "edge": {"max_latency_ms": 25.0, "max_memory_mb": 40.0, "max_params": 800_000},
    "ultra_light": {"max_latency_ms": 15.0, "max_memory_mb": 20.0, "max_params": 400_000},
}


def meets_constraints(eff: dict, profile: dict) -> bool:
    return (
        eff["latency_ms_mean"] <= profile["max_latency_ms"]
        and eff["peak_memory_mb"] <= profile["max_memory_mb"]
        and eff["num_params"] <= profile["max_params"]
    )


def estimate_flops_proxy(num_params: int, d_model: int, n_layers: int, n_mod: int = 3) -> float:
    """Rough FLOPs proxy for one forward (matmul-dominated)."""
    # graph attn ~ O(L * N^2 * D + L * N * D^2)
    graph = n_layers * (n_mod * n_mod * d_model + n_mod * d_model * d_model)
    mlp = n_layers * n_mod * (2 * d_model * d_model)
    cls = d_model * 4  # classifier rough
    return float(graph + mlp + cls + num_params * 0.01)


@torch.no_grad()
def collect_modality_stats(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    weights = []
    sparsities = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        if out.get("modality_weights") is not None:
            weights.append(out["modality_weights"].cpu().numpy())
        if out.get("graph_sparsity") is not None:
            s = out["graph_sparsity"]
            sparsities.append(float(s.detach().cpu()) if torch.is_tensor(s) else float(s))
    w = np.concatenate(weights) if weights else None
    return {
        "mean_modality_weights": w.mean(axis=0).tolist() if w is not None else None,
        "std_modality_weights": w.std(axis=0).tolist() if w is not None else None,
        "all_modality_weights": w,
        "mean_graph_sparsity": float(np.mean(sparsities)) if sparsities else 0.0,
    }


def full_eval_suite(
    teacher: nn.Module,
    student: nn.Module,
    test_loader,
    device: torch.device,
    teacher_history: dict,
    student_history: dict,
    fold_info: dict,
) -> dict[str, Any]:
    t_metrics = evaluate_model(teacher, test_loader, device)
    s_metrics = evaluate_model(student, test_loader, device)
    t_eff = measure_efficiency(teacher, test_loader, device)
    s_eff = measure_efficiency(student, test_loader, device)
    s_mod = collect_modality_stats(student, test_loader, device)

    t_eff["flops_proxy"] = estimate_flops_proxy(
        t_eff["num_params"], d_model=256, n_layers=3
    )
    s_eff["flops_proxy"] = estimate_flops_proxy(
        s_eff["num_params"], d_model=96, n_layers=2
    )

    constraint_results = {}
    for name, profile in CONSTRAINT_PROFILES.items():
        constraint_results[name] = {
            "teacher_feasible": meets_constraints(t_eff, profile),
            "student_feasible": meets_constraints(s_eff, profile),
            "profile": profile,
        }

    # Compression ratios
    compression = {
        "param_reduction": 1.0 - s_eff["num_params"] / max(t_eff["num_params"], 1),
        "size_reduction": 1.0 - s_eff["model_size_mb"] / max(t_eff["model_size_mb"], 1e-6),
        "latency_speedup": t_eff["latency_ms_mean"] / max(s_eff["latency_ms_mean"], 1e-6),
        "memory_reduction": 1.0 - s_eff["peak_memory_mb"] / max(t_eff["peak_memory_mb"], 1e-6),
        "acc_gap": t_metrics["accuracy"] - s_metrics["accuracy"],
        "f1_gap": t_metrics["f1_weighted"] - s_metrics["f1_weighted"],
    }

    return {
        "fold": fold_info,
        "teacher": {
            "metrics": {k: v for k, v in t_metrics.items() if k not in ("y_true", "y_pred", "modality_weights")},
            "efficiency": t_eff,
            "history": {k: v for k, v in teacher_history.items() if k not in ("params",)},
            "y_true": t_metrics["y_true"],
            "y_pred": t_metrics["y_pred"],
            "modality_weights": t_metrics.get("modality_weights"),
        },
        "student": {
            "metrics": {k: v for k, v in s_metrics.items() if k not in ("y_true", "y_pred", "modality_weights")},
            "efficiency": s_eff,
            "history": {k: v for k, v in student_history.items() if k not in ("params",)},
            "y_true": s_metrics["y_true"],
            "y_pred": s_metrics["y_pred"],
            "modality_weights": s_metrics.get("modality_weights"),
            "modality_stats": {
                "mean_modality_weights": s_mod["mean_modality_weights"],
                "std_modality_weights": s_mod["std_modality_weights"],
                "mean_graph_sparsity": s_mod["mean_graph_sparsity"],
            },
            "all_modality_weights": s_mod["all_modality_weights"],
        },
        "constraints": constraint_results,
        "compression": compression,
    }
