"""
End-to-end pipeline:
  IEMOCAP -> features -> Teacher MER -> Student (prune/sparse/KD) -> eval + plots

Usage (from lightweight_mer/):
  python run_pipeline.py
  python run_pipeline.py --epochs-teacher 15 --epochs-student 20
  python run_pipeline.py --max-utterances 400   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure local imports resolve when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CHECKPOINT_DIR,
    CFG,
    EMOTION_NAMES,
    FEATURE_CACHE,
    FIGURE_DIR,
    IEMOCAP_ROOT,
    MCFG,
    METRICS_CSV,
    OUTPUT_DIR,
    RESULTS_JSON,
    TrainConfig,
)
from data.dataset import make_loaders
from data.features import extract_and_cache
from data.iemocap import build_utterance_index, summarize_index
from evaluate import full_eval_suite
from plots import generate_all_plots
from train import set_seed, train_student, train_teacher


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight Graph MER on IEMOCAP")
    p.add_argument("--iemocap-root", type=str, default=str(IEMOCAP_ROOT))
    p.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--epochs-teacher", type=int, default=CFG.teacher_epochs)
    p.add_argument("--epochs-student", type=int, default=CFG.student_epochs)
    p.add_argument("--batch-size", type=int, default=CFG.batch_size)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-utterances", type=int, default=None)
    p.add_argument("--force-features", action="store_true")
    p.add_argument("--fold", type=int, default=None, help="Run a single LOSO test session")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def aggregate_results(fold_results: list[dict]) -> dict:
    def mean_std(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    t_metrics = {m: [] for m in ["accuracy", "precision", "recall", "f1_weighted", "f1_macro"]}
    s_metrics = {m: [] for m in t_metrics}
    for r in fold_results:
        for m in t_metrics:
            t_metrics[m].append(r["teacher"]["metrics"][m])
            s_metrics[m].append(r["student"]["metrics"][m])

    # Efficiency averaged
    t_eff_keys = fold_results[0]["teacher"]["efficiency"].keys()
    s_eff_keys = fold_results[0]["student"]["efficiency"].keys()
    t_eff = {k: float(np.mean([r["teacher"]["efficiency"][k] for r in fold_results])) for k in t_eff_keys}
    s_eff = {k: float(np.mean([r["student"]["efficiency"][k] for r in fold_results])) for k in s_eff_keys}

    compression = {
        k: float(np.mean([r["compression"][k] for r in fold_results]))
        for k in fold_results[0]["compression"]
    }

    teacher_agg = {m: mean_std(v)["mean"] for m, v in t_metrics.items()}
    student_agg = {m: mean_std(v)["mean"] for m, v in s_metrics.items()}
    teacher_std = {m: mean_std(v)["std"] for m, v in t_metrics.items()}
    student_std = {m: mean_std(v)["std"] for m, v in s_metrics.items()}

    return {
        "teacher": teacher_agg,
        "student": student_agg,
        "teacher_std": teacher_std,
        "student_std": student_std,
        "teacher_eff": t_eff,
        "student_eff": s_eff,
        "compression": compression,
        "n_folds": len(fold_results),
        "emotion_names": EMOTION_NAMES,
    }


def write_reports(fold_results: list[dict], agg: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON-serializable copy (drop large arrays)
    serializable = []
    for r in fold_results:
        serializable.append(
            {
                "fold": r["fold"],
                "teacher": {
                    "metrics": r["teacher"]["metrics"],
                    "efficiency": r["teacher"]["efficiency"],
                    "history": r["teacher"]["history"],
                },
                "student": {
                    "metrics": r["student"]["metrics"],
                    "efficiency": r["student"]["efficiency"],
                    "history": r["student"]["history"],
                    "modality_stats": r["student"]["modality_stats"],
                },
                "constraints": r["constraints"],
                "compression": r["compression"],
            }
        )
    payload = {"folds": serializable, "aggregate": agg}
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = []
    for r in fold_results:
        sid = r["fold"]["test_session"]
        rows.append(
            {
                "model": "teacher",
                "test_session": sid,
                **r["teacher"]["metrics"],
                **{f"eff_{k}": v for k, v in r["teacher"]["efficiency"].items()},
            }
        )
        rows.append(
            {
                "model": "student",
                "test_session": sid,
                **r["student"]["metrics"],
                **{f"eff_{k}": v for k, v in r["student"]["efficiency"].items()},
            }
        )
    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)

    # Human-readable summary
    summary = OUTPUT_DIR / "RESULTS_SUMMARY.md"
    lines = [
        "# Lightweight Graph-Based MER — Results Summary",
        "",
        f"Dataset: IEMOCAP (Sessions {sorted({r['fold']['test_session'] for r in fold_results})} LOSO)",
        f"Classes: {', '.join(EMOTION_NAMES)}",
        "",
        "## Aggregate recognition metrics (mean over LOSO folds)",
        "",
        "| Model | Accuracy | Precision | Recall | F1-W | F1-Macro |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Teacher | {agg['teacher']['accuracy']:.4f} | {agg['teacher']['precision']:.4f} | "
            f"{agg['teacher']['recall']:.4f} | {agg['teacher']['f1_weighted']:.4f} | "
            f"{agg['teacher']['f1_macro']:.4f} |"
        ),
        (
            f"| Student | {agg['student']['accuracy']:.4f} | {agg['student']['precision']:.4f} | "
            f"{agg['student']['recall']:.4f} | {agg['student']['f1_weighted']:.4f} | "
            f"{agg['student']['f1_macro']:.4f} |"
        ),
        "",
        "## Efficiency",
        "",
        "| Model | Params | Size (MB) | Latency (ms) | Memory (MB) | FLOPs proxy |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Teacher | {agg['teacher_eff']['num_params']:,.0f} | {agg['teacher_eff']['model_size_mb']:.2f} | "
            f"{agg['teacher_eff']['latency_ms_mean']:.3f} | {agg['teacher_eff']['peak_memory_mb']:.2f} | "
            f"{agg['teacher_eff']['flops_proxy']:.0f} |"
        ),
        (
            f"| Student | {agg['student_eff']['num_params']:,.0f} | {agg['student_eff']['model_size_mb']:.2f} | "
            f"{agg['student_eff']['latency_ms_mean']:.3f} | {agg['student_eff']['peak_memory_mb']:.2f} | "
            f"{agg['student_eff']['flops_proxy']:.0f} |"
        ),
        "",
        "## Compression (Student vs Teacher)",
        "",
        f"- Parameter reduction: {agg['compression']['param_reduction']*100:.1f}%",
        f"- Model size reduction: {agg['compression']['size_reduction']*100:.1f}%",
        f"- Latency speedup: {agg['compression']['latency_speedup']:.2f}x",
        f"- Memory reduction: {agg['compression']['memory_reduction']*100:.1f}%",
        f"- Accuracy gap: {agg['compression']['acc_gap']:.4f}",
        f"- F1 gap: {agg['compression']['f1_gap']:.4f}",
        "",
        f"Figures: `{FIGURE_DIR}`",
        f"Raw JSON: `{RESULTS_JSON}`",
        f"CSV: `{METRICS_CSV}`",
    ]
    summary.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))


def run_fold(
    data: dict,
    test_session: int,
    cfg: TrainConfig,
) -> dict:
    print(f"\n======== LOSO fold: test Session {test_session} ========")
    train_loader, val_loader, test_loader, info = make_loaders(
        data, test_session, batch_size=cfg.batch_size
    )
    print(f"Split sizes: {info}")

    ck_t = CHECKPOINT_DIR / f"teacher_s{test_session}.pt"
    ck_s = CHECKPOINT_DIR / f"student_s{test_session}.pt"

    teacher, th = train_teacher(train_loader, val_loader, cfg, MCFG, save_path=ck_t)
    student, sh = train_student(teacher, train_loader, val_loader, cfg, MCFG, save_path=ck_s)

    device = torch.device(cfg.device)
    result = full_eval_suite(teacher, student, test_loader, device, th, sh, info)
    print(
        f"Fold S{test_session}: Teacher Acc={result['teacher']['metrics']['accuracy']:.4f} | "
        f"Student Acc={result['student']['metrics']['accuracy']:.4f} | "
        f"Speedup={result['compression']['latency_speedup']:.2f}x"
    )
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        teacher_epochs=args.epochs_teacher,
        student_epochs=args.epochs_student,
        device=args.device,
        sessions=list(args.sessions),
        fold_session=args.fold,
        max_utterances=args.max_utterances,
    )

    print("=" * 72)
    print("Lightweight, Graph-Based Multimodal Emotion Recognition")
    print("=" * 72)
    print(f"IEMOCAP root: {args.iemocap_root}")
    print(f"Device: {cfg.device}")
    print(f"Sessions: {cfg.sessions}")

    # Quick index summary
    items = build_utterance_index(
        sessions=cfg.sessions,
        iemocap_root=Path(args.iemocap_root),
        max_utterances=cfg.max_utterances,
    )
    summary = summarize_index(items)
    print(f"Utterance index: {json.dumps(summary, indent=2)}")
    if summary["n_utterances"] == 0:
        raise SystemExit("No utterances found. Verify IEMOCAP path.")

    # Override config path if custom root
    import config as config_mod

    config_mod.IEMOCAP_ROOT = Path(args.iemocap_root)

    print("\n[1/4] Feature extraction / cache load...")
    data = extract_and_cache(
        sessions=cfg.sessions,
        cache_path=FEATURE_CACHE if cfg.max_utterances is None else OUTPUT_DIR / "features" / "iemocap_features_subset.npz",
        max_utterances=cfg.max_utterances,
        force=args.force_features,
        iemocap_root=Path(args.iemocap_root),
    )
    print(
        f"Features ready: N={len(data['labels'])}, "
        f"audio={data['audio'].shape}, visual={data['visual'].shape}, text={data['text'].shape}"
    )

    folds = [args.fold] if args.fold is not None else list(cfg.sessions)
    # Only folds that exist in data
    available = sorted(set(int(s) for s in data["sessions"]))
    folds = [f for f in folds if f in available]
    if not folds:
        raise SystemExit(f"No valid folds. Available sessions in data: {available}")

    print(f"\n[2/4] Training teacher + student over LOSO folds: {folds}")
    fold_results = []
    for sid in folds:
        fold_results.append(run_fold(data, sid, cfg))

    print("\n[3/4] Aggregating metrics...")
    agg = aggregate_results(fold_results)
    write_reports(fold_results, agg)

    print("\n[4/4] Generating plots...")
    paths = generate_all_plots(fold_results, agg)
    print(f"Generated {len(paths)} figures in {FIGURE_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
