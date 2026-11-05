"""Re-evaluate saved checkpoints and regenerate reports/plots (no retraining)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CHECKPOINT_DIR, FEATURE_CACHE, MCFG, CFG, TrainConfig
from data.dataset import make_loaders
from data.features import extract_and_cache
from evaluate import full_eval_suite
from models.student import StudentMER
from models.teacher import TeacherMER
from plots import generate_all_plots
from run_pipeline import aggregate_results, write_reports
from train import set_seed


def main() -> None:
    set_seed(42)
    cfg = TrainConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(cfg.device)
    data = extract_and_cache(sessions=cfg.sessions, cache_path=FEATURE_CACHE, force=False)
    sessions = sorted(set(int(s) for s in data["sessions"]))

    fold_results = []
    for sid in sessions:
        train_loader, val_loader, test_loader, info = make_loaders(
            data, sid, batch_size=cfg.batch_size
        )
        teacher = TeacherMER(MCFG).to(device)
        student = StudentMER(MCFG, cfg).to(device)
        student.feat_align = torch.nn.Linear(student.out_dim, teacher.out_dim).to(device)

        t_ck = torch.load(CHECKPOINT_DIR / f"teacher_s{sid}.pt", map_location=device, weights_only=False)
        s_ck = torch.load(CHECKPOINT_DIR / f"student_s{sid}.pt", map_location=device, weights_only=False)
        teacher.load_state_dict(t_ck["state_dict"])
        student.load_state_dict(s_ck["state_dict"])
        # ensure all params countable
        for p in teacher.parameters():
            p.requires_grad = True

        th = t_ck.get("history", {})
        sh = s_ck.get("history", {})
        result = full_eval_suite(teacher, student, test_loader, device, th, sh, info)
        print(
            f"S{sid}: TAcc={result['teacher']['metrics']['accuracy']:.4f} "
            f"SAcc={result['student']['metrics']['accuracy']:.4f} "
            f"Tparams={result['teacher']['efficiency']['num_params']} "
            f"Sparams={result['student']['efficiency']['num_params']}"
        )
        fold_results.append(result)

    agg = aggregate_results(fold_results)
    write_reports(fold_results, agg)
    paths = generate_all_plots(fold_results, agg)
    print(f"Regenerated {len(paths)} figures.")


if __name__ == "__main__":
    main()
