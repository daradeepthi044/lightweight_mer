# Lightweight Graph-Based MER — Results Summary

Dataset: IEMOCAP (Sessions 1-4, Leave-One-Session-Out)  
Classes: angry, happy, sad, neutral  
Utterances: 4290 (Session5 not present in this release folder)

## Aggregate recognition metrics (mean over LOSO folds)

| Model | Accuracy | Precision | Recall | F1-W | F1-Macro |
|---|---:|---:|---:|---:|---:|
| Teacher | 0.5549 | 0.5861 | 0.5549 | 0.5474 | 0.5425 |
| Student | 0.5536 | 0.5791 | 0.5536 | 0.5474 | 0.5477 |

Per-fold test accuracy (Teacher / Student):

| Test session | Teacher Acc | Teacher F1-W | Student Acc | Student F1-W |
|---|---:|---:|---:|---:|
| 1 | 0.544 | 0.519 | 0.578 | 0.562 |
| 2 | 0.586 | 0.583 | 0.583 | 0.579 |
| 3 | 0.606 | 0.607 | 0.561 | 0.557 |
| 4 | 0.484 | 0.481 | 0.493 | 0.492 |

## Efficiency

| Model | Params | Size (MB) | Latency (ms/sample) | Memory (MB) | FLOPs proxy |
|---|---:|---:|---:|---:|---:|
| Teacher | 1,370,629 | 5.23 | 0.091 | 13.07 | 1.79e6 |
| Student | 270,917 | 1.03 | 0.077 | 2.58 | 1.71e5 |

## Compression (Student vs Teacher)

- Parameter reduction: **80.2%**
- Model size reduction: **80.2%**
- Latency speedup: **1.20x**
- Memory reduction: **80.2%**
- Accuracy gap: **0.0013** (negligible)
- F1 gap: **~0.0000**
- Mean graph sparsity (student): **0.333** (1/3 of inter-modal edges pruned on average)

## Adaptive modality weights (student, example fold means)

Audio / Visual / Text typically around **0.33 / 0.38 / 0.28**, with visual often highest when MoCap is available.

## Resource-constraint feasibility

Student passes mobile/edge budgets more readily than the dense teacher (see `figures/constraint_feasibility.png`).

## Artifacts

- Figures: `outputs/figures/`
- Raw JSON: `outputs/results.json`
- CSV: `outputs/metrics_summary.csv`
- Checkpoints: `outputs/checkpoints/`
- Architecture notes: `ARCHITECTURE.md`


## High-accuracy run (speaker-dependent)
- Best accuracy: **80.42%** (xgboost)
- 90% target met: NO
- Details: `HIGHACC_SUMMARY.md`
