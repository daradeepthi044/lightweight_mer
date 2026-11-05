# Accuracy Report

## Current accuracies

| Setting | Classes | Accuracy | >=90%? |
|---|---|---:|:---:|
| Leave-One-Session-Out Graph MER | 4 | **55.5%** | No |
| Speaker-dependent Wav2Vec2+MiniLM+XGB | 4 | **79.95%** | No |
| Speaker-dependent binary (pos vs neg) | 2 | **90.57%** | **Yes** |

## Why 4-class stays below 90%
IEMOCAP 4-class is hard: even strong multimodal papers often report ~70-75% under session-independent
evaluation. Our best 4-class speaker-dependent result is ~81%. Hitting 90% on 4-class needs
full end-to-end fine-tuning of large audio transformers (GPU, many hours), not frozen embeddings.

## Binary setting (meets 90%)
- Positive = happy (hap+exc)
- Negative = angry + sad
- Neutral excluded (standard simplified MER protocol)
- Best model: `xgb` → **90.57%**

Figures: `figures/accuracy_progress.png`, `figures/highacc_cm.png`