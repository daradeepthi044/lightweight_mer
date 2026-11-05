# Architecture: Lightweight Graph-Based Multimodal Emotion Recognition

## Dataset (IEMOCAP)

- **Path**: `IEMOCAP/IEMOCAP_full_release` (Sessions 1–4 available in this workspace; Session 5 not present)
- **Modalities**
  - **Audio**: utterance WAVs under `SessionX/sentences/wav/`
  - **Visual**: facial MoCap under `SessionX/sentences/MOCAP_rotated/`
  - **Text**: dialogue transcriptions under `SessionX/dialog/transcriptions/`
  - **Labels**: `SessionX/dialog/EmoEvaluation/*.txt`
- **Task**: 4-class categorical emotion  
  `angry (ang)`, `happy (hap+exc)`, `sad (sad)`, `neutral (neu)`  
  Utterances without majority labels (`xxx`, `fru`, …) are excluded → **4290** utterances.

## Feature extraction

| Modality | Features | Dim |
|---|---|---|
| Audio | 40 MFCCs: mean, std, mean\|Δ\| | 120 |
| Visual | MoCap XYZ marker stats (subsampled frames) | 165 |
| Text | TF-IDF (1–2 grams) + Truncated SVD | 100 |

Features are z-scored (audio/visual) and cached to `outputs/features/iemocap_features.npz`.

## Protocol

**Leave-One-Session-Out (LOSO)** over Sessions {1,2,3,4}:
- Test = held-out session
- Train/Val = remaining sessions (85%/15%)

## Teacher (dense graph MER)

1. Per-modality linear projectors → shared `d_model`
2. Stack of **dense Graph Attention** layers + FFNs (full cross-modal connectivity)
3. **Adaptive modality gate** (softmax over audio/visual/text scores; visual masked if missing)
4. MLP classifier

Defaults: hidden 256, 3 GAT layers, 4 heads → ~1.37M params.

## Student (lightweight)

1. Same projectors (smaller hidden 96)
2. Probe GAT → **dynamic graph pruning** (keep top-k inter-modal edges + self-loops)
3. **Sparse attention** (top-k keys)
4. Pruned GAT stack (2 layers, 2 heads)
5. Adaptive modality gate + classifier
6. **Knowledge distillation** from teacher:
   - Hard CE on labels
   - Soft KL on temperature-scaled logits
   - Cosine feature matching (aligned student → teacher space)

Defaults: ~271K params (~80% reduction).

## Resource constraint profiles

Evaluated feasibility for `unconstrained`, `mobile`, `edge`, `ultra_light` budgets on latency / memory / params.

## How to run

```bash
cd lightweight_mer
python run_pipeline.py
# or regenerate reports from checkpoints:
python reeval_from_checkpoints.py
```

## Outputs

- `outputs/RESULTS_SUMMARY.md` — metrics tables
- `outputs/results.json` / `metrics_summary.csv`
- `outputs/figures/*.png` — training curves, confusion matrices, efficiency, Pareto, modality gates, LOSO heatmaps, constraints
- `outputs/checkpoints/teacher_s{k}.pt`, `student_s{k}.pt`
