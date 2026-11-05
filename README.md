# Lightweight Graph-Based Multimodal Emotion Recognition (IEMOCAP)
#
# Pipeline overview
# -----------------
# 1. Parse IEMOCAP Sessions (audio wav, MoCap visual, transcriptions, EmoEvaluation)
# 2. Extract multimodal features (MFCC stats, MoCap stats, TF-IDF+SVD text)
# 3. Train dense Graph-Attention **Teacher**
# 4. Distill into compact **Student** with:
#      - Dynamic graph pruning
#      - Sparse attention
#      - Adaptive per-sample modality selection
#      - Knowledge distillation (soft logits + feature matching)
# 5. Evaluate recognition + efficiency under resource constraint profiles
# 6. Export plots + RESULTS_SUMMARY.md
#
# Quick start
# -----------
#   cd lightweight_mer
#   python run_pipeline.py
#
# Smoke test (subset):
#   python run_pipeline.py --max-utterances 400 --epochs-teacher 5 --epochs-student 8
#
# Single LOSO fold:
#   python run_pipeline.py --fold 1
#
# Outputs
# -------
#   outputs/features/iemocap_features.npz
#   outputs/checkpoints/
#   outputs/figures/*.png
#   outputs/results.json
#   outputs/metrics_summary.csv
#   outputs/RESULTS_SUMMARY.md
