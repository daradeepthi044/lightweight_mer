"""Heavy teacher graph multimodal emotion model."""
from __future__ import annotations

import torch
import torch.nn as nn

from config import AUDIO_FEAT_DIM, NUM_CLASSES, TEXT_FEAT_DIM, VISUAL_FEAT_DIM, ModelConfig
from models.layers import (
    AdaptiveModalityGate,
    FeedForward,
    GraphAttentionLayer,
    ModalityProjector,
)


class TeacherMER(nn.Module):
    """Dense multi-layer graph attention teacher with full cross-modal connectivity."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.projector = ModalityProjector(
            {"audio": AUDIO_FEAT_DIM, "visual": VISUAL_FEAT_DIM, "text": TEXT_FEAT_DIM},
            cfg.d_model,
        )
        layers = []
        dims = [cfg.d_model] + [cfg.teacher_hidden] * (cfg.teacher_layers - 1) + [cfg.d_model]
        # Align dims: project once to teacher_hidden then keep
        self.in_proj = nn.Linear(cfg.d_model, cfg.teacher_hidden)
        for _ in range(cfg.teacher_layers):
            layers.append(
                GraphAttentionLayer(
                    cfg.teacher_hidden,
                    cfg.teacher_hidden,
                    n_heads=cfg.teacher_heads,
                    dropout=cfg.teacher_dropout,
                )
            )
            layers.append(FeedForward(cfg.teacher_hidden, cfg.teacher_hidden * 2, cfg.teacher_dropout))
        self.layers = nn.ModuleList(layers)
        self.gate = AdaptiveModalityGate(cfg.teacher_hidden, cfg.num_modalities)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.teacher_hidden, cfg.teacher_hidden),
            nn.GELU(),
            nn.Dropout(cfg.teacher_dropout),
            nn.Linear(cfg.teacher_hidden, NUM_CLASSES),
        )
        self.out_dim = cfg.teacher_hidden

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes = self.projector(
            {"audio": batch["audio"], "visual": batch["visual"], "text": batch["text"]}
        )
        nodes = self.in_proj(nodes)
        attn_last = None
        for layer in self.layers:
            if isinstance(layer, GraphAttentionLayer):
                nodes, attn_last = layer(nodes, adj_mask=None, return_attn=True)
            else:
                nodes = layer(nodes)
        fused, weights = self.gate(nodes, batch.get("visual_mask"))
        return fused, weights, attn_last

    def forward(self, batch: dict) -> dict:
        fused, weights, attn = self.encode(batch)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "features": fused,
            "modality_weights": weights,
            "attn": attn,
        }
