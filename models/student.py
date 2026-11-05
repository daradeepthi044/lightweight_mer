"""Lightweight student with pruning, sparse attention, adaptive fusion."""
from __future__ import annotations

import torch
import torch.nn as nn

from config import (
    AUDIO_FEAT_DIM,
    NUM_CLASSES,
    TEXT_FEAT_DIM,
    VISUAL_FEAT_DIM,
    ModelConfig,
    TrainConfig,
)
from models.layers import (
    AdaptiveModalityGate,
    FeedForward,
    GraphAttentionLayer,
    ModalityProjector,
    SparseAttention,
    dynamic_graph_prune,
)


class StudentMER(nn.Module):
    """
    Compact student:
      - dynamic graph pruning of inter-modal edges
      - sparse attention on emotionally salient connections
      - adaptive per-sample modality selection
    """

    def __init__(self, mcfg: ModelConfig | None = None, tcfg: TrainConfig | None = None):
        super().__init__()
        mcfg = mcfg or ModelConfig()
        tcfg = tcfg or TrainConfig()
        self.mcfg = mcfg
        self.tcfg = tcfg
        self.projector = ModalityProjector(
            {"audio": AUDIO_FEAT_DIM, "visual": VISUAL_FEAT_DIM, "text": TEXT_FEAT_DIM},
            mcfg.d_model,
        )
        self.in_proj = nn.Linear(mcfg.d_model, mcfg.student_hidden)
        # Warm-up dense GAT to estimate edges, then pruned GAT
        self.probe_gat = GraphAttentionLayer(
            mcfg.student_hidden,
            mcfg.student_hidden,
            n_heads=mcfg.student_heads,
            dropout=mcfg.student_dropout,
        )
        self.sparse = SparseAttention(
            mcfg.student_hidden,
            n_heads=mcfg.student_heads,
            topk=tcfg.sparse_topk,
            dropout=mcfg.student_dropout,
        )
        self.pruned_layers = nn.ModuleList()
        for _ in range(mcfg.student_layers):
            self.pruned_layers.append(
                GraphAttentionLayer(
                    mcfg.student_hidden,
                    mcfg.student_hidden,
                    n_heads=mcfg.student_heads,
                    dropout=mcfg.student_dropout,
                )
            )
            self.pruned_layers.append(
                FeedForward(mcfg.student_hidden, mcfg.student_hidden * 2, mcfg.student_dropout)
            )
        self.gate = AdaptiveModalityGate(mcfg.student_hidden, mcfg.num_modalities)
        self.classifier = nn.Sequential(
            nn.Linear(mcfg.student_hidden, mcfg.student_hidden),
            nn.GELU(),
            nn.Dropout(mcfg.student_dropout),
            nn.Linear(mcfg.student_hidden, NUM_CLASSES),
        )
        # Align student features to teacher dim for KD (learned)
        self.feat_align = nn.Linear(mcfg.student_hidden, 256)  # teacher_hidden default
        self.out_dim = mcfg.student_hidden

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes = self.projector(
            {"audio": batch["audio"], "visual": batch["visual"], "text": batch["text"]}
        )
        nodes = self.in_proj(nodes)
        nodes, attn_probe = self.probe_gat(nodes, return_attn=True)
        adj = dynamic_graph_prune(attn_probe, keep_ratio=self.tcfg.prune_ratio)
        nodes = self.sparse(nodes)
        attn_last = attn_probe
        for layer in self.pruned_layers:
            if isinstance(layer, GraphAttentionLayer):
                nodes, attn_last = layer(nodes, adj_mask=adj, return_attn=True)
            else:
                nodes = layer(nodes)
        fused, weights = self.gate(nodes, batch.get("visual_mask"))
        sparsity = 1.0 - adj.mean()  # fraction pruned (approx)
        return fused, weights, attn_last, sparsity

    def forward(self, batch: dict) -> dict:
        fused, weights, attn, sparsity = self.encode(batch)
        logits = self.classifier(fused)
        aligned = self.feat_align(fused)
        return {
            "logits": logits,
            "features": fused,
            "features_aligned": aligned,
            "modality_weights": weights,
            "attn": attn,
            "graph_sparsity": sparsity,
        }


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    return (params + buffers) * 4 / (1024 ** 2)  # float32
