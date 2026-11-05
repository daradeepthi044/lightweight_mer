"""Graph attention, sparse attention, pruning, and adaptive fusion layers."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityProjector(nn.Module):
    def __init__(self, in_dims: dict[str, int], d_model: int):
        super().__init__()
        self.projs = nn.ModuleDict(
            {k: nn.Sequential(nn.Linear(v, d_model), nn.LayerNorm(d_model), nn.GELU())
             for k, v in in_dims.items()}
        )

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        # Stack as [B, M, D]
        nodes = [self.projs[k](feats[k]) for k in ("audio", "visual", "text")]
        return torch.stack(nodes, dim=1)


class GraphAttentionLayer(nn.Module):
    """Dense multi-head graph attention over modality nodes."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_h = out_dim // n_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(n_heads, 2 * self.d_h))
        nn.init.xavier_uniform_(self.attn)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        # x: [B, N, Din]
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.n_heads, self.d_h).transpose(1, 2)  # [B,H,N,dh]
        # pairwise scores
        h_i = h.unsqueeze(3).expand(-1, -1, -1, N, -1)
        h_j = h.unsqueeze(2).expand(-1, -1, N, -1, -1)
        cat = torch.cat([h_i, h_j], dim=-1)  # [B,H,N,N,2dh]
        e = self.leaky((cat * self.attn.view(1, self.n_heads, 1, 1, -1)).sum(-1))
        if adj_mask is not None:
            # adj_mask: [B, N, N] 1=keep
            e = e.masked_fill(adj_mask.unsqueeze(1) <= 0, -1e9)
        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        out = torch.matmul(alpha, h)  # [B,H,N,dh]
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.norm(x + self.out_proj(out) if x.shape[-1] == out.shape[-1] else self.out_proj(out))
        if return_attn:
            return out, alpha.mean(dim=1)  # [B,N,N]
        return out


class SparseAttention(nn.Module):
    """Feature-wise sparse attention: keep only top-k keys per query."""

    def __init__(self, d_model: int, n_heads: int = 2, topk: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.topk = topk
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Treat last dim as sequence of feature tokens by reshaping chunks
        B, N, D = x.shape
        # Use modality nodes as sequence (already small N); sparseness on feature groups
        # Expand each node into G pseudo-tokens
        G = min(self.topk * 2, 16)
        # Project to tokens via learned reshape
        tokens = x  # [B,N,D]
        qkv = self.qkv(tokens)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.d_h).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_h).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_h).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_h)  # [B,H,N,N]
        # top-k over keys
        k_keep = min(self.topk, N)
        topv, topi = scores.topk(k_keep, dim=-1)
        mask = torch.zeros_like(scores).scatter_(-1, topi, 1.0)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return self.norm(tokens + self.out(out))


def dynamic_graph_prune(attn: torch.Tensor, keep_ratio: float = 0.45) -> torch.Tensor:
    """
    Prune inter-modal edges by keeping top-k absolute attention weights per node.
    attn: [B, N, N] -> binary adj [B, N, N]
    Always keep self-loops.
    """
    B, N, _ = attn.shape
    # Number of off-diagonal edges to keep per node
    k = max(1, int(round(keep_ratio * (N - 1))))
    k = min(k, N - 1)
    off = attn.clone()
    eye = torch.eye(N, device=attn.device).unsqueeze(0).expand(B, -1, -1)
    off = off.masked_fill(eye.bool(), -1e9)
    _, topi = off.topk(k, dim=-1)
    adj = torch.zeros_like(attn)
    adj.scatter_(-1, topi, 1.0)
    adj = torch.maximum(adj, eye)
    return adj


class AdaptiveModalityGate(nn.Module):
    """Per-sample modality importance weights."""

    def __init__(self, d_model: int, n_mod: int = 3):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.n_mod = n_mod

    def forward(
        self, nodes: torch.Tensor, visual_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # nodes [B,M,D]
        scores = self.scorer(nodes).squeeze(-1)  # [B,M]
        if visual_mask is not None:
            # visual is index 1; down-weight if missing
            penal = torch.ones_like(scores)
            penal[:, 1] = visual_mask.clamp(0, 1) + 1e-3
            scores = scores + torch.log(penal)
        weights = F.softmax(scores, dim=-1)
        fused = (nodes * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))
