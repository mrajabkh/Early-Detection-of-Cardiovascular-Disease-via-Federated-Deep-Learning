# gru_risk.py
# Simple supervised GRU for early detection on windowed time series
# NEW:
# - Optional LayerNorm on GRU outputs
# - Optional attention pooling head (auxiliary) for interpretability / sequence summary

from __future__ import annotations

import torch
import torch.nn as nn


class GRURisk(nn.Module):
    """
    GRU-based risk predictor.
    Input:  (B, T, D)

    Main output:
      logits_ts: (B, T)  per-timestep logits (unchanged pipeline)

    Optional attention pooling head (for XAI + optional auxiliary loss):
      attn_weights: (B, T) attention over timesteps
      logit_seq: (B,) pooled sequence logit
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        use_layernorm: bool = True,
        use_attention_pooling: bool = True,
        attn_hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.use_layernorm = bool(use_layernorm)
        self.use_attention_pooling = bool(use_attention_pooling)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
            bidirectional=False,
        )

        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # Per-timestep head (main)
        self.head_ts = nn.Linear(hidden_dim, 1)

        # Attention pooling head (aux)
        if self.use_attention_pooling:
            self.attn_proj = nn.Linear(hidden_dim, attn_hidden_dim)
            self.attn_v = nn.Linear(attn_hidden_dim, 1, bias=False)
            self.head_seq = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def _make_time_mask(self, lengths: torch.Tensor, t_max: int) -> torch.Tensor:
        """
        lengths: (B,)
        returns mask: (B, T) with True for valid timesteps.
        """
        # shape (T,)
        ar = torch.arange(t_max, device=lengths.device).unsqueeze(0)
        # (B, T)
        return ar < lengths.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict:
        """
        x: (B, T, D)
        lengths: (B,) true sequence lengths (optional)

        returns dict with:
          logits_ts: (B, T)
          attn_weights: (B, T) or None
          logit_seq: (B,) or None
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
            )
        else:
            out, _ = self.gru(x)

        # out: (B, T, H)
        if self.use_layernorm:
            out = self.layernorm(out)

        out = self.dropout(out)

        # Main per-timestep logits
        logits_ts = self.head_ts(out).squeeze(-1)  # (B, T)

        attn_weights = None
        logit_seq = None

        if self.use_attention_pooling:
            # Compute attention scores over time
            # score_t = v^T tanh(W h_t)
            h = torch.tanh(self.attn_proj(out))          # (B, T, A)
            scores = self.attn_v(h).squeeze(-1)          # (B, T)

            # Mask padding before softmax
            if lengths is not None:
                t_max = scores.size(1)
                time_mask = self._make_time_mask(lengths, t_max)  # (B, T) bool
                scores = scores.masked_fill(~time_mask, -1e9)

            attn_weights = torch.softmax(scores, dim=1)  # (B, T)

            # Context vector: sum_t a_t * h_t
            context = torch.sum(attn_weights.unsqueeze(-1) * out, dim=1)  # (B, H)

            # Sequence logit
            logit_seq = self.head_seq(context).squeeze(-1)  # (B,)

        return {
            "logits_ts": logits_ts,
            "attn_weights": attn_weights,
            "logit_seq": logit_seq,
        }