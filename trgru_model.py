"""Causal TrGRU adapted to the project's per-timestep prediction task.

The source paper specifies three Transformer encoder layers, two GRU layers,
global average pooling, and an MLP head. It does not report attention-head or
feed-forward dimensions, so those choices are explicit constructor settings.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrGRURisk(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        transformer_layers: int = 3,
        dim_feedforward: int = 256,
        gru_hidden_dim: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.2,
        max_len: int = 128,
        mlp_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(gru_hidden_dim),
            nn.Linear(gru_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        _, time_steps, _ = x.shape
        if time_steps > self.position_embedding.shape[1]:
            raise ValueError("Sequence exceeds configured TrGRU max_len")

        positions = self.position_embedding[:, :time_steps]
        encoded = self.input_projection(x) + positions

        # Prevent attention to future windows and ignore right-side padding.
        causal_mask = torch.triu(
            torch.ones(time_steps, time_steps, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        step_ids = torch.arange(time_steps, device=x.device)[None, :]
        padding_mask = step_ids >= lengths[:, None]
        encoded = self.transformer(
            encoded,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        recurrent, _ = self.gru(encoded)
        recurrent = recurrent.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Causal counterpart of global average pooling: prediction t pools 0..t.
        valid = (~padding_mask).unsqueeze(-1).to(recurrent.dtype)
        cumulative_sum = torch.cumsum(recurrent * valid, dim=1)
        cumulative_count = torch.cumsum(valid, dim=1).clamp_min(1.0)
        pooled = cumulative_sum / cumulative_count
        logits = self.head(pooled).squeeze(-1)
        return {"logits_ts": logits}
