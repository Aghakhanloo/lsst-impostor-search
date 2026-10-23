"""A small transformer for irregular, multi-band light curves.

Each observation becomes one token: (magnitude relative to the median, error)
plus a learned band embedding and a time encoding with sinusoids of fixed
periods from 1 to 2000 days, so irregular sampling needs no interpolation.
Tokens are mean-pooled and combined with context features (here the median
absolute magnitude and distance modulus from the host galaxy).
"""
import math
import torch
import torch.nn as nn


class TimeEncoding(nn.Module):
    def __init__(self, d_model, n_freq=16, min_period=1.0, max_period=2000.0):
        super().__init__()
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), n_freq)
        self.register_buffer("omega", 2 * math.pi / periods)
        self.proj = nn.Linear(2 * n_freq, d_model)

    def forward(self, t):                      # t: (batch, length) in days
        ang = t.unsqueeze(-1) * self.omega
        return self.proj(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))


class LightCurveTransformer(nn.Module):
    def __init__(self, n_bands=3, n_ctx=2, n_classes=3, d_model=64, n_heads=4,
                 n_layers=3, dropout=0.1):
        super().__init__()
        self.value = nn.Linear(2, d_model)
        self.band = nn.Embedding(n_bands, d_model)
        self.time = TimeEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.ctx = nn.Sequential(nn.Linear(n_ctx, d_model), nn.GELU()) if n_ctx else None
        head_in = d_model * (2 if n_ctx else 1)
        self.head = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, n_classes))

    def forward(self, t, band, x, mask, ctx=None):
        h = self.value(x) + self.band(band) + self.time(t)
        h = self.encoder(h, src_key_padding_mask=~mask)
        h = self.norm(h)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        if self.ctx is not None:
            pooled = torch.cat([pooled, self.ctx(ctx)], dim=-1)
        return self.head(pooled)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
