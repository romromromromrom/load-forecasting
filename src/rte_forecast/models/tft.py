"""Temporal Fusion Transformer compact (PyTorch pur), prévision de charge 1 série -> H heures.

Implémentation COMPACTE des blocs de Lim et al. (2021), sans dépendance à `pytorch-forecasting`
(dont les versions cassent souvent sur Colab). Ce n'est pas la bibliothèque de référence : les
différences assumées sont
  * pas de variables statiques (une seule série nationale) ni de variables catégorielles
    (tout est continu ou binaire, chaque variable passe par sa propre projection linéaire) ;
  * une seule couche d'attention interprétable (valeur partagée entre têtes, têtes moyennées).

Blocs : GRN (gated residual network) -> sélection de variables (VSN) -> LSTM encodeur/décodeur ->
gate + add&norm -> enrichissement (GRN) -> attention interprétable masquée -> feed-forward GRN ->
gate + add&norm -> projection sur les quantiles.

Entrées :  ``past``   (B, L, n_past)    charge + covariables, connues jusqu'à l'ancre (incluse)
           ``future`` (B, H, n_future)  covariables CONNUES À L'AVANCE (calendrier, température prévue)
Sortie  :  (B, H, n_quantiles)          quantiles de la charge standardisée, triés (pas de croisement)

Ce module importe torch : il n'est volontairement pas importé par ``rte_forecast.models``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


class GateAddNorm(nn.Module):
    """LayerNorm(skip + GLU(dropout(x)))."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.glu = nn.Linear(d_model, 2 * d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(skip + F.glu(self.glu(self.drop(x)), dim=-1))


class GRN(nn.Module):
    """Gated residual network : ELU(fc1) -> fc2 -> gate GLU -> add & norm sur la connexion résiduelle."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d_in, d_hidden), nn.Linear(d_hidden, d_hidden)
        self.skip = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)
        self.drop = nn.Dropout(dropout)
        self.glu = nn.Linear(d_hidden, 2 * d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(F.elu(self.fc1(x)))
        return self.norm(self.skip(x) + F.glu(self.glu(self.drop(h)), dim=-1))


class VariableSelection(nn.Module):
    """Sélection de variables : poids softmax par variable, chaque variable ayant son propre GRN."""

    def __init__(self, n_vars: int, d_model: int, dropout: float):
        super().__init__()
        self.n_vars, self.d_model = n_vars, d_model
        self.weight = nn.Parameter(torch.randn(n_vars, d_model) / math.sqrt(d_model))
        self.bias = nn.Parameter(torch.zeros(n_vars, d_model))
        self.var_grns = nn.ModuleList(GRN(d_model, d_model, d_model, dropout) for _ in range(n_vars))
        self.select = GRN(n_vars * d_model, d_model, n_vars, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = x.unsqueeze(-1) * self.weight + self.bias                    # (B, T, n, d)
        w = torch.softmax(self.select(emb.flatten(-2)), dim=-1)            # (B, T, n)
        proc = torch.stack([g(emb[..., i, :]) for i, g in enumerate(self.var_grns)], dim=-2)
        return (w.unsqueeze(-1) * proc).sum(dim=-2), w


class InterpretableAttention(nn.Module):
    """Attention multi-têtes à valeur partagée : la moyenne des têtes reste interprétable."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model doit être un multiple de n_heads")
        self.h, self.dk = n_heads, d_model // n_heads
        self.q, self.k = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.v, self.out = nn.Linear(d_model, self.dk), nn.Linear(self.dk, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, allowed: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        b, tq, _ = q.shape
        tk = kv.shape[1]
        qh = self.q(q).view(b, tq, self.h, self.dk).transpose(1, 2)
        kh = self.k(kv).view(b, tk, self.h, self.dk).transpose(1, 2)
        scores = (qh @ kh.transpose(-1, -2)) / math.sqrt(self.dk)          # (B, h, Tq, Tk)
        attn = self.drop(torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1))
        attn = attn.mean(dim=1)                                            # têtes moyennées
        return self.out(attn @ self.v(kv)), attn


class TemporalFusionTransformer(nn.Module):
    def __init__(self, n_past: int, n_future: int, d_model: int = 64, n_heads: int = 4,
                 lstm_layers: int = 1, dropout: float = 0.1,
                 quantiles: tuple[float, ...] = DEFAULT_QUANTILES):
        super().__init__()
        self.quantiles = tuple(quantiles)
        self.past_vsn = VariableSelection(n_past, d_model, dropout)
        self.future_vsn = VariableSelection(n_future, d_model, dropout)
        lstm_kw = dict(batch_first=True, num_layers=lstm_layers,
                       dropout=dropout if lstm_layers > 1 else 0.0)
        self.encoder = nn.LSTM(d_model, d_model, **lstm_kw)
        self.decoder = nn.LSTM(d_model, d_model, **lstm_kw)
        self.post_lstm = GateAddNorm(d_model, dropout)
        self.enrich = GRN(d_model, d_model, d_model, dropout)
        self.attention = InterpretableAttention(d_model, n_heads, dropout)
        self.post_attn = GateAddNorm(d_model, dropout)
        self.ff = GRN(d_model, d_model, d_model, dropout)
        self.post_ff = GateAddNorm(d_model, dropout)
        self.head = nn.Linear(d_model, len(self.quantiles))

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        n_enc, n_dec = past.shape[1], future.shape[1]
        pe, _ = self.past_vsn(past)
        fe, _ = self.future_vsn(future)
        enc_out, state = self.encoder(pe)
        dec_out, _ = self.decoder(fe, state)                     # le décodeur part de l'état encodeur
        x = self.post_lstm(torch.cat([enc_out, dec_out], dim=1), torch.cat([pe, fe], dim=1))
        e = self.enrich(x)
        # chaque pas décodé voit tout l'encodeur et les pas décodés précédents (masque causal)
        allowed = torch.zeros(n_dec, n_enc + n_dec, dtype=torch.bool, device=past.device)
        allowed[:, :n_enc] = True
        allowed[:, n_enc:] = torch.tril(torch.ones(n_dec, n_dec, dtype=torch.bool, device=past.device))
        a, _ = self.attention(e[:, n_enc:], e, allowed)
        z = self.post_ff(self.ff(self.post_attn(a, e[:, n_enc:])), x[:, n_enc:])
        return torch.sort(self.head(z), dim=-1).values


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: tuple[float, ...]
                  ) -> torch.Tensor:
    """Perte pinball moyenne. pred : (B, H, Q), target : (B, H)."""
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    err = target.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1) * err).mean()
