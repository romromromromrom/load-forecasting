"""Métriques d'erreur et tableaux par type de période."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rte_forecast.calendar import special_periods as sp

PERIOD_LABELS = {
    sp.NORMAL: "Normal days", sp.HOLIDAY: "Holidays", sp.BRIDGE: "Bridge days",
    sp.SUMMER: "Summer (Aug window)", sp.YEAR_END: "S51–S53 (year end)",
    sp.YEAR_START: "S1–S2 (year start)",
}
PERIOD_ORDER = list(PERIOD_LABELS.values()) + ["ALL"]


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def bias(y_true, y_pred) -> float:
    """Erreur moyenne signée (prévision - réel) : > 0 = surestimation."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(np.asarray(y_pred) - y_true) / np.abs(y_true)) * 100)


def summarize(y_true, y_pred) -> dict:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "bias": bias(y_true, y_pred),
            "mape": mape(y_true, y_pred), "n": int(len(y_true))}


def metrics_by_period(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """`pred` : colonnes y_true, y_pred, period_type + group_cols. Une ligne par (groupe, période)."""
    rows = []
    for keys, g in pred.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_cols, keys, strict=True))
        for ptype, gg in g.groupby("period_type"):
            rows.append({**base, "period": PERIOD_LABELS[ptype], **summarize(gg["y_true"], gg["y_pred"])})
        rows.append({**base, "period": "ALL", **summarize(g["y_true"], g["y_pred"])})
    out = pd.DataFrame(rows)
    out["period"] = pd.Categorical(out["period"], PERIOD_ORDER, ordered=True)
    return out.sort_values(group_cols + ["period"]).reset_index(drop=True)
