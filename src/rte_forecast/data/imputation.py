"""Rebouchage CAUSAL des trous de charge : aucune information future n'est jamais utilisée.

Attendu saisonnier au temps t = médiane des mêmes heures/jour de semaine des semaines
précédentes (t-168h, t-336h, …, jusqu'à `n_weeks`) déjà valides ou déjà imputées. Repli : t-24h,
puis dernière valeur connue.

Ancrage de niveau : on multiplie l'attendu par le rapport (dernière observation valide avant le
trou / son propre attendu saisonnier), amorti avec la durée écoulée depuis cette observation
(demi-vie `decay_hours`). Un trou court reste donc proche du niveau courant, un long trou revient
progressivement à l'attendu saisonnier. Traitement séquentiel : jamais de regard après t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEK = 168


def _seasonal_expectation(values: np.ndarray, i: int, n_weeks: int) -> float:
    cands = [values[i - WEEK * k] for k in range(1, n_weeks + 1) if i - WEEK * k >= 0]
    cands = [c for c in cands if not np.isnan(c)]
    if cands:
        return float(np.median(cands))
    if i >= 24 and not np.isnan(values[i - 24]):
        return float(values[i - 24])
    if i >= 1 and not np.isnan(values[i - 1]):
        return float(values[i - 1])
    return float("nan")


def impute_causal(load: pd.Series, invalid: pd.Series | None = None, n_weeks: int = 4,
                  decay_hours: float = 48.0) -> pd.Series:
    """Retourne une copie de `load` où les points `invalid` sont reconstruits causalement."""
    if invalid is None:
        invalid = load.isna()
    values = load.to_numpy(dtype=float).copy()
    bad = invalid.to_numpy(dtype=bool)
    values[bad] = np.nan
    last_valid = -1                       # dernier indice ORIGINAL valide rencontré
    for i in range(len(values)):
        if not bad[i]:
            last_valid = i
            continue
        expected = _seasonal_expectation(values, i, n_weeks)
        if np.isnan(expected):
            continue
        q = 1.0
        if last_valid >= 0:
            ref = _seasonal_expectation(values, last_valid, n_weeks)
            if not np.isnan(ref) and ref > 0:
                gap = np.exp(-(i - last_valid) / decay_hours)
                q = 1.0 + (values[last_valid] / ref - 1.0) * gap
        values[i] = expected * q
    return pd.Series(values, index=load.index, name=load.name)
