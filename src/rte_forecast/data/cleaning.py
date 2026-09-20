"""Agrégation horaire, validation de la série de charge et détection des valeurs invalides."""

from __future__ import annotations

import pandas as pd

WEEK = 168


def aggregate_hourly(eco2mix: pd.DataFrame) -> pd.DataFrame:
    """15/30 min -> horaire (moyenne des pas disponibles), étiqueté au début de l'heure.

    L'heure des fichiers eCO2mix est traitée comme un mur d'horloge local ; chaque jour compte
    exactement 96 pas, y compris les jours de changement d'heure (hypothèse documentée).
    """
    df = eco2mix.copy()
    df["hour_ts"] = df["timestamp"].dt.floor("h")
    cols = {"consommation_mw": "load_mw", "prevision_j1_mw": "rte_forecast_j1_mw",
            "prevision_j_mw": "rte_forecast_j_mw"}
    hourly = df.groupby("hour_ts")[list(cols)].mean().rename(columns=cols)
    hourly.index.name = "timestamp"
    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", name="timestamp")
    return hourly.reindex(full)


def validate_hourly(df: pd.DataFrame, target: str = "load_mw") -> None:
    """Lève ValueError si l'index est dupliqué, non monotone, irrégulier ou si la cible est NaN."""
    if df.index.has_duplicates:
        raise ValueError(f"{df.index.duplicated().sum()} timestamps dupliqués")
    if not df.index.is_monotonic_increasing:
        raise ValueError("index non trié")
    if len(df) > 1:
        steps = pd.Series(df.index).diff().dropna().unique()
        if len(steps) != 1 or steps[0] != pd.Timedelta("1h"):
            raise ValueError(f"pas temporel irrégulier : {steps}")
    if df[target].isna().any():
        raise ValueError(f"{int(df[target].isna().sum())} NaN dans la cible '{target}'")


def detect_invalid(load: pd.Series, low: float = 0.5, high: float = 1.6, n_weeks: int = 4) -> pd.Series:
    """Marque les observations manquantes ou manifestement invalides (causal, sans futur).

    Valide = >0 et ratio à la médiane des mêmes heures des `n_weeks` semaines PRÉCÉDENTES
    compris dans [low, high]. Les 4 premières semaines n'ont pas de référence : seul le test
    de positivité leur est appliqué.
    """
    bad = load.isna() | (load <= 0)
    refs = pd.concat([load.shift(WEEK * k) for k in range(1, n_weeks + 1)], axis=1)
    ratio = load / refs.median(axis=1)
    return (bad | (ratio < low) | (ratio > high)).fillna(bad)


def missing_report(load: pd.Series, invalid: pd.Series) -> pd.DataFrame:
    """Une ligne par plage contiguë d'observations invalides (vide si aucune)."""
    if not invalid.any():
        return pd.DataFrame(columns=["start", "end", "hours"])
    grp = (invalid != invalid.shift()).cumsum()
    rows = []
    for _, seg in invalid.groupby(grp):
        if seg.iloc[0]:
            rows.append({"start": seg.index[0], "end": seg.index[-1], "hours": len(seg)})
    return pd.DataFrame(rows)
