"""Baselines sans apprentissage."""

from __future__ import annotations

import pandas as pd


def persistence_j7(load: pd.Series) -> pd.Series:
    """Persistance J-7 : la charge de la même heure, une semaine plus tôt (retard 168 h >= 24 h)."""
    return load.shift(168).rename("persistence_j7")


def rte_j1_reference(df: pd.DataFrame) -> pd.Series:
    """Prévision J-1 publiée par RTE (référence externe, pas un de nos modèles).

    Attention : RTE la produit avec de vraies prévisions météo et un instant d'émission
    différent du nôtre (D 00:00) ; la comparaison est indicative.
    """
    return df["rte_forecast_j1_mw"].rename("rte_j1")
