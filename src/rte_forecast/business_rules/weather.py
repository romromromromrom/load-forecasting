"""Normalisation météo des références (ramener la charge historique à la température normale).

Pourquoi : une année analogue transférée « brute » importe son anomalie de température (ex.
décembre 2022 très doux : S51 à 0.81 de la baseline contre 0.96 en 2023). Comme la prévision de
température de la cible est la NORMALE, on ramène aussi les références à la normale :

    load_norm(t) = load(t) - g_h * (HDD_réalisée(t) - HDD_normale(t))
                           - g_c * (CDD_réalisée(t) - CDD_normale(t))

avec HDD = max(0, 15 - T), CDD = max(0, T - 22) et (g_h, g_c) en MW/°C estimés par MCO
journaliers sur les jours NORMAUX de l'historique connu à l'origine (effets mois et jour de
semaine absorbés par des indicatrices) : aucune information future n'est utilisée.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HDD_BASE, CDD_BASE = 15.0, 22.0
DEFAULT_G = {"g_heat": 2200.0, "g_cool": 500.0}   # a priori si l'historique est trop court
MIN_DAYS = 250


def hdd(t: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.maximum(0.0, HDD_BASE - t)


def cdd(t: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.maximum(0.0, t - CDD_BASE)


def estimate_thermosensitivity(load: pd.Series, temp_real: pd.Series, normal_day_mask: pd.Series
                               ) -> dict:
    """MCO journalier : load ~ mois + jour de semaine + g_heat*HDD + g_cool*CDD (jours normaux)."""
    d = pd.DataFrame({"load": load.resample("D").mean(),
                      "hdd": pd.Series(hdd(temp_real)).resample("D").mean(),
                      "cdd": pd.Series(cdd(temp_real)).resample("D").mean(),
                      "normal": normal_day_mask.resample("D").min()}).dropna()
    d = d[d["normal"].astype(bool)]
    if len(d) < MIN_DAYS:
        return {**DEFAULT_G, "n_days": len(d), "source": "prior"}
    X = pd.concat([pd.get_dummies(d.index.month, prefix="m", dtype=float).set_index(d.index),
                   pd.get_dummies(d.index.weekday, prefix="w", dtype=float).set_index(d.index)
                   .iloc[:, 1:], d[["hdd", "cdd"]]], axis=1)
    coef, *_ = np.linalg.lstsq(X.to_numpy(), d["load"].to_numpy(), rcond=None)
    names = list(X.columns)
    g_h, g_c = float(coef[names.index("hdd")]), float(coef[names.index("cdd")])
    # garde-fous physiques : un coefficient négatif n'aurait pas de sens
    return {"g_heat": max(g_h, 0.0), "g_cool": max(g_c, 0.0), "n_days": len(d),
            "source": "estimated"}


class WeatherNormalizer:
    """Ramène l'historique à la météo normale ; fournit aussi la correction du mode 'perfect'."""

    def __init__(self, temp_real: pd.Series, temp_normal: pd.Series, normal_day_mask: pd.Series,
                 temp_forecast: pd.Series | None = None):
        """`temp_forecast` = température PRÉVUE de la cible (défaut : la normale -> correction nulle)."""
        self.temp_real, self.temp_normal, self.normal_day_mask = temp_real, temp_normal, normal_day_mask
        self.temp_forecast = temp_normal if temp_forecast is None else temp_forecast

    def _delta(self, index: pd.DatetimeIndex, g: dict, source: str = "real") -> pd.Series:
        tr = (self.temp_real if source == "real" else self.temp_forecast).reindex(index)
        tn = self.temp_normal.reindex(index)
        return g["g_heat"] * (hdd(tr) - hdd(tn)) + g["g_cool"] * (cdd(tr) - cdd(tn))

    def adjust(self, history: pd.Series) -> tuple[pd.Series, dict]:
        """(historique normalisé, coefficients). `history` ne doit contenir que du connu."""
        end = history.index.max()
        g = estimate_thermosensitivity(history, self.temp_real.loc[:end].reindex(history.index),
                                       self.normal_day_mask.loc[:end].reindex(history.index))
        return history - self._delta(history.index, g).fillna(0.0), g

    def adjust_with(self, series: pd.Series, g: dict) -> pd.Series:
        """Applique des coefficients DÉJÀ estimés (à l'origine) à une série plus longue."""
        return series - self._delta(series.index, g).fillna(0.0)

    def forecast_correction(self, index: pd.DatetimeIndex, g: dict) -> pd.Series:
        """Correction (MW) à AJOUTER à une prévision « météo normale » : anomalie PRÉVUE de la cible.

        Nulle quand la température prévue est la normale ; non nulle en mode bruité/parfait."""
        return self._delta(index, g, source="forecast").fillna(0.0)
