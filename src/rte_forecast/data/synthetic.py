"""Série horaire SYNTHÉTIQUE déterministe (tests unitaires / CI, jamais pour les résultats).

Reproduit les grands effets de la charge française : thermosensibilité hivernale, cycle
hebdomadaire et journalier, baisse des fériés/ponts, creux de fin d'année et du 15 août. Sert à
tester les règles métier sans dépendre des fichiers de données (absents en intégration continue).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rte_forecast.calendar import special_periods as sp

WEEKDAY_FACTOR = np.array([1.02, 1.03, 1.03, 1.03, 1.00, 0.92, 0.87])


def make_synthetic_hourly(start: str = "2019-10-01", end: str = "2024-12-31", seed: int = 0
                          ) -> pd.DataFrame:
    """DataFrame horaire : load_mw, temperature_normal, temperature_realised, rte_forecast_j1_mw."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h", name="timestamp")
    doy, hour = idx.dayofyear.to_numpy(), idx.hour.to_numpy()
    t_norm = 12.5 - 8.0 * np.cos(2 * np.pi * (doy - 15) / 365.25) - 2.5 * np.cos(
        2 * np.pi * (hour - 15) / 24)
    days = idx.normalize().unique()
    a = np.zeros(len(days))
    for i in range(1, len(days)):
        a[i] = 0.8 * a[i - 1] + rng.normal(0, 1.6)
    anomaly = pd.Series(a, index=days).reindex(idx.normalize()).to_numpy()
    t_real = t_norm + anomaly

    cal = sp.build_hourly_calendar(idx)
    heating = np.maximum(0.0, 15.0 - t_real)
    cooling = np.maximum(0.0, t_real - 22.0)
    shape = 1.0 + 0.10 * np.sin(2 * np.pi * (hour - 13) / 24) + 0.05 * np.sin(4 * np.pi * (hour - 8) / 24)
    load = (46000 + 2100 * heating + 450 * cooling) * shape * WEEKDAY_FACTOR[cal["weekday"].to_numpy()]

    factor = np.ones(len(idx))
    ptype = cal["special_period_type"].to_numpy()
    factor[cal["is_bridge_day"].to_numpy()] *= 0.93
    factor[cal["is_public_holiday"].to_numpy()] *= 0.86
    factor[ptype == sp.YEAR_END] *= 0.90
    factor[ptype == sp.YEAR_START] *= 0.94
    xmas = (idx.month == 12) & (idx.day.isin([24, 25])) | ((idx.month == 1) & (idx.day == 1))
    factor[np.asarray(xmas)] *= 0.88
    factor[cal["is_august_special_period"].to_numpy()] *= 0.95
    load = load * factor * (1 + rng.normal(0, 0.006, len(idx)))
    return pd.DataFrame({
        "load_mw": load, "temperature_normal": t_norm, "temperature_realised": t_real,
        "rte_forecast_j1_mw": load * (1 + rng.normal(0, 0.012, len(idx)))}, index=idx)
