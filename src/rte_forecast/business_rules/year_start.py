"""Moteur « début d'année » : ramp-up S1–S2 (redémarrage de janvier)."""

from __future__ import annotations

import pandas as pd

from rte_forecast.business_rules.analogue import find_best_year_start_calendar_analogue
from rte_forecast.business_rules.period_transfer import (
    YEAR_START_METHOD,
    YEAR_START_SPEC,
    forecast_transfer_period,
)

__all__ = ["YEAR_START_METHOD", "find_best_year_start_calendar_analogue", "forecast_year_start"]


def forecast_year_start(load: pd.Series, target_year: int, covid_years=(2020,),
                        normalizer=None,
                        level_mode: str = "planning", recent_lag_days: int = 0) -> pd.DataFrame:
    """Prévision horaire de S1–S2 de `target_year`, ancrée sur la baseline S48–S50 de l'an passé."""
    return forecast_transfer_period(load, target_year, YEAR_START_SPEC, covid_years, normalizer,
                                    level_mode, recent_lag_days)
