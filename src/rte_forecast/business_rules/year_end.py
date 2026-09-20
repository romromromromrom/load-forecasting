"""Moteur « fin d'année » : ramp-down S51–S53, Noël, réveillon."""

from __future__ import annotations

import pandas as pd

from rte_forecast.business_rules.analogue import find_best_year_end_calendar_analogue
from rte_forecast.business_rules.period_transfer import (
    YEAR_END_METHOD,
    YEAR_END_SPEC,
    forecast_transfer_period,
)

__all__ = ["YEAR_END_METHOD", "find_best_year_end_calendar_analogue", "forecast_year_end"]


def forecast_year_end(load: pd.Series, target_year: int, covid_years=(2020,),
                      normalizer=None,
                      level_mode: str = "planning", recent_lag_days: int = 0) -> pd.DataFrame:
    """Prévision horaire de S51–S53 de `target_year`, avec explicabilité colonne par colonne."""
    return forecast_transfer_period(load, target_year, YEAR_END_SPEC, covid_years, normalizer,
                                    level_mode, recent_lag_days)
