"""Routeur de prévision : Random Forest / XGBoost (régime normal) OU moteur de règles métier.

    if is_year_end_period:      forecast_year_end()
    elif is_year_start_period:  forecast_year_start()
    elif is_summer_special:     forecast_summer()
    elif holiday or bridge:     forecast_isolated_holiday()
    else:                       modèle « normal » .predict()

Le routeur ne contient AUCUNE règle métier : il délègue aux modules `business_rules/`. Chaque
prévision produite par une règle est accompagnée d'une explication (`explain`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rte_forecast.business_rules import isolated_holidays as ih
from rte_forecast.business_rules.summer import forecast_summer
from rte_forecast.business_rules.weather import WeatherNormalizer
from rte_forecast.business_rules.year_end import forecast_year_end
from rte_forecast.business_rules.year_start import forecast_year_start
from rte_forecast.calendar import special_periods as sp

FALLBACK_METHOD = "FALLBACK_NORMAL_MODEL"
EXPLAIN_COLS = ["method", "reference_year", "reason", "baseline_mw", "weekly_target_level_mw",
                "day_coefficient", "day_mean_mw", "hourly_profile_source"]


class HybridForecaster:
    """Modèle « normal » + règles métier, routé jour par jour."""

    def __init__(self, load: pd.Series, cal_daily: pd.DataFrame, rules_cfg: dict,
                 normalizer: WeatherNormalizer | None = None, normal_label: str = "NORMAL_MODEL",
                 level_mode: str = "planning", recent_lag_days: int = 0):
        self.load, self.cal_daily, self.cfg, self.level_mode = load, cal_daily, rules_cfg, level_mode
        self.recent_lag_days = recent_lag_days
        self.normalizer, self.normal_label = normalizer, normal_label
        self.covid = tuple(rules_cfg.get("covid_years", (2020,)))
        self._cache: dict[tuple[str, int], pd.DataFrame | Exception] = {}
        self._stats = ih.holiday_statistics(load, cal_daily)

    # -- moteurs de période, mis en cache par (moteur, année) --------------------------------
    def _period(self, kind: str, year: int) -> pd.DataFrame | Exception:
        key = (kind, year)
        if key not in self._cache:
            try:
                if kind == sp.YEAR_END:
                    res = forecast_year_end(self.load, year, self.covid, self.normalizer,
                                            self.level_mode, self.recent_lag_days)
                elif kind == sp.YEAR_START:
                    res = forecast_year_start(self.load, year, self.covid, self.normalizer,
                                              self.level_mode, self.recent_lag_days)
                else:
                    res = forecast_summer(
                        self.load, year, self.cal_daily, self._stats,
                        self.cfg["summer_window_weeks_before"], self.cfg["summer_window_weeks_after"],
                        self.cfg["summer_anchor_weeks"], self.covid, self.normalizer,
                        self.level_mode, self.recent_lag_days)
                self._cache[key] = res
            except ValueError as exc:  # pas d'année de référence disponible
                self._cache[key] = exc
        return self._cache[key]

    def predict(self, normal_pred: pd.Series) -> pd.DataFrame:
        """Prévision finale (`yhat`) + explications, sur l'index de `normal_pred`."""
        out = pd.DataFrame({"yhat": normal_pred.to_numpy(dtype=float)}, index=normal_pred.index)
        out["method"] = pd.Series(self.normal_label, index=out.index, dtype=object)
        for c in EXPLAIN_COLS[1:]:
            text = c in {"reference_year", "reason", "hourly_profile_source"}
            out[c] = pd.Series("" if text else np.nan, index=out.index, dtype=object)
        for day, block in normal_pred.groupby(normal_pred.index.normalize()):
            row = self.cal_daily.loc[day]
            ptype = row["special_period_type"]
            res = None
            if ptype in (sp.YEAR_END, sp.YEAR_START):
                res = self._period(ptype, int(row["iso_year"]))
            elif ptype == sp.SUMMER:
                res = self._period(sp.SUMMER, int(day.year))
            elif ptype in (sp.HOLIDAY, sp.BRIDGE) and len(block) == 24:
                hol = ih.forecast_isolated_holiday(day, block.to_numpy(float), self._stats,
                                                   self.load.loc[: day - pd.Timedelta(hours=1)], row)
                if hol is not None:
                    for h, ts in enumerate(block.index):
                        out.loc[ts, "yhat"] = hol["yhat"][h]
                        for c in EXPLAIN_COLS:
                            if c in hol:
                                out.loc[ts, c] = str(hol[c]) if c == "reference_year" else hol[c]
                else:
                    out.loc[block.index, "method"] = FALLBACK_METHOD
                    out.loc[block.index, "reason"] = "aucune occurrence historique comparable"
                continue
            if isinstance(res, pd.DataFrame):
                idx = block.index.intersection(res.index)
                out.loc[idx, "yhat"] = res.loc[idx, "yhat"].to_numpy()
                for c in EXPLAIN_COLS:
                    if c in res.columns:
                        vals = res.loc[idx, c]
                        out.loc[idx, c] = vals.astype(str).to_numpy() if c == "reference_year" \
                            else vals.to_numpy()
            elif isinstance(res, Exception):
                out.loc[block.index, "method"] = FALLBACK_METHOD
                out.loc[block.index, "reason"] = str(res)
        return out


def format_explanation(ts: pd.Timestamp, rec: pd.Series) -> str:
    """Bloc texte d'explicabilité d'une prévision (spec §17)."""
    def fmt(v, unit=""):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.0f}{unit}"
    coef = rec.get("day_coefficient")
    return (f"Forecast date:\n{ts:%Y-%m-%d %H:%M}\n\nForecast method:\n{rec['method']}\n\n"
            f"Reference year:\n{rec.get('reference_year', '')}\n\nReason:\n{rec.get('reason', '')}\n\n"
            f"Weekly target level:\n{fmt(rec.get('weekly_target_level_mw'), ' MW')}\n\n"
            f"Holiday / day coefficient:\n{'n/a' if coef is None or coef != coef else f'{coef:.3f}'}"
            f"\n\nHourly profile source:\n{rec.get('hourly_profile_source', '')}\n\n"
            f"Final forecast:\n{fmt(rec['yhat'], ' MW')}")
