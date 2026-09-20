"""Tests métier : routage, année analogue, coefficients, profils, causalité des règles."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from rte_forecast.business_rules import common
from rte_forecast.business_rules import isolated_holidays as ih
from rte_forecast.business_rules.analogue import (
    calendar_similarity,
    find_best_year_end_calendar_analogue,
    find_best_year_start_calendar_analogue,
)
from rte_forecast.business_rules.summer import forecast_summer
from rte_forecast.business_rules.weather import WeatherNormalizer
from rte_forecast.business_rules.year_end import forecast_year_end
from rte_forecast.business_rules.year_start import forecast_year_start
from rte_forecast.evaluation.backtest import _train_mask
from rte_forecast.features import build_features
from rte_forecast.forecast import HybridForecaster, format_explanation

# ------------------------------------------------------------------ exclusion de l'entraînement


def test_holidays_and_bridges_never_enter_the_normal_training_set(synth, cal_h):
    Xc = build_features(synth, cal_h, clean_lags=True)
    m = _train_mask(Xc, pd.Timestamp("2024-06-01"), True, synth.index.min())
    train_days = cal_h.loc[Xc.index[m]]
    assert not train_days["is_public_holiday"].any()
    assert not train_days["is_bridge_day"].any()
    assert (train_days["special_period_type"] == "NORMAL").all()


def test_a_holiday_tuesday_is_not_a_comparable_tuesday(synth, cal_h):
    Xc = build_features(synth, cal_h, clean_lags=True)
    m = _train_mask(Xc, pd.Timestamp("2023-06-01"), True, synth.index.min())
    kept = Xc.index[m]
    assert pd.Timestamp("2022-11-01 12:00") not in kept        # Toussaint, un mardi férié
    assert pd.Timestamp("2022-10-31 12:00") not in kept        # le lundi de pont
    assert pd.Timestamp("2022-11-08 12:00") in kept            # un mardi ordinaire


def test_all_days_model_keeps_holidays_in_training(synth, cal_h):
    Xp = build_features(synth, cal_h)
    m = _train_mask(Xp, pd.Timestamp("2023-06-01"), False, synth.index.min())
    assert pd.Timestamp("2022-11-01 12:00") in Xp.index[m]


# ------------------------------------------------------------------ année analogue


def test_year_end_analogue_prefers_matching_christmas_weekday():
    # Noël 2026 = vendredi ; 2015 = vendredi, 2016 = dimanche, 2017 = lundi, 2018 = mardi
    best, ranking = find_best_year_end_calendar_analogue(2026, [2015, 2016, 2017, 2018])
    assert best == 2015
    assert ranking.iloc[0]["score"] == pytest.approx(1.0)
    assert ranking.iloc[0]["d_weekday"] == 0


def test_analogue_choice_is_reproducible():
    a = find_best_year_end_calendar_analogue(2024, [2020, 2021, 2022, 2023])
    b = find_best_year_end_calendar_analogue(2024, [2023, 2022, 2021, 2020])    # ordre différent
    assert a[0] == b[0]
    pd.testing.assert_frame_equal(a[1], b[1])


def test_analogue_tie_goes_to_most_recent_year_and_covid_year_is_deprioritised():
    # 2015 et 2020 ont Noël un vendredi (score identique 1.0 pour la cible 2026)
    assert find_best_year_end_calendar_analogue(2026, [2015, 2020])[0] == 2020
    assert find_best_year_end_calendar_analogue(2026, [2015, 2020], preferred_excluded=[2020])[0] == 2015
    # si tout est exclu, on réintègre plutôt que d'échouer
    assert find_best_year_end_calendar_analogue(2026, [2020], preferred_excluded=[2020])[0] == 2020


def test_similarity_components_are_documented_and_bounded():
    s = calendar_similarity(2026, 2015)
    assert set(s) >= {"score", "d_weekday", "d_isoweek", "d_structure"}
    assert 0.0 <= calendar_similarity(2026, 2018)["score"] < s["score"] <= 1.0


def test_year_start_analogue_uses_new_year_anchor():
    best, ranking = find_best_year_start_calendar_analogue(2027, [2021, 2022, 2023])
    # Nouvel An 2027 = vendredi ; 2021 = vendredi
    assert best == 2021 and ranking.iloc[0]["d_weekday"] == 0


# ------------------------------------------------------------------ coefficients et profils


def test_daily_coefficients_are_relative_to_their_own_weekly_level(synth):
    load = synth["load_mw"]
    monday = dt.date(2023, 11, 20)
    coefs = [common.day_coefficient(load, monday + dt.timedelta(days=i)) for i in range(7)]
    assert np.mean(coefs) == pytest.approx(1.0)              # 7 jours moyennent à leur semaine
    wk = common.week_mean(load, monday)
    d3 = common.day_mean(load, monday + dt.timedelta(days=3))
    assert coefs[3] == pytest.approx(d3 / wk)


def test_normalized_profiles_recompose_the_original_day(synth):
    load = synth["load_mw"]
    day = dt.date(2023, 12, 25)
    p = common.normalized_hourly_profile(load, day)
    assert p.mean() == pytest.approx(1.0)
    recomposed = common.day_mean(load, day) * p
    assert np.allclose(recomposed, common.day_values(load, day))


def test_forecast_recomposes_baseline_x_week_ratio_x_day_coef_x_profile(synth):
    f = forecast_year_end(synth["load_mw"], 2023)
    day = f.loc["2023-12-20"]
    level = day["baseline_mw"] * day["weekly_ratio"] * day["day_coefficient"]
    assert day["yhat"].mean() == pytest.approx(level.iloc[0])
    assert day["day_mean_mw"].iloc[0] == pytest.approx(level.iloc[0])


def test_map_reference_monday_keeps_weekday_and_aligns_christmas_week():
    # même jour de semaine de Noël -> l'écart est nul
    t = dt.date(2026, 12, 21)                # lundi de la semaine de Noël 2026 (vendredi)
    ref = common.map_reference_monday(t, dt.date(2026, 12, 25), dt.date(2015, 12, 25))
    assert ref == dt.date(2015, 12, 21) and ref.weekday() == 0
    # Noël un jour différent : toujours un lundi, écart de position <= 3 jours
    ref2 = common.map_reference_monday(t, dt.date(2026, 12, 25), dt.date(2023, 12, 25))
    assert ref2.weekday() == 0
    assert abs(((ref2 - dt.date(2023, 12, 25)).days) - ((t - dt.date(2026, 12, 25)).days)) <= 3


# ------------------------------------------------------------------ nettoyage / rebouchage


def test_baseline_is_rebuilt_from_gaps_without_future_information(synth):
    load = synth["load_mw"]
    clean = common.baseline_level(load, 2023)
    holed = load.copy()
    holed.loc["2023-12-05 08:00":"2023-12-06 20:00"] = np.nan       # trou en S49
    rebuilt = common.baseline_level(holed, 2023)
    assert np.isfinite(rebuilt) and abs(rebuilt / clean - 1) < 0.03

    # les données APRÈS la fenêtre S48–S50 ne peuvent pas influencer le niveau
    future = load.copy()
    future.loc["2023-12-18":] = future.loc["2023-12-18":] * 3
    assert common.baseline_level(future, 2023) == pytest.approx(clean)


# ------------------------------------------------------------------ routage


@pytest.fixture(scope="module")
def routed(synth, cal_h, cal_d, cfg):
    tf = synth["temperature_normal"]
    wn = WeatherNormalizer(synth["temperature_realised"], synth["temperature_normal"],
                           cal_h["is_normal_day"], tf)
    normal = synth["load_mw"].shift(168).loc["2023-01-01":"2024-12-31"]
    out = HybridForecaster(synth["load_mw"], cal_d, cfg["rules"], wn, "NORMAL").predict(normal)
    return out, cal_h.reindex(out.index)


def test_s51_s52_trigger_year_end_engine(routed):
    out, cal = routed
    idx = cal.index[(cal["iso_week"].isin([51, 52])) & (cal["iso_year"] == 2023)]
    assert len(idx) == 14 * 24
    assert (out.loc[idx, "method"] == "YEAR_END_BUSINESS_RULE").all()


def test_s1_s2_trigger_year_start_engine(routed):
    out, cal = routed
    idx = cal.index[(cal["iso_week"].isin([1, 2])) & (cal["iso_year"] == 2024)]
    assert len(idx) == 14 * 24
    assert (out.loc[idx, "method"] == "YEAR_START_BUSINESS_RULE").all()


def test_summer_window_triggers_summer_engine_and_15_august_adds_holiday_coefficient(routed):
    out, cal = routed
    win = cal.index[cal["is_august_special_period"] & (cal.index.year == 2023)]
    assert out.loc[win, "method"].str.startswith("SUMMER").all()
    assert out.loc["2023-08-15 12:00", "method"] == "SUMMER_BUSINESS_RULE+ISOLATED_HOLIDAY_BUSINESS_RULE"


def test_isolated_holiday_and_bridge_use_their_module_and_normal_days_are_untouched(routed, synth):
    out, cal = routed
    assert (out.loc["2024-05-09", "method"] == "ISOLATED_HOLIDAY_BUSINESS_RULE").all()   # Ascension
    assert (out.loc["2024-05-10", "method"] == "BRIDGE_DAY_BUSINESS_RULE").all()         # pont
    normal_rows = cal.index[cal["special_period_type"] == "NORMAL"]
    normal_pred = synth["load_mw"].shift(168).reindex(normal_rows)
    assert np.allclose(out.loc[normal_rows, "yhat"], normal_pred)
    assert (out.loc[normal_rows, "method"] == "NORMAL").all()


def test_rule_forecasts_carry_a_full_explanation(routed):
    out, _ = routed
    ts = pd.Timestamp("2023-12-25 12:00")
    text = format_explanation(ts, out.loc[ts])
    for needle in ["Forecast date", "Forecast method", "YEAR_END_BUSINESS_RULE", "Reference year",
                   "Reason", "Weekly target level", "coefficient", "Hourly profile source",
                   "Final forecast"]:
        assert needle in text
    assert out.loc[ts, "weekly_target_level_mw"] > 0


# ------------------------------------------------------------------ causalité des règles


def _perturb(load: pd.Series, start: pd.Timestamp) -> pd.Series:
    p = load.copy()
    p.loc[start:] = p.loc[start:] * 1.7 + 4000
    return p


@pytest.mark.parametrize("fn,year", [(forecast_year_end, 2023), (forecast_year_start, 2024)])
def test_planning_rules_ignore_everything_from_the_period_start(synth, fn, year):
    load = synth["load_mw"]
    base = fn(load, year, level_mode="planning")
    other = fn(_perturb(load, base.attrs["origin"]), year, level_mode="planning")
    assert np.allclose(base["yhat"], other["yhat"])


@pytest.mark.parametrize("fn,year,cut", [(forecast_year_end, 2023, "2023-12-25"),
                                         (forecast_year_start, 2024, "2024-01-08")])
def test_adaptive_rules_only_use_data_before_the_forecast_day(synth, fn, year, cut):
    load = synth["load_mw"]
    base = fn(load, year, level_mode="adaptive")
    other = fn(_perturb(load, pd.Timestamp(cut)), year, level_mode="adaptive")
    before = base.index < pd.Timestamp(cut) + pd.Timedelta(hours=24)   # jour de coupure inclus
    assert np.allclose(base.loc[before, "yhat"], other.loc[before, "yhat"])
    assert not np.allclose(base.loc[~before, "yhat"], other.loc[~before, "yhat"])  # test non vacueux


def test_summer_rules_ignore_data_from_window_start(synth, cal_d, cfg):
    load = synth["load_mw"]
    stats = ih.holiday_statistics(load, cal_d)
    r = cfg["rules"]
    args = (cal_d, stats, r["summer_window_weeks_before"], r["summer_window_weeks_after"],
            r["summer_anchor_weeks"])
    base = forecast_summer(load, 2023, *args)
    other = forecast_summer(_perturb(load, base.attrs["origin"]), 2023, *args)
    assert np.allclose(base["yhat"], other["yhat"])
    assert base["summer_trough_level_mw"].iloc[0] < base["summer_start_level_mw"].iloc[0]


def test_holiday_statistics_are_only_used_when_all_their_data_precede_the_target(synth, cal_d):
    stats = ih.holiday_statistics(synth["load_mw"], cal_d)
    target = pd.Timestamp("2024-05-09")                                  # Ascension
    occ, tier = ih.select_occurrences(stats, target, "holiday", "Ascension", 3)
    assert len(occ) >= 2 and tier.startswith("T1")
    assert (occ["last_data_date"] < target).all()
    assert (occ["date"] < target).all()
    # le férié voisin du 8 mai 2024 dépend du 15 mai : il ne doit pas fuiter dans la prévision du 9
    leaky = stats[stats["date"] == pd.Timestamp("2024-05-08")]
    if len(leaky):
        assert leaky["last_data_date"].iloc[0] >= target
        assert pd.Timestamp("2024-05-08") not in set(
            ih.select_occurrences(stats, target, "holiday", "Fête de la Victoire", 2)[0]["date"])


@pytest.mark.parametrize("fn,year,cut", [(forecast_year_end, 2023, "2023-12-25"),
                                         (forecast_year_start, 2024, "2024-01-08")])
def test_hardened_adaptive_rules_ignore_the_day_before_the_forecast_day(synth, fn, year, cut):
    """Émission à J-1 10:00 : J-1 est incomplet, le niveau récent s'arrête à J-2 (recent_lag_days=1)."""
    load = synth["load_mw"]
    base = fn(load, year, level_mode="adaptive", recent_lag_days=1)
    other = fn(_perturb(load, pd.Timestamp(cut)), year, level_mode="adaptive", recent_lag_days=1)
    before = base.index < pd.Timestamp(cut) + pd.Timedelta(hours=48)     # coupure + 1 jour d'écart
    assert np.allclose(base.loc[before, "yhat"], other.loc[before, "yhat"])
    assert not np.allclose(base.loc[~before, "yhat"], other.loc[~before, "yhat"])


def test_hardened_summer_adaptive_rule_ignores_the_day_before(synth, cal_d, cfg):
    load = synth["load_mw"]
    stats = ih.holiday_statistics(load, cal_d)
    r = cfg["rules"]
    args = (cal_d, stats, r["summer_window_weeks_before"], r["summer_window_weeks_after"],
            r["summer_anchor_weeks"])
    kw = dict(level_mode="adaptive", recent_lag_days=1)
    origin = forecast_summer(load, 2023, *args).attrs["origin"]
    cut = origin + pd.Timedelta(days=5)
    base = forecast_summer(load, 2023, *args, **kw)
    other = forecast_summer(_perturb(load, cut), 2023, *args, **kw)
    before = base.index < cut + pd.Timedelta(hours=48)
    assert np.allclose(base.loc[before, "yhat"], other.loc[before, "yhat"])
    assert not np.allclose(base.loc[~before, "yhat"], other.loc[~before, "yhat"])
