"""Tests anti-fuite : aucune feature du jour D ne doit dépendre d'une charge >= D 00:00."""

import numpy as np
import pandas as pd
import pytest

from rte_forecast.evaluation import splits
from rte_forecast.features import build_features, feature_columns

DAYS = ["2022-03-14", "2023-08-16", "2023-12-27", "2024-01-02", "2024-05-10"]


def _equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return bool(((a == b) | (a.isna() & b.isna())).all().all())


@pytest.mark.parametrize("clean", [False, True])
@pytest.mark.parametrize("day", DAYS)
def test_features_of_day_D_do_not_depend_on_load_from_D_onwards(synth, cal_h, clean, day):
    d0 = pd.Timestamp(day)
    perturbed = synth.copy()
    tail = perturbed.loc[d0:, "load_mw"]
    perturbed.loc[d0:, "load_mw"] = tail + np.random.default_rng(0).normal(0, 8000, len(tail))
    cols = feature_columns("normal")
    a = build_features(synth, cal_h, clean_lags=clean).loc[d0: d0 + pd.Timedelta(hours=23), cols]
    b = build_features(perturbed, cal_h, clean_lags=clean).loc[d0: d0 + pd.Timedelta(hours=23), cols]
    assert _equal(a, b), "fuite : une feature du jour D utilise la charge de D ou d'après"


def test_lags_are_consistent_with_the_load_series(synth, cal_h):
    X = build_features(synth, cal_h)
    load = synth["load_mw"]
    for lag, col in [(24, "load_lag_24h"), (48, "load_lag_48h"), (168, "load_lag_168h"),
                     (336, "load_lag_336h")]:
        t = pd.Timestamp("2023-06-14 15:00")
        assert X.loc[t, col] == load.loc[t - pd.Timedelta(hours=lag)]


def test_rolling_means_are_windows_ending_24h_before(synth, cal_h):
    X = build_features(synth, cal_h)
    load = synth["load_mw"]
    t = pd.Timestamp("2023-06-14 15:00")
    w24 = load.loc[t - pd.Timedelta(hours=47): t - pd.Timedelta(hours=24)]
    w7d = load.loc[t - pd.Timedelta(hours=191): t - pd.Timedelta(hours=24)]
    assert len(w24) == 24 and len(w7d) == 168
    assert X.loc[t, "rolling_mean_24h"] == pytest.approx(w24.mean())
    assert X.loc[t, "rolling_mean_7d"] == pytest.approx(w7d.mean())


def test_lag_below_24h_is_rejected(synth, cal_h):
    with pytest.raises(ValueError, match="fuite"):
        build_features(synth, cal_h, min_lag_hours=12)


def test_clean_lags_never_read_an_atypical_source_day(synth, cal_h):
    X = build_features(synth, cal_h, clean_lags=True)
    load = synth["load_mw"]
    normal = cal_h["is_normal_day"]
    for t in pd.date_range("2023-01-03", "2023-12-20", freq="37h"):
        v = X.loc[t, "load_lag_168h"]
        if pd.notna(v):
            src = [t - pd.Timedelta(weeks=k) for k in range(1, 7)
                   if load.loc[t - pd.Timedelta(weeks=k)] == v]
            assert src and normal.loc[src[0]], f"lag_168h de {t} vient d'un jour atypique"


def test_realised_temperature_is_never_a_feature_in_operational_mode():
    assert "temperature_realised" not in feature_columns("normal")
    assert "temperature_realised" not in feature_columns("noisy")
    assert "temperature_realised" in feature_columns("perfect")     # oracle explicite


def test_operational_temperature_forecast_equals_the_normal(synth, cal_h):
    X = build_features(synth, cal_h, weather_mode="normal")
    assert np.allclose(X["temperature_forecast"], synth["temperature_normal"])


def test_train_posterior_to_test_is_rejected():
    idx = pd.date_range("2023-01-01", periods=48, freq="h")
    with pytest.raises(AssertionError, match="fuite"):
        splits.assert_train_before_test(idx[:30], idx[20:])
    splits.assert_train_before_test(idx[:24], idx[24:])     # cas valide


def test_chronological_split_is_ordered_and_disjoint():
    idx = pd.date_range("2020-01-01", "2024-12-31 23:00", freq="h")
    p = splits.chronological_split(idx, "2022-12-31", "2023-12-31", "2024-12-31")
    assert p["train"].max() < p["validation"].min() <= p["validation"].max() < p["test"].min()
    assert len(p["train"]) + len(p["validation"]) + len(p["test"]) == len(idx)


def test_walk_forward_folds_train_strictly_before_each_test_window():
    folds = splits.expanding_folds("2022-01-01", "2024-12-31", months=3)
    assert len(folds) == 12
    for prev, nxt in zip(folds[:-1], folds[1:], strict=True):
        assert prev.test_end < nxt.test_start
        assert nxt.test_start - prev.test_end == pd.Timedelta(hours=1)   # pas de trou ni de recouvrement


# --- émission durcie : prévision de D émise à D-1 10:00 (dernière charge connue = D-1 09:00) -----
ISSUE_HOUR = 9


@pytest.mark.parametrize("clean", [False, True])
@pytest.mark.parametrize("day", DAYS)
def test_hardened_features_do_not_depend_on_load_after_last_known_hour(synth, cal_h, clean, day):
    d0 = pd.Timestamp(day)
    first_unknown = d0 - pd.Timedelta(hours=24 - ISSUE_HOUR - 1)          # D-1 10:00
    perturbed = synth.copy()
    tail = perturbed.loc[first_unknown:, "load_mw"]
    perturbed.loc[first_unknown:, "load_mw"] = tail + np.random.default_rng(0).normal(0, 8000, len(tail))
    cols = feature_columns("normal")
    kw = dict(clean_lags=clean, issue_hour=ISSUE_HOUR)
    a = build_features(synth, cal_h, **kw).loc[d0: d0 + pd.Timedelta(hours=23), cols]
    b = build_features(perturbed, cal_h, **kw).loc[d0: d0 + pd.Timedelta(hours=23), cols]
    assert _equal(a, b), "fuite : une feature de D lit la charge après D-1 09:00"


def test_default_emission_at_midnight_does_use_the_evening_before(synth, cal_h):
    """Contrôle non vacueux : avec l'émission à D 00:00, la même perturbation se voit."""
    d0 = pd.Timestamp("2023-08-16")
    perturbed = synth.copy()
    perturbed.loc[d0 - pd.Timedelta(hours=13):, "load_mw"] += 5000                # dès D-1 11:00
    cols = feature_columns("normal")
    a = build_features(synth, cal_h).loc[d0: d0 + pd.Timedelta(hours=23), cols]
    b = build_features(perturbed, cal_h).loc[d0: d0 + pd.Timedelta(hours=23), cols]
    assert not _equal(a, b)


def test_hardened_lags_switch_from_24h_to_48h_after_the_last_known_hour(synth, cal_h):
    X = build_features(synth, cal_h, issue_hour=ISSUE_HOUR)
    load = synth["load_mw"]
    early, late = pd.Timestamp("2023-06-14 08:00"), pd.Timestamp("2023-06-14 15:00")
    assert X.loc[early, "load_lag_24h"] == load.loc[early - pd.Timedelta(hours=24)]
    assert X.loc[late, "load_lag_24h"] == load.loc[late - pd.Timedelta(hours=48)]
    assert X.loc[late, "load_lag_48h"] == load.loc[late - pd.Timedelta(hours=72)]
    anchor = pd.Timestamp("2023-06-13 09:00")
    assert np.isclose(X.loc[late, "rolling_mean_24h"],
                      load.loc[anchor - pd.Timedelta(hours=23): anchor].mean())


@pytest.mark.parametrize("bad", [-1, 23, 24])
def test_issue_hour_out_of_range_is_rejected(synth, cal_h, bad):
    with pytest.raises(ValueError):
        build_features(synth, cal_h, issue_hour=bad)
