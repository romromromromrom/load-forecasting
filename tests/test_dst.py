"""Changement d'heure : l'index est une heure locale « murale » naïve, 24 lignes par jour.

Conséquence voulue : un retard de 168 lignes = même heure d'horloge, même jour de semaine, une
semaine plus tôt, y compris autour des passages heure d'été/hiver (la charge suit les habitudes
humaines, donc l'horloge). Ces tests verrouillent cette propriété ; un index UTC ou tz-aware avec
``shift(168)`` / ``Timedelta(hours=168)`` la casserait pendant les 7 jours qui suivent chaque
changement d'heure.
"""

import pandas as pd
import pytest

from rte_forecast.features import build_features
from rte_forecast.models import persistence_j7

LAGS = {"load_lag_24h": 24, "load_lag_48h": 48, "load_lag_168h": 168, "load_lag_336h": 336}


def _last_sunday(year: int, month: int) -> pd.Timestamp:
    last = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return last - pd.Timedelta(days=(last.weekday() + 1) % 7)


DST_DAYS = [_last_sunday(y, m) for y in range(2020, 2025) for m in (3, 10)]


@pytest.fixture(scope="module")
def encoded(synth):
    """Série dont la charge encode son propre horodatage (heures depuis le début)."""
    df = synth.copy()
    df["load_mw"] = ((df.index - df.index[0]) / pd.Timedelta(hours=1)).astype(float)
    return df


def _source_timestamp(values: pd.Series, origin: pd.Timestamp) -> pd.Series:
    return origin + pd.to_timedelta(values, unit="h")


def test_index_is_naive_regular_and_has_24_rows_on_dst_days(synth):
    assert synth.index.tz is None
    assert set(pd.Series(synth.index).diff().dropna()) == {pd.Timedelta(hours=1)}
    per_day = synth.groupby(synth.index.normalize()).size()
    assert (per_day == 24).all()
    assert all(per_day.loc[d] == 24 for d in DST_DAYS)


def test_weekly_lags_keep_clock_hour_and_weekday_across_dst(encoded, cal_h):
    X = build_features(encoded, cal_h).dropna(subset=list(LAGS))
    for col, hours in LAGS.items():
        src = _source_timestamp(X[col], encoded.index[0])
        assert (src.to_numpy() == (X.index - pd.Timedelta(hours=hours)).to_numpy()).all()
        if hours % 168 == 0:
            assert (src.dt.hour.to_numpy() == X.index.hour).all()
            assert (src.dt.weekday.to_numpy() == X.index.weekday).all()


@pytest.mark.parametrize("dst_day", DST_DAYS)
def test_lag_168h_is_the_same_clock_hour_last_week_around_dst(encoded, cal_h, dst_day):
    X = build_features(encoded, cal_h)
    window = X.loc[dst_day - pd.Timedelta(days=1): dst_day + pd.Timedelta(days=8)]
    src = _source_timestamp(window["load_lag_168h"], encoded.index[0])
    assert (src.dt.hour.to_numpy() == window.index.hour).all()
    assert (src.dt.weekday.to_numpy() == window.index.weekday).all()
    assert ((window.index - src).unique() == pd.Timedelta(days=7)).all()


def test_persistence_j7_uses_the_same_clock_hour_across_dst(encoded):
    pred = persistence_j7(encoded["load_mw"]).dropna()
    src = _source_timestamp(pred, encoded.index[0])
    assert (src.dt.hour.to_numpy() == pred.index.hour).all()
    assert (src.dt.weekday.to_numpy() == pred.index.weekday).all()


def test_elapsed_time_arithmetic_on_tz_aware_index_would_misalign():
    """Documente le piège : 168 h réelles ≠ 7 jours calendaires autour du changement d'heure."""
    t = pd.Timestamp("2024-04-01 08:00", tz="Europe/Paris")     # lundi après le passage à l'heure d'été
    assert (t - pd.Timedelta(hours=168)).hour == 7               # une heure d'horloge trop tôt
    assert (t - pd.DateOffset(days=7)).hour == 8                 # jours calendaires : correct
