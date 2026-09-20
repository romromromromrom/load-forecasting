"""SARIMAX : l'état du filtre s'arrête à la dernière charge connue à l'émission."""

import numpy as np
import pandas as pd
import pytest

from rte_forecast.features import build_features
from rte_forecast.models.sarimax import SarimaxDayAhead, sarimax_exog

PARAMS = dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, 24), train_window_days=10, maxiter=5)
START, END = pd.Timestamp("2023-06-12"), pd.Timestamp("2023-06-13")     # 2 jours prévus


@pytest.fixture(scope="module")
def exog(synth, cal_h):
    return sarimax_exog(build_features(synth, cal_h))


def _predict(load, exog, issue_hour):
    return SarimaxDayAhead(**PARAMS, issue_hour=issue_hour).fit_predict(load, exog, START, END)


def _perturbed_from(load, ts):
    p = load.copy()
    p.loc[ts:] = p.loc[ts:] * 1.5 + 3000
    return p


def test_hardened_sarimax_first_day_ignores_load_after_previous_day_09h(synth, exog):
    load = synth["load_mw"]
    unknown = START - pd.Timedelta(hours=14)                                # D-1 10:00
    a = _predict(load, exog, 9)
    b = _predict(_perturbed_from(load, unknown), exog, 9)
    day1 = a.index < START + pd.Timedelta(days=1)
    assert np.allclose(a[day1], b[day1])
    assert not np.allclose(a[~day1], b[~day1])          # le 2e jour, lui, voit la perturbation


def test_default_sarimax_does_see_the_evening_before(synth, exog):
    """Contrôle non vacueux : à D 00:00 la même perturbation change la prévision de D."""
    load = synth["load_mw"]
    unknown = START - pd.Timedelta(hours=14)
    a = _predict(load, exog, None)
    b = _predict(_perturbed_from(load, unknown), exog, None)
    assert not np.allclose(a[a.index < START + pd.Timedelta(days=1)],
                           b[b.index < START + pd.Timedelta(days=1)])
