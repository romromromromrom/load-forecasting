import numpy as np
import pandas as pd
import pytest

from rte_forecast.evaluation import metrics


def test_basic_metrics():
    y = np.array([100.0, 200.0, 300.0])
    p = np.array([110.0, 190.0, 330.0])
    assert metrics.mae(y, p) == pytest.approx((10 + 10 + 30) / 3)
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert metrics.bias(y, p) == pytest.approx((10 - 10 + 30) / 3)   # >0 = surestimation


def test_metrics_by_period_contains_all_row():
    df = pd.DataFrame({"model": "m", "y_true": [1.0, 2, 3, 4], "y_pred": [1.5, 2, 2, 5],
                       "period_type": ["NORMAL", "NORMAL", "HOLIDAY", "SUMMER"]})
    t = metrics.metrics_by_period(df, ["model"])
    assert set(t["period"].astype(str)) == {"Normal days", "Holidays", "Summer (Aug window)", "ALL"}
    assert t[t["period"] == "ALL"]["n"].iloc[0] == 4
