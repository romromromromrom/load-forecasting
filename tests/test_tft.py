"""TFT : formes, quantiles, et surtout absence de fuite dans les fenêtres (ignoré sans torch)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from rte_forecast.evaluation import tft_backtest as tb  # noqa: E402
from rte_forecast.features import build_features, feature_columns  # noqa: E402
from rte_forecast.features.build_features import LOAD_COLS  # noqa: E402
from rte_forecast.models.tft import TemporalFusionTransformer, quantile_loss  # noqa: E402

SMALL = tb.TFTConfig(lookback=48, d_model=16, n_heads=2, dropout=0.0, epochs=1, batch_size=64)


@pytest.fixture(scope="module")
def prepared(synth, cal_h):
    X = build_features(synth, cal_h, weather_mode="normal")
    cov, scale_cols = tb.build_covariates(X, "normal")
    return synth, cov, scale_cols


def test_model_shapes_quantiles_sorted_and_gradients_flow():
    torch.manual_seed(0)
    m = TemporalFusionTransformer(n_past=6, n_future=5, d_model=16, n_heads=2)
    past, fut = torch.randn(4, 30, 6), torch.randn(4, 12, 5)
    out = m(past, fut)
    assert out.shape == (4, 12, 3)
    assert (out[..., 0] <= out[..., 1]).all() and (out[..., 1] <= out[..., 2]).all()
    loss = quantile_loss(out, torch.randn(4, 12), m.quantiles)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in m.parameters())


def test_quantile_loss_is_zero_for_a_perfect_forecast_and_asymmetric():
    y = torch.tensor([[1.0, 2.0]])
    perfect = y.unsqueeze(-1).repeat(1, 1, 3)
    assert float(quantile_loss(perfect, y, (0.1, 0.5, 0.9))) == 0.0
    under, over = perfect - 1, perfect + 1                      # sous- / sur-estimation
    assert float(quantile_loss(under, y, (0.9,))) > float(quantile_loss(over, y, (0.9,)))


def test_covariates_contain_no_load_and_temperatures_are_flagged_for_scaling(prepared):
    _, cov, scale_cols = prepared
    assert not set(LOAD_COLS) & set(cov.columns)
    assert "load_mw" not in cov.columns
    assert {"temperature_forecast", "temperature_forecast_squared"} <= set(scale_cols)
    assert set(cov.columns) >= {"hour_sin", "hour_cos", "weekday_sin", "is_public_holiday"}
    assert len(cov.columns) == len([c for c in feature_columns("normal") if c not in LOAD_COLS]) + 4


def test_horizon_and_anchor_positions(prepared):
    df, _, _ = prepared
    assert tb.horizon(None) == 24 and tb.horizon(9) == 38
    d = pd.DatetimeIndex(["2023-06-14"])
    assert df.index[tb.day_anchors(df.index, d, None)[0]] == pd.Timestamp("2023-06-13 23:00")
    a = tb.day_anchors(df.index, d, 9)[0]
    assert df.index[a] == pd.Timestamp("2023-06-13 09:00")
    assert df.index[a + tb.horizon(9)] == pd.Timestamp("2023-06-14 23:00")     # dernier pas = D 23:00


def test_fold_anchor_ranges_never_let_train_targets_reach_validation_or_test():
    test_pos, lookback, val_days = 30_000, 168, 56
    for h in (24, 38):
        r = tb.fold_anchor_ranges(test_pos, lookback, h, val_days)
        val_start = test_pos - 24 * val_days
        assert r["train"][1] + h == val_start - 1                       # dernière cible d'entraînement
        assert r["val"][0] + 1 == val_start                             # 1re cible de validation
        assert r["val"][1] + h == test_pos - 1                          # dernière cible de validation
        assert r["train"][0] == lookback - 1


def test_standardization_uses_only_the_fit_rows(prepared):
    df, cov, scale_cols = prepared
    n = 20_000
    a = tb.standardize(df["load_mw"], cov, scale_cols, n)
    tampered = df["load_mw"].copy()
    tampered.iloc[n:] += 50_000
    cov2 = cov.copy()
    cov2.loc[cov2.index[n:], "temperature_forecast"] += 30
    b = tb.standardize(tampered, cov2, scale_cols, n)
    assert a[3] == b[3]                                                  # (mu, sd) de la charge
    assert np.allclose(a[1][:n], b[1][:n])                               # covariables du passé


@pytest.mark.parametrize("issue_hour", [None, 9])
def test_prediction_of_day_D_ignores_everything_after_the_anchor(prepared, issue_hour):
    df, cov, scale_cols = prepared
    P, C, Y, scale = (torch.as_tensor(x) if isinstance(x, np.ndarray) else x
                      for x in tb.standardize(df["load_mw"], cov, scale_cols, 30_000))
    torch.manual_seed(0)
    model = TemporalFusionTransformer(P.shape[1], C.shape[1], 16, 2, dropout=0.0)
    h = tb.horizon(issue_hour)
    day = pd.DatetimeIndex(["2023-06-14"])
    anchor = tb.day_anchors(df.index, day, issue_hour)
    a = tb.predict_days(model, P, C, Y, anchor, SMALL, h, scale)

    P2, Y2 = P.clone(), Y.clone()
    P2[anchor[0] + 1:, 0] += 5.0                                         # charge inconnue à l'émission
    Y2[anchor[0] + 1:] += 5.0
    b = tb.predict_days(model, P2, C, Y2, anchor, SMALL, h, scale)
    assert np.allclose(a, b), "fuite : la prévision de D lit la charge postérieure à l'ancre"

    P3 = P.clone()
    P3[anchor[0], 0] += 5.0                                              # dernière charge connue
    c = tb.predict_days(model, P3, C, Y, anchor, SMALL, h, scale)
    assert not np.allclose(a, c)                                         # contrôle non vacueux


def test_training_runs_on_a_tiny_problem_and_keeps_the_best_state(prepared):
    df, cov, scale_cols = prepared
    P, C, Y, _ = (torch.as_tensor(x) if isinstance(x, np.ndarray) else x
                  for x in tb.standardize(df["load_mw"], cov, scale_cols, 30_000))
    h = 24
    tr = torch.arange(200, 400)
    va = torch.arange(400, 460)
    model, info = tb.train_model(P, C, Y, tr, va, SMALL, h, seed=0, log=lambda *_: None)
    assert info["best_epoch"] == 1 and np.isfinite(info["best_val_pinball"])
    out = tb.predict_days(model, P, C, Y, np.array([1000]), SMALL, h, (0.0, 1.0))
    assert out.shape == (1, 24, 3)
