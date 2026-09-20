"""Fabrique XGBoost (régression sur features tabulaires)."""

from __future__ import annotations

from typing import Any

from xgboost import XGBRegressor


def make_xgboost(params: dict[str, Any] | None = None) -> XGBRegressor:
    p = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
         "colsample_bytree": 0.8, "min_child_weight": 5, "random_state": 42, "n_jobs": 4,
         "tree_method": "hist"}
    p.update(params or {})
    return XGBRegressor(**p)
