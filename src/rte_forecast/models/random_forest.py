"""Fabrique Random Forest (scikit-learn >= 1.4 gère nativement les NaN)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor


def make_random_forest(params: dict[str, Any] | None = None) -> RandomForestRegressor:
    p = {"n_estimators": 200, "max_depth": 18, "min_samples_leaf": 5, "max_features": 0.5,
         "random_state": 42, "n_jobs": 4}
    p.update(params or {})
    return RandomForestRegressor(**p)
