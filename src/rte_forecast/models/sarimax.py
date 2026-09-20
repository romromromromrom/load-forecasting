"""SARIMAX day-ahead avec température en variable exogène.

Modèle : SARIMAX(p,d,q)(P,D,Q,24) sur la charge horaire, exogènes = température prévue, son carré,
5 indicatrices de jour de semaine (lundi..vendredi -> samedi/dimanche = référence, 2 dummies week-end
distincts), férié et pont. Les paramètres sont ré-estimés au début de chaque fold sur une fenêtre
glissante des `train_window_days` derniers jours (SARIMAX sur plusieurs années horaires est
inutilement lent), puis le filtre de Kalman est simplement PROLONGÉ (`extend`, sans ré-estimation)
jour après jour : à D 00:00 on connaît la charge jusqu'à D-1 23:00 et on prévoit les 24 h de D.
Avec ``issue_hour=k`` l'émission a lieu à D-1 (k+1):00 : le filtre est arrêté à D-1 k:00 et on
prévoit D-1 (k+1):00 -> D 23:00, dont on ne garde que les 24 h de D.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

EXOG_COLS = ["temperature_forecast", "temperature_forecast_squared", "is_saturday", "is_sunday",
             "is_monday", "is_friday", "is_public_holiday", "is_bridge_day"]


def sarimax_exog(X: pd.DataFrame) -> pd.DataFrame:
    """Exogènes SARIMAX construits à partir du frame de features (aucune info sur la charge)."""
    e = pd.DataFrame(index=X.index)
    e["temperature_forecast"] = X["temperature_forecast"]
    e["temperature_forecast_squared"] = X["temperature_forecast_squared"] / 100.0
    e["is_saturday"] = (X["weekday"] == 5).astype(float)
    e["is_sunday"] = (X["weekday"] == 6).astype(float)
    e["is_monday"] = (X["weekday"] == 0).astype(float)
    e["is_friday"] = (X["weekday"] == 4).astype(float)
    e["is_public_holiday"] = X["is_public_holiday"].astype(float)
    e["is_bridge_day"] = X["is_bridge_day"].astype(float)
    return e[EXOG_COLS]


class SarimaxDayAhead:
    """Ajuste sur une fenêtre glissante puis prévoit jour par jour sans ré-estimer."""

    name = "sarimax"

    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 1, 1, 24), train_window_days=84,
                 maxiter=40, issue_hour: int | None = None):
        self.order, self.seasonal_order = tuple(order), tuple(seasonal_order)
        self.known = 23 if issue_hour is None else int(issue_hour)
        self.train_window_days, self.maxiter = train_window_days, maxiter
        self.fit_seconds = 0.0

    def fit_predict(self, load: pd.Series, exog: pd.DataFrame, start: pd.Timestamp,
                    end: pd.Timestamp) -> pd.Series:
        """Prévisions horaires pour [start, end] (jours entiers), ajustées sur données < start."""
        def last_known(day: pd.Timestamp) -> pd.Timestamp:
            return day - pd.Timedelta(hours=24 - self.known)      # D-1 known:00 (23 -> D-1 23:00)

        win_start = start - pd.Timedelta(days=self.train_window_days)
        state_end = last_known(start.normalize())
        y_tr = load.loc[win_start:state_end]
        x_tr = exog.loc[y_tr.index]
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(y_tr, exog=x_tr, order=self.order, seasonal_order=self.seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=self.maxiter)
        self.fit_seconds = time.perf_counter() - t0
        preds = []
        for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
            idx = pd.date_range(day, periods=24, freq="h")
            ahead = pd.date_range(state_end + pd.Timedelta(hours=1), idx[-1], freq="h")
            fc = res.forecast(steps=len(ahead), exog=exog.loc[ahead])
            preds.append(pd.Series(np.asarray(fc)[-24:], index=idx))
            new_end = last_known(day + pd.Timedelta(days=1))
            seen = pd.date_range(state_end + pd.Timedelta(hours=1), new_end, freq="h")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = res.extend(load.loc[seen], exog=exog.loc[seen])  # observe, sans ré-estimer
            state_end = new_end
        return pd.concat(preds).rename("sarimax")
