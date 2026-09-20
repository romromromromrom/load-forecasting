"""Découpages STRICTEMENT chronologiques (jamais de shuffle) et gardes anti-fuite."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp     # inclus (dernière heure)

    @property
    def label(self) -> str:
        return f"F{self.fold_id:02d} {self.test_start:%Y-%m} → {self.test_end:%Y-%m}"


def assert_train_before_test(train_index: pd.DatetimeIndex, test_index: pd.DatetimeIndex) -> None:
    """Lève AssertionError si une seule observation d'entraînement n'est pas antérieure au test."""
    if len(train_index) and len(test_index):
        assert train_index.max() < test_index.min(), (
            f"fuite temporelle : train jusqu'à {train_index.max()} >= test dès {test_index.min()}")


def chronological_split(index: pd.DatetimeIndex, train_end: str, val_end: str, test_end: str
                        ) -> dict[str, pd.DatetimeIndex]:
    """train <= train_end < validation <= val_end < test <= test_end."""
    t_end = pd.Timestamp(train_end) + pd.Timedelta(hours=23)
    v_end = pd.Timestamp(val_end) + pd.Timedelta(hours=23)
    e_end = pd.Timestamp(test_end) + pd.Timedelta(hours=23)
    parts = {"train": index[index <= t_end],
             "validation": index[(index > t_end) & (index <= v_end)],
             "test": index[(index > v_end) & (index <= e_end)]}
    assert_train_before_test(parts["train"], parts["validation"])
    assert_train_before_test(parts["validation"], parts["test"])
    return parts


def expanding_folds(first_test_start: str, last_day: str, months: int = 3) -> list[Fold]:
    """Folds expanding-window : l'entraînement du fold k = tout ce qui précède `test_start`."""
    folds, start, i = [], pd.Timestamp(first_test_start), 1
    last = pd.Timestamp(last_day) + pd.Timedelta(hours=23)
    while start <= last:
        end = min(start + pd.DateOffset(months=months) - pd.Timedelta(hours=1), last)
        folds.append(Fold(i, start, end))
        start, i = start + pd.DateOffset(months=months), i + 1
    return folds


def split_of(ts: pd.DatetimeIndex, val_start: str, test_start: str) -> pd.Series:
    """Étiquette train/validation/test d'un horodatage (par année de la config)."""
    lab = pd.Series("train_cv", index=ts, dtype=object)
    lab[ts >= pd.Timestamp(val_start)] = "validation"
    lab[ts >= pd.Timestamp(test_start)] = "test"
    return lab
