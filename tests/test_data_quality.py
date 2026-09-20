import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from rte_forecast.data import cleaning, imputation, loaders


def _hourly(n=24 * 60, start="2023-01-01"):
    idx = pd.date_range(start, periods=n, freq="h", name="timestamp")
    load = 50000 + 5000 * np.sin(np.arange(n) * 2 * np.pi / 24) + 3000 * np.sin(np.arange(n) * 2 * np.pi / 168)
    return pd.DataFrame({"load_mw": load}, index=idx)


def test_duplicated_timestamps_are_rejected():
    df = _hourly()
    dup = pd.concat([df, df.iloc[[10]]]).sort_index()
    with pytest.raises(ValueError, match="dupliqués"):
        cleaning.validate_hourly(dup)


def test_nan_target_is_rejected():
    df = _hourly()
    df.iloc[100, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        cleaning.validate_hourly(df)


def test_irregular_time_step_is_rejected():
    df = _hourly().drop(_hourly().index[50])
    with pytest.raises(ValueError, match="irrégulier"):
        cleaning.validate_hourly(df)


def test_unsorted_index_is_rejected():
    with pytest.raises(ValueError, match="non trié"):
        cleaning.validate_hourly(_hourly().iloc[::-1])


def test_clean_series_passes_validation():
    cleaning.validate_hourly(_hourly())


def test_invalid_observations_are_detected():
    df = _hourly()
    s = df["load_mw"].copy()
    s.iloc[900] = np.nan          # trou
    s.iloc[950] = 0.0             # valeur nulle
    s.iloc[1000] = s.iloc[1000] * 3   # aberration
    bad = cleaning.detect_invalid(s)
    assert bad.iloc[900] and bad.iloc[950] and bad.iloc[1000]
    assert bad.sum() <= 6        # pas de faux positifs en cascade


def test_imputation_is_causal_and_plausible():
    s = _hourly()["load_mw"]
    truth = s.copy()
    gap = slice(800, 830)
    s.iloc[gap] = np.nan
    invalid = s.isna()
    filled = imputation.impute_causal(s, invalid)
    assert not filled.isna().any()
    assert np.abs(filled.iloc[gap] - truth.iloc[gap]).max() < 0.05 * truth.mean()

    # causalité : changer le FUTUR du trou ne change pas la valeur reconstruite
    s2 = s.copy()
    s2.iloc[840:] = s2.iloc[840:] + 20000
    filled2 = imputation.impute_causal(s2, s2.isna())
    assert np.allclose(filled.iloc[gap], filled2.iloc[gap])


def test_missing_report_lists_contiguous_gaps():
    s = _hourly()["load_mw"]
    inv = pd.Series(False, index=s.index)
    inv.iloc[10:14] = True
    inv.iloc[50] = True
    rep = cleaning.missing_report(s, inv)
    assert list(rep["hours"]) == [4, 1]


def test_eco2mix_zip_parsing_drops_disclaimer_and_reads_latin1(tmp_path):
    header = "Périmètre\tNature\tDate\tHeures\tConsommation\tPrévision J-1\tPrévision J\tFioul\t"
    rows = ["France\tDonnées définitives\t2023-01-01\t00:00\t47571\t46600\t48000\t81\t",
            "France\tDonnées définitives\t2023-01-01\t00:15\t\t46050\t47500\t\t",
            "France\tDonnées définitives\t2023-01-01\t00:30\t46342\t45500\t47000\t68\t",
            "RTE ne pourra être tenu responsable de l'usage...\t\t\t\t\t\t\t\t"]
    payload = ("\n".join([header, *rows]) + "\n").encode("latin-1")
    path = tmp_path / "eCO2mix_test.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("eCO2mix_test.xls", payload)
    df = loaders.read_eco2mix_zip(path)
    assert len(df) == 3 and df["consommation_mw"].notna().sum() == 2
    hourly = cleaning.aggregate_hourly(df)
    assert len(hourly) == 1 and hourly["load_mw"].iloc[0] == pytest.approx((47571 + 46342) / 2)
    _ = io  # noqa: F841
