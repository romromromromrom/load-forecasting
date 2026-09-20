"""Étape 2 : statistiques historiques et vérification empirique des hypothèses métier."""

import pandas as pd

from rte_forecast.calendar import build_daily_calendar
from rte_forecast.config import load_config, resolve_path
from rte_forecast.data.pipeline import load_hourly
from rte_forecast.evaluation import analysis

if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    cfg = load_config()
    out = resolve_path(cfg, "results_dir")
    out.mkdir(parents=True, exist_ok=True)
    df = load_hourly(cfg)
    load = df["load_mw"]
    years = sorted(set(df.index.year))
    cal_d = build_daily_calendar(df.index.min() - pd.Timedelta(days=45),
                                 df.index.max() + pd.Timedelta(days=45))

    tables = {
        "analysis_summer_trough": analysis.summer_trough_check(load, years),
        "analysis_summer_shape": analysis.summer_shape(load, cal_d, years),
        "analysis_year_end": analysis.year_end_table(load, years),
        "analysis_holidays": analysis.holiday_table(load, cal_d),
        "analysis_imputation": analysis.imputation_check(load),
    }
    for name, t in tables.items():
        t.to_parquet(out / f"{name}.parquet")
        print(f"\n=== {name}\n{t.round(3).to_string(index=False)[:3500]}")
