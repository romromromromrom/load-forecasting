"""Benchmark : recherche d'hyperparamètres, walk-forward, agrégation, importances, SHAP.

Protocole (tout est chronologique, aucun shuffle) :

* split fixe : train <= 2022 < validation 2023 < test 2024 ;
* hyperparamètres choisis sur validation 2023 avec un modèle entraîné sur <= 2022 ;
* walk-forward expanding-window : 1 fold = 1 trimestre de 2022 à 2024, ré-entraînement à chaque
  fold sur tout ce qui précède le fold ; prévision jour-J émise à J 00:00 (retards >= 24 h) par
  défaut, ou à J-1 10:00 avec ``features.issue_hour: 9`` (voir features/build_features.py) ;
* les moteurs de règles n'utilisent que l'historique antérieur au début de la période prévue.
"""

from __future__ import annotations

import itertools
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from rte_forecast.business_rules.weather import WeatherNormalizer
from rte_forecast.calendar import build_daily_calendar, build_hourly_calendar
from rte_forecast.config import load_config, resolve_path
from rte_forecast.data.pipeline import load_hourly
from rte_forecast.evaluation import metrics, splits
from rte_forecast.features import build_features, feature_columns, make_temperature_forecast
from rte_forecast.forecast import HybridForecaster
from rte_forecast.models import SarimaxDayAhead, make_random_forest, make_xgboost, persistence_j7
from rte_forecast.models.sarimax import sarimax_exog

MODEL_LABELS = {
    "naive_j7": "Naive J-7", "rte_j1": "RTE J-1 (référence externe)", "sarimax": "SARIMAX",
    "xgboost": "XGBoost", "xgboost_rules": "XGBoost + règles (planning)",
    "xgboost_rules_adaptive": "XGBoost + règles (adaptatif)",
    "random_forest": "Random Forest", "random_forest_rules": "RF + règles (planning)",
    "random_forest_rules_adaptive": "RF + règles (adaptatif)",
}
WEATHER_FREE = {"naive_j7", "rte_j1"}
# modèle hybride -> (modèle « normal », mode de niveau des règles)
HYBRID_MODELS = {
    "xgboost_rules": ("xgboost_normal", "planning"),
    "xgboost_rules_adaptive": ("xgboost_normal", "adaptive"),
    "random_forest_rules": ("random_forest_normal", "planning"),
    "random_forest_rules_adaptive": ("random_forest_normal", "adaptive"),
}


def _issue_hour(cfg: dict) -> int | None:
    """Dernière heure de D-1 connue à l'émission (None = émission à D 00:00, historique)."""
    return cfg["features"].get("issue_hour")


def _recent_lag_days(issue_hour: int | None) -> int:
    """Jours entiers à ignorer avant J pour les règles adaptatives (J-1 incomplet à l'émission)."""
    return 0 if issue_hour is None else 1


def _cfg_params(cfg: dict, name: str, best: dict | None) -> dict:
    p = dict(cfg["models"][name])
    p.update((best or {}).get(name, {}))
    return p


def _train_mask(X: pd.DataFrame, before: pd.Timestamp, normal_only: bool, t0: pd.Timestamp
                ) -> pd.Series:
    m = (X.index < before) & (X.index >= t0 + pd.Timedelta(days=14)) & X["load_mw"].notna()
    if normal_only:
        m &= (X["special_period_type"] == "NORMAL").to_numpy()
    return pd.Series(m, index=X.index)


def _save_and_describe(model, name: str, stem: Path | None, n_train: int, n_features: int) -> dict:
    """Sauvegarde un modèle à arbres (XGBoost .ubj ; RF joblib compressé) et décrit ses paramètres."""
    info: dict = {"params": json.dumps(model.get_params(), default=str), "n_train": n_train,
                  "n_features": n_features, "file": "", "file_mb": float("nan")}
    if name == "xgboost":
        info["extra"] = json.dumps({"n_trees": int(model.get_booster().num_boosted_rounds())})
    else:
        nodes = [e.tree_.node_count for e in model.estimators_]
        info["extra"] = json.dumps({"n_trees": len(nodes), "mean_nodes_per_tree": float(np.mean(nodes)),
                                    "max_depth_reached": int(max(e.tree_.max_depth
                                                                 for e in model.estimators_))})
    if stem is not None:
        stem.parent.mkdir(parents=True, exist_ok=True)
        path = stem.with_suffix(".ubj" if name == "xgboost" else ".joblib")
        if name == "xgboost":
            model.save_model(path)
        else:
            joblib.dump(model, path, compress=3)
        info["file"], info["file_mb"] = path.name, path.stat().st_size / 1e6
    return info


def _fit(model, X: pd.DataFrame, y: pd.Series):
    t = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t


def _fold_task(fold: splits.Fold, Xp: pd.DataFrame, Xc: pd.DataFrame, cols: list[str],
               p_xgb: dict, p_rf: dict, t0: pd.Timestamp, save_dir: Path | None = None) -> dict:
    """Entraîne les 4 modèles à arbres d'un fold et prédit la fenêtre de test.

    Si `save_dir` est fourni, chaque modèle est sauvegardé ICI (dans le processus qui l'a entraîné)
    et seul un descriptif léger est renvoyé au processus principal.
    """
    idx = Xp.index[(Xp.index >= fold.test_start) & (Xp.index <= fold.test_end)]
    out, times, sizes, infos = {}, {}, {}, {}
    for name, factory, params in (("xgboost", make_xgboost, p_xgb),
                                  ("random_forest", make_random_forest, p_rf)):
        for variant, X, normal_only in (("", Xp, False), ("_normal", Xc, True)):
            m = _train_mask(X, fold.test_start, normal_only, t0)
            splits.assert_train_before_test(X.index[m], idx)
            model, secs = _fit(factory(params), X.loc[m, cols], X.loc[m, "load_mw"])
            key = name + variant
            out[key] = pd.Series(model.predict(X.loc[idx, cols]), index=idx)
            times[key], sizes[key] = secs, int(m.sum())
            if save_dir is not None:
                infos[key] = _save_and_describe(model, name, save_dir / f"F{fold.fold_id:02d}_{key}",
                                                sizes[key], len(cols))
    return {"fold": fold, "pred": pd.DataFrame(out), "fit_seconds": times, "n_train": sizes,
            "model_info": infos}


def _final_task(Xp: pd.DataFrame, Xc: pd.DataFrame, cols: list[str], p_xgb: dict, p_rf: dict,
                t0: pd.Timestamp, save_dir: Path) -> dict:
    """Modèles « finaux » : entraînés sur TOUTES les données disponibles (aucune évaluation possible)."""
    end = Xp.index.max() + pd.Timedelta(hours=1)
    infos = {}
    for name, factory, params in (("xgboost", make_xgboost, p_xgb),
                                  ("random_forest", make_random_forest, p_rf)):
        for variant, X, normal_only in (("", Xp, False), ("_normal", Xc, True)):
            m = _train_mask(X, end, normal_only, t0)
            model, _ = _fit(factory(params), X.loc[m, cols], X.loc[m, "load_mw"])
            infos[name + variant] = _save_and_describe(model, name, save_dir / f"final_{name}{variant}",
                                                       int(m.sum()), len(cols))
    return {"model_info": infos}


def _sarimax_task(fold: splits.Fold, load: pd.Series, exog: pd.DataFrame, params: dict,
                  save_dir: Path | None = None) -> dict:
    m = SarimaxDayAhead(**params)
    pred = m.fit_predict(load, exog, fold.test_start, fold.test_end)
    info = {}
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        # les résultats statsmodels embarquent le filtre de Kalman (~124 Mo par fold) : on ne garde que
        # la spécification et les coefficients estimés, qui suffisent à reconstruire le modèle
        spec = {"order": list(params["order"]), "seasonal_order": list(params["seasonal_order"]),
                "exog_names": m.exog_names, "train_window": list(m.train_window),
                "issue_hour": params.get("issue_hour"), "params": m.fitted_params}
        path = save_dir / f"F{fold.fold_id:02d}_sarimax.json"
        path.write_text(json.dumps(spec, indent=2))
        info["sarimax"] = {"params": json.dumps({k: v for k, v in params.items()}, default=str),
                           "n_train": params["train_window_days"] * 24,
                           "n_features": len(m.fitted_params), "file": path.name,
                           "file_mb": path.stat().st_size / 1e6,
                           "extra": json.dumps({"coefficients": m.fitted_params})}
    return {"fold": fold, "pred": pred.to_frame("sarimax"), "fit_seconds": {"sarimax": m.fit_seconds},
            "n_train": {"sarimax": params["train_window_days"] * 24}, "model_info": info}


def _tuning_task(name: str, params: dict, Xtr, ytr, Xva, yva, cols) -> dict:
    factory = make_xgboost if name == "xgboost" else make_random_forest
    model, secs = _fit(factory(params), Xtr[cols], ytr)
    pred = model.predict(Xva[cols])
    # libellés texte : la colonne mélange int et None (max_depth), illisible sinon en Parquet
    return {"model": name, **{f"hp_{k}": str(v) for k, v in params.items()},
            "val_mae": metrics.mae(yva, pred), "val_rmse": metrics.rmse(yva, pred),
            "fit_seconds": secs}


def hyperparameter_search(cfg: dict, df: pd.DataFrame, jobs: int, quick: bool = False
                          ) -> tuple[pd.DataFrame, dict]:
    """Grille sur validation 2023 (entraînement <= 2022), features en mode 'normal'."""
    cal_h = build_hourly_calendar(df.index, **_cal_kwargs(cfg))
    Xp = build_features(df, cal_h, weather_mode="normal", issue_hour=_issue_hour(cfg))
    parts = splits.chronological_split(df.index, **cfg["split"])
    t0 = df.index.min()
    tr = Xp.loc[parts["train"]]
    tr = tr[tr.index >= t0 + pd.Timedelta(days=14)]
    va = Xp.loc[parts["validation"]]
    cols = feature_columns("normal")
    tasks = []
    for name in ("xgboost", "random_forest"):
        grid = cfg["models"]["hyperparameter_grid"][name]
        keys = list(grid)
        combos = list(itertools.product(*[grid[k] for k in keys]))
        if quick:
            combos = combos[:2]
        for combo in combos:
            p = dict(cfg["models"][name])
            p.update(dict(zip(keys, combo, strict=True)))
            p["n_jobs"] = 1
            tasks.append((name, p))
    res = Parallel(n_jobs=jobs)(delayed(_tuning_task)(n, p, tr, tr["load_mw"], va, va["load_mw"], cols)
                                for n, p in tasks)
    table = pd.DataFrame(res)
    best: dict[str, dict] = {}
    for name in ("xgboost", "random_forest"):
        grid_keys = list(cfg["models"]["hyperparameter_grid"][name])
        cand = [(r["val_mae"], p) for (n, p), r in zip(tasks, res, strict=True) if n == name]
        _, p_best = min(cand, key=lambda c: c[0])
        best[name] = {k: p_best[k] for k in grid_keys}   # types Python d'origine
    return table, best


def _cal_kwargs(cfg: dict) -> dict:
    r = cfg["rules"]
    return {"year_end_weeks": tuple(r["year_end_weeks"]), "year_start_weeks": tuple(r["year_start_weeks"]),
            "summer_weeks_before": r["summer_window_weeks_before"],
            "summer_weeks_after": r["summer_window_weeks_after"]}


def _num(row: pd.Series, col: str) -> float:
    return float(row[col]) if col in row.index and pd.notna(row[col]) else float("nan")


def _rule_parameter_rows(hf: HybridForecaster, hybrid: str, mode: str) -> list[dict]:
    """Paramètres estimés par les moteurs de règles (année analogue, sensibilité thermique, niveaux)."""
    rows = []
    for (kind, year), res in hf._cache.items():
        if not isinstance(res, pd.DataFrame):
            continue
        first = res.iloc[0]
        rows.append({
            "model": hybrid, "weather_mode": mode, "kind": kind, "year": int(year),
            "level_mode": hf.level_mode, "reference_year": str(first.get("reference_year", "")),
            "similarity_score": _num(first, "similarity_score"), "baseline_mw": _num(first, "baseline_mw"),
            "baseline_ref_mw": _num(first, "baseline_ref_mw"),
            "thermosensitivity_heat_mw_per_c": _num(first, "thermosensitivity_heat_mw_per_c"),
            "summer_start_level_mw": _num(first, "summer_start_level_mw"),
            "summer_trough_level_mw": _num(first, "summer_trough_level_mw"),
            "summer_end_level_mw": _num(first, "summer_end_level_mw"),
            "shape_json": json.dumps({str(k): v for k, v in res.attrs.get("shape", {}).items()}),
            "coef_json": json.dumps({str(k): v for k, v in res.attrs.get("coefficients", {}).items()}),
            "reason": str(first.get("reason", ""))})
    return rows


def _library_versions() -> dict:
    import sklearn
    import statsmodels
    import xgboost
    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__, "joblib": joblib.__version__}


def run_all(cfg: dict | None = None, weather_modes=("normal", "noisy", "perfect"), jobs: int = 4,
            quick: bool = False, skip_sarimax: bool = False, save_models: bool = False,
            models_dir: str | Path | None = None, log=print) -> dict[str, Path]:
    """Exécute tout le benchmark et écrit les Parquet/JSON de résultats dans data/results."""
    warnings.filterwarnings("ignore")
    cfg = cfg or load_config()
    out_dir = resolve_path(cfg, "results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    mdl_dir = Path(models_dir) if models_dir else resolve_path(cfg, "models_dir")
    df = load_hourly(cfg)
    load = df["load_mw"]
    t0 = df.index.min()
    last_day = str(df.index.max().date())
    ck = _cal_kwargs(cfg)
    cal_h = build_hourly_calendar(df.index, **ck)
    cal_d = build_daily_calendar(df.index.min() - pd.Timedelta(days=45),
                                 df.index.max() + pd.Timedelta(days=45), **ck)
    folds = splits.expanding_folds(cfg["cv"]["first_test_start"], last_day, cfg["cv"]["fold_months"])
    if quick:
        folds = folds[-2:]
    parts = splits.chronological_split(df.index, **cfg["split"])
    split_info = {k: [str(v.min()), str(v.max()), len(v)] for k, v in parts.items()}

    log("== recherche d'hyperparamètres (train <= 2022 -> validation 2023)")
    tune, best = hyperparameter_search(cfg, df, jobs, quick)
    tune.to_parquet(out_dir / "hyperparameters.parquet")
    (out_dir / "best_params.json").write_text(json.dumps(best, indent=2))
    log("   meilleurs paramètres :", best)

    val_start = str(pd.Timestamp(cfg["split"]["train_end"]) + pd.Timedelta(days=1))
    test_start = str(pd.Timestamp(cfg["split"]["val_end"]) + pd.Timedelta(days=1))
    p_xgb = {**_cfg_params(cfg, "xgboost", best), "n_jobs": 1}
    p_rf = {**_cfg_params(cfg, "random_forest", best), "n_jobs": 1}
    p_sarimax = {k: cfg["models"]["sarimax"][k]
                 for k in ("order", "seasonal_order", "train_window_days", "maxiter")}
    issue = _issue_hour(cfg)
    p_sarimax["issue_hour"] = issue

    preds, fold_rows, explanations, model_rows, rule_rows = [], [], [], [], []
    ref_idx = pd.DatetimeIndex(np.concatenate([
        df.index[(df.index >= f.test_start) & (df.index <= f.test_end)] for f in folds]))
    base = pd.DataFrame({"y_true": load.reindex(ref_idx), "period_type":
                         cal_h["special_period_type"].reindex(ref_idx)})
    base["split"] = splits.split_of(ref_idx, val_start, test_start)
    fold_of = pd.Series(0, index=ref_idx)
    for f in folds:
        fold_of[(ref_idx >= f.test_start) & (ref_idx <= f.test_end)] = f.fold_id
    base["fold_id"] = fold_of

    def add(model, mode, series, method=None):
        d = base.copy()
        d["y_pred"] = series.reindex(ref_idx).to_numpy()
        d["model"], d["weather_mode"] = model, mode
        d["method"] = model if method is None else method.reindex(ref_idx).to_numpy()
        preds.append(d)

    add("naive_j7", "n/a", persistence_j7(load))
    add("rte_j1", "n/a", df["rte_forecast_j1_mw"])

    for mode in weather_modes:
        log(f"== météo '{mode}'")
        nf = cfg["models"].get("noisy_forecast", {})
        tfc = make_temperature_forecast(df, mode, **nf) if mode == "noisy" else \
            make_temperature_forecast(df, mode)
        Xp = build_features(df, cal_h, weather_mode=mode, temperature_forecast=tfc,
                            issue_hour=issue)
        Xc = build_features(df, cal_h, weather_mode=mode, temperature_forecast=tfc, clean_lags=True,
                            max_fallback_weeks=cfg["features"]["max_fallback_weeks"],
                            issue_hour=issue)
        cols = feature_columns(mode)
        save_dir = mdl_dir / mode if save_models else None
        tasks = [delayed(_fold_task)(f, Xp, Xc, cols, p_xgb, p_rf, t0, save_dir) for f in folds]
        if not skip_sarimax:
            ex = sarimax_exog(Xp)
            tasks += [delayed(_sarimax_task)(f, load, ex, p_sarimax, save_dir) for f in folds]
        t = time.perf_counter()
        results = Parallel(n_jobs=jobs)(tasks)
        log(f"   {len(tasks)} tâches en {time.perf_counter() - t:.0f}s")

        if save_models:
            final = _final_task(Xp, Xc, cols, p_xgb, p_rf, t0, save_dir)
            log(f"   modèles finaux (toutes données) sauvegardés dans {save_dir}")
            model_rows += [{"model": k, "weather_mode": mode, "fold_id": 0, **v}
                           for k, v in final["model_info"].items()]
        by_model: dict[str, list[pd.Series]] = {}
        for r in results:
            model_rows += [{"model": k, "weather_mode": mode, "fold_id": r["fold"].fold_id, **v}
                           for k, v in r.get("model_info", {}).items()]
            for c in r["pred"].columns:
                by_model.setdefault(c, []).append(r["pred"][c])
            for name, secs in r["fit_seconds"].items():
                fold_rows.append({"model": name, "weather_mode": mode, "fold_id": r["fold"].fold_id,
                                  "fold_label": r["fold"].label, "fit_seconds": secs,
                                  "n_train": r["n_train"][name]})
        series = {k: pd.concat(v).sort_index() for k, v in by_model.items()}
        for name in ("xgboost", "random_forest", "sarimax"):
            if name in series:
                add(name, mode, series[name])

        normalizer = WeatherNormalizer(df["temperature_realised"], df["temperature_normal"],
                                       cal_h["is_normal_day"], tfc)
        for hyb, (normal, level_mode) in HYBRID_MODELS.items():
            t = time.perf_counter()
            hf = HybridForecaster(load, cal_d, cfg["rules"], normalizer, normal.upper(), level_mode,
                                  _recent_lag_days(issue))
            res = hf.predict(series[normal])
            log(f"   {hyb}: règles en {time.perf_counter() - t:.0f}s ; "
                f"{res['method'].value_counts().to_dict()}")
            add(hyb, mode, res["yhat"], method=res["method"])
            rule_rows += _rule_parameter_rows(hf, hyb, mode)
            ex_rows = res[~res["method"].isin([normal.upper()])].copy()
            ex_rows["model"], ex_rows["weather_mode"] = hyb, mode
            ex_rows["y_true"] = load.reindex(ex_rows.index)
            explanations.append(ex_rows.reset_index())
            # le temps d'entraînement d'un hybride = celui de son modèle « normal » (règles ~ 0 s)
            fold_rows.extend([{**r, "model": hyb} for r in fold_rows
                              if r["model"] == normal and r["weather_mode"] == mode])

    predictions = pd.concat(preds).reset_index(names="timestamp")
    predictions["error"] = predictions["y_pred"] - predictions["y_true"]
    predictions.to_parquet(out_dir / "predictions.parquet")
    pd.concat(explanations).to_parquet(out_dir / "rule_explanations.parquet")

    key = ["model", "weather_mode"]
    metrics.metrics_by_period(predictions, key + ["split"]).to_parquet(out_dir / "metrics_split.parquet")
    metrics.metrics_by_period(predictions, key).to_parquet(out_dir / "metrics_period.parquet")
    fm = (metrics.metrics_by_period(predictions, key + ["fold_id"])
          .query("period == 'ALL'").drop(columns="period"))
    fits = pd.DataFrame(fold_rows).drop_duplicates(["model", "weather_mode", "fold_id"])
    fm.merge(fits, on=key + ["fold_id"], how="left").to_parquet(out_dir / "cv_folds.parquet")

    log("== importances / SHAP")
    _importances(cfg, df, cal_h, best, parts, out_dir, quick, jobs)
    if model_rows:
        pd.DataFrame(model_rows).to_parquet(out_dir / "model_params.parquet")
    if rule_rows:
        pd.DataFrame(rule_rows).to_parquet(out_dir / "rule_params.parquet")
    if save_models:
        manifest = {"issue_hour": issue, "cette_convention": "prévision de D émise à D-1 (issue_hour+1):00",
                    "features_par_scenario": {m: feature_columns(m) for m in weather_modes},
                    "hyperparametres": best, "sarimax": p_sarimax,
                    "train_start": str(t0), "data_end": str(df.index.max()),
                    "nommage": "F<fold>_<modele>.(ubj|joblib|json) = entraîné avant le fold ; "
                               "final_<modele> = entraîné sur toutes les données (2020-2024)",
                    "chargement": {"xgboost": "XGBRegressor().load_model(path)",
                                   "random_forest": "joblib.load(path)",
                                   "sarimax": "json (ordre + coefficients) ; reconstruction : "
                                  "SARIMAX(charge, exog, order, seasonal_order).filter(params)"},
                    "versions": _library_versions()}
        mdl_dir.mkdir(parents=True, exist_ok=True)
        (mdl_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    meta = {"issue_hour": issue, "models_saved": bool(save_models), "models_dir": str(mdl_dir), "split": split_info, "folds": [f.label for f in folds], "weather_modes": list(weather_modes),
            "best_params": best, "models": MODEL_LABELS, "n_predictions": len(predictions)}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return {"results_dir": out_dir}


def _importances(cfg, df, cal_h, best, parts, out_dir: Path, quick: bool, jobs: int = 4) -> None:
    import shap

    Xp = build_features(df, cal_h, weather_mode="normal", issue_hour=_issue_hour(cfg))
    cols = feature_columns("normal")
    t0 = df.index.min()
    tr_end = pd.Timestamp(cfg["split"]["val_end"]) + pd.Timedelta(hours=23)
    m = _train_mask(Xp, tr_end + pd.Timedelta(hours=1), False, t0)
    xgb, _ = _fit(make_xgboost({**_cfg_params(cfg, "xgboost", best), "n_jobs": jobs}), Xp.loc[m, cols],
                  Xp.loc[m, "load_mw"])
    rf, _ = _fit(make_random_forest({**_cfg_params(cfg, "random_forest", best), "n_jobs": jobs}),
                 Xp.loc[m, cols], Xp.loc[m, "load_mw"])
    imp = pd.concat([
        pd.DataFrame({"model": "xgboost", "feature": cols, "importance": xgb.feature_importances_}),
        pd.DataFrame({"model": "random_forest", "feature": cols, "importance": rf.feature_importances_}),
    ])
    imp.to_parquet(out_dir / "feature_importance.parquet")
    test = Xp.loc[parts["test"], cols]
    sample = test.sample(min(len(test), 500 if quick else 3000), random_state=0).sort_index()
    sv = shap.TreeExplainer(xgb).shap_values(sample)
    shap_df = pd.DataFrame(sv, columns=cols, index=sample.index).add_prefix("shap_")
    pd.concat([sample, shap_df], axis=1).reset_index(names="timestamp").to_parquet(
        out_dir / "shap_xgboost.parquet")
