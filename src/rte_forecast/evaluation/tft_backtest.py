"""Backtest walk-forward du TFT : mêmes folds, mêmes scénarios météo, même schéma de sortie.

Conçu pour tourner sur GPU (Colab/Kaggle) puis rapatrier ``predictions_tft.parquet``.

Protocole (identique à evaluation/backtest.py, adapté aux fenêtres) :

* une prévision = une « ancre » a = dernière charge connue. Émission à D 00:00 (défaut) : a = D-1
  23:00 et H = 24 ; émission à D-1 10:00 (``issue_hour=9``) : a = D-1 09:00 et H = 24 + 14 = 38.
  On ne garde que les 24 h de D (les 24 derniers pas décodés) ;
* le contexte (charge + covariables) s'arrête à l'ancre INCLUSE ; le décodeur ne reçoit que des
  covariables connues à l'avance (calendrier, température « prévue » du scénario météo) ;
* entraînement : toutes les ancres horaires dont la cible (a+1 .. a+H) précède la validation ;
  validation = les ``val_days`` derniers jours avant le fold (arrêt précoce), jamais le test ;
* standardisation (charge, températures) estimée sur les lignes < début du fold uniquement ;
* un fold est écrit sur disque dès qu'il est fini (``resume`` reprend après une coupure Colab).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rte_forecast.calendar import build_hourly_calendar
from rte_forecast.config import load_config
from rte_forecast.data.pipeline import load_hourly
from rte_forecast.evaluation import splits
from rte_forecast.evaluation.backtest import _cal_kwargs
from rte_forecast.features import build_features, feature_columns, make_temperature_forecast
from rte_forecast.features.build_features import LOAD_COLS, check_issue_hour
from rte_forecast.models.tft import TemporalFusionTransformer, quantile_loss

CYCLIC = {"hour": 24, "weekday": 7, "month": 12, "week_of_year": 53}
MODEL_NAME = "tft"


@dataclass
class TFTConfig:
    lookback: int = 168
    d_model: int = 64
    n_heads: int = 4
    lstm_layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 30
    patience: int = 4
    val_days: int = 56
    seeds: int = 1
    amp: bool = False
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)


# --------------------------------------------------------------------------- données / fenêtres
def horizon(issue_hour: int | None) -> int:
    """Nombre de pas à décoder : de l'ancre+1 h jusqu'à D 23:00 (24 si émission à D 00:00)."""
    return 24 + (23 - check_issue_hour(issue_hour))


def build_covariates(X: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, list[str]]:
    """Covariables connues à l'avance (sans aucune charge) + colonnes à standardiser (T°)."""
    out: dict[str, np.ndarray] = {}
    scale_cols: list[str] = []
    for c in (c for c in feature_columns(mode) if c not in LOAD_COLS):
        if c in CYCLIC:
            ang = 2 * np.pi * X[c].to_numpy(float) / CYCLIC[c]
            out[f"{c}_sin"], out[f"{c}_cos"] = np.sin(ang), np.cos(ang)
        else:
            out[c] = X[c].to_numpy(float)
            if c.startswith("temp"):
                scale_cols.append(c)
    return pd.DataFrame(out, index=X.index).astype("float32"), scale_cols


def standardize(load: pd.Series, cov: pd.DataFrame, scale_cols: list[str], fit_rows: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """(P, C, Y, (mu, sd) de la charge). Statistiques sur les `fit_rows` premières lignes SEULEMENT."""
    y = load.to_numpy(np.float64)
    mu, sd = float(y[:fit_rows].mean()), float(y[:fit_rows].std()) or 1.0
    c = cov.copy()
    for col in scale_cols:
        m, s = float(c[col].iloc[:fit_rows].mean()), float(c[col].iloc[:fit_rows].std()) or 1.0
        c[col] = (c[col] - m) / s
    ys = ((y - mu) / sd).astype(np.float32)
    cn = c.to_numpy(np.float32)
    return np.concatenate([ys[:, None], cn], axis=1), cn, ys, (mu, sd)


def fold_anchor_ranges(test_pos: int, lookback: int, h: int, val_days: int
                       ) -> dict[str, tuple[int, int]]:
    """Bornes INCLUSES des ancres d'entraînement / validation pour un fold débutant à `test_pos`.

    Cibles d'entraînement <= val_start-1 ; cibles de validation dans [val_start, test_pos-1].
    """
    val_start = test_pos - 24 * val_days
    return {"train": (lookback - 1, val_start - 1 - h), "val": (val_start - 1, test_pos - 1 - h)}


def gather(P: torch.Tensor, C: torch.Tensor, Y: torch.Tensor, anchors: torch.Tensor, lookback: int,
           h: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fenêtres (passé, futur connu, cible) pour un lot d'ancres, par indexation directe."""
    dev = anchors.device
    ip = anchors[:, None] + torch.arange(-lookback + 1, 1, device=dev)      # se termine à l'ancre
    fi = anchors[:, None] + torch.arange(1, h + 1, device=dev)
    return P[ip], C[fi], Y[fi]


def day_anchors(index: pd.DatetimeIndex, days: pd.DatetimeIndex, issue_hour: int | None) -> np.ndarray:
    """Position de l'ancre de chaque jour D : D-1 (issue_hour):00 (D-1 23:00 si issue_hour=None)."""
    off = 24 - check_issue_hour(issue_hour)
    return np.array([index.get_loc(d) - off for d in days])


# ----------------------------------------------------------------------------- entraînement
def _batches(anchors: torch.Tensor, bs: int, shuffle: bool):
    order = torch.randperm(len(anchors), device=anchors.device) if shuffle \
        else torch.arange(len(anchors), device=anchors.device)
    for i in range(0, len(anchors), bs):
        yield anchors[order[i:i + bs]]


def _val_loss(model, P, C, Y, anchors, cfg: TFTConfig, h: int) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for a in _batches(anchors, 512, False):
            past, fut, y = gather(P, C, Y, a, cfg.lookback, h)
            tot += float(quantile_loss(model(past, fut).float(), y, cfg.quantiles)) * len(a)
            n += len(a)
    return tot / max(n, 1)


def train_model(P, C, Y, tr: torch.Tensor, va: torch.Tensor, cfg: TFTConfig, h: int, seed: int,
                log=print) -> tuple[TemporalFusionTransformer, dict]:
    torch.manual_seed(seed)
    dev = P.device
    model = TemporalFusionTransformer(P.shape[1], C.shape[1], cfg.d_model, cfg.n_heads,
                                      cfg.lstm_layers, cfg.dropout, cfg.quantiles).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    use_amp = cfg.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best, best_state, best_ep, bad = float("inf"), None, 0, 0
    t0 = time.perf_counter()
    for ep in range(cfg.epochs):
        model.train()
        for a in _batches(tr, cfg.batch_size, True):
            past, fut, y = gather(P, C, Y, a, cfg.lookback, h)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                pred = model(past, fut)
            loss = quantile_loss(pred.float(), y, cfg.quantiles)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        vl = _val_loss(model, P, C, Y, va, cfg, h)
        sched.step(vl)
        log(f"      epoch {ep + 1:02d}  val pinball {vl:.4f}  ({time.perf_counter() - t0:.0f}s)")
        if vl < best - 1e-5:
            best, best_ep, bad = vl, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_val_pinball": best, "best_epoch": best_ep, "epochs_run": ep + 1,
                   "fit_seconds": time.perf_counter() - t0}


def predict_days(model, P, C, Y, anchors: np.ndarray, cfg: TFTConfig, h: int, scale: tuple[float, float]
                 ) -> np.ndarray:
    """(n_jours, 24, n_quantiles) en MW : les 24 derniers pas décodés = les 24 h du jour D."""
    model.eval()
    a_all = torch.as_tensor(anchors, device=P.device)
    outs = []
    with torch.no_grad():
        for a in _batches(a_all, 512, False):
            past, fut, _ = gather(P, C, Y, a, cfg.lookback, h)
            outs.append(model(past, fut).float()[:, -24:, :].cpu().numpy())
    mu, sd = scale
    return np.concatenate(outs) * sd + mu


# --------------------------------------------------------------------------------- orchestration
def run(cfg_tft: TFTConfig, weather_modes=("normal", "noisy", "perfect"), issue_hour: int | None = None,
        out_dir: str | Path = "tft_results", folds_subset: list[int] | None = None,
        resume: bool = False, quick: bool = False, device: str | None = None,
        project_cfg: dict | None = None, log=print) -> Path:
    """Exécute le backtest TFT et écrit predictions_tft.parquet + tft_fold_stats.parquet."""
    cfg = project_cfg or load_config()
    check_issue_hour(issue_hour)
    out = Path(out_dir)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"device : {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))

    df = load_hourly(cfg)
    load, ck = df["load_mw"], _cal_kwargs(cfg)
    cal_h = build_hourly_calendar(df.index, **ck)
    folds = splits.expanding_folds(cfg["cv"]["first_test_start"], str(df.index.max().date()),
                                   cfg["cv"]["fold_months"])
    if folds_subset:
        folds = [f for f in folds if f.fold_id in set(folds_subset)]
    if quick:
        folds, cfg_tft = folds[-1:], TFTConfig(**{**cfg_tft.__dict__, "epochs": 2, "seeds": 1})
    h = horizon(issue_hour)
    val_start = str(pd.Timestamp(cfg["split"]["train_end"]) + pd.Timedelta(days=1))
    test_start = str(pd.Timestamp(cfg["split"]["val_end"]) + pd.Timedelta(days=1))

    pred_path, stats_path = out / "predictions_tft.parquet", out / "tft_fold_stats.parquet"
    preds = [pd.read_parquet(pred_path)] if resume and pred_path.exists() else []
    stats = [pd.read_parquet(stats_path)] if resume and stats_path.exists() else []
    done = {(r.weather_mode, r.fold_id) for s in stats for r in s.itertuples()}

    for mode in weather_modes:
        tfc = make_temperature_forecast(df, mode, **cfg["models"].get("noisy_forecast", {})) \
            if mode == "noisy" else make_temperature_forecast(df, mode)
        X = build_features(df, cal_h, weather_mode=mode, temperature_forecast=tfc,
                           issue_hour=issue_hour)
        cov, scale_cols = build_covariates(X, mode)
        for f in folds:
            if (mode, f.fold_id) in done:
                log(f"== {mode} {f.label} : déjà fait, ignoré")
                continue
            log(f"== météo '{mode}'  {f.label}  (H={h}, {cfg_tft.seeds} graine(s))")
            test_pos = df.index.get_loc(f.test_start)
            Pn, Cn, Yn, scale = standardize(load, cov, scale_cols, fit_rows=test_pos)
            P, C, Y = (torch.as_tensor(a, device=dev) for a in (Pn, Cn, Yn))
            rg = fold_anchor_ranges(test_pos, cfg_tft.lookback, h, cfg_tft.val_days)
            tr = torch.arange(rg["train"][0], rg["train"][1] + 1, device=dev)
            va = torch.arange(rg["val"][0], rg["val"][1] + 1, device=dev)
            days = pd.date_range(f.test_start.normalize(), f.test_end.normalize(), freq="D")
            anchors = day_anchors(df.index, days, issue_hour)
            assert anchors.min() >= cfg_tft.lookback - 1
            splits.assert_train_before_test(df.index[[rg["train"][1] + h, rg["val"][1] + h]],
                                            df.index[[test_pos]])

            per_seed, secs, info = [], 0.0, {}
            for seed in range(cfg_tft.seeds):
                model, info = train_model(P, C, Y, tr, va, cfg_tft, h, seed, log)
                secs += info["fit_seconds"]
                per_seed.append(predict_days(model, P, C, Y, anchors, cfg_tft, h, scale))
                torch.save(model.state_dict(), out / "weights" / f"tft_{mode}_F{f.fold_id:02d}_s{seed}.pt")
            q = np.mean(per_seed, axis=0)                                  # (jours, 24, Q)
            idx = pd.DatetimeIndex(np.concatenate([pd.date_range(d, periods=24, freq="h") for d in days]))
            i50 = list(cfg_tft.quantiles).index(0.5)
            frame = pd.DataFrame({
                "timestamp": idx, "y_true": load.reindex(idx).to_numpy(),
                "period_type": cal_h["special_period_type"].reindex(idx).to_numpy(),
                "split": splits.split_of(idx, val_start, test_start).to_numpy(),
                "fold_id": f.fold_id, "y_pred": q[:, :, i50].reshape(-1),
                "model": MODEL_NAME, "weather_mode": mode, "method": MODEL_NAME,
                "q10": q[:, :, 0].reshape(-1), "q90": q[:, :, -1].reshape(-1)})
            frame["error"] = frame["y_pred"] - frame["y_true"]
            preds.append(frame)
            mae = float(frame["error"].abs().mean())
            stats.append(pd.DataFrame([{
                "model": MODEL_NAME, "weather_mode": mode, "fold_id": f.fold_id,
                "fold_label": f.label, "fit_seconds": secs, "n_train": len(tr), "seeds": cfg_tft.seeds,
                "best_epoch": info["best_epoch"], "best_val_pinball": info["best_val_pinball"],
                "fold_mae": mae, "issue_hour": -1 if issue_hour is None else issue_hour}]))
            pd.concat(preds).to_parquet(pred_path)                         # écrit après CHAQUE fold
            pd.concat(stats).to_parquet(stats_path)
            log(f"   fold MAE {mae:,.0f} MW  |  {secs:.0f}s  |  meilleure epoch {info['best_epoch']}")
    return out
