"""Dashboard Streamlit : comparer les modèles, choisir le meilleur, auditer données et erreurs.

Lancement :  streamlit run app/dashboard.py
Prérequis  :  scripts/01_prepare_data.py, 02_analyze_hypotheses.py, 03_run_benchmark.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from rte_forecast.business_rules.analogue import rank_analogues
from rte_forecast.calendar import build_hourly_calendar
from rte_forecast.config import load_config, resolve_path
from rte_forecast.evaluation.metrics import PERIOD_LABELS, PERIOD_ORDER
from rte_forecast.features import build_features, feature_columns
from rte_forecast.forecast import format_explanation

st.set_page_config(page_title="RTE load forecasting", page_icon="⚡", layout="wide")

CFG = load_config()
RES = Path(os.environ["RTE_RESULTS_DIR"]).resolve() if os.environ.get("RTE_RESULTS_DIR") \
    else resolve_path(CFG, "results_dir")   # ex. RTE_RESULTS_DIR=data/results_issue10h
PROC = resolve_path(CFG, "processed_dir")

# --------------------------------------------------------------------------- palette (validée)
# Ordre des slots de la palette catégorielle validée (light / dark). Une couleur = UN modèle,
# jamais réattribuée selon le rang ni selon le filtre.
CAT_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CAT_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
MODEL_SLOT = {"sarimax": 0, "xgboost": 1, "xgboost_rules": 2, "xgboost_rules_adaptive": 3,
              "random_forest": 4, "random_forest_rules": 5, "random_forest_rules_adaptive": 6,
              "rte_j1": 7}
DASH = {"naive_j7": "dot", "rte_j1": "dash", "xgboost_rules_adaptive": "solid"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_LIGHT = [[0, "#256abf"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#ef9a86"], [1, "#c23a3a"]]
DIV_DARK = [[0, "#3987e5"], [0.25, "#184f95"], [0.5, "#383835"], [0.75, "#b04a3a"], [1, "#e66767"]]


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return st.get_option("theme.base") == "dark"


DARK = is_dark()
INK = "#ffffff" if DARK else "#0b0b0b"
INK2 = "#c3c2b7" if DARK else "#52514e"
GRID = "#383835" if DARK else "#e6e5e1"
NEUTRAL = "#9a998f" if DARK else "#77766f"


def color(model: str) -> str:
    if model == "naive_j7":
        return NEUTRAL
    return (CAT_DARK if DARK else CAT_LIGHT)[MODEL_SLOT[model]]


def style(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, title=dict(text=title, font=dict(size=15, color=INK)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=12),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10), hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2)))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


# --------------------------------------------------------------------------- données
@st.cache_data(show_spinner=False)
def load_results():
    if not (RES / "predictions.parquet").exists():
        return None
    r = {n: pd.read_parquet(RES / f"{n}.parquet") for n in [
        "predictions", "metrics_split", "metrics_period", "cv_folds", "hyperparameters",
        "feature_importance", "shap_xgboost", "rule_explanations"]}
    r["meta"] = json.loads((RES / "run_meta.json").read_text())
    for n in ["analysis_summer_trough", "analysis_summer_shape", "analysis_year_end",
              "analysis_holidays", "analysis_imputation"]:
        p = RES / f"{n}.parquet"
        r[n] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return r


@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_parquet(PROC / "hourly.parquet")
    quality = json.loads((PROC / "data_quality.json").read_text())
    ck = {"summer_weeks_before": CFG["rules"]["summer_window_weeks_before"],
          "summer_weeks_after": CFG["rules"]["summer_window_weeks_after"]}
    cal = build_hourly_calendar(df.index, **ck)
    X = build_features(df, cal)
    return df, cal, X, quality


R = load_results()
if R is None:
    st.error("Aucun résultat : lancez scripts/03_run_benchmark.py d'abord.")
    st.stop()
DF, CAL, XF, QUALITY = load_dataset()
LABEL = R["meta"]["models"]
MODES = R["meta"]["weather_modes"]
SPLIT_LABEL = {"train_cv": "2022 (CV, entraînement partiel)", "validation": "2023 (validation)",
               "test": "2024 (test)", "all": "Tout (2022–2024)"}

# --------------------------------------------------------------------------- barre latérale
st.sidebar.title("⚡ Prévision de charge RTE")
mode = st.sidebar.radio(
    "Scénario météo", MODES, index=0,
    format_func=lambda m: {"normal": "normal — prévision T° = normale Enedis",
                           "noisy": "noisy — T° réalisée + bruit SYNTHÉTIQUE",
                           "perfect": "perfect — T° réalisée (ORACLE)"}[m])
st.sidebar.caption(
    "**normal** = mode opérationnel de référence (aucune vraie prévision météo disponible). "
    "**noisy** est synthétique. **perfect** est un plafond théorique, pas un résultat opérationnel.")
split = st.sidebar.selectbox("Période d'évaluation", ["test", "validation", "train_cv", "all"],
                             format_func=SPLIT_LABEL.get)
metric = st.sidebar.selectbox("Métrique de classement", ["mae", "rmse", "abs_bias", "mape"],
                              format_func={"mae": "MAE (MW)", "rmse": "RMSE (MW)",
                                           "abs_bias": "|Biais| (MW)", "mape": "MAPE (%)"}.get)


def in_view(d: pd.DataFrame) -> pd.DataFrame:
    return d[d["weather_mode"].isin([mode, "n/a"])]


PRED = in_view(R["predictions"])
if split != "all":
    PRED = PRED[PRED["split"] == split]
MS = in_view(R["metrics_split"])
MS = MS[MS["split"] == split] if split != "all" else in_view(R["metrics_period"])
MS = MS.assign(abs_bias=MS["bias"].abs())
ALL = MS[MS["period"] == "ALL"].set_index("model")
ranking = ALL.sort_values(metric)
BEST = ranking.index[0]
default_models = list(dict.fromkeys(["naive_j7", *ranking.index[:4]]))
models_sel = st.sidebar.multiselect("Modèles comparés (graphiques)", list(ranking.index),
                                    default=default_models, format_func=LABEL.get)
if not models_sel:
    st.warning("Sélectionnez au moins un modèle.")
    st.stop()

tabs = st.tabs(["🏆 Choix du modèle", "🗄️ Jeux de données", "🗓️ Heatmap des erreurs",
                "📊 Distribution des erreurs", "📈 Prévisions vs réel", "🔁 Walk-forward CV",
                "⚙️ Hyperparamètres & importance", "🧠 Règles métier"])


def unit(metric_name: str) -> str:
    return "%" if metric_name == "mape" else "MW"


# =========================================================================== 1. choix du modèle
with tabs[0]:
    st.subheader("Quel modèle retenir ?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Meilleur modèle", LABEL[BEST])
    c2.metric(f"{metric.upper()} ({SPLIT_LABEL[split]})", f"{ranking.loc[BEST, metric]:,.1f} {unit(metric)}")
    ref = ranking.loc["naive_j7", metric]
    c3.metric("Gain vs naïf J-7", f"{(1 - ranking.loc[BEST, metric] / ref) * 100:.0f} %")
    st.caption("Le classement change avec le scénario météo (barre latérale) : un gain obtenu en mode "
               "**perfect** n'est pas opérationnel. Comparez toujours à `normal` puis `noisy`.")

    fig = go.Figure(go.Bar(
        y=[LABEL[m] for m in ranking.index[::-1]], x=ranking[metric][::-1], orientation="h",
        marker=dict(color=[color(m) for m in ranking.index[::-1]], cornerradius=4),
        text=[f"{v:,.0f}" if metric != "mape" else f"{v:.2f}" for v in ranking[metric][::-1]],
        textposition="outside", cliponaxis=False, textfont=dict(color=INK),
        hovertemplate="%{y}<br>" + metric.upper() + " = %{x:,.1f}<extra></extra>"))
    fig.update_xaxes(title=f"{metric.upper()} ({unit(metric)})")
    st.plotly_chart(style(fig, 340), width="stretch")

    st.markdown("##### Erreur par type de période")
    piv = MS.pivot_table(index="period", columns="model", values=metric, observed=True)
    piv = piv.reindex([p for p in PERIOD_ORDER if p in piv.index])[list(ranking.index)]
    heat = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=[LABEL[m] for m in piv.columns], y=list(piv.index), colorscale=SEQ,
        text=np.round(piv.to_numpy(), 0), texttemplate="%{text:,.0f}", xgap=2, ygap=2,
        hovertemplate="%{y}<br>%{x}<br>" + metric.upper() + " = %{z:,.1f}<extra></extra>",
        colorbar=dict(title=unit(metric))))
    heat.update_yaxes(autorange="reversed")
    st.plotly_chart(style(heat, 380), width="stretch")
    best_by = piv.idxmin(axis=1).map(LABEL)
    st.markdown("**Meilleur modèle par type de période**")
    st.dataframe(pd.DataFrame({"Meilleur": best_by, f"{metric.upper()}": piv.min(axis=1).round(1),
                               "Naïf J-7": piv["naive_j7"].round(1)}), width="stretch")
    with st.expander("Tableau complet (vue table)"):
        st.dataframe(piv.rename(columns=LABEL).round(1), width="stretch")

    st.markdown("##### Précision vs temps d'entraînement")
    cv = in_view(R["cv_folds"])
    t = cv.groupby("model")["fit_seconds"].mean().dropna()
    sc = go.Figure()
    for m in t.index:
        sc.add_trace(go.Scatter(x=[t[m]], y=[ranking.loc[m, metric]], mode="markers+text",
                                text=[LABEL[m]], textposition="top center", name=LABEL[m],
                                marker=dict(size=13, color=color(m), line=dict(width=2, color="rgba(0,0,0,0)")),
                                textfont=dict(color=INK2), showlegend=False))
    sc.update_xaxes(title="Temps d'entraînement moyen par fold (s, log)", type="log")
    sc.update_yaxes(title=f"{metric.upper()} ({unit(metric)})")
    st.plotly_chart(style(sc, 380), width="stretch")

# =========================================================================== 2. données
with tabs[1]:
    st.subheader("Jeux de données")
    src = DF["temperature_source"].value_counts()
    st.warning(
        f"**Données de remplacement à noter** — la température Enedis n'existe que depuis "
        f"{QUALITY['temperature']['enedis_first'][:10]} (fenêtre glissante de 5 ans). "
        f"{src.get('openmeteo_calibrated', 0):,} h (2020-01-01 → 2021-09-18) proviennent d'un **proxy "
        f"ERA5/Open-Meteo recalé sur Enedis** (RMSE ≈ {QUALITY['temperature']['calibration']['rmse']:.2f} °C). "
        "Le scénario météo `noisy` est **synthétique**. La charge (eCO2mix) est réelle et complète.")

    splits = R["meta"]["split"]
    tl = DF["load_mw"].resample("D").mean()
    fig = go.Figure()
    bands = {"train": ("Train", "#6da7ec"), "validation": ("Validation", "#eda100"),
             "test": ("Test", "#1baf7a")}
    for name, (lab, col) in bands.items():
        a, b, n = splits[name]
        fig.add_vrect(x0=a, x1=b, fillcolor=col, opacity=0.13, line_width=0, layer="below",
                      annotation_text=f"{lab} ({n / 24:,.0f} j)", annotation_position="top left",
                      annotation_font_color=INK2)
    fig.add_trace(go.Scatter(x=tl.index, y=tl, mode="lines", line=dict(width=1.4, color=INK2),
                             name="Charge moyenne journalière",
                             hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} MW<extra></extra>"))
    fig.update_yaxes(title="MW")
    st.plotly_chart(style(fig, 360, "Découpage chronologique train / validation / test"),
                    width="stretch")

    a1, a2 = st.columns(2)
    comp = XF["special_period_type"].value_counts() / 24
    bar = go.Figure(go.Bar(x=comp.index, y=comp.values, marker=dict(color=SEQ[3], cornerradius=4),
                           text=[f"{v:,.0f}" for v in comp.values], textposition="outside",
                           textfont=dict(color=INK)))
    bar.update_yaxes(title="jours")
    a1.plotly_chart(style(bar, 320, "Jours par régime calendaire"), width="stretch")
    a1.caption("Le modèle « normal » n'apprend que **NORMAL** ; les autres régimes vont aux règles.")
    ts = DF["temperature_source"].value_counts()
    pie = go.Figure(go.Bar(x=ts.index, y=ts.values / 24, marker=dict(color=[SEQ[3], SEQ[1]][:len(ts)],
                                                                      cornerradius=4)))
    pie.update_yaxes(title="jours")
    a2.plotly_chart(style(pie, 320, "Origine de la température réalisée"), width="stretch")

    st.markdown("##### Explorer un jeu de données")
    which = st.selectbox("Table", [
        "Brut horaire (charge, RTE J-1, températures)", "Calendrier classifié",
        "Features — tous jours (modèles XGBoost / RF seuls)",
        "Jeu d'apprentissage du modèle normal (jours normaux uniquement)"])
    part = st.selectbox("Sous-ensemble", ["Tout", "train", "validation", "test"])
    d0, d1 = DF.index.min().date(), DF.index.max().date()
    rng = st.slider("Plage de dates", d0, d1, (d0, d1))
    if which.startswith("Brut"):
        table = DF
    elif which.startswith("Calendrier"):
        table = CAL
    elif which.startswith("Features"):
        table = XF[feature_columns("normal") + ["load_mw", "special_period_type"]]
    else:
        Xn = build_features(DF, CAL, clean_lags=True)
        table = Xn[Xn["special_period_type"] == "NORMAL"][feature_columns("normal") + ["load_mw"]]
    if part != "Tout":
        a, b, _ = splits[part]
        table = table[(table.index >= pd.Timestamp(a)) & (table.index <= pd.Timestamp(b))]
    table = table[(table.index.date >= rng[0]) & (table.index.date <= rng[1])]
    st.caption(f"{len(table):,} lignes × {table.shape[1]} colonnes")
    st.dataframe(table.head(5000), width="stretch", height=360)
    st.download_button("Télécharger en CSV", table.to_csv().encode("utf-8"), "dataset.csv", "text/csv")
    with st.expander("Rapport qualité des données"):
        st.json(QUALITY)

# =========================================================================== 3. heatmap
with tabs[2]:
    st.subheader("Erreurs — heatmap annuelle (x = jour du mois, y = mois)")
    years = sorted(PRED["timestamp"].dt.year.unique())
    h1, h2, h3 = st.columns(3)
    year = h1.selectbox("Année", years, index=len(years) - 1)
    kind = h2.selectbox("Grandeur", ["Biais moyen journalier (prév − réel)", "MAE journalière",
                                     "MAE journalière (% de la charge)"])
    hm_models = h3.multiselect("Modèles", list(ranking.index), default=models_sel[:4],
                               format_func=LABEL.get, key="hm_models")
    full = in_view(R["predictions"])
    full = full[full["timestamp"].dt.year == year]
    months = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

    def daily(m: str) -> pd.DataFrame:
        d = full[full["model"] == m].copy()
        d["date"] = d["timestamp"].dt.normalize()
        d["abs"] = d["error"].abs()
        d["pct"] = d["abs"] / d["y_true"] * 100
        g = d.groupby("date").agg(bias=("error", "mean"), mae=("abs", "mean"), pct=("pct", "mean"))
        return g

    mats = {}
    for m in hm_models:
        g = daily(m)
        key = "bias" if kind.startswith("Biais") else ("mae" if "%" not in kind else "pct")
        z = np.full((12, 31), np.nan)
        txt = [[""] * 31 for _ in range(12)]
        for dte, v in g[key].items():
            z[dte.month - 1, dte.day - 1] = v
            txt[dte.month - 1][dte.day - 1] = f"{dte:%Y-%m-%d}"
        mats[m] = (z, txt)
    if mats:
        zmax = max(np.nanpercentile(np.abs(z), 98) for z, _ in mats.values())
        diverging = kind.startswith("Biais")
        cols = st.columns(2 if len(mats) > 1 else 1)
        for i, (m, (z, txt)) in enumerate(mats.items()):
            fig = go.Figure(go.Heatmap(
                z=z, x=list(range(1, 32)), y=months, customdata=txt, xgap=2, ygap=2,
                colorscale=(DIV_DARK if DARK else DIV_LIGHT) if diverging else SEQ,
                zmin=-zmax if diverging else 0, zmax=zmax, zmid=0 if diverging else None,
                hovertemplate="%{customdata}<br>%{z:,.1f}" + (" %" if "%" in kind else " MW")
                + "<extra></extra>", colorbar=dict(title="%" if "%" in kind else "MW", thickness=10)))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="Jour du mois", dtick=2)
            cols[i % len(cols)].plotly_chart(style(fig, 330, LABEL[m]), width="stretch")
        st.caption("Échelle **commune** à tous les modèles. "
                   + ("Bleu = sous-estimation, rouge = surestimation, gris = neutre."
                      if diverging else "Plus foncé = erreur plus forte."))
        with st.expander("Vue table du modèle affiché en premier"):
            z0 = pd.DataFrame(mats[hm_models[0]][0], index=months, columns=range(1, 32)).round(0)
            st.dataframe(z0, width="stretch")

# =========================================================================== 4. distribution
with tabs[3]:
    st.subheader("Distribution des erreurs par modèle")
    d1c, d2c = st.columns(2)
    per = d1c.selectbox("Type de période", ["ALL", *PERIOD_LABELS.values()], key="dist_period")
    clip = d2c.slider("Rogner les queues (percentile)", 90, 100, 99)
    P = PRED[PRED["model"].isin(models_sel)].copy()
    if per != "ALL":
        inv = {v: k for k, v in PERIOD_LABELS.items()}
        P = P[P["period_type"] == inv[per]]
    lim = np.percentile(P["error"].abs(), clip)

    box = go.Figure()
    for m in models_sel:
        e = P.loc[P["model"] == m, "error"]
        box.add_trace(go.Box(x=e, name=LABEL[m], marker=dict(color=color(m)), boxmean=True,
                             line=dict(width=1.5), boxpoints=False, orientation="h"))
    box.add_vline(x=0, line_color=INK2, line_width=1)
    box.update_xaxes(title="Erreur signée prév − réel (MW)", range=[-lim, lim])
    box.update_layout(showlegend=False)
    st.plotly_chart(style(box, 120 + 60 * len(models_sel), "Boîtes (◆ = moyenne = biais)"),
                    width="stretch")

    e1, e2 = st.columns(2)
    hist = go.Figure()
    one = e1.selectbox("Histogramme du modèle", models_sel, format_func=LABEL.get)
    e = P.loc[P["model"] == one, "error"]
    hist.add_trace(go.Histogram(x=e, xbins=dict(size=lim / 30), marker=dict(color=color(one)),
                                hovertemplate="%{x:,.0f} MW : %{y} h<extra></extra>"))
    hist.add_vline(x=0, line_color=INK2, line_width=1)
    hist.update_xaxes(title="Erreur (MW)", range=[-lim, lim])
    hist.update_yaxes(title="heures")
    e1.plotly_chart(style(hist, 340), width="stretch")

    ecdf = go.Figure()
    for m in models_sel:
        a = np.sort(P.loc[P["model"] == m, "error"].abs().to_numpy())
        ecdf.add_trace(go.Scatter(x=a[:: max(1, len(a) // 400)], y=np.linspace(0, 100, len(a))[:: max(1, len(a) // 400)],
                                  mode="lines", name=LABEL[m], line=dict(color=color(m), width=2,
                                                                          dash=DASH.get(m, "solid"))))
    ecdf.update_xaxes(title="|Erreur| (MW)", range=[0, lim])
    ecdf.update_yaxes(title="% des heures avec |erreur| ≤ x")
    e2.plotly_chart(style(ecdf, 380, "Distribution cumulée de |erreur|"), width="stretch")

    q = P.groupby("model")["error"].describe(percentiles=[0.05, 0.5, 0.95, 0.99])
    q["MAE"] = P.assign(a=P["error"].abs()).groupby("model")["a"].mean()
    q = q[["count", "mean", "std", "5%", "50%", "95%", "99%", "MAE"]].round(0)
    st.dataframe(q.rename(index=LABEL, columns={"mean": "biais"}), width="stretch")

# =========================================================================== 5. prévisions
with tabs[4]:
    st.subheader("Prévisions vs réel")
    dmin, dmax = PRED["timestamp"].min().date(), PRED["timestamp"].max().date()
    start = st.date_input("Début", value=max(dmin, pd.Timestamp("2024-12-16").date())
                          if dmax >= pd.Timestamp("2024-12-16").date() else dmin,
                          min_value=dmin, max_value=dmax)
    span = st.slider("Durée (jours)", 1, 28, 10)
    w = PRED[(PRED["timestamp"] >= pd.Timestamp(start)) &
             (PRED["timestamp"] < pd.Timestamp(start) + pd.Timedelta(days=span))]
    w = w[w["model"].isin(models_sel)]
    act = w[w["model"] == models_sel[0]].set_index("timestamp")["y_true"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=act.index, y=act, name="Réel", line=dict(color=INK, width=2.4),
                             hovertemplate="%{x|%a %d %b %Hh}<br>réel %{y:,.0f} MW<extra></extra>"),
                  row=1, col=1)
    for m in models_sel:
        d = w[w["model"] == m]
        st_kw = dict(color=color(m), width=1.8, dash=DASH.get(m, "solid"))
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["y_pred"], name=LABEL[m], line=st_kw,
                                 legendgroup=m, hovertemplate=LABEL[m] + " %{y:,.0f} MW<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d["error"], name=LABEL[m], line=st_kw,
                                 legendgroup=m, showlegend=False,
                                 hovertemplate=LABEL[m] + " erreur %{y:,.0f} MW<extra></extra>"),
                      row=2, col=1)
    fig.add_hline(y=0, line_color=INK2, line_width=1, row=2, col=1)
    fig.update_yaxes(title="MW", row=1, col=1)
    fig.update_yaxes(title="Erreur (MW)", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig, 620), width="stretch")

    st.markdown("##### MAE par heure de la journée")
    H = PRED[PRED["model"].isin(models_sel)].copy()
    H["hour"] = H["timestamp"].dt.hour
    hh = H.assign(a=H["error"].abs()).groupby(["model", "hour"])["a"].mean().reset_index()
    fh = go.Figure()
    for m in models_sel:
        d = hh[hh["model"] == m]
        fh.add_trace(go.Scatter(x=d["hour"], y=d["a"], mode="lines+markers", name=LABEL[m],
                                line=dict(color=color(m), width=2, dash=DASH.get(m, "solid")),
                                marker=dict(size=7, color=color(m)),
                                hovertemplate=LABEL[m] + " %{x}h : %{y:,.0f} MW<extra></extra>"))
    fh.update_xaxes(title="Heure (locale)", dtick=2)
    fh.update_yaxes(title="MAE (MW)")
    st.plotly_chart(style(fh, 360), width="stretch")

# =========================================================================== 6. walk-forward
with tabs[5]:
    st.subheader("Validation croisée expanding-window (1 fold = 1 trimestre)")
    cvv = in_view(R["cv_folds"])
    cvv = cvv[cvv["model"].isin(models_sel)]
    fig = go.Figure()
    for m in models_sel:
        d = cvv[cvv["model"] == m].sort_values("fold_id")
        fig.add_trace(go.Scatter(x=d["fold_id"], y=d["mae"], mode="lines+markers", name=LABEL[m],
                                 line=dict(color=color(m), width=2, dash=DASH.get(m, "solid")),
                                 marker=dict(size=8, color=color(m)),
                                 hovertemplate=LABEL[m] + "<br>fold %{x}<br>MAE %{y:,.0f} MW<extra></extra>"))
    labels = R["meta"]["folds"]
    fig.update_xaxes(title="Fold (trimestre testé)", tickmode="array",
                     tickvals=list(range(1, len(labels) + 1)),
                     ticktext=[lbl.split(" ")[1] for lbl in labels])
    fig.update_yaxes(title="MAE (MW)")
    st.plotly_chart(style(fig, 380, "Performance de chaque fold"), width="stretch")

    tt = cvv.dropna(subset=["fit_seconds"])
    fig2 = go.Figure()
    for m in tt["model"].unique():
        d = tt[tt["model"] == m].sort_values("fold_id")
        fig2.add_trace(go.Bar(x=d["fold_id"], y=d["fit_seconds"], name=LABEL[m],
                              marker=dict(color=color(m), cornerradius=3)))
    fig2.update_layout(barmode="group", bargap=0.25)
    fig2.update_xaxes(title="Fold", dtick=1)
    fig2.update_yaxes(title="secondes (log)", type="log")
    st.plotly_chart(style(fig2, 340, "Temps d'entraînement par fold"), width="stretch")
    st.caption("Un modèle hybride « règles » s'entraîne comme son modèle normal ; les règles coûtent ~1 s.")
    st.dataframe(cvv.pivot_table(index="model", values=["mae", "rmse", "bias", "fit_seconds"],
                                 aggfunc=["mean", "std"]).rename(index=LABEL).round(1),
                 width="stretch")

# =========================================================================== 7. hyperparamètres
with tabs[6]:
    st.subheader("Impact des hyperparamètres (entraînement ≤ 2022 → validation 2023)")
    hp = R["hyperparameters"]
    for name, xk, yk, fk in [("xgboost", "hp_max_depth", "hp_learning_rate", "hp_n_estimators"),
                             ("random_forest", "hp_max_depth", "hp_min_samples_leaf", "hp_max_features")]:
        d = hp[hp["model"] == name]
        facets = sorted(d[fk].unique(), key=lambda s: float(s))
        fig = make_subplots(rows=1, cols=len(facets), subplot_titles=[f"{fk[3:]} = {f}" for f in facets],
                            shared_yaxes=True)
        zmin, zmax = d["val_mae"].min(), d["val_mae"].max()
        for i, f in enumerate(facets, 1):
            g = d[d[fk] == f].pivot_table(index=yk, columns=xk, values="val_mae")
            fig.add_trace(go.Heatmap(z=g.to_numpy(), x=list(g.columns), y=list(g.index), zmin=zmin,
                                     zmax=zmax, colorscale=SEQ[::-1], xgap=2, ygap=2,
                                     text=np.round(g.to_numpy(), 0), texttemplate="%{text:,.0f}",
                                     showscale=i == len(facets), colorbar=dict(title="MAE val", thickness=10),
                                     hovertemplate=f"{xk[3:]} %{{x}}<br>{yk[3:]} %{{y}}<br>MAE %{{z:,.0f}}<extra></extra>"),
                          row=1, col=i)
            fig.update_xaxes(title=xk[3:], type="category", row=1, col=i)
        fig.update_yaxes(title=yk[3:], type="category", col=1)
        st.plotly_chart(style(fig, 330, f"{LABEL[name]} — MAE de validation (plus clair = meilleur)"),
                        width="stretch")
    fig = go.Figure()
    for name in ["xgboost", "random_forest"]:
        d = hp[hp["model"] == name]
        fig.add_trace(go.Scatter(x=d["fit_seconds"], y=d["val_mae"], mode="markers", name=LABEL[name],
                                 marker=dict(size=10, color=color(name)),
                                 hovertemplate="fit %{x:.1f}s<br>MAE %{y:,.0f}<extra></extra>"))
    fig.update_xaxes(title="Temps d'entraînement (s, log)", type="log")
    fig.update_yaxes(title="MAE validation (MW)")
    st.plotly_chart(style(fig, 320, "Coût vs précision des configurations"), width="stretch")
    with st.expander("Table complète des essais"):
        st.dataframe(hp.sort_values("val_mae"), width="stretch")

    st.subheader("Importance des variables")
    imp = R["feature_importance"]
    ic = st.columns(2)
    for i, name in enumerate(["xgboost", "random_forest"]):
        d = imp[imp["model"] == name].sort_values("importance").tail(15)
        f = go.Figure(go.Bar(y=d["feature"], x=d["importance"], orientation="h",
                             marker=dict(color=color(name), cornerradius=3)))
        ic[i].plotly_chart(style(f, 420, LABEL[name]), width="stretch")

    st.subheader("SHAP (XGBoost, échantillon 2024)")
    S = R["shap_xgboost"]
    cols = feature_columns("normal")
    order = sorted(cols, key=lambda c: -S[f"shap_{c}"].abs().mean())[:12]
    rng_ = np.random.default_rng(0)
    fig = go.Figure()
    for j, c in enumerate(order[::-1]):
        v = S[c].to_numpy(dtype=float)
        norm = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-9)
        fig.add_trace(go.Scatter(
            x=S[f"shap_{c}"], y=j + rng_.uniform(-0.3, 0.3, len(S)), mode="markers", showlegend=False,
            marker=dict(size=4, color=norm, colorscale=SEQ, opacity=0.7),
            customdata=np.c_[v], hovertemplate=c + " = %{customdata[0]:,.1f}<br>SHAP %{x:,.0f} MW<extra></extra>"))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))), ticktext=order[::-1])
    fig.update_xaxes(title="Impact sur la prévision (MW) — plus foncé = valeur de la variable plus élevée")
    st.plotly_chart(style(fig, 460, "SHAP summary"), width="stretch")
    fig = go.Figure(go.Scatter(x=S["temperature_forecast"], y=S["shap_temperature_forecast"], mode="markers",
                               marker=dict(size=5, color=S["hour"], colorscale=SEQ, showscale=True,
                                           colorbar=dict(title="heure", thickness=10)),
                               hovertemplate="T %{x:.1f} °C<br>SHAP %{y:,.0f} MW<extra></extra>"))
    fig.update_xaxes(title="Température prévue (°C)")
    fig.update_yaxes(title="SHAP (MW)")
    st.plotly_chart(style(fig, 380, "SHAP dependence — température"), width="stretch")

# =========================================================================== 8. règles métier
with tabs[7]:
    st.subheader("Règles métier : hypothèses vérifiées et explicabilité")
    st.markdown("##### H1 — le creux estival est-il la semaine dont le week-end est le plus proche du 15 août ?")
    tr = R["analysis_summer_trough"]
    if len(tr):
        st.dataframe(tr, width="stretch", hide_index=True)
        st.caption("Les exceptions sont **signalées, pas masquées** : 2021 et 2022 à ±1 semaine, 2020 non "
                   "vérifiée sur juin–sept. (creux au 1er juin, post-confinement) mais vérifiée sur juil–sept.")
    sh = R["analysis_summer_shape"]
    if len(sh):
        fig = go.Figure()
        for i, (y, d) in enumerate(sh.groupby("year")):
            fig.add_trace(go.Scatter(x=d["offset_weeks"], y=d["ratio"], mode="lines+markers", name=str(y),
                                     line=dict(color=(CAT_DARK if DARK else CAT_LIGHT)[i], width=2),
                                     marker=dict(size=7)))
        fig.add_hline(y=1, line_color=INK2, line_width=1)
        fig.update_xaxes(title="Semaines relatives à la semaine du creux", dtick=1)
        fig.update_yaxes(title="Niveau / niveau d'avant-vacances")
        st.plotly_chart(style(fig, 340, "Forme de la rampe estivale : un creux net sur ~3 semaines, pas de rampe progressive"),
                        width="stretch")
    ye = R["analysis_year_end"]
    if len(ye):
        st.markdown("##### Fin / début d'année : niveau des semaines ÷ baseline S48–S50")
        st.dataframe(ye.round(3), width="stretch", hide_index=True)
    ho = R["analysis_holidays"]
    if len(ho):
        st.markdown("##### Fériés et ponts : charge du jour ÷ même jour de semaine ordinaire voisin")
        fig = go.Figure()
        for i, (cls, d) in enumerate(ho.groupby("weekday_class")):
            fig.add_trace(go.Box(y=d["ratio"], name=cls, boxpoints="all", jitter=0.4, pointpos=0,
                                 marker=dict(color=(CAT_DARK if DARK else CAT_LIGHT)[i]),
                                 text=d["name"] + " " + d["date"].dt.strftime("%Y-%m-%d"),
                                 hovertemplate="%{text}<br>%{y:.3f}<extra></extra>"))
        fig.add_hline(y=1, line_color=INK2, line_width=1)
        fig.update_yaxes(title="ratio")
        st.plotly_chart(style(fig, 360), width="stretch")
    imp_ = R["analysis_imputation"]
    if len(imp_):
        st.markdown("##### Rebouchage causal : erreur sur trous injectés dans les vraies données")
        st.dataframe(imp_.round(2), width="stretch", hide_index=True)

    st.markdown("##### Explication d'une prévision produite par une règle")
    ex = R["rule_explanations"]
    ex = ex[ex["weather_mode"] == mode]
    hy = st.selectbox("Modèle hybride", sorted(ex["model"].unique()), format_func=LABEL.get)
    ex = ex[ex["model"] == hy]
    days = sorted(ex["timestamp"].dt.normalize().unique())
    day = st.selectbox("Jour", days, index=min(len(days) - 1, days.index(pd.Timestamp("2024-12-25"))
                                               if pd.Timestamp("2024-12-25") in days else 0),
                       format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d (%a)"))
    d = ex[ex["timestamp"].dt.normalize() == day].sort_values("timestamp")
    rec = d.iloc[12].copy() if len(d) > 12 else d.iloc[0].copy()
    c1, c2 = st.columns([1, 2])
    c1.code(format_explanation(rec["timestamp"], rec), language="text")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["y_true"], name="Réel", line=dict(color=INK, width=2.4)))
    fig.add_trace(go.Scatter(x=d["timestamp"], y=d["yhat"], name=LABEL[hy],
                             line=dict(color=color(hy), width=2)))
    fig.update_yaxes(title="MW")
    c2.plotly_chart(style(fig, 360), width="stretch")
    yr = pd.Timestamp(day).year
    st.markdown(f"**Classement des années analogues pour Noël {yr}** (score de similarité calendaire)")
    st.dataframe(rank_analogues(yr, range(2020, yr), "christmas"), width="stretch", hide_index=True)
