"""
Yield Curve Builder & Recesija Predictor — Streamlit verzija
============================================================
Pokretanje:
    pip install streamlit requests pandas numpy scipy plotly
    streamlit run streamlit_app.py
"""

import os
import warnings
from datetime import datetime, timedelta

import streamlit as st
import requests
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# STRANICA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Yield Curve & Recesija Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# KONSTANTE
# ─────────────────────────────────────────────────────────────────────────────

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "SAD": {
        0.25: "DGS3MO",
        0.50: "DGS6MO",
        1:    "DGS1",
        2:    "DGS2",
        3:    "DGS3",
        5:    "DGS5",
        7:    "DGS7",
        10:   "DGS10",
        20:   "DGS20",
        30:   "DGS30",
    },
    "EU": {
        0.25: "ECBDFR",
        10:   "IRLTLT01EZM156N",
    },
    "UK": {
        0.25: "IUDSOIA",
        10:   "IRLTLT01GBM156N",
    },
}

SPREAD_SERIES     = {"SAD": "T10Y2Y", "EU": None, "UK": None}
RECESSION_SERIES  = {"SAD": "USRECM", "EU": None, "UK": None}

RECESSION_PERIODS = {
    "SAD": [("1990-07-01","1991-03-01"),("2001-03-01","2001-11-01"),
            ("2007-12-01","2009-06-01"),("2020-02-01","2020-04-01")],
    "EU":  [("2008-01-01","2009-07-01"),("2011-07-01","2013-01-01"),
            ("2020-02-01","2020-06-01")],
    "UK":  [("1990-07-01","1991-04-01"),("2008-04-01","2009-07-01"),
            ("2020-01-01","2020-07-01")],
}

NY_FED_INTERCEPT = -0.6047
NY_FED_SLOPE     = -0.7798


# ─────────────────────────────────────────────────────────────────────────────
# MODELI
# ─────────────────────────────────────────────────────────────────────────────

def nelson_siegel_yield(tau, b0, b1, b2, lam):
    tau = np.where(np.asarray(tau, dtype=float) <= 0, 1e-9, np.asarray(tau, dtype=float))
    x   = tau / lam
    f1  = (1 - np.exp(-x)) / x
    f2  = f1 - np.exp(-x)
    return b0 + b1 * f1 + b2 * f2


def fituj_nelson_siegel(maturities, yields):
    def objective(params):
        b0, b1, b2, lam = params
        if lam <= 0 or b0 <= 0:
            return 1e10
        return np.sum((nelson_siegel_yield(maturities, b0, b1, b2, lam) - yields) ** 2)

    best_res, best_val = None, np.inf
    for lam_init in [0.5, 1.0, 2.0, 3.0]:
        for b1_init in [-2, 0, 2]:
            x0 = [yields.mean(), b1_init, 0, lam_init]
            bounds = [(0.01, 20), (-10, 10), (-10, 10), (0.1, 10)]
            try:
                res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 2000, "ftol": 1e-12})
                if res.fun < best_val:
                    best_val, best_res = res.fun, res
            except Exception:
                pass

    if best_res is None:
        return (yields.mean(), 0.0, 0.0, 1.5), np.nan
    fitted = nelson_siegel_yield(maturities, *best_res.x)
    rmse   = np.sqrt(np.mean((fitted - yields) ** 2))
    return tuple(best_res.x), rmse


def logisticki_model(spread_pct):
    return float(expit(NY_FED_INTERCEPT + NY_FED_SLOPE * spread_pct))


# ─────────────────────────────────────────────────────────────────────────────
# FRED API
# ─────────────────────────────────────────────────────────────────────────────

def _fred_get(series_id, start, end, api_key):
    params = {
        "series_id": series_id,
        "observation_start": start,
        "observation_end": end,
        "api_key": api_key,
        "file_type": "json",
        "frequency": "m",
        "aggregation_method": "avg",
    }
    resp = requests.get(FRED_BASE_URL, params=params, timeout=20)
    resp.raise_for_status()
    records = {}
    for obs in resp.json().get("observations", []):
        if obs["value"] != ".":
            records[pd.to_datetime(obs["date"])] = float(obs["value"])
    return pd.Series(records).sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def povuci_podatke(zemlja, start, end, api_key):
    frames, seen = {}, set()
    for rocnost, sid in FRED_SERIES.get(zemlja, {}).items():
        if sid in seen:
            continue
        seen.add(sid)
        try:
            s = _fred_get(sid, start, end, api_key)
            if len(s) > 0:
                frames[rocnost] = s
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index().dropna(how="all")
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def povuci_spread(zemlja, start, end, api_key):
    sid = SPREAD_SERIES.get(zemlja)
    if not sid:
        return pd.Series(dtype=float)
    try:
        return _fred_get(sid, start, end, api_key)
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def povuci_recesije_series(zemlja, start, end, api_key):
    sid = RECESSION_SERIES.get(zemlja)
    if not sid:
        return pd.Series(dtype=float)
    try:
        return _fred_get(sid, start, end, api_key)
    except Exception:
        return pd.Series(dtype=float)


def izracunaj_spread_iz_df(df):
    spreads = {}
    for date, row in df.iterrows():
        valid = row.dropna()
        if len(valid) < 2:
            continue
        mats = np.array(valid.index.tolist(), dtype=float)
        ylds = valid.values.astype(float)
        if 10 in valid.index and 2 in valid.index:
            spreads[date] = valid[10] - valid[2]
        elif len(valid) >= 3:
            try:
                params, _ = fituj_nelson_siegel(mats, ylds)
                spreads[date] = (nelson_siegel_yield([10.0], *params)[0]
                                 - nelson_siegel_yield([2.0],  *params)[0])
            except Exception:
                pass
    return pd.Series(spreads, name="spread").sort_index()


def identifikuj_inverziju(spread, min_meseci=2):
    periods, in_p, start, count = [], False, None, 0
    for date, val in (spread < 0).astype(int).items():
        if val == 1:
            if not in_p:
                in_p, start, count = True, date, 1
            else:
                count += 1
        elif in_p:
            if count >= min_meseci:
                periods.append((start, date))
            in_p, count = False, 0
    if in_p and count >= min_meseci:
        periods.append((start, spread.index[-1]))
    return periods


def parse_period(s):
    s = s.strip()
    if s.isdigit() and len(s) == 4:
        return f"{s}-01-01", f"{s}-12-31"
    if len(s) == 9 and s[4] == "-" and s[:4].isdigit() and s[5:].isdigit():
        return f"{s[:4]}-01-01", f"{s[5:]}-12-31"
    if len(s) == 10:
        try:
            d = datetime.strptime(s, "%Y-%m-%d")
            return ((d - timedelta(days=180)).strftime("%Y-%m-%d"),
                    (d + timedelta(days=180)).strftime("%Y-%m-%d"))
        except ValueError:
            pass
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    return start, end


# ─────────────────────────────────────────────────────────────────────────────
# GRAFOVI
# ─────────────────────────────────────────────────────────────────────────────

def napravi_grafove(df_raw, spread_series, recession_periods, inverzija_periods):
    tau_fine  = np.linspace(0.25, 30, 200)
    dates_all = df_raw.dropna(thresh=3).index

    if len(dates_all) > 12:
        idx       = np.linspace(0, len(dates_all) - 1, 12, dtype=int)
        dates_sel = dates_all[idx]
    else:
        dates_sel = dates_all

    def get_color(i, n):
        t = i / max(n - 1, 1)
        return f"rgb({int(t*0)},{int(26 + t*186)},{int(51 + t*204)})"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["📈 Krive Prinosa (Nelson-Siegel)", "📉 Spread 10Y-2Y",
                        "⚠️ Verovatnoća Recesije (NY Fed)", "🌡️ Yield Heatmap"],
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    # subplot 1 — NS krive
    for i, date in enumerate(dates_sel):
        row_data = df_raw.loc[date].dropna()
        if len(row_data) < 3:
            continue
        mats   = np.array(row_data.index.tolist(), dtype=float)
        yields = row_data.values.astype(float)
        try:
            params, _ = fituj_nelson_siegel(mats, yields)
            y_smooth  = nelson_siegel_yield(tau_fine, *params)
            label     = date.strftime("%Y-%m")
            color     = get_color(i, len(dates_sel))
            fig.add_trace(go.Scatter(
                x=tau_fine, y=y_smooth, mode="lines", name=label,
                line=dict(color=color, width=1.5),
                opacity=0.4 + 0.6 * (i / max(len(dates_sel) - 1, 1)),
                hovertemplate=f"<b>{label}</b><br>Ročnost: %{{x:.1f}}Y<br>Prinos: %{{y:.2f}}%<extra></extra>",
                showlegend=(i == 0 or i == len(dates_sel) - 1),
            ), row=1, col=1)
            if i == len(dates_sel) - 1:
                fig.add_trace(go.Scatter(
                    x=mats, y=yields, mode="markers", name=f"Tržišne cene ({label})",
                    marker=dict(color="#ffd93d", size=8, line=dict(color="white", width=1)),
                    showlegend=True,
                ), row=1, col=1)
        except Exception:
            continue

    # subplot 2 — spread
    if len(spread_series) > 0:
        fig.add_trace(go.Scatter(
            x=spread_series.index, y=spread_series.values, mode="lines",
            name="Spread 10Y-2Y", line=dict(color="#00d4ff", width=2),
            hovertemplate="<b>%{x|%Y-%m}</b><br>Spread: %{y:.2f}%<extra></extra>",
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=spread_series.index.tolist() + spread_series.index.tolist()[::-1],
            y=spread_series.clip(upper=0).tolist() + [0] * len(spread_series),
            fill="toself", fillcolor="rgba(255,107,107,0.2)",
            line=dict(width=0), name="Inverzija zona", hoverinfo="skip",
        ), row=1, col=2)

        fig.add_hline(y=0, line_dash="dash", line_color="#ff6b6b", line_width=1.5, row=1, col=2)
        avg = spread_series.mean()
        fig.add_hline(y=avg, line_dash="dot", line_color="#ffd93d", line_width=1,
                      annotation_text=f"Prosek: {avg:.2f}%",
                      annotation_font_color="#ffd93d", row=1, col=2)

        for inv_s, inv_e in inverzija_periods:
            fig.add_vrect(x0=inv_s, x1=inv_e, fillcolor="rgba(255,107,107,0.15)",
                          line_width=0, annotation_text="INV",
                          annotation_position="top left",
                          annotation_font_color="#ff6b6b", row=1, col=2)

    # subplot 3 — verovatnoća
    if len(spread_series) > 0:
        prob_s = spread_series.apply(logisticki_model) * 100
        fig.add_trace(go.Scatter(
            x=prob_s.index, y=prob_s.values, mode="lines",
            name="P(Recesija 12M)", line=dict(color="#ffd93d", width=2.5),
            fill="tozeroy", fillcolor="rgba(255,217,61,0.1)",
            hovertemplate="<b>%{x|%Y-%m}</b><br>P: %{y:.1f}%<extra></extra>",
        ), row=2, col=1)

        for thresh, color, label in [(20,"rgba(100,200,100,0.7)","20%"),
                                      (40,"rgba(255,200,50,0.7)","40%"),
                                      (70,"rgba(255,100,100,0.7)","70%")]:
            fig.add_hline(y=thresh, line_dash="dot", line_color=color, line_width=1,
                          annotation_text=label, annotation_font_color=color,
                          annotation_font_size=9, row=2, col=1)

        for rec_s, rec_e in recession_periods:
            fig.add_vrect(x0=rec_s, x1=rec_e, fillcolor="rgba(255,107,107,0.15)",
                          line_width=0, row=2, col=1)

    # subplot 4 — heatmap
    if len(df_raw) > 0:
        df_m = df_raw.resample("ME").mean().dropna(thresh=2)
        if len(df_m) > 0:
            cols = sorted(df_m.columns)
            z    = df_m[cols].values.T
            xl   = [d.strftime("%Y-%m") for d in df_m.index]
            yl   = [f"{c}Y" if c >= 1 else f"{int(c*12)}M" for c in cols]
            fig.add_trace(go.Heatmap(
                z=z, x=xl, y=yl,
                colorscale=[[0,"#001a33"],[0.5,"#006699"],[1,"#00d4ff"]],
                colorbar=dict(title="Prinos %", len=0.4, y=0.22,
                              titlefont=dict(color="#e2e8f0"),
                              tickfont=dict(color="#e2e8f0")),
                hovertemplate="Datum: %{x}<br>Ročnost: %{y}<br>Prinos: %{z:.2f}%<extra></extra>",
            ), row=2, col=2)

    fig.update_layout(
        height=700,
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0", size=11),
        legend=dict(bgcolor="rgba(17,24,39,0.8)", bordercolor="#1e293b", borderwidth=1),
        margin=dict(t=60, b=40, l=40, r=40),
        hovermode="x unified",
    )
    for ann in fig.layout.annotations:
        ann.font.color = "#e2e8f0"
        ann.font.size  = 12

    for row, col in [(1,1),(1,2),(2,1),(2,2)]:
        fig.update_xaxes(gridcolor="#1e293b", zeroline=False, row=row, col=col)
        fig.update_yaxes(gridcolor="#1e293b", zeroline=False, row=row, col=col)

    fig.update_yaxes(title_text="Prinos (%)", row=1, col=1)
    fig.update_xaxes(title_text="Ročnost (god.)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (%)", row=1, col=2)
    fig.update_yaxes(title_text="Verovatnoća (%)", range=[0, 100], row=2, col=1)
    fig.update_xaxes(tickangle=-45, row=2, col=2)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.metric-card .label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.metric-card .value { font-size: 26px; font-weight: 600; font-family: monospace; }
.signal-box { border-radius: 8px; padding: 12px 16px; margin: 10px 0; font-size: 14px; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parametri")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 FRED API Ključ",
        value=os.environ.get("FRED_API_KEY", ""),
        type="password",
        help="Besplatan ključ: https://fred.stlouisfed.org/docs/api/api_key.html",
    )

    zemlja = st.selectbox("🌍 Zemlja", ["SAD", "EU", "UK"])

    datum_ili_period = st.text_input(
        "📅 Period",
        value="2019-2024",
        help="Primeri: '2023', '2019-2024', '2007-2010'",
    )

    st.markdown("---")
    st.markdown("**Primeri perioda:**")
    st.code("2023\n2019-2024\n2007-2010\n2020-2023")

    st.markdown("---")
    analiziraj_btn = st.button("🔍 Pokreni analizu", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
**O modelu:**
- Nelson-Siegel fitovanje krive
- Spread = 10Y prinos − 2Y prinos
- NY Fed logistički model:
  `P = σ(−0.605 − 0.780·spread)`
""")

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.title("📈 Yield Curve Builder & Recesija Predictor")
st.caption("Seminarski rad | Finansijska analiza & Ekonometrika")

if not api_key:
    st.info("👈 Unesite FRED API ključ u bočnoj traci. "
            "Besplatan ključ na: https://fred.stlouisfed.org/docs/api/api_key.html")
    st.stop()

if not analiziraj_btn and "last_result" not in st.session_state:
    st.info("👈 Podesite parametre i kliknite **Pokreni analizu**.")
    st.stop()

if analiziraj_btn or "last_result" not in st.session_state:
    start, end   = parse_period(datum_ili_period)
    start_hist   = (pd.Timestamp(start) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

    with st.spinner("Preuzimanje podataka sa FRED API-ja..."):
        df_raw = povuci_podatke(zemlja, start_hist, end, api_key)

    if df_raw.empty:
        st.error("❌ Nisu preuzeti podaci. Proverite API ključ i pokušajte ponovo.")
        st.stop()

    with st.spinner("Izračunavanje spreada i fitovanje NS modela..."):
        spread_direct = povuci_spread(zemlja, start_hist, end, api_key)
        spread_series = spread_direct if len(spread_direct) > 10 else izracunaj_spread_iz_df(df_raw)

        rec_raw = povuci_recesije_series(zemlja, start_hist, end, api_key)
        if len(rec_raw) > 0:
            in_rec, rec_periods, r_start = False, [], None
            for d, v in rec_raw.items():
                if v == 1 and not in_rec:
                    in_rec, r_start = True, d
                elif v == 0 and in_rec:
                    in_rec = False
                    rec_periods.append((r_start, d))
            if in_rec:
                rec_periods.append((r_start, rec_raw.index[-1]))
        else:
            rec_periods = [
                (pd.Timestamp(s), pd.Timestamp(e))
                for s, e in RECESSION_PERIODS.get(zemlja, [])
                if pd.Timestamp(e) >= pd.Timestamp(start_hist) and pd.Timestamp(s) <= pd.Timestamp(end)
            ]

        inverzija = identifikuj_inverziju(spread_series)
        df_disp   = df_raw[(df_raw.index >= start) & (df_raw.index <= end)]
        spread_disp = spread_series[(spread_series.index >= start) & (spread_series.index <= end)]

    st.session_state.last_result = dict(
        df_disp=df_disp, df_raw=df_raw,
        spread_disp=spread_disp, rec_periods=rec_periods,
        inverzija=inverzija, zemlja=zemlja, datum_ili_period=datum_ili_period,
    )

# Učitavamo rezultate
r            = st.session_state.last_result
df_disp      = r["df_disp"]
spread_disp  = r["spread_disp"]
rec_periods  = r["rec_periods"]
inverzija    = r["inverzija"]

# ── METRIKE ───────────────────────────────────────────────────────────────────
cur_spread = float(spread_disp.iloc[-1]) if len(spread_disp) > 0 else float("nan")
avg_spread = float(spread_disp.mean())   if len(spread_disp) > 0 else float("nan")
prob       = logisticki_model(cur_spread) * 100 if not pd.isna(cur_spread) else float("nan")
is_inv     = cur_spread < 0 if not pd.isna(cur_spread) else False

col1, col2, col3, col4 = st.columns(4)

with col1:
    color = "#ff6b6b" if is_inv else "#6bcb77"
    st.markdown(f'<div class="metric-card"><div class="label">Trenutni spread</div>'
                f'<div class="value" style="color:{color}">{cur_spread:+.2f}%</div></div>',
                unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="metric-card"><div class="label">Istorijski prosek</div>'
                f'<div class="value" style="color:#94a3b8">{avg_spread:+.2f}%</div></div>',
                unsafe_allow_html=True)

with col3:
    pc = "#ff6b6b" if prob >= 60 else "#ffd93d" if prob >= 35 else "#6bcb77"
    st.markdown(f'<div class="metric-card"><div class="label">P(Recesija 12M)</div>'
                f'<div class="value" style="color:{pc}">{prob:.1f}%</div></div>',
                unsafe_allow_html=True)

with col4:
    badge = "🔴 Invertovana" if is_inv else "🟢 Normalna"
    bc    = "#ff6b6b" if is_inv else "#6bcb77"
    st.markdown(f'<div class="metric-card"><div class="label">Status krive</div>'
                f'<div class="value" style="color:{bc}; font-size:18px">{badge}</div></div>',
                unsafe_allow_html=True)

# Signal box
if not pd.isna(prob):
    if prob < 25:
        bg, border, tekst = "#052e16","#166534","🟢 <b>Nizak rizik</b> — Kriva prinosa je normalna."
    elif prob < 45:
        bg, border, tekst = "#422006","#92400e","🟡 <b>Umereni rizik</b> — Kriva se spljošćuje, pratiti razvoj."
    elif prob < 65:
        bg, border, tekst = "#450a0a","#991b1b","🟠 <b>Povišen rizik</b> — Kriva je invertovana! Istorijski prethodi recesiji."
    else:
        bg, border, tekst = "#3b0000","#7f1d1d","🔴 <b>Visok rizik</b> — Duboka inverzija. NY Fed: recesija verovatna."
    st.markdown(
        f'<div class="signal-box" style="background:{bg};border:1px solid {border}">{tekst}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── GRAFOVI ───────────────────────────────────────────────────────────────────
with st.spinner("Generisanje grafova..."):
    fig = napravi_grafove(df_disp, spread_disp, rec_periods, inverzija)

st.plotly_chart(fig, use_container_width=True)

# ── DETALJI U EXPANDER-IMA ────────────────────────────────────────────────────
with st.expander("📐 Nelson-Siegel parametri (poslednji datum)"):
    latest = df_disp.dropna(thresh=3)
    if len(latest) > 0:
        row = latest.iloc[-1].dropna()
        if len(row) >= 3:
            mats = np.array(row.index.tolist(), dtype=float)
            ylds = row.values.astype(float)
            try:
                params, rmse = fituj_nelson_siegel(mats, ylds)
                b0, b1, b2, lam = params
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("β₀ — Dugoročni nivo",         f"{b0:.4f}%")
                    st.metric("β₁ — Kratkoročna komponenta", f"{b1:.4f}%")
                with c2:
                    st.metric("β₂ — Hump",               f"{b2:.4f}%")
                    st.metric("λ  — Parametar oblika",    f"{lam:.4f}")
                st.info(f"RMSE fitovanja: {rmse:.4f}%")
            except Exception as e:
                st.warning(f"NS fit nije uspeo: {e}")

with st.expander("📋 Periodi inverzije"):
    if inverzija:
        rows = [{"Početak": pd.Timestamp(s).strftime("%Y-%m"),
                 "Kraj": pd.Timestamp(e).strftime("%Y-%m"),
                 "Trajanje (meseci)": (pd.Timestamp(e) - pd.Timestamp(s)).days // 30}
                for s, e in inverzija]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.info("Nema identifikovanih perioda inverzije u analiziranom periodu.")

with st.expander("🔢 Logistički model — referentne vrednosti"):
    ref_spreads = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    st.dataframe(pd.DataFrame({
        "Spread (%)":        [f"{s:+.1f}" for s in ref_spreads],
        "P(Recesija 12M)": [f"{logisticki_model(s)*100:.1f}%" for s in ref_spreads],
    }), hide_index=True, use_container_width=True)
    st.caption(f"Formula: P = 1 / (1 + exp(-({NY_FED_INTERCEPT} + {NY_FED_SLOPE}·spread)))")

with st.expander("📊 Sirovi podaci"):
    st.dataframe(df_disp.style.format("{:.3f}"), use_container_width=True)
    if len(spread_disp) > 0:
        st.markdown("**Spread 10Y-2Y:**")
        st.dataframe(spread_disp.rename("Spread (%)").to_frame().style.format("{:.3f}"),
                     use_container_width=True)

st.markdown("---")
st.caption("Model baziran na NY Fed metodologiji | FRED podaci | Seminarski rad — Finansijska analiza")
