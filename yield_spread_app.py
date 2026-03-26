import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analiza Krive Prinosa",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark financial theme */
.main { background-color: #0d1117; }
[data-testid="stAppViewContainer"] { background-color: #0d1117; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

/* Headers */
h1, h2, h3 { color: #e6edf3 !important; }
p, label, .stMarkdown { color: #8b949e !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #58a6ff; }
[data-testid="stMetricValue"] { color: #58a6ff !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; }

/* Divider */
hr { border-color: #30363d !important; }

/* Info boxes */
.info-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #c9d1d9 !important;
    font-size: 0.88rem;
    line-height: 1.6;
}
.info-box strong { color: #58a6ff; }

/* Section title */
.section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #58a6ff !important;
    margin-bottom: 6px;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 8px;
    padding: 8px;
}

/* Buttons */
.stButton > button {
    background: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #30363d;
    border-color: #58a6ff;
    color: #58a6ff;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #c9d1d9 !important;
}

/* Plot background override */
.js-plotly-plot { border-radius: 8px; overflow: hidden; }

/* Recession band label */
.rec-label {
    font-size: 0.75rem;
    color: #f85149;
    font-family: 'IBM Plex Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────
CLR_BG       = "#0d1117"
CLR_PAPER    = "#161b22"
CLR_GRID     = "#21262d"
CLR_TEXT     = "#8b949e"
CLR_TITLE    = "#e6edf3"
CLR_10Y      = "#58a6ff"   # blue  – 10-year
CLR_3M       = "#3fb950"   # green – 3-month
CLR_SPREAD   = "#d29922"   # amber – spread
CLR_NEG      = "#f85149"   # red   – inversion
CLR_ZERO     = "#30363d"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CLR_PAPER,
    plot_bgcolor=CLR_BG,
    font=dict(family="IBM Plex Sans", color=CLR_TEXT),
    title_font=dict(color=CLR_TITLE, size=15),
    xaxis=dict(gridcolor=CLR_GRID, zerolinecolor=CLR_GRID, tickcolor=CLR_TEXT),
    yaxis=dict(gridcolor=CLR_GRID, zerolinecolor=CLR_GRID, tickcolor=CLR_TEXT),
    legend=dict(bgcolor=CLR_PAPER, bordercolor=CLR_GRID, borderwidth=1, font=dict(color=CLR_TEXT)),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=CLR_PAPER, bordercolor=CLR_GRID, font=dict(color=CLR_TITLE, family="IBM Plex Mono")),
    margin=dict(l=50, r=20, t=50, b=40),
)

# ─────────────────────────────────────────────
#  DEMO DATA (koristi se dok nema uploada)
# ─────────────────────────────────────────────
@st.cache_data
def get_demo_data():
    np.random.seed(42)
    dates = pd.date_range("2005-01-01", "2024-12-01", freq="MS")
    n = len(dates)
    trend10 = 5.0 + np.cumsum(np.random.randn(n) * 0.08)
    trend3  = trend10 - 1.5 + np.cumsum(np.random.randn(n) * 0.10)
    # inject inversion 2006-07 to 2007-06
    inv1 = (dates >= "2006-07-01") & (dates <= "2007-06-30")
    trend3[inv1] = trend10[inv1] + np.abs(np.random.randn(inv1.sum()) * 0.3)
    # inject inversion 2022-07 to 2023-06
    inv2 = (dates >= "2022-07-01") & (dates <= "2023-12-31")
    trend3[inv2] = trend10[inv2] + np.abs(np.random.randn(inv2.sum()) * 0.4) + 0.5
    trend10 = np.clip(trend10, 0.5, 9)
    trend3  = np.clip(trend3,  0.1, 9)
    df = pd.DataFrame({"Date": dates, "Y10": np.round(trend10, 3), "Y3M": np.round(trend3, 3)})
    df["Spread"] = np.round(df["Y10"] - df["Y3M"], 3)
    return df

# ─────────────────────────────────────────────
#  PARSE UPLOADED FILE
# ─────────────────────────────────────────────
def parse_upload(file) -> pd.DataFrame | None:
    try:
        if file.name.endswith(".csv"):
            df_raw = pd.read_csv(file)
        else:
            df_raw = pd.read_excel(file)

        # Normalize column names
        df_raw.columns = [str(c).strip() for c in df_raw.columns]

        # ── Try to auto-detect columns ──────────────────────────────────
        # Date column
        date_col = None
        for c in df_raw.columns:
            if any(k in c.lower() for k in ["date", "datum", "period", "time"]):
                date_col = c; break
        if date_col is None:
            # First column likely date
            date_col = df_raw.columns[0]

        # 10-year column
        y10_col = None
        for c in df_raw.columns:
            if any(k in c.lower() for k in ["10y", "10-y", "10 y", "dugo", "long", "10g"]):
                y10_col = c; break

        # 3-month column
        y3m_col = None
        for c in df_raw.columns:
            if any(k in c.lower() for k in ["3m", "3-m", "3 m", "krat", "short", "tbill", "3mj"]):
                y3m_col = c; break

        # If auto-detection failed, pick numerics in order
        num_cols = [c for c in df_raw.columns if c != date_col and pd.api.types.is_numeric_dtype(df_raw[c])]
        if y10_col is None and len(num_cols) >= 1:
            y10_col = num_cols[0]
        if y3m_col is None and len(num_cols) >= 2:
            y3m_col = num_cols[1]

        if y10_col is None or y3m_col is None:
            return None

        df = pd.DataFrame()
        df["Date"]   = pd.to_datetime(df_raw[date_col], dayfirst=True, errors="coerce")
        df["Y10"]    = pd.to_numeric(df_raw[y10_col],   errors="coerce")
        df["Y3M"]    = pd.to_numeric(df_raw[y3m_col],   errors="coerce")
        df = df.dropna(subset=["Date","Y10","Y3M"]).sort_values("Date").reset_index(drop=True)

        # Handle percentages > 1 stored as 0–1
        if df["Y10"].max() <= 1.0:
            df["Y10"] *= 100
            df["Y3M"] *= 100

        df["Spread"] = df["Y10"] - df["Y3M"]
        return df, y10_col, y3m_col, date_col

    except Exception as e:
        st.error(f"Greška pri čitanju fajla: {e}")
        return None

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-title">📂 Podaci</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Učitaj Excel / CSV", type=["xlsx","xls","csv"])

    st.markdown("---")
    st.markdown('<p class="section-title">⚙️ Podešavanja</p>', unsafe_allow_html=True)
    show_ma  = st.checkbox("Prikaži pokretni prosjek (MA)", value=True)
    ma_win   = st.slider("Prozor MA (mj.)", 3, 24, 12) if show_ma else 12
    show_rec = st.checkbox("Prikaži period inverzije", value=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🗂️ Mapiranje kolona</p>', unsafe_allow_html=True)
    st.caption("Automatski detekovano. Prikaži detalje:")
    show_mapping = st.checkbox("Prikaži mapiranje", value=False)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#484f58; line-height:1.7'>
    <strong style='color:#58a6ff'>Finansijska matematika</strong><br>
    Analiza krive prinosa (Yield Curve)<br>
    Trezorski zapisi SAD – FRED podaci<br><br>
    <em>Inverzija krive prinosa = razlika<br>
    između 10Y i 3M prinosa &lt; 0</em>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
col_map = {}
if uploaded:
    result = parse_upload(uploaded)
    if result:
        df, c10, c3m, cdate = result
        col_map = {"Datum": cdate, "10Y kolona": c10, "3M kolona": c3m}
        using_demo = False
    else:
        st.error("Ne mogu automatski prepoznati kolone. Provjeri format fajla.")
        df = get_demo_data()
        using_demo = True
else:
    df = get_demo_data()
    using_demo = True

if show_mapping and col_map:
    with st.sidebar:
        for k, v in col_map.items():
            st.caption(f"{k}: **{v}**")

if show_ma:
    df["Y10_MA"]   = df["Y10"].rolling(ma_win, min_periods=1).mean()
    df["Y3M_MA"]   = df["Y3M"].rolling(ma_win, min_periods=1).mean()
    df["Spread_MA"]= df["Spread"].rolling(ma_win, min_periods=1).mean()

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style='color:#e6edf3; font-size:1.9rem; font-weight:700; margin-bottom:4px'>
    📈 Analiza Krive Prinosa
</h1>
<p style='color:#58a6ff; font-size:0.85rem; font-family:"IBM Plex Mono"; margin-top:0'>
    Trezorski zapisi — 10-godišnji vs 3-mjesečni · Yield Spread Analysis
</p>
""", unsafe_allow_html=True)

if using_demo:
    st.info("ℹ️ Prikazuju se **demo podaci**. Učitajte vaš Excel fajl u lijevoj bočnoj traci.")

st.markdown("---")

# ─────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────
latest   = df.iloc[-1]
prev     = df.iloc[-2] if len(df) > 1 else latest
period   = f"{df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}"
inv_pct  = (df["Spread"] < 0).mean() * 100
max_sp   = df["Spread"].max()
min_sp   = df["Spread"].min()
avg_sp   = df["Spread"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("10Y Prinos (zadnji)", f"{latest['Y10']:.2f}%",
          f"{latest['Y10']-prev['Y10']:+.2f}%")
c2.metric("3M Prinos (zadnji)",  f"{latest['Y3M']:.2f}%",
          f"{latest['Y3M']-prev['Y3M']:+.2f}%")
c3.metric("Spread (zadnji)",     f"{latest['Spread']:.2f}%",
          f"{latest['Spread']-prev['Spread']:+.2f}%",
          delta_color="inverse" if latest['Spread'] < 0 else "normal")
c4.metric("% inverzije",         f"{inv_pct:.1f}%",
          help="Postotak perioda kada je spread bio negativan")
c5.metric("Period podataka",     period)

st.markdown("---")

# ─────────────────────────────────────────────
#  HELPER: inversion shading
# ─────────────────────────────────────────────
def add_inversion_shading(fig, df, row=None, col=None):
    if not show_rec:
        return
    in_inv, start = False, None
    for _, r in df.iterrows():
        if r["Spread"] < 0 and not in_inv:
            in_inv, start = True, r["Date"]
        elif r["Spread"] >= 0 and in_inv:
            kw = dict(type="rect", xref="x", yref="paper",
                      x0=start, x1=r["Date"], y0=0, y1=1,
                      fillcolor=CLR_NEG, opacity=0.08, line_width=0, layer="below")
            if row: kw.update(row=row, col=col)
            fig.add_shape(**kw)
            in_inv = False
    if in_inv:
        kw = dict(type="rect", xref="x", yref="paper",
                  x0=start, x1=df["Date"].iloc[-1], y0=0, y1=1,
                  fillcolor=CLR_NEG, opacity=0.08, line_width=0, layer="below")
        if row: kw.update(row=row, col=col)
        fig.add_shape(**kw)

# ─────────────────────────────────────────────
#  TAB LAYOUT
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Opći pregled",
    "📉 Kretanje obveznica",
    "🔀 Yield Spread",
    "📋 Tabela podataka"
])

# ══════════════════════════════════════════════
#  TAB 1 – OPĆI PREGLED
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### Opće statističke karakteristike")

    c_l, c_r = st.columns(2)

    # ── Box statistics ──────────────────────────
    with c_l:
        st.markdown('<p class="section-title">Statistički sažetak</p>', unsafe_allow_html=True)
        stats = df[["Y10","Y3M","Spread"]].describe().T
        stats.index = ["10Y prinos (%)","3M prinos (%)","Spread (%)"]
        stats.columns = ["N","Prosjek","Std. dev","Min","25%","Medijana","75%","Max"]
        st.dataframe(stats.round(3).style
                     .background_gradient(cmap="Blues", subset=["Prosjek"])
                     .format("{:.3f}"),
                     use_container_width=True)

        st.markdown('<div class="info-box">'
            '<strong>Prinos (Yield)</strong> je godišnji povrat koji investitor ostvaruje '
            'držanjem obveznice do dospijeća. '
            '<br><strong>10Y prinos</strong> odražava dugoročna očekivanja, dok je '
            '<strong>3M prinos</strong> pod neposrednim uticajem monetarne politike centralne banke.'
            '</div>', unsafe_allow_html=True)

    # ── Distribution chart ───────────────────────
    with c_r:
        st.markdown('<p class="section-title">Distribucija prinosa</p>', unsafe_allow_html=True)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df["Y10"], name="10Y prinos", nbinsx=40,
            marker_color=CLR_10Y, opacity=0.7))
        fig_dist.add_trace(go.Histogram(
            x=df["Y3M"], name="3M prinos", nbinsx=40,
            marker_color=CLR_3M, opacity=0.7))
        fig_dist.update_layout(**PLOTLY_LAYOUT,
            title="Distribucija historijskih prinosa",
            xaxis_title="Prinos (%)", yaxis_title="Frekvencija",
            barmode="overlay", height=300)
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # ── Scatter: 10Y vs 3M ───────────────────────
    st.markdown('<p class="section-title">Korelaciona analiza</p>', unsafe_allow_html=True)
    c_s1, c_s2 = st.columns([2, 1])
    with c_s1:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=df["Y3M"], y=df["Y10"],
            mode="markers",
            marker=dict(
                color=df["Spread"].fillna(0),  # popuni NaN sa 0 ili nekim defaultom
                colorscale=[[0, CLR_NEG], [0.5, CLR_GRID], [1, CLR_10Y]],  # boje u HEX ili CSS
                size=5,
                opacity=0.7,
                colorbar=dict(
                    title=dict(
                        text="Spread %",        # naziv colorbara
                        font=dict(color=CLR_TEXT)  # font naslova
                    ), tickfont=dict(color=CLR_TEXT))   # font tickova,
                ),
            text=df["Date"].dt.strftime("%b %Y"),
            hovertemplate="<b>%{text}</b><br>3M: %{x:.2f}%<br>10Y: %{y:.2f}%<extra></extra>"
        ))
        # 45° line
        mn, mx = min(df["Y3M"].min(), df["Y10"].min()), max(df["Y3M"].max(), df["Y10"].max())
        fig_sc.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color=CLR_ZERO, dash="dash", width=1), name="3M = 10Y (inverzija)"))
        fig_sc.update_layout(**PLOTLY_LAYOUT,
            title="Disperzija: 3M prinos vs 10Y prinos",
            xaxis_title="3M prinos (%)", yaxis_title="10Y prinos (%)",
            height=350)
        st.plotly_chart(fig_sc, use_container_width=True)

    with c_s2:
        corr = df[["Y10","Y3M","Spread"]].corr()
        st.markdown('<div class="info-box">'
            f'<strong>Korelacija 10Y/3M:</strong> {corr.loc["Y10","Y3M"]:.3f}<br><br>'
            f'<strong>Maksimalni spread:</strong> {max_sp:.2f}%<br>'
            f'<strong>Minimalni spread:</strong> {min_sp:.2f}%<br>'
            f'<strong>Prosječni spread:</strong> {avg_sp:.2f}%<br><br>'
            f'<strong>Periodi inverzije:</strong> {inv_pct:.1f}% ukupnog vremena<br><br>'
            '<em>Tačke ispod dijagonalne linije označavaju inverziju krive prinosa '
            '(3M prinos veći od 10Y).</em>'
            '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 – KRETANJE OBVEZNICA (oba na jednom grafikonu)
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### Historijsko kretanje prinosa")

    fig_both = go.Figure()

    # 10Y
    fig_both.add_trace(go.Scatter(
        x=df["Date"], y=df["Y10"], name="10Y trezorski zapis",
        line=dict(color=CLR_10Y, width=1.5), opacity=0.8,
        hovertemplate="10Y: <b>%{y:.2f}%</b><extra></extra>"))
    if show_ma:
        fig_both.add_trace(go.Scatter(
            x=df["Date"], y=df["Y10_MA"],
            name=f"10Y MA({ma_win}m)", line=dict(color=CLR_10Y, width=2.5, dash="dot"),
            hovertemplate=f"10Y MA: <b>%{{y:.2f}}%</b><extra></extra>"))

    # 3M
    fig_both.add_trace(go.Scatter(
        x=df["Date"], y=df["Y3M"], name="3M trezorski zapis",
        line=dict(color=CLR_3M, width=1.5), opacity=0.8,
        hovertemplate="3M: <b>%{y:.2f}%</b><extra></extra>"))
    if show_ma:
        fig_both.add_trace(go.Scatter(
            x=df["Date"], y=df["Y3M_MA"],
            name=f"3M MA({ma_win}m)", line=dict(color=CLR_3M, width=2.5, dash="dot"),
            hovertemplate=f"3M MA: <b>%{{y:.2f}}%</b><extra></extra>"))

    add_inversion_shading(fig_both, df)
    fig_both.update_layout(**PLOTLY_LAYOUT,
        title="Kretanje prinosa: 10-godišnji vs 3-mjesečni trezorski zapisi",
        xaxis_title="Datum", yaxis_title="Prinos (%)",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor=CLR_PAPER, bordercolor=CLR_GRID, font=dict(color=CLR_TEXT)))
    st.plotly_chart(fig_both, use_container_width=True)

    if show_rec:
        st.caption("🔴 Crveno osjenčanje = period inverzije krive prinosa (3M > 10Y)")

    st.markdown("---")

    # ── Separate sparklines ───────────────────────
    st.markdown("#### Zasebni prikaz svake obveznice")
    fig_sub = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("10-godišnji trezorski zapis", "3-mjesečni trezorski zapis"),
                            vertical_spacing=0.1)

    fig_sub.add_trace(go.Scatter(x=df["Date"], y=df["Y10"],
        line=dict(color=CLR_10Y, width=1.4), name="10Y",
        fill="tozeroy", fillcolor=f"rgba(88,166,255,0.06)"), row=1, col=1)
    if show_ma:
        fig_sub.add_trace(go.Scatter(x=df["Date"], y=df["Y10_MA"],
            line=dict(color=CLR_10Y, width=2.2, dash="dot"), name=f"10Y MA"), row=1, col=1)

    fig_sub.add_trace(go.Scatter(x=df["Date"], y=df["Y3M"],
        line=dict(color=CLR_3M, width=1.4), name="3M",
        fill="tozeroy", fillcolor=f"rgba(63,185,80,0.06)"), row=2, col=1)
    if show_ma:
        fig_sub.add_trace(go.Scatter(x=df["Date"], y=df["Y3M_MA"],
            line=dict(color=CLR_3M, width=2.2, dash="dot"), name=f"3M MA"), row=2, col=1)

    add_inversion_shading(fig_sub, df, row=1, col=1)
    add_inversion_shading(fig_sub, df, row=2, col=1)

    fig_sub.update_layout(**PLOTLY_LAYOUT,
        height=500,
        showlegend=False,
        title="")
    fig_sub.update_yaxes(title_text="Prinos (%)")
    st.plotly_chart(fig_sub, use_container_width=True)

    # ── Key differences ───────────────────────────
    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="info-box">'
            '<strong>10-godišnji trezorski zapis (T-Note)</strong><br><br>'
            '• Dugoročni instrument sa dospijećem od 10 godina<br>'
            '• Odražava dugoročna inflacijska očekivanja<br>'
            '• Referentna stopa za dugoročne hipoteke i kredite<br>'
            '• Osjetljiv na fiskalne projekcije i rast BDP-a<br>'
            '• Tipično viši prinos usljed premije ročnosti'
            '</div>', unsafe_allow_html=True)
    with cb:
        st.markdown('<div class="info-box">'
            '<strong>3-mjesečni trezorski zapis (T-Bill)</strong><br><br>'
            '• Kratkoročni instrument sa dospijećem od 91 dan<br>'
            '• Gotovo direktno vezan za politiku Fed-a (federal funds rate)<br>'
            '• Smatra se "nerizičnom" investicijom u kratkom roku<br>'
            '• Reaguje brzo na promjene monetarne politike<br>'
            '• Koristi se kao benchmark za gotovinske ekvivalente'
            '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 3 – SPREAD (Kriva prinosa)
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### Yield Spread — Kriva prinosa (10Y − 3M)")

    # ── Main spread chart ─────────────────────────
    fig_sp = go.Figure()

    # Fill above/below zero
    spread_pos = df["Spread"].copy(); spread_pos[spread_pos < 0]  = 0
    spread_neg = df["Spread"].copy(); spread_neg[spread_neg >= 0] = 0

    fig_sp.add_trace(go.Scatter(
        x=df["Date"], y=spread_pos,
        fill="tozeroy", fillcolor=f"rgba(88,166,255,0.18)",
        line=dict(width=0), name="Normalna kriva (spread > 0)", showlegend=True))
    fig_sp.add_trace(go.Scatter(
        x=df["Date"], y=spread_neg,
        fill="tozeroy", fillcolor=f"rgba(248,81,73,0.25)",
        line=dict(width=0), name="Inverzija krive (spread < 0)", showlegend=True))
    fig_sp.add_trace(go.Scatter(
        x=df["Date"], y=df["Spread"],
        line=dict(color=CLR_SPREAD, width=2),
        name="Yield spread (10Y−3M)",
        hovertemplate="Spread: <b>%{y:.2f}%</b><extra></extra>"))
    if show_ma:
        fig_sp.add_trace(go.Scatter(
            x=df["Date"], y=df["Spread_MA"],
            line=dict(color=CLR_SPREAD, width=2.5, dash="dot"),
            name=f"Spread MA({ma_win}m)"))

    # Zero line
    fig_sp.add_hline(y=0, line_dash="dash", line_color=CLR_NEG,
                     line_width=1.2, annotation_text="0%",
                     annotation_font=dict(color=CLR_NEG, size=11))

    fig_sp.update_layout(**PLOTLY_LAYOUT,
        title="Yield Spread: 10-godišnji minus 3-mjesečni trezorski zapis",
        xaxis_title="Datum", yaxis_title="Spread (%)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor=CLR_PAPER, bordercolor=CLR_GRID, font=dict(color=CLR_TEXT)))
    st.plotly_chart(fig_sp, use_container_width=True)

    # ── Spread stats row ───────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Maks. spread", f"{max_sp:.2f}%",
              df.loc[df["Spread"].idxmax(), "Date"].strftime("%b %Y"))
    k2.metric("Min. spread", f"{min_sp:.2f}%",
              df.loc[df["Spread"].idxmin(), "Date"].strftime("%b %Y"),
              delta_color="inverse")
    k3.metric("Prosječni spread", f"{avg_sp:.2f}%")
    k4.metric("Trajanje inverzije", f"{int((df['Spread']<0).sum())} mj.",
              f"od {len(df)} mj. ukupno")

    st.markdown("---")

    # ── Waterfall: spread change ───────────────────
    st.markdown("#### Godišnji prosjek spreada")
    df_yr = df.groupby(df["Date"].dt.year)["Spread"].mean().reset_index()
    df_yr.columns = ["Godina","Spread"]
    fig_yr = go.Figure(go.Bar(
        x=df_yr["Godina"].astype(str),
        y=df_yr["Spread"],
        marker_color=[CLR_10Y if v >= 0 else CLR_NEG for v in df_yr["Spread"]],
        text=df_yr["Spread"].round(2).astype(str) + "%",
        textposition="outside",
        textfont=dict(size=9, color=CLR_TEXT),
    ))
    fig_yr.update_layout(**PLOTLY_LAYOUT,
        title="Godišnji prosjek yield spreada (10Y − 3M)",
        xaxis_title="Godina", yaxis_title="Spread (%)",
        height=320)
    fig_yr.add_hline(y=0, line_dash="dash", line_color=CLR_NEG, line_width=1)
    st.plotly_chart(fig_yr, use_container_width=True)

    # ── Explanation ───────────────────────────────
    st.markdown("---")
    e1, e2 = st.columns(2)
    with e1:
        st.markdown('<div class="info-box">'
            '<strong>Šta znači yield spread?</strong><br><br>'
            'Yield spread (kriva prinosa) mjeri razliku između dugoročnih i kratkoročnih '
            'kamatnih stopa. Normalna kriva prinosa ima <strong style="color:#58a6ff">pozitivan spread</strong> '
            '— investitori zahtijevaju više prinose za dugoročna ulaganja zbog neizvjesnosti.<br><br>'
            'Kada je spread <strong style="color:#f85149">negativan (inverzija)</strong>, '
            'kratkoročni prinosi prevazilaze dugoročne. Ovo se historijski smatralo '
            'jednim od najpouzdanijih indikatora predstojeće recesije.'
            '</div>', unsafe_allow_html=True)
    with e2:
        st.markdown('<div class="info-box">'
            '<strong>Zašto je inverzija važna?</strong><br><br>'
            '• Svaka recesija u SAD od 1950. bila je prethodila inverzijom krive prinosa<br>'
            '• Tipično se recesija desi 6–18 mjeseci nakon inverzije<br>'
            '• Inverzija signalizira da tržišta očekuju pad stopa u budućnosti<br>'
            '• Fed koristi ovaj indikator u procjeni ekonomskih uslova<br><br>'
            '<em>Napomena: korelacija nije uzročnost — inverzija je signal, ne garancija.</em>'
            '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 4 – TABELA
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### Tabela podataka")

    col_filter, col_dl = st.columns([3, 1])
    with col_filter:
        years = sorted(df["Date"].dt.year.unique())
        sel_years = st.multiselect("Filtriraj po godini", years, default=years[-5:] if len(years) > 5 else years)

    df_show = df[df["Date"].dt.year.isin(sel_years)].copy()
    df_show["Datum"] = df_show["Date"].dt.strftime("%d.%m.%Y.")
    df_show = df_show[["Datum","Y10","Y3M","Spread"]].rename(columns={
        "Y10": "10Y prinos (%)", "Y3M": "3M prinos (%)", "Spread": "Spread (%)"
    })

    def colour_spread(val):
        if val < 0:   return f"color: {CLR_NEG}; font-weight:600"
        elif val > 2: return f"color: {CLR_10Y}"
        return f"color: {CLR_3M}"

    st.dataframe(
        df_show.style
            .applymap(colour_spread, subset=["Spread (%)"])
            .format({"10Y prinos (%)": "{:.3f}", "3M prinos (%)": "{:.3f}", "Spread (%)": "{:.3f}"}),
        use_container_width=True, height=480
    )

    with col_dl:
        csv_bytes = df_show.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
        st.download_button("⬇ Preuzmi CSV", data=csv_bytes,
                           file_name="yield_spread_podaci.csv", mime="text/csv")

    st.caption(f"Prikazano {len(df_show)} redova · Crveno = inverzija krive prinosa")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:0.72rem; color:#484f58; padding:8px 0 16px'>
    Seminarski rad · Finansijska matematika · Analiza krive prinosa trezorskih zapisa<br>
    Izgrađeno sa Streamlit · Plotly · Pandas
</div>
""", unsafe_allow_html=True)
