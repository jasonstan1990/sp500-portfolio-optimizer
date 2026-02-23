# app.py — PUBLIC (snapshot-only, same layout as your original)
import os
import json
import zlib
import pickle
import sqlite3

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.covariance import LedoitWolf

st.set_page_config(page_title="S&P Portfolio Builder — BL & Risk", layout="wide")

CACHE_DIR = ".portfolio_cache"
DB_PATH = os.path.join(CACHE_DIR, "cache.db")


# ============================
# HERO SECTION (Brunson style)
# ============================

st.markdown("""
# 🚀 Build a Wall-Street Level Portfolio  
## From 500 Stocks → To a Precision-Optimized Portfolio  

Stop guessing. Start allocating like a quant. Without Overpaying Advisors.
""")

st.markdown("---")

# ----------------------------
# DB / Snapshot helpers (with migration)
# ----------------------------
def _ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)


def _unpack(blob: bytes):
    raw = zlib.decompress(blob)
    return pickle.loads(raw)


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _get_columns(conn, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]


def _ensure_schema(conn: sqlite3.Connection):
    if not _table_exists(conn, "published_snapshots"):
        conn.execute("""
            CREATE TABLE published_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_key TEXT NOT NULL,
                published_at TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                payload_blob BLOB NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX idx_published_snapshots_key_id
            ON published_snapshots(snapshot_key, id)
        """)
        conn.commit()
        return

    cols = _get_columns(conn, "published_snapshots")
    if "snapshot_key" not in cols:
        conn.execute("ALTER TABLE published_snapshots ADD COLUMN snapshot_key TEXT")
        conn.commit()
        conn.execute("""
            UPDATE published_snapshots
            SET snapshot_key = COALESCE(snapshot_key, 'daily')
            WHERE snapshot_key IS NULL
        """)
        conn.commit()

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_published_snapshots_key_id
        ON published_snapshots(snapshot_key, id)
    """)
    conn.commit()


def _conn():
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    _ensure_schema(conn)
    return conn


@st.cache_data(show_spinner=False)
def load_snapshot_by_key(snapshot_key: str):
    conn = _conn()
    row = conn.execute("""
        SELECT published_at, meta_json, payload_blob
        FROM published_snapshots
        WHERE snapshot_key = ?
        ORDER BY id DESC
        LIMIT 1
    """, (snapshot_key,)).fetchone()

    if not row:
        # disk fallback (optional)
        disk_path = os.path.join(CACHE_DIR, f"published_{snapshot_key}_latest.pklz")
        if os.path.exists(disk_path):
            with open(disk_path, "rb") as f:
                payload = _unpack(f.read())
            meta = payload.get("meta", {})
            published_at = meta.get("published_at", "unknown")
            return published_at, meta, payload
        return None

    published_at, meta_json, payload_blob = row
    meta = json.loads(meta_json)
    payload = _unpack(payload_blob)
    return published_at, meta, payload


# ============================
# Sidebar — Inputs (SAME SECTIONS, but universe hidden)
# ============================
st.sidebar.header("Portfolio Configuration")

# ✅ Universe hidden: no text_area tickers in public
bench = "^GSPC"

start_date = st.sidebar.date_input("Start date:", value=pd.Timestamp("2020-01-01"))
end_date = st.sidebar.date_input("End date:", value=pd.Timestamp.today())

frequency = st.sidebar.selectbox("Data frequency:", ["daily", "weekly", "monthly"], index=0)
risk_free_rate = st.sidebar.number_input("Risk-free (annual)", min_value=0.0, max_value=1.0, value=0.02, step=0.005)

# ============================
# SPRINT 2 — Screening controls
# ============================
st.sidebar.header("Screening (Sprint 2)")
adtv_usd_min = st.sidebar.number_input("Min ADTV ($)", min_value=0, value=1_000_000, step=100_000)
vol_mult_spx = st.sidebar.slider("Max Volatility vs SPX (×)", 0.5, 5.0, 3.0, 0.1)
beta_min, beta_max = st.sidebar.slider("Beta range", 0.3, 5.0, (0.8, 1.3), 0.05)
min_growth = st.sidebar.slider(
    "Min expected growth (analysts, YoY)",
    -0.50, 1.00, 0.00, 0.05,
    help="Uses published snapshot fundamentals (no yfinance)."
)

# ============================
# Ranking controls
# ============================
st.sidebar.header("Ranking")
top_n = st.sidebar.slider("Select Top-N tickers", 5, 80, 30, 1)
w_mom  = st.sidebar.slider("Weight: Momentum (6/12m)", 0.0, 1.0, 0.40, 0.05)
w_qual = st.sidebar.slider("Weight: Quality (1y Sharpe)", 0.0, 1.0, 0.30, 0.05)
w_val  = st.sidebar.slider("Weight: Value (EY + 1/PEG)", 0.0, 1.0, 0.20, 0.05)
w_lvol = st.sidebar.slider("Weight: LowVol (1/σ)", 0.0, 1.0, 0.10, 0.05)
if abs((w_mom + w_qual + w_val + w_lvol) - 1.0) > 1e-9:
    st.sidebar.warning("Ranking weights should sum to 1.0")

# ============================
# EF constraints
# ============================
st.sidebar.header("EF Constraints")
max_weight = st.sidebar.number_input("Max weight per stock", 0.0, 1.0, 0.10, 0.01)
min_weight = st.sidebar.number_input("Min weight per stock", 0.0, 1.0, 0.00, 0.01)
num_portfolios = st.sidebar.slider("Monte Carlo portfolios", 100, 15000, 4000, 100)

# ============================
# SPRINT 3 — Black–Litterman controls
# ============================
st.sidebar.header("Black–Litterman (Sprint 3)")
enable_bl = st.sidebar.checkbox("Enable Black–Litterman views", value=False)
bl_delta = st.sidebar.number_input("Risk aversion δ", 0.1, 10.0, 2.5, 0.1)
bl_tau   = st.sidebar.number_input("Tau (τ)", 0.0, 1.0, 0.05, 0.01)


# ============================
# Helpers (same as original)
# ============================
def pct_change_safe(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.ffill().pct_change().dropna(how="all")

def annualize_mean_std(ret: pd.Series, freq: str = "daily"):
    f = {"daily": 252, "weekly": 52, "monthly": 12}[freq]
    mu = ret.mean() * f
    sigma = ret.std() * np.sqrt(f)
    return mu, sigma, f

def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    lw = LedoitWolf().fit(returns.dropna().values)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)

def compute_adtv_usd(close: pd.DataFrame, vol: pd.DataFrame, lookback_days: int = 60) -> pd.Series:
    recent = close.iloc[-lookback_days:].mul(vol.iloc[-lookback_days:], axis=0)
    return recent.mean(axis=0)

def compute_beta(returns: pd.DataFrame, bench_col: str) -> pd.Series:
    if bench_col not in returns.columns:
        return pd.Series(index=returns.columns, dtype=float)
    rm = returns[bench_col].dropna()
    betas = {}
    for c in returns.columns:
        if c == bench_col:
            continue
        ri = returns[c].reindex(rm.index).dropna()
        common = ri.index.intersection(rm.index)
        if len(common) < 30:
            betas[c] = np.nan
            continue
        cov = np.cov(ri.loc[common], rm.loc[common])[0, 1]
        varm = np.var(rm.loc[common])
        betas[c] = cov / varm if varm > 0 else np.nan
    return pd.Series(betas)

def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    return (s - s.mean()) / std if std and not np.isnan(std) and std != 0 else s * 0


# ============================
# DATA INGEST (PUBLIC) — LOAD SNAPSHOT ONLY
# ============================
latest = load_snapshot_by_key(frequency)
if latest is None:
    st.error(f"No published snapshot for '{frequency}'. Open admin.py and click PUBLISH ALL.")
    st.stop()

published_at, meta, payload = latest

close_df = payload["data"]["close_df"]
vol_df   = payload["data"]["vol_df"]
info_df_base = payload["data"]["info_df"]
tickers_all = payload["data"]["tickers"]   # hidden universe

# Clip by UI date range (no downloads!)
close_df = close_df.loc[pd.Timestamp(start_date):pd.Timestamp(end_date)]
vol_df = vol_df.reindex(close_df.index)

if close_df.empty or bench not in close_df.columns:
    st.error("No valid snapshot data in selected date range OR benchmark missing.")
    st.stop()

returns_all = pct_change_safe(close_df)
returns_all = returns_all.dropna(axis=1, how="any")

# Hidden universe resolved against snapshot
available_tickers = [t for t in tickers_all if t in returns_all.columns]
if not available_tickers:
    st.error("Snapshot loaded but no tickers with complete data in selected period.")
    st.stop()

stock_returns_full = returns_all[available_tickers]
cov_lw_full = ledoit_wolf_cov(stock_returns_full)

# ============================
# SCREENING (Sprint 2) + Growth & Beta range (NO yfinance)
# ============================
st.write("")
with st.expander("Screening & Ranking — Details", expanded=True):
    st.markdown("**Screening Filters**: ADTV $, Volatility vs SPX, Beta range, Analysts' expected growth (YoY).")

    adtv_usd = compute_adtv_usd(
        close_df[available_tickers],
        vol_df[available_tickers],
        lookback_days=min(60, len(close_df))
    )
    adtv_pass = adtv_usd[adtv_usd >= adtv_usd_min].index.tolist()

    spx_ret = returns_all[bench].dropna()
    _, spx_sigma_ann, ann_f = annualize_mean_std(spx_ret, frequency)
    vols = pd.Series({t: annualize_mean_std(stock_returns_full[t], frequency)[1] for t in available_tickers})
    vol_pass = vols[vols <= (vol_mult_spx * spx_sigma_ann)].index.tolist()

    betas = compute_beta(returns_all[available_tickers + [bench]], bench)
    beta_pass = betas[(betas >= beta_min) & (betas <= beta_max)].index.tolist()

    # Fundamentals from snapshot
    info_df_all = info_df_base.reindex(available_tickers).copy()
    growth_series_all = info_df_all.get("AnalystGrowth", pd.Series(index=available_tickers, dtype=float))
    growth_pass = growth_series_all[growth_series_all >= min_growth].index.tolist()

    screened = sorted(set(adtv_pass) & set(vol_pass) & set(beta_pass) & set(growth_pass))
    st.write(f"Screened universe: **{len(screened)} / {len(available_tickers)}** tickers pass filters.")

    scr_df = pd.DataFrame({
        "ADTV $ (60d)": adtv_usd.reindex(available_tickers),
        "Vol (ann)": vols.reindex(available_tickers),
        "Beta": betas.reindex(available_tickers),
        "Analyst Growth (YoY)": growth_series_all.reindex(available_tickers)
    }).loc[available_tickers].sort_values(by="ADTV $ (60d)", ascending=False)

    st.dataframe(
        scr_df.style.format({
            "ADTV $ (60d)": "{:,.0f}",
            "Vol (ann)": "{:.2%}",
            "Beta": "{:.2f}",
            "Analyst Growth (YoY)": "{:.2%}"
        }),
        use_container_width=True
    )

# ============================
# RANKING (Sprint 2) — same logic
# ============================
if not screened:
    st.error("No tickers passed screening. Relax filters and try again.")
    st.stop()

lookback_days = min(len(stock_returns_full), 252)
ret_win = stock_returns_full.iloc[-lookback_days:].copy()

def cumulative_return(window_days):
    if len(ret_win) < window_days:
        return pd.Series(index=ret_win.columns, dtype=float)
    return (1 + ret_win.iloc[-window_days:]).prod() - 1

mom6 = cumulative_return(126)
mom12 = cumulative_return(252) if len(ret_win) >= 252 else mom6
momentum = 0.5 * mom6.add(0, fill_value=0) + 0.5 * mom12.add(0, fill_value=0)
momentum = momentum.reindex(screened)

excess_daily = ret_win.sub(risk_free_rate / {"daily": 252, "weekly": 52, "monthly": 12}[frequency], axis=0)
sharpe_1y = (excess_daily.mean() / excess_daily.std()).replace([np.inf, -np.inf], np.nan)
quality = sharpe_1y.reindex(screened)

sigma_ann_series = ret_win.std() * np.sqrt({"daily": 252, "weekly": 52, "monthly": 12}[frequency])
lowvol = (1.0 / sigma_ann_series.replace(0, np.nan)).reindex(screened)

# Value: Earnings Yield = 1/PE (from snapshot)
info_df = info_df_base.reindex(screened).copy()
earn_yield = (1.0 / info_df["PE"]).replace([np.inf, -np.inf], np.nan)

Z_mom = zscore(momentum.fillna(momentum.median()))
Z_qual = zscore(quality.fillna(quality.median()))
Z_val = zscore(earn_yield.fillna(earn_yield.median()))
Z_lvol = zscore(lowvol.fillna(lowvol.median()))

composite = (w_mom * Z_mom + w_qual * Z_qual + w_val * Z_val + w_lvol * Z_lvol).sort_values(ascending=False)
ranked = composite.index.tolist()

rank_df = pd.DataFrame({
    "Sector": info_df["Sector"].reindex(ranked),
    "Momentum(6/12m)": momentum.reindex(ranked),
    "Sharpe(1y)": quality.reindex(ranked),
    "Earnings Yield (1/PE)": earn_yield.reindex(ranked),
    "LowVol (1/σ)": lowvol.reindex(ranked),
    "Composite Z": composite
}).head(top_n)

st.subheader("Top-N ranked universe")
st.dataframe(rank_df.style.format({
    "Momentum(6/12m)": "{:.2%}", "Sharpe(1y)": "{:.2f}",
    "Earnings Yield (1/PE)": "{:.2f}", "LowVol (1/σ)": "{:.4f}",
    "Composite Z": "{:.2f}"
}), use_container_width=True)

csv_bytes = rank_df.reset_index().rename(columns={"index": "Ticker"}).to_csv(index=False).encode()
st.download_button("⬇️ Download Top-N CSV", data=csv_bytes, file_name="topN_ranked.csv", mime="text/csv")

selected = rank_df.index.tolist()
stock_returns = returns_all[selected]
cov_lw = ledoit_wolf_cov(stock_returns)

date_range = st.slider(
    "Select optimization date range:",
    min_value=stock_returns.index.min().to_pydatetime(),
    max_value=stock_returns.index.max().to_pydatetime(),
    value=(stock_returns.index.min().to_pydatetime(), stock_returns.index.max().to_pydatetime()),
    format="YYYY-MM-DD"
)
stock_returns_window = stock_returns.loc[date_range[0]:date_range[1]]
mean_ret_win = stock_returns_window.mean()
cov_win = ledoit_wolf_cov(stock_returns_window)

sector_map = info_df["Sector"].reindex(selected).fillna("Unknown").to_dict()
market_caps = info_df["MarketCap"].reindex(selected)
if market_caps.isna().all():
    w_mkt = pd.Series(1 / len(selected), index=selected)
else:
    mc = market_caps.fillna(market_caps.median())
    w_mkt = mc / mc.sum()

f_ann = {"daily": 252, "weekly": 52, "monthly": 12}[frequency]

# ============================
# EF Monte Carlo (Baseline)
# ============================
st.subheader("Efficient Frontier (Monte Carlo on Top-N)")

def sample_weights(n, min_w, max_w):
    for _ in range(5000):
        w = np.random.random(n)
        w = w / np.sum(w)
        if (w >= min_w - 1e-12).all() and (w <= max_w + 1e-12).all():
            return w
    w = np.maximum(w, min_w)
    w = np.minimum(w, max_w)
    return w / np.sum(w)

n = len(selected)
results = np.zeros((4, num_portfolios))
weights_record = []

covM = cov_win.values
muV = mean_ret_win.values
rf = risk_free_rate

for i in range(num_portfolios):
    w = sample_weights(n, min_weight, max_weight)
    weights_record.append(w)
    port_mu = (muV @ w) * f_ann
    port_sigma = np.sqrt(w.T @ covM @ w) * np.sqrt(f_ann)
    sharpe = (port_mu - rf) / port_sigma if port_sigma > 0 else np.nan
    results[0, i] = port_mu
    results[1, i] = port_sigma
    results[2, i] = sharpe
    results[3, i] = i

max_sharpe_idx = np.nanargmax(results[2])
min_risk_idx = np.nanargmin(results[1])

w_opt = weights_record[int(results[3, max_sharpe_idx])]
w_min = weights_record[int(results[3, min_risk_idx])]

opt_df = pd.DataFrame({"Stock": selected, "Optimal Weights": w_opt}).sort_values("Optimal Weights", ascending=False)
min_df = pd.DataFrame({"Stock": selected, "Min Risk Weights": w_min}).sort_values("Min Risk Weights", ascending=False)

# ============================
# BLACK–LITTERMAN (Sprint 3) — same logic
# ============================
sectors_in_universe = sorted(pd.Series(sector_map).unique())

bl_views = []
bl_conf = []
if enable_bl:
    st.sidebar.markdown("—")
    st.sidebar.subheader("BL Sector Views")
    chosen_sectors = st.sidebar.multiselect("Pick up to 2 sectors for views", sectors_in_universe, max_selections=2)
    for s in chosen_sectors:
        exp = st.sidebar.slider(f"{s}: Expected excess (annualized, %)", -20.0, 20.0, 3.0, 0.5)
        conf = st.sidebar.slider(f"{s}: Confidence (%)", 1, 99, 60, 1)
        bl_views.append((s, exp / 100.0))
        bl_conf.append(conf / 100.0)

mu_bl = None
w_bl = None

if enable_bl and bl_views:
    Sigma_ann = cov_win * f_ann
    pi = bl_delta * (Sigma_ann @ w_mkt.values)
    pi = pd.Series(pi, index=selected)

    P_rows, Q, omegas = [], [], []
    for (sec, q_ann), conf in zip(bl_views, bl_conf):
        mask = [1.0 if sector_map[t] == sec else 0.0 for t in selected]
        msum = sum(mask)
        if msum == 0:
            continue
        p = np.array([m / msum for m in mask], dtype=float)
        P_rows.append(p)
        Q.append(q_ann)
        var_view = p @ (Sigma_ann.values * bl_tau) @ p.T
        omega_i = var_view * (1.0 - conf) / max(conf, 1e-6)
        omegas.append(omega_i)

    if P_rows:
        P = np.vstack(P_rows)
        Q = np.array(Q)
        Omega = np.diag(omegas)
        tauSigma = Sigma_ann.values * bl_tau
        inv_tauSigma = np.linalg.pinv(tauSigma)
        middle = inv_tauSigma + P.T @ np.linalg.pinv(Omega) @ P
        rhs = inv_tauSigma @ pi.values + P.T @ np.linalg.pinv(Omega) @ Q
        mu_bl_vec = np.linalg.pinv(middle) @ rhs
        mu_bl = pd.Series(mu_bl_vec, index=selected)

        results_bl = np.zeros((4, num_portfolios))
        weights_bl_rec = []
        for i in range(num_portfolios):
            w = sample_weights(n, min_weight, max_weight)
            weights_bl_rec.append(w)
            port_mu = (mu_bl.values @ w)
            port_sigma = np.sqrt(w.T @ Sigma_ann.values @ w)
            sharpe = (port_mu - rf) / port_sigma if port_sigma > 0 else np.nan
            results_bl[0, i] = port_mu
            results_bl[1, i] = port_sigma
            results_bl[2, i] = sharpe
            results_bl[3, i] = i

        max_sharpe_idx_bl = np.nanargmax(results_bl[2])
        w_bl = weights_bl_rec[int(results_bl[3, max_sharpe_idx_bl])]





# ============================
# Charts & Tabs (same)
# ============================
port_returns_opt = (stock_returns @ w_opt)
port_returns_min = (stock_returns @ w_min)
spx = returns_all[bench]
cum_opt = (1 + port_returns_opt).cumprod()
cum_min = (1 + port_returns_min).cumprod()
cum_spx = (1 + spx.reindex(cum_opt.index, method="ffill")).cumprod()

if w_bl is not None:
    port_returns_bl = (stock_returns @ w_bl)
    cum_bl = (1 + port_returns_bl).cumprod()
else:
    port_returns_bl = None
    cum_bl = None

mu_opt, sigma_opt, _ = annualize_mean_std(port_returns_opt, frequency)
sh_opt = (mu_opt - risk_free_rate) / sigma_opt if sigma_opt > 0 else np.nan
downside = port_returns_opt[port_returns_opt < 0]
sortino_opt = (mu_opt - risk_free_rate) / (downside.std() * np.sqrt({"daily": 252, "weekly": 52, "monthly": 12}[frequency])) if downside.std() not in (0, np.nan) else np.nan
mdd_opt = (cum_opt / cum_opt.cummax() - 1).min()

risk_summary = pd.DataFrame({
    "Metric": ["Annual Return", "Annual Volatility", "Sharpe", "Sortino", "Max Drawdown"],
    "Value": [f"{mu_opt:.2%}", f"{sigma_opt:.2%}", f"{sh_opt:.2f}", f"{sortino_opt:.2f}", f"{mdd_opt:.2%}"]
})

fig_frontier = go.Figure()
fig_frontier.add_trace(go.Scatter(
    x=results[1], y=results[0], mode="markers",
    marker=dict(color=results[2], colorscale="Viridis", size=5, showscale=True),
    name="Baseline Portfolios",
    text=[f"Ret {r:.2%} | Vol {v:.2%} | Sharpe {s:.2f}" for r, v, s in zip(results[0], results[1], results[2])]
))
fig_frontier.add_trace(go.Scatter(
    x=[results[1, max_sharpe_idx]], y=[results[0, max_sharpe_idx]],
    mode="markers", marker=dict(color="red", size=12, symbol="star"),
    name="Max Sharpe (Baseline)"
))
fig_frontier.add_trace(go.Scatter(
    x=[results[1, min_risk_idx]], y=[results[0, min_risk_idx]],
    mode="markers", marker=dict(color="blue", size=12, symbol="star"),
    name="Min Vol (Baseline)"
))
if w_bl is not None:
    mu_bl_opt = float(mu_bl.values @ w_bl)
    sigma_bl_opt = float(np.sqrt(w_bl.T @ (cov_win.values * f_ann) @ w_bl))
    fig_frontier.add_trace(go.Scatter(
        x=[sigma_bl_opt], y=[mu_bl_opt],
        mode="markers", marker=dict(color="orange", size=12, symbol="diamond"),
        name="Max Sharpe (BL)"
    ))
fig_frontier.update_layout(title="Efficient Frontier (Top-N Universe)", xaxis_title="Volatility (ann)", yaxis_title="Return (ann)")

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(x=cum_opt.index, y=cum_opt, name="Optimal", mode="lines"))
fig_cum.add_trace(go.Scatter(x=cum_min.index, y=cum_min, name="Min Risk", mode="lines"))
fig_cum.add_trace(go.Scatter(x=cum_spx.index, y=cum_spx, name="S&P 500", mode="lines"))
if cum_bl is not None:
    fig_cum.add_trace(go.Scatter(x=cum_bl.index, y=cum_bl, name="BL Optimal", mode="lines"))
fig_cum.update_layout(title="Cumulative Returns", xaxis_title="Date", yaxis_title="Growth of 1")

Sigma_ann = cov_win * f_ann
sigma_p_opt = float(np.sqrt(np.array(w_opt).T @ Sigma_ann.values @ np.array(w_opt)))
if sigma_p_opt > 0:
    mrc = (Sigma_ann.values @ np.array(w_opt)) / sigma_p_opt
    rc_vol = pd.Series(np.array(w_opt) * mrc, index=selected)
    rc_vol_pct = rc_vol / rc_vol.sum()
else:
    rc_vol = pd.Series(0, index=selected)
    rc_vol_pct = rc_vol

asset_sector = pd.Series(sector_map)
rc_sector = rc_vol.groupby(asset_sector).sum().sort_values(ascending=False)
rc_sector_pct = rc_sector / rc_sector.sum()

opt_series = port_returns_opt.dropna()
if len(opt_series) > 50:
    q95 = np.quantile(opt_series, 0.05)
    q99 = np.quantile(opt_series, 0.01)
    var95 = -q95
    var99 = -q99
    cvar95 = -opt_series[opt_series <= q95].mean()
    cvar99 = -opt_series[opt_series <= q99].mean()
else:
    var95 = var99 = cvar95 = cvar99 = np.nan

risk_table = pd.DataFrame({
    "Metric": ["VaR 95% (1-day)", "CVaR 95% (1-day)", "VaR 99% (1-day)", "CVaR 99% (1-day)"],
    "Value": [f"{var95:.2%}", f"{cvar95:.2%}", f"{var99:.2%}", f"{cvar99:.2%}"]
})

stress_rows = []
try:
    port_hist = (stock_returns @ w_opt).copy()
    if port_hist.index.min() <= pd.Timestamp("2020-03-01") <= port_hist.index.max():
        shock = (1 + port_hist.loc["2020-03-01":"2020-03-31"]).prod() - 1
        stress_rows.append(["COVID Mar-2020 (hist)", f"{shock:.2%}"])
except Exception:
    pass

sector_effect = {
    "Technology": -0.12,
    "Communication Services": -0.08,
    "Consumer Discretionary": -0.07,
    "Financial Services": -0.03,
    "Industrials": -0.04,
    "Health Care": -0.02,
    "Energy": 0.05,
    "Materials": -0.03,
    "Consumer Defensive": -0.01,
    "Real Estate": -0.06,
    "Utilities": -0.03,
    "Unknown": -0.04,
}
if len(selected) > 0:
    weights_opt_series = pd.Series(w_opt, index=selected)
    sector_w = weights_opt_series.groupby(asset_sector).sum()
    est_shock = float((sector_w * pd.Series(sector_effect)).sum())
    stress_rows.append(["Rates↑/Tech↓/Energy↑ (proxy)", f"{est_shock:.2%}"])

stress_df = pd.DataFrame(stress_rows, columns=["Scenario", "Portfolio Return"])

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Allocations", "Frontier", "Black–Litterman", "Risk", "Reports"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Return (opt)", f"{mu_opt:.2%}")
    c2.metric("Volatility (opt)", f"{sigma_opt:.2%}")
    c3.metric("Sharpe (opt)", f"{sh_opt:.2f}")
    st.plotly_chart(fig_cum, use_container_width=True, key="cum_base")

with tab2:
    st.subheader("Optimal Weights")
    st.dataframe(opt_df.style.format({"Optimal Weights": "{:.2%}"}), use_container_width=True)
    st.subheader("Minimum Risk Weights")
    st.dataframe(min_df.style.format({"Min Risk Weights": "{:.2%}"}), use_container_width=True)

with tab3:
    st.plotly_chart(fig_frontier, use_container_width=True)

with tab4:
    st.markdown("**Black–Litterman Settings**")
    if not enable_bl:
        st.info("Enable BL in the sidebar to define sector views.")
    elif w_bl is None:
        st.warning("No valid BL views defined (pick 1–2 sectors and set expectations).")
    else:
        bl_df = pd.DataFrame({"Stock": selected, "BL Optimal Weights": w_bl, "Sector": [sector_map[t] for t in selected]})
        st.subheader("BL Optimal Weights")
        st.dataframe(bl_df.sort_values("BL Optimal Weights", ascending=False).style.format({"BL Optimal Weights": "{:.2%}"}), use_container_width=True)
        if cum_bl is not None:
            st.subheader("Cumulative Returns (BL vs Baseline)")
            fig_cum_bl_tab = go.Figure(fig_cum)
            st.plotly_chart(fig_cum_bl_tab, use_container_width=True, key="cum_bl")

with tab5:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Risk Metrics (Optimal)")
        st.table(risk_summary)
        st.subheader("VaR / CVaR")
        st.table(risk_table)
    with c2:
        st.subheader("Volatility Risk Contributions (Assets)")
        rc_assets_df = pd.DataFrame({"RC Vol": rc_vol, "RC Vol %": rc_vol_pct}).sort_values("RC Vol %", ascending=False)
        st.dataframe(rc_assets_df.style.format({"RC Vol": "{:.4f}", "RC Vol %": "{:.2%}"}), use_container_width=True)
        st.subheader("Risk Contributions by Sector")
        rc_sector_df = pd.DataFrame({"RC Vol": rc_sector, "RC Vol %": rc_sector_pct}).sort_values("RC Vol %", ascending=False)
        st.dataframe(rc_sector_df.style.format({"RC Vol": "{:.4f}", "RC Vol %": "{:.2%}"}), use_container_width=True)
    st.subheader("Stress Tests")
    if stress_df.empty:
        st.info("Not enough history for historical stress. Synthetic scenario shown if defined.")
    else:
        st.table(stress_df)

with tab6:
    st.download_button(
        "⬇️ Download Optimal Weights (CSV)",
        data=opt_df.to_csv(index=False).encode(),
        file_name="optimal_weights.csv",
        mime="text/csv"
    )
    st.download_button(
        "⬇️ Download Min-Risk Weights (CSV)",
        data=min_df.to_csv(index=False).encode(),
        file_name="minrisk_weights.csv",
        mime="text/csv"
    )
    if w_bl is not None:
        bl_out = pd.DataFrame({"Stock": selected, "BL Weights": w_bl})
        st.download_button(
            "⬇️ Download BL Weights (CSV)",
            data=bl_out.to_csv(index=False).encode(),
            file_name="bl_weights.csv",
            mime="text/csv"
        )

# ----------------------------
# Extra: Investment allocation helper (NO yfinance; uses snapshot last close)
# ----------------------------
st.sidebar.header("Investment Helper")
investment_amount = st.sidebar.number_input("Investment amount", min_value=0.0, value=40_000.0, step=100.0)
allow_fractional = st.sidebar.checkbox("Allow fractional shares", value=True)

if st.sidebar.button("Compute allocation for Optimal"):
    # Use last available close in snapshot (no fetching)
    last_prices = close_df[selected].ffill().iloc[-1].dropna()

    common = pd.DataFrame({"Stock": selected}).merge(
        last_prices.rename("Price"),
        left_on="Stock",
        right_index=True,
        how="inner"
    )
    common = common.merge(
        pd.DataFrame({"Stock": selected, "Optimal Weights": w_opt}),
        on="Stock",
        how="left",
    )

    alloc = investment_amount * common["Optimal Weights"].values
    if allow_fractional:
        shares = alloc / common["Price"].values
    else:
        shares = np.floor(alloc / common["Price"].values)

    total_cost = (shares * common["Price"].values).sum()
    remaining = investment_amount - total_cost

    out_df = pd.DataFrame({
        "Stock": common["Stock"],
        "Weight": common["Optimal Weights"],
        "Price": common["Price"],
        "Shares": shares,
        "Cost": shares * common["Price"].values
    }).sort_values("Weight", ascending=False)

    st.subheader("Portfolio Allocation (Optimal)")
    st.dataframe(
        out_df.style.format({
            "Weight": "{:.2%}",
            "Price": "${:,.2f}",
            "Shares": "{:.4f}",
            "Cost": "${:,.2f}"
        }),
        use_container_width=True
    )
    st.write(f"**Total Cost:** ${total_cost:,.2f} | **Remaining:** ${remaining:,.2f}")