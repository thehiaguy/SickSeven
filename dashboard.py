import base64
import json
import os
import textwrap
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

# All indicator and signal logic lives in strategy.py — no duplication here
from strategy import add_chart_indicators, generate_signal, compute_indicators

load_dotenv()

GECKO_API_KEY   = os.getenv("GECKO_API", "")
KALSHI_KEY_ID   = os.getenv("KALSHI_API_KEY", "")
KALSHI_PRIV_RAW = os.getenv("KALSHI_PRIV", "")
GECKO_BASE      = "https://api.coingecko.com/api/v3"
KALSHI_BASE_URL = "https://external-api.kalshi.com"
KALSHI_API_PFX  = "/trade-api/v2"
STATE_FILE      = Path("trading_state.json")
CONFIG_FILE     = Path("trading_config.json")

DEFAULT_CONFIG = {
    "enabled":            False,
    "dry_run":            True,
    "series_ticker":      "KXBTCD",
    "max_contracts":      5,
    "max_open_risk_usd":  50.0,
    "stop_loss_pct":      0.40,
    "only_on_change":     True,
    "cooldown_minutes":   15,
    "loop_interval_sec":  60,
}

SIGNAL_COLORS = {
    "STRONG BUY":  "#00C853",
    "BUY":         "#69F0AE",
    "HOLD":        "#FFD600",
    "SELL":        "#FF5252",
    "STRONG SELL": "#D50000",
}


# ---------------------------------------------------------------------------
# Kalshi auth (fresh signature per call — fixes stale-timestamp retry bug)
# ---------------------------------------------------------------------------

def _load_kalshi_key():
    if not KALSHI_PRIV_RAW:
        return None
    raw_b64 = "".join(KALSHI_PRIV_RAW.split())
    lines   = textwrap.wrap(raw_b64, 64)
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        for hdr in ("RSA PRIVATE KEY", "PRIVATE KEY"):
            try:
                pem = (f"-----BEGIN {hdr}-----\n"
                       + "\n".join(lines)
                       + f"\n-----END {hdr}-----")
                return serialization.load_pem_private_key(
                    pem.encode(), password=None, backend=default_backend()
                )
            except Exception:
                pass
    except Exception as e:
        st.warning(f"Kalshi key load error: {e}")
    return None


_KAL_KEY = _load_kalshi_key()


def _kalshi_headers(path: str) -> dict:
    """Generate fresh signed headers. Called individually per request."""
    if not _KAL_KEY or not KALSHI_KEY_ID:
        return {}
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    ts  = str(int(time.time() * 1000))
    msg = (ts + "GET" + KALSHI_API_PFX + path).encode()
    sig = _KAL_KEY.sign(msg, padding.PKCS1v15(), hashes.SHA256())
    return {
        "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def fetch_btc_price() -> dict:
    url = (f"{GECKO_BASE}/simple/price?ids=bitcoin&vs_currencies=usd"
           f"&include_24hr_change=true&include_24hr_vol=true"
           f"&x_cg_demo_api_key={GECKO_API_KEY}")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()["bitcoin"]


@st.cache_data(ttl=120)
def fetch_ohlc(days: int = 7) -> pd.DataFrame:
    url = (f"{GECKO_BASE}/coins/bitcoin/ohlc?vs_currency=usd"
           f"&days={days}&x_cg_demo_api_key={GECKO_API_KEY}")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["ts", "open", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df


@st.cache_data(ttl=120)
def fetch_price_history(days: int = 30) -> pd.DataFrame:
    """Returns hourly price + volume DataFrame for indicator computation."""
    url = (f"{GECKO_BASE}/coins/bitcoin/market_chart?vs_currency=usd"
           f"&days={days}&interval=hourly&x_cg_demo_api_key={GECKO_API_KEY}")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data    = r.json()
    prices  = pd.DataFrame(data["prices"],        columns=["ts", "price"])
    volumes = pd.DataFrame(data["total_volumes"],  columns=["ts", "volume"])
    df      = prices.merge(volumes, on="ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df


@st.cache_data(ttl=60)
def fetch_kalshi_markets() -> list:
    """Try KXBTCD series first; fall back to keyword search. Fresh signature each attempt."""
    path = "/markets"
    for query in ("?series_ticker=KXBTCD&limit=20&status=open",
                  "?limit=50&status=open"):
        headers = _kalshi_headers(path)   # fresh signature every attempt
        if not headers:
            break
        try:
            r = requests.get(
                KALSHI_BASE_URL + KALSHI_API_PFX + path + query,
                headers=headers, timeout=10,
            )
            if r.status_code == 200:
                markets = r.json().get("markets", [])
                if "series_ticker=KXBTCD" in query:
                    if markets:
                        return markets
                else:
                    btc = [m for m in markets
                           if "btc"     in m.get("ticker", "").lower()
                           or "bitcoin" in m.get("title",  "").lower()]
                    if btc:
                        return btc
        except requests.RequestException:
            pass
    return []


# ---------------------------------------------------------------------------
# Config / state helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text()))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def load_trader_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="BTC Dashboard", layout="wide", page_icon="₿")
st_autorefresh(interval=30_000, key="btc_refresh")

st.title("₿  BTC/USD Live Dashboard")
st.caption(f"Auto-refreshes every 30 s  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

with st.spinner("Loading market data…"):
    price_data   = fetch_btc_price()
    ohlc_df      = fetch_ohlc(days=7)
    hist_df      = fetch_price_history(days=30)
    kalshi_mkts  = fetch_kalshi_markets()
    trader_state = load_trader_state()
    config       = load_config()

# Build full indicator series (from strategy.py — single source of truth)
hist_df = add_chart_indicators(hist_df)

# Get current signal using latest indicator values
ind_latest = compute_indicators(hist_df["price"])
signal     = generate_signal(ind_latest)
sig_label  = signal["label"]
sig_color  = SIGNAL_COLORS.get(sig_label, "#FFD600")
sig_score  = signal["bull_score"]

price   = price_data["usd"]
chg_24h = price_data.get("usd_24h_change", 0.0)
vol_24h = price_data.get("usd_24h_vol",    0.0)
cur_rsi = ind_latest["rsi"]

# ---------------------------------------------------------------------------
# Metric row
# ---------------------------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("BTC Price",  f"${price:,.2f}",      f"{chg_24h:+.2f}%")
c2.metric("24h Volume", f"${vol_24h / 1e9:.2f}B")

rsi_zone = "Overbought" if cur_rsi > 70 else ("Oversold" if cur_rsi < 30 else "Neutral")
c3.metric("RSI (14)", f"{cur_rsi:.1f}", rsi_zone)

macd_val  = ind_latest["macd"]
macd_sig  = ind_latest["macd_signal"]
macd_str  = f"{'▲' if macd_val > macd_sig else '▼'} {macd_val:+.1f}"
c4.markdown(
    f"<div style='text-align:center;padding:12px 6px;border-radius:8px;"
    f"background:{sig_color}18;border:2px solid {sig_color}'>"
    f"<div style='font-size:0.75em;color:#aaa;margin-bottom:2px'>Strategy Signal</div>"
    f"<div style='font-size:1.5em;font-weight:700;color:{sig_color}'>{sig_label}</div>"
    f"<div style='font-size:0.75em;color:#aaa'>score {sig_score:+d} · MACD {macd_str}</div>"
    f"</div>",
    unsafe_allow_html=True,
)

# Kalshi card: show the side relevant to our current signal direction
if kalshi_mkts:
    m0 = kalshi_mkts[0]
    if signal["direction"] == "up":
        mkt_price = m0.get("yes_ask")
        mkt_label = "YES ask"
    elif signal["direction"] == "down":
        mkt_price = m0.get("no_ask")
        mkt_label = "NO ask"
    else:
        mkt_price = m0.get("yes_ask") or m0.get("last_price")
        mkt_label = "YES ask"
    ticker_short = m0.get("ticker", "BTC")[:16]
    c5.metric(
        f"Kalshi: {ticker_short}",
        f"{mkt_price}¢  ({mkt_price}% prob)" if isinstance(mkt_price, (int, float)) else "N/A",
        mkt_label,
    )
else:
    c5.metric("Kalshi", "N/A", "no open markets")

st.divider()

# ---------------------------------------------------------------------------
# Main chart: Price + BB + MAs | RSI | MACD
# ---------------------------------------------------------------------------

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.22, 0.23],
    vertical_spacing=0.03,
    subplot_titles=(
        "BTC/USD — 7-day OHLC  ·  Bollinger Bands (20, 2σ)  ·  Moving Averages",
        "RSI (14)",
        "MACD (12 / 26 / 9)",
    ),
    specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
)

# ── Row 1: Candlesticks ───────────────────────────────────────────────────
fig.add_trace(go.Candlestick(
    x=ohlc_df["ts"],
    open=ohlc_df["open"], high=ohlc_df["high"],
    low=ohlc_df["low"],   close=ohlc_df["close"],
    name="OHLC",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
    showlegend=True,
), row=1, col=1, secondary_y=False)

# ── Row 1: Bollinger Bands (filled channel) ───────────────────────────────
fig.add_trace(go.Scatter(
    x=hist_df["ts"], y=hist_df["bb_upper"], name="BB Upper",
    line=dict(color="rgba(100,149,237,0.6)", width=1, dash="dot"),
    showlegend=True,
), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(
    x=hist_df["ts"], y=hist_df["bb_lower"], name="BB Lower",
    line=dict(color="rgba(100,149,237,0.6)", width=1, dash="dot"),
    fill="tonexty",
    fillcolor="rgba(100,149,237,0.05)",
    showlegend=True,
), row=1, col=1, secondary_y=False)

fig.add_trace(go.Scatter(
    x=hist_df["ts"], y=hist_df["bb_mid"], name="BB Mid (SMA20)",
    line=dict(color="rgba(100,149,237,0.4)", width=1),
    showlegend=False,
), row=1, col=1, secondary_y=False)

# ── Row 1: Moving averages ────────────────────────────────────────────────
for col_name, label, color, dash in [
    ("sma50",  "SMA 50",  "#42A5F5", "solid"),
    ("ema20",  "EMA 20",  "#AB47BC", "dash"),
    ("ema200", "EMA 200", "#FF7043", "dot"),
]:
    fig.add_trace(go.Scatter(
        x=hist_df["ts"], y=hist_df[col_name], name=label,
        line=dict(color=color, width=1.5, dash=dash),
    ), row=1, col=1, secondary_y=False)

# ── Row 1: Volume bars (secondary y) ────────────────────────────────────
vol_colors = [
    "#26a69a" if close >= open_ else "#ef5350"
    for close, open_ in zip(ohlc_df["close"], ohlc_df["open"])
]
fig.add_trace(go.Bar(
    x=ohlc_df["ts"], y=ohlc_df["close"] - ohlc_df["close"],  # placeholder for alignment
    name="", showlegend=False, visible=False,
), row=1, col=1, secondary_y=True)  # dummy to initialise secondary axis

fig.add_trace(go.Bar(
    x=hist_df["ts"], y=hist_df["volume"],
    name="Volume",
    marker_color="rgba(150,150,150,0.25)",
    showlegend=True,
), row=1, col=1, secondary_y=True)

# ── Row 2: RSI ────────────────────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=hist_df["ts"], y=hist_df["rsi14"], name="RSI 14",
    line=dict(color="#66BB6A", width=2),
), row=2, col=1)

fig.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.07, row=2, col=1, line_width=0)
fig.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.07, row=2, col=1, line_width=0)
for y_val, clr in [(70, "red"), (50, "grey"), (30, "green")]:
    fig.add_hline(
        y=y_val,
        line_dash="dash" if y_val != 50 else "dot",
        line_color=clr, line_width=1,
        row=2, col=1,
    )

# ── Row 3: MACD ───────────────────────────────────────────────────────────
hist_colors = [
    "#26a69a" if v >= 0 else "#ef5350"
    for v in hist_df["macd_hist"].fillna(0)
]
fig.add_trace(go.Bar(
    x=hist_df["ts"], y=hist_df["macd_hist"],
    name="MACD Hist",
    marker_color=hist_colors,
    opacity=0.6,
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=hist_df["ts"], y=hist_df["macd"], name="MACD",
    line=dict(color="#42A5F5", width=1.5),
), row=3, col=1)

fig.add_trace(go.Scatter(
    x=hist_df["ts"], y=hist_df["macd_signal"], name="Signal",
    line=dict(color="#FF7043", width=1.5),
), row=3, col=1)

fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1, row=3, col=1)

# ── Layout ────────────────────────────────────────────────────────────────
fig.update_layout(
    height=820,
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=70, b=10),
    bargap=0,
)
fig.update_yaxes(title_text="Price (USD)", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Volume",      row=1, col=1, secondary_y=True,
                 showgrid=False, tickformat=".2s")
fig.update_yaxes(title_text="RSI",         row=2, col=1, range=[0, 100])
fig.update_yaxes(title_text="MACD",        row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Kalshi markets table
# ---------------------------------------------------------------------------

st.subheader("Kalshi BTC Markets")
if kalshi_mkts:
    rows = []
    for m in kalshi_mkts:
        ct = (m.get("close_time") or "")[:10]
        # Highlight the market our signal would trade
        would_trade = (
            signal["direction"] in ("up", "down")
            and m == kalshi_mkts[0]   # first = best by select_market logic
        )
        rows.append({
            "★":             "●" if would_trade else "",
            "Ticker":        m.get("ticker", ""),
            "Title":         m.get("title", ""),
            "Yes Ask (¢)":   m.get("yes_ask", "-"),
            "No Ask (¢)":    m.get("no_ask",  "-"),
            "Last (¢)":      m.get("last_price", "-"),
            "Volume":        f"{m.get('volume', 0):,}",
            "Closes":        ct or "-",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No open Kalshi BTC markets — check your API credentials.")

st.divider()

# ---------------------------------------------------------------------------
# Trading section
# ---------------------------------------------------------------------------

st.header("Automated Trading")

is_running = trader_state.get("is_running", False)
is_enabled = trader_state.get("enabled",    False)
is_dry_run = trader_state.get("dry_run",    True)

if   is_running and is_enabled and not is_dry_run:
    st.error("🔴  LIVE TRADING ACTIVE — real orders are being placed on Kalshi")
elif is_running and is_enabled and is_dry_run:
    st.warning("🟡  Trader running in DRY RUN — signals logged, no real orders sent")
elif is_running:
    st.info("⚪  Trader daemon is running but trading is disabled")
else:
    st.info("⚫  Trader daemon is not running.  Start it with:  `python trader.py`")

# Portfolio snapshot
tc1, tc2, tc3, tc4, tc5 = st.columns(5)
bal_cents = trader_state.get("balance_cents")
tc1.metric("Kalshi Balance",
           f"${bal_cents/100:.2f}" if bal_cents is not None else "—")

positions    = trader_state.get("active_positions", [])
total_unreal = sum((p.get("unrealized_pnl") or 0) for p in positions)
total_real   = sum((p.get("realized_pnl")   or 0) for p in positions)
tc2.metric("Open Positions", len(positions))
tc3.metric("Unrealized P&L", f"${total_unreal/100:.2f}" if positions else "—")
tc4.metric("Realized P&L",   f"${total_real/100:.2f}"   if positions else "—")

last_sig_time = trader_state.get("last_signal_time", "")
last_ord_time = trader_state.get("last_order_time",  "")
tc5.metric("Last Order",
           last_ord_time[:16].replace("T", " ") if last_ord_time else "Never")

# Open positions detail
if positions:
    st.subheader("Open Positions")
    pos_rows = []
    for p in positions:
        cost      = abs(p.get("total_cost",     0) or 0)
        unreal    =     p.get("unrealized_pnl", 0) or 0
        contracts = abs(p.get("position",       0) or 0)
        avg_cost  = round(cost / max(contracts, 1), 1)
        pnl_pct   = (unreal / cost * 100) if cost else 0
        pos_rows.append({
            "Ticker":         p.get("ticker", ""),
            "Contracts":      p.get("position", 0),
            "Avg Cost (¢)":   avg_cost,
            "Unrealized P&L": f"${unreal/100:.2f}  ({pnl_pct:+.1f}%)",
            "Realized P&L":   f"${(p.get('realized_pnl') or 0)/100:.2f}",
        })
    st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

# Recent orders
recent_orders = trader_state.get("recent_orders", [])
if recent_orders:
    st.subheader("Recent Orders")
    ord_rows = []
    for o in recent_orders[:20]:
        result = o.get("result")
        if isinstance(result, dict):
            status = result.get("order", {}).get("status", "sent")
        else:
            status = str(result) if result else "—"
        ord_rows.append({
            "Time":      (o.get("timestamp") or "")[:16].replace("T", " "),
            "Signal":    o.get("signal", ""),
            "Score":     f"{o.get('bull_score', 0):+d}",
            "Ticker":    o.get("ticker", ""),
            "Side":      o.get("side", "").upper(),
            "Count":     o.get("count", 0),
            "Price (¢)": o.get("price_cents", "—"),
            "Cost":      f"${o.get('cost_usd', 0):.2f}",
            "Status":    status,
            "Dry Run":   "✓" if o.get("dry_run") else "✗",
        })
    st.dataframe(pd.DataFrame(ord_rows), use_container_width=True, hide_index=True)

# Error log
errs = trader_state.get("errors", [])
if errs:
    with st.expander(f"⚠ {len(errs)} error(s) from last cycle"):
        for e in errs:
            st.text(e)

st.divider()

# ---------------------------------------------------------------------------
# Configuration panel
# ---------------------------------------------------------------------------

st.subheader("Trading Configuration")
st.caption(
    "Written to `trading_config.json`.  "
    "The trader daemon picks up changes on its next cycle — no restart needed."
)

with st.form("trading_config_form"):
    col_a, col_b = st.columns(2)

    with col_a:
        dry_run = st.checkbox(
            "Dry Run  (log signals — no real orders)",
            value=config.get("dry_run", True),
        )
        enabled = st.checkbox(
            "Enable live trading",
            value=config.get("enabled", False),
            disabled=dry_run,
            help="Only activates when Dry Run is unchecked",
        )
        only_on_change = st.checkbox(
            "Only trade on signal change",
            value=config.get("only_on_change", True),
            help="Prevents repeated orders when the signal is flat",
        )
        series_ticker = st.text_input(
            "Kalshi series ticker",
            value=config.get("series_ticker", "KXBTCD"),
        )

    with col_b:
        max_contracts = st.number_input(
            "Max contracts per trade",
            min_value=1, max_value=100,
            value=int(config.get("max_contracts", 5)),
            help="STRONG signal uses 100%, regular signal uses 50%.  Scaled down further in high-volatility.",
        )
        max_risk = st.number_input(
            "Max open risk (USD)",
            min_value=1.0, max_value=10_000.0, step=5.0,
            value=float(config.get("max_open_risk_usd", 50.0)),
            help="No new orders are placed once total cost basis exceeds this.",
        )
        stop_loss_pct = st.slider(
            "Stop-loss threshold (%)",
            min_value=5, max_value=80,
            value=int(config.get("stop_loss_pct", 0.40) * 100),
            help="Close a position when its unrealized loss exceeds this % of cost.",
        )
        cooldown_min = st.number_input(
            "Cooldown between orders (minutes)",
            min_value=1, max_value=1440,
            value=int(config.get("cooldown_minutes", 15)),
            help="Minimum gap between consecutive order placements.",
        )
        loop_interval = st.number_input(
            "Cycle interval (seconds)",
            min_value=30, max_value=3600,
            value=int(config.get("loop_interval_sec", 60)),
        )

    if enabled and not dry_run:
        st.warning(
            "⚠  LIVE TRADING will be enabled. Real money will be placed on Kalshi. "
            "Confirm your risk limits are correct before saving."
        )

    if st.form_submit_button("Save Configuration"):
        new_cfg = {
            "enabled":            bool(enabled and not dry_run),
            "dry_run":            bool(dry_run),
            "series_ticker":      series_ticker.strip().upper(),
            "max_contracts":      int(max_contracts),
            "max_open_risk_usd":  float(max_risk),
            "stop_loss_pct":      stop_loss_pct / 100.0,
            "only_on_change":     bool(only_on_change),
            "cooldown_minutes":   int(cooldown_min),
            "loop_interval_sec":  int(loop_interval),
        }
        save_config(new_cfg)
        st.success("Configuration saved. Trader daemon picks it up on its next cycle.")
        st.cache_data.clear()
