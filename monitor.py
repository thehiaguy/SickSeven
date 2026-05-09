"""
Short-term BTC Market Monitor.

Run with:  streamlit run monitor.py --server.port 8502

Displays real-time 1m / 5m / 15m Kraken candlestick charts with short-term
technical indicators, an LLM probability gauge, Fear & Greed Index, live
BTC news feed, and Kalshi market odds — all in one screen.

The LLM probability estimate is cached for 5 minutes to control API costs.
Market data refreshes every 15 seconds.
"""
import base64
import json
import os
import textwrap
import threading
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

from strategy import (
    add_short_term_chart_indicators,
    compute_short_term_indicators,
    generate_short_term_signal,
)

load_dotenv()

GECKO_API_KEY   = os.getenv("GECKO_API", "")
KALSHI_KEY_ID   = os.getenv("KALSHI_API_KEY", "")
KALSHI_PRIV_RAW = os.getenv("KALSHI_PRIV", "")
GECKO_BASE      = "https://api.coingecko.com/api/v3"
KRAKEN_BASE     = "https://api.kraken.com/0/public"
KALSHI_BASE_URL = "https://external-api.kalshi.com"
KALSHI_API_PFX  = "/trade-api/v2"
STATE_FILE      = Path("trading_state.json")

SIGNAL_COLORS = {
    "STRONG BUY":  "#00C853",
    "BUY":         "#69F0AE",
    "HOLD":        "#FFD600",
    "SELL":        "#FF5252",
    "STRONG SELL": "#D50000",
}

TIMEFRAMES = ["1m", "5m", "15m"]

# ---------------------------------------------------------------------------
# Kalshi auth
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
    except Exception:
        pass
    return None

_KAL_KEY = _load_kalshi_key()


def _kalshi_headers(path: str) -> dict:
    if not _KAL_KEY or not KALSHI_KEY_ID:
        return {}
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    ts  = str(int(time.time() * 1000))
    msg = (ts + "GET" + KALSHI_API_PFX + path).encode()
    sig = _KAL_KEY.sign(msg, padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.DIGEST_LENGTH,
    ), hashes.SHA256())
    return {
        "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

_TF_TO_KRAKEN_INTERVAL = {"1m": 1, "5m": 5, "15m": 15}


@st.cache_data(ttl=15)
def fetch_kraken_ohlcv(interval: str = "5m", limit: int = 200) -> pd.DataFrame:
    """Fetch Kraken public OHLCV — no API key required."""
    ki = _TF_TO_KRAKEN_INTERVAL.get(interval, 5)
    url = f"{KRAKEN_BASE}/OHLC?pair=XBTUSD&interval={ki}"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        if data.get("error"):
            raise ValueError(f"Kraken: {data['error']}")
        result = data["result"]
        pair_key = next(k for k in result if k != "last")
        rows = result[pair_key][-limit:]
        # Format: [time(s), open, high, low, close, vwap, volume, count]
        df = pd.DataFrame(rows, columns=[
            "time", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df["ts"] = pd.to_datetime(df["time"].astype(int), unit="s")
        return df[["ts", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        st.warning(f"Kraken data unavailable: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=10)
def fetch_composite_price() -> dict:
    """
    BRTI-approximate price from four constituent exchanges in parallel.
    Falls back to Kraken-only if the composite feed fails entirely.
    """
    try:
        from price_feed import get_composite_price
        return get_composite_price()
    except Exception:
        pass
    # Kraken-only fallback
    try:
        r = requests.get(f"{KRAKEN_BASE}/Ticker?pair=XBTUSD", timeout=8)
        r.raise_for_status()
        result = r.json()["result"]
        t = result[next(iter(result))]
        price = (float(t["a"][0]) + float(t["b"][0])) / 2
        return {"price": price, "sources": {"kraken": price}, "source_count": 1, "spread_pct": 0.0}
    except Exception:
        return {"price": 0.0, "sources": {}, "source_count": 0, "spread_pct": 0.0}


@st.cache_data(ttl=30)
def fetch_btc_ticker() -> dict:
    """24h stats from Kraken (high, low, volume, change %)."""
    try:
        r = requests.get(f"{KRAKEN_BASE}/Ticker?pair=XBTUSD", timeout=8)
        r.raise_for_status()
        data = r.json()
        result = data["result"]
        t = result[next(iter(result))]
        price = float(t["c"][0])
        open_ = float(t["o"])
        change_pct = (price - open_) / open_ * 100 if open_ else 0.0
        return {
            "price":      price,
            "change_pct": change_pct,
            "high_24h":   float(t["h"][1]),
            "low_24h":    float(t["l"][1]),
            "volume_24h": float(t["v"][1]),
        }
    except Exception:
        return {"price": 0, "change_pct": 0, "high_24h": 0, "low_24h": 0, "volume_24h": 0}


@st.cache_data(ttl=60)
def fetch_kalshi_markets() -> list:
    path = "/markets"
    for query in ("?series_ticker=KXBTCD&limit=20&status=open",
                  "?limit=50&status=open"):
        headers = _kalshi_headers(path)
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


@st.cache_data(ttl=120)
def fetch_fear_greed() -> dict:
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        r.raise_for_status()
        d = r.json()["data"][0]
        return {"value": int(d["value"]), "classification": d["value_classification"]}
    except Exception:
        return {"value": 50, "classification": "Neutral"}


class _NewsCache:
    def __init__(self):
        self._headlines: list = []
        self._lock = threading.Lock()

    def set(self, headlines: list):
        with self._lock:
            self._headlines = headlines

    def get(self) -> list:
        with self._lock:
            return list(self._headlines)


@st.cache_resource
def _start_news_background_thread() -> _NewsCache:
    """
    Starts a single background thread that fetches news every 5 minutes.
    st.cache_resource ensures this runs exactly once regardless of how many
    times Streamlit reruns the script — the thread and cache persist forever.
    """
    cache = _NewsCache()

    def _worker():
        while True:
            try:
                from news_fetcher import fetch_all_headlines
                cache.set(fetch_all_headlines(max_age_hours=3, max_total=25))
            except Exception:
                pass
            time.sleep(300)  # refresh every 5 minutes

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return cache


_news_cache = _start_news_background_thread()


@st.cache_data(ttl=300)  # LLM estimate cached 5 minutes
def fetch_llm_probability(bull_score: int, price: float) -> dict:
    """
    Run the full LLM probability pipeline.
    Cached for 5 minutes to control Claude API costs.
    bull_score and price are in the cache key so a big signal change forces refresh.
    """
    try:
        from probability_model import get_combined_probability
        from strategy import compute_indicators
        import numpy as np

        # We pass a minimal indicators dict — the model fetches fresh data internally
        # Use price as a stand-in for the quick indicators snapshot
        fake_ind = {
            "price": price, "sma20": price, "sma50": price,
            "ema20": price, "ema200": price,
            "rsi": 50.0, "macd": 0.0, "macd_signal": 0.0,
            "macd_hist": 0.0, "bb_upper": price, "bb_lower": price,
            "bb_pct_b": 0.5, "atr_pct": 0.003,
        }
        markets = fetch_kalshi_markets()
        return get_combined_probability(fake_ind, bull_score, markets)
    except Exception as e:
        return {"combined_prob": 0.5, "llm_error": str(e), "llm_detail": None}


def load_trader_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

def build_chart(df: pd.DataFrame, timeframe: str, signal: dict) -> go.Figure:
    df = add_short_term_chart_indicators(df, timeframe)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.03,
        subplot_titles=(
            f"BTC/USDT  {timeframe}  ·  Bollinger Bands  ·  EMAs  ·  VWAP",
            f"RSI",
            f"MACD",
        ),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df["ts"],
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1, secondary_y=False)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["bb_upper"], name="BB Upper",
        line=dict(color="rgba(100,149,237,0.5)", width=1, dash="dot"), showlegend=False,
    ), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["bb_lower"], name="BB",
        line=dict(color="rgba(100,149,237,0.5)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(100,149,237,0.05)",
    ), row=1, col=1, secondary_y=False)

    # EMAs
    for col, label, color, dash in [
        ("ema_fast", f"EMA fast", "#FFA726", "solid"),
        ("ema_slow", f"EMA slow", "#42A5F5", "solid"),
    ]:
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df[col], name=label,
            line=dict(color=color, width=1.5, dash=dash),
        ), row=1, col=1, secondary_y=False)

    # VWAP
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["vwap"], name="VWAP",
        line=dict(color="#E040FB", width=1.5, dash="dash"),
    ), row=1, col=1, secondary_y=False)

    # Volume bars
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["ts"], y=df["volume"], name="Volume",
        marker_color=vol_colors, opacity=0.4, showlegend=False,
    ), row=1, col=1, secondary_y=True)

    # Signal line on chart
    if signal["direction"] != "neutral":
        last_price = df["close"].iloc[-1]
        sig_color  = SIGNAL_COLORS.get(signal["label"], "#FFD600")
        fig.add_hline(
            y=last_price, line_dash="dash",
            line_color=sig_color, line_width=1.5,
            annotation_text=signal["label"],
            annotation_font_color=sig_color,
            row=1, col=1,
        )

    # RSI
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["rsi"], name="RSI",
        line=dict(color="#66BB6A", width=2),
    ), row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.07, row=2, col=1, line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.07, row=2, col=1, line_width=0)
    for y, c, d in [(70, "red", "dash"), (50, "grey", "dot"), (30, "green", "dash")]:
        fig.add_hline(y=y, line_dash=d, line_color=c, line_width=1, row=2, col=1)

    # MACD
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350"
                   for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(
        x=df["ts"], y=df["macd_hist"], name="MACD Hist",
        marker_color=hist_colors, opacity=0.6,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["macd"], name="MACD",
        line=dict(color="#42A5F5", width=1.5),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["macd_signal"], name="Signal",
        line=dict(color="#FF7043", width=1.5),
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1, row=3, col=1)

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", y=1.03, x=1, xanchor="right"),
        margin=dict(l=10, r=10, t=60, b=10),
        bargap=0,
    )
    fig.update_yaxes(title_text="Price",  row=1, col=1, secondary_y=False)
    fig.update_yaxes(showgrid=False,      row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RSI",    row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD",   row=3, col=1)
    return fig


# ---------------------------------------------------------------------------
# Probability gauge
# ---------------------------------------------------------------------------

def build_gauge(prob: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 36}},
        delta={"reference": 50, "valueformat": "+.1f"},
        title={"text": f"<b>{label}</b><br><span style='font-size:0.8em'>BTC up in ~4h</span>"},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": color, "thickness": 0.25},
            "steps": [
                {"range": [0,  28], "color": "#D50000"},
                {"range": [28, 40], "color": "#FF5252"},
                {"range": [40, 60], "color": "#424242"},
                {"range": [60, 72], "color": "#69F0AE"},
                {"range": [72, 100],"color": "#00C853"},
            ],
            "threshold": {
                "line":  {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": prob * 100,
            },
        },
    ))
    fig.update_layout(
        height=280,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="BTC Monitor", layout="wide", page_icon="📡")
st_autorefresh(interval=15_000, key="monitor_refresh")

st.title("📡  BTC Short-Term Market Monitor")
st.caption(f"Kraken data · refreshes every 15 s  —  {datetime.now().strftime('%H:%M:%S')}")

# Timeframe selector
col_tf, col_spacer = st.columns([3, 9])
with col_tf:
    tf = st.radio("Timeframe", TIMEFRAMES, horizontal=True, index=1)

# Fetch all data
with st.spinner(""):
    df_ohlcv      = fetch_kraken_ohlcv(tf, limit=200)
    composite     = fetch_composite_price()
    ticker        = fetch_btc_ticker()
    kalshi_mkts   = fetch_kalshi_markets()
    fng           = fetch_fear_greed()
    headlines     = _news_cache.get()
    trader_state  = load_trader_state()

if df_ohlcv.empty:
    st.error("Could not fetch Kraken data. Check your internet connection.")
    st.stop()

# Compute short-term indicators + signal
ind    = compute_short_term_indicators(df_ohlcv, tf)
signal = generate_short_term_signal(ind)
sig_color = SIGNAL_COLORS.get(signal["label"], "#FFD600")

# LLM probability (cached 5 min — only fetches when bull_score changes significantly)
prob_bucket = round(signal["bull_score"])  # cache key granularity
llm_result  = fetch_llm_probability(prob_bucket, round(ind["price"], -2))
combined_prob = llm_result.get("combined_prob", 0.5)
llm_label, _, _ = (
    ("STRONG BUY", "up", 1.0) if combined_prob >= 0.72 else
    ("BUY",        "up", 0.5) if combined_prob >= 0.60 else
    ("STRONG SELL","down",1.0) if combined_prob <= 0.28 else
    ("SELL",       "down",0.5) if combined_prob <= 0.40 else
    ("HOLD",       "neutral", 0.0)
)
llm_color = SIGNAL_COLORS.get(llm_label, "#FFD600")

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------

m1, m2, m3, m4, m5, m6 = st.columns(6)
_src_count  = composite.get("source_count", 0)
_spread     = composite.get("spread_pct", 0.0)
_src_label  = f"{_src_count}/4 exchanges  spread {_spread:.3f}%"
m1.metric("BRTI ≈ Price", f"${composite['price']:,.2f}",
          f"{ticker['change_pct']:+.2f}%  ({_src_label})")
m2.metric("24h High",    f"${ticker['high_24h']:,.0f}")
m3.metric("24h Low",     f"${ticker['low_24h']:,.0f}")
m4.metric(f"RSI ({tf})", f"{ind['rsi']:.1f}",
          "Overbought" if ind["rsi"] > 70 else ("Oversold" if ind["rsi"] < 30 else "Neutral"))
m5.metric("VWAP",        f"${ind['vwap']:,.2f}",
          f"{'Above' if ind['price'] > ind['vwap'] else 'Below'} VWAP")
fng_delta = "↑ Improving" if fng.get("trend") == "improving" else ("↓ Worsening" if fng.get("trend") == "worsening" else "")
m6.metric("Fear & Greed", f"{fng['value']}/100",  fng["classification"])

st.divider()

# ---------------------------------------------------------------------------
# Main layout: chart | signal + LLM gauge | news + Kalshi
# ---------------------------------------------------------------------------

chart_col, right_col = st.columns([7, 3])

with chart_col:
    chart_fig = build_chart(df_ohlcv, tf, signal)
    st.plotly_chart(chart_fig, use_container_width=True)

with right_col:
    # Short-term technical signal
    st.markdown(
        f"<div style='text-align:center;padding:14px 8px;border-radius:10px;"
        f"background:{sig_color}18;border:2px solid {sig_color};margin-bottom:12px'>"
        f"<div style='font-size:0.75em;color:#aaa'>{tf} Technical Signal</div>"
        f"<div style='font-size:1.8em;font-weight:800;color:{sig_color}'>{signal['label']}</div>"
        f"<div style='font-size:0.8em;color:#aaa'>"
        f"score {signal['bull_score']:+d} · RSI {signal['rsi']:.0f} · "
        f"MACD {'▲' if signal['macd'] > signal['macd_signal'] else '▼'}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # LLM probability gauge
    gauge_fig = build_gauge(combined_prob, llm_label, llm_color)
    st.plotly_chart(gauge_fig, use_container_width=True)

    # LLM detail
    detail = llm_result.get("llm_detail")
    if detail:
        confidence = detail.get("confidence", "?")
        reasoning  = detail.get("reasoning", "")
        factors    = detail.get("key_factors", [])
        with st.expander(f"LLM reasoning  ({confidence} confidence)"):
            st.write(reasoning)
            if factors:
                st.write("**Key factors:**")
                for f in factors:
                    st.write(f"• {f}")
    elif llm_result.get("llm_error"):
        st.caption(f"⚠ LLM unavailable: {llm_result['llm_error'][:80]}")

    llm_ts = llm_result.get("timestamp", "")
    if llm_ts:
        st.caption(f"LLM estimate as of {llm_ts[:16].replace('T',' ')} UTC  (refreshes every 5 min)")

    # Kalshi markets
    st.subheader("Kalshi Markets")
    if kalshi_mkts:
        for m in kalshi_mkts[:6]:
            yes = m.get("yes_ask", "-")
            no  = m.get("no_ask",  "-")
            title_short = (m.get("title") or m.get("ticker", ""))[:40]
            st.markdown(
                f"<div style='font-size:0.8em;padding:4px 0;border-bottom:1px solid #333'>"
                f"<b style='color:#aaa'>{title_short}</b><br>"
                f"YES <b style='color:#69F0AE'>{yes}¢</b>  ·  "
                f"NO <b style='color:#FF5252'>{no}¢</b>  ·  "
                f"vol {m.get('volume',0):,}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No open Kalshi BTC markets")

st.divider()

# ---------------------------------------------------------------------------
# News feed
# ---------------------------------------------------------------------------

news_col, macro_col = st.columns([7, 3])

with news_col:
    st.subheader(f"BTC News  ({len(headlines)} headlines, last 3h)")
    if headlines:
        for h in headlines[:20]:
            age  = f"{h['age_minutes']:.0f}m" if h["age_minutes"] < 60 else f"{h['age_minutes']/60:.1f}h"
            src  = h["source"]
            url  = h.get("url", "#")
            title = h["title"]
            st.markdown(
                f"<div style='padding:5px 0;border-bottom:1px solid #2a2a2a;font-size:0.85em'>"
                f"<span style='color:#666'>[{src} · {age} ago]</span>  "
                f"<a href='{url}' target='_blank' style='color:#90CAF9;text-decoration:none'>{title}</a>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No recent headlines loaded (news_fetcher running…)")

with macro_col:
    st.subheader("Macro Snapshot")

    # Fear & Greed gauge
    fng_val = fng["value"]
    fng_color = (
        "#D50000" if fng_val <= 25 else
        "#FF5252" if fng_val <= 45 else
        "#FFD600" if fng_val <= 55 else
        "#69F0AE" if fng_val <= 75 else
        "#00C853"
    )
    st.markdown(
        f"<div style='text-align:center;padding:14px;border-radius:8px;"
        f"background:{fng_color}18;border:1px solid {fng_color};margin-bottom:12px'>"
        f"<div style='font-size:0.75em;color:#aaa'>Fear & Greed Index</div>"
        f"<div style='font-size:2.5em;font-weight:800;color:{fng_color}'>{fng_val}</div>"
        f"<div style='font-size:0.85em;color:{fng_color}'>{fng['classification']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Trump tweet signal
    st.subheader("Trump Signal")
    try:
        from trump_watcher import get_trump_signal
        trump_sig = get_trump_signal(max_age_minutes=60)
    except Exception:
        trump_sig = None

    if trump_sig:
        imp     = trump_sig.get("impact", "neutral")
        urgency = trump_sig.get("urgency", "low")
        adj     = trump_sig.get("probability_adjustment", 0.0)
        ts_raw  = trump_sig.get("timestamp", "")
        try:
            ts_dt  = datetime.fromisoformat(ts_raw)
            age_m  = (datetime.now(timezone.utc) - ts_dt.replace(tzinfo=timezone.utc if ts_dt.tzinfo is None else ts_dt.tzinfo)).total_seconds() / 60
            age_str = f"{age_m:.0f}m ago"
        except Exception:
            age_str = ""

        imp_color = (
            "#00C853" if imp in ("btc_bullish", "usd_bearish") else
            "#D50000" if imp in ("btc_bearish", "usd_bullish") else
            "#FFD600"
        )
        urgency_color = {"high": "#D50000", "medium": "#FFD600", "low": "#aaa"}.get(urgency, "#aaa")
        st.markdown(
            f"<div style='padding:10px;border-radius:8px;border:1px solid {imp_color};"
            f"background:{imp_color}18;margin-bottom:8px'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
            f"<span style='font-weight:700;color:{imp_color};font-size:0.85em'>{imp.replace('_',' ').upper()}</span>"
            f"<span style='color:{urgency_color};font-size:0.75em'>{urgency.upper()} · {age_str}</span>"
            f"</div>"
            f"<div style='font-size:0.8em;color:#ccc'>{trump_sig.get('reasoning','')}</div>"
            f"<div style='font-size:0.75em;color:#888;margin-top:4px;font-style:italic'>"
            f"\"{trump_sig.get('tweet_text','')[:120]}...\"</div>"
            f"<div style='font-size:0.85em;color:{imp_color};margin-top:4px'>"
            f"Prob adj: {adj:+.0%}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No recent Trump signal (watcher not running or no new tweets)")

    # Trader daemon status
    st.subheader("Trader Daemon")
    is_running = trader_state.get("is_running", False)
    is_enabled = trader_state.get("enabled",    False)
    is_dry_run = trader_state.get("dry_run",    True)
    last_signal = trader_state.get("last_signal", "—")

    if is_running and is_enabled and not is_dry_run:
        st.error("🔴 LIVE TRADING")
    elif is_running and is_dry_run:
        st.warning("🟡 Dry Run")
    elif is_running:
        st.info("⚪ Running / disabled")
    else:
        st.info("⚫ Stopped")

    st.metric("Daemon signal", last_signal)
    bal = trader_state.get("balance_cents")
    if bal is not None:
        st.metric("Balance", f"${bal/100:.2f}")
    positions = trader_state.get("active_positions", [])
    st.metric("Open positions", len(positions))
