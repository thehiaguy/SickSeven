"""
Signal generation, indicator computation, and order decision logic.

Imported by trader.py, dashboard.py, monitor.py, and probability_model.py.
All indicator logic lives here — never duplicate it elsewhere.
"""
import math
import re
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core indicator functions
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    # When loss == 0 (all-gain run), RS is infinite → RSI = 100 (not NaN)
    rs = np.where(loss == 0, np.inf, gain / loss)
    return pd.Series(100 - (100 / (1 + rs)), index=series.index)


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast    = series.ewm(span=fast,          adjust=False).mean()
    ema_slow    = series.ewm(span=slow,          adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, mid, lower) Bollinger Bands."""
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ---------------------------------------------------------------------------
# Indicator snapshot (latest values — used by trader.py)
# ---------------------------------------------------------------------------

def compute_indicators(prices: pd.Series) -> dict:
    """
    Return a dict of the most recent value for every indicator.
    Pass this to generate_signal().
    """
    sma20  = prices.rolling(20).mean()
    sma50  = prices.rolling(50).mean()
    ema20  = prices.ewm(span=20,  adjust=False).mean()
    ema200 = prices.ewm(span=200, adjust=False).mean()

    rsi14            = compute_rsi(prices, 14)
    macd_l, macd_s, macd_h = compute_macd(prices)
    bb_upper, bb_mid, bb_lower = compute_bollinger(prices)

    # Bollinger %B: 0 = price at lower band, 1 = price at upper band
    # Clamp to [0,1] so near-zero bandwidth doesn't produce extreme values that
    # falsely trigger the band-extreme scoring.
    bb_width = bb_upper - bb_lower
    bb_pct_b = ((prices - bb_lower) / bb_width.replace(0, np.nan)).clip(0.0, 1.0)

    # Volatility proxy: rolling std of log returns over 14 periods
    log_ret = np.log(prices / prices.shift(1))
    valid_std = log_ret.rolling(14).std().dropna()
    atr_pct = float(valid_std.iloc[-1]) if not valid_std.empty else 0.003
    if math.isnan(atr_pct) or atr_pct <= 0:
        atr_pct = 0.003

    def _safe(s: pd.Series, fallback: float = 0.0) -> float:
        arr = s.to_numpy()
        if len(arr) == 0:
            return fallback
        v = float(arr[-1])
        return fallback if math.isnan(v) else v

    return {
        "price":       _safe(prices),
        "sma20":       _safe(sma20),
        "sma50":       _safe(sma50),
        "ema20":       _safe(ema20),
        "ema200":      _safe(ema200),
        "rsi":         _safe(rsi14, 50.0),
        "macd":        _safe(macd_l),
        "macd_signal": _safe(macd_s),
        "macd_hist":   _safe(macd_h),
        "bb_upper":    _safe(bb_upper),
        "bb_lower":    _safe(bb_lower),
        "bb_pct_b":    _safe(bb_pct_b, 0.5),
        "atr_pct":     atr_pct,
    }


# ---------------------------------------------------------------------------
# Full series for charting (used by dashboard.py)
# ---------------------------------------------------------------------------

def add_chart_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all indicator columns to a DataFrame that has a 'price' column.
    The DataFrame is returned with new columns added in-place on a copy.
    """
    df = df.copy()
    p = df["price"]

    df["sma20"]  = p.rolling(20).mean()
    df["sma50"]  = p.rolling(50).mean()
    df["ema20"]  = p.ewm(span=20,  adjust=False).mean()
    df["ema200"] = p.ewm(span=200, adjust=False).mean()

    df["rsi14"] = compute_rsi(p, 14)

    macd_l, macd_s, macd_h = compute_macd(p)
    df["macd"]        = macd_l
    df["macd_signal"] = macd_s
    df["macd_hist"]   = macd_h

    bb_upper, bb_mid, bb_lower = compute_bollinger(p)
    df["bb_upper"] = bb_upper
    df["bb_mid"]   = bb_mid
    df["bb_lower"] = bb_lower

    return df


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signal(ind: dict) -> dict:
    """
    Five-factor signal engine.  Returns label, direction, strength, and
    the raw bull_score so callers can reason about how close to a boundary we are.

    Factors and their max contribution (each can add OR subtract):
      RSI          : ±2  (oversold/overbought)
      MACD         : ±2  (line vs signal, and position vs zero)
      MA crossover : ±1  (SMA20 vs SMA50 golden/death cross)
      Long trend   : ±1  (price vs EMA200)
      Bollinger    : ±1  (price at band extremes)
    ─────────────────────
    Total range    : ±7

    Score → signal:
      ≥ 4  : STRONG BUY
      2–3  : BUY
      -1–1 : HOLD
      -2–-3: SELL
      ≤ -4 : STRONG SELL
    """
    rsi      = ind["rsi"]
    macd     = ind["macd"]
    macd_sig = ind["macd_signal"]
    price    = ind["price"]
    sma20    = ind["sma20"]
    sma50    = ind["sma50"]
    ema200   = ind["ema200"]
    bb_pct_b = ind["bb_pct_b"]  # 0 = at lower band, 1 = at upper band

    score = 0

    # 1. RSI (±2)
    if rsi < 30:    score += 2
    elif rsi < 40:  score += 1
    elif rsi > 70:  score -= 2
    elif rsi > 60:  score -= 1

    # 2. MACD (±2)
    #    First point: is MACD above or below its signal line?
    #    Second point (bonus): is the MACD line itself above or below zero?
    if macd > macd_sig:
        score += 1
        if macd > 0:   score += 1   # confirmed: above zero
    else:
        score -= 1
        if macd < 0:   score -= 1   # confirmed: below zero

    # 3. Medium-term MA crossover (±1)
    score += 1 if sma20 > sma50 else -1

    # 4. Long-term trend filter — EMA200 (±1)
    #    Only trade with the dominant trend.
    score += 1 if price > ema200 else -1

    # 5. Bollinger Band extremes (±1)
    #    Near lower band = oversold pressure.  Near upper = extended.
    if bb_pct_b <= 0.05:    score += 1
    elif bb_pct_b >= 0.95:  score -= 1

    # Map to label
    if score >= 4:
        label, direction, strength = "STRONG BUY",  "up",      1.0
    elif score >= 2:
        label, direction, strength = "BUY",          "up",      0.5
    elif score <= -4:
        label, direction, strength = "STRONG SELL",  "down",    1.0
    elif score <= -2:
        label, direction, strength = "SELL",          "down",    0.5
    else:
        label, direction, strength = "HOLD",          "neutral", 0.0

    # RSI ceiling: never enter bullish when RSI is extreme (>72)
    if direction == "up" and rsi > 72:
        label, direction, strength = "HOLD", "neutral", 0.0

    return {
        "label":      label,
        "direction":  direction,
        "strength":   strength,
        "bull_score": score,
        "rsi":        rsi,
        "macd":       macd,
        "macd_hist":  ind["macd_hist"],
        "atr_pct":    ind["atr_pct"],
    }


# ---------------------------------------------------------------------------
# Options Greeks (Black-Scholes cash-or-nothing binary)
# ---------------------------------------------------------------------------

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _parse_strike(market: dict) -> Optional[float]:
    """
    Extract BTC strike price from a Kalshi market dict.
    Prefers the direct floor_strike field (new API), falls back to
    ticker regex (T95000 pattern) then title ($95,000).
    """
    if market.get("floor_strike") is not None:
        return float(market["floor_strike"])
    m = re.search(r'[Tt](\d{4,7})\b', market.get("ticker", ""))
    if m:
        return float(m.group(1))
    m = re.search(r'\$([0-9,]+)', market.get("title", ""))
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _parse_hours_to_expiry(market: dict) -> float:
    """Hours until market trading closes; returns 4.0 when field is missing."""
    for key in ("close_time", "expected_expiration_time", "expiration_time", "expiry_time"):
        val = market.get(key)
        if not val:
            continue
        try:
            if isinstance(val, (int, float)):
                exp = datetime.fromtimestamp(float(val), tz=timezone.utc)
            else:
                exp = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
            hours = (exp - datetime.now(timezone.utc)).total_seconds() / 3600
            return max(0.1, hours)
        except Exception:
            continue
    return 4.0


def compute_greeks(
    contract_price_cents: float,
    btc_price: float,
    strike: float,
    hours_to_expiry: float,
    annual_vol: float,
) -> dict:
    """
    Black-Scholes cash-or-nothing binary call Greeks (YES side, risk-free rate = 0).

    Parameters:
      contract_price_cents — current YES ask in cents (1–99)
      btc_price            — BTC spot price in USD
      strike               — option strike price in USD
      hours_to_expiry      — hours until settlement (> 0)
      annual_vol           — annualised BTC vol (e.g. 0.80 for 80%)

    Key outputs:
      delta          — cents per $1 BTC move (sensitivity)
      gamma          — cents per $1² BTC (rate of delta change)
      theta_per_hour — cents/hr the contract earns (+ITM) or loses (−OTM) to time decay
      vega_per_1pct  — cents per +1% annual vol (negative near ATM: vol pushes price to 50¢)
      delta_per_1k   — practical: cents per $1,000 BTC move
      high_gamma     — bool: option moves >30¢ per $1k BTC (elevated regime risk)
      near_expiry    — bool: < 2 hours remaining (gamma spikes, avoid or size down hard)
      gamma_factor   — 0–1 position-sizing multiplier for compute_order
    """
    _empty: dict = dict(
        delta=0.0, gamma=0.0, theta_per_hour=0.0, vega_per_1pct=0.0,
        delta_per_1k=0.0, high_gamma=False, near_expiry=True,
        gamma_factor=0.0, hours_to_expiry=hours_to_expiry,
    )
    if hours_to_expiry <= 0 or annual_vol <= 0 or btc_price <= 0 or strike <= 0:
        return _empty

    T           = max(hours_to_expiry / 8760.0, 1e-7)
    sigma_sqrtT = annual_vol * math.sqrt(T)
    if sigma_sqrtT < 1e-8:
        return {**_empty, "near_expiry": True, "high_gamma": True}

    log_SK = math.log(btc_price / strike)
    d2     = (log_SK - 0.5 * annual_vol ** 2 * T) / sigma_sqrtT
    d1     = d2 + sigma_sqrtT
    phi    = _norm_pdf(d2)
    S      = btc_price

    # All Greeks scaled to cents (×100 converts [0,1] option value to cents)
    delta          = phi / (S * sigma_sqrtT) * 100
    gamma          = -phi * d1 / (S * S * annual_vol ** 2 * T) * 100
    theta_per_hour = phi * d1 / (2.0 * T) / 8760.0 * 100   # +ITM / −OTM
    vega_per_1pct  = -phi * d1 / annual_vol * 0.01 * 100

    delta_per_1k = delta * 1000
    near_expiry  = hours_to_expiry < 2.0
    high_gamma   = abs(delta_per_1k) > 30.0  # >30¢ per $1k BTC move = elevated risk

    # Sizing multiplier: reference 15¢/$1k = typical 4-hour near-ATM binary
    gamma_factor = min(1.0, 15.0 / max(abs(delta_per_1k), 0.5))
    if math.isnan(gamma_factor):
        gamma_factor = 0.3
    if near_expiry:
        gamma_factor = min(gamma_factor, 0.3)  # hard cap at 30% size near expiry

    return {
        "delta":           round(delta, 5),
        "gamma":           round(gamma, 8),
        "theta_per_hour":  round(theta_per_hour, 4),
        "vega_per_1pct":   round(vega_per_1pct, 4),
        "delta_per_1k":    round(delta_per_1k, 3),
        "high_gamma":      high_gamma,
        "near_expiry":     near_expiry,
        "gamma_factor":    round(gamma_factor, 3),
        "hours_to_expiry": round(hours_to_expiry, 2),
    }


# ---------------------------------------------------------------------------
# Market selection
# ---------------------------------------------------------------------------

def select_market(
    markets: list,
    direction: str,
    btc_price: float = 0.0,
    atr_pct: float = 0.003,
) -> Optional[dict]:
    """
    Pick the best Kalshi BTC market for the given direction.

    Scoring criteria:
    1. Relevant-side ask must be 15–85 cents (maximum edge range)
    2. Prefer markets closest to 50 cents (maximum uncertainty = maximum edge)
    3. Among similar odds, prefer highest volume (tightest spread, easiest exit)
    4. When btc_price is provided, apply Greek-based gamma_factor to down-weight
       contracts near expiry or with high delta sensitivity.

    The selected market dict will have a '_greeks' key when Greeks are available.
    """
    if not markets:
        return None

    annual_vol = atr_pct * math.sqrt(8760)

    scored = []
    for m in markets:
        yes_ask = m.get("yes_ask")
        no_ask  = m.get("no_ask")
        volume  = m.get("volume") or 0
        if yes_ask is None or no_ask is None:
            continue

        target = yes_ask if direction == "up" else no_ask
        if 20 <= target <= 80:
            proximity    = 1.0 - abs(target - 50) / 50.0
            vol_score    = math.log1p(volume)
            greek_factor = 1.0
            greeks: Optional[dict] = None

            if btc_price > 0:
                strike = _parse_strike(m)
                hours  = _parse_hours_to_expiry(m)
                if strike:
                    greeks       = compute_greeks(target, btc_price, strike, hours, annual_vol)
                    greek_factor = greeks["gamma_factor"]

            score  = (proximity * 0.6 + vol_score * 0.4) * greek_factor
            m_copy = dict(m, _greeks=greeks) if greeks else m
            scored.append((score, m_copy))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    return max(markets, key=lambda m: m.get("volume") or 0)


# ---------------------------------------------------------------------------
# Order sizing
# ---------------------------------------------------------------------------

def compute_order(
    signal: dict,
    market: dict,
    max_contracts: int,
    atr_pct: float = 0.003,
    btc_price: float = 0.0,
    use_greeks: bool = True,
    market_order: bool = False,
) -> Optional[dict]:
    """
    Convert signal + market into order parameters.
    Returns None when no trade should be placed (HOLD or missing price).

    Position sizing (three multipliers, all ≤ 1.0):
    - Signal strength: 1.0 for STRONG, 0.5 for regular BUY/SELL
    - Volatility factor: scales down when BTC vol exceeds the 0.3%/hr reference
    - Greek factor: scales down when delta/gamma is elevated (near expiry or ATM
      with little time left). Caps at 0.3 when < 2 hours to expiry.

    Greeks are sourced from the market dict's '_greeks' key (set by select_market)
    or computed fresh when btc_price is provided.
    """
    if signal["direction"] == "neutral" or signal["strength"] == 0:
        return None

    side = "yes" if signal["direction"] == "up" else "no"
    ask  = market.get("yes_ask" if side == "yes" else "no_ask")
    if ask is None:
        return None

    # Volatility-adjusted factor
    base_vol   = 0.003
    vol_factor = min(1.0, base_vol / max(atr_pct, base_vol * 0.1))

    # Greek-adjusted factor — skipped for short-duration contracts (e.g. KXBTC15M)
    # where near-zero T causes delta to blow up and gamma_factor to collapse to ~0.
    greeks = None
    if use_greeks:
        greeks = market.get("_greeks")
        if greeks is None and btc_price > 0:
            strike = _parse_strike(market)
            hours  = _parse_hours_to_expiry(market)
            if strike:
                annual_vol = atr_pct * math.sqrt(8760)
                greeks     = compute_greeks(ask, btc_price, strike, hours, annual_vol)

    greek_factor = greeks["gamma_factor"] if greeks else 1.0

    raw_count = max_contracts * signal["strength"] * vol_factor * greek_factor
    count     = max(1, round(raw_count))

    # Tighten stop-loss recommendation when near expiry
    suggested_stop_pct: Optional[float] = None
    if greeks and greeks.get("near_expiry"):
        suggested_stop_pct = 0.20

    greek_note = ""
    if greeks:
        greek_note = (
            f" | Δ={greeks['delta_per_1k']:.1f}¢/$1k"
            + (" [near-expiry]" if greeks["near_expiry"] else "")
        )

    if market_order:
        # Kalshi requires a price even for aggressive taker orders.
        # Ask + 5¢ (capped at 99¢) crosses the spread and fills immediately.
        limit_price = min(99, max(1, ask + 5))
        cost_usd    = round(count * limit_price / 100, 2)
        return {
            "ticker":             market["ticker"],
            "action":             "buy",
            "side":               side,
            "count":              count,
            "order_type":         "limit",
            "price_cents":        limit_price,
            "cost_usd":           cost_usd,
            "suggested_stop_pct": suggested_stop_pct,
            "greeks":             greeks,
            "rationale": (
                f"{signal['label']} | RSI={signal['rsi']:.1f} "
                f"MACD={'▲' if signal.get('macd_hist', 0) > 0 else '▼'} "
                f"score={signal['bull_score']:+d} | "
                f"vol={vol_factor:.2f} greek={greek_factor:.2f} → "
                f"{count}×@ {limit_price}¢ [taker]{greek_note}"
            ),
        }

    # Limit 1 cent above ask — quick fill without chasing the book
    limit_price = min(99, max(1, ask + 1))
    cost_usd    = round(count * limit_price / 100, 2)

    return {
        "ticker":             market["ticker"],
        "action":             "buy",
        "side":               side,
        "count":              count,
        "order_type":         "limit",
        "price_cents":        limit_price,
        "cost_usd":           cost_usd,
        "suggested_stop_pct": suggested_stop_pct,
        "greeks":             greeks,
        "rationale": (
            f"{signal['label']} | RSI={signal['rsi']:.1f} "
            f"MACD={'▲' if signal.get('macd_hist', 0) > 0 else '▼'} "
            f"score={signal['bull_score']:+d} | "
            f"vol={vol_factor:.2f} greek={greek_factor:.2f} → "
            f"{count}×@ {limit_price}¢{greek_note}"
        ),
    }


# ---------------------------------------------------------------------------
# Short-term indicators  (1m / 5m / 15m OHLCV data)
# ---------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Rolling windowed VWAP over `window` periods.
    Uses typical price = (high + low + close) / 3.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol     = df["volume"].replace(0, float("nan"))
    vwap    = (typical * vol).rolling(window).sum() / vol.rolling(window).sum()
    return vwap


def compute_short_term_indicators(df: pd.DataFrame, timeframe: str = "1m") -> dict:
    """
    Compute short-term indicators for OHLCV data.

    df must have columns: open, high, low, close, volume
    timeframe: "1m" | "5m" | "15m"  — adjusts indicator periods accordingly.

    Indicator periods by timeframe:
      1m:  EMA5/10/21, RSI7,  MACD(5,13,3),  BB(14), VWAP(60)
      5m:  EMA9/21/55, RSI10, MACD(12,26,9), BB(20), VWAP(48)
      15m: EMA9/21/55, RSI14, MACD(12,26,9), BB(20), VWAP(24)
    """
    periods = {
        "1m":  dict(ema_fast=5,  ema_mid=10, ema_slow=21, rsi=7,  macd=(5,13,3),  bb=14, vwap=60),
        "5m":  dict(ema_fast=9,  ema_mid=21, ema_slow=55, rsi=10, macd=(12,26,9), bb=20, vwap=48),
        "15m": dict(ema_fast=9,  ema_mid=21, ema_slow=55, rsi=14, macd=(12,26,9), bb=20, vwap=24),
    }
    p = periods.get(timeframe, periods["5m"])
    close = df["close"]

    ema_f  = close.ewm(span=p["ema_fast"],  adjust=False).mean()
    ema_m  = close.ewm(span=p["ema_mid"],   adjust=False).mean()
    ema_s  = close.ewm(span=p["ema_slow"],  adjust=False).mean()
    rsi_s  = compute_rsi(close, p["rsi"])
    mf, ms, mh = compute_macd(close, *p["macd"])
    bb_u, bb_mid, bb_l = compute_bollinger(close, p["bb"])
    bb_w   = (bb_u - bb_l).replace(0, float("nan"))
    bb_pct = (close - bb_l) / bb_w
    vwap   = compute_vwap(df, p["vwap"])

    log_ret = np.log(close / close.shift(1))
    valid_std = log_ret.rolling(14).std().dropna()
    atr_pct = float(valid_std.iloc[-1]) if not valid_std.empty else 0.003
    if math.isnan(atr_pct) or atr_pct <= 0:
        atr_pct = 0.003

    def _s(series: pd.Series, fallback: float = 0.0) -> float:
        arr = series.to_numpy()
        if len(arr) == 0:
            return fallback
        v = float(arr[-1])
        return fallback if math.isnan(v) else v

    price = _s(close)
    return {
        "price":      price,
        "ema_fast":   _s(ema_f),
        "ema_mid":    _s(ema_m),
        "ema_slow":   _s(ema_s),
        "rsi":        _s(rsi_s, 50.0),
        "macd":       _s(mf),
        "macd_signal":_s(ms),
        "macd_hist":  _s(mh),
        "bb_upper":   _s(bb_u),
        "bb_lower":   _s(bb_l),
        "bb_pct_b":   _s(bb_pct, 0.5),
        "vwap":       _s(vwap, price),
        "atr_pct":    atr_pct,
        "timeframe":  timeframe,
    }


def generate_short_term_signal(ind: dict) -> dict:
    """
    Five-factor signal for short-term (1m/5m/15m) data.

    Factors:
      RSI         : ±2  (same as long-term but faster periods)
      MACD        : ±2  (line vs signal + zero line bonus)
      EMA cross   : ±1  (fast EMA vs slow EMA)
      VWAP        : ±1  (price vs rolling VWAP)
      Bollinger   : ±1  (price at band extremes)
    """
    rsi      = ind["rsi"]
    macd     = ind["macd"]
    macd_sig = ind["macd_signal"]
    price    = ind["price"]
    ema_fast = ind["ema_fast"]
    ema_slow = ind["ema_slow"]
    vwap     = ind["vwap"]
    bb_pct_b = ind["bb_pct_b"]

    score = 0

    # RSI (±2) — tightened thresholds vs long-term to catch overbought/oversold faster
    if rsi < 30:    score += 2
    elif rsi < 40:  score += 1
    elif rsi > 70:  score -= 2
    elif rsi > 55:  score -= 1   # was >60; RSI 55-70 now scores -1 instead of 0

    # MACD (±2)
    if macd > macd_sig:
        score += 1
        if macd > 0: score += 1
    else:
        score -= 1
        if macd < 0: score -= 1

    # Fast EMA crossover (±1) — replaces SMA50/EMA200 for short-term
    score += 1 if ema_fast > ema_slow else -1

    # VWAP (±1) — institutional reference price
    score += 1 if price > vwap else -1

    # Bollinger (±1)
    if bb_pct_b <= 0.05:    score += 1
    elif bb_pct_b >= 0.95:  score -= 1

    # Wider HOLD band: require ≥3 for BUY (was ≥2) so a moderate uptrend
    # with RSI slightly overbought doesn't auto-trigger entry.
    if score >= 4:
        label, direction, strength = "STRONG BUY",  "up",      1.0
    elif score >= 3:
        label, direction, strength = "BUY",          "up",      0.5
    elif score <= -4:
        label, direction, strength = "STRONG SELL",  "down",    1.0
    elif score <= -3:
        label, direction, strength = "SELL",          "down",    0.5
    else:
        label, direction, strength = "HOLD",          "neutral", 0.0

    # RSI ceiling: never enter bullish when RSI is extreme (>72)
    if direction == "up" and rsi > 72:
        label, direction, strength = "HOLD", "neutral", 0.0

    return {
        "label":      label,
        "direction":  direction,
        "strength":   strength,
        "bull_score": score,
        "rsi":         rsi,
        "macd":        macd,
        "macd_signal": macd_sig,
        "macd_hist":   ind["macd_hist"],
        "atr_pct":     ind["atr_pct"],
        "timeframe":   ind.get("timeframe", "?"),
    }


def add_short_term_chart_indicators(df: pd.DataFrame, timeframe: str = "5m") -> pd.DataFrame:
    """Add all short-term indicator columns to an OHLCV DataFrame."""
    p = {
        "1m":  dict(ema_fast=5,  ema_mid=10, ema_slow=21, rsi=7,  macd=(5,13,3),  bb=14, vwap=60),
        "5m":  dict(ema_fast=9,  ema_mid=21, ema_slow=55, rsi=10, macd=(12,26,9), bb=20, vwap=48),
        "15m": dict(ema_fast=9,  ema_mid=21, ema_slow=55, rsi=14, macd=(12,26,9), bb=20, vwap=24),
    }.get(timeframe, {
        "ema_fast":5, "ema_mid":10, "ema_slow":21, "rsi":7, "macd":(5,13,3), "bb":14, "vwap":60
    })

    df = df.copy()
    c  = df["close"]
    df["ema_fast"]    = c.ewm(span=p["ema_fast"],  adjust=False).mean()
    df["ema_mid"]     = c.ewm(span=p["ema_mid"],   adjust=False).mean()
    df["ema_slow"]    = c.ewm(span=p["ema_slow"],  adjust=False).mean()
    df["rsi"]         = compute_rsi(c, p["rsi"])
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(c, *p["macd"])
    df["bb_upper"], df["bb_mid"], df["bb_lower"]   = compute_bollinger(c, p["bb"])
    df["vwap"]        = compute_vwap(df, p["vwap"])
    return df
