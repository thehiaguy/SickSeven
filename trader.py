"""
Autonomous BTC/Kalshi trading daemon.

Usage:
    python trader.py

Reads trading_config.json each cycle — change settings (including enable/disable,
stop-loss, and cooldown) at runtime without restarting.  Writes all state to
trading_state.json for the dashboard to read.
"""
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

import kalshi_client as kc
from strategy import compute_indicators, generate_signal, select_market, compute_order

load_dotenv()

# Force UTF-8 on Windows terminals so Unicode log characters (→ ▲ ▼) don't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GECKO_API_KEY = os.getenv("GECKO_API", "")
GECKO_BASE    = "https://api.coingecko.com/api/v3"
STATE_FILE    = Path("trading_state.json")
CONFIG_FILE   = Path("trading_config.json")
LOG_FILE      = Path("trader.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("trader")

DEFAULT_CONFIG: dict = {
    "enabled":            False,   # master switch — must be True to place real orders
    "dry_run":            True,    # log orders but never actually send them
    "series_ticker":      "KXBTCD",
    "max_contracts":      2,       # conservative: ~$1 per trade at 50c/contract
    "max_open_risk_usd":  5.0,     # hard cap on total open exposure in USD
    "stop_loss_pct":      0.40,    # close a position when its loss exceeds 40% of cost
    "only_on_change":     True,    # skip order if signal label hasn't changed
    "cooldown_minutes":   15,      # minimum minutes between consecutive orders
    "loop_interval_sec":  60,
}


# ---------------------------------------------------------------------------
# Config / state I/O
# ---------------------------------------------------------------------------

def load_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            cfg.update(json.loads(CONFIG_FILE.read_text()))
        except Exception as e:
            log.warning(f"Bad config file: {e}")
    return cfg


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return _empty_state()


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _empty_state() -> dict:
    return {
        "is_running":             False,
        "enabled":                False,
        "dry_run":                True,
        "last_run":               None,
        "last_signal":            None,
        "last_signal_time":       None,
        "last_order_time":        None,
        "current_price":          None,
        "current_indicators":     {},
        "balance_cents":          None,
        "active_positions":       [],
        "recent_orders":          [],
        "total_realized_pnl_cents": 0,
        "errors":                 [],
    }


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch_price_history() -> pd.Series:
    url = (f"{GECKO_BASE}/coins/bitcoin/market_chart"
           f"?vs_currency=usd&days=30&interval=hourly"
           f"&x_cg_demo_api_key={GECKO_API_KEY}")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return pd.Series([p[1] for p in r.json()["prices"]])


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------

def open_risk_usd(positions: list) -> float:
    """Total cost basis of all open positions in USD."""
    total = 0.0
    for p in positions:
        # total_cost is the aggregate cents paid for this position
        cost_cents = abs(p.get("total_cost", 0) or 0)
        total += cost_cents / 100
    return total


def positions_to_close(positions: list, stop_loss_pct: float) -> list:
    """
    Return positions whose unrealized loss exceeds stop_loss_pct of cost.
    e.g. stop_loss_pct=0.40 → close when down 40%.
    """
    to_close = []
    for p in positions:
        total_cost   = abs(p.get("total_cost",     0) or 0)
        unrealized   =     p.get("unrealized_pnl", 0) or 0
        if total_cost > 0 and (unrealized / total_cost) < -stop_loss_pct:
            to_close.append(p)
    return to_close


def already_positioned(positions: list, ticker: str) -> bool:
    """True if we already hold a non-zero position on this market."""
    return any(p.get("ticker") == ticker and (p.get("position") or 0) != 0
               for p in positions)


def cooldown_remaining(last_order_iso: Optional[str], cooldown_minutes: float) -> float:
    """Returns seconds remaining in cooldown, or 0 if cooldown has expired."""
    if not last_order_iso:
        return 0.0
    try:
        last = datetime.fromisoformat(last_order_iso)
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        remaining = cooldown_minutes * 60 - elapsed
        return max(0.0, remaining)
    except Exception:
        return 0.0


# Bring Optional into scope (used by cooldown_remaining type hint)
from typing import Optional


# ---------------------------------------------------------------------------
# Core trading cycle
# ---------------------------------------------------------------------------

def run_cycle(config: dict, state: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    state["last_run"] = now
    state["enabled"]  = config["enabled"]
    state["dry_run"]  = config["dry_run"]
    state["errors"]   = []

    # ── 1. Compute signal ────────────────────────────────────────────────────
    try:
        prices     = fetch_price_history()
        indicators = compute_indicators(prices)
        signal     = generate_signal(indicators)

        # Overlay composite BRTI-approximate price for display and Greeks.
        # The 30-day hourly series from CoinGecko drives the signal; the
        # composite gives a fresher price for risk management calculations.
        try:
            from price_feed import get_composite_price
            _cp = get_composite_price()
            live_price = _cp["price"]
            state["price_feed"] = _cp
        except Exception as _pf_err:
            log.debug(f"Composite price feed failed, using CoinGecko last: {_pf_err}")
            live_price = indicators["price"]
            state["price_feed"] = None

        state["current_price"]      = live_price
        state["current_indicators"] = indicators

        prev_label = state.get("last_signal")
        if signal["label"] != prev_label:
            log.info(
                f"Signal change: {prev_label} → {signal['label']}  "
                f"RSI={signal['rsi']:.1f}  MACD={signal['macd']:.1f}  "
                f"score={signal['bull_score']:+d}"
            )
            state["last_signal"]      = signal["label"]
            state["last_signal_time"] = now
        else:
            log.info(
                f"Signal unchanged: {signal['label']}  "
                f"RSI={signal['rsi']:.1f}  score={signal['bull_score']:+d}"
            )
    except Exception as e:
        log.error(f"Signal computation failed: {e}", exc_info=True)
        state["errors"].append(f"Signal: {e}")
        return state

    # ── 2. Refresh portfolio ─────────────────────────────────────────────────
    try:
        bal = kc.get_balance()
        state["balance_cents"]   = bal.get("balance", 0)
        positions                = kc.get_positions()
        state["active_positions"] = positions
    except Exception as e:
        log.warning(f"Portfolio refresh failed: {e}")
        state["errors"].append(f"Portfolio: {e}")
        positions = state.get("active_positions", [])

    # ── 3. Stop-loss exits ───────────────────────────────────────────────────
    if config.get("enabled") and not config.get("dry_run"):
        stop_pct   = config.get("stop_loss_pct", 0.40)
        for p in positions_to_close(positions, stop_pct):
            ticker  = p.get("ticker", "")
            net_pos = p.get("position", 0)
            pnl_pct = ((p.get("unrealized_pnl") or 0) / max(abs(p.get("total_cost") or 1), 1)) * 100
            log.warning(
                f"Stop-loss triggered on {ticker}: "
                f"P&L={pnl_pct:.1f}% < -{stop_pct*100:.0f}%"
            )
            try:
                kc.close_position(ticker, net_pos)
                state["errors"].append(f"Stop-loss closed {ticker} ({pnl_pct:.1f}%)")
            except Exception as e:
                log.error(f"Stop-loss close failed for {ticker}: {e}")
                state["errors"].append(f"Stop-loss error {ticker}: {e}")

    # ── 4. Guard rails ───────────────────────────────────────────────────────
    if not config["enabled"]:
        log.info("Trading disabled — no order placed")
        return state

    if signal["direction"] == "neutral":
        log.info("HOLD — no order placed")
        return state

    if config["only_on_change"] and signal["label"] == prev_label:
        log.info("Signal unchanged and only_on_change=True — skipping")
        return state

    # Cooldown check
    cd_secs = cooldown_remaining(state.get("last_order_time"), config.get("cooldown_minutes", 15))
    if cd_secs > 0:
        log.info(f"Cooldown active — {cd_secs:.0f}s remaining")
        return state

    # Risk cap
    risk = open_risk_usd(state.get("active_positions", []))
    if risk >= config["max_open_risk_usd"]:
        msg = f"Open risk ${risk:.2f} >= cap ${config['max_open_risk_usd']} — skipping"
        log.warning(msg)
        state["errors"].append(msg)
        return state

    # ── 5. Select market and size order ──────────────────────────────────────
    try:
        series  = config.get("series_ticker", "KXBTCD")
        markets = kc.get_markets(series_ticker=series)
        if not markets:
            log.warning(f"No markets for {series} — falling back to BTC keyword search")
            markets = kc.search_markets("btc")

        market = select_market(
            markets,
            signal["direction"],
            btc_price=live_price,
            atr_pct=indicators.get("atr_pct", 0.003),
        )
        if not market:
            state["errors"].append("No suitable market found")
            return state

        # Don't double into an existing position on the same market
        if already_positioned(state.get("active_positions", []), market["ticker"]):
            log.info(f"Already positioned on {market['ticker']} — skipping")
            return state

        order_params = compute_order(
            signal,
            market,
            config["max_contracts"],
            atr_pct=indicators.get("atr_pct", 0.003),
            btc_price=live_price,
        )
        if not order_params:
            log.info("No order computed (missing price data?)")
            return state
    except Exception as e:
        log.error(f"Market/order selection failed: {e}", exc_info=True)
        state["errors"].append(f"Selection: {e}")
        return state

    # ── 6. Execute (or dry-run) ───────────────────────────────────────────────
    # Log Greek warning before placing
    greeks = order_params.get("greeks") or {}
    if greeks.get("near_expiry"):
        log.warning(
            f"Greek alert: contract near expiry ({greeks.get('hours_to_expiry', '?'):.1f}h) "
            f"Δ={greeks.get('delta_per_1k', 0):.1f}¢/$1k — sizing capped at 30%"
        )
    elif greeks.get("high_gamma"):
        log.info(
            f"Greek note: elevated delta Δ={greeks.get('delta_per_1k', 0):.1f}¢/$1k "
            f"— position scaled by greek_factor={greeks.get('gamma_factor', 1):.2f}"
        )

    order_record = {
        "timestamp":          now,
        "signal":             signal["label"],
        "bull_score":         signal["bull_score"],
        "ticker":             order_params["ticker"],
        "side":               order_params["side"],
        "count":              order_params["count"],
        "price_cents":        order_params["price_cents"],
        "cost_usd":           order_params["cost_usd"],
        "rationale":          order_params["rationale"],
        "suggested_stop_pct": order_params.get("suggested_stop_pct"),
        "greeks":             greeks or None,
        "dry_run":            config["dry_run"],
        "result":             None,
    }

    if config["dry_run"]:
        log.info(f"[DRY RUN] Would place: {order_params['rationale']}")
        order_record["result"] = "dry_run"
    else:
        try:
            result = kc.place_order(
                ticker      = order_params["ticker"],
                action      = order_params["action"],
                side        = order_params["side"],
                count       = order_params["count"],
                order_type  = order_params["order_type"],
                price_cents = order_params["price_cents"],
            )
            order_record["result"] = result
            state["last_order_time"] = now
            log.info(f"Order placed: {result}")
        except Exception as e:
            log.error(f"Order placement failed: {e}", exc_info=True)
            order_record["result"] = f"ERROR: {e}"
            state["errors"].append(f"Order: {e}")

    state["recent_orders"] = ([order_record] + state.get("recent_orders", []))[:100]
    return state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        log.info(f"Created default config at {CONFIG_FILE}")

    log.info("=== BTC Trader daemon started ===")
    state = load_state()
    state["is_running"] = True
    save_state(state)

    try:
        while True:
            config = load_config()
            state  = load_state()
            state["is_running"] = True

            try:
                state = run_cycle(config, state)
            except Exception as e:
                log.error(f"Unexpected cycle error: {e}", exc_info=True)
                state["errors"] = state.get("errors", []) + [str(e)]

            save_state(state)
            interval = config.get("loop_interval_sec", 60)
            log.info(f"Next cycle in {interval}s")
            time.sleep(interval)

    except KeyboardInterrupt:
        log.info("Trader stopped by keyboard interrupt")
    finally:
        state = load_state()
        state["is_running"] = False
        save_state(state)
        log.info("=== Trader daemon stopped ===")


if __name__ == "__main__":
    main()
