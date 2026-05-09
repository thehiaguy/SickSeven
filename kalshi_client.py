"""Kalshi Trading API v2 client with RSA authentication and retry logic."""
import base64
import logging
import os
import textwrap
import time
import uuid
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

_KALSHI_BASE_URL = "https://external-api.kalshi.com"
_API_PREFIX      = "/trade-api/v2"
_KEY_ID          = os.getenv("KALSHI_API_KEY", "")
_PRIV_RAW        = os.getenv("KALSHI_PRIV", "")


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------

def _load_private_key():
    if not _PRIV_RAW:
        raise EnvironmentError("KALSHI_PRIV not set in .env")
    raw_b64 = "".join(_PRIV_RAW.split())
    lines   = textwrap.wrap(raw_b64, 64)
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    for header in ("RSA PRIVATE KEY", "PRIVATE KEY"):
        try:
            pem = (f"-----BEGIN {header}-----\n"
                   + "\n".join(lines)
                   + f"\n-----END {header}-----")
            return serialization.load_pem_private_key(
                pem.encode(), password=None, backend=default_backend()
            )
        except Exception:
            continue
    raise ValueError("Could not load Kalshi private key — check KALSHI_PRIV in .env")


_PRIVATE_KEY = _load_private_key()


# ---------------------------------------------------------------------------
# Request signing and transport
# ---------------------------------------------------------------------------

def _signed_headers(method: str, path: str) -> dict:
    """
    Kalshi RSA auth.  Signed message = timestamp_ms + METHOD + full_path.
    A fresh timestamp is generated on every call so retries get new signatures.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    ts  = str(int(time.time() * 1000))
    msg = (ts + method.upper() + _API_PREFIX + path).encode()
    sig = _PRIVATE_KEY.sign(msg, padding.PKCS1v15(), hashes.SHA256())
    return {
        "KALSHI-ACCESS-KEY":       _KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type":            "application/json",
    }


def _url(path: str) -> str:
    return _KALSHI_BASE_URL + _API_PREFIX + path


def _request(method: str, path: str, *, params=None, json=None, timeout=10) -> dict:
    """
    Single HTTP request with up to 3 attempts and exponential backoff.
    Signing is refreshed on each attempt so the timestamp is always current.
    """
    last_exc = None
    for attempt in range(3):
        try:
            r = requests.request(
                method,
                _url(path),
                headers=_signed_headers(method, path),
                params=params,
                json=json,
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            # 4xx errors are not transient — don't retry
            if e.response is not None and 400 <= e.response.status_code < 500:
                raise
            last_exc = e
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e

        wait = 2 ** attempt  # 1s, 2s, 4s
        log.warning(f"Kalshi request failed (attempt {attempt+1}/3): {last_exc}. Retrying in {wait}s")
        time.sleep(wait)

    raise last_exc


def _get(path: str, params: dict = None) -> dict:
    return _request("GET", path, params=params)


def _post(path: str, body: dict) -> dict:
    return _request("POST", path, json=body)


def _delete(path: str) -> dict:
    return _request("DELETE", path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_balance() -> dict:
    """Returns {'balance': cents, ...}"""
    return _get("/portfolio/balance")


def get_markets(series_ticker: str = "KXBTCD", status: str = "open") -> list:
    data = _get("/markets", {"series_ticker": series_ticker,
                              "status": status, "limit": 50})
    return data.get("markets", [])


def search_markets(keyword: str, status: str = "open") -> list:
    """Fallback: keyword search across all open markets."""
    data = _get("/markets", {"status": status, "limit": 100})
    kw   = keyword.lower()
    return [m for m in data.get("markets", [])
            if kw in m.get("title", "").lower() or kw in m.get("ticker", "").lower()]


def get_positions() -> list:
    data = _get("/portfolio/positions")
    return data.get("market_positions", [])


def get_orders(status: str = "resting") -> list:
    data = _get("/portfolio/orders", {"status": status, "limit": 100})
    return data.get("orders", [])


def get_order(order_id: str) -> dict:
    return _get(f"/orders/{order_id}")


def place_order(
    ticker: str,
    action: str,
    side: str,
    count: int,
    order_type: str,
    price_cents: Optional[int] = None,
) -> dict:
    """
    action:      "buy" | "sell"
    side:        "yes" | "no"
    order_type:  "limit" | "market"
    price_cents: required for limit orders (1–99)
    """
    body: dict = {
        "ticker":          ticker,
        "client_order_id": str(uuid.uuid4()),
        "action":          action,
        "type":            order_type,
        "side":            side,
        "count":           count,
    }
    if order_type == "limit" and price_cents is not None:
        body["yes_price" if side == "yes" else "no_price"] = price_cents
    return _post("/orders", body)


def close_position(ticker: str, net_position: int) -> Optional[dict]:
    """
    Close an open position at market.
    net_position > 0 → long YES → sell YES
    net_position < 0 → long NO  → sell NO
    Returns the order response, or None if nothing to close.
    """
    if net_position == 0:
        return None
    side  = "yes" if net_position > 0 else "no"
    count = abs(net_position)
    log.info(f"Closing position: sell {count} {side} @ market on {ticker}")
    return place_order(ticker, "sell", side, count, "market")


def cancel_order(order_id: str) -> dict:
    return _delete(f"/orders/{order_id}")


def cancel_all_resting() -> list:
    """Cancel every resting order. Returns list of cancel responses."""
    results = []
    for order in get_orders(status="resting"):
        oid = order.get("order_id", "")
        try:
            results.append(cancel_order(oid))
            log.info(f"Cancelled order {oid}")
        except Exception as e:
            log.error(f"Failed to cancel order {oid}: {e}")
    return results
