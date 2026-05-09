"""
LLM-based Bitcoin directional probability model.

Combines five data streams:
  1. Technical indicators (RSI, MACD, Bollinger, MAs)   from strategy.py
  2. Multi-source BTC news headlines                     from news_fetcher.py
  3. Kalshi prediction-market implied odds               from kalshi_client.py
  4. Macro sentiment (Fear & Greed, BTC dominance)       from free public APIs
  5. Trump tweet signal                                  from trump_watcher.py (if running)

Returns a blended probability (0.0–1.0) that BTC will be higher in ~4 hours,
combining the technical score and the LLM's contextual assessment.
"""
import logging
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional

import requests
from dotenv import load_dotenv

import anthropic

load_dotenv()

log = logging.getLogger(__name__)

GECKO_BASE = "https://api.coingecko.com/api/v3"
GECKO_API  = os.getenv("GECKO_API", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Probability helpers (no LLM — pure math)
# ---------------------------------------------------------------------------

def score_to_probability(bull_score: int, max_score: int = 7) -> float:
    """
    Map the strategy bull_score (−max to +max) to a probability in [0.10, 0.90].
    Score 0 → 50%, score +max → 90%, score -max → 10%.
    """
    clamped    = max(-max_score, min(max_score, bull_score))
    normalized = (clamped + max_score) / (2 * max_score)  # 0.0 to 1.0
    return round(0.10 + 0.80 * normalized, 4)


def probability_to_signal(prob: float) -> tuple[str, str, float]:
    """
    Returns (label, direction, strength) from a blended probability.
    Thresholds are tighter than pure technical (require more conviction).
    """
    if prob >= 0.72:   return "STRONG BUY",  "up",      1.0
    if prob >= 0.60:   return "BUY",          "up",      0.5
    if prob <= 0.28:   return "STRONG SELL",  "down",    1.0
    if prob <= 0.40:   return "SELL",          "down",    0.5
    return "HOLD", "neutral", 0.0


def blend_probabilities(
    tech_prob: float,
    llm_prob: float,
    tech_weight: float = 0.50,
    llm_weight: float  = 0.50,
) -> float:
    return round(tech_weight * tech_prob + llm_weight * llm_prob, 4)


# ---------------------------------------------------------------------------
# Macro data fetchers
# ---------------------------------------------------------------------------

def fetch_fear_greed() -> dict:
    """Fear & Greed Index from alternative.me — free, no auth."""
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=2", timeout=8)
        r.raise_for_status()
        data = r.json()["data"]
        today = data[0]
        prev  = data[1] if len(data) > 1 else data[0]
        trend = "improving" if int(today["value"]) > int(prev["value"]) else (
                "worsening" if int(today["value"]) < int(prev["value"]) else "flat")
        return {
            "value":          int(today["value"]),
            "classification": today["value_classification"],
            "trend":          trend,
        }
    except Exception as e:
        log.warning(f"Fear & Greed fetch failed: {e}")
        return {"value": 50, "classification": "Neutral", "trend": "unknown"}


def fetch_btc_dominance() -> dict:
    """BTC market dominance % from CoinGecko global endpoint."""
    try:
        r = requests.get(
            f"{GECKO_BASE}/global?x_cg_demo_api_key={GECKO_API}", timeout=8
        )
        r.raise_for_status()
        g = r.json()["data"]
        dom = g.get("market_cap_percentage", {}).get("btc", 50.0)
        chg = g.get("market_cap_change_percentage_24h_usd", 0.0)
        return {
            "btc_dominance_pct":  round(dom, 1),
            "total_mcap_chg_24h": round(chg, 2),
        }
    except Exception as e:
        log.warning(f"BTC dominance fetch failed: {e}")
        return {"btc_dominance_pct": 50.0, "total_mcap_chg_24h": 0.0}


# ---------------------------------------------------------------------------
# LLM tool definition
# ---------------------------------------------------------------------------

_PROBABILITY_TOOL: dict = {
    "name": "submit_probability_estimate",
    "description": (
        "Submit your structured probability estimate for BTC price direction "
        "in the next ~4 hours."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "probability_up": {
                "type": "number",
                "description": (
                    "Probability (0.0–1.0) that BTC will be HIGHER ~4 hours from now. "
                    "0.50 = genuinely uncertain. Be calibrated, not cautious."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Your confidence in this estimate.",
            },
            "key_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Top 3–5 factors (bullish or bearish) driving your estimate.",
                "minItems": 2,
                "maxItems": 5,
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Top 2–3 risks that could invalidate your estimate.",
            },
            "reasoning": {
                "type": "string",
                "description": "1–2 sentence summary of your overall reasoning.",
            },
        },
        "required": ["probability_up", "confidence", "key_factors", "reasoning"],
    },
}

_SYSTEM_PROMPT = """\
You are an expert quantitative analyst specializing in short-term Bitcoin price prediction
for Kalshi binary option trades (next 1–4 hours horizon).

Your job is to synthesize technical indicators, news headlines, prediction-market odds,
and macro sentiment into a single calibrated directional probability.

Guidelines:
- Weight RECENT momentum and NEWS heavily — the 4-hour horizon is news-sensitive.
- Use the Kalshi market's own implied probability as a Bayesian prior before adjusting.
- 50% means genuinely uncertain — not "I don't know." Use the full 0–1 range.
- Identify specific named catalysts or risks from the headlines when present.
- A technical score near 0 with a strong news catalyst should move you away from 50%.
- You MUST respond by calling the submit_probability_estimate tool.
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _build_user_prompt(
    indicators:  dict,
    bull_score:  int,
    tech_prob:   float,
    kalshi_mkts: list,
    headlines:   str,
    fng:         dict,
    macro:       dict,
) -> str:
    price    = indicators.get("price",  0)
    rsi      = indicators.get("rsi",   50)
    macd     = indicators.get("macd",   0)
    macd_sig = indicators.get("macd_signal", 0)
    bb_pct   = indicators.get("bb_pct_b", 0.5)
    sma20    = indicators.get("sma20", price)
    sma50    = indicators.get("sma50", price)
    ema200   = indicators.get("ema200", price)
    atr_pct  = indicators.get("atr_pct", 0.003)

    ma_trend  = "BULLISH" if sma20 > sma50 else "BEARISH"
    lt_trend  = "above" if price > ema200 else "below"
    macd_dir  = "above signal (bullish)" if macd > macd_sig else "below signal (bearish)"
    rsi_zone  = "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral")

    # Kalshi summary
    if kalshi_mkts:
        kal_lines = []
        for m in kalshi_mkts[:5]:
            title    = m.get("title", m.get("ticker", ""))[:60]
            yes_ask  = m.get("yes_ask", "?")
            no_ask   = m.get("no_ask",  "?")
            vol      = m.get("volume", 0)
            kal_lines.append(
                f'  "{title}" — YES {yes_ask}¢ / NO {no_ask}¢  (vol {vol:,})'
            )
        kalshi_block = "\n".join(kal_lines)
    else:
        kalshi_block = "  No open Kalshi BTC markets available."

    return f"""\
## Live Technical Snapshot  ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})
BTC Price:     ${price:,.2f}
RSI (14):      {rsi:.1f}  [{rsi_zone}]
MACD (12/26/9): line {macd:+.1f}  {macd_dir}
Bollinger %B:  {bb_pct:.2f}  (0=lower band extreme, 1=upper band extreme)
MA crossover:  SMA20 {'>' if sma20 > sma50 else '<'} SMA50  [{ma_trend}]
Long-term:     Price {lt_trend} EMA200
1h Volatility: {atr_pct*100:.2f}% per hour (ATR)

Technical bull score:  {bull_score:+d} / 7
Technical probability: {tech_prob:.1%}

## Kalshi BTC Markets (prediction-market implied odds)
{kalshi_block}

## Macro & Sentiment
Fear & Greed Index: {fng['value']}/100  ({fng['classification']}, {fng['trend']})
BTC Market Dominance: {macro['btc_dominance_pct']}%
Total Crypto Cap 24h Change: {macro['total_mcap_chg_24h']:+.2f}%

## Recent BTC News & Community (last 6h)
{headlines}

---
Based on ALL of the above, what is the probability that BTC will be HIGHER ~4 hours from now?
Call submit_probability_estimate with your structured assessment.
"""


def run_llm_estimate(
    indicators:  dict,
    bull_score:  int,
    kalshi_mkts: list,
    headlines:   str,
    fng:         dict,
    macro:       dict,
) -> dict:
    """
    Call Claude to get a probability estimate. Returns the tool input dict.
    Raises on API errors — caller should handle and fall back to technical-only.
    """
    client     = _get_client()
    tech_prob  = score_to_probability(bull_score)
    user_msg   = _build_user_prompt(
        indicators, bull_score, tech_prob, kalshi_mkts, headlines, fng, macro
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # cache the static system prompt
            }
        ],
        tools=[_PROBABILITY_TOOL],
        tool_choice={"type": "tool", "name": "submit_probability_estimate"},
        messages=[{"role": "user", "content": user_msg}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_probability_estimate":
            return block.input

    raise ValueError("LLM did not call the expected tool")


# ---------------------------------------------------------------------------
# Full pipeline — entry point for trader.py and monitor.py
# ---------------------------------------------------------------------------

def get_combined_probability(
    indicators:  dict,
    bull_score:  int,
    kalshi_mkts: list,
    tech_weight: float = 0.50,
    llm_weight:  float = 0.50,
) -> dict:
    """
    Run the full pipeline: gather all context → call LLM → blend probabilities.

    Returns a result dict that is safe to store in trading_state.json and display
    in the dashboard. Falls back to technical-only if LLM call fails.
    """
    now       = datetime.now(timezone.utc).isoformat()
    tech_prob = score_to_probability(bull_score)

    # Gather context (all independently failable)
    fng   = fetch_fear_greed()
    macro = fetch_btc_dominance()

    try:
        from news_fetcher import fetch_all_headlines, headlines_for_llm
        raw_headlines = fetch_all_headlines(max_age_hours=6)
        news_text     = headlines_for_llm(raw_headlines, limit=20)
        news_count    = len(raw_headlines)
    except Exception as e:
        log.warning(f"News fetch failed: {e}")
        news_text  = "News unavailable."
        news_count = 0
        raw_headlines = []

    # LLM estimate
    llm_result: Optional[dict] = None
    llm_error:  Optional[str]  = None
    try:
        llm_result = run_llm_estimate(
            indicators, bull_score, kalshi_mkts, news_text, fng, macro
        )
        llm_prob       = max(0.0, min(1.0, float(llm_result["probability_up"])))
        combined_prob  = blend_probabilities(tech_prob, llm_prob, tech_weight, llm_weight)
        combined_prob  = max(0.0, min(1.0, combined_prob))
        label, direction, strength = probability_to_signal(combined_prob)
    except Exception as e:
        log.warning(f"LLM estimate failed: {e} — falling back to technical only")
        llm_prob      = tech_prob
        combined_prob = tech_prob
        llm_error     = str(e)
        label, direction, strength = probability_to_signal(tech_prob)

    # Trump tweet adjustment — applied on top of the blended estimate.
    # trump_watcher.py must be running separately; if not, this is a silent no-op.
    trump_signal: Optional[dict] = None
    try:
        from trump_watcher import get_trump_signal
        trump_signal = get_trump_signal(max_age_minutes=30)
        if trump_signal:
            adj = float(trump_signal.get("probability_adjustment", 0.0))
            combined_prob = round(max(0.05, min(0.95, combined_prob + adj)), 4)
            label, direction, strength = probability_to_signal(combined_prob)
            log.info(
                f"Trump signal applied: {trump_signal['impact']} adj={adj:+.2f} "
                f"-> combined_prob={combined_prob:.2%}"
            )
    except Exception:
        pass

    return {
        "timestamp":         now,
        "tech_prob":         tech_prob,
        "llm_prob":          llm_prob,
        "combined_prob":     combined_prob,
        "tech_weight":       tech_weight,
        "llm_weight":        llm_weight,
        "signal_label":      label,
        "signal_direction":  direction,
        "signal_strength":   strength,
        "bull_score":        bull_score,
        "fear_greed":        fng,
        "btc_dominance":     macro,
        "news_count":        news_count,
        "recent_headlines":  raw_headlines[:10],
        "llm_detail":        llm_result,
        "llm_error":         llm_error,
        "trump_signal":      trump_signal,
    }
