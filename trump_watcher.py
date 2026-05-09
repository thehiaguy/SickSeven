"""
Trump tweet watcher — polls Truth Social and Twitter/X (via Nitter) for new
posts from @realDonaldTrump and classifies their BTC/USD market impact via
Claude Haiku (fast + cheap for repetitive polling).

Run standalone:  python trump_watcher.py
Also imported by probability_model.py via get_trump_signal().

State is persisted to trump_state.json so the watcher survives restarts
without re-classifying tweets it has already seen.
"""
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import feedparser
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
STATE_FILE        = Path("trump_state.json")
POLL_INTERVAL     = 30  # seconds

# RSS sources tried in order — Truth Social is most reliable (no rate limiting),
# Nitter instances are fallbacks for anything not cross-posted.
RSS_SOURCES = [
    "https://truthsocial.com/@realDonaldTrump.rss",
    "https://nitter.net/realDonaldTrump/rss",
    "https://nitter.1d4.us/realDonaldTrump/rss",
    "https://nitter.poast.org/realDonaldTrump/rss",
    "https://nitter.privacydev.net/realDonaldTrump/rss",
]

_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")
        _client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# LLM classification (Claude Haiku — cheap + fast for this repetitive task)
# ---------------------------------------------------------------------------

_CLASSIFY_TOOL = {
    "name": "classify_tweet_impact",
    "description": "Classify a Trump tweet's impact on BTC and USD markets.",
    "input_schema": {
        "type": "object",
        "properties": {
            "impact": {
                "type": "string",
                "enum": ["btc_bullish", "btc_bearish", "usd_bearish", "usd_bullish", "neutral"],
                "description": (
                    "Primary market impact category. "
                    "usd_bearish = dollar-weakening policy (inflation, rate cuts) — proxy bullish for BTC. "
                    "usd_bullish = strong dollar / rate hawkishness — proxy bearish for BTC."
                ),
            },
            "urgency": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": (
                    "How quickly markets are likely to react. "
                    "high = direct crypto/Bitcoin/dollar policy statement or major geopolitical shock — act within minutes. "
                    "medium = tariff/trade escalation or indirect financial policy with hours-scale impact. "
                    "low = political posturing or social commentary with minor financial relevance."
                ),
            },
            "probability_adjustment": {
                "type": "number",
                "description": (
                    "How much to shift the BTC-up probability. "
                    "Range: -0.25 (strongly bearish for BTC) to +0.25 (strongly bullish). "
                    "Positive = BTC likely to rise. Negative = BTC likely to fall. "
                    "Use 0.0 for genuinely neutral tweets."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining the expected market reaction and why.",
            },
        },
        "required": ["impact", "urgency", "probability_adjustment", "reasoning"],
    },
}

_SYSTEM_PROMPT = """\
You are a financial analyst specializing in how Donald Trump's public statements move BTC and USD markets.

Classification guide:
- btc_bullish  (+adj): direct support for Bitcoin/crypto, strategic reserve mention, deregulation, pro-crypto policy
- btc_bearish  (-adj): anti-crypto statement, tighter regulation threat, crackdown on exchanges or stablecoins
- usd_bearish  (+adj): dollar-weakening policy — tariff inflation, pressure on Fed to cut rates, deficit expansion, sanctions bypass → BTC as inflation hedge rises
- usd_bullish  (-adj): strong dollar stance, fiscal tightening, rate hike support, dollar dominance rhetoric → reduces BTC appeal as alternative asset
- neutral (0.0):       personal attacks, sports/golf, political endorsements, cultural commentary with no financial angle

Urgency:
- high:   direct named policy statement on crypto, Bitcoin, the dollar, Fed, or a major geopolitical shock
- medium: tariff/trade news, sanctions, indirect fiscal/monetary signals
- low:    political theatre or social commentary where financial implications are vague or speculative

Be calibrated: most tweets are neutral. Only flag high urgency for genuinely market-moving statements."""


def classify_tweet(tweet_text: str) -> dict:
    """Classify a tweet's BTC market impact using Claude Haiku."""
    try:
        resp = _get_client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f'Classify this tweet:\n\n"{tweet_text}"'}],
            tools=[_CLASSIFY_TOOL],
            tool_choice={"type": "tool", "name": "classify_tweet_impact"},
        )
        for block in resp.content:
            if block.type == "tool_use":
                result = dict(block.input)
                result["probability_adjustment"] = max(
                    -0.25, min(0.25, float(result.get("probability_adjustment", 0.0)))
                )
                return result
    except Exception as e:
        log.warning(f"Tweet classification failed: {e}")
    return {
        "impact": "neutral",
        "urgency": "low",
        "probability_adjustment": 0.0,
        "reasoning": "Classification unavailable",
    }


# ---------------------------------------------------------------------------
# RSS fetching
# ---------------------------------------------------------------------------

def fetch_latest_tweets(limit: int = 5) -> list:
    """
    Try each RSS source in order. Returns a list of tweet dicts on first success.
    Strips HTML tags from entry text (Nitter sometimes includes <a> and <img> tags).
    """
    for url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                continue
            tweets = []
            for entry in feed.entries[:limit]:
                tweet_id = entry.get("id") or entry.get("link") or ""
                if not tweet_id:
                    continue  # skip entries with no identifier — can't deduplicate them
                raw      = entry.get("title") or entry.get("summary") or ""
                text     = re.sub(r"<[^>]+>", " ", raw).strip()
                text     = re.sub(r"\s+", " ", text)
                tweets.append({
                    "id":        tweet_id,
                    "text":      text,
                    "url":       entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "source":    url.split("/")[2],
                })
            if tweets:
                return tweets
        except Exception as e:
            log.debug(f"RSS source {url} failed: {e}")
    return []


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"seen_ids": [], "latest_signal": None, "last_checked": None}


def _save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ---------------------------------------------------------------------------
# Public API — called by probability_model.py
# ---------------------------------------------------------------------------

def get_trump_signal(max_age_minutes: int = 30) -> Optional[dict]:
    """
    Returns the latest classified Trump signal if it is recent and non-neutral.
    Returns None if no signal exists, the signal is stale, or the tweet was neutral.

    Called by probability_model.get_combined_probability() to apply a probability
    adjustment on top of the technical + LLM blended estimate.
    """
    state = _load_state()
    sig   = state.get("latest_signal")
    if not sig:
        return None
    try:
        ts = datetime.fromisoformat(sig["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60
        if age_min > max_age_minutes:
            return None
        if sig.get("impact") == "neutral" or sig.get("probability_adjustment", 0.0) == 0.0:
            return None
        # Re-clamp adjustment so a hand-edited state file can't produce extreme shifts
        adj = float(sig.get("probability_adjustment", 0.0))
        sig["probability_adjustment"] = max(-0.25, min(0.25, adj))
        return sig
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Standalone watcher loop
# ---------------------------------------------------------------------------

def run_watcher():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.FileHandler("trump_watcher.log"),
            logging.StreamHandler(),
        ],
    )
    log.info(f"Trump watcher started — polling every {POLL_INTERVAL}s")
    log.info(f"Sources: {[s.split('/')[2] for s in RSS_SOURCES]}")

    state    = _load_state()
    seen_ids = set(state.get("seen_ids", [])[-200:])

    while True:
        try:
            state["last_checked"] = datetime.now(timezone.utc).isoformat()
            tweets     = fetch_latest_tweets(limit=5)
            # Deduplicate within this batch too (same tweet from multiple RSS sources)
            seen_this_batch: set = set()
            new_tweets = []
            for t in tweets:
                if t["id"] not in seen_ids and t["id"] not in seen_this_batch:
                    new_tweets.append(t)
                    seen_this_batch.add(t["id"])

            if not tweets:
                log.warning("All RSS sources returned empty — check connectivity")

            for tweet in reversed(new_tweets):  # oldest first
                log.info(f"New tweet [{tweet['source']}]: {tweet['text'][:120]}")
                classification = classify_tweet(tweet["text"])

                signal = {
                    **classification,
                    "tweet_text": tweet["text"][:280],
                    "tweet_url":  tweet["url"],
                    "published":  tweet["published"],
                    "timestamp":  datetime.now(timezone.utc).isoformat(),
                    "source":     tweet["source"],
                }

                log.info(
                    f"  => impact={signal['impact']}  urgency={signal['urgency']}  "
                    f"adj={signal['probability_adjustment']:+.2f}  | {signal['reasoning']}"
                )

                seen_ids.add(tweet["id"])
                state["latest_signal"] = signal

            state["seen_ids"] = list(seen_ids)[-200:]
            _save_state(state)

        except Exception as e:
            log.error(f"Watcher cycle error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_watcher()
