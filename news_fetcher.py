"""
Multi-source BTC news aggregator. All sources are free — no API keys required.

Fetches RSS feeds and Reddit concurrently and returns a unified, deduplicated,
time-sorted list of recent headlines. Used by probability_model.py.
"""
import concurrent.futures
import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import feedparser
import requests

log = logging.getLogger(__name__)

REDDIT_UA = "BTCMonitor/1.0 (trading research; non-commercial)"

RSS_SOURCES: list[tuple[str, str]] = [
    ("CoinDesk",         "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph",    "https://cointelegraph.com/rss"),
    ("Bitcoin Magazine", "https://bitcoinmagazine.com/feed"),
    ("Decrypt",          "https://decrypt.co/feed"),
    ("Bitcoinist",       "https://bitcoinist.com/feed/"),
    ("NewsbtC",          "https://www.newsbtc.com/feed/"),
    ("CryptoNews",       "https://cryptonews.com/news/feed/"),
    ("BeInCrypto",       "https://beincrypto.com/feed/"),
]

REDDIT_SOURCES: list[tuple[str, str]] = [
    ("Reddit/r/Bitcoin",
     "https://www.reddit.com/r/Bitcoin/new.json?limit=15"),
    ("Reddit/r/CryptoCurrency",
     "https://www.reddit.com/r/CryptoCurrency/search.json?q=bitcoin&sort=new&limit=10&restrict_sr=1"),
    ("Reddit/r/btc",
     "https://www.reddit.com/r/btc/new.json?limit=10"),
]


# ---------------------------------------------------------------------------
# Per-source fetchers
# ---------------------------------------------------------------------------

def _fetch_rss(name: str, url: str, max_items: int = 8) -> list[dict]:
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            pub = None
            for attr in ("published", "updated"):
                raw = getattr(entry, attr, None)
                if raw:
                    try:
                        pub = parsedate_to_datetime(raw).astimezone(timezone.utc)
                    except Exception:
                        pass
                    break
            if pub is None:
                pub = datetime.now(timezone.utc)

            age_min = (datetime.now(timezone.utc) - pub).total_seconds() / 60
            items.append({
                "source":      name,
                "title":       entry.get("title", "").strip(),
                "url":         entry.get("link", ""),
                "published":   pub.isoformat(),
                "age_minutes": round(age_min, 1),
                "summary":     (entry.get("summary") or "")[:300].strip(),
            })
        return items
    except Exception as e:
        log.debug(f"RSS fetch failed [{name}]: {e}")
        return []


def _fetch_reddit(name: str, url: str, max_items: int = 8) -> list[dict]:
    try:
        r = requests.get(url, headers={"User-Agent": REDDIT_UA}, timeout=15)
        r.raise_for_status()
        posts = r.json().get("data", {}).get("children", [])
        items = []
        for post in posts[:max_items]:
            d = post.get("data", {})
            title = d.get("title", "").strip()
            # Skip meme / image posts
            if not title or d.get("is_video") or d.get("post_hint") == "image":
                continue
            ts  = d.get("created_utc", time.time())
            pub = datetime.fromtimestamp(ts, tz=timezone.utc)
            age_min = (datetime.now(timezone.utc) - pub).total_seconds() / 60
            items.append({
                "source":      name,
                "title":       title,
                "url":         f"https://reddit.com{d.get('permalink', '')}",
                "published":   pub.isoformat(),
                "age_minutes": round(age_min, 1),
                "summary":     (d.get("selftext") or "")[:300].strip(),
                "score":       d.get("score", 0),
            })
        return items
    except Exception as e:
        log.debug(f"Reddit fetch failed [{name}]: {e}")
        return []


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_all_headlines(
    max_age_hours: float = 6.0,
    max_total: int = 40,
) -> list[dict]:
    """
    Fetch from all sources in parallel.
    Returns deduplicated headlines sorted newest-first, filtered to the last
    `max_age_hours` hours.
    """
    tasks: list[tuple] = (
        [("rss",    name, url) for name, url in RSS_SOURCES] +
        [("reddit", name, url) for name, url in REDDIT_SOURCES]
    )

    all_items: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {}
        for kind, name, url in tasks:
            if kind == "rss":
                fut = pool.submit(_fetch_rss, name, url)
            else:
                fut = pool.submit(_fetch_reddit, name, url)
            futures[fut] = name

        for fut in concurrent.futures.as_completed(futures, timeout=15):
            try:
                all_items.extend(fut.result())
            except Exception:
                pass

    # Deduplicate: include source so similar titles from different outlets both appear
    seen: set[str] = set()
    unique: list[dict] = []
    for item in all_items:
        key = f"{item.get('source', '')}|{item['title'][:60].lower().strip()}"
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    # Filter by age and sort newest first
    max_age_min = max_age_hours * 60
    recent = [i for i in unique if i["age_minutes"] <= max_age_min]
    recent.sort(key=lambda x: x["age_minutes"])

    return recent[:max_total]


def headlines_for_llm(headlines: list[dict], limit: int = 20) -> str:
    """
    Format headlines into a compact string for the LLM prompt.
    """
    if not headlines:
        return "No recent headlines available."
    lines = []
    for h in headlines[:limit]:
        age = f"{h['age_minutes']:.0f}m ago" if h["age_minutes"] < 60 else f"{h['age_minutes']/60:.1f}h ago"
        lines.append(f"[{h['source']} · {age}] {h['title']}")
    return "\n".join(lines)
