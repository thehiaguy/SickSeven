# SickSeven — BTC / Kalshi Automated Trading System

A real-time Bitcoin analysis and autonomous trading engine. Generates directional
signals from technical indicators, an LLM probability model, live news, and Trump
tweet monitoring, then executes binary-option orders on Kalshi.

---

## What it does

**Long-term strategy (trader daemon)**
Runs every 10–60 seconds (configurable). Pulls 30 days of hourly BTC prices from
CoinGecko, scores 5 technical factors (RSI, MACD, MA crossover, EMA200 trend,
Bollinger Bands), and places Kalshi binary-option orders when there is a clear
directional edge.

**Short-term monitor (1m / 5m / 15m)**
A separate Streamlit dashboard with live Kraken candlestick charts, short-term
indicators (VWAP, fast/slow EMA cross), a real-time signal badge, and a Trump
signal card. Refreshes every 15 seconds.

**LLM probability model**
Uses the Claude API (claude-sonnet-4-6) to blend the technical signal with live BTC
news headlines, Fear & Greed Index, BTC dominance, and any recent Trump tweet signal
into a 0–100% probability estimate. Displayed as a gauge on the short-term monitor.

**Trump tweet watcher**
Polls Trump's Truth Social and Twitter/X feeds every 30 seconds around the clock.
Every new tweet is classified by Claude Haiku into a market impact category
(BTC bullish / BTC bearish / USD bearish / USD bullish / neutral) and a probability
adjustment (±0–25%). The signal feeds directly into the probability model.

---

## Project Structure

```
sickseven/
├── .env                    # API keys and secrets (never commit)
├── requirements.txt        # Python dependencies
│
├── strategy.py             # All indicator + signal + sizing logic (long + short term)
├── kalshi_client.py        # Kalshi API v2 wrapper (RSA-PSS auth + retry)
├── news_fetcher.py         # BTC news aggregator (8 RSS feeds + 3 Reddit, no keys)
├── probability_model.py    # LLM probability model (Claude API + Trump signal)
├── trump_watcher.py        # 24/7 Trump tweet monitor and classifier
├── price_feed.py           # Multi-exchange composite BTC price (Kraken, Coinbase, etc.)
│
├── trader.py               # Autonomous trading daemon
├── dashboard.py            # Long-term trading control UI (port 8501)
├── monitor.py              # Short-term market monitor (port 8502)
│
├── trading_config.json     # Runtime config (auto-created on first run)
├── trading_state.json      # Live state written by daemon, read by dashboards
├── trump_state.json        # Latest Trump tweet + classification (written by watcher)
├── trump_watcher.log       # Log of every detected tweet and its classification
└── trader.log              # Rolling log of every trader cycle
```

---

## File Descriptions

### `.env`
Holds all secrets. Never share or commit this file.

| Variable | What it is |
|---|---|
| `GECKO_API` | CoinGecko demo API key for 30-day hourly price history |
| `KALSHI_API_KEY` | Your Kalshi key UUID (identifies who you are) |
| `KALSHI_PRIV` | RSA private key used to sign every Kalshi request |
| `ANTHROPIC_API_KEY` | Claude API key — used by both the probability model and the tweet classifier |

Kalshi uses RSA-PSS signature authentication — every API request is signed with
your private key, not a simple bearer token.

---

### `strategy.py`
All trading logic in one place. No API calls, no I/O — pure computation.
Everything else imports from here; never duplicate indicator logic in other files.

**Long-term signal (5 factors, ±7 max score):**

| Factor | Bullish | Bearish |
|---|---|---|
| RSI (±2) | RSI < 30 → +2, RSI < 40 → +1 | RSI > 70 → -2, RSI > 60 → -1 |
| MACD (±2) | Line > signal +1, line > 0 +1 | Line < signal -1, line < 0 -1 |
| MA crossover (±1) | SMA20 > SMA50 | SMA20 < SMA50 |
| EMA200 trend (±1) | Price > EMA200 | Price < EMA200 |
| Bollinger %B (±1) | %B ≤ 0.05 (at lower band) | %B ≥ 0.95 (at upper band) |

Score → signal: ≥4 = STRONG BUY, 3 = BUY, -2–+2 = HOLD, -3 = SELL, ≤-4 = STRONG SELL

**Short-term signal (1m / 5m / 15m):** Same RSI and MACD factors, but replaces
SMA/EMA200 filters with a fast/slow EMA cross (±1) and VWAP comparison (±1).
Indicator periods adapt to the selected timeframe.

Short-term RSI thresholds are tightened vs long-term: RSI > 55 scores −1 (not > 60),
and HOLD requires score −2 to +2 (requires ≥3 for BUY). This prevents a moderate
uptrend from auto-triggering entry on every cycle.

**Position sizing:** ATR-based volatility scaling. At 2× normal volatility, position
size halves to keep dollar risk roughly constant. Entry price must be 20–80¢ —
trades outside this range are refused because the risk/reward is unfavourable.

**15M profit-take:** When holding a 15-minute contract, the daemon checks on every
cycle whether the current bid has reached 80¢. If so, it sells at market to lock in
the gain rather than risk theta decay to zero.

---

### `kalshi_client.py`
Low-level Kalshi API v2 wrapper. Every request is RSA-PSS signed with a fresh
timestamp. Includes 3-attempt exponential backoff retry. HTTP 429 (rate limit) is
retried with backoff; other 4xx errors are raised immediately.

Key functions: `get_balance`, `get_markets`, `get_positions`, `get_orders`,
`place_order`, `close_position` (for stop-loss), `cancel_all_resting` (emergency).

**Important:** The Kalshi positions API has a lag after binary option fills —
newly executed orders may not appear in `get_positions()` for several seconds.
The trader daemon works around this by tracking ordered tickers in its own state.

---

### `news_fetcher.py`
Aggregates BTC headlines from 11 free sources with no API keys required:
- **8 RSS feeds**: CoinDesk, CoinTelegraph, Bitcoin Magazine, Decrypt, Bitcoinist,
  NewsBTC, CryptoNews, BeInCrypto
- **3 Reddit JSON feeds**: r/Bitcoin, r/CryptoCurrency (BTC-filtered), r/btc

Fetches all sources in parallel, deduplicates by source + title prefix, and returns
headlines sorted newest-first with age in minutes.

---

### `trump_watcher.py`
Polls Trump's Truth Social (primary) and Nitter/Twitter instances (fallback) every
30 seconds. On a new tweet, calls **Claude Haiku** to classify market impact:

| Classification | Meaning | Probability adjustment |
|---|---|---|
| `btc_bullish` | Direct crypto support, strategic reserve, deregulation | +0.05 to +0.25 |
| `usd_bearish` | Tariff inflation, Fed rate cut pressure, dollar weakness | +0.05 to +0.20 |
| `btc_bearish` | Anti-crypto statements, regulation threats | -0.05 to -0.25 |
| `usd_bullish` | Strong dollar stance, fiscal tightening | -0.05 to -0.20 |
| `neutral` | Sports, personal attacks, unrelated content | 0.00 |

Haiku is used here (not Sonnet) because this task runs 24/7 at 30-second intervals —
it is roughly 12× cheaper and fast enough for simple classification.

The result is written to `trump_state.json`. The probability model reads this file
automatically. Probability adjustments are hard-clamped to ±0.25 at both write time
and read time. The watcher is **optional** — everything else works without it.

---

### `probability_model.py`
Calls the Claude API to estimate the probability BTC moves up over the next ~4 hours.

Signal pipeline:
1. Technical score → raw probability (10%–90%)
2. Claude Sonnet-4-6 LLM call → contextual estimate from news + macro
3. 50/50 blend of technical and LLM (clamped to [0, 1])
4. Trump tweet adjustment applied on top (additive, clamped to 5%–95%)

Falls back gracefully at each step if any data source is unavailable.

---

### `trader.py`
The autonomous daemon. Run it in a terminal and leave it running.

Each cycle (every 10s by default in 15M mode, configurable):
1. Fetch price data (Kraken 15m candles in KXBTC15M mode; CoinGecko 30-day hourly otherwise)
2. Compute indicators and generate a signal
3. Refresh Kalshi portfolio (balance + open positions)
4. Run stop-loss checks — close any position down more than `stop_loss_pct`
5. **Profit-take check (15M)** — sell at market if current bid ≥ 80¢
6. **Signal-reversal exit (15M)** — close position if signal flips direction
7. Check all guard rails (enabled? cooldown? risk limit? already in this contract?)
8. Select the best Kalshi BTC market and size the order
9. Place the order (or log it in dry-run mode)
10. Write everything to `trading_state.json`

**15M position deduplication:** The daemon tracks the active contract ticker and
expiry in state. It will not place a second order on the same contract within the
same window, even if the Kalshi positions API has not yet reflected the first fill.

**`only_on_change: true` is strongly recommended** for 15M mode — it prevents
placing multiple orders on a flat signal (the single biggest source of fee drain).

---

### `dashboard.py`
Long-term trading control UI. Auto-refreshes every 30 seconds.

- 7-day OHLC candlestick chart with Bollinger Bands, SMA20/50, EMA20, EMA200,
  RSI subplot, and MACD subplot
- Live Kalshi market odds table
- Portfolio snapshot: balance, open positions, unrealized P&L
- Trading configuration form (enable/disable, risk limits, stop-loss, cooldown)
- **🚨 Emergency Controls panel**: cancel all resting orders and market-sell every
  open position with one click (two-step confirmation required). Also disables the
  daemon automatically.

---

### `monitor.py`
Short-term market monitor. Auto-refreshes every 15 seconds.

- 1m / 5m / 15m Kraken candlestick charts with EMAs, VWAP, Bollinger Bands,
  RSI, and MACD
- Short-term technical signal badge (BUY / SELL / HOLD + score)
- LLM probability gauge (Claude API estimate, refreshes every 5 minutes)
- Live BTC news feed (last 3 hours, up to 20 headlines)
- Trump signal card — shows latest tweet impact, urgency, and probability adjustment
- Fear & Greed Index + trader daemon status

---

## Setup

**1. Install dependencies**
```
pip install -r requirements.txt
```

**2. Configure `.env`**
```
GECKO_API         = your_coingecko_demo_key
KALSHI_API_KEY    = your_kalshi_uuid
KALSHI_PRIV       = 'your_rsa_private_key_base64'
ANTHROPIC_API_KEY = your_anthropic_key
```

**3. Start the trader daemon** (terminal 1)
```
python trader.py
```
Creates `trading_config.json` with `dry_run: true` and `enabled: false` on first run.
No orders will be placed until you explicitly enable them.

**4. Start the long-term dashboard** (terminal 2)
```
streamlit run dashboard.py
```
Open `http://localhost:8501`

**5. Start the short-term monitor** (terminal 3)
```
streamlit run monitor.py --server.port 8502
```
Open `http://localhost:8502`

**6. Start the Trump tweet watcher** (terminal 4, optional)
```
python trump_watcher.py
```
Runs silently in the background. Writes to `trump_state.json` and `trump_watcher.log`.
The probability model and monitor pick up its output automatically — no restart needed.

---

## Position Sizing

The three config fields that control how much money is at risk:

**`max_contracts`** — how many contracts per single trade. At ~50 cents per contract
on average, `max_contracts=2` costs about $1 per trade.

**`max_open_risk_usd`** — the hard cap on total open exposure across all positions
simultaneously. The daemon will not place new orders once this is reached. This is
your primary bankroll protection. In 15M mode this uses state-tracked cost when the
Kalshi positions API has not yet caught up.

**`stop_loss_pct`** — exits a position early when its unrealised loss exceeds this
percentage of what you paid. At 0.35, a $1.00 position is cut when it falls to $0.65.

### Recommended settings by bankroll

| Bankroll | max_contracts | max_open_risk_usd | stop_loss_pct | cooldown_minutes |
|---|---|---|---|---|
| < $50    | 1             | $3                | 0.30          | 30               |
| $200     | 3             | $20               | 0.35          | 30               |
| $500     | 5             | $50               | 0.40          | 15               |
| $1,000+  | 8             | $100              | 0.40          | 15               |

The rule of thumb: `max_open_risk_usd` should be 10% of your total bankroll. Even a
complete wipeout of all open positions costs you at most 10%, and you keep trading.

The defaults in the codebase (`max_contracts=2`, `max_open_risk_usd=5.0`) are set for
minimal liquidity. Adjust them in the dashboard config form as your account grows.

---

## Going Live

Before enabling real trading, run in dry-run mode for at least several cycles to
confirm signals look correct and order sizing is reasonable.

When ready:
1. Open `http://localhost:8501`
2. Scroll to **Trading Configuration**
3. Uncheck **Dry Run** → check **Enable live trading**
4. Set `max_contracts` and `max_open_risk_usd` for your bankroll (see table above)
5. Ensure `only_on_change` is checked
6. Click **Save Configuration**

The daemon picks up the change within one cycle.

**Emergency stop:** Open the **🚨 Emergency Controls** panel in the dashboard —
it cancels all resting orders and closes all positions at market in one click.
Or set `"enabled": false` directly in `trading_config.json` (takes effect within
one cycle, but does not close existing positions).

---

## How the Signal Maps to Kalshi

Kalshi BTC markets are binary: *"Will Bitcoin be above $X on [date]?"*

| Signal | Side | Logic |
|---|---|---|
| STRONG BUY / BUY | Buy YES | Expect BTC to rise above the strike |
| STRONG SELL / SELL | Buy NO | Expect BTC to stay below the strike |
| HOLD | No order | No clear edge — stay flat |

The market selector targets contracts where the relevant side is priced 20–80 cents.
Contracts priced outside this range (near-certain outcomes) are skipped — the
risk/reward is poor.

---

## Known Limitations

**Kalshi positions API lag (15M mode)**
After a binary option order fills, the Kalshi `/portfolio/positions` endpoint may
not reflect the new position for several seconds. The daemon works around this by
tracking ordered tickers in its own state, but the stop-loss check still relies on
the API. If the API is slow, a losing contract may not be stopped out mid-life.
The primary protection in 15M mode is the `max_open_risk_usd` cap (enforced via
state-tracked cost) and the signal-reversal early exit.

**Binary options and stop-loss**
Stop-loss for binary options works differently from continuous markets. The contract
value moves between 0¢ and 100¢ based on current market probability — the stop-loss
triggers if the market price drops far enough from your entry. However, binary
options can settle at exactly zero with very little warning in the final minutes.
The 80¢ profit-take is more reliable protection than waiting for stop-loss to fire.

**Signal bias in strong trends**
The short-term signal can become one-sided in persistent trends (MACD and EMA cross
both stay positive for hours). `only_on_change: true` mitigates this by suppressing
repeated orders on an unchanged signal.

---

## Risk Warnings

- This system places real financial bets on Kalshi using your account funds.
- Past indicator signals do not guarantee future performance.
- Kalshi binary options can expire worthless — you can lose 100% of what you bet on a single contract.
- Start in dry-run mode and validate the system over multiple cycles before going live.
- Always monitor `trader.log` when live trading is active.
- Never set `max_open_risk_usd` above 15% of your total available capital.
- Use the 🚨 Emergency Controls panel in the dashboard to close all positions instantly if needed.
