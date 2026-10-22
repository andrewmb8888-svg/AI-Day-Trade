# RSI/MACD Alpaca paper-trading bot

**AI Day Trade V2** is available separately in [AI-Day-Trade-V2](AI-Day-Trade-V2/README.md).
It uses a 20-symbol watchlist, completed 30-minute bars, 10-second checks, and a
shared cash-backed budget. Stop V1 before starting V2 on the same account.
The original `trading_bot.py` and its strategy are preserved below.

This replaces the uploaded educational notebook export with an executable,
long-only Python application. It does not use AI or establish profitability.

## Install and test

Use Python 3.12 or 3.13 on Windows, Linux, or macOS.

Use the existing checkout; no separate worktree is needed.

On Windows, open Command Prompt in the extracted project folder:

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m unittest -v
if not exist .env copy .env.example .env
```

Enter your paper credentials in `.env`, then run the read-only connection check:

```bat
.venv\Scripts\python.exe trading_bot.py --check
```

After that succeeds, start paper trading with:

```bat
.venv\Scripts\python.exe trading_bot.py
```

On Linux or macOS:

```sh
cd AI-Day-Trade  # or the directory where you extracted this project
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest -v
cp .env.example .env  # only if .env does not already exist
```

Enter paper-account credentials in the ignored `.env` file. Do not commit keys.
First run `.venv/bin/python trading_bot.py --check` for a read-only check of
credentials, account access, symbol eligibility, and market data. It never places
or cancels orders. It checks connectivity even while the market is closed;
entry-time quote freshness is enforced separately in the trading loop.
Run `.venv/bin/python trading_bot.py` to start; Ctrl-C stops it.
`.venv/bin/python trading_bot.py --once` runs one cycle and CAN submit a paper
order when entry conditions are met. Logs appear in the terminal and trade_log.txt (rotated at 5 MB; three backups).
The process remains alive overnight and checks the broker clock each minute.
Run it under a process supervisor for unattended use; this repository does not
configure hosting, monitoring, or automatic restart. Platform-specific file locking
prevents a second local instance; do not run other copies against the same account.

## Behavior

- Trades TQQQ by default; change SYMBOL in `.env` if desired.
- Uses split-adjusted IEX data (not the consolidated market feed).
- Excludes incomplete hourly bars and the current session's daily bar. Uses
  Alpaca's exchange calendar to retain only full hourly intervals within regular
  sessions, including early closes. Intervals overlapping the opening or closing
  boundary are excluded; this may omit part of the session depending on the
  provider's hourly alignment. No premarket or after-hours intervals drive signals.
- Requires the daily history to include the previous exchange session for entries;
  stale or insufficient daily history disables entries without disabling hourly exits.
- Requires at least 200 daily and 100 hourly observations, with daily
  50 > 100 > 200 moving averages for entry.
- Uses Wilder RSI(14) and MACD(12,26,9). Entry requires RSI crossing above 30
  and MACD crossing above its signal, both within the most recent six completed
  bars (at most five bars apart).
- Allocates at most 2% of equity per entry, capped by available buying power.
  This differs from the original leveraged buying-power allocation.
- Places a GTC limit entry at ask + 0.1%, with a broker-held bracket: stop 3%
  below the entry limit and take-profit 6% above it. These defaults are explicit
  strategy changes. Stops do not guarantee execution price or cap account loss;
  bracket legs activate after the entry fills and normally operate in regular hours.
- Pending orders prevent new entries. Unfilled GTC entries can remain overnight;
  monitor them in Alpaca if no longer wanted. Automatic expiry is not implemented:
  canceling a bracket during a partial-fill race can also remove protection.
  Review fills and remaining exit protection before manually canceling a bracket.
  This limitation means the bot is not intended for unattended live deployment.
- RSI crossing below 65 plus a bearish MACD event gives an indicator exit.
  **When a position has active orders, these orders are retained and indicator
  exits are deferred.** Positions opened by this bot therefore normally exit by
  the bracket. It does not cancel stops to force an indicator exit.
- Existing long positions without orders may be sold by the indicator exit,
  through Alpaca's close-position endpoint, including fractional quantities. Use a dedicated paper account: manually
  opened positions in the same symbol are not isolated from this strategy.
- Deterministic IDs based on the RSI/MACD event timestamps and broker lookups reconcile restarts and
  ambiguous submission errors. Rejected/canceled orders are not retried for the
  same signal pair, including on later bars after the position closes. Position/account/API failures abort that cycle rather than assuming
  success or an empty account. Orders are read before positions to reduce fill
  races, and indicator exits ask the broker to close its current position rather
  than selling a cached quantity. Both API clients use 10-second connect and
  30-second read timeouts; transient failures are reconciled on the next cycle. No order fill is implied by a submission log.

Paper mode is the default. PAPER is parsed as a Boolean. Live mode additionally
requires ALPACA_API_KEY, ALPACA_SECRET_KEY and
ALLOW_LIVE_TRADING=I_UNDERSTAND. Custom TRADE_API_URL overrides are rejected.
Do not enable live mode before observing paper entries, partial fills, exits,
restarts, and disconnections. Tests simulate the broker; they do not prove actual
Alpaca connectivity, fill behavior, or investment returns.

## Validation and operating limits

The automated suite uses synthetic prices and simulated HTTP/broker responses.
It covers RSI smoothing, stable signal identity across bars and restarts, daily
trend-history failures, regular-session filtering, duplicate orders, timeout
reconciliation, close-position races, request timeouts, and read-only checks.
Tests make no real brokerage requests and place no orders.

Before relying on execution, run `--check` with your own paper credentials, then
observe paper entries, unfilled and partial orders, bracket exits, restarts, and
network interruptions. Paper fills do not establish live execution quality or
profitability. Monitor unfilled GTC entries and overnight exposure; the strategy
does not close positions at the end of each day. Existing manual positions in the
same symbol are managed by this strategy, so use a dedicated account.
