# AI Day Trade V2

A separate, long-only, multi-symbol Alpaca bot for **paper testing**. Version 1
remains in the parent folder as `trading_bot.py`.

V2 checks the account and orders about every **10 seconds**, evaluates completed
**30-minute regular-session bars**, and shares one cash-backed spending budget
across the watchlist. It has **no daily trade-count quota**. A fresh signal,
eligible cash, and the configured dollar exposure limits are required for entry.

## Start from your existing Windows installation

1. Stop Version 1 with **Ctrl+C**. This stops its program; existing orders and
   positions remain at Alpaca. Review those in the paper dashboard.
2. Place this `AI-Day-Trade-V2` folder **inside your existing project folder**,
   alongside `trading_bot.py`, `.env`, and `.venv`.
3. Open Command Prompt in that existing project folder. Run these one at a time:

```bat
.venv\Scripts\python.exe -m pip install -r AI-Day-Trade-V2\requirements.txt
.venv\Scripts\python.exe -m unittest discover -s AI-Day-Trade-V2 -v
.venv\Scripts\python.exe AI-Day-Trade-V2\ai_day_trade_v2.py --list
.venv\Scripts\python.exe AI-Day-Trade-V2\ai_day_trade_v2.py --check
```

`--list` lists symbols without connecting to a broker. **`--check` reads broker
and data endpoints only: it never submits/cancels orders or changes account
settings.** It also works outside market hours. It reports each symbol's data
readiness, a conservative cash-policy budget, and whether the cash-only broker
settings are already enabled. Quote freshness is enforced separately when
placing an entry.

After the check succeeds, start V2:

```bat
.venv\Scripts\python.exe AI-Day-Trade-V2\ai_day_trade_v2.py
```

The program prints **AI Day Trade V2** and **PAPER** at startup. Leave that window
open, keep the computer awake and connected, and monitor Alpaca's paper dashboard.
Use Ctrl+C to stop it. Existing positions and broker orders remain after stopping.

V2 first looks for `.env` inside this folder. If none exists, it reads the `.env`
in the parent folder, so you can reuse your current paper keys. `SYMBOL=TQQQ`
is a V1 setting; V2 uses its own 20-name default or `V2_SYMBOLS`.
`V2_SYMBOLS` can select a subset of the reviewed list. Adding other instruments
requires reviewing their exposure group and leverage metadata in `common.py`.
Do not upload your real `.env` or keys to GitHub or chat.

## Fresh installation / Linux / macOS

Use Python **3.12 or 3.13** with the pinned dependencies. From this folder:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -v
```

Create `.env` from `.env.example` only if it does not already exist. Enter your
paper keys locally. Then run:

```sh
.venv/bin/python ai_day_trade_v2.py --check
.venv/bin/python ai_day_trade_v2.py
```

For a fresh Windows installation, use `python -m venv .venv` and replace
`.venv/bin/python` with `.venv\Scripts\python.exe` in those commands.

## Cash-only policy and account settings

**At trading startup, V2 sets Alpaca's account-wide `max_margin_multiplier` to
`1` and `no_shorting` to `True`, preserving other account configuration fields.
Only these two fields are sent, and a fresh read verifies the settings before
entering the trading loop.** This is an
account-wide change, not a separate subaccount or a conversion to a formal cash
account. If the broker does not confirm it, V2 refuses to trade. `--check` and
`--status` never make this change.

V2 does not size entries from margin buying power. It uses the smallest positive
bound from cash, non-marginable buying power, and broker buying power, then:

- Reserves remaining limit-order cost for all account-wide pending buys.
- Holds local reservations for submissions whose outcome is unknown.
- Holds recent sale proceeds through the **entire expected T+1 settlement day**,
  using exchange sessions and federal bank holidays. A normal Monday sale is
  therefore withheld through Tuesday and becomes eligible Wednesday, assuming
  no holiday and that the broker's balances support it.
- Accounts for buys filling during reconciliation, reported fees, outgoing
  transfers, and an additional 5% haircut on the cycle's calculated budget.

The haircut is a per-cycle spending cushion, not a guaranteed 5% portfolio cash
floor. Some broker reservations may be counted conservatively twice, so the
usable budget can be lower than the dashboard's cash balance. Positive cash
alone is not treated as settled sale proceeds.

Alpaca does not expose an explicit settled-cash field in the pinned SDK. REST
snapshots are not atomic. Use a **dedicated paper account with no concurrent V1,
other bots, or manual trading/withdrawals** while V2 is operating. These controls
reduce unintended borrowing; they cannot provide an absolute guarantee against
external account activity, stale broker state, fees, or broker errors. Account
and instrument restrictions still apply.

An unsupported open market-buy order disables new entries until it is resolved,
because its maximum cost cannot be established from a limit price. Non-stock
assets, short positions, malformed responses, and incomplete activity/order
coverage also disable spending. Existing broker-held protection remains at Alpaca.

## Full default watchlist

| Symbol | Company/fund | Exposure group | Sizing multiplier |
|---|---|---|---|
| NVDA | Nvidia | Semiconductors | 1 |
| AMD | Advanced Micro Devices | Semiconductors | 1 |
| SOXL | Direxion Semiconductor Bull | Semiconductors | 3 |
| TSLA | Tesla | EV / growth | 1 |
| TSLL | Direxion Tesla Bull | EV / growth | 2 |
| COIN | Coinbase | Crypto-related | 1 |
| IBIT | iShares Bitcoin Trust | Crypto-related | 1 |
| XBI | SPDR S&P Biotech | Biotech | 1 |
| LABU | Direxion Biotech Bull | Biotech | 3 |
| OXY | Occidental Petroleum | Energy | 1 |
| XOP | SPDR Oil & Gas Exploration & Production | Energy | 1 |
| GUSH | Direxion Oil & Gas Bull | Energy | 2 |
| NEM | Newmont | Gold miners | 1 |
| GDX | VanEck Gold Miners | Gold miners | 1 |
| NUGT | Direxion Gold Miners Bull | Gold miners | 2 |
| IWM | iShares Russell 2000 | Small companies | 1 |
| TNA | Direxion Small Cap Bull | Small companies | 3 |
| JPM | JPMorgan Chase | Financials | 1 |
| KRE | SPDR S&P Regional Banking | Financials | 1 |
| FAS | Direxion Financial Bull | Financials | 3 |

Membership means eligible for evaluation, not an instruction to buy everything.
Symbols must pass broker eligibility and market-data checks. This is a research
universe, not a verified current volatility or profitability ranking. Verify
current issuer objectives and instrument eligibility before relying on the list.
Leveraged funds target daily multiples; longer holding-period returns differ.
Groups are approximate, not a measurement of all correlations or ETF constituents.

## Strategy, sizing, and execution

- Uses **IEX**, not consolidated all-exchange market data.
- The first normal completed bar is 9:30-10:00 a.m. New York time. V2 waits three
  seconds for publication and retries late data without inventing a signal.
- Uses Wilder RSI(14), MACD(12,26,9), and recent events at most five bars apart.
  Both entry events must still be within the recent six completed bars.
- Entry requires daily 50 > 100 > 200 moving averages and current previous-session
  daily data. Insufficient/stale daily data prevents entry without suppressing a
  valid 30-minute exit signal.
- Each symbol has independent indicator/cache state. Event-based order IDs stop
  the same setup buying again after a restart or a quick exit. A genuinely new
  setup may trade again; there is no trades-per-day quota.
- Full 60-day intraday and 500-day daily histories refresh each New York session
  so split-adjusted history uses one price basis. Intraday updates are batched
  and incremental thereafter. Completed bars are cached between checks.
- Default per-position approximate daily exposure is **2% of equity**. Cash
  allocation divides by the configured leverage: a 3x fund receives roughly
  0.67% of equity in dollars, subject to whole shares and other budget limits.
- Default approximate group exposure cap is **10% of equity**, including pending
  buys. This is a dollar exposure control, not a loss guarantee or a trade quota.
  Actual volatility/beta differs across securities; unknown holdings outside the
  reviewed list are not assigned a group, although their orders/cash affect the
  account-wide budget. Review such holdings before using this strategy.
- New entries are whole-share GTC limit brackets, with a limit at ask +0.1%, a
  **3% stop** below the limit, and a **6% take-profit** above it. Spread must be
  at most 1%; quotes must be recent. Stocks below $1 are unsupported.
- Active broker-held bracket exits take priority over indicator exits. V2 does
  not cancel protection to force an RSI/MACD close. An unprotected existing long
  in a watched symbol may be closed via Alpaca's close-position endpoint.
- **Unfilled GTC entries can remain overnight and fill after a signal ages.**
  Automatic cancellation is not implemented because cancellation and partial
  fills require a separate protection-reconciliation workflow. Monitor these
  orders in Alpaca and verify protection before changing them.
- Bracket exits generally become active after the parent completely fills and
  operate in regular hours. Partial positions and overnight gaps require
  monitoring. Stops do not guarantee execution price or cap losses at 3%.
- Positions can remain overnight; there is no end-of-day liquidation rule.

The 30-minute strategy and fixed exits have not been demonstrated profitable or
optimal for these symbols. Paper execution must be observed before any live use.

## State, recovery, and concurrent operation

Cash reservations are saved atomically before submission in an ignored,
account-bound `.v2-state-*.json` journal. Do not delete or manually edit it to
make more cash available. A submission timeout keeps its reservation even if an
immediate broker lookup returns 404. Corrupt/wrong-account state stops trading.

From the original Windows project folder, inspect outstanding reservations with:

```bat
.venv\Scripts\python.exe AI-Day-Trade-V2\ai_day_trade_v2.py --status
```

For an unresolved entry, compare its client ID with Alpaca's orders and fills.
Do not resubmit blindly or erase the journal. Broker rejection/filled/canceled
status releases the remaining reservation through reconciliation. A never-accepted
ambiguous intent can remain reserved and needs a reviewed recovery; this version
has no unsafe "clear all state" command.

Account-specific locks prevent concurrent V2 processes on the same computer.
When V2 is inside the original project, it also acquires V1's local lock. This
cannot coordinate copies on different machines or independently located V1
folders: stop those yourself. Keep one strategy in control of the account.

Logs rotate at 5 MB with three backups in `trade_v2_log.txt*`. A process
supervisor/hosting service is needed for unattended uptime; publication or file
copying does not keep a computer awake or start a process automatically.

## Validation

The offline tests simulate market data, HTTP transport, and broker state. They
exercise multiple simultaneous symbols, capital reservations, partial fills,
timeouts/restarts, duplicate signals, bank holidays, short/unsupported assets,
splits, session boundaries, cache reuse, broker configuration, and read-only CLI.
They do not place orders, establish profitability, or prove actual Alpaca fills.

`--once` runs a real trading cycle and **can submit orders**. It also enforces the
cash-only account configuration. It is not a dry-run or unit-test command.
Paper mode defaults to true. Live mode requires separate live credentials,
`PAPER=false`, and `ALLOW_LIVE_TRADING=I_UNDERSTAND`; it has not been validated.
