"""AI Day Trade V2: cash-backed, multi-symbol, 30-minute paper-trading bot."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import tempfile
import time

from dotenv import load_dotenv

from broker import Broker
from cash import IntentStore, budget
from common import BY_SYMBOL, UTC, Settings, instance_lock, money
from engine import Engine
from market_data import MarketData

LOG = logging.getLogger("AI Day Trade V2")
ROOT = Path(__file__).resolve().parent


def state_path(account_id):
    token = hashlib.sha256(str(account_id).encode()).hexdigest()[:20]
    return ROOT / f".v2-state-{token}.json"


def connection_check(broker, market_data, settings):
    """Only read broker/data endpoints; never configure, submit, or cancel."""
    now = datetime.now(UTC)
    clock = broker.clock()
    snapshot = broker.snapshot(now)
    if any(getattr(snapshot.account, field, False) for field in
           ("trading_blocked", "account_blocked", "trade_suspended_by_user")):
        raise ValueError("Account is blocked or suspended")
    store = IntentStore(state_path(snapshot.account.id), snapshot.account.id, settings.paper)
    calendar = broker.calendar(now)
    cash = budget(snapshot, store, calendar, datetime.now(UTC), settings.cash_buffer)
    guard = broker.cash_settings()
    data = market_data.update(now, calendar, force_history=True)
    quotes = broker.quotes(list(settings.symbols))
    failures = []
    for symbol in settings.symbols:
        try:
            broker.asset(symbol)
            if symbol not in data:
                raise ValueError(market_data.errors.get(symbol, "No market data"))
            bars, daily = data[symbol]
            if len(bars) < 100 or len(daily) < 200:
                raise ValueError("Insufficient 30-minute or daily history")
            if not daily.attrs.get("current_session_data", False):
                raise ValueError("Daily history is missing the previous exchange session")
            quote = quotes[symbol]
            if not 0 < money(quote.bid_price) <= money(quote.ask_price):
                raise ValueError("No usable bid/ask quote")
            LOG.info("%s: data check passed; 30-minute bars=%d; daily bars=%d", symbol, len(bars), len(daily))
        except Exception as exc:
            failures.append(symbol)
            LOG.error("%s: %s", symbol, str(exc))
    LOG.info("READ-ONLY check: %d/%d symbols passed; market_open=%s; cash-policy budget=$%.2f; cash-only guard=%s",
             len(settings.symbols) - len(failures), len(settings.symbols), clock.is_open, cash, guard)
    if not guard:
        LOG.info("Trading startup will set broker max_margin_multiplier=1 and no_shorting=True; --check changed nothing")
    if store.active():
        LOG.warning("%d unresolved/pending cash intents remain reserved; review --status if unexpected", len(store.active()))
    if failures:
        raise SystemExit("Some symbols are not ready; see the messages above")


def main(argv=None):
    # Reuse the user's existing ignored V1 .env when V2 has no local .env.
    local_env = ROOT / ".env"
    load_dotenv(local_env if local_env.exists() else ROOT.parent / ".env")
    parser = argparse.ArgumentParser(prog="AI Day Trade V2", description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only connectivity, data and cash-policy check")
    mode.add_argument("--once", action="store_true", help="Run one trading cycle; may submit orders")
    mode.add_argument("--list", action="store_true", help="List configured symbols; no credentials required")
    mode.add_argument("--status", action="store_true", help="Read-only pending reservation journal summary")
    args = parser.parse_args(argv)
    settings = Settings.from_environment()
    if args.list:
        print("AI Day Trade V2 — configured watchlist")
        for symbol in settings.symbols:
            item = BY_SYMBOL[symbol]
            print(f"{symbol:5s}  {item.name}  [{item.group}, sizing multiplier={item.leverage}x]")
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), RotatingFileHandler(
                            ROOT / "trade_v2_log.txt", maxBytes=5_000_000, backupCount=3)])
    LOG.info("AI Day Trade V2 | %s | %d symbols | completed 30-minute bars | %.1f-second checks | cash-backed",
             "PAPER" if settings.paper else "LIVE", len(settings.symbols), settings.interval)
    broker = Broker(settings)
    market_data = MarketData(broker.data, settings)
    if args.check:
        connection_check(broker, market_data, settings)
        return
    account = broker.account()
    store_file = state_path(account.id)
    if args.status:
        store = IntentStore(store_file, account.id, settings.paper)
        print(f"AI Day Trade V2: {len(store.active())} pending/unresolved cash intents")
        for item in store.active():
            print(f"{item['symbol']} reserved=${item['reserved']} status={item['status']} client_id={item['client_id']}")
        return
    token = hashlib.sha256(f"{account.id}:{settings.paper}".encode()).hexdigest()[:24]
    with ExitStack() as stack:
        # Covers other V2 copies on this computer, including different folders.
        stack.enter_context(instance_lock(Path(tempfile.gettempdir()) / f"ai-day-trade-account-{token}.lock"))
        # Cooperate with V1 when both versions live in this downloaded checkout.
        if (ROOT.parent / "trading_bot.py").exists():
            stack.enter_context(instance_lock(ROOT.parent / f".bot-{'paper' if settings.paper else 'live'}.lock"))
        store = IntentStore(store_file, account.id, settings.paper)
        LOG.info("Enforcing account-wide cash guard: max_margin_multiplier=1, no_shorting=True")
        broker.enforce_cash_only()
        engine = Engine(broker, market_data, settings, store)
        previous = None
        last_report = 0.0
        while True:
            started = time.monotonic()
            try:
                results = engine.step(datetime.now(UTC))
                summary = dict(Counter(results.values()))
                if summary != previous or started - last_report >= 300 or args.once:
                    LOG.info("Cycle: %s", summary)
                    previous, last_report = summary, started
            except Exception:
                LOG.exception("Cycle failed; no new blind orders. Broker-held exits remain independent.")
                if args.once:
                    raise
            if args.once:
                return
            # No overlapping cycles. Slow API responses can extend the interval.
            time.sleep(max(0, settings.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.info("AI Day Trade V2 stopped. Existing broker orders and positions remain active.")
