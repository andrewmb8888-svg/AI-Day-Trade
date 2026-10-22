"""Coordinated multi-symbol decisions; cash is shared across the watchlist."""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import logging

import pandas as pd

from common import (BY_SYMBOL, TERMINAL, UTC, money, order_key, signals, text_enum,
                    timestamp)
from cash import budget, reservations
from market_data import expected_bar

LOG = logging.getLogger("AI Day Trade V2")
ZERO = Decimal("0")
CENT = Decimal("0.01")


def group_exposures(snapshot, store, now):
    """Approximate daily exposure; overlapping themes can still be correlated."""
    result = {}
    for position in snapshot.positions:
        if money(position.qty) <= 0:
            raise ValueError("Short/zero positions require account review")
        spec = BY_SYMBOL.get(position.symbol)
        if spec:
            value = abs(money(position.market_value)) * spec.leverage
            result[spec.group] = result.get(spec.group, ZERO) + value
    for pending in reservations(snapshot, store).values():
        spec = BY_SYMBOL.get(pending["symbol"])
        if spec:
            result[spec.group] = result.get(spec.group, ZERO) + money(pending["cost"]) * spec.leverage
    # A just-filled buy can disappear from orders before positions catch up.
    # Temporarily counting it twice when already present is conservative.
    for fill in snapshot.fills:
        spec = BY_SYMBOL.get(fill.symbol)
        if spec and fill.side == "buy" and fill.at >= snapshot.started_at - timedelta(seconds=30):
            result[spec.group] = result.get(spec.group, ZERO) + fill.gross * spec.leverage
    return result


def entry_prices(quote, now, settings):
    ask, bid = money(quote.ask_price), money(quote.bid_price)
    age = (now - timestamp(quote.timestamp)).total_seconds()
    if not 0 <= age <= 120:
        raise ValueError("Stale or future quote")
    if not 0 < bid <= ask or ask < 1 or (ask - bid) / ask > Decimal("0.01"):
        raise ValueError("Invalid quote, stock below $1, or spread over 1%")
    limit = (ask * Decimal("1.001")).quantize(CENT, rounding=ROUND_CEILING)
    stop = (limit * (1 - settings.stop_loss)).quantize(CENT, rounding=ROUND_FLOOR)
    take = (limit * (1 + settings.take_profit)).quantize(CENT, rounding=ROUND_CEILING)
    if not 0 < stop < bid <= limit < take:
        raise ValueError("Invalid bracket prices")
    return limit, stop, take


class Engine:
    def __init__(self, broker, market_data, settings, store, clock=None):
        self.broker, self.market_data = broker, market_data
        self.settings, self.store = settings, store
        self.decisions = {}
        self.clock = clock or (lambda: datetime.now(UTC))

    def reconcile_intents(self):
        # Do this before the cash snapshot so fills seen by lookup precede the
        # balance/fill/order reconciliation used for new spending decisions.
        for intent in self.store.active():
            order = self.broker.lookup(intent["client_id"])
            if order is not None:
                self.store.observe(order)
            else:
                LOG.warning("%s: submission unresolved; cash remains reserved (%s)",
                            intent["symbol"], intent["client_id"])

    def step(self, now):
        market_clock = self.broker.clock()
        if not market_clock.is_open:
            return {"market": "closed"}
        if not self.broker.cash_settings():
            raise ValueError("Broker cash-only configuration changed; new orders disabled")
        self.reconcile_intents()
        snapshot = self.broker.snapshot(now)
        if any(getattr(snapshot.account, field, False) for field in
               ("trading_blocked", "account_blocked", "trade_suspended_by_user")):
            raise ValueError("Account is blocked or suspended; trading disabled")
        for order in snapshot.orders:
            self.store.observe(order)
        calendar = self.broker.calendar(now)
        available = budget(snapshot, self.store, calendar, max(now, self.clock()), self.settings.cash_buffer)
        exposure = group_exposures(snapshot, self.store, now)
        expected = expected_bar(now, calendar)
        if expected is None:
            return {"market": "waiting for first completed regular-session 30-minute bar"}
        data = self.market_data.update(now, calendar)
        orders_by_symbol = {}
        for order in snapshot.orders:
            if text_enum(order.status) not in TERMINAL:
                orders_by_symbol.setdefault(order.symbol, []).append(order)
        positions = {position.symbol: position for position in snapshot.positions}
        active_symbols = {item["symbol"] for item in self.store.active()}
        results, candidates = {}, []

        for symbol in self.settings.symbols:
            try:
                self.broker.asset(symbol)
                if symbol not in data:
                    results[symbol] = self.market_data.errors.get(symbol, "market data unavailable")
                    continue
                bars, daily = data[symbol]
                if bars.empty or bars.index[-1] != expected:
                    results[symbol] = "waiting for current completed bar"
                    continue
                signature = (bars.index[-1], daily.index[-1] if not daily.empty else None,
                             daily.attrs.get("current_session_data", False))
                old = self.decisions.get(symbol)
                if old is None or old[0] != signature:
                    self.decisions[symbol] = (signature, signals(bars, daily, self.settings.window))
                decision = self.decisions[symbol][1]
                if symbol in positions:
                    if decision.exit_signal and not orders_by_symbol.get(symbol) and symbol not in active_symbols:
                        closing_at = self.clock()
                        session_end = getattr(market_clock, "next_close", None)
                        if ((session_end is not None and closing_at >= timestamp(session_end)) or
                                expected_bar(closing_at, calendar) != expected):
                            results[symbol] = "bar/session changed during cycle; waiting for fresh data"
                            continue
                        result = self.broker.close(symbol)
                        results[symbol] = "close submitted" if result is not None else "already closed"
                        # Sale proceeds never increase this cycle's budget.
                    else:
                        results[symbol] = "holding; broker protective orders retained"
                    continue
                if orders_by_symbol.get(symbol) or symbol in active_symbols:
                    results[symbol] = "pending order or unresolved submission"
                    continue
                if not decision.entry_key:
                    results[symbol] = "no entry signal"
                    continue
                client_id = order_key(symbol, decision.entry_key)
                if self.store.consumed(client_id):
                    results[symbol] = "signal already submitted"
                    continue
                existing = self.broker.lookup(client_id)
                if existing is not None:
                    # A broker-held ID survives loss of an in-memory cache.
                    results[symbol] = "signal already exists at broker"
                    continue
                candidates.append((symbol, client_id))
            except Exception as exc:
                LOG.exception("%s: evaluation skipped", symbol)
                results[symbol] = f"skipped: {type(exc).__name__}"

        if not candidates or available <= 0:
            for symbol, _ in candidates:
                results[symbol] = "no eligible cash available"
            return results
        quotes = self.broker.quotes([symbol for symbol, _ in candidates])
        market_clock = self.broker.clock()
        if not market_clock.is_open:
            results.update({symbol: "market closed before submission" for symbol, _ in candidates})
            return results
        equity = money(snapshot.account.equity)
        if equity <= 0:
            raise ValueError("Account equity must be positive")
        for symbol, client_id in candidates:
            try:
                spec = BY_SYMBOL[symbol]
                sending_at = self.clock()
                close_at = getattr(market_clock, "next_close", None)
                if ((close_at is not None and sending_at >= timestamp(close_at)) or
                        expected_bar(sending_at, calendar) != expected):
                    results[symbol] = "bar/session changed during cycle; waiting for fresh data"
                    continue
                limit, stop, take = entry_prices(quotes[symbol], sending_at, self.settings)
                group_room = max(ZERO, equity * self.settings.group_cap - exposure.get(spec.group, ZERO))
                dollars = min(available, equity * self.settings.allocation / spec.leverage,
                              group_room / spec.leverage)
                qty = int((dollars / limit).to_integral_value(rounding=ROUND_FLOOR))
                if qty < 1:
                    results[symbol] = "cash/allocation/exposure budget below one share"
                    continue
                cost = limit * qty
                # Durable reservation precedes the HTTP request. This remains
                # deducted for the WHOLE cycle even if the response says filled.
                self.store.reserve(client_id, symbol, qty, limit, spec.group, spec.leverage, now)
                available -= cost
                exposure[spec.group] = exposure.get(spec.group, ZERO) + cost * spec.leverage
                try:
                    order = self.broker.buy(symbol, qty, client_id, limit, stop, take)
                except Exception as exc:
                    if self.broker.definitive_rejection(exc):
                        self.store.reject(client_id)
                    # Unknown outcomes keep their reservation across restarts.
                    raise
                self.store.observe(order)
                LOG.info("BUY submitted: %s qty=%s limit=%s stop=%s take=%s client_id=%s status=%s",
                         symbol, qty, limit, stop, take, client_id, text_enum(order.status))
                results[symbol] = f"buy submitted: {qty} shares; status={text_enum(order.status)}"
            except Exception as exc:
                LOG.exception("%s: order skipped/failed; reconciliation required before retry", symbol)
                results[symbol] = f"skipped/failed: {type(exc).__name__}"
        return results
