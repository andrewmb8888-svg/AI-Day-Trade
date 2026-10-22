"""Alpaca boundary for AI Day Trade V2; all spending uses account-wide reads."""
from __future__ import annotations

from copy import copy
from datetime import date, datetime, timedelta
import logging
import os
from zoneinfo import ZoneInfo

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetCalendarRequest, GetOrdersRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest,
)

from common import Fill, Snapshot, TERMINAL, TimeoutAdapter, UTC, flatten_orders, money, text_enum, timestamp

LOG = logging.getLogger(__name__)
OPEN_STATUSES = {
    "accepted", "pending_new", "accepted_for_bidding", "new", "partially_filled",
    "done_for_day", "pending_cancel", "pending_replace", "stopped", "suspended",
    "calculated", "held",
    "pending_review",
}


def _utc_now():
    return datetime.now(UTC)


def definitive_rejection(exc):
    """A documented request rejection, rather than an ambiguous transport error."""
    if not isinstance(exc, APIError) or exc.status_code not in {400, 403, 422}:
        return False
    message = str(exc).lower()
    # Duplicate-ID errors may refer to an earlier accepted order whose response
    # was lost. Reconcile that ID instead of releasing its cash reservation.
    return not ("client_order_id" in message and any(word in message for word in ("unique", "duplicate", "already")))


class Broker:
    definitive_rejection = staticmethod(definitive_rejection)

    def __init__(self, settings):
        self.settings = settings
        key_name = "ALPACA_PAPER_API_KEY" if settings.paper else "ALPACA_API_KEY"
        secret_name = "ALPACA_PAPER_SECRET_KEY" if settings.paper else "ALPACA_SECRET_KEY"
        key, secret = os.getenv(key_name), os.getenv(secret_name)
        if not key or not secret:
            raise ValueError("Set Alpaca credentials in .env; see .env.example")
        if not settings.paper and os.getenv("ALLOW_LIVE_TRADING") != "I_UNDERSTAND":
            raise ValueError("Live trading requires ALLOW_LIVE_TRADING=I_UNDERSTAND")
        if os.getenv("TRADE_API_URL"):
            raise ValueError("Remove TRADE_API_URL; V2 uses the selected standard Alpaca endpoint")
        self.trade = TradingClient(key, secret, paper=settings.paper)
        self.data = StockHistoricalDataClient(key, secret)
        for client in (self.trade, self.data):
            client._session.mount("https://", TimeoutAdapter())
        self._calendar = None
        self._calendar_day = None
        self._assets = {}
        self.disabled = {}

    def clock(self):
        return self.trade.get_clock()

    def account(self):
        return self.trade.get_account()

    def calendar(self, now):
        today = timestamp(now).astimezone(ZoneInfo("America/New_York")).date()
        if today != self._calendar_day:
            rows = self.trade.get_calendar(GetCalendarRequest(
                start=today - timedelta(days=110), end=today + timedelta(days=20)))
            dates = [row.date for row in rows]
            if (not dates or any(type(day) is not date for day in dates) or
                    len(dates) != len(set(dates)) or
                    min(dates) > today - timedelta(days=100) or max(dates) < today + timedelta(days=10)):
                raise ValueError("Exchange calendar does not cover the required history and settlement horizon")
            self._calendar = sorted(rows, key=lambda row: row.date)
            self._calendar_day = today
        return self._calendar

    def _asset_record(self, symbol):
        if not symbol or not isinstance(symbol, str):
            raise ValueError("A stock symbol is required")
        if symbol not in self._assets:
            self._assets[symbol] = self.trade.get_asset(symbol)
        asset = self._assets[symbol]
        if text_enum(asset.asset_class) != AssetClass.US_EQUITY.value:
            raise ValueError("V2 supports only US stocks and stock-listed ETFs in this account")
        return asset

    def asset(self, symbol):
        asset = self._asset_record(symbol)
        if not asset.tradable or text_enum(asset.status) != "active":
            raise ValueError("Instrument is not active and tradable")
        return asset

    def assets(self):
        result = {}
        self.disabled = {}
        for symbol in self.settings.symbols:
            try:
                result[symbol] = self.asset(symbol)
            except ValueError as exc:
                self.disabled[symbol] = str(exc)
                LOG.warning("%s disabled: %s", symbol, exc)
            except APIError as exc:
                if exc.status_code != 404:
                    raise
                self.disabled[symbol] = "Instrument unavailable from broker"
                LOG.warning("%s disabled: instrument unavailable from broker", symbol)
        return result

    def cash_settings(self):
        config = self.trade.get_account_configurations()
        return money(config.max_margin_multiplier) == 1 and config.no_shorting is True

    def enforce_cash_only(self):
        """Explicit trading startup policy; never used by the read-only check."""
        config = self.trade.get_account_configurations()
        if money(config.max_margin_multiplier) != 1 or config.no_shorting is not True:
            LOG.warning("Applying account-wide policy: max margin multiplier=1; no shorting")
            updated = config.model_copy(update={"max_margin_multiplier": "1", "no_shorting": True})
            self.trade.set_account_configurations(updated)
        verified = self.trade.get_account_configurations()
        if money(verified.max_margin_multiplier) != 1 or verified.no_shorting is not True:
            raise ValueError("Broker cash-only/long-only restrictions were not confirmed; trading disabled")
        return verified

    def fills(self, now):
        """Read every fill in a fixed 14-calendar-day window, including old GTC fills."""
        until = timestamp(now)
        start = until - timedelta(days=14)
        params = {"after": start.isoformat(), "until": until.isoformat(),
                  "direction": "asc", "page_size": 100}
        found = {}
        cursors = set()
        # Defensive coverage bound; exceeding it disables entries, never silently truncates.
        for _ in range(1000):
            page = self.trade.get("/account/activities/FILL", data=dict(params))
            if not isinstance(page, list) or len(page) > 100:
                raise ValueError("Invalid fill-activity response; coverage cannot be confirmed")
            for item in page:
                if not isinstance(item, dict):
                    raise ValueError("Malformed fill activity")
                try:
                    if item.get("activity_type") != "FILL":
                        raise ValueError("Unexpected account activity type")
                    identity, order_id, symbol = item["id"], item["order_id"], item["symbol"]
                    if not all(isinstance(value, str) and value.strip() for value in (identity, order_id, symbol)):
                        raise ValueError("Fill identity is missing")
                    side = text_enum(item["side"])
                    qty, price = money(item["qty"]), money(item["price"])
                    at = timestamp(item["transaction_time"])
                    if side not in {"buy", "sell"} or qty <= 0 or price <= 0:
                        raise ValueError("Invalid fill amount or side")
                    if not start <= at <= max(until, _utc_now()) + timedelta(seconds=5):
                        raise ValueError("Fill is outside requested activity coverage")
                    fill = Fill(identity, order_id, symbol, side, qty, price, at)
                except (KeyError, TypeError, AttributeError) as exc:
                    raise ValueError("Malformed fill activity") from exc
                previous = found.get(identity)
                if previous is not None and previous != fill:
                    raise ValueError("Conflicting fill activities")
                found[identity] = fill
            if len(page) < 100:
                return sorted(found.values(), key=lambda fill: (fill.at, fill.id))
            cursor = page[-1]["id"]
            if cursor in cursors:
                raise ValueError("Fill pagination did not advance; coverage is incomplete")
            cursors.add(cursor)
            params["page_token"] = cursor
        raise ValueError("Fill activity coverage exceeds the supported pagination bound")

    def snapshot(self, now):
        started = _utc_now()
        first = self.account()
        fills = self.fills(_utc_now())
        parents = self.trade.get_orders(filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN, limit=500, nested=True))
        if not isinstance(parents, list) or len(parents) >= 500:
            raise ValueError("Open-order coverage cannot be confirmed; trading disabled")
        orders = []
        for order in flatten_orders(parents):
            status = text_enum(order.status)
            if status in TERMINAL:
                continue
            if status not in OPEN_STATUSES:
                raise ValueError("Unrecognized open-order status")
            if getattr(order, "asset_class", None) is not None and text_enum(order.asset_class) != "us_equity":
                raise ValueError("Non-equity orders are unsupported in the V2 account")
            orders.append(order)
        positions = self.trade.get_all_positions()
        if not isinstance(positions, list):
            raise ValueError("Invalid position response")
        for position in positions:
            if getattr(position, "asset_class", None) is not None and text_enum(position.asset_class) != "us_equity":
                raise ValueError("Non-equity positions are unsupported in the V2 account")
            if money(position.qty) <= 0:
                raise ValueError("Short or zero positions require manual review")
        second = self.account()
        if not getattr(first, "id", None) or str(first.id) != str(second.id):
            raise ValueError("Broker account changed during reconciliation")
        for account in (first, second):
            if account.currency != "USD" or text_enum(account.status) != "active":
                raise ValueError("An active USD trading account is required")
            if money(getattr(account, "short_market_value", 0)) != 0:
                raise ValueError("Existing short exposure is unsupported")
        account = copy(first)
        for name in ("cash", "buying_power", "non_marginable_buying_power", "equity"):
            setattr(account, name, min(money(getattr(first, name)), money(getattr(second, name))))
        for name in ("pending_transfer_out", "accrued_fees"):
            setattr(account, name, max(money(getattr(first, name, None) or 0),
                                       money(getattr(second, name, None) or 0)))
        for name in ("trading_blocked", "account_blocked", "trade_suspended_by_user"):
            setattr(account, name, bool(getattr(first, name, False) or getattr(second, name, False)))
        # Fill records lack an asset-class field: validate every symbol through the
        # cached asset endpoint so option/crypto amounts cannot be misinterpreted.
        for symbol in {row.symbol for row in fills + positions + orders}:
            self._asset_record(symbol)
        return Snapshot(account, orders, positions, fills, started)

    def lookup(self, client_id):
        try:
            return self.trade.get_order_by_client_id(client_id)
        except APIError as exc:
            if exc.status_code == 404:
                return None
            raise

    def quotes(self, symbols):
        if not symbols:
            return {}
        return self.data.get_stock_latest_quote(StockLatestQuoteRequest(
            symbol_or_symbols=list(symbols), feed=DataFeed.IEX))

    def buy(self, symbol, qty, client_id, limit, stop, take):
        if type(qty) is not int or qty <= 0:
            raise ValueError("V2 bracket entries require a positive whole-share quantity")
        prices = tuple(money(value) for value in (limit, stop, take))
        if not 0 < prices[1] < prices[0] < prices[2]:
            raise ValueError("Invalid entry and bracket prices")
        return self.trade.submit_order(order_data=LimitOrderRequest(
            symbol=symbol, qty=qty, client_order_id=client_id,
            side=OrderSide.BUY, limit_price=float(prices[0]),
            time_in_force=TimeInForce.GTC, extended_hours=False,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=float(prices[1])),
            take_profit=TakeProfitRequest(limit_price=float(prices[2]))))

    def close(self, symbol):
        try:
            return self.trade.close_position(symbol)
        except APIError as exc:
            if exc.status_code == 404:
                return None
            raise
