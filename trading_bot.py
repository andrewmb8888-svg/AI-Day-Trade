"""Hourly, long-only Alpaca RSI/MACD strategy. Defaults to paper trading."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import math
import os
from pathlib import Path
import time
from logging.handlers import RotatingFileHandler

from requests.adapters import HTTPAdapter

import pandas as pd

LOG = logging.getLogger(__name__)
UTC = timezone.utc


def boolean(value: str) -> bool:
    if value.strip().lower() in {"true", "1", "yes"}:
        return True
    if value.strip().lower() in {"false", "0", "no"}:
        return False
    raise ValueError("PAPER must be true or false")


@dataclass(frozen=True)
class Config:
    symbol: str = "TQQQ"
    allocation: float = 0.02
    window: int = 5
    paper: bool = True
    stop_loss: float = 0.03
    take_profit: float = 0.06

    def __post_init__(self):
        if not self.symbol or not self.symbol.isascii() or not self.symbol.replace(".", "").isalnum():
            raise ValueError("A valid stock symbol is required")
        if not math.isfinite(self.allocation) or not 0 < self.allocation <= 1:
            raise ValueError("Allocation must be in (0, 1]")
        if type(self.window) is not int or self.window < 1:
            raise ValueError("Signal window must be a positive integer")
        if (not math.isfinite(self.stop_loss) or not 0 < self.stop_loss < 1 or
                not math.isfinite(self.take_profit) or self.take_profit <= 0):
            raise ValueError("Invalid exit percentages")
        if type(self.paper) is not bool:
            raise ValueError("Paper mode must be a Boolean")


@dataclass(frozen=True)
class Decision:
    entry_key: str | None = None
    exit_signal: bool = False


class TimeoutAdapter(HTTPAdapter):
    """Keep requests' proxy/TLS handling, with bounded connect/read waits."""
    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (10, 30)
        return super().send(request, **kwargs)


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI, seeded with the arithmetic average of the first period."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    result = pd.Series(float("nan"), index=prices.index)
    if len(prices) <= period:
        return result
    avg_gain = float(gain.iloc[1:period + 1].mean())
    avg_loss = float(loss.iloc[1:period + 1].mean())
    for i in range(period, len(prices)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period
        result.iloc[i] = (100 - 100 / (1 + avg_gain / avg_loss) if avg_loss else
                          100.0 if avg_gain else 50.0)
    return result


def signals(hourly: pd.DataFrame, daily: pd.DataFrame, window: int = 5) -> Decision:
    """Identify recent events, using their timestamps as a stable entry key."""
    if len(hourly) < 100:
        raise ValueError("Need at least 100 completed hourly bars")
    prices = hourly.close
    strength = rsi(prices)
    macd = prices.ewm(span=12, adjust=False).mean() - prices.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    bounce = (strength.shift() <= 30) & (strength > 30)
    cross = (macd.shift() <= signal.shift()) & (macd > signal)
    retreat = (strength.shift() >= 65) & (strength < 65)
    bearish = ((macd.shift() >= signal.shift()) & (macd < signal)) | ((macd.shift() >= 0) & (macd < 0))

    def latest_event(events):
        recent = events.iloc[-(window + 1):]
        return recent.index[recent][-1] if recent.any() else None

    # Exits need no daily trend data. Daily-history problems only prevent entry.
    exit_signal = latest_event(retreat) is not None and latest_event(bearish) is not None
    if len(daily) < 200 or not daily.attrs.get("current_session_data", True):
        return Decision(exit_signal=exit_signal)
    averages = [daily.close.rolling(n).mean().iloc[-1] for n in (50, 100, 200)]
    uptrend = all(math.isfinite(x) for x in averages) and averages[0] > averages[1] > averages[2]
    bounce_at, cross_at = latest_event(bounce), latest_event(cross)
    key = f"{bounce_at}:{cross_at}" if uptrend and bounce_at is not None and cross_at is not None else None
    return Decision(key, exit_signal)


def completed_bars(frame: pd.DataFrame, symbol: str, now: datetime, hourly: bool) -> pd.DataFrame:
    frame = frame.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.xs(symbol, level="symbol")
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if hourly:
        frame = frame[frame.index <= pd.Timestamp(now) - pd.Timedelta(hours=1)]
    else:
        # Exclude the current New York session, including its partial daily bar.
        today = pd.Timestamp(now).tz_convert("America/New_York").date()
        frame = frame[frame.index.tz_convert("America/New_York").date < today]
    if frame.empty or not frame.close.map(lambda x: math.isfinite(x) and x > 0).all():
        raise ValueError("Missing or invalid market data")
    return frame


def session_time(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value.tz_localize("America/New_York").tz_convert("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def regular_hours(frame: pd.DataFrame, calendar) -> pd.DataFrame:
    """Keep full hourly intervals contained in a regular exchange session."""
    keep = pd.Series(False, index=frame.index)
    for session in calendar:
        opening, closing = session_time(session.open), session_time(session.close)
        keep |= (frame.index >= opening) & (frame.index <= closing - pd.Timedelta(hours=1))
    result = frame.loc[keep]
    if result.empty:
        raise ValueError("No completed regular-session hourly bars")
    return result


class AlpacaBroker:
    def __init__(self, config: Config):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient
        key = os.getenv("ALPACA_PAPER_API_KEY" if config.paper else "ALPACA_API_KEY")
        secret = os.getenv("ALPACA_PAPER_SECRET_KEY" if config.paper else "ALPACA_SECRET_KEY")
        if not key or not secret:
            raise ValueError("Set Alpaca credentials in the environment; see .env.example")
        if not config.paper and os.getenv("ALLOW_LIVE_TRADING") != "I_UNDERSTAND":
            raise ValueError("Live trading requires ALLOW_LIVE_TRADING=I_UNDERSTAND")
        if os.getenv("TRADE_API_URL"):
            raise ValueError("TRADE_API_URL overrides are unsupported; remove it to use the selected Alpaca endpoint")
        self.trade = TradingClient(key, secret, paper=config.paper)
        self.data = StockHistoricalDataClient(key, secret)
        # alpaca-py 0.43.2 exposes no public request-timeout option.
        # Mounting an adapter preserves SDK authentication, proxies and TLS.
        for client in (self.trade, self.data):
            client._session.mount("https://", TimeoutAdapter())
        self.config = config
        self._calendar_day = None
        self._calendar = None

    def calendar(self, now):
        from alpaca.trading.requests import GetCalendarRequest
        today = pd.Timestamp(now).tz_convert("America/New_York").date()
        if self._calendar_day != today:
            calendar = self.trade.get_calendar(GetCalendarRequest(
                start=today - timedelta(days=110), end=today))
            if not calendar:
                raise ValueError("Exchange calendar is unavailable")
            self._calendar, self._calendar_day = calendar, today
        return self._calendar

    def clock(self):
        return self.trade.get_clock()

    def bars(self, now):
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed, Adjustment
        def fetch(timeframe, days, hourly):
            request = StockBarsRequest(symbol_or_symbols=[self.config.symbol], timeframe=timeframe,
                                       start=now - timedelta(days=days), end=now,
                                       feed=DataFeed.IEX, adjustment=Adjustment.SPLIT)
            return completed_bars(self.data.get_stock_bars(request).df, self.config.symbol, now, hourly)
        calendar = self.calendar(now)
        hourly = regular_hours(fetch(TimeFrame.Hour, 100, True), calendar)
        daily = fetch(TimeFrame.Day, 500, False)
        today = pd.Timestamp(now).tz_convert("America/New_York").date()
        previous = max(session.date for session in calendar if session.date < today)
        daily.attrs["current_session_data"] = daily.index[-1].tz_convert("America/New_York").date() == previous
        if not daily.attrs["current_session_data"]:
            LOG.warning("Daily data does not include previous exchange session %s; entries disabled", previous)
        return hourly, daily

    def position(self):
        # A failed API request must never be interpreted as an empty account.
        return next((p for p in self.trade.get_all_positions() if p.symbol == self.config.symbol), None)

    def orders(self):
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        return self.trade.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN,
                                                           symbols=[self.config.symbol], nested=True))

    def already_submitted(self, client_id):
        from alpaca.common.exceptions import APIError
        try:
            self.trade.get_order_by_client_id(client_id)
            return True
        except APIError as exc:
            if exc.status_code == 404:
                return False
            raise

    def account(self):
        return self.trade.get_account()

    def quote(self):
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca.data.enums import DataFeed
        return self.data.get_stock_latest_quote(StockLatestQuoteRequest(
            symbol_or_symbols=[self.config.symbol], feed=DataFeed.IEX))[self.config.symbol]

    def buy(self, qty, client_id, limit, stop, take):
        from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        return self.trade.submit_order(order_data=LimitOrderRequest(
            symbol=self.config.symbol, qty=qty, side=OrderSide.BUY,
            limit_price=limit, time_in_force=TimeInForce.GTC, extended_hours=False,
            client_order_id=client_id, order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop),
            take_profit=TakeProfitRequest(limit_price=take)))

    def close(self):
        # Ask the broker to close its current position, rather than selling a
        # stale locally cached quantity that could accidentally open a short.
        from alpaca.common.exceptions import APIError
        try:
            return self.trade.close_position(self.config.symbol)
        except APIError as exc:
            if exc.status_code == 404:
                return None  # Position closed between reconciliation and call.
            raise

    def check(self, now):
        """Read-only connectivity and data check; never submits/cancels orders."""
        from alpaca.trading.enums import AssetClass
        clock = self.clock()
        account = self.account()
        asset = self.trade.get_asset(self.config.symbol)
        if not asset.tradable or asset.asset_class != AssetClass.US_EQUITY:
            raise ValueError("Symbol must be a tradable US equity")
        if account.account_blocked or account.trading_blocked:
            raise ValueError("Account is blocked from trading")
        hourly, daily = self.bars(now)
        signals(hourly, daily, self.config.window)
        self.position()
        self.orders()
        self.quote()
        LOG.info("Read-only connection check passed; market_open=%s; hourly_bars=%d; daily_bars=%d; daily_current=%s",
                 clock.is_open, len(hourly), len(daily), daily.attrs["current_session_data"])


@dataclass
class Engine:
    broker: object
    config: Config
    processed: object = field(default=None, init=False)

    def step(self, now: datetime) -> str:
        clock = self.broker.clock()
        if not clock.is_open:
            return "market closed"
        hourly, daily = self.broker.bars(now)
        bar = hourly.index[-1]
        if now - bar.to_pydatetime() > timedelta(hours=2):
            raise ValueError("Hourly data is stale; skipping orders")
        if bar == self.processed:
            return "bar already processed"
        decision = signals(hourly, daily, self.config.window)
        # Orders first, then position: a buy filling during reconciliation is
        # either seen as an order or as a position, rather than missed by both.
        orders = self.broker.orders()
        position = self.broker.position()
        if position is not None and (not math.isfinite(float(position.qty)) or float(position.qty) <= 0):
            raise ValueError("Unexpected short or zero position; manual review required")
        if position is not None and decision.exit_signal:
            # Do not duplicate an existing sell or remove protective brackets. A
            # bracket's broker-held exits manage this position until it closes.
            if orders:
                return "position has active orders; broker exits retained"
            side = "sell"
        elif position is None and decision.entry_key:
            if orders:
                return "pending orders; entry skipped"
            side = "buy"
        else:
            self.processed = bar
            return "hold"
        if side == "sell":
            result = self.broker.close()
            self.processed = bar
            return "close submitted" if result is not None else "position already closed"
        else:
            client_id = "rsi-" + hashlib.sha256(f"{self.config.symbol}:{decision.entry_key}".encode()).hexdigest()[:40]
            if self.broker.already_submitted(client_id):
                self.processed = bar
                return "order already submitted"
            account = self.broker.account()
            if account.trading_blocked or account.account_blocked:
                raise ValueError("Account is blocked from trading")
            quote = self.broker.quote()
            age = (now - quote.timestamp).total_seconds()
            ask, bid = float(quote.ask_price), float(quote.bid_price)
            if not all(math.isfinite(x) and x > 0 for x in (ask, bid)) or not bid <= ask:
                raise ValueError("Invalid quote")
            if not 0 <= age <= 120 or (ask - bid) / ask > 0.01:
                raise ValueError("Quote is stale or spread exceeds 1%")
            limit = round(ask * 1.001, 2)
            equity, buying_power = float(account.equity), float(account.buying_power)
            if not all(math.isfinite(x) and x > 0 for x in (equity, buying_power)):
                raise ValueError("Invalid account balances")
            budget = min(equity * self.config.allocation, buying_power * 0.95)
            qty = math.floor(budget / limit)
            if qty < 1:
                self.processed = bar
                return "insufficient allocation for one share"
            stop = round(limit * (1 - self.config.stop_loss), 2)
            take = round(limit * (1 + self.config.take_profit), 2)
            if not 0 < stop < limit < take or limit < 1:
                raise ValueError("Entry and bracket prices must be distinct; stocks below $1 are unsupported")
            # Avoid a stop immediately marketable after a wide spread/price move.
            if stop >= bid:
                raise ValueError("Protective stop would be above the current bid")
            self.broker.buy(qty, client_id, limit, stop, take)
        self.processed = bar
        return f"{side} submitted"


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one cycle (may submit paper orders)")
    parser.add_argument("--check", action="store_true", help="Read-only broker/data check; never places orders")
    args = parser.parse_args()
    if args.once and args.check:
        parser.error("Choose --once or --check")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), RotatingFileHandler(Path(__file__).with_name("trade_log.txt"), maxBytes=5_000_000, backupCount=3)])
    config = Config(symbol=os.getenv("SYMBOL", "TQQQ"), paper=boolean(os.getenv("PAPER", "true")))
    # Prevent concurrent local instances from racing on the same account/symbol.
    import fcntl
    lock_path = Path(__file__).with_name(f".bot-{'paper' if config.paper else 'live'}.lock")
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another local bot instance is already running")
        broker = AlpacaBroker(config)
        if args.check:
            broker.check(datetime.now(UTC))
            return
        engine = Engine(broker, config)
        while True:
            try:
                LOG.info(engine.step(datetime.now(UTC)))
            except Exception:
                LOG.exception("Cycle failed; no blind order retry. Next cycle reconciles broker state.")
                if args.once:
                    raise
            if args.once:
                return
            # Check each minute, process each completed hourly bar at most once.
            # Closed-market checks resume automatically on the next session.
            time.sleep(60)


if __name__ == "__main__":
    main()
