"""Batched, cached IEX data for completed regular-session 30-minute bars."""
from __future__ import annotations

from datetime import timedelta
import math

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from common import BAR_MINUTES, session_time

BAR = pd.Timedelta(minutes=BAR_MINUTES)
PUBLICATION_DELAY = pd.Timedelta(seconds=3)


def _now(value):
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("Market data requires a timezone-aware current time")
    return value.tz_convert("UTC")


def _day(session):
    return pd.Timestamp(session.date).date()


def expected_bar(now, calendar) -> pd.Timestamp | None:
    """Latest completed bar in today's session, allowing publication time."""
    now = _now(now)
    today = now.tz_convert("America/New_York").date()
    for session in calendar:
        if _day(session) != today:
            continue
        opening, closing = session_time(session.open), session_time(session.close)
        end = min(now - PUBLICATION_DELAY, closing)
        completed = int((end - opening) // BAR)
        if completed < 1:
            return None
        return opening + (completed - 1) * BAR
    return None


def _empty():
    return pd.DataFrame({"close": pd.Series(dtype=float)},
                        index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))


def _clean(frame):
    frame = frame.copy()
    if frame.empty:
        return _empty()
    if "close" not in frame.columns:
        raise ValueError("Market data has no close prices")
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.isna().any():
        raise ValueError("Market data has invalid timestamps")
    frame["close"] = pd.to_numeric(frame.close, errors="raise")
    if not frame.close.map(lambda x: math.isfinite(x) and x > 0).all():
        raise ValueError("Market data has invalid close prices")
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def regular30(frame, now, calendar):
    """Return aligned, completed intervals entirely inside regular sessions."""
    now = _now(now)
    if frame.empty:
        return frame.copy()
    keep = pd.Series(False, index=frame.index)
    effective = now - PUBLICATION_DELAY
    for session in calendar:
        opening, closing = session_time(session.open), session_time(session.close)
        aligned = (frame.index - opening) % BAR == pd.Timedelta(0)
        keep |= ((frame.index >= opening) & aligned &
                 (frame.index <= min(effective, closing) - BAR))
    return frame.loc[keep].copy()


class MarketData:
    def __init__(self, data_client, settings):
        self.client = data_client
        self.settings = settings
        self.errors: dict[str, str] = {}
        self._intraday: dict[str, pd.DataFrame] = {}
        self._daily: dict[str, pd.DataFrame] = {}
        self._completed: dict[str, pd.DataFrame] = {}
        self._confirmed: dict[str, pd.Timestamp] = {}
        self._daily_day = {}
        self._history_day = {}
        self._intraday_day = {}

    def _fetch(self, symbols, now, timeframe, days):
        request = StockBarsRequest(
            symbol_or_symbols=list(symbols), timeframe=timeframe,
            start=(now - pd.Timedelta(days=days)).to_pydatetime(),
            end=now.to_pydatetime(), feed=DataFeed.IEX,
            adjustment=Adjustment.SPLIT,
        )
        # Do not set limit: Alpaca's limit applies to the combined result, not
        # each ticker, and could omit later symbols. The SDK paginates all rows.
        return self.client.get_stock_bars(request).df

    @staticmethod
    def _extract(frame, symbol, symbols):
        if frame.empty:
            return _empty()
        if isinstance(frame.index, pd.MultiIndex):
            if "symbol" not in frame.index.names:
                raise ValueError("Batched market data is missing its symbol index")
            try:
                return frame.xs(symbol, level="symbol")
            except KeyError:
                return _empty()
        if len(symbols) != 1:
            raise ValueError("Batched market data is missing symbol labels")
        return frame

    @staticmethod
    def _merge(previous, new, maximum=3000):
        if previous is None or previous.empty:
            merged = new
        elif new.empty:
            merged = previous
        else:
            merged = pd.concat([previous, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        return merged.iloc[-maximum:].copy()

    def update(self, now, calendar, *, force_history=False):
        """Return per-symbol (intraday,daily) frames; retry late symbols only."""
        now = _now(now)
        calendar = list(calendar)
        self.errors = {}
        target = expected_bar(now, calendar)
        if target is None and not force_history:
            return {}  # No current-session completed bar; no historical polling.
        today = now.tz_convert("America/New_York").date()
        earlier = [_day(session) for session in calendar if _day(session) < today]
        if not earlier:
            raise ValueError("Calendar does not include the previous trading session")
        previous_session = max(earlier)
        invalid = set()
        symbols = tuple(self.settings.symbols)

        needed = ([s for s in symbols if self._history_day.get(s) != today]
                  if target is None else
                  [s for s in symbols if self._confirmed.get(s) != target])
        # Corporate actions can rescale the entire split-adjusted prefix.
        # Replace history once per NY date, then overlap one day intraday.
        full = [s for s in needed if self._intraday_day.get(s) != today]
        incremental = [s for s in needed if self._intraday_day.get(s) == today]
        for days, group in ((60, full), (1, incremental)):
            if not group:
                continue
            raw = self._fetch(group, now, TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute), days)
            for symbol in group:
                replace = days == 60
                try:
                    fresh = _clean(self._extract(raw, symbol, group))
                    if replace:
                        self._intraday[symbol] = self._merge(None, fresh)
                        self._completed.pop(symbol, None)
                        self._confirmed.pop(symbol, None)
                        self._intraday_day.pop(symbol, None)
                    elif not fresh.empty:
                        self._intraday[symbol] = self._merge(self._intraday.get(symbol), fresh)
                    complete = regular30(fresh, now, calendar)
                    if target is None and not complete.empty:
                        # Read-only startup history is useful before opening,
                        # but first trading bar still confirms today's basis.
                        self._history_day[symbol] = today
                        self._completed[symbol] = regular30(self._intraday[symbol], now, calendar)
                    elif target is not None and target in complete.index:
                        self._confirmed[symbol] = target
                        self._intraday_day[symbol] = today
                        self._completed[symbol] = regular30(self._intraday[symbol], now, calendar)
                    else:
                        self.errors[symbol] = "Latest completed 30-minute bar is missing"
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    if replace:
                        self._intraday.pop(symbol, None)
                        self._completed.pop(symbol, None)
                        self._confirmed.pop(symbol, None)
                        self._intraday_day.pop(symbol, None)
                    self.errors[symbol] = str(exc)
                    invalid.add(symbol)

        daily_needed = [s for s in symbols if self._daily_day.get(s) != today]
        if daily_needed:
            raw = self._fetch(daily_needed, now, TimeFrame.Day, 500)
            for symbol in daily_needed:
                try:
                    fresh = _clean(self._extract(raw, symbol, daily_needed))
                    # Replace the whole prefix on each date: an overnight split
                    # must not mix newly adjusted prices with yesterday's basis.
                    fresh = fresh[fresh.index.tz_convert("America/New_York").date < today]
                    self._daily[symbol] = self._merge(None, fresh)
                    if not fresh.empty and fresh.index[-1].tz_convert("America/New_York").date() == previous_session:
                        self._daily_day[symbol] = today
                    else:
                        self._daily_day.pop(symbol, None)
                        self.errors.setdefault(symbol, "Daily trend data is stale or missing; entries disabled")
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    self._daily.pop(symbol, None)
                    self._daily_day.pop(symbol, None)
                    self.errors[symbol] = str(exc)
                    invalid.add(symbol)

        result = {}
        for symbol in symbols:
            if (symbol in invalid or
                    (target is not None and self._confirmed.get(symbol) != target) or
                    (target is None and self._history_day.get(symbol) != today)):
                self.errors.setdefault(symbol, "Latest completed 30-minute bar is missing")
                continue
            intraday = self._completed.get(symbol, _empty()).copy()
            if intraday.empty or (target is not None and intraday.index[-1] != target):
                self.errors[symbol] = "Latest completed 30-minute bar is missing"
                continue
            daily = self._daily.get(symbol, _empty()).copy()
            daily = daily[daily.index.tz_convert("America/New_York").date < today]
            daily.attrs["current_session_data"] = (
                self._daily_day.get(symbol) == today and not daily.empty and
                daily.index[-1].tz_convert("America/New_York").date() == previous_session)
            if not daily.attrs["current_session_data"]:
                self.errors.setdefault(symbol, "Daily trend data is stale or missing; entries disabled")
            result[symbol] = intraday, daily
        return result
