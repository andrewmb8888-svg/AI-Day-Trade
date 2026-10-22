"""Shared types and strategy rules for AI Day Trade V2."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import errno
import hashlib
import math
import os
from pathlib import Path
import sys
from contextlib import contextmanager

import pandas as pd
from requests.adapters import HTTPAdapter

UTC = timezone.utc
BAR_MINUTES = 30
TERMINAL = {"filled", "canceled", "expired", "rejected", "replaced"}


def text_enum(value):
    return str(getattr(value, "value", value)).lower()


def money(value) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Invalid monetary value") from exc
    if not result.is_finite():
        raise ValueError("Non-finite monetary value")
    return result


def timestamp(value) -> datetime:
    value = pd.Timestamp(value)
    if pd.isna(value) or value.tzinfo is None:
        raise ValueError("A timezone-aware timestamp is required")
    return value.tz_convert("UTC").to_pydatetime()


def session_time(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    if value.tzinfo is None:
        value = value.tz_localize("America/New_York")
    return value.tz_convert("UTC")


def boolean(value):
    value = value.strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise ValueError("Boolean settings must be true or false")


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    group: str
    leverage: int = 1


WATCHLIST = (
    Instrument("NVDA", "Nvidia", "semiconductors"),
    Instrument("AMD", "Advanced Micro Devices", "semiconductors"),
    Instrument("SOXL", "Direxion Semiconductor Bull", "semiconductors", 3),
    Instrument("TSLA", "Tesla", "ev_growth"),
    Instrument("TSLL", "Direxion Tesla Bull", "ev_growth", 2),
    Instrument("COIN", "Coinbase", "crypto"),
    Instrument("IBIT", "iShares Bitcoin Trust", "crypto"),
    Instrument("XBI", "SPDR S&P Biotech", "biotech"),
    Instrument("LABU", "Direxion Biotech Bull", "biotech", 3),
    Instrument("OXY", "Occidental Petroleum", "energy"),
    Instrument("XOP", "SPDR Oil & Gas Exploration & Production", "energy"),
    Instrument("GUSH", "Direxion Oil & Gas Bull", "energy", 2),
    Instrument("NEM", "Newmont", "gold_miners"),
    Instrument("GDX", "VanEck Gold Miners", "gold_miners"),
    Instrument("NUGT", "Direxion Gold Miners Bull", "gold_miners", 2),
    Instrument("IWM", "iShares Russell 2000", "small_caps"),
    Instrument("TNA", "Direxion Small Cap Bull", "small_caps", 3),
    Instrument("JPM", "JPMorgan Chase", "financials"),
    Instrument("KRE", "SPDR S&P Regional Banking", "financials"),
    Instrument("FAS", "Direxion Financial Bull", "financials", 3),
)
BY_SYMBOL = {item.symbol: item for item in WATCHLIST}


@dataclass(frozen=True)
class Settings:
    symbols: tuple[str, ...] = tuple(BY_SYMBOL)
    paper: bool = True
    interval: float = 10
    allocation: Decimal = Decimal("0.02")
    group_cap: Decimal = Decimal("0.10")
    cash_buffer: Decimal = Decimal("0.05")
    stop_loss: Decimal = Decimal("0.03")
    take_profit: Decimal = Decimal("0.06")
    window: int = 5

    def __post_init__(self):
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Select unique V2 symbols")
        if set(self.symbols) - set(BY_SYMBOL):
            raise ValueError("V2_SYMBOLS must use symbols in the reviewed watchlist")
        if type(self.paper) is not bool or not math.isfinite(self.interval) or self.interval < 1:
            raise ValueError("Invalid paper mode or check interval")
        for name in ("allocation", "group_cap", "cash_buffer", "stop_loss", "take_profit"):
            value = money(getattr(self, name))
            object.__setattr__(self, name, value)
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.group_cap < self.allocation:
            raise ValueError("Group cap must cover at least one position allocation")
        if type(self.window) is not int or self.window < 1:
            raise ValueError("Signal window must be a positive integer")

    @classmethod
    def from_environment(cls):
        return cls(
            symbols=tuple(x.strip().upper() for x in os.getenv("V2_SYMBOLS", ",".join(BY_SYMBOL)).split(",")),
            paper=boolean(os.getenv("PAPER", "true")),
            interval=float(os.getenv("V2_CHECK_SECONDS", "10")),
            allocation=money(os.getenv("V2_ALLOCATION_PCT", "0.02")),
            group_cap=money(os.getenv("V2_GROUP_EXPOSURE_PCT", "0.10")),
            cash_buffer=money(os.getenv("V2_CASH_BUFFER_PCT", "0.05")),
            stop_loss=money(os.getenv("V2_STOP_LOSS_PCT", "0.03")),
            take_profit=money(os.getenv("V2_TAKE_PROFIT_PCT", "0.06")),
        )


@dataclass(frozen=True)
class Fill:
    id: str
    order_id: str
    symbol: str
    side: str
    qty: Decimal
    price: Decimal
    at: datetime

    @property
    def gross(self):
        return money(self.qty) * money(self.price)


@dataclass
class Snapshot:
    account: object
    orders: list
    positions: list
    fills: list[Fill]
    started_at: datetime


@dataclass(frozen=True)
class Decision:
    entry_key: str | None = None
    exit_signal: bool = False


def rsi(prices: pd.Series, period=14):
    delta = prices.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    result = pd.Series(float("nan"), index=prices.index)
    if len(prices) <= period:
        return result
    avg_gain = float(gain.iloc[1:period + 1].mean())
    avg_loss = float(loss.iloc[1:period + 1].mean())
    for i in range(period, len(prices)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period
        result.iloc[i] = (100 - 100 / (1 + avg_gain / avg_loss) if avg_loss else 100.0 if avg_gain else 50.0)
    return result


def signals(bars: pd.DataFrame, daily: pd.DataFrame, window=5) -> Decision:
    if len(bars) < 100:
        raise ValueError("Need 100 completed 30-minute bars")
    prices = bars.close
    strength = rsi(prices)
    macd = prices.ewm(span=12, adjust=False).mean() - prices.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    bounce = (strength.shift() <= 30) & (strength > 30)
    cross = (macd.shift() <= signal.shift()) & (macd > signal)
    retreat = (strength.shift() >= 65) & (strength < 65)
    bearish = ((macd.shift() >= signal.shift()) & (macd < signal)) | ((macd.shift() >= 0) & (macd < 0))

    def latest(events):
        recent = events.iloc[-(window + 1):]
        return recent.index[recent][-1] if recent.any() else None

    exit_signal = latest(retreat) is not None and latest(bearish) is not None
    if len(daily) < 200 or not daily.attrs.get("current_session_data", False):
        return Decision(exit_signal=exit_signal)
    averages = [daily.close.rolling(n).mean().iloc[-1] for n in (50, 100, 200)]
    uptrend = all(math.isfinite(x) for x in averages) and averages[0] > averages[1] > averages[2]
    bounce_at, cross_at = latest(bounce), latest(cross)
    key = f"{bounce_at}:{cross_at}" if uptrend and bounce_at is not None and cross_at is not None else None
    return Decision(key, exit_signal)


def order_key(symbol, event_key):
    return "adv2-" + hashlib.sha256(f"v2:30m:{symbol}:{event_key}".encode()).hexdigest()[:40]


def flatten_orders(orders):
    result = {}
    for order in orders:
        if not getattr(order, "id", None):
            raise ValueError("Order is missing its ID")
        result[str(order.id)] = order
        for leg in flatten_orders(getattr(order, "legs", None) or []):
            result[str(leg.id)] = leg
    return list(result.values())


class TimeoutAdapter(HTTPAdapter):
    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (10, 30)
        return super().send(request, **kwargs)


@contextmanager
def instance_lock(path: Path):
    with path.open("a+b") as lock:
        try:
            if sys.platform == "win32":
                import msvcrt
                if path.stat().st_size == 0:
                    lock.write(b"\0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise SystemExit("Another local bot instance is already running") from exc
            raise
        yield
