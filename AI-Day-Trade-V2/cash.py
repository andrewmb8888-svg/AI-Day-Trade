"""Conservative cash reservations and durable submission intents for V2.

Alpaca's account model has no settled-cash field. This module deliberately
holds recent sale proceeds longer than normal settlement and may double-reserve
funds already reflected in broker buying power. It never assumes a missing
order proves that an ambiguous submission was not accepted.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta
from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from common import BY_SYMBOL, TERMINAL, flatten_orders, money, text_enum, timestamp

ZERO = Decimal("0")
NY = "America/New_York"


def _positive(value, name):
    result = money(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value, name):
    result = money(value)
    if result < 0:
        raise ValueError(f"{name} cannot be negative")
    return result


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key in cash intent state")
        result[key] = value
    return result


class IntentStore:
    """Atomic, account-bound journal written before an order can be sent."""
    def __init__(self, path, account_id, paper):
        self.path = Path(path)
        self.account_id = str(account_id)
        if not account_id or not self.account_id or type(paper) is not bool:
            raise ValueError("Account ID and Boolean paper mode are required")
        self.paper = paper
        self._intents = {}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"), object_pairs_hook=_object_pairs)
                if (not isinstance(data, dict) or data.get("version") != 1 or
                        data.get("account_id") != self.account_id or
                        type(data.get("paper")) is not bool or data["paper"] != paper or
                        not isinstance(data.get("intents"), dict)):
                    raise ValueError("Wrong account, mode, or schema in cash intent state")
                for key, record in data["intents"].items():
                    self._validate(key, record)
                self._intents = data["intents"]
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError("Corrupt cash intent state; manual review required") from exc

    @staticmethod
    def _validate(key, record):
        if not isinstance(key, str) or not key or not isinstance(record, dict):
            raise ValueError("Invalid cash intent record")
        required = {"client_id", "symbol", "qty", "limit", "reserved", "group", "leverage",
                    "created_at", "status", "filled_qty", "order_id"}
        if set(record) != required or record["client_id"] != key:
            raise ValueError("Invalid cash intent record fields")
        for name in ("symbol", "group", "status"):
            if not isinstance(record[name], str) or not record[name]:
                raise ValueError("Invalid cash intent metadata")
        if type(record["leverage"]) is not int or record["leverage"] < 1:
            raise ValueError("Invalid cash intent leverage")
        for name in ("qty", "limit", "reserved", "filled_qty"):
            if not isinstance(record[name], str):
                raise ValueError("Cash intent amounts must be stored as strings")
        qty = _positive(record["qty"], "Intent quantity")
        limit = _positive(record["limit"], "Intent limit")
        filled = _nonnegative(record["filled_qty"], "Filled quantity")
        reserved = _nonnegative(record["reserved"], "Intent reservation")
        if filled > qty or record["order_id"] is not None and not isinstance(record["order_id"], str):
            raise ValueError("Invalid reconciled intent")
        timestamp(record["created_at"])
        expected = ZERO if record["status"] in TERMINAL else (qty - filled) * limit
        if reserved != expected:
            raise ValueError("Cash intent reservation does not match its quantity and price")
        if record["status"] == "filled" and filled != qty:
            raise ValueError("Filled intent must report the entire intended quantity")
        if record["status"] == "rejected" and filled != ZERO:
            raise ValueError("Rejected intent cannot have fills")
        if record["status"] in TERMINAL - {"rejected"} and not record["order_id"]:
            raise ValueError("Resolved intent requires its broker order ID")

    def _save(self):
        data = {"version": 1, "account_id": self.account_id, "paper": self.paper,
                "intents": self._intents}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=f".{self.path.name}.", suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
                json.dump(data, file, sort_keys=True, separators=(",", ":"))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
            temporary = None
            if os.name != "nt":
                descriptor = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def consumed(self, client_id):
        return str(client_id) in self._intents

    def active(self):
        return [deepcopy(record) for record in self._intents.values() if record["status"] not in TERMINAL]

    def reserve(self, client_id, symbol, qty, limit, group, leverage, now):
        if not isinstance(client_id, str) or not client_id or client_id in self._intents:
            raise ValueError("Signal already consumed or invalid client ID")
        qty, limit = _positive(qty, "Quantity"), _positive(limit, "Limit price")
        record = {"client_id": client_id, "symbol": symbol, "qty": str(qty), "limit": str(limit),
                  "reserved": str(qty * limit), "group": group, "leverage": leverage,
                  "created_at": timestamp(now).isoformat(), "status": "reserved",
                  "filled_qty": "0", "order_id": None}
        self._validate(client_id, record)
        self._intents[client_id] = record
        self._save()

    def observe(self, order):
        client_id = str(getattr(order, "client_order_id", ""))
        if client_id not in self._intents:
            return
        record = deepcopy(self._intents[client_id])
        status = text_enum(getattr(order, "status", None))
        if status in {"", "none"}:
            raise ValueError("Order is missing reconciliation status")
        if getattr(order, "symbol", None) != record["symbol"]:
            raise ValueError("Order symbol does not match its intent")
        if money(getattr(order, "qty", None)) != money(record["qty"]):
            raise ValueError("Order quantity does not match its intent")
        if text_enum(getattr(order, "side", None)) != "buy":
            raise ValueError("Order side does not match its buy intent")
        filled = _nonnegative(getattr(order, "filled_qty", None), "Filled quantity")
        if filled > money(record["qty"]) or filled < money(record["filled_qty"]):
            raise ValueError("Filled quantity is invalid or moved backwards")
        raw_order_id = getattr(order, "id", None)
        if not raw_order_id:
            raise ValueError("Order is missing reconciliation ID")
        order_id = str(raw_order_id)
        if record["status"] in TERMINAL and status not in TERMINAL:
            raise ValueError("A terminal intent cannot become pending again")
        record.update(status=status, filled_qty=str(filled), order_id=order_id)
        observed_limit = getattr(order, "limit_price", None)
        if observed_limit is not None:
            record["limit"] = str(max(money(record["limit"]), _positive(observed_limit, "Observed limit")))
        record["reserved"] = str(ZERO if status in TERMINAL else
                                 (money(record["qty"]) - filled) * money(record["limit"]))
        self._validate(client_id, record)
        if record != self._intents[client_id]:
            self._intents[client_id] = record
            self._save()

    def reject(self, client_id):
        if client_id not in self._intents:
            raise ValueError("Cannot reject an unknown cash intent")
        record = self._intents[client_id]
        if money(record["filled_qty"]) > 0:
            raise ValueError("A partially filled intent cannot be treated as rejected")
        record["status"], record["reserved"] = "rejected", "0"
        self._save()


def reservations(snapshot, store):
    """All-account pending buy cash; local/broker duplicates reserve the larger cost."""
    result = {}
    for record in store.active():
        result[record["client_id"]] = {"symbol": record["symbol"], "cost": money(record["reserved"]),
                                       "group": record["group"], "leverage": record["leverage"]}
    for order in flatten_orders(snapshot.orders):
        status, side = text_enum(getattr(order, "status", None)), text_enum(getattr(order, "side", None))
        if status in {"", "none"} or side not in {"buy", "sell"}:
            raise ValueError("Invalid open order status or side")
        if status in TERMINAL or side != "buy":
            continue
        kind = text_enum(getattr(order, "type", None) or getattr(order, "order_type", None))
        if kind != "limit":
            raise ValueError("Unsupported open buy order; cannot safely reserve cash")
        qty = _positive(getattr(order, "qty", None), "Open buy quantity")
        filled = _nonnegative(getattr(order, "filled_qty", None), "Open buy filled quantity")
        price = _positive(getattr(order, "limit_price", None), "Open buy limit")
        if filled > qty:
            raise ValueError("Open buy filled quantity exceeds total")
        symbol = getattr(order, "symbol", None)
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Open buy is missing its symbol")
        key = str(getattr(order, "client_order_id", None) or f"order:{order.id}")
        cost = (qty - filled) * price
        instrument = BY_SYMBOL.get(symbol)
        candidate = {"symbol": symbol, "cost": cost,
                     "group": instrument.group if instrument else "other",
                     "leverage": instrument.leverage if instrument else 1}
        if key in result:
            if result[key]["symbol"] != symbol:
                raise ValueError("Duplicate client ID has conflicting symbols")
            result[key]["cost"] = max(result[key]["cost"], cost)
        else:
            result[key] = candidate
    return result


def _calendar_dates(calendar, now):
    today = pd.Timestamp(now).tz_convert(NY).date()
    dates = []
    for session in calendar:
        value = getattr(session, "date", None)
        if isinstance(value, datetime):
            value = value.date()
        elif isinstance(value, str):
            value = date.fromisoformat(value)
        if not isinstance(value, date) or value.weekday() > 4:
            raise ValueError("Invalid settlement exchange calendar")
        dates.append(value)
    if (not dates or min(dates) > today - timedelta(days=14) or
            max(dates) < today + timedelta(days=10)):
        raise ValueError("Exchange calendar does not cover settlement lookback and future sessions")
    dates = sorted(set(dates))
    holidays = set(USFederalHolidayCalendar().holidays(start=min(dates), end=max(dates)).date)
    return dates, [day for day in dates if day not in holidays]


def budget(snapshot, store, calendar, now, buffer_pct):
    """Cash-only spending bound, deliberately conservative about pending/settled funds."""
    now, started = timestamp(now), timestamp(snapshot.started_at)
    if started > now + timedelta(seconds=5):
        raise ValueError("Snapshot timestamp is in the future")
    account = snapshot.account
    account_id = getattr(account, "id", None)
    if not account_id or str(account_id) != store.account_id:
        raise ValueError("Cash snapshot belongs to a different account")
    buffer_pct = money(buffer_pct)
    if not ZERO <= buffer_pct < 1:
        raise ValueError("Cash buffer must be between zero and one")
    available = max(ZERO, min(money(getattr(account, "cash", None)),
                              money(getattr(account, "non_marginable_buying_power", None)),
                              money(getattr(account, "buying_power", None))))
    available -= sum((record["cost"] for record in reservations(snapshot, store).values()), ZERO)
    for field in ("pending_transfer_out", "accrued_fees"):
        available -= _nonnegative(getattr(account, field, None) or "0", field)
    dates, settlement_days = _calendar_dates(calendar, now)
    today = pd.Timestamp(now).tz_convert(NY).date()
    seen = {}
    for fill in snapshot.fills:
        fill_id = getattr(fill, "id", None)
        symbol, order_id = getattr(fill, "symbol", None), getattr(fill, "order_id", None)
        if not isinstance(fill_id, str) or not fill_id or not symbol or not order_id:
            raise ValueError("Fill is missing identifying information")
        side = text_enum(fill.side)
        qty, price, at = _positive(fill.qty, "Fill quantity"), _positive(fill.price, "Fill price"), timestamp(fill.at)
        if (side not in {"buy", "sell"} or at > now + timedelta(seconds=5) or
                at < started - timedelta(days=14, seconds=30)):
            raise ValueError("Invalid fill side or out-of-window fill timestamp")
        identity = (str(order_id), symbol, side, qty, price, at)
        if fill_id in seen:
            if seen[fill_id] != identity:
                raise ValueError("Conflicting duplicate fill ID")
            continue
        seen[fill_id] = identity
        gross = qty * price
        if side == "buy":
            # Protect against a fill between the cash read and order-list read.
            if at >= started - timedelta(seconds=30):
                available -= gross
        else:
            trade_day = pd.Timestamp(at).tz_convert(NY).date()
            if trade_day < min(dates) or trade_day not in dates:
                raise ValueError("Fill date is not covered by exchange calendar")
            settlement = next((day for day in settlement_days if day > trade_day), None)
            if settlement is None:
                raise ValueError("Missing future cash settlement session")
            # Hold through the full expected T+1 day, not merely until its open.
            if today <= settlement:
                available -= gross
    return max(ZERO, available) * (1 - buffer_pct)
