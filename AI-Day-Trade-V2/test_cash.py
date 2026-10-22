"""Cash accounting regressions; all accounts, orders, and fills are simulated."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from common import Fill, Snapshot
from cash import IntentStore, budget, reservations


UTC = timezone.utc
D = Decimal


def moment(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def calendar_for(now):
    """Exchange-like weekdays; bank-only holidays deliberately stay open."""
    day = now.date() - timedelta(days=30)
    end = now.date() + timedelta(days=20)
    result = []
    exchange_holidays = {"2026-04-03", "2026-11-26", "2026-12-25", "2026-07-03"}
    while day <= end:
        if day.weekday() < 5 and day.isoformat() not in exchange_holidays:
            result.append(NS(date=day,
                             open=datetime.combine(day, datetime.min.time()).replace(hour=9, minute=30),
                             close=datetime.combine(day, datetime.min.time()).replace(hour=16)))
        day += timedelta(days=1)
    return result


def order(client_id="manual", symbol="NVDA", qty="5", filled_qty="0", limit="20",
          side="buy", status="new", kind="limit", order_id=None):
    return NS(id=order_id or "order-" + client_id, client_order_id=client_id,
              symbol=symbol, qty=qty, filled_qty=filled_qty, limit_price=limit,
              side=side, status=status, type=kind, order_type=kind,
              notional=None, legs=[])


def fill(identity, side, at, qty="2", price="100", symbol="NVDA"):
    return Fill(identity, "order-" + identity, symbol, side, D(qty), D(price), at)


class CashFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cash-state.json"
        self.store = IntentStore(self.path, "paper-account", True)
        self.now = moment("2026-10-21T15:00:00Z")

    def snapshot(self, *, cash="1000", buying_power="4000", non_marginable="1000",
                 orders=None, fills=None, started_at=None, fees="0", transfer_out="0"):
        account = NS(id="paper-account", cash=cash, buying_power=buying_power,
                     non_marginable_buying_power=non_marginable,
                     accrued_fees=fees, pending_transfer_out=transfer_out,
                     pending_transfer_in="0")
        return Snapshot(account, orders or [], [], fills or [], started_at or self.now)

    def available(self, snapshot=None, now=None, calendar=None):
        now = now or self.now
        return budget(snapshot or self.snapshot(), self.store,
                      calendar if calendar is not None else calendar_for(now), now, D(".05"))

    def reserve(self, client_id="intent", qty="5", limit="20", symbol="NVDA"):
        self.store.reserve(client_id, symbol, D(qty), D(limit), "semiconductors", 1, self.now)


class IntentStoreTests(CashFixture):
    def test_reservation_survives_restart_with_decimal_strings(self):
        self.reserve(qty="3", limit="20.12")
        self.assertTrue(self.path.exists())
        self.assertTrue(self.store.consumed("intent"))
        restored = IntentStore(self.path, "paper-account", True)
        self.assertTrue(restored.consumed("intent"))
        self.assertEqual(len(restored.active()), 1)
        held = reservations(self.snapshot(), restored)["intent"]
        self.assertEqual(held["cost"], D("60.36"))
        # JSON must not use binary floats for quantities and monetary amounts.
        raw = self.path.read_text()
        self.assertIn('"20.12"', raw)
        self.assertNotIn('NaN', raw)

    def test_wrong_account_or_mode_is_rejected(self):
        self.reserve()
        with self.assertRaises(ValueError):
            IntentStore(self.path, "different-account", True)
        with self.assertRaises(ValueError):
            IntentStore(self.path, "paper-account", False)

    def test_corrupt_state_fails_closed_without_replacing_file(self):
        bad = '{"broken":'
        self.path.write_text(bad)
        with self.assertRaises(ValueError):
            IntentStore(self.path, "paper-account", True)
        self.assertEqual(self.path.read_text(), bad)

    def test_malformed_json_state_fails_closed(self):
        self.path.write_text(json.dumps({"account_id": "paper-account", "paper": True,
                                         "intents": "not-an-intent-table"}))
        with self.assertRaises(ValueError):
            IntentStore(self.path, "paper-account", True)

    def test_unknown_order_keeps_reservation_and_consumes_signal(self):
        self.reserve()
        # No observation is supplied when a broker client-ID lookup returns 404.
        restored = IntentStore(self.path, "paper-account", True)
        self.assertTrue(restored.consumed("intent"))
        self.assertEqual(reservations(self.snapshot(), restored)["intent"]["cost"], D("100"))

    def test_partial_fill_reduces_remaining_reservation(self):
        self.reserve()
        self.store.observe(order("intent", filled_qty="2", status="partially_filled"))
        self.assertEqual(reservations(self.snapshot(), self.store)["intent"]["cost"], D("60"))
        self.assertTrue(self.store.consumed("intent"))

    def test_observed_higher_limit_increases_remaining_reservation(self):
        self.reserve()
        self.store.observe(order("intent", filled_qty="2", status="partially_filled", limit="30"))
        self.assertEqual(reservations(self.snapshot(), self.store)["intent"]["cost"], D("90"))

    def test_filled_status_without_full_fill_does_not_release_cash(self):
        self.reserve()
        with self.assertRaises(ValueError):
            self.store.observe(order("intent", status="filled", filled_qty="0"))
        self.assertEqual(reservations(self.snapshot(), self.store)["intent"]["cost"], D("100"))

    def test_terminal_orders_release_cash_but_do_not_reuse_signal(self):
        for status in ("filled", "canceled", "expired", "rejected"):
            with self.subTest(status=status):
                client_id = "intent-" + status
                self.reserve(client_id)
                self.store.observe(order(client_id, filled_qty="5" if status == "filled" else "0", status=status))
                self.assertTrue(self.store.consumed(client_id))
                self.assertNotIn(client_id, reservations(self.snapshot(), self.store))

    def test_definitive_rejection_releases_cash_and_keeps_signal_consumed(self):
        self.reserve()
        self.store.reject("intent")
        self.assertTrue(self.store.consumed("intent"))
        self.assertEqual(self.store.active(), [])

    def test_observing_other_orders_does_not_release_an_unknown_intent(self):
        self.reserve()
        self.store.observe(order("unrelated", status="filled", filled_qty="5"))
        self.assertEqual(len(self.store.active()), 1)

    def test_observation_without_broker_id_cannot_release_reservation(self):
        self.reserve()
        bad = order("intent", status="filled", filled_qty="5")
        bad.id = None
        with self.assertRaises(ValueError):
            self.store.observe(bad)
        self.assertEqual(reservations(self.snapshot(), self.store)["intent"]["cost"], D("100"))

    def test_duplicate_reserve_preserves_the_original_intent(self):
        self.reserve()
        with self.assertRaises(ValueError):
            self.reserve(qty="20")
        self.assertEqual(reservations(self.snapshot(), self.store)["intent"]["cost"], D("100"))

    def test_invalid_quantities_and_prices_never_reserve_cash(self):
        for qty, price in (("0", "20"), ("-1", "20"), ("1", "0"),
                           ("1", "NaN"), ("Infinity", "20")):
            with self.subTest(qty=qty, price=price):
                with self.assertRaises(ValueError):
                    self.reserve("bad", qty=qty, limit=price)
        self.assertEqual(self.store.active(), [])


class ReservationTests(CashFixture):
    def test_manual_buys_outside_watchlist_are_reserved(self):
        result = reservations(self.snapshot(orders=[order(symbol="AAPL", qty="3", limit="50")]), self.store)
        self.assertEqual(sum(item["cost"] for item in result.values()), D("150"))

    def test_local_and_broker_reservation_are_not_double_counted(self):
        self.reserve()
        result = reservations(self.snapshot(orders=[order("intent")]), self.store)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["intent"]["cost"], D("100"))

    def test_broker_and_local_disagreement_uses_larger_cost(self):
        self.reserve()
        result = reservations(self.snapshot(orders=[order("intent", qty="8")]), self.store)
        self.assertEqual(result["intent"]["cost"], D("160"))

    def test_partial_broker_order_reserves_remaining_shares(self):
        result = reservations(self.snapshot(orders=[order(qty="10", filled_qty="4", limit="25", status="partially_filled")]), self.store)
        self.assertEqual(sum(item["cost"] for item in result.values()), D("150"))

    def test_sell_and_terminal_buy_orders_do_not_reserve_cash(self):
        result = reservations(self.snapshot(orders=[order("sell", side="sell"),
                                                     order("filled", status="filled", filled_qty="5")]), self.store)
        self.assertEqual(result, {})

    def test_unknown_open_market_buy_fails_closed(self):
        with self.assertRaises(ValueError):
            reservations(self.snapshot(orders=[order(kind="market", limit=None)]), self.store)

    def test_invalid_pending_order_amounts_fail_closed(self):
        for changes in ({"qty": None}, {"qty": "-1"}, {"filled_qty": "6"},
                        {"filled_qty": "-1"}, {"limit": "0"}, {"limit": "NaN"}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    reservations(self.snapshot(orders=[order(**changes)]), self.store)


class BudgetTests(CashFixture):
    def test_cash_bound_prevents_using_margin_multiplier(self):
        self.assertEqual(self.available(), D("950"))
        self.assertEqual(self.available(self.snapshot(non_marginable="600")), D("570"))
        self.assertEqual(self.available(self.snapshot(buying_power="400")), D("380"))

    def test_pending_buys_fees_and_outgoing_transfers_reduce_budget(self):
        snapshot = self.snapshot(orders=[order(qty="5", limit="20")], fees="10", transfer_out="40")
        self.assertEqual(self.available(snapshot), D("807.50"))

    def test_unknown_local_intent_reserves_money_without_broker_order(self):
        self.reserve()
        self.assertEqual(self.available(), D("855"))

    def test_reserved_cost_above_cash_never_makes_negative_budget(self):
        self.reserve(qty="100", limit="20")
        self.assertEqual(self.available(), D("0"))

    def test_negative_cash_provides_no_spending_capacity(self):
        self.assertEqual(self.available(self.snapshot(cash="-100")), D("0"))

    def test_nonfinite_balance_fails_closed(self):
        with self.assertRaises(ValueError):
            self.available(self.snapshot(cash="NaN"))

    def test_account_identity_is_required_and_must_match_state(self):
        for identity in (None, "unrelated-account"):
            with self.subTest(identity=identity):
                snapshot = self.snapshot()
                snapshot.account.id = identity
                with self.assertRaises(ValueError):
                    self.available(snapshot)
        snapshot = self.snapshot()
        del snapshot.account.id
        with self.assertRaises(ValueError):
            self.available(snapshot)

    def test_recent_buy_fill_covers_race_after_cash_snapshot(self):
        snapshot = self.snapshot(started_at=self.now - timedelta(seconds=10),
                                 fills=[fill("recent-buy", "buy", self.now - timedelta(seconds=5))])
        self.assertEqual(self.available(snapshot), D("760"))

    def test_buy_fill_clock_skew_overlap_is_reserved(self):
        snapshot = self.snapshot(fills=[fill("skew-buy", "buy", self.now - timedelta(seconds=20))])
        self.assertEqual(self.available(snapshot), D("760"))

    def test_old_buy_fill_is_already_reflected_in_cash(self):
        snapshot = self.snapshot(fills=[fill("old-buy", "buy", self.now - timedelta(minutes=10))])
        self.assertEqual(self.available(snapshot), D("950"))

    def test_duplicate_fill_id_is_not_subtracted_twice(self):
        trade = fill("same-buy", "buy", self.now)
        self.assertEqual(self.available(self.snapshot(fills=[trade, trade])), D("760"))

    def test_conflicting_duplicate_fill_ids_fail_closed(self):
        one = fill("same-id", "buy", self.now)
        other = fill("same-id", "buy", self.now, qty="3")
        with self.assertRaises(ValueError):
            self.available(self.snapshot(fills=[one, other]))

    def test_invalid_fill_amount_or_future_timestamp_fails_closed(self):
        bad_fills = [fill("negative", "buy", self.now, qty="-1"),
                     fill("invalid-price", "sell", self.now, price="0"),
                     fill("future", "sell", self.now + timedelta(seconds=6))]
        for trade in bad_fills:
            with self.subTest(trade=trade.id):
                with self.assertRaises(ValueError):
                    self.available(self.snapshot(fills=[trade]))

    def test_fill_older_than_authoritative_history_window_fails_closed(self):
        at = self.now - timedelta(days=14, seconds=31)
        for side in ("buy", "sell"):
            with self.subTest(side=side):
                snapshot = self.snapshot(fills=[fill("old-" + side, side, at)])
                with self.assertRaises(ValueError):
                    self.available(snapshot)

    def test_insufficient_calendar_coverage_fails_closed(self):
        snapshot = self.snapshot(fills=[fill("sell", "sell", self.now - timedelta(days=1))])
        with self.assertRaises(ValueError):
            self.available(snapshot, calendar=[NS(date=self.now.date())])


class SettlementTests(CashFixture):
    def assert_sale_budget(self, sold_at, now, expected):
        now = moment(now)
        snapshot = self.snapshot(started_at=now, fills=[fill("sale", "sell", moment(sold_at))])
        self.assertEqual(self.available(snapshot, now=now), D(expected))

    def test_sale_is_held_during_trade_day_and_settlement_day(self):
        self.assert_sale_budget("2026-10-19T15:00:00Z", "2026-10-19T16:00:00Z", "760")
        self.assert_sale_budget("2026-10-19T15:00:00Z", "2026-10-20T16:00:00Z", "760")
        self.assert_sale_budget("2026-10-19T15:00:00Z", "2026-10-21T16:00:00Z", "950")

    def test_weekend_does_not_count_as_settlement_session(self):
        self.assert_sale_budget("2026-10-16T15:00:00Z", "2026-10-19T16:00:00Z", "760")
        self.assert_sale_budget("2026-10-16T15:00:00Z", "2026-10-20T16:00:00Z", "950")

    def test_columbus_day_is_bank_holiday_even_when_exchange_open(self):
        self.assert_sale_budget("2026-10-09T15:00:00Z", "2026-10-13T16:00:00Z", "760")
        self.assert_sale_budget("2026-10-09T15:00:00Z", "2026-10-14T16:00:00Z", "950")

    def test_veterans_day_is_bank_holiday_even_when_exchange_open(self):
        self.assert_sale_budget("2026-11-10T15:00:00Z", "2026-11-12T16:00:00Z", "760")
        self.assert_sale_budget("2026-11-10T15:00:00Z", "2026-11-13T16:00:00Z", "950")

    def test_exchange_holiday_does_not_count_as_settlement_session(self):
        self.assert_sale_budget("2026-04-02T15:00:00Z", "2026-04-06T16:00:00Z", "760")
        self.assert_sale_budget("2026-04-02T15:00:00Z", "2026-04-07T16:00:00Z", "950")

    def test_release_uses_new_york_date_instead_of_utc_date(self):
        self.assert_sale_budget("2026-10-19T19:00:00Z", "2026-10-21T01:00:00Z", "760")
        self.assert_sale_budget("2026-10-19T19:00:00Z", "2026-10-21T05:00:00Z", "950")


if __name__ == "__main__":
    unittest.main()
