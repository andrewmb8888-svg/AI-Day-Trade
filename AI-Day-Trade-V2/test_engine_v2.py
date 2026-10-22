"""Multi-symbol engine tests: simulated broker, real cash reservations, no sockets."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from cash import IntentStore, reservations
from common import BY_SYMBOL, Decision, Fill, Settings, Snapshot, order_key
from engine import Engine, entry_prices

D = Decimal
UTC = timezone.utc


def calendar_for(now):
    day = now.date() - timedelta(days=40)
    end = now.date() + timedelta(days=25)
    sessions = []
    while day <= end:
        if day.weekday() < 5:
            sessions.append(NS(date=day,
                               open=datetime.combine(day, datetime.min.time()).replace(hour=9, minute=30),
                               close=datetime.combine(day, datetime.min.time()).replace(hour=16)))
        day += timedelta(days=1)
    return sessions


def order(symbol, client_id, qty=1, limit=D("100.10"), *, status="new", filled=0,
          side="buy", identity=None, kind="limit"):
    return NS(id=identity or f"broker-{client_id}", client_order_id=client_id,
              symbol=symbol, qty=str(qty), filled_qty=str(filled), limit_price=str(limit),
              status=status, side=side, type=kind, legs=[])


class Rejected(ValueError):
    pass


class EngineFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state.json"
        self.store = IntentStore(self.path, "paper-fixture", True)
        self.now = datetime(2026, 10, 21, 15, 5, tzinfo=UTC)
        self.settings = Settings(symbols=("NVDA", "TSLA"))
        self.account = NS(id="paper-fixture", cash="10000", non_marginable_buying_power="10000",
                          buying_power="40000", equity="100000", pending_transfer_out="0", accrued_fees="0")
        self.snapshot = Snapshot(self.account, [], [], [], self.now)
        self.broker = Mock()
        self.broker.clock.return_value = NS(is_open=True)
        self.broker.cash_settings.return_value = True
        self.broker.lookup.return_value = None
        self.broker.snapshot.return_value = self.snapshot
        self.broker.calendar.return_value = calendar_for(self.now)
        self.broker.asset.return_value = NS(tradable=True)
        self.broker.buy.side_effect = self.accept
        self.broker.definitive_rejection.side_effect = lambda exc: isinstance(exc, Rejected)
        self.market = NS(update=Mock(), errors={})
        self.frame_data = {}
        self.quotes = {}
        self.set_frames()
        self.broker.quotes.side_effect = lambda symbols: {symbol: self.quotes[symbol] for symbol in symbols}
        self.decisions = {symbol: Decision("initial-events") for symbol in BY_SYMBOL}
        self.signal_patch = patch("engine.signals", side_effect=lambda bars, daily, window:
                                  self.decisions[bars.attrs["symbol"]])
        self.signals = self.signal_patch.start()
        self.addCleanup(self.signal_patch.stop)
        self.log_patch = patch("engine.LOG")
        self.log_patch.start()
        self.addCleanup(self.log_patch.stop)
        self.engine = self.build_engine()

    def set_frames(self, symbols=None, latest=None):
        symbols = symbols or self.settings.symbols
        latest = latest or pd.Timestamp(self.now).floor("30min") - pd.Timedelta(minutes=30)
        for symbol in symbols:
            bars = pd.DataFrame({"close": [100.] * 100},
                                index=pd.date_range(end=latest, periods=100, freq="30min"))
            bars.attrs["symbol"] = symbol
            daily = pd.DataFrame({"close": range(1, 251)},
                                 index=pd.date_range(end="2026-10-20T04:00Z", periods=250, freq="D"))
            daily.attrs["current_session_data"] = True
            self.frame_data[symbol] = bars, daily
            self.quotes[symbol] = NS(ask_price="100", bid_price="99.99", timestamp=self.now)
        self.market.update.return_value = self.frame_data

    def build_engine(self, store=None):
        return Engine(self.broker, self.market, self.settings, store or self.store,
                      clock=lambda: self.now)

    def accept(self, symbol, qty, client_id, limit, stop, take):
        # The durable reservation must exist before any network submission.
        self.assertTrue(self.store.consumed(client_id))
        self.assertTrue(self.path.exists())
        return order(symbol, client_id, qty, limit)

    def buy_costs(self):
        return [D(call.args[1]) * call.args[3] for call in self.broker.buy.call_args_list]


class MultiSymbolEngineTests(EngineFixture):
    def test_two_qualifying_symbols_can_buy_in_same_cycle(self):
        results = self.engine.step(self.now)
        self.assertIn("buy submitted", results["NVDA"])
        self.assertIn("buy submitted", results["TSLA"])
        self.assertEqual(self.broker.buy.call_count, 2)
        self.broker.quotes.assert_called_once_with(["NVDA", "TSLA"])
        self.assertLessEqual(sum(self.buy_costs()), D("9500"))
        self.assertNotEqual(self.broker.buy.call_args_list[0].args[2],
                            self.broker.buy.call_args_list[1].args[2])

    def test_immediately_filled_first_order_cannot_reuse_same_cash(self):
        self.account.cash = self.account.non_marginable_buying_power = "1800"
        self.settings = Settings(symbols=("NVDA", "TSLA"), allocation=D(".5"), group_cap=D(".9"))
        self.engine = self.build_engine()

        def immediate(symbol, qty, client_id, limit, stop, take):
            self.assertTrue(self.store.consumed(client_id))
            return order(symbol, client_id, qty, limit, status="filled", filled=qty)

        self.broker.buy.side_effect = immediate
        results = self.engine.step(self.now)
        self.broker.buy.assert_called_once()
        self.assertIn("buy submitted", results["NVDA"])
        self.assertIn("below one share", results["TSLA"])
        self.assertLessEqual(sum(self.buy_costs()), D("1710"))
        self.assertEqual(self.store.active(), [])

    def test_all_twenty_qualifying_symbols_have_no_trade_count_quota(self):
        self.settings = Settings()
        self.account.cash = self.account.non_marginable_buying_power = "100000"
        self.set_frames()
        for quote in self.quotes.values():
            quote.ask_price = quote.bid_price = "1"
        self.engine = self.build_engine()
        results = self.engine.step(self.now)
        self.assertEqual(self.broker.buy.call_count, 20)
        self.assertEqual(sum("buy submitted" in value for value in results.values()), 20)
        self.assertLessEqual(sum(self.buy_costs()), D("95000"))
        self.assertEqual(len(self.store.active()), 20)

    def test_timeout_reserves_money_and_survives_restart_without_duplicate(self):
        def submit(symbol, qty, client_id, limit, stop, take):
            if symbol == "NVDA":
                raise TimeoutError("submission result unknown")
            return self.accept(symbol, qty, client_id, limit, stop, take)

        self.broker.buy.side_effect = submit
        results = self.engine.step(self.now)
        self.assertIn("failed", results["NVDA"])
        self.assertIn("buy submitted", results["TSLA"])
        restored = IntentStore(self.path, "paper-fixture", True)
        self.engine = self.build_engine(restored)
        self.engine.step(self.now)
        self.assertEqual(self.broker.buy.call_count, 2)
        self.assertGreater(sum(item["cost"] for item in reservations(self.snapshot, restored).values()), 0)

    def test_same_signal_is_consumed_on_later_bar_after_restart(self):
        self.engine.step(self.now)
        self.now += timedelta(minutes=30)
        self.snapshot.started_at = self.now
        self.set_frames()
        restored = IntentStore(self.path, "paper-fixture", True)
        for intent in restored.active():
            restored.observe(order(intent["symbol"], intent["client_id"], intent["qty"],
                                   D(intent["limit"]), status="filled", filled=intent["qty"]))
        self.engine = self.build_engine(restored)
        results = self.engine.step(self.now)
        self.assertEqual(self.broker.buy.call_count, 2)
        self.assertTrue(all(value == "signal already submitted" for value in results.values()))

    def test_a_genuinely_new_event_after_closed_position_can_trade(self):
        self.engine.step(self.now)
        for intent in self.store.active():
            self.store.observe(order(intent["symbol"], intent["client_id"], intent["qty"],
                                     D(intent["limit"]), status="filled", filled=intent["qty"]))
        self.now += timedelta(minutes=30)
        self.snapshot.started_at = self.now
        self.set_frames()
        self.decisions = {symbol: Decision("genuinely-new-events") for symbol in BY_SYMBOL}
        self.engine.step(self.now)
        self.assertEqual(self.broker.buy.call_count, 4)

    def test_missing_current_bar_prevents_that_symbol_only(self):
        bars, daily = self.frame_data["NVDA"]
        self.frame_data["NVDA"] = bars.iloc[:-1], daily
        results = self.engine.step(self.now)
        self.assertIn("waiting", results["NVDA"])
        self.assertIn("buy submitted", results["TSLA"])
        self.broker.buy.assert_called_once()

    def test_first_bar_must_complete_before_entries(self):
        self.now = self.now.replace(hour=13, minute=55)
        self.snapshot.started_at = self.now
        result = self.engine.step(self.now)
        self.assertIn("waiting", result["market"])
        self.market.update.assert_not_called()
        self.broker.buy.assert_not_called()

    def test_bad_quote_does_not_starve_other_symbol(self):
        self.quotes["NVDA"].ask_price = "nan"
        results = self.engine.step(self.now)
        self.assertIn("failed", results["NVDA"])
        self.assertIn("buy submitted", results["TSLA"])
        self.broker.buy.assert_called_once()

    def test_unavailable_asset_does_not_starve_other_symbol(self):
        self.broker.asset.side_effect = lambda symbol: (_ for _ in ()).throw(ValueError("disabled")) if symbol == "NVDA" else NS(tradable=True)
        results = self.engine.step(self.now)
        self.assertIn("skipped", results["NVDA"])
        self.assertIn("buy submitted", results["TSLA"])

    def test_existing_position_prevents_additional_buy(self):
        self.snapshot.positions = [NS(symbol="NVDA", qty="2", market_value="200")]
        results = self.engine.step(self.now)
        self.assertIn("holding", results["NVDA"])
        self.assertEqual(self.broker.buy.call_args.args[0], "TSLA")
        self.broker.buy.assert_called_once()

    def test_open_buy_prevents_duplicate_entry(self):
        self.snapshot.orders = [order("NVDA", "manual-pending")]
        results = self.engine.step(self.now)
        self.assertIn("pending", results["NVDA"])
        self.broker.buy.assert_called_once()
        self.assertEqual(self.broker.buy.call_args.args[0], "TSLA")

    def test_protective_bracket_is_retained_on_exit_signal(self):
        self.snapshot.positions = [NS(symbol="NVDA", qty="2", market_value="200")]
        self.snapshot.orders = [order("NVDA", "protective-stop", side="sell", kind="stop")]
        self.decisions["NVDA"] = Decision(exit_signal=True)
        results = self.engine.step(self.now)
        self.assertIn("protective orders retained", results["NVDA"])
        self.broker.close.assert_not_called()

    def test_unprotected_exit_uses_broker_close_current_position(self):
        self.snapshot.positions = [NS(symbol="NVDA", qty="1.5", market_value="150")]
        self.decisions["NVDA"] = Decision(exit_signal=True)
        results = self.engine.step(self.now)
        self.assertEqual(results["NVDA"], "close submitted")
        self.broker.close.assert_called_once_with("NVDA")

    def test_disappearing_position_at_close_is_not_a_sell(self):
        self.snapshot.positions = [NS(symbol="NVDA", qty="1.5", market_value="150")]
        self.decisions["NVDA"] = Decision(exit_signal=True)
        self.broker.close.return_value = None
        results = self.engine.step(self.now)
        self.assertEqual(results["NVDA"], "already closed")

    def test_submitted_sell_does_not_fund_other_buy(self):
        self.account.cash = self.account.non_marginable_buying_power = "0"
        self.snapshot.positions = [NS(symbol="NVDA", qty="5", market_value="500")]
        self.decisions["NVDA"] = Decision(exit_signal=True)
        results = self.engine.step(self.now)
        self.assertEqual(results["NVDA"], "close submitted")
        self.assertEqual(results["TSLA"], "no eligible cash available")
        self.broker.buy.assert_not_called()

    def test_margin_setting_change_prevents_all_new_orders(self):
        self.broker.cash_settings.return_value = False
        with self.assertRaisesRegex(ValueError, "cash-only"):
            self.engine.step(self.now)
        self.broker.snapshot.assert_not_called()
        self.broker.buy.assert_not_called()

    def test_each_account_block_flag_prevents_entries_and_closes(self):
        self.snapshot.positions = [NS(symbol="NVDA", qty="2", market_value="200")]
        self.decisions["NVDA"] = Decision(exit_signal=True)
        for field in ("trading_blocked", "account_blocked", "trade_suspended_by_user"):
            with self.subTest(field=field):
                setattr(self.account, field, True)
                with self.assertRaisesRegex(ValueError, "blocked or suspended"):
                    self.engine.step(self.now)
                setattr(self.account, field, False)
        self.broker.buy.assert_not_called()
        self.broker.close.assert_not_called()
        self.broker.quotes.assert_not_called()
        self.market.update.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_quote_freshness_uses_clock_after_batch_fetch(self):
        fetched = False

        def fetch_quotes(symbols):
            nonlocal fetched
            fetched = True
            return {symbol: NS(ask_price="100", bid_price="99.99",
                               timestamp=self.now + timedelta(seconds=19))
                    for symbol in symbols}

        # The latest quote arrived after cycle start but before the response.
        # Comparing it to cycle start would incorrectly label it a future quote.
        self.engine.clock = lambda: self.now + timedelta(seconds=20) if fetched else self.now
        self.broker.quotes.side_effect = fetch_quotes
        results = self.engine.step(self.now)
        self.assertTrue(fetched)
        self.assertTrue(all("buy submitted" in value for value in results.values()))
        self.assertEqual(self.broker.buy.call_count, 2)

    def test_market_closing_during_quote_fetch_prevents_submission(self):
        self.broker.clock.side_effect = [NS(is_open=True), NS(is_open=False)]
        results = self.engine.step(self.now)
        self.assertEqual(self.broker.clock.call_count, 2)
        self.broker.quotes.assert_called_once()
        self.assertTrue(all(value == "market closed before submission" for value in results.values()))
        self.broker.buy.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_bar_boundary_crossed_during_quote_fetch_requires_fresh_data(self):
        fetched = False

        def fetch_quotes(symbols):
            nonlocal fetched
            fetched = True
            return {symbol: self.quotes[symbol] for symbol in symbols}

        # The cycle evaluated the 14:30 bar; after 15:30:03 the completed target
        # is the 15:00 bar. Do not submit against an old target after a slow call.
        self.engine.clock = lambda: self.now.replace(minute=30, second=4) if fetched else self.now
        self.broker.quotes.side_effect = fetch_quotes
        results = self.engine.step(self.now)
        self.assertTrue(all("bar/session changed" in value for value in results.values()))
        self.broker.buy.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_reported_close_time_passed_before_buy_prevents_submission(self):
        fetched = False
        close_at = self.now + timedelta(seconds=10)
        self.broker.clock.return_value = NS(is_open=True, next_close=close_at)

        def fetch_quotes(symbols):
            nonlocal fetched
            fetched = True
            return {symbol: self.quotes[symbol] for symbol in symbols}

        # Even if the cached/refreshed broker response still says open, the
        # known next_close is authoritative once the local wall clock passes it.
        self.engine.clock = lambda: close_at + timedelta(seconds=1) if fetched else self.now
        self.broker.quotes.side_effect = fetch_quotes
        results = self.engine.step(self.now)
        self.assertTrue(all("bar/session changed" in value for value in results.values()))
        self.broker.buy.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_reported_close_time_passed_before_unprotected_close_is_skipped(self):
        data_received = False
        close_at = self.now + timedelta(seconds=10)
        self.broker.clock.return_value = NS(is_open=True, next_close=close_at)
        self.snapshot.positions = [NS(symbol="NVDA", qty="2", market_value="200")]
        self.decisions["NVDA"] = Decision(exit_signal=True)

        def fetch_data(*args):
            nonlocal data_received
            data_received = True
            return self.frame_data

        self.engine.clock = lambda: close_at + timedelta(seconds=1) if data_received else self.now
        self.market.update.side_effect = fetch_data
        results = self.engine.step(self.now)
        self.assertIn("bar/session changed", results["NVDA"])
        self.broker.close.assert_not_called()
        self.broker.buy.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_broker_cash_guard_stops_before_pending_intent_reconciliation(self):
        self.store.reserve("awaiting-client-id", "NVDA", 1, D("100.10"),
                           "semiconductors", 1, self.now)
        self.broker.cash_settings.return_value = False
        with self.assertRaisesRegex(ValueError, "cash-only"):
            self.engine.step(self.now)
        self.broker.lookup.assert_not_called()
        self.broker.snapshot.assert_not_called()
        self.market.update.assert_not_called()
        self.broker.buy.assert_not_called()
        self.broker.close.assert_not_called()
        self.assertEqual(len(self.store.active()), 1)

    def test_closed_market_does_not_fetch_or_trade(self):
        self.broker.clock.return_value.is_open = False
        self.assertEqual(self.engine.step(self.now), {"market": "closed"})
        self.market.update.assert_not_called()
        self.broker.snapshot.assert_not_called()

    def test_leveraged_position_has_smaller_dollar_allocation(self):
        self.settings = Settings(symbols=("NVDA", "SOXL"))
        self.set_frames()
        self.engine = self.build_engine()
        self.engine.step(self.now)
        calls = {call.args[0]: call.args for call in self.broker.buy.call_args_list}
        self.assertEqual(calls["NVDA"][1], 19)
        self.assertEqual(calls["SOXL"][1], 6)
        self.assertLessEqual(calls["SOXL"][1] * calls["SOXL"][3] * 3, D("2000"))

    def test_existing_group_exposure_reduces_new_position(self):
        self.settings = Settings(symbols=("NVDA", "SOXL"))
        self.snapshot.positions = [NS(symbol="AMD", qty="100", market_value="9900")]
        self.set_frames()
        self.engine = self.build_engine()
        results = self.engine.step(self.now)
        self.broker.buy.assert_not_called()  # Only $100 group room; one share costs $100.10.
        self.assertTrue(all("below one share" in value for value in results.values()))

    def test_pending_group_order_uses_group_exposure_budget(self):
        self.account.cash = self.account.non_marginable_buying_power = "50000"
        self.snapshot.orders = [order("AMD", "pending-other", qty=100, limit=D("100"))]
        results = self.engine.step(self.now)
        self.assertIn("below one share", results["NVDA"])
        self.assertIn("buy submitted", results["TSLA"])

    def test_stale_or_future_quotes_never_submit(self):
        for seconds in (-121, 1):
            with self.subTest(offset=seconds):
                self.quotes["NVDA"].timestamp = self.now + timedelta(seconds=seconds)
                self.quotes["TSLA"].timestamp = self.now + timedelta(seconds=seconds)
                self.engine.step(self.now)
        self.broker.buy.assert_not_called()

    def test_repeated_poll_reuses_indicator_calculations(self):
        self.decisions = {symbol: Decision() for symbol in BY_SYMBOL}
        self.engine.step(self.now)
        self.engine.step(self.now + timedelta(seconds=10))
        self.assertEqual(self.signals.call_count, 2)

    def test_broker_lookup_existing_id_prevents_resubmission(self):
        self.broker.lookup.side_effect = lambda cid: order("NVDA", cid) if cid == order_key("NVDA", "initial-events") else None
        results = self.engine.step(self.now)
        self.assertIn("exists at broker", results["NVDA"])
        self.broker.buy.assert_called_once()
        self.assertEqual(self.broker.buy.call_args.args[0], "TSLA")

    def test_definitive_rejection_releases_reservation_but_consumes_signal(self):
        self.broker.buy.side_effect = Rejected("not tradable")
        self.engine.step(self.now)
        self.assertEqual(self.store.active(), [])
        self.engine.step(self.now)
        self.assertEqual(self.broker.buy.call_count, 2)

    def test_journal_failure_prevents_submission(self):
        with patch.object(self.store, "_save", side_effect=OSError("disk full")):
            results = self.engine.step(self.now)
        self.assertTrue(all("failed" in value for value in results.values()))
        self.broker.buy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
