"""Offline SDK contracts; HTTP transport is intercepted before any socket opens."""
import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import requests
from requests.adapters import HTTPAdapter
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderClass, TimeInForce
from alpaca.trading.models import AccountConfiguration

from broker import Broker, definitive_rejection
from common import Settings, TimeoutAdapter

UTC = timezone.utc
NOW = datetime(2026, 10, 22, 15, 0, tzinfo=UTC)
PAPER = {"ALPACA_PAPER_API_KEY": "offline-key", "ALPACA_PAPER_SECRET_KEY": "offline-secret"}


def error(status):
    return APIError('{"message":"offline test"}', requests.HTTPError(response=NS(status_code=status)))


def account(**changes):
    fields = dict(id="account-one", currency="USD", status="ACTIVE", cash="1000",
                  buying_power="4000", non_marginable_buying_power="1000", equity="10000",
                  short_market_value="0", pending_transfer_out="0", accrued_fees="0",
                  trading_blocked=False, account_blocked=False, trade_suspended_by_user=False)
    return NS(**{**fields, **changes})


def activity(identity="fill-one", **changes):
    fields = dict(id=identity, activity_type="FILL", order_id="order-one", symbol="NVDA",
                  side="buy", qty="2", price="100", transaction_time=NOW.isoformat())
    return {**fields, **changes}


def order(identity="order-one", **changes):
    fields = dict(id=identity, client_order_id="adv2-offline", symbol="NVDA", asset_class="us_equity",
                  status="new", side="buy", qty="2", filled_qty="0", limit_price="100", legs=[])
    return NS(**{**fields, **changes})


def broker():
    result = object.__new__(Broker)
    result.settings = Settings(symbols=("NVDA", "AMD"))
    result.trade = Mock()
    result.data = Mock()
    result._assets = {}
    result._calendar = None
    result._calendar_day = None
    result.disabled = {}
    result.trade.get_asset.return_value = NS(asset_class="us_equity", tradable=True, status="active")
    return result


def config(**changes):
    fields = dict(dtbp_check="entry", fractional_trading=True, max_margin_multiplier="4",
                  no_shorting=False, pdt_check="both", suspend_trade=False,
                  trade_confirm_email="none", ptp_no_exception_entry=True,
                  max_options_trading_level=0)
    return AccountConfiguration(**{**fields, **changes})


class CredentialsAndTransportTests(unittest.TestCase):
    def test_credentials_and_live_gate(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "credentials"):
                Broker(Settings())
        with patch.dict(os.environ, {"ALPACA_API_KEY": "offline", "ALPACA_SECRET_KEY": "offline"}, clear=True):
            with self.assertRaisesRegex(ValueError, "ALLOW_LIVE_TRADING"):
                Broker(Settings(paper=False))
        with patch.dict(os.environ, {**PAPER, "TRADE_API_URL": "https://example.invalid"}, clear=True):
            with self.assertRaisesRegex(ValueError, "TRADE_API_URL"):
                Broker(Settings())

    def test_real_sdk_batch_quotes_and_transport_timeouts(self):
        with patch.dict(os.environ, PAPER, clear=True):
            item = Broker(Settings(symbols=("NVDA", "AMD")))
        for client in (item.trade, item.data):
            self.assertIsInstance(client._session.get_adapter("https://example.test"), TimeoutAdapter)
        payloads = [
            {"timestamp": NOW.isoformat(), "is_open": True,
             "next_open": "2026-10-23T13:30:00Z", "next_close": "2026-10-22T20:00:00Z"},
            {"quotes": {symbol: {"t": NOW.isoformat(), "ap": 100, "as": 10, "bp": 99.99, "bs": 10}
                        for symbol in ("NVDA", "AMD")}},
        ]
        responses = []
        for payload in payloads:
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps(payload).encode()
            responses.append(response)
        with patch.object(HTTPAdapter, "send", side_effect=responses) as send:
            self.assertTrue(item.clock().is_open)
            self.assertEqual(set(item.quotes(["NVDA", "AMD"])), {"NVDA", "AMD"})
        self.assertEqual(send.call_count, 2)
        for call in send.call_args_list:
            self.assertEqual(call.kwargs["timeout"], (10, 30))
        self.assertIn("paper-api.alpaca.markets/v2/clock", send.call_args_list[0].args[0].url)
        quote_url = send.call_args_list[1].args[0].url
        self.assertIn("feed=iex", quote_url)
        self.assertIn("symbols=NVDA%2CAMD", quote_url)
        item.trade._session.close()
        item.data._session.close()


class CashPolicyTests(unittest.TestCase):
    def test_cash_guard_is_read_only(self):
        item = broker()
        item.trade.get_account_configurations.return_value = config()
        self.assertFalse(item.cash_settings())
        item.trade.set_account_configurations.assert_not_called()
        item.trade.get_account_configurations.return_value = config(max_margin_multiplier="1", no_shorting=True)
        self.assertTrue(item.cash_settings())

    def test_enforcement_preserves_other_settings_and_verifies(self):
        item = broker()
        original = config()
        verified = original.model_copy(update={"max_margin_multiplier": "1", "no_shorting": True})
        item.trade.get_account_configurations.side_effect = [original, verified]
        self.assertIs(item.enforce_cash_only(), verified)
        updated = item.trade.set_account_configurations.call_args.args[0]
        self.assertEqual(updated.max_margin_multiplier, "1")
        self.assertTrue(updated.no_shorting)
        for name in type(original).model_fields:
            if name not in {"max_margin_multiplier", "no_shorting"}:
                self.assertEqual(getattr(updated, name), getattr(original, name))
        self.assertEqual(original.max_margin_multiplier, "4")

    def test_failed_enforcement_prevents_trading(self):
        item = broker()
        item.trade.get_account_configurations.return_value = config()
        with self.assertRaisesRegex(ValueError, "not confirmed"):
            item.enforce_cash_only()
        item.trade.submit_order.assert_not_called()


class FillHistoryTests(unittest.TestCase):
    def test_pagination_and_correct_time_parameters(self):
        item = broker()
        page = [activity(f"fill-{i:03}") for i in range(100)]
        item.trade.get.side_effect = [page, [activity("last-fill", side="sell")]]
        with patch("broker._utc_now", return_value=NOW):
            fills = item.fills(NOW)
        self.assertEqual(len(fills), 101)
        self.assertEqual(fills[0].gross, Decimal("200"))
        first = item.trade.get.call_args_list[0]
        self.assertEqual(first.args, ("/account/activities/FILL",))
        self.assertIn("after", first.kwargs["data"])
        self.assertIn("until", first.kwargs["data"])
        self.assertNotIn("before", first.kwargs["data"])
        self.assertEqual(item.trade.get.call_args_list[1].kwargs["data"]["page_token"], "fill-099")

    def test_duplicate_fill_deduplication_and_conflict_rejection(self):
        item = broker()
        item.trade.get.return_value = [activity(), activity()]
        self.assertEqual(len(item.fills(NOW)), 1)
        item.trade.get.return_value = [activity(), activity(price="101")]
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            item.fills(NOW)

    def test_repeated_page_fails_coverage(self):
        item = broker()
        item.trade.get.return_value = [activity(f"fill-{i}") for i in range(100)]
        with self.assertRaisesRegex(ValueError, "pagination"):
            item.fills(NOW)
        self.assertEqual(item.trade.get.call_count, 2)

    def test_bad_fill_data_fails_closed(self):
        bad_rows = [activity(qty="NaN"), activity(price="0"), activity(side="sell_short"),
                    activity(transaction_time="2026-10-22T15:00:00"),
                    activity(transaction_time=(NOW + timedelta(days=1)).isoformat()),
                    activity(transaction_time=(NOW - timedelta(days=15)).isoformat()),
                    activity(activity_type="DIV"), {"id": "incomplete"}]
        for row in bad_rows:
            with self.subTest(row=row):
                item = broker()
                item.trade.get.return_value = [row]
                with patch("broker._utc_now", return_value=NOW):
                    with self.assertRaises(ValueError):
                        item.fills(NOW)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        current_time = patch("broker._utc_now", return_value=NOW)
        current_time.start()
        self.addCleanup(current_time.stop)
        self.item = broker()
        self.item.trade.get_account.side_effect = [account(), account(cash="800", buying_power="3200",
            non_marginable_buying_power="800", accrued_fees="2", pending_transfer_out="5", trading_blocked=True)]
        self.item.trade.get.return_value = []
        self.item.trade.get_orders.return_value = []
        self.item.trade.get_all_positions.return_value = []

    def test_minimum_balances_maximum_fees_and_cash_first_order(self):
        with patch("broker._utc_now", return_value=NOW):
            snapshot = self.item.snapshot(NOW - timedelta(minutes=5))
        self.assertEqual(snapshot.account.cash, Decimal("800"))
        self.assertEqual(snapshot.account.buying_power, Decimal("3200"))
        self.assertEqual(snapshot.account.accrued_fees, Decimal("2"))
        self.assertEqual(snapshot.account.pending_transfer_out, Decimal("5"))
        self.assertTrue(snapshot.account.trading_blocked)
        self.assertEqual(snapshot.started_at, NOW)
        calls = [call[0] for call in self.item.trade.mock_calls]
        self.assertEqual(calls[:4], ["get_account", "get", "get_orders", "get_all_positions"])
        self.assertEqual(calls[4], "get_account")
        self.assertEqual(self.item.trade.get.call_args.kwargs["data"]["until"], NOW.isoformat())

    def test_flatten_nested_orders_and_keep_active_protective_legs(self):
        leg = order("exit-leg", client_order_id="exit-client", side="sell", status="held")
        parent = order("entry-parent", status="filled", filled_qty="2", legs=[leg])
        self.item.trade.get_orders.return_value = [parent, leg]
        snapshot = self.item.snapshot(NOW)
        self.assertEqual([row.id for row in snapshot.orders], ["exit-leg"])

    def test_full_order_page_is_not_assumed_complete(self):
        self.item.trade.get_orders.return_value = [order(str(i)) for i in range(500)]
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.item.snapshot(NOW)

    def test_pending_review_order_remains_reserved_in_snapshot(self):
        pending = order(status="pending_review")
        self.item.trade.get_orders.return_value = [pending]
        snapshot = self.item.snapshot(NOW)
        self.assertEqual(snapshot.orders, [pending])

    def test_wrong_account_or_currency_fails_closed(self):
        for second in [account(id="different"), account(currency="EUR"), account(short_market_value="-10")]:
            with self.subTest(second=second):
                self.item.trade.get_account.side_effect = [account(), second]
                with self.assertRaises(ValueError):
                    self.item.snapshot(NOW)

    def test_crypto_fill_is_not_treated_as_a_stock_fill(self):
        self.item.trade.get.return_value = [activity(symbol="BTCUSD")]
        self.item.trade.get_asset.return_value = NS(asset_class="crypto", tradable=True, status="active")
        with self.assertRaisesRegex(ValueError, "US stocks"):
            self.item.snapshot(NOW)

    def test_short_positions_and_unsupported_orders_fail_closed(self):
        self.item.trade.get_all_positions.return_value = [NS(symbol="NVDA", qty="-2", asset_class="us_equity")]
        with self.assertRaisesRegex(ValueError, "Short"):
            self.item.snapshot(NOW)
        self.item.trade.get_all_positions.return_value = []
        self.item.trade.get_orders.return_value = [order(asset_class="us_option")]
        self.item.trade.get_account.side_effect = [account(), account()]
        with self.assertRaisesRegex(ValueError, "Non-equity"):
            self.item.snapshot(NOW)


class OrderAndAssetTests(unittest.TestCase):
    def test_real_sdk_bracket_request(self):
        item = broker()
        item.buy("NVDA", 2, "adv2-test", Decimal("100.10"), Decimal("97.10"), Decimal("106.11"))
        request = item.trade.submit_order.call_args.kwargs["order_data"]
        self.assertEqual(request.order_class, OrderClass.BRACKET)
        self.assertEqual(request.time_in_force, TimeInForce.GTC)
        self.assertEqual(request.symbol, "NVDA")
        self.assertEqual(request.qty, 2)
        self.assertFalse(request.extended_hours)
        self.assertEqual(request.stop_loss.stop_price, 97.1)
        for qty in (0, -1, 1.5, True):
            with self.subTest(qty=qty):
                with self.assertRaises(ValueError):
                    item.buy("NVDA", qty, "bad", 100, 97, 106)

    def test_lookup_and_close_only_catch_404(self):
        item = broker()
        for method, args, target in [(item.lookup, ("client",), item.trade.get_order_by_client_id),
                                     (item.close, ("NVDA",), item.trade.close_position)]:
            target.side_effect = error(404)
            self.assertIsNone(method(*args))
            for code in (401, 403, 429, 500):
                target.side_effect = error(code)
                with self.assertRaises(APIError):
                    method(*args)
            target.side_effect = requests.Timeout("offline")
            with self.assertRaises(requests.Timeout):
                method(*args)

    def test_definitive_rejections_do_not_include_ambiguous_failures(self):
        for code in (400, 403, 422):
            self.assertTrue(definitive_rejection(error(code)))
        for exc in (error(409), error(429), error(500), requests.Timeout("offline")):
            self.assertFalse(definitive_rejection(exc))
        duplicate = APIError('{"message":"client_order_id must be unique"}',
                             requests.HTTPError(response=NS(status_code=422)))
        self.assertFalse(definitive_rejection(duplicate))
        self.assertFalse(Broker.definitive_rejection(duplicate))

    def test_asset_lookup_is_cached_and_bad_symbols_are_isolated(self):
        item = broker()
        item.trade.get_asset.side_effect = [NS(asset_class="us_equity", tradable=True, status="active"), error(404)]
        self.assertEqual(set(item.assets()), {"NVDA"})
        self.assertIn("AMD", item.disabled)
        item.asset("NVDA")
        self.assertEqual(item.trade.get_asset.call_count, 2)

    def test_calendar_coverage_and_daily_cache(self):
        item = broker()
        today = NOW.date()
        item.trade.get_calendar.return_value = [NS(date=today - timedelta(days=105)), NS(date=today + timedelta(days=15))]
        first = item.calendar(NOW)
        self.assertIs(first, item.calendar(NOW + timedelta(minutes=5)))
        item.trade.get_calendar.assert_called_once()
        item._calendar_day = None
        item.trade.get_calendar.return_value = []
        with self.assertRaisesRegex(ValueError, "calendar"):
            item.calendar(NOW)


if __name__ == "__main__":
    unittest.main()
