"""Offline contracts against the installed Alpaca SDK; no network or orders."""
import json
import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pandas as pd
import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import AssetClass, OrderClass, OrderSide
from requests.adapters import HTTPAdapter

from trading_bot import AlpacaBroker, Config, TimeoutAdapter


PAPER_ENV = {
    "ALPACA_PAPER_API_KEY": "offline-test-key",
    "ALPACA_PAPER_SECRET_KEY": "offline-test-secret",
}


def api_error(status):
    return APIError('{"message":"offline test"}',
                    requests.HTTPError(response=NS(status_code=status)))


def response(payload):
    result = requests.Response()
    result.status_code = 200
    result._content = json.dumps(payload).encode()
    result.headers["Content-Type"] = "application/json"
    return result


class TimeoutTests(unittest.TestCase):
    def test_timeout_defaults_and_explicit_values(self):
        request = requests.Request("GET", "https://example.test/").prepare()
        adapter = TimeoutAdapter()
        with patch.object(HTTPAdapter, "send", return_value=Mock()) as send:
            adapter.send(request)
            self.assertEqual(send.call_args.kwargs["timeout"], (10, 30))
            adapter.send(request, timeout=None)
            self.assertEqual(send.call_args.kwargs["timeout"], (10, 30))
            adapter.send(request, timeout=(2, 4))
            self.assertEqual(send.call_args.kwargs["timeout"], (2, 4))

    def test_real_sdk_clients_use_adapter_with_timeout(self):
        with patch.dict(os.environ, PAPER_ENV, clear=True):
            broker = AlpacaBroker(Config())
        for client in (broker.trade, broker.data):
            self.assertIsInstance(client._session.get_adapter("https://example.test/"), TimeoutAdapter)
        payloads = [
            {"timestamp": "2026-10-20T15:05:00Z", "is_open": True,
             "next_open": "2026-10-21T13:30:00Z", "next_close": "2026-10-20T20:00:00Z"},
            {"quotes": {"TQQQ": {"t": "2026-10-20T15:05:00Z", "ap": 100.0,
                                   "as": 10, "bp": 99.99, "bs": 10}}},
        ]
        # Intercept the base transport. SDK routing, headers and model parsing
        # run normally, but no socket or broker endpoint is contacted.
        with patch.object(HTTPAdapter, "send", side_effect=[response(p) for p in payloads]) as send:
            self.assertTrue(broker.clock().is_open)
            self.assertEqual(broker.quote().ask_price, 100.0)
        self.assertEqual(send.call_count, 2)
        for call in send.call_args_list:
            self.assertEqual(call.kwargs["timeout"], (10, 30))
        self.assertIn("paper-api.alpaca.markets/v2/clock", send.call_args_list[0].args[0].url)
        self.assertIn("data.alpaca.markets/v2/stocks/quotes/latest", send.call_args_list[1].args[0].url)


class CredentialTests(unittest.TestCase):
    def test_missing_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "credentials"):
                AlpacaBroker(Config())

    def test_default_paper_endpoint(self):
        with patch.dict(os.environ, PAPER_ENV, clear=True):
            broker = AlpacaBroker(Config())
        self.assertIn("paper-api.alpaca.markets", broker.trade._base_url.value)

    def test_live_requires_explicit_acknowledgment(self):
        live = {"ALPACA_API_KEY": "offline-key", "ALPACA_SECRET_KEY": "offline-secret"}
        with patch.dict(os.environ, live, clear=True):
            with self.assertRaisesRegex(ValueError, "ALLOW_LIVE_TRADING"):
                AlpacaBroker(Config(paper=False))
        live["ALLOW_LIVE_TRADING"] = "I_UNDERSTAND"
        with patch.dict(os.environ, live, clear=True):
            broker = AlpacaBroker(Config(paper=False))
        self.assertIn("api.alpaca.markets", broker.trade._base_url.value)
        self.assertNotIn("paper-api", broker.trade._base_url.value)

    def test_custom_endpoint_is_rejected(self):
        with patch.dict(os.environ, {**PAPER_ENV, "TRADE_API_URL": "https://example.test/"}, clear=True):
            with self.assertRaisesRegex(ValueError, "overrides"):
                AlpacaBroker(Config())


class BrokerRequestTests(unittest.TestCase):
    def setUp(self):
        self.broker = object.__new__(AlpacaBroker)
        self.broker.config = Config()
        self.broker.trade = Mock()

    def test_bracket_uses_real_sdk_request(self):
        self.broker.buy(2, "offline-entry", 100.1, 97.1, 106.11)
        request = self.broker.trade.submit_order.call_args.kwargs["order_data"]
        self.assertEqual(request.order_class, OrderClass.BRACKET)
        self.assertEqual(request.side, OrderSide.BUY)
        self.assertEqual(request.client_order_id, "offline-entry")
        self.assertEqual(request.stop_loss.stop_price, 97.1)
        self.assertEqual(request.take_profit.limit_price, 106.11)

    def test_close_uses_position_endpoint(self):
        self.assertIs(self.broker.close(), self.broker.trade.close_position.return_value)
        self.broker.trade.close_position.assert_called_once_with("TQQQ")
        self.broker.trade.submit_order.assert_not_called()

    def test_close_only_swallows_not_found(self):
        self.broker.trade.close_position.side_effect = api_error(404)
        self.assertIsNone(self.broker.close())
        for status in (401, 403, 429, 500):
            with self.subTest(status=status):
                self.broker.trade.close_position.side_effect = api_error(status)
                with self.assertRaises(APIError):
                    self.broker.close()
        self.broker.trade.close_position.side_effect = requests.Timeout("offline")
        with self.assertRaises(requests.Timeout):
            self.broker.close()


class ReadOnlyCheckTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 20, 15, 5, tzinfo=timezone.utc)
        self.broker = object.__new__(AlpacaBroker)
        self.broker.config = Config()
        self.broker.trade = Mock()
        self.broker.trade.get_asset.return_value = NS(tradable=True, asset_class=AssetClass.US_EQUITY)
        self.broker.clock = Mock(return_value=NS(is_open=True))
        self.broker.account = Mock(return_value=NS(account_blocked=False, trading_blocked=False))
        hourly = pd.DataFrame({"close": [100.] * 100}, index=pd.date_range("2026-10-01", periods=100, freq="h", tz="UTC"))
        daily = pd.DataFrame({"close": [100.] * 250}, index=pd.date_range("2025-10-01", periods=250, tz="UTC"))
        daily.attrs["current_session_data"] = True
        self.broker.bars = Mock(return_value=(hourly, daily))
        self.broker.position = Mock(return_value=None)
        self.broker.orders = Mock(return_value=[])
        self.broker.quote = Mock(return_value=NS(ask_price=100., bid_price=99.99, timestamp=self.now))

    def test_check_is_read_only(self):
        self.broker.check(self.now)
        for method in (self.broker.clock, self.broker.account, self.broker.position,
                       self.broker.orders, self.broker.quote):
            method.assert_called_once_with()
        self.broker.bars.assert_called_once_with(self.now)
        self.broker.trade.get_asset.assert_called_once_with("TQQQ")
        for method in (self.broker.trade.submit_order, self.broker.trade.close_position,
                       self.broker.trade.cancel_order_by_id):
            method.assert_not_called()

    def test_invalid_asset_rejected_before_data_or_orders(self):
        for tradable, asset_class in ((False, AssetClass.US_EQUITY), (True, AssetClass.CRYPTO)):
            with self.subTest(tradable=tradable, asset_class=asset_class):
                self.broker.trade.get_asset.return_value = NS(tradable=tradable, asset_class=asset_class)
                with self.assertRaisesRegex(ValueError, "tradable US equity"):
                    self.broker.check(self.now)
        self.broker.bars.assert_not_called()
        self.broker.trade.submit_order.assert_not_called()

    def test_blocked_account_rejected(self):
        self.broker.account.return_value.trading_blocked = True
        with self.assertRaisesRegex(ValueError, "blocked"):
            self.broker.check(self.now)
        self.broker.bars.assert_not_called()


if __name__ == "__main__":
    unittest.main()
