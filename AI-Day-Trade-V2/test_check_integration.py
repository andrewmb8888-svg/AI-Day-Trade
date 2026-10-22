"""Complete read-only check through real SDK models; all HTTP stays offline."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

import ai_day_trade_v2 as app
from broker import Broker
from common import Settings
from market_data import MarketData


class ReadOnlyCheckIntegrationTests(unittest.TestCase):
    def test_all_symbols_use_real_sdk_models_with_modern_account_configuration(self):
        now = datetime(2026, 10, 22, 2, 44, 24, tzinfo=timezone.utc)
        today = pd.Timestamp(now).tz_convert("America/New_York").date()
        settings = Settings()
        calendar = [
            {"date": str(day.date()), "open": "09:30", "close": "16:00"}
            for day in pd.bdate_range(pd.Timestamp(today) - pd.Timedelta(days=110),
                                      pd.Timestamp(today) + pd.Timedelta(days=20))
        ]
        recent_sessions = [pd.Timestamp(row["date"]) for row in calendar
                           if pd.Timestamp(row["date"]).date() <= today][-12:]

        def bar(at):
            return {"t": at.isoformat(), "o": 100., "h": 101., "l": 99.,
                    "c": 100., "v": 10000, "n": 100, "vw": 100.}

        intraday = [bar(day.tz_localize("America/New_York") +
                        pd.Timedelta(hours=9, minutes=30 + 30 * interval))
                    for day in recent_sessions for interval in range(13)]
        daily = [bar(day.tz_localize("America/New_York")) for day in pd.bdate_range(
            end=pd.Timestamp(today) - pd.Timedelta(days=1), periods=230)]
        account = dict(id="00000000-0000-0000-0000-000000000001", account_number="OFFLINE001",
                       status="ACTIVE", currency="USD", cash="1000", buying_power="4000",
                       non_marginable_buying_power="1000", equity="10000", short_market_value="0",
                       pending_transfer_out="0", accrued_fees="0", trading_blocked=False,
                       account_blocked=False, trade_suspended_by_user=False)
        # Real responses can omit these two legacy fields even though the SDK's
        # AccountConfiguration model still requires both. The guard starts off
        # so this also catches any attempt to configure an account during --check.
        configuration = dict(fractional_trading=True, max_margin_multiplier="4", no_shorting=False,
                             suspend_trade=False, trade_confirm_email="all",
                             ptp_no_exception_entry=True, max_options_trading_level=0,
                             closing_transactions_only=False)
        self.assertNotIn("dtbp_check", configuration)
        self.assertNotIn("pdt_check", configuration)
        calls = []

        def send(adapter, request, **kwargs):
            parsed = urlparse(request.url)
            path, query = parsed.path, parse_qs(parsed.query)
            calls.append((request.method, path))
            self.assertEqual(request.method, "GET", "Read-only check attempted a write")
            self.assertEqual(parsed.scheme, "https")
            self.assertIn(parsed.netloc, {"paper-api.alpaca.markets", "data.alpaca.markets"})
            self.assertEqual(kwargs["timeout"], (10, 30))
            if path == "/v2/clock":
                payload = {"timestamp": now.isoformat(), "is_open": False,
                           "next_open": "2026-10-22T13:30:00Z", "next_close": "2026-10-22T20:00:00Z"}
            elif path == "/v2/account":
                payload = account
            elif path in {"/v2/account/activities/FILL", "/v2/orders", "/v2/positions"}:
                payload = []
            elif path == "/v2/calendar":
                payload = calendar
            elif path == "/v2/account/configurations":
                payload = configuration
            elif path == "/v2/stocks/bars":
                self.assertEqual(query["feed"], ["iex"])
                self.assertEqual(query["adjustment"], ["split"])
                self.assertIn(query["timeframe"], [["1Day"], ["30Min"]])
                self.assertEqual(set(query["symbols"][0].split(",")), set(settings.symbols))
                rows = daily if query["timeframe"] == ["1Day"] else intraday
                payload = {"bars": {symbol: rows for symbol in settings.symbols}, "next_page_token": None}
            elif path == "/v2/stocks/quotes/latest":
                self.assertEqual(query["feed"], ["iex"])
                self.assertEqual(set(query["symbols"][0].split(",")), set(settings.symbols))
                payload = {"quotes": {symbol: {"t": now.isoformat(), "ap": 100.01, "bp": 100.,
                                               "as": 100, "bs": 100} for symbol in settings.symbols}}
            elif path.startswith("/v2/assets/"):
                symbol = path.rsplit("/", 1)[-1]
                self.assertIn(symbol, settings.symbols)
                payload = {"id": "00000000-0000-0000-0000-000000000002", "class": "us_equity",
                           "exchange": "NASDAQ", "symbol": symbol, "status": "active", "tradable": True,
                           "marginable": True, "shortable": True, "easy_to_borrow": True, "fractionable": True}
            else:
                self.fail("Unexpected HTTP path: " + path)
            response = requests.Response()
            response.status_code = 200
            response._content = json.dumps(payload).encode()
            return response

        with patch.dict(os.environ, {"ALPACA_PAPER_API_KEY": "offline-key",
                                     "ALPACA_PAPER_SECRET_KEY": "offline-secret"}, clear=True):
            broker = Broker(settings)
        self.addCleanup(broker.trade._session.close)
        self.addCleanup(broker.data._session.close)
        market = MarketData(broker.data, settings)
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            with patch.object(app, "datetime") as clock, patch("broker._utc_now", return_value=now), \
                    patch.object(app, "state_path", return_value=state_path), \
                    patch.object(HTTPAdapter, "send", new=send), \
                    self.assertLogs(app.LOG, level="INFO") as logged:
                clock.now.return_value = now
                app.connection_check(broker, market, settings)
            self.assertFalse(state_path.exists())
        messages = "\n".join(logged.output)
        self.assertIn("READ-ONLY check: 20/20 symbols passed", messages)
        self.assertIn("cash-only guard=False", messages)
        self.assertIn("--check changed nothing", messages)
        self.assertEqual(len(calls), 31)
        self.assertTrue(all(method == "GET" for method, _ in calls))
        paths = Counter(path for _, path in calls)
        self.assertEqual(paths["/v2/account"], 2)
        self.assertEqual(paths["/v2/account/configurations"], 1)
        self.assertEqual(paths["/v2/calendar"], 1)
        self.assertEqual(paths["/v2/stocks/bars"], 2)
        self.assertEqual(paths["/v2/stocks/quotes/latest"], 1)
        self.assertEqual(sum(count for path, count in paths.items() if path.startswith("/v2/assets/")), 20)


if __name__ == "__main__":
    unittest.main()
