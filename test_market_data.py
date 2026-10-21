"""Regression coverage for price data, session boundaries, and strategy events."""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from trading_bot import Config, completed_bars, regular_hours, signals


class MarketDataTests(unittest.TestCase):
    @staticmethod
    def rising_daily(length=250):
        return pd.DataFrame(
            {"close": [float(n) for n in range(1, length + 1)]},
            index=pd.date_range("2025-01-01", periods=length, freq="B", tz="UTC"),
        )

    @staticmethod
    def hourly(values):
        return pd.DataFrame(
            {"close": values},
            index=pd.date_range("2026-09-01T14:00Z", periods=len(values), freq="h"),
        )

    def entry_frame(self):
        # A sustained decline followed by a sharp rebound crosses both real
        # Wilder RSI and real MACD, rather than replacing those calculations.
        return self.hourly([100.] * 80 + [float(n) for n in range(100, 80, -1)] + [150.])

    def exit_frame(self):
        hourly = self.hourly([100.] * 98 + [120., 80.])
        strength = pd.Series([50.] * 98 + [70., 60.], index=hourly.index)
        return hourly, strength

    def test_rising_daily_trend_permits_real_indicator_entry(self):
        result = signals(self.entry_frame(), self.rising_daily())
        self.assertIsNotNone(result.entry_key)

    def test_entry_identity_survives_next_hour_and_recalculation(self):
        hourly = self.entry_frame()
        first = signals(hourly, self.rising_daily())
        self.assertIsNotNone(first.entry_key)
        extra = pd.DataFrame({"close": [150.]}, index=[hourly.index[-1] + pd.Timedelta(hours=1)])
        later = signals(pd.concat([hourly, extra]), self.rising_daily())
        self.assertEqual(first.entry_key, later.entry_key)
        # A restarted process reconstructs the same identity from source data.
        self.assertEqual(first.entry_key, signals(hourly.copy(), self.rising_daily()).entry_key)

    def test_entry_events_expire_outside_window(self):
        hourly = self.entry_frame()
        extra = pd.DataFrame(
            {"close": [150.] * 7},
            index=pd.date_range(hourly.index[-1] + pd.Timedelta(hours=1), periods=7, freq="h"),
        )
        self.assertIsNone(signals(pd.concat([hourly, extra]), self.rising_daily()).entry_key)

    def test_downtrend_blocks_entry(self):
        daily = self.rising_daily()
        daily["close"] = daily.close.iloc[::-1].to_numpy()
        self.assertIsNone(signals(self.entry_frame(), daily).entry_key)

    def test_short_daily_history_blocks_entry(self):
        self.assertIsNone(signals(self.entry_frame(), self.rising_daily(199)).entry_key)

    def test_stale_daily_session_blocks_entry(self):
        daily = self.rising_daily()
        daily.attrs["current_session_data"] = False
        self.assertIsNone(signals(self.entry_frame(), daily).entry_key)

    def test_insufficient_daily_history_does_not_suppress_hourly_exit(self):
        hourly, strength = self.exit_frame()
        with patch("trading_bot.rsi", return_value=strength):
            result = signals(hourly, self.rising_daily(10))
        self.assertTrue(result.exit_signal)
        self.assertIsNone(result.entry_key)

    def test_stale_daily_session_does_not_suppress_hourly_exit(self):
        hourly, strength = self.exit_frame()
        daily = self.rising_daily()
        daily.attrs["current_session_data"] = False
        with patch("trading_bot.rsi", return_value=strength):
            self.assertTrue(signals(hourly, daily).exit_signal)

    def test_insufficient_hourly_history_is_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "100"):
            signals(self.hourly([100.] * 99), self.rising_daily())

    def test_regular_hours_exclude_partial_open_and_extended_hours(self):
        frame = pd.DataFrame(
            {"close": [100.] * 9}, index=pd.date_range("2026-10-20T12:00Z", periods=9, freq="h")
        )
        calendar = [SimpleNamespace(open=datetime(2026, 10, 20, 9, 30), close=datetime(2026, 10, 20, 16))]
        kept = regular_hours(frame, calendar)
        self.assertEqual(list(kept.index.hour), [14, 15, 16, 17, 18, 19])

    def test_early_close_excludes_interval_crossing_close(self):
        frame = pd.DataFrame(
            {"close": [100.] * 8}, index=pd.date_range("2026-11-27T13:00Z", periods=8, freq="h")
        )
        calendar = [SimpleNamespace(open=datetime(2026, 11, 27, 9, 30), close=datetime(2026, 11, 27, 13))]
        self.assertEqual(list(regular_hours(frame, calendar).index.hour), [15, 16, 17])

    def test_exchange_session_localization_handles_dst_transition(self):
        timestamps = pd.to_datetime([
            "2026-03-06T14:00Z", "2026-03-06T15:00Z", "2026-03-06T20:00Z", "2026-03-06T21:00Z",
            "2026-03-09T13:00Z", "2026-03-09T14:00Z", "2026-03-09T19:00Z", "2026-03-09T20:00Z",
        ])
        frame = pd.DataFrame({"close": [100.] * len(timestamps)}, index=timestamps)
        calendar = [
            SimpleNamespace(open=datetime(2026, 3, 6, 9, 30), close=datetime(2026, 3, 6, 16)),
            SimpleNamespace(open=datetime(2026, 3, 9, 9, 30), close=datetime(2026, 3, 9, 16)),
        ]
        expected = timestamps[[1, 2, 5, 6]]
        pd.testing.assert_index_equal(regular_hours(frame, calendar).index, expected)

    def test_timezone_aware_exchange_session_is_supported(self):
        frame = pd.DataFrame({"close": [100., 100.]}, index=pd.to_datetime(["2026-10-20T14:00Z", "2026-10-20T20:00Z"]))
        calendar = [SimpleNamespace(open=pd.Timestamp("2026-10-20T13:30Z"), close=pd.Timestamp("2026-10-20T20:00Z"))]
        self.assertEqual(list(regular_hours(frame, calendar).index), [frame.index[0]])

    def test_no_regular_session_data_is_explicit_error(self):
        frame = pd.DataFrame({"close": [100.]}, index=pd.to_datetime(["2026-10-20T12:00Z"]))
        with self.assertRaisesRegex(ValueError, "regular-session"):
            regular_hours(frame, [])

    def test_completed_bar_boundary_excludes_partial_and_future(self):
        now = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)
        frame = pd.DataFrame(
            {"close": [100., 101., 102.]},
            index=pd.to_datetime(["2026-10-20T14:00Z", "2026-10-20T14:01Z", "2026-10-20T16:00Z"]),
        )
        result = completed_bars(frame, "TQQQ", now, True)
        self.assertEqual(list(result.index), [frame.index[0]])

    def test_bad_prices_are_rejected(self):
        for price in [float("nan"), float("inf"), 0, -1]:
            with self.subTest(price=price):
                frame = pd.DataFrame({"close": [price]}, index=pd.to_datetime(["2026-10-20T14:00Z"]))
                with self.assertRaisesRegex(ValueError, "invalid"):
                    completed_bars(frame, "TQQQ", datetime(2026, 10, 20, 15, tzinfo=timezone.utc), True)

    def test_nonfinite_configuration_values_are_rejected(self):
        for name in ["allocation", "stop_loss", "take_profit"]:
            for value in [float("nan"), float("inf"), float("-inf")]:
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        Config(**{name: value})

    def test_window_must_be_positive_integer(self):
        for value in [0, -1, 1.5, True]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Config(window=value)


if __name__ == "__main__":
    unittest.main()
