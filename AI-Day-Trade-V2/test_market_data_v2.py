import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pandas as pd

from common import Settings
from market_data import MarketData, expected_bar, regular30

UTC = timezone.utc


def session(day, opening="09:30", closing="16:00"):
    day = date.fromisoformat(day)
    return NS(date=day, open=datetime.fromisoformat(f"{day}T{opening}:00"),
              close=datetime.fromisoformat(f"{day}T{closing}:00"))


def prices(times, values=None):
    return pd.DataFrame({"close": values or [100.] * len(times)},
                        index=pd.DatetimeIndex(pd.to_datetime(times, utc=True), name="timestamp"))


def combined(frames, symbols):
    existing = {s: frames[s] for s in symbols if s in frames and not frames[s].empty}
    if existing:
        return pd.concat(existing, names=["symbol", "timestamp"])
    return pd.DataFrame({"close": []}, index=pd.MultiIndex.from_arrays([[], []], names=["symbol", "timestamp"]))


class FakeData:
    def __init__(self, symbols):
        self.calls = []
        self.intraday = {s: prices(["2026-10-19T19:30Z", "2026-10-20T13:30Z", "2026-10-20T14:00Z"])
                         for s in symbols}
        self.daily = {s: prices(["2026-10-16T04:00Z", "2026-10-19T04:00Z", "2026-10-20T04:00Z"])
                      for s in symbols}
        self.failure = None

    def get_stock_bars(self, request):
        self.calls.append(request)
        if self.failure:
            raise self.failure
        source = self.daily if str(request.timeframe) == "1Day" else self.intraday
        return NS(df=combined(source, request.symbol_or_symbols))


class SessionBoundaryTests(unittest.TestCase):
    def test_normal_open_and_publication_delay(self):
        calendar = [session("2026-10-20")]
        self.assertIsNone(expected_bar(pd.Timestamp("2026-10-20T13:59:59Z"), calendar))
        self.assertIsNone(expected_bar(pd.Timestamp("2026-10-20T14:00:02Z"), calendar))
        self.assertEqual(expected_bar(pd.Timestamp("2026-10-20T14:00:03Z"), calendar),
                         pd.Timestamp("2026-10-20T13:30Z"))
        self.assertEqual(expected_bar(pd.Timestamp("2026-10-20T15:03Z"), calendar),
                         pd.Timestamp("2026-10-20T14:30Z"))

    def test_early_close_and_after_close_cap(self):
        calendar = [session("2026-11-27", closing="13:00")]
        self.assertEqual(expected_bar(pd.Timestamp("2026-11-27T21:00Z"), calendar),
                         pd.Timestamp("2026-11-27T17:30Z"))
        raw = prices(["2026-11-27T17:00Z", "2026-11-27T17:30Z", "2026-11-27T18:00Z"])
        actual = regular30(raw, pd.Timestamp("2026-11-27T21:00Z"), calendar)
        self.assertEqual(list(actual.index), list(raw.index[:2]))

    def test_dst_and_holiday(self):
        calendar = [session("2026-10-30"), session("2026-11-02")]
        self.assertEqual(expected_bar(pd.Timestamp("2026-10-30T14:00:03Z"), calendar),
                         pd.Timestamp("2026-10-30T13:30Z"))
        self.assertEqual(expected_bar(pd.Timestamp("2026-11-02T15:00:03Z"), calendar),
                         pd.Timestamp("2026-11-02T14:30Z"))
        self.assertIsNone(expected_bar(pd.Timestamp("2026-11-01T18:00Z"), calendar))

    def test_filter_excludes_unaligned_partial_and_extended_intervals(self):
        raw = prices(["2026-10-20T13:00Z", "2026-10-20T13:30Z", "2026-10-20T13:45Z",
                      "2026-10-20T14:00Z", "2026-10-20T20:00Z"])
        actual = regular30(raw, pd.Timestamp("2026-10-20T14:00:03Z"), [session("2026-10-20")])
        self.assertEqual(list(actual.index), [pd.Timestamp("2026-10-20T13:30Z")])

    def test_timezone_aware_calendar(self):
        entry = session("2026-10-20")
        entry.open = pd.Timestamp("2026-10-20T13:30Z")
        entry.close = pd.Timestamp("2026-10-20T20:00Z")
        self.assertEqual(expected_bar(pd.Timestamp("2026-10-20T14:00:03Z"), [entry]), entry.open)

    def test_naive_now_rejected(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            expected_bar(datetime(2026, 10, 20, 14), [session("2026-10-20")])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(symbols=("NVDA", "AMD"))
        self.fake = FakeData(self.settings.symbols)
        self.data = MarketData(self.fake, self.settings)
        self.now = pd.Timestamp("2026-10-20T14:00:03Z")
        self.calendar = [session("2026-10-16"), session("2026-10-19"), session("2026-10-20")]

    def test_initial_requests_batch_every_symbol_without_limit(self):
        result = self.data.update(self.now, self.calendar)
        self.assertEqual(set(result), set(self.settings.symbols))
        self.assertEqual(len(self.fake.calls), 2)
        for request in self.fake.calls:
            self.assertEqual(request.symbol_or_symbols, list(self.settings.symbols))
            self.assertIsNone(request.limit)
            self.assertEqual(request.feed.value, "iex")
            self.assertEqual(request.adjustment.value, "split")
        self.assertEqual(str(self.fake.calls[0].timeframe), "30Min")
        self.assertEqual(self.now.to_pydatetime().replace(tzinfo=None) - self.fake.calls[0].start, timedelta(days=60))
        self.assertEqual(self.now.to_pydatetime().replace(tzinfo=None) - self.fake.calls[1].start, timedelta(days=500))
        for bars, daily in result.values():
            self.assertEqual(bars.index[-1], pd.Timestamp("2026-10-20T13:30Z"))
            self.assertEqual(daily.index[-1], pd.Timestamp("2026-10-19T04:00Z"))
            self.assertTrue(daily.attrs["current_session_data"])

    def test_same_bar_checks_do_not_refetch_histories(self):
        self.data.update(self.now, self.calendar)
        self.data.update(self.now + pd.Timedelta(seconds=10), self.calendar)
        self.data.update(self.now + pd.Timedelta(minutes=20), self.calendar)
        self.assertEqual(len(self.fake.calls), 2)

    def test_next_bar_refreshes_incrementally_and_replaces_partial(self):
        self.data.update(self.now, self.calendar)
        for frame in self.fake.intraday.values():
            frame.loc[pd.Timestamp("2026-10-20T14:00Z"), "close"] = 123.
        result = self.data.update(self.now + pd.Timedelta(minutes=30), self.calendar)
        self.assertEqual(len(self.fake.calls), 3)
        self.assertEqual(self.fake.calls[-1].end - self.fake.calls[-1].start, timedelta(days=1))
        self.assertEqual(result["NVDA"][0].close.iloc[-1], 123.)
        self.assertTrue(result["NVDA"][0].index.is_unique)

    def test_missing_symbol_retries_only_that_symbol_without_starvation(self):
        del self.fake.intraday["AMD"]
        first = self.data.update(self.now, self.calendar)
        self.assertEqual(set(first), {"NVDA"})
        self.assertIn("AMD", self.data.errors)
        self.fake.intraday["AMD"] = prices(["2026-10-20T13:30Z"])
        result = self.data.update(self.now + pd.Timedelta(seconds=10), self.calendar)
        self.assertEqual(set(result), {"NVDA", "AMD"})
        self.assertEqual(self.fake.calls[-1].symbol_or_symbols, ["AMD"])
        # A missing initialization response retries full history, not one day.
        self.assertEqual(self.fake.calls[-1].end - self.fake.calls[-1].start, timedelta(days=60))

    def test_cached_partial_is_not_used_if_completed_refetch_missing(self):
        self.data.update(self.now, self.calendar)
        self.fake.intraday["AMD"] = prices(["2026-10-20T13:30Z"])
        result = self.data.update(self.now + pd.Timedelta(minutes=30), self.calendar)
        self.assertIn("NVDA", result)
        self.assertNotIn("AMD", result)
        self.assertIn("AMD", self.data.errors)

    def test_bad_symbol_prices_do_not_block_healthy_symbol(self):
        self.fake.intraday["AMD"].iloc[0, 0] = float("nan")
        result = self.data.update(self.now, self.calendar)
        self.assertEqual(set(result), {"NVDA"})
        self.assertIn("invalid", self.data.errors["AMD"])

    def test_stale_daily_disables_entry_but_returns_intraday_for_exits(self):
        self.fake.daily["AMD"] = prices(["2026-10-16T04:00Z"])
        result = self.data.update(self.now, self.calendar)
        self.assertIn("AMD", result)
        self.assertFalse(result["AMD"][1].attrs["current_session_data"])
        self.assertTrue(result["NVDA"][1].attrs["current_session_data"])
        self.fake.daily["AMD"] = prices(["2026-10-19T04:00Z"])
        result = self.data.update(self.now + pd.Timedelta(seconds=10), self.calendar)
        self.assertTrue(result["AMD"][1].attrs["current_session_data"])
        self.assertEqual(self.fake.calls[-1].symbol_or_symbols, ["AMD"])
        self.assertEqual(str(self.fake.calls[-1].timeframe), "1Day")

    def test_empty_daily_does_not_suppress_intraday(self):
        del self.fake.daily["AMD"]
        result = self.data.update(self.now, self.calendar)
        self.assertIn("AMD", result)
        self.assertTrue(result["AMD"][1].empty)
        self.assertFalse(result["AMD"][1].attrs["current_session_data"])
        self.fake.daily["AMD"] = prices(["2026-10-19T04:00Z"])
        self.data.update(self.now + pd.Timedelta(seconds=10), self.calendar)
        self.assertEqual(self.fake.calls[-1].end - self.fake.calls[-1].start, timedelta(days=500))

    def test_next_session_refreshes_daily_and_cannot_use_cached_partial_daily(self):
        self.data.update(self.now, self.calendar)
        next_now = pd.Timestamp("2026-10-21T14:00:03Z")
        calendar = self.calendar + [session("2026-10-21")]
        for symbol in self.settings.symbols:
            self.fake.intraday[symbol] = prices(["2026-10-21T13:30Z"])
            # A stale response fails to supply yesterday, despite that row being
            # present as a partial daily bar in the first API response.
            self.fake.daily[symbol] = prices(["2026-10-19T04:00Z"])
        result = self.data.update(next_now, calendar)
        for bars, daily in result.values():
            self.assertFalse(daily.attrs["current_session_data"])
            self.assertEqual(daily.index[-1], pd.Timestamp("2026-10-19T04:00Z"))
        self.assertEqual(self.fake.calls[-1].end - self.fake.calls[-1].start, timedelta(days=500))

    def test_overnight_split_replaces_entire_intraday_and_daily_prefix(self):
        self.fake.intraday["NVDA"] = prices([
            "2026-10-16T19:30Z", "2026-10-19T19:30Z", "2026-10-20T13:30Z"])
        self.data.update(self.now, self.calendar)
        next_now = pd.Timestamp("2026-10-21T14:00:03Z")
        calendar = self.calendar + [session("2026-10-21")]
        # Full histories now reflect an overnight 10-for-1 split. A row absent
        # from the new full result must not survive from the previous basis.
        self.fake.intraday["NVDA"] = prices([
            "2026-10-16T19:30Z", "2026-10-21T13:30Z"], [10., 10.])
        self.fake.daily["NVDA"] = prices([
            "2026-10-16T04:00Z", "2026-10-20T04:00Z"], [10., 10.])
        self.fake.intraday["AMD"] = prices(["2026-10-21T13:30Z"])
        self.fake.daily["AMD"] = prices(["2026-10-20T04:00Z"])
        result = self.data.update(next_now, calendar)
        bars, daily = result["NVDA"]
        self.assertEqual(list(bars.close), [10., 10.])
        self.assertEqual(list(daily.close), [10., 10.])
        self.assertNotIn(pd.Timestamp("2026-10-19T19:30Z"), bars.index)
        self.assertNotIn(pd.Timestamp("2026-10-19T04:00Z"), daily.index)
        self.assertEqual(self.fake.calls[-2].end - self.fake.calls[-2].start, timedelta(days=60))
        self.assertEqual(self.fake.calls[-1].end - self.fake.calls[-1].start, timedelta(days=500))
        # Within the same session, the next bar uses only a one-day overlap.
        self.fake.intraday["NVDA"] = prices(["2026-10-21T14:00Z"], [11.])
        self.fake.intraday["AMD"] = prices(["2026-10-21T14:00Z"])
        self.data.update(next_now + pd.Timedelta(minutes=30), calendar)
        self.assertEqual(self.fake.calls[-1].end - self.fake.calls[-1].start, timedelta(days=1))

    def test_new_session_missing_full_history_cannot_reuse_previous_basis(self):
        self.data.update(self.now, self.calendar)
        next_now = pd.Timestamp("2026-10-21T14:00:03Z")
        calendar = self.calendar + [session("2026-10-21")]
        del self.fake.intraday["NVDA"]
        self.fake.intraday["AMD"] = prices(["2026-10-21T13:30Z"])
        for symbol in self.settings.symbols:
            self.fake.daily[symbol] = prices(["2026-10-20T04:00Z"])
        result = self.data.update(next_now, calendar)
        self.assertEqual(set(result), {"AMD"})
        self.assertNotIn("NVDA", self.data._intraday_day)
        self.assertTrue(self.data._intraday["NVDA"].empty)
        self.fake.intraday["NVDA"] = prices(["2026-10-21T13:30Z"], [10.])
        result = self.data.update(next_now + pd.Timedelta(seconds=10), calendar)
        self.assertEqual(result["NVDA"][0].close.iloc[-1], 10.)
        self.assertEqual(self.fake.calls[-1].symbol_or_symbols, ["NVDA"])
        self.assertEqual(self.fake.calls[-1].end - self.fake.calls[-1].start, timedelta(days=60))

    def test_new_session_empty_daily_discards_old_prefix(self):
        self.data.update(self.now, self.calendar)
        next_now = pd.Timestamp("2026-10-21T14:00:03Z")
        calendar = self.calendar + [session("2026-10-21")]
        for symbol in self.settings.symbols:
            self.fake.intraday[symbol] = prices(["2026-10-21T13:30Z"])
            self.fake.daily[symbol] = prices(["2026-10-20T04:00Z"])
        del self.fake.daily["NVDA"]
        result = self.data.update(next_now, calendar)
        self.assertTrue(result["NVDA"][1].empty)
        self.assertFalse(result["NVDA"][1].attrs["current_session_data"])


    def test_no_historical_requests_before_first_completed_bar(self):
        self.assertEqual(self.data.update(pd.Timestamp("2026-10-20T13:45Z"), self.calendar), {})
        self.assertEqual(self.fake.calls, [])

    def test_force_history_allows_read_only_checks_before_first_bar(self):
        now = pd.Timestamp("2026-10-20T13:45Z")
        result = self.data.update(now, self.calendar, force_history=True)
        self.assertEqual(set(result), set(self.settings.symbols))
        self.assertEqual(result["NVDA"][0].index[-1], pd.Timestamp("2026-10-19T19:30Z"))
        self.assertEqual(len(self.fake.calls), 2)
        self.data.update(now + pd.Timedelta(seconds=10), self.calendar, force_history=True)
        self.assertEqual(len(self.fake.calls), 2)
        self.data.update(self.now, self.calendar)
        self.assertEqual(len(self.fake.calls), 3)  # First real completed bar refresh.

    def test_force_history_works_on_closed_nontrading_day(self):
        now = pd.Timestamp("2026-10-18T18:00Z")
        for symbol in self.settings.symbols:
            self.fake.intraday[symbol] = prices(["2026-10-16T19:30Z"])
            self.fake.daily[symbol] = prices(["2026-10-16T04:00Z"])
        result = self.data.update(now, self.calendar, force_history=True)
        self.assertEqual(set(result), set(self.settings.symbols))
        self.assertTrue(result["NVDA"][1].attrs["current_session_data"])

    def test_full_twenty_symbol_watchlist_uses_two_initial_batch_requests(self):
        settings = Settings()
        fake = FakeData(settings.symbols)
        data = MarketData(fake, settings)
        result = data.update(self.now, self.calendar)
        self.assertEqual(len(result), 20)
        self.assertEqual(len(fake.calls), 2)
        for request in fake.calls:
            self.assertEqual(set(request.symbol_or_symbols), set(settings.symbols))

    def test_raw_cache_is_bounded_and_returned_frames_are_independent(self):
        times = pd.date_range("2026-08-01", periods=4000, freq="30min", tz="UTC")
        history = pd.DataFrame({"close": [99.] * len(times)}, index=times)
        current = prices(["2026-10-20T13:30Z"])
        self.fake.intraday["NVDA"] = pd.concat([history, current])
        first = self.data.update(self.now, self.calendar)
        self.assertLessEqual(len(self.data._intraday["NVDA"]), 3000)
        first["NVDA"][0].loc[pd.Timestamp("2026-10-20T13:30Z"), "close"] = 999.
        second = self.data.update(self.now + pd.Timedelta(seconds=10), self.calendar)
        self.assertEqual(second["NVDA"][0].close.iloc[-1], 100.)

    def test_api_exception_propagates_for_cycle_retry(self):
        self.fake.failure = TimeoutError("network timeout")
        with self.assertRaises(TimeoutError):
            self.data.update(self.now, self.calendar)
        self.fake.failure = None
        result = self.data.update(self.now, self.calendar)
        self.assertEqual(set(result), set(self.settings.symbols))

    def test_index_sort_dedup_and_no_future_output(self):
        self.fake.intraday["NVDA"] = prices(["2026-10-20T13:30Z", "2026-10-20T14:00Z",
                                             "2026-10-20T13:30Z"], [100., 999., 101.])
        result = self.data.update(self.now, self.calendar)
        self.assertEqual(len(result["NVDA"][0]), 1)
        self.assertEqual(result["NVDA"][0].close.iloc[-1], 101.)


if __name__ == "__main__":
    unittest.main()
