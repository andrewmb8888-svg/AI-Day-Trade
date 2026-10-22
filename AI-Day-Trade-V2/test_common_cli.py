"""Strategy, Windows locking, and read-only CLI tests; no brokerage calls."""
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import ai_day_trade_v2 as app
from common import (BY_SYMBOL, WATCHLIST, Settings, boolean, flatten_orders,
                    instance_lock, money, order_key, rsi, signals)

UTC = timezone.utc


class CommonTests(unittest.TestCase):
    def test_exact_reviewed_watchlist_and_leverage(self):
        self.assertEqual(tuple(BY_SYMBOL), ("NVDA", "AMD", "SOXL", "TSLA", "TSLL", "COIN", "IBIT", "XBI", "LABU",
                                            "OXY", "XOP", "GUSH", "NEM", "GDX", "NUGT", "IWM", "TNA", "JPM", "KRE", "FAS"))
        self.assertEqual(len(WATCHLIST), 20)
        self.assertEqual(BY_SYMBOL['SOXL'].leverage, 3)
        self.assertEqual(BY_SYMBOL['TSLL'].leverage, 2)

    def test_defaults_and_environment_are_cash_only_and_paper(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_environment()
        self.assertTrue(settings.paper)
        self.assertEqual(settings.interval, 10)
        self.assertEqual(settings.allocation, Decimal('.02'))
        self.assertEqual(settings.group_cap, Decimal('.10'))
        self.assertFalse(hasattr(settings, 'allow_margin'))
        with patch.dict(os.environ, {'V2_SYMBOLS': 'NVDA, SOXL', 'V2_CHECK_SECONDS': '15'}, clear=True):
            settings = Settings.from_environment()
        self.assertEqual(settings.symbols, ('NVDA', 'SOXL'))
        self.assertEqual(settings.interval, 15)

    def test_invalid_settings_and_finite_money(self):
        for kwargs in ({'symbols': ()}, {'symbols': ('NVDA', 'NVDA')}, {'symbols': ('UNKNOWN',)},
                       {'interval': 0}, {'interval': float('nan')}, {'paper': 'true'},
                       {'allocation': 'NaN'}, {'cash_buffer': '1'}, {'stop_loss': '-1'},
                       {'take_profit': '0'}, {'window': 1.5}, {'group_cap': '.001'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Settings(**kwargs)
        for value in ('NaN', 'Infinity', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                money(value)
        self.assertFalse(boolean(' False '))
        with self.assertRaises(ValueError):
            boolean('sometimes')

    def test_signal_ids_are_symbol_specific_stable_and_broker_length(self):
        first = order_key('NVDA', 'event:pair')
        self.assertEqual(first, order_key('NVDA', 'event:pair'))
        self.assertNotEqual(first, order_key('AMD', 'event:pair'))
        self.assertNotEqual(first, order_key('NVDA', 'different:pair'))
        self.assertLessEqual(len(first), 48)

    def test_wilder_smoothing_and_flat_series(self):
        result = rsi(pd.Series([10., 11., 10., 12., 11.]), 3)
        self.assertAlmostEqual(result.iloc[3], 75.)
        self.assertAlmostEqual(result.iloc[4], 54.545454545)
        self.assertEqual(rsi(pd.Series([1.] * 30)).iloc[-1], 50)

    def test_signal_pair_requires_current_daily_data_and_exits_do_not(self):
        bars = pd.DataFrame({'close': [100.] * 98 + [90., 120.]},
                            index=pd.date_range('2026-10-01', periods=100, freq='30min', tz='UTC'))
        daily = pd.DataFrame({'close': range(1, 251)})
        strength = pd.Series([50.] * 98 + [20., 40.], index=bars.index)
        with patch('common.rsi', return_value=strength):
            self.assertIsNone(signals(bars, daily).entry_key)
            daily.attrs['current_session_data'] = True
            self.assertIsNotNone(signals(bars, daily).entry_key)
        down = bars.copy()
        down['close'] = [100.] * 98 + [110., 80.]
        strength = pd.Series([50.] * 98 + [80., 60.], index=down.index)
        with patch('common.rsi', return_value=strength):
            self.assertTrue(signals(down, daily.iloc[:10]).exit_signal)
        with self.assertRaises(ValueError):
            signals(bars.iloc[:99], daily)

    def test_nested_orders_deduplicate_children(self):
        child = NS(id='child', legs=[])
        parent = NS(id='parent', legs=[child])
        self.assertEqual(len(flatten_orders([parent, child])), 2)

    def test_lock_excludes_concurrent_local_instances_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'account.lock'
            with instance_lock(path):
                with self.assertRaises(SystemExit):
                    with instance_lock(path):
                        self.fail('Double lock must not succeed')
            with instance_lock(path):
                self.assertTrue(path.exists())

    def test_windows_lock_uses_nonblocking_first_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'account.lock'
            descriptors = []
            def acquire(fd, mode, count):
                self.assertEqual((mode, count), (10, 1))
                self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), 0)
                self.assertGreaterEqual(os.fstat(fd).st_size, 1)
                descriptors.append(fd)
            windows = NS(LK_NBLCK=10, locking=Mock(side_effect=acquire))
            with patch.dict('sys.modules', {'msvcrt': windows}), patch('common.sys.platform', 'win32'):
                with instance_lock(path):
                    windows.locking.assert_called_once()
            with self.assertRaises(OSError):
                os.fstat(descriptors[0])


class CLITests(unittest.TestCase):
    def test_list_requires_no_credentials_and_creates_no_broker(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch.object(app, 'load_dotenv'), \
                patch.object(app, 'Broker') as broker, redirect_stdout(output):
            app.main(['--list'])
        broker.assert_not_called()
        self.assertIn('AI Day Trade V2', output.getvalue())
        self.assertIn('NVDA', output.getvalue())
        self.assertIn('FAS', output.getvalue())

    def test_check_is_read_only_and_keeps_existing_credentials_file(self):
        from test_cash import calendar_for
        now = datetime.now(UTC)
        account = NS(id='test-account', cash='10000', buying_power='40000', non_marginable_buying_power='10000')
        snapshot = NS(account=account, orders=[], positions=[], fills=[], started_at=now)
        bars = pd.DataFrame({'close': [100.] * 100})
        daily = pd.DataFrame({'close': range(250)})
        daily.attrs['current_session_data'] = True
        broker = Mock()
        broker.clock.return_value = NS(is_open=False)
        broker.snapshot.return_value = snapshot
        broker.calendar.return_value = calendar_for(now)
        broker.cash_settings.return_value = False
        broker.quotes.return_value = {'NVDA': NS(bid_price='100', ask_price='100.01')}
        market = Mock()
        market.update.return_value = {'NVDA': (bars, daily)}
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / '.state.json'
            with patch.object(app, 'state_path', return_value=file):
                app.connection_check(broker, market, Settings(symbols=('NVDA',)))
            self.assertFalse(file.exists())
        broker.enforce_cash_only.assert_not_called()
        broker.buy.assert_not_called()
        broker.close.assert_not_called()

    def test_check_main_never_locks_or_starts_trading(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(app, 'load_dotenv'), \
                patch.object(app, 'RotatingFileHandler'), patch.object(app.logging, 'basicConfig'), \
                patch.object(app, 'Broker') as broker, patch.object(app, 'MarketData'), \
                patch.object(app, 'connection_check') as check, patch.object(app, 'instance_lock') as lock:
            app.main(['--check'])
        check.assert_called_once()
        lock.assert_not_called()
        broker.return_value.enforce_cash_only.assert_not_called()

    def test_v1_checkout_lock_blocks_v2_before_account_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / 'AI-Day-Trade-V2'
            root.mkdir()
            (parent / 'trading_bot.py').write_text('# Existing V1 checkout\n')
            with patch.dict(os.environ, {}, clear=True), patch.object(app, 'ROOT', root), \
                    patch.object(app, 'load_dotenv'), patch.object(app, 'RotatingFileHandler'), \
                    patch.object(app.logging, 'basicConfig'), patch.object(app, 'MarketData'), \
                    patch.object(app, 'Broker') as broker, patch.object(app, 'Engine') as engine:
                broker.return_value.account.return_value = NS(id='account-' + parent.name)
                with instance_lock(parent / '.bot-paper.lock'):
                    with self.assertRaises(SystemExit):
                        app.main(['--once'])
                broker.return_value.enforce_cash_only.assert_not_called()
                engine.assert_not_called()


if __name__ == '__main__':
    unittest.main()
