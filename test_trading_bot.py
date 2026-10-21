import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace as NS
from datetime import datetime, timezone
import pandas as pd
from trading_bot import Config, Engine, Decision, boolean, completed_bars, rsi, signals


class StrategyTests(unittest.TestCase):
    def test_boolean(self):
        self.assertFalse(boolean('False'))
        self.assertTrue(boolean('true'))
        with self.assertRaises(ValueError):
            boolean('maybe')

    def test_wilder_seed_and_smoothing(self):
        prices = pd.Series([10, 11, 10, 12, 11], dtype=float)
        values = rsi(prices, 3)
        self.assertAlmostEqual(values.iloc[3], 75)
        self.assertAlmostEqual(values.iloc[4], 54.54545454545)
        self.assertEqual(rsi(pd.Series([1.] * 20)).iloc[-1], 50)

    def test_partial_bars_excluded(self):
        now = datetime(2026, 10, 20, 15, 5, tzinfo=timezone.utc)
        frame = pd.DataFrame({'close': [10., 11.]}, index=pd.to_datetime(['2026-10-20T14:00Z', '2026-10-20T15:00Z']))
        self.assertEqual(len(completed_bars(frame, 'TQQQ', now, True)), 1)
        daily = pd.DataFrame({'close': [10., 11.]}, index=pd.to_datetime(['2026-10-19T04:00Z', '2026-10-20T04:00Z']))
        self.assertEqual(len(completed_bars(daily, 'TQQQ', now, False)), 1)

    def test_trend_warmup_does_not_block_valid_entry(self):
        daily = pd.DataFrame({'close': range(1, 251)})
        hourly = pd.DataFrame({'close': [100.] * 100})
        strength = pd.Series([50.] * 98 + [20., 40.])
        # Supply a price path whose MACD crosses upward on the last bar.
        hourly['close'] = [100.] * 98 + [90., 120.]
        with patch('trading_bot.rsi', return_value=strength):
            self.assertTrue(signals(hourly, daily).entry_key)

    def test_old_signals_expire(self):
        daily = pd.DataFrame({'close': range(1, 251)})
        hourly = pd.DataFrame({'close': [100.] * 90 + [90., 120.] + [120.] * 10})
        strength = pd.Series([50.] * 90 + [20., 40.] + [50.] * 10)
        with patch('trading_bot.rsi', return_value=strength):
            self.assertFalse(signals(hourly, daily).entry_key)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 20, 15, 5, tzinfo=timezone.utc)
        self.b = Mock()
        self.b.clock.return_value = NS(is_open=True)
        frame = pd.DataFrame({'close': [100.]}, index=pd.to_datetime(['2026-10-20T14:00Z']))
        self.b.bars.return_value = frame, frame
        self.b.position.return_value = None
        self.b.orders.return_value = []
        self.b.already_submitted.return_value = False
        self.b.account.return_value = NS(equity='10000', buying_power='20000', trading_blocked=False, account_blocked=False)
        self.b.quote.return_value = NS(timestamp=self.now, ask_price=100., bid_price=99.99)
        self.engine = Engine(self.b, Config())
        self.signal = patch('trading_bot.signals', return_value=Decision("bounce:cross", False))
        self.signal.start()
        self.addCleanup(self.signal.stop)

    def test_buy_budget_bracket_and_duplicate_bar(self):
        self.assertEqual(self.engine.step(self.now), 'buy submitted')
        args = self.b.buy.call_args.args
        self.assertEqual(args[0], 1)  # 2% equity, with limit-price buffer
        self.assertLess(args[3], args[2])
        self.assertGreater(args[4], args[2])
        self.assertEqual(self.engine.step(self.now), 'bar already processed')
        self.b.buy.assert_called_once()

    def test_restart_reconciles_order_id(self):
        self.b.already_submitted.return_value = True
        self.assertEqual(self.engine.step(self.now), 'order already submitted')
        self.b.buy.assert_not_called()

    def test_pending_order_prevents_buy(self):
        self.b.orders.return_value = [NS(id='pending')]
        self.assertEqual(self.engine.step(self.now), 'pending orders; entry skipped')
        self.b.buy.assert_not_called()

    def test_position_error_fails_closed(self):
        self.b.position.side_effect = RuntimeError('network failure')
        with self.assertRaises(RuntimeError):
            self.engine.step(self.now)
        self.b.buy.assert_not_called()

    def test_closed_market(self):
        self.b.clock.return_value.is_open = False
        self.assertEqual(self.engine.step(self.now), 'market closed')
        self.b.bars.assert_not_called()

    def test_stale_quote(self):
        self.b.quote.return_value.timestamp = self.now.replace(hour=14)
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.engine.step(self.now)
        self.b.buy.assert_not_called()

    def test_stale_bars(self):
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.engine.step(self.now.replace(hour=17))
        self.b.buy.assert_not_called()

    def test_protective_orders_retained(self):
        self.b.position.return_value = NS(qty='2')
        self.b.orders.return_value = [NS(id='stop')]
        with patch('trading_bot.signals', return_value=Decision(exit_signal=True)):
            self.assertIn('retained', self.engine.step(self.now))
        self.b.close.assert_not_called()

    def test_fractional_existing_position_exit(self):
        self.b.position.return_value = NS(qty='1.5')
        with patch('trading_bot.signals', return_value=Decision(exit_signal=True)):
            self.assertEqual(self.engine.step(self.now), 'close submitted')
        self.b.close.assert_called_once_with()

    def test_submission_timeout_reconciles_same_id(self):
        self.b.buy.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            self.engine.step(self.now)
        first_id = self.b.buy.call_args.args[1]
        self.b.already_submitted.return_value = True
        self.assertEqual(self.engine.step(self.now), 'order already submitted')
        self.assertEqual(self.b.already_submitted.call_args.args[0], first_id)
        self.b.buy.assert_called_once()


    def test_entry_signal_consumed_across_bars_and_restart(self):
        self.assertEqual(self.engine.step(self.now), 'buy submitted')
        used_id = self.b.buy.call_args.args[1]
        # A broker exit has closed the position, but the old pair is still recent.
        self.b.already_submitted.side_effect = lambda client_id: client_id == used_id
        frame, daily = self.b.bars.return_value
        next_frame = frame.copy()
        next_frame.index = next_frame.index + pd.Timedelta(hours=1)
        self.b.bars.return_value = next_frame, daily
        self.now = self.now.replace(hour=16)
        # Restart means no in-memory processed bar remains.
        restarted = Engine(self.b, Config())
        self.assertEqual(restarted.step(self.now), 'order already submitted')
        self.b.buy.assert_called_once()

    def test_orders_read_before_position(self):
        self.engine.step(self.now)
        names = [call[0] for call in self.b.mock_calls]
        self.assertLess(names.index('orders'), names.index('position'))

    def test_bracket_closes_before_indicator_exit(self):
        self.b.position.return_value = NS(qty='2')
        self.b.close.return_value = None
        with patch('trading_bot.signals', return_value=Decision(exit_signal=True)):
            self.assertEqual(self.engine.step(self.now), 'position already closed')
        self.b.buy.assert_not_called()

    def test_close_timeout_reconciles_pending_order(self):
        self.b.position.return_value = NS(qty='1.5')
        self.b.close.side_effect = TimeoutError()
        with patch('trading_bot.signals', return_value=Decision(exit_signal=True)):
            with self.assertRaises(TimeoutError):
                self.engine.step(self.now)
            self.b.orders.return_value = [NS(id='pending-close')]
            self.assertIn('retained', self.engine.step(self.now))
        self.b.close.assert_called_once()

    def test_blocked_account_never_buys(self):
        self.b.account.return_value.trading_blocked = True
        with self.assertRaisesRegex(ValueError, 'blocked'):
            self.engine.step(self.now)
        self.b.buy.assert_not_called()

    def test_invalid_position_never_closes_or_buys(self):
        for qty in ['nan', '-2', '0']:
            with self.subTest(qty=qty):
                self.b.position.return_value = NS(qty=qty)
                with self.assertRaises(ValueError):
                    self.engine.step(self.now)
        self.b.buy.assert_not_called()
        self.b.close.assert_not_called()


class SDKContractTests(unittest.TestCase):
    def test_real_sdk_builds_bracket_and_exit_requests(self):
        from trading_bot import AlpacaBroker
        from alpaca.trading.enums import OrderClass
        broker = object.__new__(AlpacaBroker)
        broker.config = Config()
        broker.trade = Mock()
        broker.buy(2, 'test-entry', 100.1, 97.1, 106.11)
        request = broker.trade.submit_order.call_args.kwargs['order_data']
        self.assertEqual(request.order_class, OrderClass.BRACKET)
        self.assertEqual(request.stop_loss.stop_price, 97.1)
        broker.close()
        broker.trade.close_position.assert_called_once_with('TQQQ')

    def test_lookup_only_swallows_not_found(self):
        from trading_bot import AlpacaBroker
        from alpaca.common.exceptions import APIError
        from requests.exceptions import HTTPError
        broker = object.__new__(AlpacaBroker)
        broker.trade = Mock()
        error = HTTPError(response=NS(status_code=404))
        broker.trade.get_order_by_client_id.side_effect = APIError('{}', error)
        self.assertFalse(broker.already_submitted('unknown'))
        error.response.status_code = 500
        with self.assertRaises(APIError):
            broker.already_submitted('unknown')


class InstanceLockTests(unittest.TestCase):
    def test_another_process_is_blocked_then_can_acquire_after_release(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from trading_bot import instance_lock
        script = """
from pathlib import Path
import sys
from trading_bot import instance_lock
try:
    with instance_lock(Path(sys.argv[1])):
        print('acquired')
except SystemExit as exc:
    print(str(exc))
    sys.exit(7)
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bot.lock'
            marker = b'existing lock file content'
            path.write_bytes(marker)
            command = [sys.executable, '-c', script, str(path)]
            with instance_lock(path):
                blocked = subprocess.run(command, cwd=Path(__file__).parent,
                                         capture_output=True, text=True, timeout=20)
                self.assertEqual(blocked.returncode, 7, blocked.stderr)
                self.assertIn('already running', blocked.stdout)
            acquired = subprocess.run(command, cwd=Path(__file__).parent,
                                      capture_output=True, text=True, timeout=20)
            self.assertEqual(acquired.returncode, 0, acquired.stderr)
            self.assertIn('acquired', acquired.stdout)
            self.assertEqual(path.read_bytes(), marker)

    def test_body_exception_is_preserved_and_lock_released(self):
        import tempfile
        from pathlib import Path
        from trading_bot import instance_lock
        failure = RuntimeError('original cycle failure')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bot.lock'
            with self.assertRaises(RuntimeError) as raised:
                with instance_lock(path):
                    raise failure
            self.assertIs(raised.exception, failure)
            with instance_lock(path):
                self.assertTrue(path.exists())

    def test_windows_locks_first_byte_with_flushed_content(self):
        import os
        import tempfile
        from pathlib import Path
        from trading_bot import instance_lock
        with tempfile.TemporaryDirectory() as directory:
            for content in (b'', b'keep existing bytes'):
                with self.subTest(content=content):
                    path = Path(directory) / 'bot.lock'
                    path.write_bytes(content)
                    descriptors = []

                    def acquire(fd, mode, byte_count):
                        self.assertEqual(mode, 4242)
                        self.assertEqual(byte_count, 1)
                        self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), 0)
                        self.assertGreaterEqual(os.fstat(fd).st_size, 1)
                        self.assertEqual(os.read(fd, 1), content[:1] or b'\0')
                        descriptors.append(fd)

                    msvcrt = NS(LK_NBLCK=4242, locking=Mock(side_effect=acquire))
                    with patch.dict('sys.modules', {'msvcrt': msvcrt}), patch('trading_bot.sys.platform', 'win32'):
                        with instance_lock(path):
                            self.assertEqual(len(descriptors), 1)
                        msvcrt.locking.assert_called_once()
                    self.assertEqual(path.read_bytes(), content or b'\0')
                    with self.assertRaises(OSError):
                        os.fstat(descriptors[0])

    def test_windows_contention_is_reported_but_io_errors_propagate(self):
        import errno
        import tempfile
        from pathlib import Path
        from trading_bot import instance_lock
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bot.lock'
            msvcrt = NS(LK_NBLCK=2, locking=Mock())
            with patch.dict('sys.modules', {'msvcrt': msvcrt}), patch('trading_bot.sys.platform', 'win32'):
                for code in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    with self.subTest(errno=code):
                        msvcrt.locking.side_effect = OSError(code, 'locked')
                        with self.assertRaisesRegex(SystemExit, 'already running'):
                            with instance_lock(path):
                                self.fail('Contention must not enter the bot')
                failure = OSError(errno.EIO, 'disk error')
                msvcrt.locking.side_effect = failure
                with self.assertRaises(OSError) as raised:
                    with instance_lock(path):
                        self.fail('I/O errors must not enter the bot')
                self.assertIs(raised.exception, failure)

    def test_read_only_main_check_never_locks_or_starts_engine(self):
        import trading_bot
        with patch('sys.argv', ['trading_bot.py', '--check']), \
             patch('dotenv.load_dotenv'), \
             patch.dict('os.environ', {'PAPER': 'true', 'SYMBOL': 'TQQQ'}), \
             patch('trading_bot.RotatingFileHandler'), \
             patch('trading_bot.logging.basicConfig'), \
             patch('trading_bot.instance_lock') as lock, \
             patch('trading_bot.Engine') as engine, \
             patch('trading_bot.AlpacaBroker') as broker:
            trading_bot.main()
        lock.assert_not_called()
        engine.assert_not_called()
        broker.assert_called_once()
        broker.return_value.check.assert_called_once()


if __name__ == '__main__':
    unittest.main()
