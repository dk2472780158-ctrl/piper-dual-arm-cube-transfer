"""Hardware-free tests: every child is replaced with a dummy Python process."""
import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('run_cube.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Input:
    def __init__(self):
        self.event = threading.Event()

    def isatty(self):
        return True

    def __iter__(self):
        self.event.wait()
        yield 'stop\n'


class RunnerTests(unittest.TestCase):
    def run_dummy(self, program, plan, stop=False):
        real_popen = subprocess.Popen
        commands = []
        inp = Input()
        saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}

        def dummy(command, **kwargs):
            commands.append(command)
            return real_popen([sys.executable, '-u', '-c', program], **kwargs)

        try:
            with tempfile.TemporaryDirectory() as tmp:
                with patch.object(runner, 'HERE', Path(tmp)), patch.object(runner, 'PROJECT', Path(tmp)):
                    with patch.object(runner.sys, 'stdin', inp), patch.object(runner.subprocess, 'Popen', dummy):
                        if stop:
                            threading.Timer(0.5, inp.event.set).start()
                        result = runner.run(plan)
            return result, commands
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)

    def test_ten_rounds(self):
        self.assertEqual(runner.sequence('loop', 10), ['l2r', 'r2l'] * 10)
        self.assertEqual(runner.sequence('l2r', 10), ['l2r'])

    def test_success(self):
        code, calls = self.run_dummy("print('[piper-auto-reset] complete')", ['l2r', 'r2l'])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)

    def test_complete_ten_rounds(self):
        code, calls = self.run_dummy("print('[piper-auto-reset] complete')", runner.sequence('loop', 10))
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 20)

    def test_error_stops_sequence(self):
        code, calls = self.run_dummy('raise SystemExit(1)', ['l2r', 'r2l'])
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)

    def test_no_reset_stops_sequence(self):
        code, calls = self.run_dummy("print('early exit')", ['l2r', 'r2l'])
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)

    def test_stop_interrupts_child(self):
        code, calls = self.run_dummy('import time; time.sleep(30)', ['l2r', 'r2l'], stop=True)
        self.assertEqual(code, 130)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
