import unittest
import resident_new_env as runner


class SequenceTests(unittest.TestCase):
    def test_ten_rounds(self):
        events = []
        runner.execute_sequence(runner.plan('loop', 10),
            events.append, events.append, lambda *args: None)
        self.assertEqual(events, ['l2r', 'left', 'r2l', 'right'] * 10)

    def test_interrupt(self):
        events = []
        def stop(direction):
            events.append(direction)
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            runner.execute_sequence(['l2r', 'r2l'], stop, events.append, lambda *args: None)
        self.assertEqual(events, ['l2r'])

    def test_reset_failure(self):
        events = []
        def fail(arm):
            raise RuntimeError('reset error')
        with self.assertRaises(RuntimeError):
            runner.execute_sequence(['l2r', 'r2l'], events.append, fail, lambda *args: None)
        self.assertEqual(events, ['l2r'])

    def test_checkpoints(self):
        self.assertIn('new_env_ft_from_80ep20k', str(runner.CHECKPOINTS['r2l']))
        self.assertIn('050000', str(runner.CHECKPOINTS['l2r']))


if __name__ == '__main__':
    unittest.main()
