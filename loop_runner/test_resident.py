import unittest
import resident


class ResidentTests(unittest.TestCase):
    def test_order_and_limit(self):
        events = []
        resident.execute_sequence(resident.plan('loop', 10),
            lambda d: events.append(d), lambda a: events.append(a), lambda *args: None)
        self.assertEqual(events, ['l2r', 'left', 'r2l', 'right'] * 10)

    def test_failure_no_reset_or_next(self):
        events = []

        def fail(d):
            events.append(d)
            raise RuntimeError('mock inference failed')

        with self.assertRaises(RuntimeError):
            resident.execute_sequence(['l2r', 'r2l'], fail,
                lambda a: events.append(a), lambda *args: None)
        self.assertEqual(events, ['l2r'])

    def test_stop_no_reset_or_next(self):
        events = []

        def stop(d):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            resident.execute_sequence(['l2r', 'r2l'], stop,
                lambda a: events.append(a), lambda *args: None)
        self.assertEqual(events, [])

    def test_reset_failure_no_next(self):
        events = []

        def fail(a):
            raise RuntimeError('mock reset failed')

        with self.assertRaises(RuntimeError):
            resident.execute_sequence(['l2r', 'r2l'], events.append, fail, lambda *args: None)
        self.assertEqual(events, ['l2r'])


if __name__ == '__main__':
    unittest.main()
