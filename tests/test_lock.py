#!/usr/bin/env python3
"""The workdir lock.

Byte-range locks are enforced by the OS across handles even WITHIN one
process (LockFile on Windows, flock elsewhere), which is what makes these
tests meaningful without spawning a second process - and what makes the
real protection work against a genuinely concurrent one.
"""

import os
import shutil
import tempfile
import unittest

from a2a.lock import WorkdirBusy, WorkdirLock


class TestLock(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="a2a-lock-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_acquire_and_release(self):
        lock = WorkdirLock(self.dir)
        lock.acquire()
        self.assertTrue(os.path.exists(os.path.join(self.dir, ".lock")))
        lock.release()
        self.assertIsNone(lock._fh)

    def test_released_lock_can_be_taken_again(self):
        first = WorkdirLock(self.dir)
        first.acquire()
        first.release()
        second = WorkdirLock(self.dir)
        second.acquire()
        second.release()

    def test_second_taker_is_rejected(self):
        first = WorkdirLock(self.dir)
        first.acquire()
        second = WorkdirLock(self.dir)
        with self.assertRaises(WorkdirBusy):
            second.acquire()
        # the rejected taker must not have leaked its handle
        self.assertIsNone(second._fh)
        first.release()

    def test_reentrant_acquire_is_a_noop(self):
        lock = WorkdirLock(self.dir)
        lock.acquire()
        lock.acquire()  # same object: not a conflict
        lock.release()

    def test_release_when_not_held_is_safe(self):
        lock = WorkdirLock(self.dir)
        lock.release()  # must not raise

    def test_context_manager(self):
        with WorkdirLock(self.dir) as lock:
            self.assertIsNotNone(lock._fh)
        self.assertIsNone(lock._fh)

    def test_busy_message_names_the_workdir(self):
        first = WorkdirLock(self.dir)
        first.acquire()
        try:
            second = WorkdirLock(self.dir)
            with self.assertRaises(WorkdirBusy) as ctx:
                second.acquire()
            self.assertIn(self.dir, str(ctx.exception))
        finally:
            first.release()


if __name__ == "__main__":
    unittest.main()
