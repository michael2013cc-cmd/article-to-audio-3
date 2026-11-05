#!/usr/bin/env python3
"""A workdir lock, so two runs on the same output cannot overwrite each other.

WHY THIS EXISTS
The workdir is located by OUTPUT NAME, not by run: ``out/ep.mp3`` always
uses ``out/ep_work/``, so a second process building the same output - a
 forgotten earlier run, another terminal, another agent - shares its
raw_/norm_/joined files with the first. The results interleave silently:
v1 measured a job log where a ``--verify-only`` pass overwrote a finished
``transcript.txt`` and the verification numbers (25/25 and 20/25) came out
interleaved, which reads exactly like "the audio is damaged".

The lock is an OS-level exclusive lock on ``<workdir>/.lock`` (LockFile on
Windows, flock elsewhere), held by the open file handle:

- the OS releases it when the process exits, even on a crash - a stale
  lock file left by a killed run blocks nobody;
- a second process fails to acquire it and exits with code 5 instead of
  racing.

That is the whole point of using the OS lock rather than a PID file: a PID
file must be judged ("is that PID alive? is it OUR pid recycled by another
program?"), and both wrong answers are worse than not locking.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "win32":  # pragma: no branch - platform selection
    import msvcrt

    def _try_lock(fh) -> bool:
        try:
            fh.seek(0)
            # Lock one byte at position 0, non-blocking.
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fh) -> None:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

else:
    import fcntl

    def _try_lock(fh) -> bool:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fh) -> None:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


class WorkdirBusy(RuntimeError):
    """Another process holds the lock on this workdir."""

    def __init__(self, workdir: str, lockfile: str):
        self.workdir = workdir
        self.lockfile = lockfile
        super().__init__(
            f"另一个进程正在使用任务目录 {workdir}（锁文件 {lockfile}）。"
            "同一输出文件同时只能有一个运行：等它结束，或换一个 -o 输出文件名。"
        )


class WorkdirLock:
    """Exclusive, crash-safe lock on a workdir. Also a context manager."""

    def __init__(self, workdir: str):
        self.workdir = workdir
        self.lockfile = os.path.join(workdir, ".lock")
        self._fh = None

    def acquire(self) -> None:
        """Take the lock, or raise :class:`WorkdirBusy`.

        Re-acquiring from the same object is a no-op (one run, one lock).
        """
        if self._fh is not None:
            return
        os.makedirs(self.workdir, exist_ok=True)
        # "a+b" never truncates an existing holder's file and works on a
        # fresh (empty) file alike; the lock is on the handle, not the
        # bytes, so the content is only for humans reading the directory.
        fh = open(self.lockfile, "a+b")
        try:
            if not _try_lock(fh):
                raise WorkdirBusy(self.workdir, self.lockfile)
        except BaseException:
            fh.close()
            raise
        self._fh = fh
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(f"{os.getpid()}\n".encode())
            fh.flush()
        except OSError:
            pass  # diagnostics only - the lock itself already holds

    def release(self) -> None:
        """Drop the lock. Safe to call when not held.

        The lock FILE is deliberately left behind: unlinking it after the
        unlock opens the classic race where the next taker already holds
        the old inode while a third process creates a fresh file and locks
        that - two "holders", no exclusion. A leftover empty .lock file
        blocks nobody, because the lock lives on the handle, not the file.
        """
        if self._fh is None:
            return
        fh, self._fh = self._fh, None
        _unlock(fh)
        fh.close()

    def __enter__(self) -> "WorkdirLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
