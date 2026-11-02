"""One broker per state directory (SPEC 10).

A starting broker clears the batch and serve directories in its ``state_dir`` and reaps
leftover serve containers, so a second broker started by mistake would destroy the
running one's state. :class:`BrokerLock` holds an exclusive lock on ``broker.lock`` there
for the broker's lifetime; the operating system releases it when the process exits,
however it exits.
"""

from __future__ import annotations

import os
import sys

LOCK_NAME = "broker.lock"
# Windows locks a byte range, and a locked range cannot be read by another process. The
# lock is taken on a byte past the PID, so a second broker can still read who holds it.
_WIN_LOCK_OFFSET = 1 << 20


class BrokerRunning(RuntimeError):
    """Another broker holds the lock."""


class BrokerLock:
    def __init__(self, state_dir: str):
        self.path = os.path.join(state_dir, LOCK_NAME)
        self._fd: int | None = None

    def acquire(self) -> None:
        """Take the lock and write this process's PID into the file, or raise BrokerRunning
        naming the PID of the broker that holds it."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _lock(fd)
        except OSError:
            pid = _read_pid(fd)
            os.close(fd)
            raise BrokerRunning(
                f"another broker (pid {pid or 'unknown'}) is using {os.path.dirname(self.path)}; "
                "stop it first, or give this one its own state_dir"
            ) from None
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            _unlock(self._fd)
        except OSError:
            pass
        os.close(self._fd)
        self._fd = None


def _read_pid(fd: int) -> str:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        return os.read(fd, 32).decode("ascii", "replace").strip()
    except OSError:
        return ""


if sys.platform == "win32":
    import msvcrt

    def _lock(fd: int) -> None:
        os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd: int) -> None:
        os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
