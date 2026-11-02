"""File-share queue transport (SPEC 7).

Each client has its own queue tree::

    <queue_root>/inbox       a client drops a job bundle (.zip) here
                /processing  the broker claims a bundle here
                /outbox      the broker writes the result bundle here
                /archive     inputs whose result has been written
                /failed      bundles that could not be processed, each with an .error.txt

A client needs only write access to ``inbox/`` and read access to ``outbox/``, but a
share often grants more, so the broker does not rely on the rest of the tree staying as
it was (SPEC 6.3). It pins each subdirectory on first use and works only through that
handle, so a subdirectory later replaced by a link is never followed.

Guarantees:
  * **atomic claim**: inbox to processing in one step that never replaces a file.
  * **one bundle per name**: a name already in processing/ or with a result is refused.
  * **batch-failure reporting**: a bundle that cannot be unpacked still reports every
    job ``failed``, so no ``--wait`` submitter hangs.
"""

from __future__ import annotations

import os
import shutil
import stat
import time
import uuid
from dataclasses import dataclass, field
from typing import IO

from . import bundle
from .manifest import JobManifest, is_valid_job_id, now_iso

SUBDIRS = ("inbox", "processing", "outbox", "archive", "failed")
# A failed batch lists at most this many manifests it could not read (SPEC 7).
MAX_UNREADABLE_ENTRIES = 100

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_BINARY = getattr(os, "O_BINARY", 0)
# Where directory fds are missing (Windows), each operation re-checks the pinned identity
# first, which leaves a check-to-use race (SPEC 6.3).
_DIR_FD = bundle._FD_WALK and all(
    f in os.supports_dir_fd for f in (os.link, os.rename, os.unlink, os.stat, os.utime)
)
_LINK_NOFOLLOW = os.link in os.supports_follow_symlinks


class ClaimRejected(ValueError):
    """An inbox entry that the broker refused to claim. It was moved to failed/."""


class UnsafeQueueDir(OSError):
    """A queue subdirectory is no longer the directory the broker pinned."""


def _is_link(st: os.stat_result) -> bool:
    return stat.S_ISLNK(st.st_mode) or bool(
        getattr(st, "st_file_attributes", 0) & bundle._REPARSE_POINT
    )


class QueueDir:
    """One queue subdirectory, pinned when the broker starts (SPEC 6.3), so a directory
    replaced by a link or moved later does not redirect the broker's reads and writes.

    Where the OS has directory-fd calls (Linux, macOS) it holds a directory fd opened
    with ``O_NOFOLLOW``, and every list, stat, link, rename, unlink and open is made
    relative to it: the broker keeps using the directory it pinned. On Windows it keeps
    the (device, inode) identity the directory had when pinned and checks, before every
    use, that the path still names it and is not a link or junction; the check and the
    operation are two steps. A path that is already a link when pinned is refused."""

    def __init__(self, path: str):
        self.path = path
        st = os.lstat(path)
        if _is_link(st) or not stat.S_ISDIR(st.st_mode):
            raise ValueError(f"queue directory {path} is a symbolic link or not a directory")
        self._ident = (st.st_dev, st.st_ino)
        self.fd: int | None = None
        if _DIR_FD:
            fd = os.open(path, os.O_RDONLY | bundle._DIRECTORY | _NOFOLLOW)
            fst = os.fstat(fd)
            if (fst.st_dev, fst.st_ino) != self._ident:
                os.close(fd)
                raise ValueError(f"queue directory {path} changed while it was opened")
            self.fd = fd

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _verify(self) -> None:
        try:
            st = os.lstat(self.path)
        except FileNotFoundError:
            raise UnsafeQueueDir(f"queue directory {self.path} is gone") from None
        if _is_link(st) or (st.st_dev, st.st_ino) != self._ident:
            raise UnsafeQueueDir(f"queue directory {self.path} was replaced")

    def at(self, name: str) -> tuple[str, dict]:
        """The (path, kwargs) pair that names *name* in this directory for an os call."""
        if self.fd is not None:
            return name, {"dir_fd": self.fd}
        self._verify()
        return os.path.join(self.path, name), {}

    def listdir(self) -> list[str]:
        if self.fd is not None:
            return os.listdir(self.fd)
        self._verify()
        return os.listdir(self.path)

    def lstat(self, name: str) -> os.stat_result:
        p, kw = self.at(name)
        return os.stat(p, follow_symlinks=False, **kw)

    def exists(self, name: str) -> bool:
        try:
            self.lstat(name)
            return True
        except FileNotFoundError:
            return False

    def unlink(self, name: str) -> None:
        p, kw = self.at(name)
        os.unlink(p, **kw)

    def touch(self, name: str) -> None:
        """Set *name*'s times to now. Through an fd of the file itself where the OS allows
        it, so a link swapped in after the claim is never followed to its target."""
        fd = self.open_regular(name)
        try:
            if os.utime in os.supports_fd:
                os.utime(fd)
                return
        finally:
            os.close(fd)
        # Windows: by path, after open_regular refused a link (SPEC 6.3, residual risks).
        p, kw = self.at(name)
        os.utime(p, None, **kw)

    def open_regular(self, name: str) -> int:
        """Open *name* for reading without following a link or blocking on a FIFO, and
        refuse anything but a regular file. Without O_NOFOLLOW (Windows) the entry is
        checked first and the file opened must be the one checked, which is two steps
        (SPEC 6.3)."""
        p, kw = self.at(name)
        checked = None
        if not _NOFOLLOW:
            checked = os.stat(p, follow_symlinks=False, **kw)
            if _is_link(checked):
                raise bundle.UnsafeMember(f"{name!r} is a symbolic link")
        try:
            fd = os.open(p, os.O_RDONLY | _NOFOLLOW | _NONBLOCK | _BINARY, **kw)
        except OSError as exc:
            raise bundle.UnsafeMember(f"{name!r} cannot be opened safely: {exc}") from None
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise bundle.UnsafeMember(f"{name!r} is not a regular file")
        if checked is not None and (st.st_dev, st.st_ino) != (checked.st_dev, checked.st_ino):
            os.close(fd)
            raise bundle.UnsafeMember(f"{name!r} changed while it was opened")
        return fd

    def write_new(self, name: str, write) -> str:
        if self.fd is not None:
            return bundle.write_new(self.path, name, write, dir_fd=self.fd)
        self._verify()
        return bundle.write_new(self.path, name, write)


def _move_noreplace(src: QueueDir, name: str, dst: QueueDir, new_name: str) -> None:
    """Move *name* from *src* to *new_name* in *dst*, raising FileExistsError instead of
    replacing an entry there.

    A POSIX rename replaces its destination, so this links then unlinks. Without hard links
    (some network shares) it checks then renames, which is fine for one broker but not for
    two draining one tree. Windows renames never replace."""
    if src.fd is None or dst.fd is None:
        s, _ = src.at(name)
        d, _ = dst.at(new_name)
        os.rename(s, d)
        return
    kw = {"follow_symlinks": False} if _LINK_NOFOLLOW else {}
    try:
        os.link(name, new_name, src_dir_fd=src.fd, dst_dir_fd=dst.fd, **kw)
    except FileExistsError:
        raise
    except OSError:
        if dst.exists(new_name):
            raise FileExistsError(new_name) from None
        os.rename(name, new_name, src_dir_fd=src.fd, dst_dir_fd=dst.fd)
        return
    os.unlink(name, dir_fd=src.fd)


@dataclass
class QueueTree:
    """One client's queue directories under a root."""

    root: str
    _pinned: dict[str, QueueDir] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dirs = {name: os.path.join(self.root, name) for name in SUBDIRS}

    def ensure(self) -> QueueTree:
        """Create and pin the subdirectories, refusing any that is a link (SPEC 6.3)."""
        for d in self.dirs.values():
            if os.path.islink(d):
                raise ValueError(f"queue directory {d} is a symbolic link")
            os.makedirs(d, exist_ok=True)
        if not self._pinned:
            pinned = {}
            try:
                for key, path in self.dirs.items():
                    pinned[key] = QueueDir(path)
            except BaseException:
                for qd in pinned.values():
                    qd.close()
                raise
            self._pinned = pinned
        return self

    def close(self) -> None:
        for qd in self._pinned.values():
            qd.close()
        self._pinned = {}

    def _dir(self, key: str) -> QueueDir:
        if not self._pinned:
            self.ensure()
        return self._pinned[key]

    # ---- claim / move ------------------------------------------------------

    def list_inbox(self) -> list[str]:
        """Bundle names waiting in inbox/. Creates nothing before the tree is pinned."""
        try:
            names = (
                self._pinned["inbox"].listdir() if self._pinned else os.listdir(self.dirs["inbox"])
            )
        except FileNotFoundError:
            return []
        return sorted(fn for fn in names if fn.endswith(".zip"))

    def claim(self, filename: str) -> str | None:
        """Move a bundle inbox->processing and return its path, or None if another broker
        won. Raises :class:`ClaimRejected` for an entry that is not a plain file with one
        link, or whose name is already in ``processing/`` or has a result, so a reused name
        can neither replace an active bundle nor overwrite an earlier result. The move
        never replaces an entry (:func:`_move_noreplace`), and the claimed file's mtime is
        set through the pinned directory, since stale reclaim ages bundles from claim."""
        inbox, proc = self._dir("inbox"), self._dir("processing")
        try:
            st = inbox.lstat(filename)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(st.st_mode):
            # A link would make the broker read a file the client cannot read itself.
            self._reject(inbox, filename, "not a regular file")
        try:
            pst: os.stat_result | None = proc.lstat(filename)
        except FileNotFoundError:
            pst = None
        if pst is not None:
            if (pst.st_dev, pst.st_ino) == (st.st_dev, st.st_ino):
                # A claim interrupted between its link and its unlink: finish it.
                inbox.unlink(filename)
                return self._stamp(filename)
            self._reject(inbox, filename, "a bundle with this name is already in processing/")
        if self.has_result(filename):
            self._reject(inbox, filename, "a bundle with this name already has a result")
        try:
            _move_noreplace(inbox, filename, proc, filename)
        except (FileExistsError, FileNotFoundError):
            return None
        st = proc.lstat(filename)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
            # Swapped between check and move, or hard-linked from elsewhere.
            self._reject(proc, filename, "not a regular file with a single link")
        return self._stamp(filename)

    def _stamp(self, filename: str) -> str:
        try:
            self._dir("processing").touch(filename)  # reclaim_stale ages from claim
        except OSError:
            pass
        return os.path.join(self.dirs["processing"], filename)

    def _reject(self, qd: QueueDir, name: str, reason: str) -> None:
        """Move a refused entry to failed/ under a fresh name, with a reason file, and
        raise. No result is written: the outbox name may belong to an earlier bundle."""
        dead = f"{name}.rejected-{uuid.uuid4().hex[:8]}"
        failed = self._dir("failed")
        try:
            _move_noreplace(qd, name, failed, dead)
            failed.write_new(
                dead + ".error.txt", lambda fh: fh.write(f"{now_iso()} {reason}\n".encode())
            )
        except OSError:
            pass
        raise ClaimRejected(f"{name}: {reason}")

    def open_processing(self, filename: str) -> IO[bytes]:
        """Open a claimed bundle without following a link or blocking on a FIFO
        (``O_NOFOLLOW | O_NONBLOCK`` and a regular-file check). Intake unpacks from this one
        file object, so an entry swapped in after the claim is never read."""
        return os.fdopen(self._dir("processing").open_regular(filename), "rb")

    def fail_batch(
        self, processing_path: str, reason: str, max_members: int = bundle.MAX_MEMBERS
    ) -> list[str]:
        """Report every job of an unprocessable batch ``failed`` and move it to ``failed/``.
        Returns the job_ids reported. Never raises on a bad zip, and reads it only within
        the limits of :func:`bundle.read_pending_manifests`."""
        fn = os.path.basename(processing_path)
        reported: list[str] = []
        results: list[dict] = []
        problems: list[str] = []
        try:
            with self.open_processing(fn) as fh:
                manifests = bundle.read_pending_manifests(
                    fh, max_members=max_members, problems=problems
                )
        except (bundle.UnsafeMember, OSError) as exc:
            manifests = []
            problems.append(f"bundle cannot be read ({exc})")
        for raw in manifests:
            if not isinstance(raw, dict):
                raw = {}  # a bare list, string or number: treat as unparseable
            try:
                jm = JobManifest.from_dict(raw)
            except Exception:
                # The raw job_id names a zip member, so only a valid one is echoed back.
                raw_id = raw.get("job_id")
                jid = (
                    raw_id.strip()
                    if isinstance(raw_id, str) and is_valid_job_id(raw_id.strip())
                    else f"unknown-{len(results)}"
                )
                jtype = str(raw.get("job_type") or "batch.unreadable")[:200]
                msg = (
                    reason
                    if jid != f"unknown-{len(results)}" or raw_id is None
                    else (f"{reason} (submitted job_id {str(raw_id)[:200]!r} is not a valid id)")
                )
                results.append(
                    {"job_id": jid, "job_type": jtype, "status": "failed", "message": msg}
                )
                reported.append(jid)
                continue
            jm.mark_failed(reason)
            results.append(jm.to_dict())
            reported.append(jm.job_id)
        # An unreadable manifest still gets an entry, so no job just goes missing.
        for problem in problems[:MAX_UNREADABLE_ENTRIES]:
            jid = f"unknown-{len(results)}"
            results.append(
                {
                    "job_id": jid,
                    "job_type": "batch.unreadable",
                    "status": "failed",
                    "message": f"{reason} ({problem})",
                }
            )
            reported.append(jid)
        # client.wait_result needs at least one terminal manifest.
        if not results:
            jid = os.path.splitext(fn)[0] or "unreadable-batch"
            results.append(
                {
                    "job_id": jid if is_valid_job_id(jid) else "unreadable-batch",
                    "job_type": "batch.unreadable",
                    "status": "failed",
                    "message": reason,
                }
            )
            reported.append(jid)
        self._dir("outbox").write_new(fn, lambda fh: bundle.pack(fh, results, root=self.root))
        self.move_to_failed(fn, reason)
        return reported

    def move_to_failed(self, filename: str, reason: str) -> str | None:
        """Move a bundle from processing/ to failed/ beside a ``.error.txt`` holding
        *reason*, and return its new name (None if it is gone)."""
        dead = self._move_aside(filename, "failed")
        if dead is not None:
            self._dir("failed").write_new(
                dead + ".error.txt", lambda fh: fh.write(f"{now_iso()} {reason}\n".encode())
            )
        return dead

    def _move_aside(self, filename: str, key: str) -> str | None:
        """Move a bundle out of processing/ into *key* and return its new name, or None if
        it is gone. A name clash gets a fresh name. Across filesystems it is copied and then
        removed, since a copy left in processing/ would be reclaimed and run again. If that
        removal fails, the copy is removed instead and OSError raised: the bundle has not
        moved, and the caller keeps it."""
        proc, dest = self._dir("processing"), self._dir(key)
        name = filename
        for _ in range(2):
            try:
                _move_noreplace(proc, filename, dest, name)
                return name
            except FileExistsError:
                name = f"{filename}.{uuid.uuid4().hex[:8]}"
            except FileNotFoundError:
                return None
            except UnsafeQueueDir:
                raise
            except OSError:
                break  # cross-device or similar: copy below
        with os.fdopen(proc.open_regular(filename), "rb") as src:
            dest.write_new(name, lambda fh: shutil.copyfileobj(src, fh))
        try:
            proc.unlink(filename)
        except OSError:
            try:
                dest.unlink(name)
            except OSError:
                pass
            raise
        return name

    def write_result(
        self,
        filename: str,
        results: list[dict],
        files_root: str | None = None,
        files: list[str] | None = None,
    ) -> str:
        """Write a result bundle to outbox atomically, without following a link in its way.
        It packs the manifests and each listed file from *files_root*, every one read
        again without following symlinks."""
        return self._dir("outbox").write_new(
            filename,
            lambda fh: bundle.pack(
                fh, results, root=files_root or self.root, extra_members=files or [], nofollow=True
            ),
        )

    def archive(self, processing_path: str) -> str | None:
        return self._move_aside(os.path.basename(processing_path), "archive")

    def has_result(self, filename: str) -> bool:
        """True if outbox/ or archive/ holds this name. Archive counts because an input is
        archived only after its result is written (Scheduler._retire). Raises OSError if
        the share cannot say, so no caller takes a storage error for "no result"."""
        return any(self._dir(key).exists(filename) for key in ("outbox", "archive"))

    def archive_leftovers(self, skip: set[str] | None = None) -> list[str]:
        """Archive bundles still in processing/ that already have a result, except those in
        *skip*: an archive step that failed after delivery (Scheduler._retire), or a broker
        stopped between the two. Reclaim skips them, so without this they stay forever.
        Returns the names archived."""
        done: list[str] = []
        proc = self._dir("processing")
        for fn in proc.listdir():
            if not fn.endswith(".zip") or (skip and fn in skip):
                continue
            try:
                if self.has_result(fn) and self._move_aside(fn, "archive"):
                    done.append(fn)
            except bundle.UnsafeMember:
                continue  # not a regular file; nothing the broker delivered
            except UnsafeQueueDir:
                raise
            except OSError:
                continue  # the next reclaim tries again
        return done

    def left_in_processing(self) -> list[tuple[str, float]]:
        """(name, claim time) of each bundle in processing/ without a result, for the start
        of a broker: none of them is live then, and stale reclaim runs each again."""
        proc, left = self._dir("processing"), []
        for fn in sorted(proc.listdir()):
            if not fn.endswith(".zip"):
                continue
            try:
                if not self.has_result(fn):
                    left.append((fn, proc.lstat(fn).st_mtime))
            except UnsafeQueueDir:
                raise
            except OSError:
                continue
        return left

    def reclaim_stale(
        self, max_age_s: float, now: float | None = None, skip: set[str] | None = None
    ) -> list[str]:
        """Move bundles older than *max_age_s* from processing/ back to inbox, except those
        in *skip* or with a result (a re-run could overwrite collected outputs). Returns
        the reclaimed filenames. Age is weak evidence: *skip* does not survive a restart."""
        now = time.time() if now is None else now
        reclaimed: list[str] = []
        proc, inbox = self._dir("processing"), self._dir("inbox")
        for fn in proc.listdir():
            if not fn.endswith(".zip") or (skip and fn in skip):
                continue
            try:
                if self.has_result(fn):
                    continue
                age = now - proc.lstat(fn).st_mtime
            except UnsafeQueueDir:
                raise
            except OSError:
                continue  # gone, or the share cannot say whether it has a result
            if age >= max_age_s:
                try:
                    # Never over a new bundle the client has since put in inbox/.
                    _move_noreplace(proc, fn, inbox, fn)
                    reclaimed.append(fn)
                except (FileExistsError, FileNotFoundError):
                    pass
        return reclaimed


def discover_queues(queue_roots: list[str]) -> list[QueueTree]:
    """One QueueTree per client root; nothing is opened until first use."""
    return [QueueTree(r) for r in queue_roots]
