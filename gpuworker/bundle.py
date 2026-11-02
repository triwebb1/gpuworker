"""Portable job bundles (zip) with a path-traversal guard on unpack.

SPEC 4.2 / 7. A bundle is a
zip whose members are paths relative to a bundle root, carrying job manifests under
``jobs/pending/<job_id>.json`` plus the jobs' input files. Three safety-critical
behaviours:

  * ``safe_join`` rejects any member that escapes the destination root, so a
    corrupt or mistaken bundle cannot write outside the unpack dir (SPEC 6.3).
  * ``open_regular`` reads paths a job could have written without following
    symlinks, so a link left in a job's output does not pull a host file into a result.
  * ``unpack`` accepts a member only if its name is already in the one canonical
    spelling (:func:`~gpuworker.manifest.is_bundle_path`), and no two members name the
    same file. Intake then reads the manifests back from the unpacked files with
    ``read_unpacked_manifests``, the same bytes a worker reads.
  * ``read_pending_manifests`` reads manifests straight from the zip bytes without
    extracting, so even an un-unpackable batch can still report every job back as
    failed (the batch-failure reliability contract, SPEC 7, 10).

``write_new`` is how the broker writes into a queue directory, which a client may also
be able to write: it never follows a link it finds in its way.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import stat
import time
import uuid
import zipfile
from collections.abc import Callable
from typing import IO, Any

from .manifest import is_bundle_path

PENDING_PREFIX = "jobs/pending/"

# Intake limits for one bundle (SPEC 6.3). The scheduler passes its configured values;
# these are the defaults for direct callers.
MAX_MEMBERS = 10_000
MAX_BUNDLE_BYTES = 8 * 1024**3
# A job manifest is a small JSON object; anything larger is not one.
MAX_MANIFEST_BYTES = 1024 * 1024
# Total manifest bytes read from one bundle, so reading a corrupt bundle's manifests (also
# done for a bundle that is being failed) stays bounded however many it holds.
MAX_MANIFEST_TOTAL_BYTES = 64 * 1024 * 1024

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
# Walk paths one directory fd at a time where the OS supports it (Linux, macOS), so no
# component can be swapped for a symlink between the check and the open.
_FD_WALK = bool(_NOFOLLOW and _DIRECTORY) and os.open in os.supports_dir_fd
_REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT: Windows symlinks and junctions


class UnsafeMember(ValueError):
    """A path escapes its root, goes through a symlink, or is not a regular file."""


class BundleTooLarge(ValueError):
    """A bundle exceeds the intake member-count or size limit."""


class OutputTooLarge(ValueError):
    """A job output is larger than the broker's per-output limit."""


def safe_join(root: str, member: str) -> str:
    """Resolve *member* under *root*, refusing any path that escapes the tree."""
    root_abs = os.path.abspath(root)
    dest = os.path.normpath(os.path.join(root_abs, member))
    if dest != root_abs and not dest.startswith(root_abs + os.sep):
        raise ValueError(f"unsafe path in bundle: {member!r}")
    return dest


def _parts(root: str, member: str) -> list[str]:
    dest = safe_join(root, member)
    rel = os.path.relpath(dest, os.path.abspath(root))
    if rel == ".":
        raise UnsafeMember(f"not a file path: {member!r}")
    return rel.split(os.sep)


def open_regular(root: str, member: str) -> int:
    """Open *member* under *root* for reading and return the fd, refusing to follow a
    symlink at any component and refusing anything but a regular file.

    For paths a sandboxed job could have written (its output dir): :func:`safe_join` is
    lexical only, so a job that links ``out/x`` to a host file, or replaces a directory
    with a link, would otherwise make the broker read that file into a result. Where the
    OS has ``dir_fd`` opens, the path is walked one directory fd at a time with
    ``O_NOFOLLOW``; elsewhere (Windows) every component is ``lstat``-checked and the
    resolved path must stay under *root*, which is two steps (SPEC 6.3). Raises
    :class:`UnsafeMember`, or ``FileNotFoundError`` if the member does not exist.
    """
    parts = _parts(root, member)
    if _FD_WALK:
        dfd = os.open(root, os.O_RDONLY | _DIRECTORY)
        try:
            for name in parts[:-1]:
                try:
                    nfd = os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=dfd)
                except OSError as exc:
                    # A symlink here fails with ELOOP or, because of O_DIRECTORY, ENOTDIR;
                    # ENOTDIR also means a plain file, which is just a missing path.
                    if exc.errno in (errno.ELOOP, errno.ENOTDIR) and stat.S_ISLNK(
                        os.stat(name, dir_fd=dfd, follow_symlinks=False).st_mode
                    ):
                        raise UnsafeMember(f"{member!r}: {name!r} is a symbolic link") from None
                    raise
                os.close(dfd)
                dfd = nfd
            try:
                # O_NONBLOCK so a FIFO where a file should be cannot block the open.
                fd = os.open(parts[-1], os.O_RDONLY | _NOFOLLOW | os.O_NONBLOCK, dir_fd=dfd)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise UnsafeMember(f"{member!r} is a symbolic link") from None
                raise
        finally:
            os.close(dfd)
    else:
        # No dir-fd walk on this OS: lstat every component, then require the resolved
        # path to stay under the resolved root. Not race-free, which is acceptable only
        # because container jobs run on Linux, where the branch above is used.
        cur = os.path.abspath(root)
        for name in parts:
            cur = os.path.join(cur, name)
            st = os.lstat(cur)
            if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & _REPARSE_POINT:
                raise UnsafeMember(f"{member!r}: {name!r} is a symbolic link")
        if not os.path.realpath(cur).startswith(os.path.realpath(root) + os.sep):
            raise UnsafeMember(f"{member!r} resolves outside its root")
        if not stat.S_ISREG(st.st_mode):
            raise UnsafeMember(f"{member!r} is not a regular file")
        fd = os.open(cur, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise UnsafeMember(f"{member!r} is not a regular file")
    return fd


def copy_regular(src_root: str, member: str, dest: str, max_bytes: int | None = None) -> None:
    """Copy *member* of *src_root* (read with :func:`open_regular`) to the broker-owned
    path *dest*, creating its parent directories. With *max_bytes*, a member larger than
    that raises :class:`OutputTooLarge` and leaves no partial copy. The bytes are counted
    as they are copied, because a serve container is still running and can grow the file
    after it was checked."""
    fd = open_regular(src_root, member)
    with os.fdopen(fd, "rb") as src:
        if max_bytes is not None and os.fstat(src.fileno()).st_size > max_bytes:
            raise OutputTooLarge(f"{member!r} is larger than {max_bytes} bytes")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        copied = 0
        try:
            with open(dest, "wb") as out:
                while chunk := src.read(1 << 20):
                    copied += len(chunk)
                    if max_bytes is not None and copied > max_bytes:
                        raise OutputTooLarge(f"{member!r} is larger than {max_bytes} bytes")
                    out.write(chunk)
        except OutputTooLarge:
            os.unlink(dest)
            raise


def collect_outputs(
    src_root: str, outputs: list[str], dest_root: str, max_bytes: int | None = None
) -> list[str]:
    """Copy each declared output that exists under *src_root* to the same relative path
    under *dest_root*, and return the ones found. A missing output is skipped (the caller
    decides what an incomplete set means). A symlink or non-regular file raises
    :class:`UnsafeMember`: a job that leaves one is not reporting an output. An output over
    *max_bytes* raises :class:`OutputTooLarge`."""
    found: list[str] = []
    for rel in outputs:
        try:
            copy_regular(src_root, rel, safe_join(dest_root, rel), max_bytes)
        except (FileNotFoundError, NotADirectoryError):
            continue
        found.append(rel)
    return found


def write_new(
    dir_path: str, name: str, write: Callable[[IO[bytes]], Any], dir_fd: int | None = None
) -> str:
    """Write a file *name* in *dir_path* through *write(fileobj)*, atomically, without
    following a link anywhere on the way, and return its path.

    For queue directories (SPEC 6.3), which a client may also write: if a symlink is left
    where the broker is about to write, a plain ``open(path, "w")`` would write through it
    to whatever file it points at. So the data goes to a new temporary name
    created with ``O_CREAT | O_EXCL | O_NOFOLLOW`` (which fails on any existing entry,
    links included), then is renamed over *name* (a rename replaces a link, never its
    target). Where the OS supports it, both happen relative to a directory fd opened with
    ``O_NOFOLLOW``, so *dir_path* itself cannot be swapped for a link in between. A caller
    that pinned the directory earlier passes its fd as *dir_fd*, which is used instead and
    left open.
    """
    # A fixed-length name: one built from *name* fails on a name near the length limit.
    tmp = f".gpuworker-{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | getattr(os, "O_BINARY", 0)
    if _FD_WALK and os.rename in os.supports_dir_fd:
        dfd = (
            dir_fd
            if dir_fd is not None
            else os.open(dir_path, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
        )
        try:
            fd = os.open(tmp, flags, 0o644, dir_fd=dfd)
            try:
                with os.fdopen(fd, "wb") as fh:
                    write(fh)
                os.replace(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
            except BaseException:
                try:
                    os.unlink(tmp, dir_fd=dfd)
                except OSError:
                    pass
                raise
        finally:
            if dir_fd is None:
                os.close(dfd)
        return os.path.join(dir_path, name)
    # No dir-fd calls on this OS (Windows): refuse a linked directory, then rely on
    # O_EXCL for the file itself. Not race-free for the directory; see SPEC 6.3.
    st = os.lstat(dir_path)
    if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & _REPARSE_POINT:
        raise UnsafeMember(f"{dir_path!r} is a symbolic link")
    tmp_path = os.path.join(dir_path, tmp)
    fd = os.open(tmp_path, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
        os.replace(tmp_path, os.path.join(dir_path, name))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return os.path.join(dir_path, name)


def write_new_bytes(dir_path: str, name: str, data: bytes) -> str:
    """:func:`write_new` for a small payload."""
    return write_new(dir_path, name, lambda fh: fh.write(data))


def is_regular_member(root: str, member: str) -> bool:
    """True if *member* is a regular file under *root*, reached without symlinks."""
    try:
        os.close(open_regular(root, member))
        return True
    except (ValueError, OSError):
        return False


def check_member_name(member: str) -> str:
    """Return the path *member* names (without a directory entry's trailing ``/``), or
    raise :class:`UnsafeMember` if it is not in canonical form. A name such as
    ``jobs/./pending/x.json`` would unpack onto the same file as ``jobs/pending/x.json``
    while a check of the raw name sees two different paths, so it is refused, not
    normalized."""
    path = member[:-1] if member.endswith("/") else member
    if not is_bundle_path(path):
        raise UnsafeMember(f"bundle member name is not a plain relative path: {member[:200]!r}")
    return path


def read_unpacked_manifests(
    root: str, members: list[str] | None = None, max_total_bytes: int = MAX_MANIFEST_TOTAL_BYTES
) -> list[tuple[str, Any]]:
    """Read the job manifests of an unpacked bundle from ``<root>/jobs/pending/`` and
    return ``(file name, parsed JSON)`` pairs, in the order of *members* (the names
    :func:`unpack` returned), so jobs keep their order in the bundle. These are the files a
    worker reads, so they are what intake validates. The directory may hold only regular
    ``*.json`` files; a file that is too large or does not parse raises ValueError."""
    pending = os.path.join(root, *PENDING_PREFIX.rstrip("/").split("/"))
    try:
        names = os.listdir(pending)
    except FileNotFoundError:
        return []
    rank = {m[len(PENDING_PREFIX) :]: i for i, m in enumerate(members or [])}
    names.sort(key=lambda n: (rank.get(n, len(rank)), n))
    out: list[tuple[str, Any]] = []
    total = 0
    for name in names:
        if not name.endswith(".json"):
            raise ValueError(f"{PENDING_PREFIX} holds {name[:200]!r}, which is not a .json file")
        with os.fdopen(open_regular(pending, name), "rb") as fh:
            raw = fh.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ValueError(f"{PENDING_PREFIX}{name}: larger than {MAX_MANIFEST_BYTES} bytes")
        total += len(raw)
        if total > max_total_bytes:
            raise ValueError(f"manifests exceed {max_total_bytes} bytes")
        try:
            out.append((name, json.loads(raw.decode("utf-8"))))
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"{PENDING_PREFIX}{name}: unreadable ({type(exc).__name__})") from None
    return out


def read_pending_manifests(
    zip_path: str | IO[bytes],
    max_members: int = MAX_MEMBERS,
    max_total_bytes: int = MAX_MANIFEST_TOTAL_BYTES,
    problems: list[str] | None = None,
) -> list[Any]:
    """Return the pending job manifests embedded in a bundle without extracting
    anything to disk. Never raises, so this is safe on a corrupt or malformed
    bundle. Each parsed entry is returned as JSON gives it; callers check that it is an
    object. An entry that cannot be read is left out and, if *problems* is given, named
    there with the reason, so the caller can fail it instead of losing it.

    Bounded, because the batch-failure path calls this on bundles that intake refused: a
    bundle with more than *max_members* entries yields nothing, and reading stops once
    *max_total_bytes* of manifests have been read. Any parse error counts as unreadable,
    including the RecursionError a deeply nested document raises.
    """
    out: list[Any] = []
    bad = problems if problems is not None else []
    read = 0
    try:
        if not isinstance(zip_path, str):
            zip_path.seek(0)
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            if len(names) > max_members:
                bad.append(f"bundle has {len(names)} members (limit {max_members})")
                return out
            for member in names:
                if not (member.startswith(PENDING_PREFIX) and member.endswith(".json")):
                    continue
                if read >= max_total_bytes:
                    bad.append(f"manifests exceed {max_total_bytes} bytes; the rest were not read")
                    break
                try:
                    with zf.open(member) as fh:
                        raw = fh.read(MAX_MANIFEST_BYTES + 1)
                    read += len(raw)
                    if len(raw) > MAX_MANIFEST_BYTES:
                        bad.append(f"{member[:200]}: larger than {MAX_MANIFEST_BYTES} bytes")
                        continue
                    out.append(json.loads(raw.decode("utf-8")))
                except Exception as exc:
                    bad.append(f"{member[:200]}: unreadable ({type(exc).__name__})")
    except Exception as exc:
        bad.append(f"not a readable zip ({type(exc).__name__})")
    return out


def pack(
    zip_path: str | IO[bytes],
    manifests: list[dict[str, Any]],
    root: str,
    extra_members: list[str] | None = None,
    nofollow: bool = False,
) -> str | IO[bytes]:
    """Build a bundle zip at a path or into an open binary file. *manifests* are written
    under ``jobs/pending/``; *extra_members* are root-relative paths copied in from disk
    (inputs). With *nofollow*, each member is read through :func:`open_regular`, and one
    that is a symlink or not a regular file raises; the broker uses this for result
    bundles."""
    if isinstance(zip_path, str):
        os.makedirs(os.path.dirname(os.path.abspath(zip_path)) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in manifests:
            jid = m["job_id"]
            zf.writestr(f"{PENDING_PREFIX}{jid}.json", json.dumps(m, indent=2, sort_keys=True))
        for rel in extra_members or []:
            check_member_name(rel)
            if nofollow:
                _write_regular(zf, root, rel)
                continue
            src = safe_join(root, rel)
            if os.path.isfile(src):
                zf.write(src, rel)
    return zip_path


def _write_regular(zf: zipfile.ZipFile, root: str, rel: str) -> None:
    with os.fdopen(open_regular(root, rel), "rb") as src:
        st = os.fstat(src.fileno())
        # Zip timestamps start in 1980.
        stamp = time.localtime(max(st.st_mtime, 315619200))[:6]
        zi = zipfile.ZipInfo(rel, stamp)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = (st.st_mode & 0xFFFF) << 16
        with zf.open(zi, "w", force_zip64=True) as dst:
            shutil.copyfileobj(src, dst)


def extract_outputs(
    zip_path: str, dest_root: str, skip_prefixes: tuple[str, ...] = (PENDING_PREFIX,)
) -> list[str]:
    """Extract a result bundle's output members (everything except the job manifests
    under ``jobs/pending/``) into *dest_root*, guarded against traversal and streamed
    (so large binary outputs don't load into memory). Returns extracted member names.
    Used by the client collect path to retrieve output files."""
    os.makedirs(dest_root, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            check_member_name(member)
            if member.endswith("/") or any(member.startswith(p) for p in skip_prefixes):
                continue
            dest = safe_join(dest_root, member)  # raises on traversal
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as fh:
                shutil.copyfileobj(src, fh)
            extracted.append(member)
    return extracted


def unpack(
    zip_path: str | IO[bytes],
    dest_root: str,
    max_members: int = MAX_MEMBERS,
    max_bytes: int = MAX_BUNDLE_BYTES,
) -> list[str]:
    """Extract a bundle into *dest_root*, guarding every member. Returns the list of
    extracted member names. Raises ValueError on the first unsafe path, and
    :class:`BundleTooLarge` past *max_members* entries or *max_bytes* extracted. The byte
    limit counts what is actually written, not the sizes the zip declares.

    Every member name must already be canonical (:func:`check_member_name`), and no two
    may name the same path, compared case-insensitively because the unpack directory may
    be on a filesystem that ignores case. So each file on disk comes from exactly one
    member, whose name is the path it was written to."""
    os.makedirs(dest_root, exist_ok=True)
    extracted: list[str] = []
    written = 0
    if not isinstance(zip_path, str):
        zip_path.seek(0)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if len(names) > max_members:
            raise BundleTooLarge(f"bundle has {len(names)} members (limit {max_members})")
        seen: set[str] = set()
        for member in names:
            key = check_member_name(member).casefold()
            if key in seen:
                raise UnsafeMember(f"two bundle members name the same path: {member[:200]!r}")
            seen.add(key)
        for member in names:
            dest = safe_join(dest_root, member)  # raises on traversal
            if member.endswith("/"):
                os.makedirs(dest, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as fh:
                while chunk := src.read(1 << 20):
                    written += len(chunk)
                    if written > max_bytes:
                        raise BundleTooLarge(f"bundle unpacks to more than {max_bytes} bytes")
                    fh.write(chunk)
            extracted.append(member)
    return extracted
