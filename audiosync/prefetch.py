"""Reading a window's stretch of a large file ahead of FFmpeg, one reader per drive.

FFmpeg reaches a Matroska file's audio by reading everything stored around
it -- on a 4K remux the picture is ~95% of the bytes -- in pieces about the
size of one video frame. Four pairs measured at once is four FFmpegs doing
that on the same drive, and a hard disk then spends its time moving its head
between them rather than reading. Here each window's stretch of the file is
read once, start to end in large pieces, by one reader per drive at a time,
into the operating system's file cache; FFmpeg then decodes it from memory,
as many at once as there are pairs.

Only drives that pay for seeking are read this way. An SSD does not, and an
NVMe one reads faster for four readers than for one taking turns -- holding
them to one at a time would make a run slower there, so it is not done.
Where the drive cannot be identified, nothing changes.

This is only a cache warm-up. Whatever it misses or gets wrong, FFmpeg reads
from the disk as it always did, so the decoded samples are the same either
way, and any failure here is silent.
"""

from __future__ import annotations

import bisect
import functools
import os
import sys
import threading
from typing import Dict, List, Optional, Tuple

# Smaller files are read quickly however it is done.
MIN_FILE_BYTES = 256 * 1024 * 1024
# A stretch shorter than this is not worth a turn at the drive.
MIN_RANGE_BYTES = 16 * 1024 * 1024
# Large enough that a hard disk spends nearly all its time reading rather
# than seeking, small enough to stop promptly on cancel.
CHUNK_BYTES = 16 * 1024 * 1024
# The most of a file's head read while looking for its index.
HEAD_BYTES = 1024 * 1024

EBML = 0x1A45DFA3
SEGMENT = 0x18538067
SEEK_HEAD = 0x114D9B74
SEEK = 0x4DBB
SEEK_ID = 0x53AB
SEEK_POSITION = 0x53AC
INFO = 0x1549A966
TIMESTAMP_SCALE = 0x2AD7B1
CUES = 0x1C53BB6B
CUE_POINT = 0xBB
CUE_TIME = 0xB3
CUE_TRACK_POSITIONS = 0xB7
CUE_TRACK = 0xF7
CUE_CLUSTER_POSITION = 0xF1

_locks: Dict[int, threading.Lock] = {}
_locks_guard = threading.Lock()


def warm(path: str, first_s: float, last_s: float, token=None) -> None:
    """Read the part of ``path`` that holds ``first_s``..``last_s`` of its
    timeline (absolute timestamps, as FFmpeg's ``-copyts`` sees them) into the
    file cache, waiting for this drive's previous reader to finish first.

    ``AUDIOSYNC_PREFETCH`` set to 0 turns it off, and to 1 on for every drive.
    """
    setting = os.environ.get("AUDIOSYNC_PREFETCH", "auto")
    if setting == "0":
        return
    try:
        stat = os.stat(path)
        if stat.st_size < MIN_FILE_BYTES:
            return
        if setting != "1" and not _seeks_slowly(path, stat.st_dev):
            return
        index = _cue_index(path, stat.st_size, stat.st_mtime_ns)
        if not index:
            return
        span = _byte_range(index, first_s, last_s, stat.st_size)
        if span is None or span[1] - span[0] < MIN_RANGE_BYTES:
            return
        with _lock_for(stat.st_dev):
            _read(path, span[0], span[1], token)
    except Exception:  # noqa: BLE001 - a warm-up must never stop a decode
        return


_seek_penalty: Dict[int, Optional[bool]] = {}


def _seeks_slowly(path: str, device: int) -> bool:
    """Whether the drive holding ``path`` is a spinning disk, asked once per
    drive. False where it cannot be told."""
    with _locks_guard:
        if device in _seek_penalty:
            return bool(_seek_penalty[device])
    try:
        if sys.platform == "win32":
            answer = _windows_seek_penalty(path)
        elif sys.platform.startswith("linux"):
            answer = _linux_rotational(device)
        else:
            answer = None
    except Exception:  # noqa: BLE001 - a drive that cannot be asked is not a hard disk
        answer = None
    with _locks_guard:
        _seek_penalty[device] = answer
    return bool(answer)


def _windows_seek_penalty(path: str) -> Optional[bool]:
    """The drive's own answer to whether it pays for seeking -- what Windows
    asks to tell a hard disk from an SSD. None for a network share, a folder
    mount, or a drive that does not say (some USB enclosures)."""
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    root = ctypes.create_unicode_buffer(1024)
    if not kernel32.GetVolumePathNameW(ctypes.c_wchar_p(os.path.abspath(path)), root, 1024):
        return None
    if len(root.value) < 2 or root.value[1] != ":":
        return None

    create = kernel32.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    # No access rights asked for: querying a property needs none, and so no
    # administrator either. FILE_SHARE_READ | FILE_SHARE_WRITE, OPEN_EXISTING.
    handle = create("\\\\.\\" + root.value[:2], 0, 3, None, 3, 0, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        return None

    class Query(ctypes.Structure):
        _fields_ = [("PropertyId", wintypes.DWORD), ("QueryType", wintypes.DWORD),
                    ("AdditionalParameters", ctypes.c_ubyte * 1)]

    class SeekPenalty(ctypes.Structure):
        _fields_ = [("Version", wintypes.DWORD), ("Size", wintypes.DWORD),
                    ("IncursSeekPenalty", wintypes.BOOLEAN)]

    control = kernel32.DeviceIoControl
    control.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                        wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    control.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    try:
        # StorageDeviceSeekPenaltyProperty, PropertyStandardQuery;
        # IOCTL_STORAGE_QUERY_PROPERTY.
        query, answer, returned = Query(7, 0), SeekPenalty(), wintypes.DWORD(0)
        ok = control(handle, 0x2D1400, ctypes.byref(query), ctypes.sizeof(query),
                     ctypes.byref(answer), ctypes.sizeof(answer), ctypes.byref(returned), None)
        if not ok or returned.value < SeekPenalty.IncursSeekPenalty.offset + 1:
            return None
        return bool(answer.IncursSeekPenalty)
    finally:
        kernel32.CloseHandle(handle)


def _linux_rotational(device: int) -> Optional[bool]:
    """The kernel's flag for a spinning disk, on the partition's own disk."""
    partition = os.path.realpath(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}")
    for candidate in (partition, os.path.dirname(partition)):
        flag = os.path.join(candidate, "queue", "rotational")
        if os.path.exists(flag):
            with open(flag) as handle:
                return handle.read().strip() == "1"
    return None


def _lock_for(device: int) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(device)
        if lock is None:
            lock = _locks[device] = threading.Lock()
        return lock


def _read(path: str, start: int, end: int, token) -> None:
    buffer = bytearray(CHUNK_BYTES)
    view = memoryview(buffer)
    with open(path, "rb", buffering=0) as handle:
        handle.seek(start)
        remaining = end - start
        while remaining > 0:
            if token is not None and getattr(token, "cancelled", False):
                return
            got = handle.readinto(view[: min(CHUNK_BYTES, remaining)])
            if not got:
                return
            remaining -= got


# How far past the end of a window to read: audio is stored a little after
# the picture it plays with, and FFmpeg stops on the first packet beyond it.
END_MARGIN_S = 1.5


def _byte_range(index: List[Tuple[float, int]], first_s: float, last_s: float, size: int) -> Optional[Tuple[int, int]]:
    """From the cluster a seek to ``first_s`` lands on to just past ``last_s``.

    The seek lands on the cue point at or before ``first_s``; with several
    tracks indexed, on whichever of theirs is earliest. Cue points are
    seconds apart, so the end is placed between the two around ``last_s`` in
    proportion to time -- reading on to the next cue point instead was up to
    a fifth more than the window needed.
    """
    times = [time for time, _ in index]
    at = bisect.bisect_right(times, first_s)
    if at == 0:
        start = min(position for _, position in index)
    else:
        lands = times[at - 1]
        start = min(position for time, position in index[:at] if time >= lands - 1e-9)

    target = last_s + END_MARGIN_S
    after = bisect.bisect_right(times, target)
    if after >= len(index):
        end = size
    elif after == 0:
        end = index[0][1]
    else:
        (t0, p0), (t1, p1) = index[after - 1], index[after]
        end = p1 if t1 <= t0 else int(p0 + (p1 - p0) * (target - t0) / (t1 - t0))
    end = min(end, size)
    return (start, end) if end > start else None


@functools.lru_cache(maxsize=64)
def _cue_index(path: str, size: int, mtime_ns: int) -> Optional[List[Tuple[float, int]]]:
    """(time in seconds, absolute byte position of its cluster) for every cue
    point of a Matroska file, sorted by time; None for anything else, or a
    file whose index cannot be found from its head."""
    with open(path, "rb") as handle:
        head = handle.read(HEAD_BYTES)
        if len(head) < 4 or int.from_bytes(head[:4], "big") != EBML:
            return None
        pos = 0
        element, length, pos = _header(head, pos)
        pos += length
        element, _, pos = _header(head, pos)
        if element != SEGMENT:
            return None
        segment = pos

        scale = 1_000_000
        cues_at = None
        info_at = None
        # The top-level elements before the first cluster: the seek head and
        # the segment info are always among them.
        while pos < len(head) - 12:
            element, length, body = _header(head, pos)
            if element == SEEK_HEAD and body + length <= len(head):
                for seek_id, offset in _seek_entries(head[body: body + length]):
                    if seek_id == CUES:
                        cues_at = segment + offset
                    elif seek_id == INFO:
                        info_at = segment + offset
            elif element == INFO and body + length <= len(head):
                scale = _timestamp_scale(head[body: body + length]) or scale
                info_at = None
            elif element == CUES:
                cues_at = pos
                break
            elif element == 0x1F43B675:  # Cluster: the index is not up front
                break
            if length < 0:
                break
            pos = body + length

        if info_at is not None:
            handle.seek(info_at)
            block = handle.read(4096)
            element, length, body = _header(block, 0)
            if element == INFO and body + length <= len(block):
                scale = _timestamp_scale(block[body: body + length]) or scale
        if cues_at is None or cues_at >= size:
            return None
        handle.seek(cues_at)
        block = handle.read(16)
        element, length, body = _header(block, 0)
        if element != CUES or length <= 0 or length > 64 * 1024 * 1024:
            return None
        handle.seek(cues_at + body)
        cues = handle.read(length)

    index = []
    for element, start, end in _children(cues, 0, len(cues)):
        if element != CUE_POINT:
            continue
        time = None
        positions = []
        for child, child_start, child_end in _children(cues, start, end):
            if child == CUE_TIME:
                time = int.from_bytes(cues[child_start:child_end], "big")
            elif child == CUE_TRACK_POSITIONS:
                for grandchild, g_start, g_end in _children(cues, child_start, child_end):
                    if grandchild == CUE_CLUSTER_POSITION:
                        positions.append(int.from_bytes(cues[g_start:g_end], "big"))
        if time is not None:
            for position in positions:
                index.append((time * scale / 1e9, segment + position))
    index.sort()
    return index or None


def _header(data: bytes, pos: int) -> Tuple[int, int, int]:
    """(element id, body length or -1 if unknown, body start) at ``pos``."""
    first = data[pos]
    width = 1
    while width <= 4 and not first & (0x80 >> (width - 1)):
        width += 1
    if width > 4:
        raise ValueError("bad element id")
    element = int.from_bytes(data[pos: pos + width], "big")
    pos += width
    first = data[pos]
    size_width = 1
    while size_width <= 8 and not first & (0x80 >> (size_width - 1)):
        size_width += 1
    if size_width > 8:
        raise ValueError("bad element size")
    value = first & (0xFF >> size_width)
    for byte in data[pos + 1: pos + size_width]:
        value = (value << 8) | byte
    unknown = value == (1 << (7 * size_width)) - 1
    return element, (-1 if unknown else value), pos + size_width


def _children(data: bytes, start: int, end: int):
    pos = start
    while pos < end:
        element, length, body = _header(data, pos)
        if length < 0 or body + length > end:
            return
        yield element, body, body + length
        pos = body + length


def _seek_entries(data: bytes):
    for element, start, end in _children(data, 0, len(data)):
        if element != SEEK:
            continue
        seek_id = position = None
        for child, child_start, child_end in _children(data, start, end):
            if child == SEEK_ID:
                seek_id = int.from_bytes(data[child_start:child_end], "big")
            elif child == SEEK_POSITION:
                position = int.from_bytes(data[child_start:child_end], "big")
        if seek_id is not None and position is not None:
            yield seek_id, position


def _timestamp_scale(data: bytes) -> Optional[int]:
    for element, start, end in _children(data, 0, len(data)):
        if element == TIMESTAMP_SCALE:
            return int.from_bytes(data[start:end], "big") or None
    return None
