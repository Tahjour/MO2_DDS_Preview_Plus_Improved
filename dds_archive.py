from __future__ import annotations

import ctypes
import mmap
import os
import struct
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_BSA_HEADER = struct.Struct("<4s8I")
_BSA_FOLDER_104 = struct.Struct("<QII")
_BSA_FOLDER_105 = struct.Struct("<QIIQ")
_BSA_FILE = struct.Struct("<QII")
_FLAG_DIRECTORY_NAMES = 0x1
_FLAG_FILE_NAMES = 0x2
_FLAG_COMPRESSED = 0x4
_FLAG_EMBED_FILE_NAMES = 0x100
_FILE_COMPRESSION_TOGGLE = 0x40000000
_FILE_SIZE_MASK = 0x3FFFFFFF

_INDEX_CACHE: dict[tuple[str, int, int], "BsaIndex"] = {}
_INDEX_CACHE_LOCK = threading.Lock()
_LZ4_CACHE: dict[str, object] = {}


def normalize_virtual_path(path: str) -> str:
    return path.replace("\\", "/").strip("/").lower()


@dataclass(frozen=True)
class BsaMember:
    virtual_path: str
    offset: int
    stored_size: int
    compressed: bool


@dataclass(frozen=True)
class BsaIndex:
    path: Path
    version: int
    archive_flags: int
    members: tuple[BsaMember, ...]


def _fingerprint(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()).lower(), stat.st_size, stat.st_mtime_ns)


def _read_bzstring(data: memoryview, pos: int) -> tuple[str, int]:
    if pos >= len(data):
        raise ValueError("BSA directory name is truncated")
    length = data[pos]
    pos += 1
    end = pos + length
    if end > len(data):
        raise ValueError("BSA directory name is truncated")
    value = bytes(data[pos:end]).rstrip(b"\x00").decode("utf-8", errors="ignore")
    return value, end


def _read_cstring(data: memoryview, pos: int) -> tuple[str, int]:
    end = pos
    while end < len(data) and data[end] != 0:
        end += 1
    if end >= len(data):
        raise ValueError("BSA filename table is truncated")
    value = bytes(data[pos:end]).decode("utf-8", errors="ignore")
    return value, end + 1


def _parse_bsa_index_data(path: Path, data: memoryview) -> BsaIndex:
    if len(data) < _BSA_HEADER.size:
        raise ValueError("Archive is smaller than a BSA header")

    (
        magic,
        version,
        folder_offset,
        archive_flags,
        folder_count,
        file_count,
        _folder_name_length,
        _file_name_length,
        _file_flags,
    ) = _BSA_HEADER.unpack_from(data, 0)

    if magic != b"BSA\x00":
        raise ValueError("Not a BSA archive")
    if version not in (104, 105):
        raise ValueError(f"Unsupported BSA version: {version}")
    if not (archive_flags & _FLAG_FILE_NAMES):
        raise ValueError("BSA archives without filenames are unsupported")

    folder_struct = _BSA_FOLDER_105 if version == 105 else _BSA_FOLDER_104
    pos = folder_offset
    folder_file_counts: list[int] = []
    for _ in range(folder_count):
        if pos + folder_struct.size > len(data):
            raise ValueError("BSA folder records are truncated")
        values = folder_struct.unpack_from(data, pos)
        folder_file_counts.append(values[1])
        pos += folder_struct.size

    raw_records: list[tuple[str, int, int]] = []
    for count in folder_file_counts:
        folder_name = ""
        if archive_flags & _FLAG_DIRECTORY_NAMES:
            folder_name, pos = _read_bzstring(data, pos)
        for _ in range(count):
            if pos + _BSA_FILE.size > len(data):
                raise ValueError("BSA file records are truncated")
            _name_hash, raw_size, file_offset = _BSA_FILE.unpack_from(data, pos)
            raw_records.append((folder_name, raw_size, file_offset))
            pos += _BSA_FILE.size

    if len(raw_records) != file_count:
        raise ValueError("BSA file count does not match folder records")

    names: list[str] = []
    for _ in range(file_count):
        name, pos = _read_cstring(data, pos)
        names.append(name)

    default_compressed = bool(archive_flags & _FLAG_COMPRESSED)
    members: list[BsaMember] = []
    for name, (folder, raw_size, file_offset) in zip(names, raw_records):
        virtual_path = normalize_virtual_path(f"{folder}/{name}" if folder else name)
        if not virtual_path:
            continue
        compressed = default_compressed ^ bool(raw_size & _FILE_COMPRESSION_TOGGLE)
        members.append(
            BsaMember(
                virtual_path=virtual_path,
                offset=file_offset,
                stored_size=raw_size & _FILE_SIZE_MASK,
                compressed=compressed,
            )
        )

    return BsaIndex(path=path, version=version, archive_flags=archive_flags, members=tuple(members))


def _parse_bsa_index(path: Path) -> BsaIndex:
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            data = memoryview(mapped)
            try:
                return _parse_bsa_index_data(path, data)
            finally:
                data.release()


def _get_bsa_index(path: Path) -> BsaIndex:
    key = _fingerprint(path)
    with _INDEX_CACHE_LOCK:
        cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    index = _parse_bsa_index(path)
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE[key] = index
    return index


def _load_lz4(explicit_path: Optional[Path] = None):
    candidates: list[Path | str] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    env_path = os.environ.get("MO2_LZ4_PATH")
    if env_path:
        candidates.append(Path(env_path))
    try:
        candidates.append(Path(__file__).resolve().parent / "dlls" / "liblz4.dll")
    except IndexError:
        pass
    candidates.extend(["liblz4.dll", "lz4.dll"])

    errors: list[str] = []
    for candidate in candidates:
        key = str(candidate)
        if key in _LZ4_CACHE:
            return _LZ4_CACHE[key]
        try:
            lib = ctypes.CDLL(key)
            lib.LZ4_decompress_safe.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.LZ4_decompress_safe.restype = ctypes.c_int
            lib.LZ4F_createDecompressionContext.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint,
            ]
            lib.LZ4F_createDecompressionContext.restype = ctypes.c_size_t
            lib.LZ4F_freeDecompressionContext.argtypes = [ctypes.c_void_p]
            lib.LZ4F_freeDecompressionContext.restype = ctypes.c_size_t
            lib.LZ4F_decompress.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_void_p,
            ]
            lib.LZ4F_decompress.restype = ctypes.c_size_t
            lib.LZ4F_isError.argtypes = [ctypes.c_size_t]
            lib.LZ4F_isError.restype = ctypes.c_uint
            lib.LZ4F_getErrorName.argtypes = [ctypes.c_size_t]
            lib.LZ4F_getErrorName.restype = ctypes.c_char_p
            _LZ4_CACHE[key] = lib
            return lib
        except (OSError, AttributeError) as exc:
            errors.append(f"{candidate}: {exc}")

    raise RuntimeError("Could not load liblz4 for BSA decompression: " + "; ".join(errors))


def _lz4_error(lib, code: int) -> str:
    if not lib.LZ4F_isError(code):
        return ""
    name = lib.LZ4F_getErrorName(code)
    return name.decode("ascii", errors="ignore") if name else "unknown LZ4 error"


def _decompress_lz4_frame(packed: bytes, expected_size: int, lib) -> bytes:
    context = ctypes.c_void_p()
    result = lib.LZ4F_createDecompressionContext(ctypes.byref(context), 100)
    error = _lz4_error(lib, result)
    if error:
        raise ValueError(f"LZ4 context creation failed: {error}")

    source = ctypes.create_string_buffer(packed)
    output = bytearray()
    source_pos = 0
    try:
        while True:
            remaining = max(expected_size - len(output), 1)
            target = ctypes.create_string_buffer(remaining)
            target_size = ctypes.c_size_t(remaining)
            source_size = ctypes.c_size_t(len(packed) - source_pos)
            source_ptr = ctypes.cast(ctypes.byref(source, source_pos), ctypes.c_void_p)
            result = lib.LZ4F_decompress(
                context,
                target,
                ctypes.byref(target_size),
                source_ptr,
                ctypes.byref(source_size),
                None,
            )
            error = _lz4_error(lib, result)
            if error:
                raise ValueError(f"LZ4 decompression failed: {error}")
            output.extend(target.raw[: target_size.value])
            source_pos += source_size.value
            if result == 0:
                break
            if source_size.value == 0 and target_size.value == 0:
                raise ValueError("LZ4 decompression made no progress")
    finally:
        lib.LZ4F_freeDecompressionContext(context)

    if len(output) != expected_size:
        raise ValueError("LZ4 decompression size mismatch")
    return bytes(output)


class BsaArchive:
    def __init__(self, path: Path, lz4_path: Optional[Path] = None):
        self.path = Path(path)
        self._lz4_path = lz4_path
        self._index: Optional[BsaIndex] = None
        self._by_path: Optional[dict[str, BsaMember]] = None
        self._handle = None
        self._read_lock = threading.Lock()

    @property
    def index(self) -> BsaIndex:
        if self._index is None:
            self._index = _get_bsa_index(self.path)
        return self._index

    @property
    def members(self) -> tuple[BsaMember, ...]:
        return self.index.members

    def find_member(self, virtual_path: str) -> Optional[BsaMember]:
        if self._by_path is None:
            self._by_path = {member.virtual_path: member for member in self.members}
        return self._by_path.get(normalize_virtual_path(virtual_path))

    def extract(self, member: BsaMember) -> bytes:
        with self._read_lock:
            if self._handle is None or self._handle.closed:
                self._handle = self.path.open("rb")
            self._handle.seek(member.offset)
            payload = self._handle.read(member.stored_size)
        if len(payload) != member.stored_size:
            raise ValueError(f"Truncated BSA member: {member.virtual_path}")

        if self.index.archive_flags & _FLAG_EMBED_FILE_NAMES:
            if not payload:
                raise ValueError(f"Missing embedded name: {member.virtual_path}")
            name_size = payload[0] + 1
            payload = payload[name_size:]

        if not member.compressed:
            return payload
        if len(payload) < 4:
            raise ValueError(f"Missing uncompressed size: {member.virtual_path}")

        expected_size = struct.unpack_from("<I", payload, 0)[0]
        packed = payload[4:]
        if self.index.version == 105:
            lib = _load_lz4(self._lz4_path)
            if packed.startswith(b"\x04\x22\x4d\x18"):
                return _decompress_lz4_frame(packed, expected_size, lib)
            out = ctypes.create_string_buffer(expected_size)
            source = ctypes.create_string_buffer(packed)
            result = lib.LZ4_decompress_safe(source, out, len(packed), expected_size)
            if result < 0 or result != expected_size:
                raise ValueError(f"LZ4 decompression failed: {member.virtual_path}")
            return out.raw[:result]

        raw = zlib.decompress(packed)
        if len(raw) != expected_size:
            raise ValueError(f"Zlib size mismatch: {member.virtual_path}")
        return raw

    def close(self) -> None:
        with self._read_lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "BsaArchive",
    "BsaIndex",
    "BsaMember",
    "normalize_virtual_path",
]
