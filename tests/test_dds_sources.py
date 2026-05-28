import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from dds_sources import resolve_dds_sources, virtual_path_for  # noqa: E402


def _write_bsa(path: Path, version: int, files: dict[str, bytes], compress: bool = False) -> None:
    grouped: dict[str, list[tuple[str, bytes]]] = {}
    for virtual_path, data in files.items():
        folder, name = virtual_path.rsplit("/", 1)
        grouped.setdefault(folder, []).append((name, data))

    folder_struct = struct.Struct("<QIIQ" if version == 105 else "<QII")
    flags = 0x1 | 0x2
    blocks: list[tuple[str, list[tuple[str, bytes, int]]]] = []
    names_blob = bytearray()
    folder_blocks_size = 0

    for folder, entries in grouped.items():
        stored_entries = []
        folder_bytes = folder.encode("utf-8") + b"\x00"
        folder_blocks_size += 1 + len(folder_bytes) + 16 * len(entries)
        for name, raw in entries:
            payload = struct.pack("<I", len(raw)) + zlib.compress(raw) if compress else raw
            raw_size = len(payload) | (0x40000000 if compress else 0)
            stored_entries.append((name, payload, raw_size))
            names_blob.extend(name.encode("utf-8") + b"\x00")
        blocks.append((folder, stored_entries))

    folder_records_size = folder_struct.size * len(blocks)
    data_offset = 36 + folder_records_size + folder_blocks_size + len(names_blob)
    payload_offset = data_offset
    folder_records = bytearray()
    folder_blocks = bytearray()
    payloads = bytearray()

    for folder, entries in blocks:
        if version == 105:
            folder_records.extend(folder_struct.pack(0, len(entries), 0, 0))
        else:
            folder_records.extend(folder_struct.pack(0, len(entries), 0))
        folder_bytes = folder.encode("utf-8") + b"\x00"
        folder_blocks.extend(bytes([len(folder_bytes)]) + folder_bytes)
        for _name, payload, raw_size in entries:
            folder_blocks.extend(struct.pack("<QII", 0, raw_size, payload_offset))
            payloads.extend(payload)
            payload_offset += len(payload)

    header = struct.pack(
        "<4s8I",
        b"BSA\x00",
        version,
        36,
        flags,
        len(blocks),
        len(files),
        sum(len(folder.encode("utf-8")) + 1 for folder in grouped),
        len(names_blob),
        0,
    )
    path.write_bytes(header + folder_records + folder_blocks + names_blob + payloads)


class FakeDirectory:
    def __init__(self, path: Path):
        self.path = path

    def absolutePath(self):
        return str(self.path)


class FakeGame:
    def __init__(self, data_path: Path):
        self.data_path = data_path

    def dataDirectory(self):
        return FakeDirectory(self.data_path)


class FakeMod:
    def __init__(self, path: Path):
        self.path = path

    def absolutePath(self):
        return str(self.path)


class FakeModList:
    def __init__(self, mods: dict[str, FakeMod], order: list[str]):
        self.mods = mods
        self.order = order

    def allModsByProfilePriority(self):
        return list(self.order)

    def allMods(self):
        return list(self.order)

    def displayName(self, name):
        return name

    def getMod(self, name):
        return self.mods.get(name)

    def state(self, _name):
        return 1


class FakeOrganizer:
    def __init__(self, root: Path, origins: list[str], mods: dict[str, FakeMod], order: list[str]):
        self.root = root
        self.origins = origins
        self.mod_list = FakeModList(mods, order)
        self.game_data = root / "Game" / "Data"
        self.game_data.mkdir(parents=True, exist_ok=True)

    def modList(self):
        return self.mod_list

    def getFileOrigins(self, _virtual_path):
        return list(self.origins)

    def modsPath(self):
        return str(self.root / "mods")

    def managedGame(self):
        return FakeGame(self.game_data)

    def gameFeatures(self):
        raise RuntimeError("No game archive feature in unit tests")


class DdsSourceResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "mods").mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _make_mod(self, name: str) -> FakeMod:
        path = self.root / "mods" / name
        path.mkdir(parents=True)
        return FakeMod(path)

    def test_virtual_path_and_loose_ordering(self) -> None:
        virtual = "textures/demo/glow.dds"
        winner = self._make_mod("Winner")
        loser = self._make_mod("Loser")
        for mod, payload in ((winner, b"winner"), (loser, b"loser")):
            file_path = Path(mod.absolutePath()) / Path(*virtual.split("/"))
            file_path.parent.mkdir(parents=True)
            file_path.write_bytes(payload)

        organizer = FakeOrganizer(
            self.root,
            origins=["Winner", "Loser"],
            mods={"Winner": winner, "Loser": loser},
            order=["Winner", "Loser"],
        )
        loser_path = str(Path(loser.absolutePath()) / Path(*virtual.split("/")))

        result = resolve_dds_sources(organizer, loser_path, b"")

        self.assertEqual(virtual_path_for(organizer, loser_path), virtual)
        self.assertEqual([p.owner for p in result.providers], ["Winner", "Loser"])
        self.assertEqual(result.current_index, 1)

    def test_bsa_provider_and_current_archive_byte_match(self) -> None:
        virtual = "textures/demo/core.dds"
        data = b"dds bytes from archive"
        archive_mod = self._make_mod("ArchiveMod")
        _write_bsa(Path(archive_mod.absolutePath()) / "ArchiveAssets.bsa", 105, {virtual: data})
        organizer = FakeOrganizer(
            self.root,
            origins=[],
            mods={"ArchiveMod": archive_mod},
            order=["ArchiveMod"],
        )

        result = resolve_dds_sources(organizer, virtual, data)

        self.assertEqual(len(result.providers), 1)
        self.assertEqual(result.providers[0].source_kind, "bsa")
        self.assertEqual(result.providers[0].data, data)
        self.assertEqual(result.current_index, 0)

    def test_unmatched_archive_preview_falls_back_to_memory_provider(self) -> None:
        virtual = "textures/demo/missing.dds"
        data = b"opened from an archive MO2 did not expose"
        organizer = FakeOrganizer(self.root, origins=[], mods={}, order=[])

        result = resolve_dds_sources(organizer, virtual, data)

        self.assertEqual(len(result.providers), 1)
        self.assertEqual(result.providers[0].source_kind, "memory")
        self.assertEqual(result.providers[0].display_name, "Current Archive Preview")

    def test_corrupt_archive_is_ignored(self) -> None:
        virtual = "textures/demo/corrupt.dds"
        archive_mod = self._make_mod("BrokenArchive")
        (Path(archive_mod.absolutePath()) / "Broken.bsa").write_bytes(b"BSA\x00")
        organizer = FakeOrganizer(
            self.root,
            origins=[],
            mods={"BrokenArchive": archive_mod},
            order=["BrokenArchive"],
        )

        result = resolve_dds_sources(organizer, virtual, b"")

        self.assertEqual(result.providers, [])


if __name__ == "__main__":
    unittest.main()
