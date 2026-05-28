from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from dds_archive import BsaArchive, normalize_virtual_path


GAME_DATA_OWNER = "[Game Data]"


@dataclass
class DdsSourceProvider:
    display_name: str
    virtual_path: str
    source_kind: str
    owner: str = ""
    physical_path: str = ""
    archive_path: str = ""
    archive_name: str = ""
    data: bytes = b""
    is_game: bool = False

    @property
    def filename(self) -> str:
        source = self.virtual_path or self.physical_path or self.archive_name or self.display_name
        return source.replace("\\", "/").rsplit("/", 1)[-1]

    @property
    def location_text(self) -> str:
        if self.source_kind == "bsa":
            return f"{self.archive_path} :: {self.virtual_path}"
        return self.physical_path or self.virtual_path

    @property
    def is_manageable_loose_file(self) -> bool:
        return self.source_kind == "loose" and bool(self.owner) and bool(self.physical_path) and not self.is_game


@dataclass
class DdsSourceSet:
    providers: list[DdsSourceProvider]
    virtual_path: str
    current_index: int = 0

    def current_provider(self) -> Optional[DdsSourceProvider]:
        if 0 <= self.current_index < len(self.providers):
            return self.providers[self.current_index]
        return None


def normalize_data_path(path: str) -> str:
    return normalize_virtual_path(path)


def _display_name(mod_list, mod_name: str) -> str:
    try:
        value = mod_list.displayName(mod_name)
        if value:
            return str(value)
    except Exception:
        pass
    return mod_name


def _path_from_qdirish(value) -> Optional[Path]:
    try:
        if hasattr(value, "absolutePath"):
            return Path(value.absolutePath())
    except Exception:
        pass
    try:
        return Path(str(value))
    except Exception:
        return None


def _relative_to_base(base: Path, file_name: Path) -> str:
    try:
        return normalize_data_path(str(file_name.resolve().relative_to(base.resolve())))
    except Exception:
        return ""


def _mod_names_by_priority(mod_list) -> list[str]:
    for attr in ("allModsByProfilePriority", "allMods"):
        try:
            values = getattr(mod_list, attr)()
            if values:
                return [str(value) for value in values]
        except Exception:
            continue
    return []


def virtual_path_for(organizer, file_name: str) -> str:
    normalized = file_name.replace("\\", "/").strip()
    if not normalized:
        return ""

    if not os.path.isabs(normalized):
        return normalize_data_path(normalized)

    file_path = Path(normalized)
    if organizer is None:
        return _fallback_data_path(normalized)

    try:
        game = organizer.managedGame()
        game_data = _path_from_qdirish(game.dataDirectory())
        if game_data:
            relative = _relative_to_base(game_data, file_path)
            if relative:
                return relative
    except Exception:
        pass

    try:
        mod_list = organizer.modList()
        for mod_name in _mod_names_by_priority(mod_list):
            mod = mod_list.getMod(mod_name)
            if not mod:
                continue
            relative = _relative_to_base(Path(mod.absolutePath()), file_path)
            if relative:
                return relative
    except Exception:
        pass

    try:
        mods_path = Path(organizer.modsPath())
        relative = file_path.resolve().relative_to(mods_path.resolve())
        parts = relative.parts
        if len(parts) > 1:
            return normalize_data_path(str(Path(*parts[1:])))
    except Exception:
        pass

    return _fallback_data_path(normalized)


def _fallback_data_path(path: str) -> str:
    lower = path.lower().replace("\\", "/")
    for marker in ("/textures/", "textures/"):
        index = lower.find(marker)
        if index >= 0:
            start = index + (1 if marker.startswith("/") else 0)
            return normalize_data_path(path[start:])
    return normalize_data_path(path)


def order_origins_for_preview(origins: Iterable[str]) -> list[str]:
    ordered = [str(origin) for origin in origins]
    if len(ordered) > 1:
        ordered[1:] = reversed(ordered[1:])
    return ordered


def _same_file(lhs: str, rhs: str) -> bool:
    try:
        return Path(lhs).resolve().samefile(Path(rhs).resolve())
    except Exception:
        return str(Path(lhs)).lower() == str(Path(rhs)).lower()


def _has_provider(providers: list[DdsSourceProvider], provider: DdsSourceProvider) -> bool:
    for existing in providers:
        if existing.source_kind != provider.source_kind:
            continue
        if provider.source_kind == "loose" and existing.physical_path and provider.physical_path:
            if _same_file(existing.physical_path, provider.physical_path):
                return True
        elif provider.source_kind == "bsa":
            if (
                existing.archive_path.lower() == provider.archive_path.lower()
                and normalize_data_path(existing.virtual_path) == normalize_data_path(provider.virtual_path)
            ):
                return True
        elif provider.source_kind == "memory" and existing.data == provider.data:
            return True
    return False


def _add_provider(providers: list[DdsSourceProvider], provider: DdsSourceProvider) -> None:
    if not _has_provider(providers, provider):
        providers.append(provider)


def _loose_provider(mod_list, mod_name: str, mod, virtual_path: str) -> Optional[DdsSourceProvider]:
    try:
        candidate = Path(mod.absolutePath()) / Path(*virtual_path.split("/"))
    except Exception:
        return None
    if not candidate.exists() or not candidate.is_file():
        return None
    return DdsSourceProvider(
        display_name=_display_name(mod_list, mod_name),
        virtual_path=virtual_path,
        source_kind="loose",
        owner=mod_name,
        physical_path=str(candidate),
    )


def _archive_paths_from_mod_root(root: Path) -> list[Path]:
    try:
        return sorted(
            (path for path in root.rglob("*.bsa") if path.is_file()),
            key=lambda path: str(path).lower(),
        )
    except Exception:
        return []


def _feature_archive_names(organizer) -> list[str]:
    names: list[str] = []
    try:
        import mobase

        feature = organizer.gameFeatures().gameFeature(mobase.DataArchives)
        profile = organizer.profile()
        for value in list(feature.archives(profile)) + list(feature.vanillaArchives()):
            value = str(value)
            if value not in names:
                names.append(value)
    except Exception:
        pass
    return names


def _game_data_path(organizer) -> Optional[Path]:
    try:
        game = organizer.managedGame()
        return _path_from_qdirish(game.dataDirectory())
    except Exception:
        return None


def _archive_paths_from_game(organizer) -> list[Path]:
    game_data = _game_data_path(organizer)
    if not game_data:
        return []

    paths: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
            key = str(resolved).lower()
        except Exception:
            return
        if key in seen or not resolved.exists() or resolved.suffix.lower() != ".bsa":
            return
        seen.add(key)
        paths.append(resolved)

    for name in _feature_archive_names(organizer):
        candidate = Path(name)
        add(candidate if candidate.is_absolute() else game_data / candidate)

    try:
        for candidate in sorted(game_data.glob("*.bsa"), key=lambda path: str(path).lower()):
            add(candidate)
    except Exception:
        pass

    return paths


def _add_archive_provider(
    providers: list[DdsSourceProvider],
    display_name: str,
    owner: str,
    virtual_path: str,
    archive_path: Path,
    is_game: bool = False,
) -> None:
    archive = None
    try:
        archive = BsaArchive(archive_path)
        member = archive.find_member(virtual_path)
        if member is None:
            return
        data = archive.extract(member)
    except Exception:
        return
    finally:
        if archive is not None:
            archive.close()

    archive_name = archive_path.name
    _add_provider(
        providers,
        DdsSourceProvider(
            display_name=f"{display_name} - {archive_name}" if display_name else archive_name,
            virtual_path=normalize_data_path(virtual_path),
            source_kind="bsa",
            owner=owner,
            archive_path=str(archive_path),
            archive_name=archive_name,
            data=data,
            is_game=is_game,
        ),
    )


def _mod_exists(mod_list, mod_name: str) -> bool:
    try:
        import mobase

        return bool(mod_list.state(mod_name) & mobase.ModState.EXISTS)
    except Exception:
        return True


def _add_mod_archives(
    providers: list[DdsSourceProvider],
    mod_list,
    mod_name: str,
    mod,
    virtual_path: str,
) -> None:
    try:
        root = Path(mod.absolutePath())
    except Exception:
        return
    display_name = _display_name(mod_list, mod_name)
    for archive_path in _archive_paths_from_mod_root(root):
        _add_archive_provider(providers, display_name, mod_name, virtual_path, archive_path)


def _provider_index_for_path(providers: list[DdsSourceProvider], path: str) -> int:
    for index, provider in enumerate(providers):
        if provider.source_kind == "loose" and provider.physical_path and _same_file(provider.physical_path, path):
            return index
    return -1


def _provider_index_for_data(providers: list[DdsSourceProvider], data: bytes) -> int:
    if not data:
        return -1
    for index, provider in enumerate(providers):
        if provider.source_kind == "bsa" and provider.data == data:
            return index
    return -1


def resolve_dds_sources(organizer, file_name: str, file_data: bytes | None = None) -> DdsSourceSet:
    file_data = file_data or b""
    virtual_path = virtual_path_for(organizer, file_name)
    providers: list[DdsSourceProvider] = []

    if organizer is not None and virtual_path:
        try:
            mod_list = organizer.modList()
            origins = order_origins_for_preview(organizer.getFileOrigins(virtual_path))
            for mod_name in origins:
                mod = mod_list.getMod(mod_name)
                if not mod:
                    continue
                loose = _loose_provider(mod_list, mod_name, mod, virtual_path)
                if loose:
                    _add_provider(providers, loose)
                _add_mod_archives(providers, mod_list, mod_name, mod, virtual_path)

            for mod_name in _mod_names_by_priority(mod_list):
                if not _mod_exists(mod_list, mod_name):
                    continue
                mod = mod_list.getMod(mod_name)
                if not mod:
                    continue
                _add_mod_archives(providers, mod_list, mod_name, mod, virtual_path)
        except Exception:
            pass

        for archive_path in _archive_paths_from_game(organizer):
            _add_archive_provider(providers, GAME_DATA_OWNER, GAME_DATA_OWNER, virtual_path, archive_path, True)

    current_index = 0
    if file_data:
        match = _provider_index_for_data(providers, file_data)
        if match >= 0:
            current_index = match
        else:
            providers.insert(
                0,
                DdsSourceProvider(
                    display_name="Current Archive Preview",
                    virtual_path=virtual_path or normalize_data_path(file_name),
                    source_kind="memory",
                    physical_path=file_name,
                    data=file_data,
                ),
            )
            current_index = 0
    elif file_name and os.path.exists(file_name):
        match = _provider_index_for_path(providers, file_name)
        if match >= 0:
            current_index = match
        else:
            path = Path(file_name)
            providers.insert(
                0,
                DdsSourceProvider(
                    display_name=path.name,
                    virtual_path=virtual_path or normalize_data_path(path.name),
                    source_kind="loose",
                    physical_path=str(path),
                ),
            )
            current_index = 0

    if providers and not (0 <= current_index < len(providers)):
        current_index = 0

    return DdsSourceSet(providers=providers, virtual_path=virtual_path, current_index=current_index)


__all__ = [
    "DdsSourceProvider",
    "DdsSourceSet",
    "GAME_DATA_OWNER",
    "normalize_data_path",
    "order_origins_for_preview",
    "resolve_dds_sources",
    "virtual_path_for",
]
