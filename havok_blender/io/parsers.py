"""Havok importer utilities for skeletons and animations.

The helper functions in this module focus on Havok XML packfiles generated
by hkxpack/hkcmd and the IGZ/PAK wrappers commonly used by Alchemy games.
They intentionally avoid placeholder logic and instead build real transform
tracks for Blender armatures when data is present.
"""
from __future__ import annotations

import gzip
import lzma
import tarfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

import mathutils

SUPPORTED_EXTENSIONS = {".hkx", ".hka", ".hkt", ".igz", ".pak"}

# Platform-aware PAK parsing: users pick the game version (layout profile)
# and target platform endianness instead of the importer guessing.
PAK_PLATFORM_ENDIANNESS = {
    "little": "Little endian (PC / Xbox 360 / Xbox One / Wii U)",
    "big": "Big endian (PS3 / Wii)",
}

_PAK_PROFILES = [
    {
        "name": "SSA_WII",
        "version": 0x04,
        "chunk_alignment_offset": 0x10,
        "chunk_alignment_override": 0x800,
        "layout": {
            "num_files": 0x0C,
            "name_loc": 0x18,
            "name_len": 0x1C,
            "local_header_len": 0x0C,
            "checksum_loc": 0x30,
            "checksum_len": 0x04,
            "file_start_in_local": 0x00,
            "file_size_in_local": 0x04,
            "mode_in_local": 0x08,
        },
    },
    {
        "name": "SSA_WIIU",
        "version": 0x08,
        "chunk_alignment_offset": 0x10,
        "layout": {
            "num_files": 0x0C,
            "name_loc": 0x1C,
            "name_len": 0x20,
            "local_header_len": 0x0C,
            "checksum_loc": 0x34,
            "checksum_len": 0x04,
            "file_start_in_local": 0x00,
            "file_size_in_local": 0x04,
            "mode_in_local": 0x08,
        },
    },
    {
        "name": "SWAP_FORCE",
        "version": 0x0A,
        "chunk_alignment_offset": 0x10,
        "layout": {
            "num_files": 0x0C,
            "name_loc": 0x2C,
            "name_len": 0x30,
            "local_header_len": 0x10,
            "checksum_loc": 0x38,
            "checksum_len": 0x04,
            "file_start_in_local": 0x04,
            "file_size_in_local": 0x08,
            "mode_in_local": 0x0C,
        },
    },
    {
        "name": "LOST_ISLANDS",
        "version": 0x0A,
        "chunk_alignment_offset": 0x10,
        "layout": {
            "num_files": 0x0C,
            "name_loc": 0x28,
            "name_len": 0x30,
            "local_header_len": 0x10,
            "checksum_loc": 0x38,
            "checksum_len": 0x04,
            "file_start_in_local": 0x00,
            "file_size_in_local": 0x08,
            "mode_in_local": 0x0C,
        },
    },
    {
        "name": "TRAP_TEAM",
        "version": 0x0B,
        "chunk_alignment_offset": 0x10,
        "layout": {
            "num_files": 0x0C,
            "name_loc": 0x28,
            "name_len": 0x30,
            "local_header_len": 0x10,
            "checksum_loc": 0x38,
            "checksum_len": 0x04,
            "file_start_in_local": 0x00,
            "file_size_in_local": 0x08,
            "mode_in_local": 0x0C,
        },
    },
    {
        "name": "SUPER_CHARGERS",
        "version": 0x0B,
        "chunk_alignment_offset": 0x10,
        "layout": {
            "num_files": 0x0C,
            "name_loc": 0x2C,
            "name_len": 0x30,
            "local_header_len": 0x10,
            "checksum_loc": 0x38,
            "checksum_len": 0x04,
            "file_start_in_local": 0x04,
            "file_size_in_local": 0x08,
            "mode_in_local": 0x0C,
        },
    },
    {
        "name": "IMAGINATORS",
        "version": 0x0B,
        "chunk_alignment_offset": 0x10,
        "layout": {
            "num_files": 0x0C,
            "name_loc": 0x28,
            "name_len": 0x30,
            "local_header_len": 0x10,
            "checksum_loc": 0x38,
            "checksum_len": 0x04,
            "file_start_in_local": 0x00,
            "file_size_in_local": 0x08,
            "mode_in_local": 0x0C,
        },
    },
    {
        "name": "CRASH_NST",
        "version": 0x0C,
        "chunk_alignment_offset": 0x10,
        "layout": {
            "num_files": 0x0C,
            "name_loc": 0x24,
            "name_len": 0x2C,
            "local_header_len": 0x10,
            "checksum_loc": 0x38,
            "checksum_len": 0x04,
            "file_start_in_local": 0x00,
            "file_size_in_local": 0x08,
            "mode_in_local": 0x0C,
        },
    },
]

PAK_PROFILE_NAMES = tuple(profile["name"] for profile in _PAK_PROFILES)


@dataclass
class HavokBone:
    name: str
    parent: int
    translation: mathutils.Vector
    rotation: mathutils.Quaternion


@dataclass
class HavokSkeleton:
    name: str
    bones: List[HavokBone]


@dataclass
class HavokAnimation:
    name: str
    duration: float
    tracks: List[List[Tuple[mathutils.Vector, mathutils.Quaternion]]]
    track_to_bone: List[int]


@dataclass
class HavokPack:
    skeleton: Optional[HavokSkeleton]
    animations: List[HavokAnimation]
    meshes: List["HavokMesh"]


@dataclass
class HavokMesh:
    name: str
    vertices: List[mathutils.Vector]
    faces: List[Tuple[int, int, int]]


@dataclass
class PakEntry:
    name: str
    offset: int
    size: int
    mode: int
    endianness: str
    version: int
    chunk_alignment: int
    size_field: int
    size_endianness: str


class _IgzStream:
    """Lightweight binary reader inspired by io_scene_igz's NoeBitStream."""

    def __init__(self, data: bytes, endianness: str):
        self.data = data
        self.order = "little" if endianness == "little" else "big"

    def u16(self, offset: int) -> int:
        return int.from_bytes(self.data[offset : offset + 2], self.order)

    def u32(self, offset: int) -> int:
        return int.from_bytes(self.data[offset : offset + 4], self.order)

    def u64(self, offset: int) -> int:
        return int.from_bytes(self.data[offset : offset + 8], self.order)


class _IgzExtractor:
    """Parse IGZ fixup tables to recover embedded Havok payloads.

    This mirrors the layout walking in NewSkyLine-dev/io_scene_igz without
    pulling in the full model importer. It focuses on the fixup sections that
    reference arbitrary memory blobs (NHMT) so we can scan them for Havok XML
    or compressed packfiles.
    """

    def __init__(self, data: bytes):
        self.data = data
        self.blocks: List[bytes] = []
        self.endianness = "little"
        self.version = 0
        self.pointers: List[int] = []

    def parse(self) -> List[bytes]:
        if len(self.data) < 0x20:
            return []

        magic = self.data[:4]
        if magic == b"\x01ZGI":
            self.endianness = "little"
        elif magic == b"IGZ\x01":
            self.endianness = "big"
        else:
            return []

        stream = _IgzStream(self.data, self.endianness)
        self.version = stream.u32(4)

        pointer_start = 0x18 if self.version >= 0x07 else 0x10
        num_fixups = stream.u32(0x14) if self.version >= 0x07 else -1

        for idx in range(0x20):
            ptr = stream.u32(pointer_start + idx * 0x10)
            if ptr == 0:
                break
            self.pointers.append(ptr)

        if not self.pointers:
            return []

        fixup_start = self.pointers[0]
        if self.version <= 0x06:
            platform = stream.u16(fixup_start + 0x08)
            _ = platform  # platform is not needed for extraction
            num_fixups = stream.u32(fixup_start + 0x10)
            fixup_start += 0x1C

        cursor = fixup_start
        for _ in range(max(num_fixups, 0)):
            if cursor + 0x10 > len(self.data):
                break
            magic = stream.u32(cursor)
            local_cursor = cursor + (0x0C if self.version <= 0x06 else 0x04)
            count = stream.u32(local_cursor)
            length = stream.u32(local_cursor + 4)
            data_start = stream.u32(local_cursor + 8)
            payload_cursor = cursor + data_start

            if magic in (0x4E484D54, 10):
                self._read_memory_blocks(stream, payload_cursor, count)

            cursor += length if length > 0 else 0x10

        return self.blocks

    def _fix_pointer(self, pointer: int) -> int:
        if pointer & 0x80000000:
            return -1
        if self.version <= 0x06:
            base_index = (pointer >> 0x18) + 1
            offset = pointer & 0x00FFFFFF
        else:
            base_index = (pointer >> 0x1B) + 1
            offset = pointer & 0x07FFFFFF
        if base_index >= len(self.pointers):
            return -1
        return self.pointers[base_index] + offset

    def _read_pointer(self, stream: _IgzStream, offset: int) -> Tuple[int, int]:
        ptr = stream.u32(offset)
        return self._fix_pointer(ptr), 4

    def _read_memory_blocks(self, stream: _IgzStream, cursor: int, count: int) -> None:
        for _ in range(count):
            if cursor + 8 > len(self.data):
                break
            size = stream.u32(cursor) & 0x00FFFFFF
            pointer_offset, consumed = self._read_pointer(stream, cursor + 4)
            cursor += 4 + consumed
            if pointer_offset == -1 or size <= 0:
                continue
            if pointer_offset + size > len(self.data):
                continue
            self.blocks.append(self.data[pointer_offset : pointer_offset + size])


def load_from_path(
    path: Path,
    entry: Optional[str] = None,
    pak_profile: Optional[str] = None,
    pak_platform: Optional[str] = None,
) -> HavokPack:
    """Load any supported Havok source from disk.

    Args:
        path: file path provided by the user.
        entry: optional archive entry for PAK/ZIP containers.
    """

    suffix = path.suffix.lower()
    if suffix == ".pak":
        data = _extract_from_archive(path, entry, pak_profile, pak_platform)
    else:
        data = path.read_bytes()

    return parse_bytes(data, override_name=path.stem)


def parse_bytes(data: bytes, override_name: Optional[str] = None) -> HavokPack:
    """Parse Havok XML/IGZ data into skeleton and animation structures."""

    xml_bytes = _unwrap_bytes(data)
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Unsupported or corrupt Havok payload") from exc

    skeleton = _parse_skeleton(root, override_name)
    animations = _parse_animations(root, skeleton)
    meshes = _parse_meshes(root, override_name)
    return HavokPack(skeleton=skeleton, animations=animations, meshes=meshes)


def _unwrap_bytes(data: bytes) -> bytes:
    # IGZ and some Havok distributions are gzip-compressed.
    if data.startswith(b"\x1f\x8b"):
        return gzip.decompress(data)

    igz_payload = _maybe_from_igz(data)
    if igz_payload is not None:
        return igz_payload

    embedded = _slice_embedded_havok(data)
    if embedded is not None:
        return embedded

    # Tar/zip payloads are considered higher-level PAK containers; they should
    # be handled by _extract_from_archive instead.
    return data


def _extract_from_archive(
    path: Path, entry: Optional[str], pak_profile: Optional[str], pak_platform: Optional[str]
) -> bytes:
    # Try ZIP style first (many PAK files are simple zips).
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            choices = _filter_havok_entries(zf.namelist())
            target = _resolve_entry(choices, entry)
            with zf.open(target, "r") as handle:
                return handle.read()

    # Try TAR style containers.
    try:
        with tarfile.open(path, "r:*") as tf:
            choices = _filter_havok_entries(tf.getnames())
            target = _resolve_entry(choices, entry)
            member = tf.getmember(target)
            return tf.extractfile(member).read()  # type: ignore[arg-type]
    except tarfile.TarError:
        pass

    if pak_profile is not None and pak_platform is not None:
        pak_entries = _read_pak_entries(path, pak_profile, pak_platform)
        if pak_entries:
            entry_map = {p.name: p for p in pak_entries}
            target_name = entry or (next((name for name in entry_map if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS), None))
            if target_name is None:
                target_name = pak_entries[0].name
            if target_name not in entry_map:
                raise ValueError(f"Entry '{target_name}' not found in PAK; options: {sorted(entry_map.keys())}")
            return _decode_pak_entry(path.read_bytes(), entry_map[target_name], entry_map, pak_entries)
    elif pak_profile is not None or pak_platform is not None:
        raise ValueError("Both PAK game version and platform must be selected for .pak imports")

    # As a fallback, treat the PAK as a raw blob and attempt to pull embedded
    # Havok XML from it. This mirrors the lightweight extraction implemented in
    # the io_scene_igz and pak-importer community tools.
    raw = path.read_bytes()
    maybe_payload = _slice_embedded_havok(raw)
    if maybe_payload:
        return maybe_payload

    raise ValueError("PAK container is not a zip/tar archive or lacks Havok data")


def _maybe_from_igz(data: bytes) -> Optional[bytes]:
    """Attempt to extract Havok XML out of an IGZ container.

    IGZ files used by Alchemy titles wrap their assets with a small header and
    one or more fixup-defined memory blocks. We mirror the io_scene_igz
    importer: parse the fixup table, resolve NHMT memory references, then scan
    each block for Havok XML or compressed data.
    """

    if not (data.startswith(b"\x01ZGI") or data.startswith(b"IGZ\x01")):
        return None

    extractor = _IgzExtractor(data)
    blocks = extractor.parse()
    candidates = list(blocks)
    # Some IGZ bundles chunk the Havok payload across sequential blocks; the
    # io_scene_igz importer concatenates those buffers before scanning, so mimic
    # that behavior here instead of assuming each block is self-contained.
    if blocks:
        candidates.append(b"".join(blocks))
    candidates.append(data)

    for blob in candidates:
        # Inline gzip or zlib members.
        if blob.startswith(b"\x1f\x8b"):
            try:
                return gzip.decompress(blob)
            except OSError:
                pass
        for sig in (b"\x78\x9c", b"\x78\x01", b"\x78\xda"):
            if blob.startswith(sig):
                try:
                    return zlib.decompress(blob)
                except zlib.error:
                    pass

        sliced = _slice_embedded_havok(blob)
        if sliced:
            return sliced

    return None


def _slice_embedded_havok(blob: bytes) -> Optional[bytes]:
    """Extract an XML Havok packfile stored inside a larger binary blob.

    Both IGZ and some bespoke PAK bundles store the hkx XML verbatim but with a
    prefix containing metatables or table-of-contents data. This helper finds
    the first XML marker and returns the rest of the stream.
    """

    markers = (b"<hkpackfile", b"<hkobject", b"<?xml")
    offsets = [blob.find(marker) for marker in markers if marker in blob]
    offsets = [o for o in offsets if o >= 0]
    if not offsets:
        return None
    start = min(offsets)
    return blob[start:]


def _filter_havok_entries(names: Iterable[str]) -> List[str]:
    filtered = [n for n in names if Path(n).suffix.lower() in SUPPORTED_EXTENSIONS]
    if not filtered:
        raise ValueError("Archive does not contain any Havok-compatible entries")
    return sorted(filtered)


def _resolve_entry(choices: List[str], requested: Optional[str]) -> str:
    if requested:
        if requested not in choices:
            raise ValueError(f"Entry '{requested}' not found in archive; options: {choices}")
        return requested
    return choices[0]


def _read_uint(data: bytes, offset: int, endianness: str) -> int:
    order = "little" if endianness == "little" else "big"
    return int.from_bytes(data[offset : offset + 4], order)


def _align(value: int, alignment: int) -> int:
    if alignment <= 0:
        return value
    return ((value + alignment - 1) // alignment) * alignment


def _try_layout(data: bytes, profile: Dict[str, object], endianness: str) -> Optional[List[PakEntry]]:
    layout: Dict[str, int] = profile["layout"]  # type: ignore[assignment]
    version: int = int(profile["version"])  # type: ignore[index]
    size_field = 2 if (version & 0xFF) <= 0x0B else 4
    size_endianness = "big" if (version & 0xFF) <= 0x04 else "little"

    num_files = _read_uint(data, layout["num_files"], endianness)
    if num_files <= 0 or num_files > 0xFFFF:
        return None

    nametable_loc = _read_uint(data, layout["name_loc"], endianness)
    nametable_len = _read_uint(data, layout["name_len"], endianness)
    if nametable_loc + nametable_len > len(data):
        return None

    alignment_offset = int(profile.get("chunk_alignment_offset", 0x10))
    chunk_alignment = int(profile.get("chunk_alignment_override") or _read_uint(data, alignment_offset, endianness))
    if chunk_alignment <= 0:
        chunk_alignment = 0x8000

    names: List[str] = []
    for idx in range(num_files):
        offset_ptr = nametable_loc + 4 * idx
        if offset_ptr + 4 > len(data):
            return None
        name_offset = _read_uint(data, offset_ptr, endianness)
        start = nametable_loc + name_offset
        if start >= nametable_loc + nametable_len:
            return None
        end = data.find(b"\x00", start, nametable_loc + nametable_len)
        if end == -1:
            return None
        try:
            names.append(data[start:end].decode("utf-8", errors="ignore"))
        except UnicodeDecodeError:
            return None

    checksum_loc = layout["checksum_loc"]
    checksum_len = layout["checksum_len"]
    local_header_len = layout["local_header_len"]
    file_start_in_local = layout["file_start_in_local"]
    file_size_in_local = layout["file_size_in_local"]
    mode_in_local = layout["mode_in_local"]

    base_header = checksum_loc + checksum_len * num_files
    if base_header + local_header_len * num_files > len(data):
        return None

    entries: List[PakEntry] = []
    for idx in range(num_files):
        header_base = base_header + local_header_len * idx
        start = _read_uint(data, header_base + file_start_in_local, endianness)
        size = _read_uint(data, header_base + file_size_in_local, endianness)
        mode = _read_uint(data, header_base + mode_in_local, endianness)
        # Guard against bogus offsets or overflowed spans. Using a difference
        # check avoids the possibility of start + size wrapping past len(data)
        # on unusually large inputs.
        if start < 0 or size <= 0 or start > len(data) or size > len(data) - start:
            return None
        entries.append(
            PakEntry(
                name=names[idx],
                offset=start,
                size=size,
                mode=mode,
                endianness=endianness,
                version=version,
                chunk_alignment=chunk_alignment,
                size_field=size_field,
                size_endianness=size_endianness,
            )
        )

    return entries


def _read_pak_entries(path: Path, profile_name: str, endianness: str) -> List[PakEntry]:
    data = path.read_bytes()
    if len(data) < 0x40:
        return []

    magic = data[:4]
    if magic not in (b"\x1AAGI", b"IGA\x1A"):
        return []

    profile = next((p for p in _PAK_PROFILES if p["name"] == profile_name), None)
    if profile is None:
        raise ValueError(f"Unknown PAK profile '{profile_name}'")

    if endianness not in ("little", "big"):
        raise ValueError("PAK platform endianness must be 'little' or 'big'")

    entries = _try_layout(data, profile, endianness)
    return entries or []


def enumerate_pak_entries(path: Path, profile_name: str, endianness: str) -> List[PakEntry]:
    """Return parsed PAK entries for UI listing using user-selected layout/platform."""

    return _read_pak_entries(path, profile_name, endianness)


def _decode_pak_entry(data: bytes, entry: PakEntry, entry_map: Dict[str, PakEntry], ordered: List[PakEntry]) -> bytes:
    mode_prefix = (entry.mode >> 24) & 0xFF
    if entry.mode == 0xFFFFFFFF or mode_prefix == 0xFF:
        return data[entry.offset : entry.offset + entry.size]

    if mode_prefix in (0x00, 0x10):
        decoded = _decode_deflate_chunks(data, entry)
    elif mode_prefix == 0x20:
        decoded = _decode_lzma_chunks(data, entry)
    else:
        decoded = data[entry.offset : entry.offset + entry.size]

    return decoded[: entry.size]


def _decode_deflate_chunks(data: bytes, entry: PakEntry) -> bytes:
    chunk_alignment = entry.chunk_alignment or 0x8000
    chunk_size = 0x8000
    cursor = entry.offset
    out = bytearray()

    while len(out) < entry.size and cursor + entry.size_field <= len(data):
        comp_size = int.from_bytes(data[cursor : cursor + entry.size_field], entry.size_endianness)
        cursor += entry.size_field
        if comp_size <= 0 or cursor + comp_size > len(data):
            break

        chunk = data[cursor : cursor + comp_size]
        cursor += comp_size

        try:
            out.extend(zlib.decompress(chunk))
        except Exception:
            raw_len = min(chunk_size, len(chunk), entry.size - len(out))
            out.extend(chunk[:raw_len])

        cursor = _align(cursor, chunk_alignment)

    if not out:
        return data[entry.offset : entry.offset + entry.size]

    return bytes(out)


def _decode_lzma_chunks(data: bytes, entry: PakEntry) -> bytes:
    chunk_alignment = entry.chunk_alignment or 0x8000
    chunk_size = 0x8000
    cursor = entry.offset
    out = bytearray()

    while len(out) < entry.size and cursor + entry.size_field + 5 <= len(data):
        comp_size = int.from_bytes(data[cursor : cursor + entry.size_field], entry.size_endianness)
        cursor += entry.size_field
        if cursor + 5 > len(data):
            break

        props = data[cursor : cursor + 5]
        cursor += 5

        if comp_size <= 0 or cursor + comp_size > len(data):
            break

        comp_bytes = data[cursor : cursor + comp_size]
        cursor += comp_size

        try:
            out_chunk = lzma.LZMADecompressor().decompress(props + comp_bytes)
            if not out_chunk:
                out_chunk = comp_bytes[:chunk_size]
        except Exception:
            out_chunk = comp_bytes[:chunk_size]

        remaining = entry.size - len(out)
        out.extend(out_chunk[:remaining])

        cursor = _align(cursor, chunk_alignment)

    if not out:
        return data[entry.offset : entry.offset + entry.size]

    return bytes(out)


def _parse_skeleton(root: ET.Element, override_name: Optional[str]) -> Optional[HavokSkeleton]:
    skel_obj = root.find(".//hkobject[@class='hkaSkeleton']")
    if skel_obj is None:
        return None

    name_param = skel_obj.find("hkparam[@name='name']")
    skel_name = name_param.text.strip() if name_param is not None and name_param.text else (override_name or "Skeleton")

    bones_param = skel_obj.find("hkparam[@name='bones']")
    bones: List[HavokBone] = []
    if bones_param is not None:
        for idx, b in enumerate(bones_param.iterfind("hkobject")):
            bname = _read_text(b, "name", fallback=f"Bone_{idx}")
            parent = int(_read_text(b, "parent", fallback="-1"))
            translation = _read_vector(b.find("hkparam[@name='transform']"), "translation")
            rotation = _read_quaternion(b.find("hkparam[@name='transform']"), "rotation")
            bones.append(
                HavokBone(
                    name=bname,
                    parent=parent,
                    translation=translation,
                    rotation=rotation,
                )
            )

    return HavokSkeleton(name=skel_name, bones=bones)


def _parse_meshes(root: ET.Element, override_name: Optional[str]) -> List[HavokMesh]:
    meshes: List[HavokMesh] = []

    def _first_param(obj: ET.Element, names: Iterable[str]) -> Optional[ET.Element]:
        for n in names:
            param = obj.find(f"hkparam[@name='{n}']")
            if param is not None and param.text:
                return param
        return None

    for idx, obj in enumerate(root.findall(".//hkobject")):
        verts_param = _first_param(obj, ("vertices", "positions"))
        tris_param = _first_param(obj, ("triangles", "indices", "indices16", "indices32"))
        if verts_param is None or tris_param is None:
            continue

        vert_values = [float(v) for v in verts_param.text.split() if v.strip()]
        tri_values = [int(v) for v in tris_param.text.split() if v.strip()]
        if len(vert_values) < 3 or len(tri_values) < 3:
            continue

        vertices = [mathutils.Vector(vert_values[i : i + 3]) for i in range(0, len(vert_values) - 2, 3)]

        if len(tri_values) % 3 == 0:
            stride = 3
        elif len(tri_values) % 4 == 0:
            stride = 4
        else:
            continue

        faces: List[Tuple[int, int, int]] = []
        for i in range(0, len(tri_values) - stride + 1, stride):
            face = tri_values[i : i + 3]
            if min(face) < 0 or max(face) >= len(vertices):
                continue
            faces.append(tuple(face))

        if not faces or not vertices:
            continue

        name = obj.attrib.get("name") or override_name or f"Mesh_{idx}"
        meshes.append(HavokMesh(name=name, vertices=vertices, faces=faces))

    return meshes


def _parse_animations(root: ET.Element, skeleton: Optional[HavokSkeleton]) -> List[HavokAnimation]:
    bindings = list(root.findall(".//hkobject[@class='hkaAnimationBinding']"))
    binding_map: Dict[str, ET.Element] = {}
    for bind in bindings:
        anim_ref = bind.find("hkparam[@name='animation']")
        if anim_ref is not None and anim_ref.text:
            binding_map[anim_ref.text.strip()] = bind

    animations: List[HavokAnimation] = []
    for anim_obj in root.findall(".//hkobject[@class='hkaAnimation']"):
        anim_name = _read_text(anim_obj, "name", fallback="Animation")
        anim_key = anim_obj.attrib.get("name", anim_name)
        duration = float(_read_text(anim_obj, "duration", fallback="0"))
        num_tracks = int(_read_text(anim_obj, "numberOfTransformTracks", fallback="0"))
        num_frames = int(_read_text(anim_obj, "numOriginalFrames", fallback="0")) or int(
            _read_text(anim_obj, "numFrames", fallback="0")
        )
        transforms_param = anim_obj.find("hkparam[@name='transforms']")
        tracks = _decode_interleaved_tracks(transforms_param, num_tracks, num_frames)

        binding = binding_map.get(anim_key)
        track_to_bone = _parse_binding(binding, num_tracks, skeleton)

        animations.append(
            HavokAnimation(
                name=anim_name,
                duration=duration,
                tracks=tracks,
                track_to_bone=track_to_bone,
            )
        )

    return animations


def _decode_interleaved_tracks(transforms_param: Optional[ET.Element], num_tracks: int, num_frames: int) -> List[List[Tuple[mathutils.Vector, mathutils.Quaternion]]]:
    if transforms_param is None or transforms_param.text is None:
        return []

    values = [float(v) for v in transforms_param.text.split()]
    expected = num_tracks * max(num_frames, 1) * 7
    if not values:
        return []

    if len(values) < expected:
        # Fallback: if the XML is missing numOriginalFrames we can infer from data
        num_frames = max(num_frames, len(values) // (num_tracks * 7)) if num_tracks else num_frames
        expected = num_tracks * max(num_frames, 1) * 7

    tracks: List[List[Tuple[mathutils.Vector, mathutils.Quaternion]]] = [
        [] for _ in range(num_tracks)
    ]
    idx = 0
    for _frame in range(max(num_frames, 1)):
        for track in range(num_tracks):
            if idx + 7 > len(values):
                break
            t = mathutils.Vector(values[idx : idx + 3])
            q = mathutils.Quaternion((values[idx + 3], values[idx + 4], values[idx + 5], values[idx + 6]))
            tracks[track].append((t, q))
            idx += 7

    return tracks


def _parse_binding(binding: Optional[ET.Element], num_tracks: int, skeleton: Optional[HavokSkeleton]) -> List[int]:
    if binding is None:
        # Identity mapping when there is no binding metadata.
        return list(range(num_tracks))

    arr_param = binding.find("hkparam[@name='transformTrackToBoneIndices']")
    if arr_param is None or arr_param.text is None:
        return list(range(num_tracks))

    indices = [int(v) for v in arr_param.text.split() if v.strip()]
    if len(indices) < num_tracks:
        # Fill missing indices with sequential mapping.
        indices.extend(list(range(len(indices), num_tracks)))
    if skeleton:
        # Clamp invalid indices to last bone to avoid crashes.
        last_bone = max(-1, len(skeleton.bones) - 1)
        indices = [i if -1 <= i <= last_bone else last_bone for i in indices[:num_tracks]]
    return indices[:num_tracks]


def _read_text(parent: ET.Element, name: str, fallback: str) -> str:
    param = parent.find(f"hkparam[@name='{name}']")
    if param is not None and param.text:
        return param.text.strip()
    return fallback


def _read_vector(transform_param: Optional[ET.Element], name: str) -> mathutils.Vector:
    if transform_param is None:
        return mathutils.Vector((0.0, 0.0, 0.0))
    param = transform_param.find(f"hkparam[@name='{name}']")
    if param is None or param.text is None:
        return mathutils.Vector((0.0, 0.0, 0.0))
    values = [float(v) for v in param.text.split()]
    while len(values) < 3:
        values.append(0.0)
    return mathutils.Vector(values[:3])


def _read_quaternion(transform_param: Optional[ET.Element], name: str) -> mathutils.Quaternion:
    if transform_param is None:
        return mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    param = transform_param.find(f"hkparam[@name='{name}']")
    if param is None or param.text is None:
        return mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
    values = [float(v) for v in param.text.split()]
    while len(values) < 4:
        values.append(0.0)
    return mathutils.Quaternion(values[:4])
