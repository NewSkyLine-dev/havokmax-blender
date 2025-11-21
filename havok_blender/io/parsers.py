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

from .havoklib_bridge import convert_bytes_to_xml, convert_packfile_to_xml

SUPPORTED_EXTENSIONS = {".hkx", ".hka", ".hkt", ".igz", ".pak"}

_PAK_LAYOUTS: Dict[str, Dict[str, int]] = {
    "SSA_WII": {
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
    "SSA_WIIU": {
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
    "SWAP_FORCE": {
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
    "LOST_ISLANDS": {
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
    "TRAP_TEAM": {
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
    "SUPER_CHARGERS": {
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
    "IMAGINATORS": {
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
    "CRASH_NST": {
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
}


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


@dataclass
class PakEntry:
    name: str
    offset: int
    size: int
    mode: int
    endianness: str


def load_from_path(path: Path, entry: Optional[str] = None) -> HavokPack:
    """Load any supported Havok source from disk.

    Args:
        path: file path provided by the user.
        entry: optional archive entry for PAK/ZIP containers.
    """

    suffix = path.suffix.lower()
    if suffix == ".pak":
        data = _extract_from_archive(path, entry)
    else:
        data = path.read_bytes()

    converted = convert_packfile_to_xml(path)
    if converted:
        data = converted

    return parse_bytes(data, override_name=path.stem)


def parse_bytes(data: bytes, override_name: Optional[str] = None) -> HavokPack:
    """Parse Havok XML/IGZ data into skeleton and animation structures."""

    xml_bytes = _unwrap_bytes(data)
    if b"<hkpackfile" not in xml_bytes:
        # Sometimes IGZ payloads contain a full XML blob preceded by metadata.
        marker = xml_bytes.find(b"<hkobject")
        if marker > -1:
            xml_bytes = xml_bytes[marker:]
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Unsupported or corrupt Havok payload") from exc

    skeleton = _parse_skeleton(root, override_name)
    animations = _parse_animations(root, skeleton)
    return HavokPack(skeleton=skeleton, animations=animations)


def _unwrap_bytes(data: bytes) -> bytes:
    # IGZ and some Havok distributions are gzip-compressed.
    if data.startswith(b"\x1f\x8b"):
        return gzip.decompress(data)

    igz_payload = _maybe_from_igz(data)
    if igz_payload is not None:
        return igz_payload

    converted = convert_bytes_to_xml(data)
    if converted is not None:
        return converted

    embedded = _slice_embedded_havok(data)
    if embedded is not None:
        return embedded

    # Tar/zip payloads are considered higher-level PAK containers; they should
    # be handled by _extract_from_archive instead.
    return data


def _extract_from_archive(path: Path, entry: Optional[str]) -> bytes:
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

    pak_entries = _read_pak_entries(path)
    if pak_entries:
        entry_map = {p.name: p for p in pak_entries}
        target_name = entry or (next((name for name in entry_map if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS), None))
        if target_name is None:
            target_name = pak_entries[0].name
        if target_name not in entry_map:
            raise ValueError(f"Entry '{target_name}' not found in PAK; options: {sorted(entry_map.keys())}")
        return _decode_pak_entry(path.read_bytes(), entry_map[target_name], entry_map, pak_entries)

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
    one or more compressed blocks. To avoid re-implementing the full Noesis
    parser, we reuse the community technique of scanning for compressed streams
    and Havok XML markers.
    """

    if not (data.startswith(b"\x01ZGI") or data.startswith(b"IGZ\x01")):
        return None

    # 1) Look for inline gzip members.
    gzip_idx = data.find(b"\x1f\x8b")
    if gzip_idx != -1:
        try:
            return gzip.decompress(data[gzip_idx:])
        except OSError:
            pass

    # 2) Search for common zlib headers, as used by Skylanders IGZ payloads.
    for sig in (b"\x78\x9c", b"\x78\x01", b"\x78\xda"):
        z_idx = data.find(sig)
        if z_idx != -1:
            try:
                return zlib.decompress(data[z_idx:])
            except zlib.error:
                continue

    # 3) If compression markers are absent, fall back to slicing out the XML.
    return _slice_embedded_havok(data)


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


def _try_layout(data: bytes, layout: Dict[str, int], endianness: str) -> Optional[List[PakEntry]]:
    num_files = _read_uint(data, layout["num_files"], endianness)
    if num_files <= 0 or num_files > 10_000:
        return None

    name_loc = layout["name_loc"]
    name_len = layout["name_len"]
    name_table_end = name_loc + _read_uint(data, name_len, endianness)
    if name_table_end > len(data):
        return None

    names: List[str] = []
    for idx in range(num_files):
        offset_ptr = name_loc + 4 * idx
        if offset_ptr + 4 > len(data):
            return None
        name_offset = _read_uint(data, offset_ptr, endianness)
        start = name_loc + name_offset
        if start >= len(data):
            return None
        end = data.find(b"\x00", start, len(data))
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

    entries: List[PakEntry] = []
    base_header = checksum_loc + checksum_len * num_files
    for idx in range(num_files):
        header_base = base_header + local_header_len * idx
        if header_base + max(file_start_in_local, file_size_in_local, mode_in_local) + 4 > len(data):
            return None
        start = _read_uint(data, header_base + file_start_in_local, endianness)
        size = _read_uint(data, header_base + file_size_in_local, endianness)
        mode = _read_uint(data, header_base + mode_in_local, endianness)
        if start >= len(data) or size <= 0:
            return None
        entries.append(PakEntry(name=names[idx], offset=start, size=size, mode=mode, endianness=endianness))

    return entries


def _read_pak_entries(path: Path) -> List[PakEntry]:
    data = path.read_bytes()
    if len(data) < 0x40:
        return []

    magic = data[:4]
    if magic not in (b"\x1AAGI", b"IGA\x1A"):
        return []
    endianness = "little" if magic == b"\x1AAGI" else "big"

    for layout in _PAK_LAYOUTS.values():
        entries = _try_layout(data, layout, endianness)
        if entries:
            return entries
    return []


def enumerate_pak_entries(path: Path) -> List[PakEntry]:
    """Return parsed PAK entries for UI listing."""

    return _read_pak_entries(path)


def _decode_pak_entry(data: bytes, entry: PakEntry, entry_map: Dict[str, PakEntry], ordered: List[PakEntry]) -> bytes:
    # Heuristic: the compressed blob spans until the next entry or EOF.
    ordered_sorted = sorted(ordered, key=lambda e: e.offset)
    next_offsets = [e.offset for e in ordered_sorted if e.offset > entry.offset]
    end = min(next_offsets) if next_offsets else len(data)
    blob = data[entry.offset:end]

    mode_prefix = (entry.mode >> 24) & 0xFF
    if entry.mode == 0xFFFFFFFF or mode_prefix == 0xFF:
        return blob[: entry.size]

    # Try straightforward zlib first; many variants use deflate-chunked blocks.
    if mode_prefix in (0x00, 0x10):
        try:
            return zlib.decompress(blob, bufsize=entry.size)
        except Exception:
            pass
        # Fallback: treat the stream as a sequence of length-prefixed chunks.
        out = bytearray()
        idx = 0
        while idx + 2 <= len(blob) and len(out) < entry.size:
            comp_len = int.from_bytes(blob[idx : idx + 2], entry.endianness)
            idx += 2
            if comp_len <= 0 or idx + comp_len > len(blob):
                break
            chunk = blob[idx : idx + comp_len]
            idx += comp_len
            try:
                out.extend(zlib.decompress(chunk))
            except Exception:
                out.extend(chunk)
        if out:
            return bytes(out[: entry.size])

    if mode_prefix == 0x20:
        # LZMA blocks: assume 5-byte properties followed by compressed size.
        try:
            props = blob[:5]
            lzma_blob = blob[5:]
            return lzma.LZMADecompressor().decompress(props + lzma_blob)[: entry.size]
        except Exception:
            pass

    return blob[: entry.size]


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
