"""Havok importer utilities for skeletons and animations.

The helper functions in this module focus on Havok XML packfiles generated
by hkxpack/hkcmd and the IGZ/PAK wrappers commonly used by Alchemy games.
They intentionally avoid placeholder logic and instead build real transform
tracks for Blender armatures when data is present.
"""
from __future__ import annotations

import gzip
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
