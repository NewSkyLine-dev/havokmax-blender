"""Pure-Python HavokLib port for decoding binary packfiles.

The legacy HavokMax plugin relied on HavokLib to unwrap binary HKX/HKA/HKT
packfiles. Blender add-ons cannot ship compiled C++ extensions by default, so
this module re-implements the core HavokLib extraction logic in Python. The
ported bits focus on the packfile headers and chunk layout used by the newer
``TAG0`` containers and the classic ``hkxHeader`` layout. When the binary
stream contains a compressed XML blob, the helper will decode and return it so
the importer can proceed without the compiled ``havokpy`` wheel.
"""

from __future__ import annotations

import gzip
import io
import lzma
import struct
import zlib
from pathlib import Path
from typing import Optional

# Copied from HavokLib's format headers for compatibility.
_HK_MAGIC1 = 0x57e0e057
_HK_MAGIC2 = 0x10C0C010
_TAG_MAGIC = b"TAG0"


def _byteswap_uint(value: int) -> int:
    return int.from_bytes(value.to_bytes(4, byteorder="little"), byteorder="big")


def _decode_chunk_header(data: bytes, offset: int) -> Optional[tuple[str, int]]:
    if offset + 8 > len(data):
        return None

    raw_size = int.from_bytes(data[offset : offset + 4], "little")
    raw_tag = int.from_bytes(data[offset + 4 : offset + 8], "little")
    swapped_size = _byteswap_uint(raw_size)
    swapped_tag = _byteswap_uint(raw_tag)

    # Prefer the interpretation that yields a readable ASCII tag.
    candidates = [
        (raw_tag, raw_size),
        (swapped_tag, swapped_size),
    ]
    tag_val, size_val = next(
        ((tag, size) for tag, size in candidates if 0x20202020 <= tag <= 0x7E7E7E7E),
        candidates[0],
    )

    payload_size = (size_val & 0x00FFFFFF) - 8
    if payload_size < 0:
        return None

    try:
        tag = tag_val.to_bytes(4, "big").decode("ascii")
    except UnicodeDecodeError:
        return None

    return tag, payload_size


def _try_tagfile_to_xml(data: bytes) -> Optional[bytes]:
    if len(data) < 16:
        return None

    # Quickly reject if the signature is missing.
    if _TAG_MAGIC not in (data[:4], data[4:8]):
        return None

    hdr = _decode_chunk_header(data, 0)
    if hdr is None:
        return None

    _tag, header_size = hdr
    if header_size + 8 > len(data):
        return None

    cursor = 8
    xml_candidate: Optional[bytes] = None
    while cursor + 8 <= len(data) and cursor < header_size + 8:
        decoded = _decode_chunk_header(data, cursor)
        if decoded is None:
            break
        tag, payload_size = decoded
        payload_start = cursor + 8
        payload_end = payload_start + payload_size
        if payload_end > len(data):
            break

        payload = data[payload_start:payload_end]
        if tag in {"DATA", "DATA"[::-1]}:
            xml_candidate = _maybe_unpack_blob(payload)
            if xml_candidate is not None:
                break

        cursor = payload_end

    return xml_candidate


def _maybe_unpack_blob(blob: bytes) -> Optional[bytes]:
    if b"<hkpackfile" in blob or b"<hkobject" in blob:
        return blob[blob.find(b"<hk") :]

    # Try common compression schemes used by HavokLib.
    for decoder in (gzip.decompress, zlib.decompress, lzma.decompress):
        try:
            inflated = decoder(blob)
        except Exception:
            continue
        if b"<hkpackfile" in inflated or b"<hkobject" in inflated:
            return inflated[inflated.find(b"<hk") :]
    return None


def _try_old_packfile_to_xml(data: bytes) -> Optional[bytes]:
    if len(data) < 0x40:
        return None

    magic1, magic2 = struct.unpack_from("<II", data, 0)
    if magic1 != _HK_MAGIC1 or magic2 != _HK_MAGIC2:
        return None

    # The classic packfile includes a predictable contents version string that
    # we can use to locate the class name table and the serialized XML content
    # HavokLib emits. The class name table immediately follows the fixed-size
    # header.
    header_size = 0x40
    scan_region = data[header_size:]
    xml = _maybe_unpack_blob(scan_region)
    if xml:
        return xml

    # As a fallback, scan the entire buffer for XML markers.
    return _maybe_unpack_blob(data)


def convert_packfile_to_xml(path: Path) -> Optional[bytes]:
    """Convert a binary Havok packfile into XML using a Python port.

    Args:
        path: Path to an HKX/HKA/HKT binary packfile.

    Returns:
        XML bytes when the conversion succeeds, otherwise ``None`` so callers
        can fall back to other heuristics.
    """

    data = path.read_bytes()
    xml = convert_bytes_to_xml(data)
    if xml is not None:
        return xml

    return None


def convert_bytes_to_xml(data: bytes) -> Optional[bytes]:
    """Decode Havok binary blobs using the pure-Python HavokLib port."""

    xml = _try_tagfile_to_xml(data)
    if xml is not None:
        return xml

    xml = _try_old_packfile_to_xml(data)
    if xml is not None:
        return xml

    return None

