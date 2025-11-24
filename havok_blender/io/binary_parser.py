"""Havok binary packfile parsing and animation decoding.

This module provides a clean, Pythonic reimplementation of the minimal
Havok packfile reader used by the Blender importer. It focuses on
animation data and mirrors the concepts from HavokLib/HavokMax without
reusing any of the previous logic.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .spline_decompressor import SplineDecompressor


class BinaryReader:
    def __init__(self, data: bytes):
        self.data = memoryview(data)
        self.offset = 0
        self.endian = "<"

    def set_endian(self, little_endian: bool) -> None:
        self.endian = "<" if little_endian else ">"

    def read(self, fmt: str) -> Tuple[int, ...]:
        size = struct.calcsize(fmt)
        values = struct.unpack_from(self.endian + fmt, self.data, self.offset)
        self.offset += size
        return values

    def read_bytes(self, size: int) -> bytes:
        chunk = self.data[self.offset : self.offset + size]
        self.offset += size
        return bytes(chunk)

    def read_string(self, size: int) -> str:
        raw = self.read_bytes(size)
        end = raw.find(b"\0")
        if end != -1:
            raw = raw[:end]
        return raw.decode("ascii", errors="ignore")

    def tell(self) -> int:
        return self.offset

    def seek(self, offset: int, whence: int = 0) -> None:
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset += offset
        elif whence == 2:
            self.offset = len(self.data) + offset


@dataclass
class HkxLayout:
    bytes_in_pointer: int
    little_endian: bool
    reuse_padding: bool
    empty_base_class: bool


@dataclass
class Section:
    tag: str
    absolute_data_start: int
    local_fixups_offset: int
    global_fixups_offset: int
    virtual_fixups_offset: int
    exports_offset: int
    imports_offset: int
    buffer_size: int
    data: bytearray
    pointer_map: Dict[int, Tuple[int, int]]


class hkxHeader:
    def __init__(self) -> None:
        self.layout: Optional[HkxLayout] = None
        self.sections: List[Section] = []
        self.contents_section_index = 0
        self.contents_section_offset = 0
        self.contents_class_name_section_index = 0
        self.contents_class_name_section_offset = 0
        self.contents_version = "hk_2015"

    # Header parsing --------------------------------------------------
    def load(self, reader: BinaryReader) -> None:
        magic1, magic2, user_tag, version = reader.read("IIII")
        layout_bytes = reader.read("BBBB")
        bytes_in_pointer, little_endian, reuse_padding, empty_base = layout_bytes
        self.layout = HkxLayout(
            bytes_in_pointer=bytes_in_pointer,
            little_endian=bool(little_endian),
            reuse_padding=bool(reuse_padding),
            empty_base_class=bool(empty_base),
        )
        reader.set_endian(self.layout.little_endian)

        (
            num_sections,
            self.contents_section_index,
            self.contents_section_offset,
            self.contents_class_name_section_index,
            self.contents_class_name_section_offset,
        ) = reader.read("iiiii")
        self.contents_version = reader.read_string(16)
        _flags = reader.read("I")
        _max_predicate, predicate_array_size = reader.read("hh")
        if _max_predicate != -1:
            reader.seek(predicate_array_size, 1)

        self.sections = []
        for section_id in range(num_sections):
            tag = reader.read_string(20)
            (
                absolute_data_start,
                local_fixups_offset,
                global_fixups_offset,
                virtual_fixups_offset,
                exports_offset,
                imports_offset,
                buffer_size,
            ) = reader.read("IIIIIII")
            if version > 9:
                reader.seek(16, 1)
            self.sections.append(
                Section(
                    tag=tag,
                    absolute_data_start=absolute_data_start,
                    local_fixups_offset=local_fixups_offset,
                    global_fixups_offset=global_fixups_offset,
                    virtual_fixups_offset=virtual_fixups_offset,
                    exports_offset=exports_offset,
                    imports_offset=imports_offset,
                    buffer_size=buffer_size,
                    data=bytearray(),
                    pointer_map={},
                )
            )

        for section in self.sections:
            self._load_section(reader, section)

    def _load_section(self, reader: BinaryReader, section: Section) -> None:
        if section.buffer_size == 0:
            return
        reader.seek(section.absolute_data_start)
        section.data = bytearray(reader.read_bytes(section.local_fixups_offset))
        virtual_eof = (
            section.imports_offset if section.exports_offset == 0xFFFFFFFF else section.exports_offset
        )
        num_local = (section.global_fixups_offset - section.local_fixups_offset) // 8
        num_global = (section.virtual_fixups_offset - section.global_fixups_offset) // 12
        reader.seek(section.absolute_data_start + section.local_fixups_offset)
        local_fixups = [reader.read("ii") for _ in range(num_local)]
        reader.seek(section.absolute_data_start + section.global_fixups_offset)
        global_fixups = [reader.read("iii") for _ in range(num_global)]
        section.pointer_map = {}
        for pointer, destination in local_fixups:
            if pointer != -1:
                section.pointer_map[pointer] = (self.sections.index(section), destination)
        for pointer, target_section, destination in global_fixups:
            if pointer != -1:
                section.pointer_map[pointer] = (target_section, destination)

    # Pointer helpers -------------------------------------------------
    def _section(self, index: int) -> Optional[Section]:
        if 0 <= index < len(self.sections):
            return self.sections[index]
        return None

    def read_pointer(self, section_index: int, offset: int) -> Optional[Tuple[int, int]]:
        section = self._section(section_index)
        if not section:
            return None
        if offset in section.pointer_map:
            return section.pointer_map[offset]
        ptr_size = self.layout.bytes_in_pointer if self.layout else 4
        if offset + ptr_size > len(section.data):
            return None
        raw_val = int.from_bytes(
            section.data[offset : offset + ptr_size],
            "little" if self.layout and self.layout.little_endian else "big",
        )
        if raw_val == 0:
            return None
        return (section_index, raw_val)

    def read_hkarray(self, section_index: int, offset: int) -> Tuple[Optional[Tuple[int, int]], int]:
        ptr_size = self.layout.bytes_in_pointer if self.layout else 4
        data_ptr = self.read_pointer(section_index, offset)
        count_offset = offset + ptr_size
        section = self._section(section_index)
        if not section:
            return None, 0
        count = int.from_bytes(
            section.data[count_offset : count_offset + 4],
            "little" if self.layout and self.layout.little_endian else "big",
        )
        return data_ptr, count

    def read_string_at(self, section_index: int, offset: int) -> str:
        section = self._section(section_index)
        if not section:
            return ""
        end = section.data.find(b"\0", offset)
        if end == -1:
            raw = section.data[offset:]
        else:
            raw = section.data[offset:end]
        return raw.decode("ascii", errors="ignore")

    def read_string_ptr(self, section_index: int, offset: int) -> str:
        ptr = self.read_pointer(section_index, offset)
        if not ptr:
            return ""
        sid, soff = ptr
        return self.read_string_at(sid, soff)

    # Root variant helpers --------------------------------------------
    def get_root_level_container(self) -> List[Dict[str, object]]:
        if not self.layout:
            return []
        ptr_size = self.layout.bytes_in_pointer
        variants_ptr, variants_size = self.read_hkarray(
            self.contents_section_index, self.contents_section_offset
        )
        variants: List[Dict[str, object]] = []
        if not variants_ptr:
            return variants
        sid, soff = variants_ptr
        variant_size = ptr_size * 3
        for idx in range(variants_size):
            cursor = soff + idx * variant_size
            name = self.read_string_ptr(sid, cursor)
            class_name = self.read_string_ptr(sid, cursor + ptr_size)
            variant_ptr = self.read_pointer(sid, cursor + 2 * ptr_size)
            variants.append(
                {"name": name, "class_name": class_name, "variant_ptr": variant_ptr}
            )
        return variants

    # Skeleton parsing ------------------------------------------------
    def read_hka_skeleton(self, section_index: int, offset: int) -> Dict[str, object]:
        ptr_size = self.layout.bytes_in_pointer if self.layout else 4
        header_skip = 16 if ptr_size == 8 else 8
        cursor = offset + header_skip
        name = self.read_string_ptr(section_index, cursor)
        cursor += ptr_size
        parent_ptr, parent_count = self.read_hkarray(section_index, cursor)
        cursor += 12 if ptr_size == 4 else 16
        bones_ptr, bone_count = self.read_hkarray(section_index, cursor)
        cursor += 12 if ptr_size == 4 else 16
        ref_pose_ptr, ref_pose_count = self.read_hkarray(section_index, cursor)

        parents: List[int] = []
        if parent_ptr:
            psid, poff = parent_ptr
            parent_section = self._section(psid)
            for idx in range(parent_count):
                raw = parent_section.data[poff + idx * 2 : poff + idx * 2 + 2]
                parents.append(int.from_bytes(raw, "little" if self.layout.little_endian else "big", signed=True))

        bones: List[Dict[str, object]] = []
        if bones_ptr:
            bsid, boff = bones_ptr
            bone_section = self._section(bsid)
            bone_size = 16 if ptr_size == 8 else 8
            for idx in range(bone_count):
                b_offset = boff + idx * bone_size
                b_name = self.read_string_ptr(bsid, b_offset)
                bones.append({"name": b_name, "parent": parents[idx] if idx < len(parents) else -1})

        ref_poses: List[Dict[str, Tuple[float, float, float]]] = []
        if ref_pose_ptr:
            rsid, roff = ref_pose_ptr
            ref_section = self._section(rsid)
            for idx in range(ref_pose_count):
                t_off = roff + idx * 48
                floats = struct.unpack_from(
                    ("<" if self.layout.little_endian else ">") + "f" * 12,
                    ref_section.data,
                    t_off,
                )
                translation = floats[0:3]
                rotation = floats[4:8]
                scale = floats[8:11]
                ref_poses.append(
                    {"translation": translation, "rotation": rotation, "scale": scale}
                )
        return {"name": name, "bones": bones, "ref_poses": ref_poses}

    # Animation container and bindings --------------------------------
    def read_hka_animation_container(self, section_index: int, offset: int) -> Dict[str, object]:
        ptr_size = self.layout.bytes_in_pointer if self.layout else 4
        cursor = offset + (16 if ptr_size == 8 else 8)
        skeletons_ptr, skeletons_size = self.read_hkarray(section_index, cursor)
        cursor += 12 if ptr_size == 4 else 16
        animations_ptr, animations_size = self.read_hkarray(section_index, cursor)
        cursor += 12 if ptr_size == 4 else 16
        bindings_ptr, bindings_size = self.read_hkarray(section_index, cursor)
        cursor += 12 if ptr_size == 4 else 16
        attachments_ptr, attachments_size = self.read_hkarray(section_index, cursor)
        cursor += 12 if ptr_size == 4 else 16
        skins_ptr, skins_size = self.read_hkarray(section_index, cursor)
        return {
            "skeletons": (skeletons_ptr, skeletons_size),
            "animations": (animations_ptr, animations_size),
            "bindings": (bindings_ptr, bindings_size),
            "attachments": (attachments_ptr, attachments_size),
            "skins": (skins_ptr, skins_size),
        }

    def read_hka_animation_binding(self, section_index: int, offset: int) -> Dict[str, object]:
        ptr_size = self.layout.bytes_in_pointer if self.layout else 4
        cursor = offset + (16 if ptr_size == 8 else 8)
        original_skeleton_name = self.read_string_ptr(section_index, cursor)
        cursor += ptr_size
        animation_ptr = self.read_pointer(section_index, cursor)
        cursor += ptr_size
        track_to_bone_ptr, track_to_bone_size = self.read_hkarray(section_index, cursor)
        cursor += 12 if ptr_size == 4 else 16
        _float_map_ptr, _float_map_size = self.read_hkarray(section_index, cursor)
        cursor += 12 if ptr_size == 4 else 16
        section = self._section(section_index)
        blend_hint = section.data[cursor]

        track_to_bone: List[int] = []
        if track_to_bone_ptr:
            sid, soff = track_to_bone_ptr
            t_section = self._section(sid)
            for idx in range(track_to_bone_size):
                raw = t_section.data[soff + idx * 2 : soff + idx * 2 + 2]
                track_to_bone.append(int.from_bytes(raw, "little" if self.layout.little_endian else "big", signed=True))

        return {
            "original_skeleton_name": original_skeleton_name,
            "animation_ptr": animation_ptr,
            "track_to_bone": track_to_bone,
            "blend_hint": blend_hint,
        }

    # Animation parsing ------------------------------------------------
    def _animation_layout(self) -> List[Tuple[int, int, int, int]]:
        year = _year_from_version(self.contents_version)
        ptr_size = self.layout.bytes_in_pointer if self.layout else 4
        reuse_padding = self.layout.reuse_padding if self.layout else False
        # Offsets correspond to enum order in hka_animation_spline.inl
        if ptr_size == 8:
            if year >= 2016:
                return [
                    0,
                    80,
                    84,
                    96,
                    160,
                    112,
                    144,
                    88,
                    76,
                    72,
                    104,
                    68,
                    168,
                    120,
                    152,
                    64,
                    136,
                    128,
                ]
            return [
                0,
                72,
                76,
                88,
                152,
                104,
                136,
                80,
                68,
                64,
                96,
                60,
                160,
                112,
                144,
                56,
                128,
                120,
            ]
        if year >= 2016:
            return [
                0,
                60,
                64,
                72,
                120,
                84,
                108,
                68,
                56,
                52,
                76,
                48,
                124,
                88,
                112,
                44,
                100,
                96,
            ]
        return [
            0,
            56,
            60,
            68,
            116,
            80,
            104,
            64,
            52,
            48,
            72,
            44,
            120,
            84,
            108,
            40,
            96,
            92,
        ]

    def _base_animation_layout(self) -> List[int]:
        year = _year_from_version(self.contents_version)
        ptr_size = self.layout.bytes_in_pointer if self.layout else 4
        if ptr_size == 8:
            if year >= 2016:
                return [24, 48, 0, 28, 40, 56, 36, 32]
            return [16, 40, 0, 20, 32, 48, 28, 24]
        if year >= 2016:
            return [24, 48, 0, 28, 40, 56, 36, 32]
        return [8, 28, 0, 12, 24, 32, 20, 16]

    def read_hka_animation(self, section_index: int, offset: int) -> Dict[str, object]:
        ptr_size = self.layout.bytes_in_pointer if self.layout else 4
        offsets = self._animation_layout()
        base_offsets = self._base_animation_layout()
        section = self._section(section_index)
        endian = "little" if self.layout.little_endian else "big"
        def read_f32(rel: int) -> float:
            return struct.unpack_from(("<" if endian == "little" else ">") + "f", section.data, offset + rel)[0]

        def read_u32(rel: int) -> int:
            return int.from_bytes(section.data[offset + rel : offset + rel + 4], endian)

        duration = read_f32(base_offsets[3])
        num_transform_tracks = read_u32(base_offsets[7])
        num_float_tracks = read_u32(base_offsets[6]) if base_offsets[6] >= 0 else 0
        num_frames = read_u32(offsets[15])
        num_blocks = read_u32(offsets[11])
        max_frames_per_block = read_u32(offsets[9])
        block_duration = read_f32(offsets[1])
        block_inverse_duration = read_f32(offsets[2])
        frame_duration = read_f32(offsets[7])

        animation_name = self.read_string_ptr(section_index, offset + base_offsets[0])
        if not animation_name:
            animation_name = "Animation"

        def read_array_ptr(rel: int, count_rel: int) -> Tuple[List[int], int]:
            arr_ptr = self.read_pointer(section_index, offset + rel)
            count = read_u32(count_rel)
            values: List[int] = []
            if arr_ptr:
                sid, soff = arr_ptr
                arr_section = self._section(sid)
                for idx in range(count):
                    values.append(int.from_bytes(arr_section.data[soff + idx * 4 : soff + idx * 4 + 4], endian))
            return values, count

        block_offsets, _ = read_array_ptr(offsets[3], offsets[10])
        if num_frames == 0:
            if block_offsets:
                num_frames = block_offsets[-1]
            elif duration > 0.0 and frame_duration > 0.0:
                num_frames = max(1, int(round(duration / frame_duration)) + 1)
        data_ptr = self.read_pointer(section_index, offset + offsets[4])
        data_size = read_u32(offsets[12])
        data_buffer = b""
        if data_ptr:
            dsid, doff = data_ptr
            data_section = self._section(dsid)
            data_buffer = bytes(data_section.data[doff : doff + data_size])

        decompressor = SplineDecompressor(little_endian=self.layout.little_endian)
        decompressor.decompress(
            data_buffer=data_buffer,
            block_offsets=block_offsets,
            num_transform_tracks=num_transform_tracks,
            num_float_tracks=num_float_tracks,
        )
        tracks = decompressor.sample_all_tracks(num_frames, frame_duration)
        return {
            "name": animation_name,
            "duration": duration,
            "tracks": tracks,
            "num_frames": num_frames,
            "block_duration": block_duration,
            "block_inverse_duration": block_inverse_duration,
            "frame_duration": frame_duration,
            "num_blocks": num_blocks,
            "max_frames_per_block": max_frames_per_block,
        }


def _year_from_version(version_string: str) -> int:
    match = re.search(r"(20\d{2})", version_string)
    if match:
        return int(match.group(1))
    return 2015
