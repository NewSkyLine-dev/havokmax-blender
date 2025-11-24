import struct
import sys
import types
from pathlib import Path

import pytest

ImportHelper = type("ImportHelper", (), {})
Operator = type("Operator", (ImportHelper,), {})
bpy_stub = types.SimpleNamespace(
    utils=types.SimpleNamespace(
        register_class=lambda _: None, unregister_class=lambda _: None
    ),
    types=types.SimpleNamespace(
        PropertyGroup=object,
        Operator=Operator,
        UIList=object,
        AddonPreferences=object,
        Panel=object,
    ),
)
mathutils_stub = types.SimpleNamespace(
    Vector=lambda *args, **kwargs: None,
    Quaternion=lambda *args, **kwargs: None,
    Matrix=lambda *args, **kwargs: None,
)
sys.modules.setdefault("bpy", bpy_stub)
sys.modules.setdefault("mathutils", mathutils_stub)
sys.modules.setdefault(
    "bpy_extras.io_utils",
    types.SimpleNamespace(ImportHelper=ImportHelper, axis_conversion=lambda **_: None),
)

from havok_blender.io.binary_parser import BinaryReader, HkxLayout, Section, hkxHeader


def _build_identity_animation_header(num_frames: int = 3):
    hk = hkxHeader()
    hk.layout = HkxLayout(
        bytes_in_pointer=4,
        little_endian=True,
        reuse_padding=False,
        empty_base_class=False,
    )
    hk.contents_version = "hk_2015"

    anim_section = Section(
        tag="__data",
        absolute_data_start=0,
        local_fixups_offset=0,
        global_fixups_offset=0,
        virtual_fixups_offset=0,
        exports_offset=0,
        imports_offset=0,
        buffer_size=512,
        data=bytearray(512),
        pointer_map={},
    )
    data_section = Section(
        tag="__data",
        absolute_data_start=0,
        local_fixups_offset=0,
        global_fixups_offset=0,
        virtual_fixups_offset=0,
        exports_offset=0,
        imports_offset=0,
        buffer_size=256,
        data=bytearray(256),
        pointer_map={},
    )
    hk.sections = [anim_section, data_section]

    offsets = hk._animation_layout()
    base_offsets = hk._base_animation_layout()

    duration = 0.1
    frame_duration = 1.0 / 30.0
    num_transform_tracks = 1
    num_float_tracks = 0
    num_blocks = 1

    struct.pack_into("<f", anim_section.data, base_offsets[3], duration)
    anim_section.data[base_offsets[7] : base_offsets[7] + 4] = (
        num_transform_tracks.to_bytes(4, "little")
    )
    anim_section.data[base_offsets[6] : base_offsets[6] + 4] = (
        num_float_tracks.to_bytes(4, "little")
    )

    struct.pack_into("<f", anim_section.data, offsets[1], frame_duration * num_frames)
    struct.pack_into(
        "<f", anim_section.data, offsets[2], 1.0 / (frame_duration * num_frames)
    )
    struct.pack_into("<f", anim_section.data, offsets[7], frame_duration)

    anim_section.data[offsets[9] : offsets[9] + 4] = num_frames.to_bytes(4, "little")
    anim_section.data[offsets[10] : offsets[10] + 4] = (1).to_bytes(4, "little")
    anim_section.data[offsets[11] : offsets[11] + 4] = num_blocks.to_bytes(4, "little")
    anim_section.data[offsets[15] : offsets[15] + 4] = num_frames.to_bytes(4, "little")

    block_offsets_offset = 0
    struct.pack_into("<I", data_section.data, block_offsets_offset, 0)
    anim_section.pointer_map[offsets[3]] = (1, block_offsets_offset)

    data_buffer_offset = 16
    block_payload = bytearray(64)
    block_payload[0:4] = b"\x00\x00\x00\x00"  # mask declaring identity tracks
    data_section.data[data_buffer_offset : data_buffer_offset + len(block_payload)] = (
        block_payload
    )
    anim_section.pointer_map[offsets[4]] = (1, data_buffer_offset)
    struct.pack_into(
        "<I",
        anim_section.data,
        offsets[4] + hk.layout.bytes_in_pointer,
        len(block_payload),
    )

    name_storage_offset = 200
    name_bytes = b"TestAnim\0"
    anim_section.data[name_storage_offset : name_storage_offset + len(name_bytes)] = (
        name_bytes
    )
    anim_section.pointer_map[base_offsets[0]] = (0, name_storage_offset)

    return hk


def test_binary_parser_imports_identity_animation():
    hk = _build_identity_animation_header(num_frames=3)
    animation = hk.read_hka_animation(0, 0)

    assert animation["name"] == "TestAnim"
    assert animation["num_frames"] == 3
    assert len(animation["tracks"]) == 1
    assert len(animation["tracks"][0]) == 3

    for frame in animation["tracks"][0]:
        assert frame.position == (0.0, 0.0, 0.0)
        assert frame.rotation == (0.0, 0.0, 0.0, 1.0)
        assert frame.scale == (1.0, 1.0, 1.0)


SPYRO_HKA_PATH = Path(
    "/Users/fabianoppermann/Documents/spyro/spyro/Temporary/BuildServer/ps4/Output/anims/Skylanders/Spyro.hka"
)


@pytest.mark.skipif(
    not SPYRO_HKA_PATH.exists(),
    reason="Spyro.hka test fixture not present on this machine",
)
def test_spyro_pack_can_be_parsed():
    data = SPYRO_HKA_PATH.read_bytes()
    header = hkxHeader()
    header.load(BinaryReader(data))

    variants = header.get_root_level_container()
    assert variants, "expected at least one variant"

    container_ptr = variants[0]["variant_ptr"]
    assert container_ptr is not None
    container = header.read_hka_animation_container(*container_ptr)

    (anim_ptrs, anim_count) = container["animations"]
    assert anim_count > 0

    sid, soff = anim_ptrs
    first_anim_ptr = header.read_pointer(sid, soff)
    assert first_anim_ptr is not None

    animation = header.read_hka_animation(*first_anim_ptr)
    assert animation["tracks"], "expected real animation tracks from Spyro.hka"
