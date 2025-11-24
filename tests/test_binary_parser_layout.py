import math
import struct
import sys
import types
from pathlib import Path
from unittest.mock import patch

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

from havok_blender.io.binary_parser import (
    BinaryReader,
    HkxLayout,
    Section,
    _year_from_version,
    hkxHeader,
)


def test_year_from_version():
    assert _year_from_version("hk_2015.2.0-r1") == 2015
    assert _year_from_version("hk_2019.1.0-r1") == 2019
    assert _year_from_version("unknown") == 2015


def test_header_parses_basic_layout():
    # minimal fake packfile header with one empty section
    magic = b"\x57\xe0\xe0\x57"
    header = magic + b"\x57\xe0\xe0\x57"  # duplicate magic for magic2
    header += struct.pack("<II", 0, 0)  # userTag, version
    header += bytes([8, 1, 0, 0])  # 64-bit pointers, little endian
    header += struct.pack("<iiiii", 1, 0, 0, 0, 0)
    header += b"hk_2015\0" + b"\0" * 8
    header += struct.pack("<Ihh", 0, -1, 0)
    # section header
    header += b"__data\0" + b"\0" * 14
    header += struct.pack("<IIIIIII", len(header) + 8, 0, 0, 0, 0, 0, 0)

    reader = BinaryReader(header + b"\0" * 8)
    hk = hkxHeader()
    hk.load(reader)
    assert hk.layout.bytes_in_pointer == 8
    assert hk.layout.little_endian is True
    assert len(hk.sections) == 1


def test_animation_frame_count_falls_back_to_duration():
    hk = hkxHeader()
    hk.layout = HkxLayout(
        bytes_in_pointer=4,
        little_endian=True,
        reuse_padding=False,
        empty_base_class=False,
    )
    hk.contents_version = "hk_2015"

    data_size = 140
    section = Section(
        tag="__data",
        absolute_data_start=0,
        local_fixups_offset=0,
        global_fixups_offset=0,
        virtual_fixups_offset=0,
        exports_offset=0,
        imports_offset=0,
        buffer_size=data_size,
        data=bytearray(data_size),
        pointer_map={},
    )
    hk.sections = [section]

    offsets = hk._animation_layout()
    base_offsets = hk._base_animation_layout()

    struct.pack_into("<f", section.data, base_offsets[3], 2.0)  # duration seconds
    struct.pack_into("<f", section.data, offsets[7], 0.5)  # frame duration seconds
    section.data[base_offsets[7] : base_offsets[7] + 4] = (1).to_bytes(
        4, "little"
    )  # transform tracks
    section.data[base_offsets[6] : base_offsets[6] + 4] = (0).to_bytes(
        4, "little"
    )  # float tracks
    section.data[offsets[10] : offsets[10] + 4] = (0).to_bytes(
        4, "little"
    )  # block offset count
    section.data[offsets[11] : offsets[11] + 4] = (0).to_bytes(
        4, "little"
    )  # num blocks

    animation = hk.read_hka_animation(0, 0)
    assert animation["num_frames"] == 5
    assert (
        animation["frame_duration"]
        == struct.unpack_from("<f", section.data, offsets[7])[0]
    )


def test_animation_reads_tracks_from_64bit_2014_layout():
    hk = hkxHeader()
    hk.layout = HkxLayout(
        bytes_in_pointer=8,
        little_endian=True,
        reuse_padding=True,
        empty_base_class=True,
    )
    hk.contents_version = "hk_2014.1.0-r1"

    anim_section = Section(
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
    data_section = Section(
        tag="__data",
        absolute_data_start=0,
        local_fixups_offset=0,
        global_fixups_offset=0,
        virtual_fixups_offset=0,
        exports_offset=0,
        imports_offset=0,
        buffer_size=64,
        data=bytearray(64),
        pointer_map={},
    )
    hk.sections = [anim_section, data_section]

    duration = 0.75
    num_transform_tracks = 5
    num_float_tracks = 2

    struct.pack_into("<f", anim_section.data, 16, duration)
    anim_section.data[20:24] = num_transform_tracks.to_bytes(4, "little")
    anim_section.data[24:28] = num_float_tracks.to_bytes(4, "little")

    offsets = hk._animation_layout()
    struct.pack_into("<f", anim_section.data, offsets[1], 0.5)  # block duration
    struct.pack_into("<f", anim_section.data, offsets[2], 2.0)  # inverse duration
    struct.pack_into("<f", anim_section.data, offsets[7], 1.0 / 30.0)
    anim_section.data[offsets[9] : offsets[9] + 4] = (10).to_bytes(4, "little")
    anim_section.data[offsets[10] : offsets[10] + 4] = (1).to_bytes(4, "little")
    anim_section.data[offsets[11] : offsets[11] + 4] = (1).to_bytes(4, "little")
    anim_section.data[offsets[12] : offsets[12] + 4] = (4).to_bytes(4, "little")
    anim_section.data[offsets[15] : offsets[15] + 4] = (10).to_bytes(4, "little")

    anim_section.pointer_map[offsets[3]] = (1, 0)
    anim_section.pointer_map[offsets[4]] = (1, 4)
    data_section.data[0:4] = (0).to_bytes(4, "little")

    with patch("havok_blender.io.binary_parser.SplineDecompressor") as mock_decomp:
        instance = mock_decomp.return_value
        instance.sample_all_tracks.return_value = [
            [] for _ in range(num_transform_tracks)
        ]
        animation = hk.read_hka_animation(0, 0)

    kwargs = mock_decomp.return_value.decompress.call_args.kwargs
    assert kwargs["num_transform_tracks"] == num_transform_tracks
    assert kwargs["num_float_tracks"] == num_float_tracks
    assert math.isclose(animation["duration"], duration)


def test_animation_sampling_can_be_skipped():
    hk = hkxHeader()
    hk.layout = HkxLayout(
        bytes_in_pointer=8,
        little_endian=True,
        reuse_padding=True,
        empty_base_class=True,
    )
    hk.contents_version = "hk_2014.1.0-r1"

    anim_section = Section(
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
    data_section = Section(
        tag="__data",
        absolute_data_start=0,
        local_fixups_offset=0,
        global_fixups_offset=0,
        virtual_fixups_offset=0,
        exports_offset=0,
        imports_offset=0,
        buffer_size=64,
        data=bytearray(64),
        pointer_map={},
    )
    hk.sections = [anim_section, data_section]

    struct.pack_into("<f", anim_section.data, 16, 1.0)
    anim_section.data[20:24] = (1).to_bytes(4, "little")

    offsets = hk._animation_layout()
    struct.pack_into("<f", anim_section.data, offsets[1], 0.5)
    struct.pack_into("<f", anim_section.data, offsets[2], 2.0)
    struct.pack_into("<f", anim_section.data, offsets[7], 1.0 / 30.0)
    anim_section.data[offsets[9] : offsets[9] + 4] = (10).to_bytes(4, "little")
    anim_section.data[offsets[10] : offsets[10] + 4] = (1).to_bytes(4, "little")
    anim_section.data[offsets[11] : offsets[11] + 4] = (1).to_bytes(4, "little")
    anim_section.data[offsets[12] : offsets[12] + 4] = (4).to_bytes(4, "little")
    anim_section.data[offsets[15] : offsets[15] + 4] = (10).to_bytes(4, "little")

    anim_section.pointer_map[offsets[3]] = (1, 0)
    anim_section.pointer_map[offsets[4]] = (1, 4)
    data_section.data[0:4] = (0).to_bytes(4, "little")

    with patch("havok_blender.io.binary_parser.SplineDecompressor") as mock_decomp:
        animation = hk.read_hka_animation(0, 0, sample_tracks=False)

    mock_decomp.assert_not_called()
    assert animation["tracks"] == []


@pytest.mark.skipif(
    not Path(
        "/Users/fabianoppermann/Documents/spyro/spyro/Temporary/BuildServer/ps4/Output/anims/Skylanders/Spyro.hka"
    ).exists(),
    reason="Spyro.hka fixture unavailable",
)
def test_spyro_hka_has_tracks():
    path = Path(
        "/Users/fabianoppermann/Documents/spyro/spyro/Temporary/BuildServer/ps4/Output/anims/Skylanders/Spyro.hka"
    )
    data = path.read_bytes()
    reader = BinaryReader(data)
    hk = hkxHeader()
    hk.load(reader)
    variants = hk.get_root_level_container()
    assert variants, "Expected at least one variant"
    container_ptr = variants[0]["variant_ptr"]
    assert container_ptr is not None
    container = hk.read_hka_animation_container(*container_ptr)
    (anim_ptrs, anim_count) = container["animations"]
    assert anim_count > 0
    sid, soff = anim_ptrs
    ptr_size = hk.layout.bytes_in_pointer
    anim_ptr = hk.read_pointer(sid, soff)
    assert anim_ptr is not None
    anim = hk.read_hka_animation(*anim_ptr)
    assert anim["tracks"], "Expected non-empty transform tracks"
