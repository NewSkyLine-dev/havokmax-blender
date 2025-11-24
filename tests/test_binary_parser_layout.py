import struct
import sys
import types

ImportHelper = type("ImportHelper", (), {})
Operator = type("Operator", (ImportHelper,), {})
bpy_stub = types.SimpleNamespace(
    utils=types.SimpleNamespace(register_class=lambda _: None, unregister_class=lambda _: None),
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
    hk.layout = HkxLayout(bytes_in_pointer=4, little_endian=True, reuse_padding=False, empty_base_class=False)
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
    section.data[base_offsets[7] : base_offsets[7] + 4] = (1).to_bytes(4, "little")  # transform tracks
    section.data[base_offsets[6] : base_offsets[6] + 4] = (0).to_bytes(4, "little")  # float tracks
    section.data[offsets[10] : offsets[10] + 4] = (0).to_bytes(4, "little")  # block offset count
    section.data[offsets[11] : offsets[11] + 4] = (0).to_bytes(4, "little")  # num blocks

    animation = hk.read_hka_animation(0, 0)
    assert animation["num_frames"] == 5
    assert animation["frame_duration"] == struct.unpack_from("<f", section.data, offsets[7])[0]
