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

from havok_blender.io.binary_parser import _year_from_version, hkxHeader, BinaryReader


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
