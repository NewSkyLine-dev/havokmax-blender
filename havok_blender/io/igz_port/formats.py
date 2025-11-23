"""
Classes for handling the various data formats in Skylanders files
"""

import struct
import bpy
import bmesh
from mathutils import Matrix, Vector
from typing import Any

from . import utils
from . import constants

# ------------------------------------------------------------------------------
# Unpack functions for various vertex formats
# ------------------------------------------------------------------------------


def unpack_FLOAT1(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    return [struct.unpack(f"{endian}f", data[element._offset:element._offset + 4])[0], 0.0, 0.0, 1.0]


def unpack_FLOAT2(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    floats = struct.unpack(
        f"{endian}ff", data[element._offset:element._offset + 8])
    return [floats[0], floats[1], 0.0, 1.0]


def unpack_FLOAT3(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    floats = struct.unpack(
        f"{endian}fff", data[element._offset:element._offset + 12])
    return [floats[0], floats[1], floats[2], 1.0]


def unpack_FLOAT4(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    floats = struct.unpack(
        f"{endian}ffff", data[element._offset:element._offset + 16])
    return [floats[0], floats[1], floats[2], floats[3]]


def unpack_INT1(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}i", data[element._offset:element._offset + 4])
    return [float(ints[0]), 0.0, 0.0, 1.0]


def unpack_INT2(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}ii", data[element._offset:element._offset + 8])
    return [float(ints[0]), float(ints[1]), 0.0, 1.0]


def unpack_INT3(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}iii", data[element._offset:element._offset + 12])
    return [float(ints[0]), float(ints[1]), float(ints[2]), 1.0]


def unpack_INT4(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}iiii", data[element._offset:element._offset + 16])
    return [float(ints[0]), float(ints[1]), float(ints[2]), float(ints[3])]


def unpack_INT1N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}i", data[element._offset:element._offset + 4])
    return [float(ints[0]) / 0x7FFFFFFF, 0.0, 0.0, 1.0]


def unpack_INT2N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}ii", data[element._offset:element._offset + 8])
    return [float(ints[0]) / 0x7FFFFFFF, float(ints[1]) / 0x7FFFFFFF, 0.0, 1.0]


def unpack_INT3N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}iii", data[element._offset:element._offset + 12])
    return [float(ints[0]) / 0x7FFFFFFF, float(ints[1]) / 0x7FFFFFFF, float(ints[2]) / 0x7FFFFFFF, 1.0]


def unpack_INT4N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}iiii", data[element._offset:element._offset + 16])
    return [float(ints[0]) / 0x7FFFFFFF, float(ints[1]) / 0x7FFFFFFF, float(ints[2]) / 0x7FFFFFFF, float(ints[3]) / 0x7FFFFFFF]


def unpack_UINT1(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}I", data[element._offset:element._offset + 4])
    return [float(ints[0]), 0.0, 0.0, 1.0]


def unpack_UINT2(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}II", data[element._offset:element._offset + 8])
    return [float(ints[0]), float(ints[1]), 0.0, 1.0]


def unpack_UINT3(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}III", data[element._offset:element._offset + 12])
    return [float(ints[0]), float(ints[1]), float(ints[2]), 1.0]


def unpack_UINT4(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}IIII", data[element._offset:element._offset + 16])
    return [float(ints[0]), float(ints[1]), float(ints[2]), float(ints[3])]


def unpack_UINT1N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}I", data[element._offset:element._offset + 4])
    return [float(ints[0]) / 0xFFFFFFFF, 0.0, 0.0, 1.0]


def unpack_UINT2N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}II", data[element._offset:element._offset + 8])
    return [float(ints[0]) / 0xFFFFFFFF, float(ints[1]) / 0xFFFFFFFF, 0.0, 1.0]


def unpack_UINT3N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}III", data[element._offset:element._offset + 12])
    return [float(ints[0]) / 0xFFFFFFFF, float(ints[1]) / 0xFFFFFFFF, float(ints[2]) / 0xFFFFFFFF, 1.0]


def unpack_UINT4N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ints = struct.unpack(
        f"{endian}IIII", data[element._offset:element._offset + 16])
    return [float(ints[0]) / 0xFFFFFFFF, float(ints[1]) / 0xFFFFFFFF, float(ints[2]) / 0xFFFFFFFF, float(ints[3]) / 0xFFFFFFFF]


def unpack_SHORT1(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    shorts = struct.unpack(
        f"{endian}h", data[element._offset:element._offset + 2])
    return [float(shorts[0]), 0.0, 0.0, 1.0]


def unpack_SHORT2(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    shorts = struct.unpack(
        f"{endian}hh", data[element._offset:element._offset + 4])
    return [float(shorts[0]), float(shorts[1]), 0.0, 1.0]


def unpack_SHORT3(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    shorts = struct.unpack(
        f"{endian}hhh", data[element._offset:element._offset + 6])
    return [float(shorts[0]), float(shorts[1]), float(shorts[2]), 1.0]


def unpack_SHORT4(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    shorts = struct.unpack(
        f"{endian}hhhh", data[element._offset:element._offset + 8])
    return [float(shorts[0]), float(shorts[1]), float(shorts[2]), float(shorts[3])]


def unpack_SHORT1N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    shorts = struct.unpack(
        f"{endian}h", data[element._offset:element._offset + 2])
    return [float(shorts[0]) / 0x7FFF, 0.0, 0.0, 1.0]


def unpack_SHORT2N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    shorts = struct.unpack(
        f"{endian}hh", data[element._offset:element._offset + 4])
    return [float(shorts[0]) / 0x7FFF, float(shorts[1]) / 0x7FFF, 0.0, 1.0]


def unpack_SHORT3N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    shorts = struct.unpack(
        f"{endian}hhh", data[element._offset:element._offset + 6])
    return [float(shorts[0]) / 0x7FFF, float(shorts[1]) / 0x7FFF, float(shorts[2]) / 0x7FFF, 1.0]


def unpack_SHORT4N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    shorts = struct.unpack(
        f"{endian}hhhh", data[element._offset:element._offset + 8])
    return [float(shorts[0]) / 0x7FFF, float(shorts[1]) / 0x7FFF, float(shorts[2]) / 0x7FFF, float(shorts[3]) / 0x7FFF]


def unpack_USHORT1(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ushorts = struct.unpack(
        f"{endian}H", data[element._offset:element._offset + 2])
    return [float(ushorts[0]), 0.0, 0.0, 1.0]


def unpack_USHORT2(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ushorts = struct.unpack(
        f"{endian}HH", data[element._offset:element._offset + 4])
    return [float(ushorts[0]), float(ushorts[1]), 0.0, 1.0]


def unpack_USHORT3(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ushorts = struct.unpack(
        f"{endian}HHH", data[element._offset:element._offset + 6])
    return [float(ushorts[0]), float(ushorts[1]), float(ushorts[2]), 1.0]


def unpack_USHORT4(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ushorts = struct.unpack(
        f"{endian}HHHH", data[element._offset:element._offset + 8])
    return [float(ushorts[0]), float(ushorts[1]), float(ushorts[2]), float(ushorts[3])]


def unpack_USHORT1N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ushorts = struct.unpack(
        f"{endian}H", data[element._offset:element._offset + 2])
    return [float(ushorts[0]) / 0xFFFF, 0.0, 0.0, 1.0]


def unpack_USHORT2N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ushorts = struct.unpack(
        f"{endian}HH", data[element._offset:element._offset + 4])
    return [float(ushorts[0]) / 0xFFFF, float(ushorts[1]) / 0xFFFF, 0.0, 1.0]


def unpack_USHORT3N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ushorts = struct.unpack(
        f"{endian}HHH", data[element._offset:element._offset + 6])
    return [float(ushorts[0]) / 0xFFFF, float(ushorts[1]) / 0xFFFF, float(ushorts[2]) / 0xFFFF, 1.0]


def unpack_USHORT4N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    ushorts = struct.unpack(
        f"{endian}HHHH", data[element._offset:element._offset + 8])
    return [float(ushorts[0]) / 0xFFFF, float(ushorts[1]) / 0xFFFF, float(ushorts[2]) / 0xFFFF, float(ushorts[3]) / 0xFFFF]


def unpack_BYTE1(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    sbytes = struct.unpack(
        f"{endian}b", data[element._offset:element._offset + 1])
    return [float(sbytes[0]), 0.0, 0.0, 1.0]


def unpack_BYTE2(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    sbytes = struct.unpack(
        f"{endian}bb", data[element._offset:element._offset + 2])
    return [float(sbytes[0]), float(sbytes[1]), 0.0, 1.0]


def unpack_BYTE3(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    sbytes = struct.unpack(
        f"{endian}bbb", data[element._offset:element._offset + 3])
    return [float(sbytes[0]), float(sbytes[1]), float(sbytes[2]), 1.0]


def unpack_BYTE4(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    sbytes = struct.unpack(
        f"{endian}bbbb", data[element._offset:element._offset + 4])
    return [float(sbytes[0]), float(sbytes[1]), float(sbytes[2]), float(sbytes[3])]


def unpack_BYTE1N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    sbytes = struct.unpack(
        f"{endian}b", data[element._offset:element._offset + 1])
    return [float(sbytes[0]) / 0x7F, 0.0, 0.0, 1.0]


def unpack_BYTE2N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    sbytes = struct.unpack(
        f"{endian}bb", data[element._offset:element._offset + 2])
    return [float(sbytes[0]) / 0x7F, float(sbytes[1]) / 0x7F, 0.0, 1.0]


def unpack_BYTE3N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    sbytes = struct.unpack(
        f"{endian}bbb", data[element._offset:element._offset + 3])
    return [float(sbytes[0]) / 0x7F, float(sbytes[1]) / 0x7F, float(sbytes[2]) / 0x7F, 1.0]


def unpack_BYTE4N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    sbytes = struct.unpack(
        f"{endian}bbbb", data[element._offset:element._offset + 4])
    return [float(sbytes[0]) / 0x7F, float(sbytes[1]) / 0x7F, float(sbytes[2]) / 0x7F, float(sbytes[3]) / 0x7F]


def unpack_UBYTE1(data: bytes, element: Any, endarg: str) -> list[float]:
    return [float(data[element._offset + 0]), 0.0, 0.0, 1.0]


def unpack_UBYTE2(data: bytes, element: Any, endarg: str) -> list[float]:
    return [float(data[element._offset + 0]), float(data[element._offset + 1]), 0.0, 1.0]


def unpack_UBYTE3(data: bytes, element: Any, endarg: str) -> list[float]:
    return [float(data[element._offset + 0]), float(data[element._offset + 1]), float(data[element._offset + 2]), 1.0]


def unpack_UBYTE4(data: bytes, element: Any, endarg: str) -> list[float]:
    return [float(data[element._offset + 0]), float(data[element._offset + 1]), float(data[element._offset + 2]), float(data[element._offset + 3])]


def unpack_UBYTE4_ENDIAN(data: bytes, element: Any, endarg: str) -> list[float]:
    return [float(data[element._offset + 3]), float(data[element._offset + 2]), float(data[element._offset + 1]), float(data[element._offset + 0])]


def unpack_UBYTE1N(data: bytes, element: Any, endarg: str) -> list[float]:
    return [float(data[element._offset + 0] / 0xFF), 0.0, 0.0, 1.0]


def unpack_UBYTE2N(data: bytes, element: Any, endarg: str) -> list[float]:
    return [float(data[element._offset + 0] / 0xFF), float(data[element._offset + 1] / 0xFF), 0.0, 1.0]


def unpack_UBYTE3N(data: bytes, element: Any, endarg: str) -> list[float]:
    return [float(data[element._offset + 0] / 0xFF), float(data[element._offset + 1] / 0xFF), float(data[element._offset + 2] / 0xFF), 1.0]


def unpack_UBYTE4N(data: bytes, element: Any, endarg: str) -> list[float]:
    return [float(data[element._offset + 0]) / 0xFF, float(data[element._offset + 1]) / 0xFF, float(data[element._offset + 2]) / 0xFF, float(data[element._offset + 3]) / 0xFF]


def unpack_UBYTE4_X4(data: bytes, element: Any, endarg: str) -> list[float]:
    return [float(data[element._offset + 0] * 0.25), float(data[element._offset + 1] * 0.25), float(data[element._offset + 2] * 0.25), float(data[element._offset + 3] * 0.25)]


def unpack_HALF2(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    bs = utils.NoeBitStream(
        data, constants.Endianness.BIG if endian == '>' else constants.Endianness.LITTLE)
    bs.seek(element._offset)
    return [float(bs.readHalfFloat()), float(bs.readHalfFloat()), 0.0, 1.0]


def unpack_HALF4(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    bs = utils.NoeBitStream(
        data, constants.Endianness.BIG if endian == '>' else constants.Endianness.LITTLE)
    bs.seek(element._offset)
    return [float(bs.readHalfFloat()), float(bs.readHalfFloat()), float(bs.readHalfFloat()), float(bs.readHalfFloat())]


def unpack_UNUSED(data: bytes, element: Any, endarg: str) -> list[float]:
    vec4 = unpack_SHORT4(data, element, endarg)
    return [vec4[0] / vec4[3], vec4[1] / vec4[3], vec4[2] / vec4[3], vec4[3]]


def unpack_UNDEFINED_0(data: bytes, element: Any, endarg: str) -> list[float]:
    print("Got IG_VERTEX_TYPE_UNDEFINED_0")
    return [0.0, 0.0, 0.0, 0.0]


def unpack_UBYTE4N_COLOR_ARGB(data: bytes, element: Any, endarg: str) -> list[float]:
    color = unpack_UBYTE4N(data, element, endarg)
    return [color[1], color[2], color[3], color[0]]


def unpack_UBYTE2N_COLOR_5650(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    color = struct.unpack(
        f"{endian}H", data[element._offset:element._offset + 2])[0]
    return [((color >> 11) & 31) / 31, ((color >> 5) & 63) / 63, (color & 31) / 31, 1]


def unpack_UBYTE2N_COLOR_5551(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    color = struct.unpack(
        f"{endian}H", data[element._offset:element._offset + 2])[0]
    return [(color & 31) / 31, ((color >> 5) & 31) / 31, ((color >> 10) & 31) / 31, (color >> 15) & 1]


def unpack_UBYTE2N_COLOR_4444(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    color = struct.unpack(
        f"{endian}H", data[element._offset:element._offset + 2])[0]
    return [(color & 15) / 15, ((color >> 4) & 15) / 15, ((color >> 8) & 15) / 15, ((color >> 12) & 15) / 15]


def unpack_UDEC3(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    raw = struct.unpack(
        f"{endian}I", data[element._offset:element._offset + 4])[0]
    return [float(raw & 0x3FF), float((raw >> 10) & 0x3FF), float((raw >> 20) & 0x3FF), 1.0]


def unpack_UDEC3_OES(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    raw = struct.unpack(
        f"{endian}I", data[element._offset:element._offset + 4])[0]
    return [float(raw >> 22), float((raw >> 12) & 0x3FF), float((raw >> 2) & 0x3FF), 1.0]


def unpack_DEC3N(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    raw = struct.unpack(
        f"{endian}I", data[element._offset:element._offset + 4])[0]
    return [
        float((((raw >> 0) & 0x1FF) / 511.0) *
              (1 if (raw >> 0) & 0x200 == 0 else -1)),
        float((((raw >> 10) & 0x1FF) / 511.0) *
              (1 if (raw >> 10) & 0x200 == 0 else -1)),
        float((((raw >> 20) & 0x1FF) / 511.0) *
              (1 if (raw >> 20) & 0x200 == 0 else -1)),
        1.0
    ]


def unpack_DEC3N_OES(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    raw = struct.unpack(
        f"{endian}I", data[element._offset:element._offset + 4])[0]
    return [
        float((((raw >> 2) & 0x1FF) / 511.0) *
              (1 if (raw >> 2) & 0x200 == 0 else -1)),
        float((((raw >> 12) & 0x1FF) / 511.0) *
              (1 if (raw >> 12) & 0x200 == 0 else -1)),
        float((((raw >> 22) & 0x1FF) / 511.0) *
              (1 if (raw >> 22) & 0x200 == 0 else -1)),
        1.0
    ]


def unpack_DEC3N_S11_11_10(data: bytes, element: Any, endarg: str) -> list[float]:
    endian = endarg.value if hasattr(endarg, 'value') else endarg
    raw = struct.unpack(
        f"{endian}I", data[element._offset:element._offset + 4])[0]
    return [
        float((((raw >> 0) & 0x3FF) / 1023.0) *
              (1 if (raw >> 0) & 0x400 == 0 else -1)),
        float((((raw >> 11) & 0x3FF) / 1023.0) *
              (1 if (raw >> 11) & 0x400 == 0 else -1)),
        float((((raw >> 22) & 0x1FF) / 511.0) *
              (1 if (raw >> 22) & 0x200 == 0 else -1)),
        1.0
    ]

# Edge geometry unpack functions


def edgeUnpack_I16N(data: bytes, offset: int) -> bytes:
    return struct.pack('>f', float(struct.unpack('>h', data[offset:offset + 2])[0]) / 0x7FFF)


def edgeUnpack_F32(data: bytes, offset: int) -> bytes:
    return data[offset:offset + 4]


def edgeUnpack_F16(data: bytes, offset: int) -> bytes:
    bs = utils.NoeBitStream(data, constants.Endianness.BIG)
    bs.seek(offset)
    return struct.pack('>f', float(bs.readHalfFloat()))


def edgeUnpack_U8N(data: bytes, offset: int) -> bytes:
    return struct.pack('>f', float(data[offset]) / 0x7F)


def edgeUnpack_I16(data: bytes, offset: int) -> bytes:
    return struct.pack('>f', float(struct.unpack('>h', data[offset:offset + 2])[0]))


def edgeUnpack_X11Y11Z10N(data: bytes, offset: int) -> bytes:
    raw = struct.unpack('>I', data[offset:offset + 4])[0]
    return struct.pack('>fff', ((raw & 0x000007FF) >> 0) / 0x7FF, ((raw & 0x003FF800) >> 11) / 0x7FF, ((raw & 0xFFC00000) >> 22) / 0x3FF)


def edgeUnpack_U8(data: bytes, offset: int) -> float:
    return float(data[offset])


# Edge unpacking functions array
edgeUnpackFunctions = [
    (None, 0),
    (edgeUnpack_I16N, 2),
    (edgeUnpack_F32, 4),
    (edgeUnpack_F16, 2),
    (edgeUnpack_U8N, 1),
    (edgeUnpack_I16, 2),
    (edgeUnpack_X11Y11Z10N, 4),
    (edgeUnpack_U8, 1),
    (None, 0),  # edgeUnpack_FIXED_POINT,
    (None, 0),  # edgeUnpack_UNIT_VECTOR,
]

# Vertex unpacking functions dictionary for Superchargers
sscvertexUnpackFunctions = [
    unpack_FLOAT1,
    unpack_FLOAT2,
    unpack_FLOAT3,
    unpack_FLOAT4,
    unpack_UBYTE4N,    # UBYTE4N_COLOR is identical to UBYTE4N
    unpack_UBYTE4N_COLOR_ARGB,
    unpack_UBYTE4N,    # unpack_UBYTE4N_COLOR_RGBA,
    unpack_UNDEFINED_0,
    unpack_UBYTE2N_COLOR_5650,
    unpack_UBYTE2N_COLOR_5551,
    unpack_UBYTE2N_COLOR_4444,
    unpack_INT1,
    unpack_INT2,
    unpack_INT4,
    unpack_UINT1,
    unpack_UINT2,
    unpack_UINT4,
    unpack_INT1N,
    unpack_INT2N,
    unpack_INT4N,
    unpack_UINT1N,
    unpack_UINT2N,
    unpack_UINT4N,
    unpack_UBYTE4,
    unpack_UBYTE4_X4,
    unpack_BYTE4,
    unpack_UBYTE4N,
    unpack_UNDEFINED_0,  # unpack_UNDEFINED_1,
    unpack_BYTE4N,
    unpack_SHORT2,
    unpack_SHORT4,
    unpack_USHORT2,
    unpack_USHORT4,
    unpack_SHORT2N,
    unpack_SHORT3N,
    unpack_SHORT4N,
    unpack_USHORT2N,
    unpack_USHORT3N,
    unpack_USHORT4N,
    unpack_UDEC3,
    unpack_DEC3N,
    unpack_DEC3N_S11_11_10,
    unpack_HALF2,
    unpack_HALF4,
    unpack_UNUSED,
    unpack_BYTE3N,
    unpack_SHORT3,
    unpack_USHORT3,
    unpack_UBYTE4_ENDIAN,
    unpack_UBYTE4,     # unpack_UBYTE4_COLOR,
    unpack_BYTE3,
    unpack_UBYTE2N_COLOR_5650,  # unpack_UBYTE2N_COLOR_5650_RGB,
    unpack_UDEC3_OES,
    unpack_DEC3N_OES,
    unpack_SHORT4N,    # unpack_SHORT4N_EDGE, identical to unpack_SHORT4N
    unpack_UNDEFINED_0  # unpack_MAX
]

# ------------------------------------------------------------------------------
# Classes for data structures in Skylanders files
# ------------------------------------------------------------------------------


class EdgeGeometryVertexDescriptor:
    def __init__(self, data):
        self.count = 0
        self.vertexStride = 0
        self.elements = []
        if len(data) != 0:
            self.count = data[0]
            self.vertexStride = data[1]
            print(f"count:  {self.count}")
            print(f"stride: {self.vertexStride}")
            for i in range(self.count):
                print(f"  processing element: {i}")
                attributeBlock = EdgeGeometryAttributeBlock()
                attributeBlock.readFromFile(data[(i+1)*0x08:(i+2)*0x08])
                self.elements.append(attributeBlock)


class EdgeGeometryAttributeBlock:
    def __init__(self):
        self.offset = 0
        self.format = 0                  # See Formats section of PS3 Reference
        self.componentCount = 0
        self.edgeAttributeId = 0         # See Attribute Ids section of PS3 Reference
        self.size = 0
        self.vertexProgramSlotIndex = 0
        self.fixedBlockOffset = 0
        self.padding = 0

    def readFromFile(self, data):
        self.offset = data[0]
        self.format = data[1]
        self.componentCount = data[2]
        self.edgeAttributeId = data[3]
        self.size = data[4]
        self.vertexProgramSlotIndex = data[5]
        self.fixedBlockOffset = data[6]
        self.padding = data[7]

    def unpack(self, vertexBuffer, vertexCount, stride):
        vattributes = []
        for i in range(vertexCount):
            vattributes.extend(self.unpackVertex(
                vertexBuffer[stride * i: stride * (i + 1)]))
        return bytes(vattributes)

    def unpackVertex(self, data):
        ret = []

        if self.edgeAttributeId == 1 and self.componentCount == 4:
            raw = struct.unpack('>hhhh', data[self.offset:self.offset+8])
            ret.extend(bytes(struct.pack(
                '>ffff', raw[0] / raw[3], raw[1] / raw[3], raw[2] / raw[3], raw[3])))
            return ret

        componentSize = 0
        if self.format < 10:
            unpackFunction = edgeUnpackFunctions[self.format][0]
            componentSize = edgeUnpackFunctions[self.format][1]
        else:
            print(f"unimplemented format type: {self.format}")
            return ret

        for i in range(4):
            if i < self.componentCount:
                ret.extend(bytes(unpackFunction(
                    data, self.offset + componentSize * i)))
            elif i == 3:
                ret.extend(bytes(struct.pack('>f', 1.0)))
            else:
                ret.extend(bytes(struct.pack('>f', 0.0)))

        return ret


class EdgeGeomSpuConfigInfo:
    def __init__(self, data):
        self.flagsAndUniformTableCount = data[0]
        self.commandBufferHoleSize = data[1]
        self.inputVertexFormatId = data[2]
        self.secondaryInputVertexFormatId = data[3]
        self.outputVertexFormatId = data[4]
        self.vertexDeltaFormatId = data[5]
        self.indexesFlavorAndSkinningFlavor = data[6]
        self.skinningMatrixFormat = data[7]
        self.numVertexes = struct.unpack('>H', data[8:10])[0]
        self.numIndexes = struct.unpack('>H', data[10:12])[0]
        self.indexesOffset = struct.unpack('>I', data[12:16])[0]

        # Not part of this struct, but needed for storage
        self.skinMatrixOffset0 = 0
        self.skinMatrixOffset1 = 0
        self.skinMatrixSize0 = 0
        self.skinMatrixSize1 = 0


class igVertexElement:
    def __init__(self, data, endarg):
        self._type = data[0]
        self._stream = data[1]
        self._mapToElement = data[2]
        self._count = data[3]
        self._usage = data[4]
        self._usageIndex = data[5]
        self._packDataOffset = data[6]
        self._packTypeAndFracHint = data[7]
        self._offset = struct.unpack(f"{endarg}H", data[8:10])[0]
        self._freq = struct.unpack(f"{endarg}H", data[10:12])[0]

    def unpack(self, vertexBuffer, stride, packData, endarg, debugPrint=False):
        vattributes = []

        scale = 1
        if (self._packTypeAndFracHint & 7) == 2 and packData is not None:
            scale /= 1 << struct.unpack(f"{endarg}I", bytes(
                packData[self._packDataOffset:self._packDataOffset + 4]))[0]
            print(f"scale is 1 / {1 / scale}")

        magnitude = 0
        for i in range(len(vertexBuffer) // stride):
            attribute = sscvertexUnpackFunctions[self._type](
                vertexBuffer[i * stride:(i + 1) * stride], self, endarg)
            if debugPrint:
                currMag = (attribute[0]*attribute[0] + attribute[1]
                           * attribute[1] + attribute[2]*attribute[2])
                print(attribute)
                if magnitude < currMag:
                    magnitude = currMag
            vattributes.extend(bytes(struct.pack(
                f"{endarg}ffff", attribute[0] * scale, attribute[1] * scale, attribute[2] * scale, attribute[3])))
        if debugPrint:
            print(f"magnitude: {magnitude * (scale*scale)}")
        return bytes(vattributes)

    def getElemNormaliser(self):
        return constants.vertexMaxMags[self._type]


class PS3MeshObject:
    def __init__(self):
        self.vertexBuffers = []
        self.vertexStrides = []
        self.vertexCount = 0
        self.indexBuffer = None
        self.spuConfigInfo = None
        self.vertexElements = []
        self.indexCount = None
        self.boneMapIndex = None

    def getBufferForAttribute(self, attributeId):
        if attributeId == 1:  # Position attribute
            if self.vertexElements[0].count == 0:
                elem = EdgeGeometryAttributeBlock()
                elem.componentCount = 3
                elem.format = 2
                elem.offset = 0
                elem.edgeAttributeId = 1
                return elem.unpack(self.vertexBuffers[0], self.vertexCount, 0x0C)

        for i in range(3):
            if self.vertexElements[i].count != 0:
                for elem in self.vertexElements[i].elements:
                    if elem.edgeAttributeId == attributeId:
                        # print(f"stream: {hex(i)}; attr: {hex(elem.edgeAttributeId)}; offset: {hex(elem.offset)}; format: {hex(elem.format)}; componentCount: {hex(elem.componentCount)}; size: {hex(elem.size)}; vertexProgramSlotIndex: {hex(elem.vertexProgramSlotIndex)}; fixedBlockOffset: {hex(elem.fixedBlockOffset)}")
                        unpackedBuffer = elem.unpack(
                            self.vertexBuffers[i], self.vertexCount, self.vertexStrides[i])
                        return unpackedBuffer

        return None

    def getPs3BoneStuff(self):
        skinningFlags = self.spuConfigInfo.indexesFlavorAndSkinningFlavor & 0xF
        if skinningFlags == constants.EdgeGeomSkinType.NONE:
            return None

        useOneBone = skinningFlags in [constants.EdgeGeomSkinType.SINGLE_BONE_NO_SCALING,
                                       constants.EdgeGeomSkinType.SINGLE_BONE_UNIFORM_SCALING,
                                       constants.EdgeGeomSkinType.SINGLE_BONE_NON_UNIFORM_SCALING]

        boneMapOffset0 = self.spuConfigInfo.skinMatrixOffset0 // 0x30
        boneMapOffset1 = self.spuConfigInfo.skinMatrixOffset1 // 0x30
        boneMapSize0 = self.spuConfigInfo.skinMatrixSize0 // 0x30

        vertexCount = self.vertexCount
        # build the buffers
        skinBuffer = self.vertexBuffers[3]
        highestIndex = 0

        if useOneBone:
            bwBuffer = [0xFF, 0x00, 0x00, 0x00] * vertexCount
            biBuffer = []
            for i in range(vertexCount):
                biBuffer.extend(
                    [skinBuffer[i] + boneMapOffset0, 0x00, 0x00, 0x00])
        else:
            bwBuffer = []
            biBuffer = []
            for i in range(vertexCount):
                for j in range(4):
                    weightIdx = i*8+j*2+0
                    # Or try i*8+(j+1)*2+1 if weights are shifted
                    boneIdx = i*8+(j+1)*2+1  # i*8+j*2+1

                    bwBuffer.append(skinBuffer[weightIdx])

                    boneIndex = skinBuffer[boneIdx]
                    if boneIndex < boneMapSize0:
                        boneIndex += boneMapOffset0
                    else:
                        boneIndex += boneMapOffset1 - boneMapSize0

                    biBuffer.append(boneIndex)

                    if skinBuffer[i*8+j*2+1] > highestIndex:
                        highestIndex = skinBuffer[i*8+j*2+1]

        return (bwBuffer, biBuffer)


class MeshObject:
    def __init__(self):
        self.name = ""
        self.vertexBuffers = []
        self.vertexStrides = []
        self.vertexCount = 0
        self.indexBuffer = None
        self.isPs3 = False
        self.ps3Segments = []
        self.spuConfigInfo = False  # PS3 Exclusive
        self.skipBuild = False
        self.vertexElements = []
        self.vertexStreams = []
        self.primType = constants.PrimitiveType.TRIANGLE
        self.indexCount = 0
        self.boneMapIndex = 0
        self.transformation = None
        self.packData = None
        self.platform = 0
        self.platformData = None

        # For Blender mesh construction
        self.vertices = []
        self.faces = []
        self.normals = []
        self.uvs = []
        self.colors = []
        self.weights = []
        self.boneIndices = []

    def buildMesh(self, boneMapList, endianness, version, platform):
        if self.vertexCount == 0:
            return

        endarg = '>' if endianness == "BE" else '<'

        print(f"vertex count    {hex(self.vertexCount)}")
        print(f"vertex stride   {hex(self.vertexStrides[0])}")
        print(f"index count     {hex(self.indexCount)}")
        print(f"name:           {self.name}")
        print(f"bone map index: {hex(self.boneMapIndex)}")

        # Process vertex data
        if platform == 2 and struct.unpack(">H", self.vertexBuffers[0][0:2])[0] == 0x9F:
            self.vertexBuffers[0] = bytes(self.vertexBuffers[0][4:])

        if version >= 6:
            packData = self.packData[2] if self.packData is not None else None
        else:
            packDataOffset = 0
            for elem in self.vertexElements:
                if elem._type == 0x2C:
                    continue
                print(f"processing... {elem._packDataOffset}")
                if (elem._packTypeAndFracHint & 7) == 2:
                    if packDataOffset < elem._packDataOffset:
                        packDataOffset = elem._packDataOffset
            packData = bytes(self.vertexBuffers[0][len(
                self.vertexBuffers[0]) - packDataOffset - 4:])

        for elem in self.vertexElements:
            if elem._type == 0x2C:
                continue

            streamOffset = 0
            for i in range(elem._stream):
                streamOffset += (((self.vertexStreams[i] *
                                 self.vertexCount) + 0x1F) // 0x20) * 0x20
            print(
                f"Getting bytes for stream from {hex(streamOffset)} to {hex(streamOffset + self.vertexCount * self.vertexStreams[elem._stream])}")
            stream = bytes(self.vertexBuffers[0][streamOffset:streamOffset +
                           self.vertexCount * self.vertexStreams[elem._stream]])
            streamSize = self.vertexStreams[elem._stream]

            print(f"usage: {hex(elem._usage)}; offset: {hex(elem._offset)}; stream: {hex(elem._stream)}; count: {hex(elem._count)}; type: {hex(elem._type)}; mapToElement: {hex(elem._mapToElement)}; usageIndex: {hex(elem._usageIndex)}; packDataOffset: {hex(elem._packDataOffset)}; packTypeAndFracHint: {hex(elem._packTypeAndFracHint)}; freq: {hex(elem._freq)}; streamOffset: {hex(streamOffset)}")

            if elem._usage == 0:  # IG_VERTEX_USAGE_POSITION
                stride = 0x10
                if elem._type == 0x23:
                    fakeVertexBuffer = self.superchargersFunkiness(endarg)
                    stride = 0x0C
                else:
                    fakeVertexBuffer = elem.unpack(
                        stream, streamSize, packData, endarg)

                # Extract positions
                for i in range(self.vertexCount):
                    x, y, z = struct.unpack(
                        endarg + 'fff', fakeVertexBuffer[i * stride:i * stride + 12])
                    self.vertices.append((x, y, z))

            if elem._usage == 1:  # IG_VERTEX_USAGE_NORMAL
                vnormals = elem.unpack(stream, streamSize, packData, endarg)

                # Extract normals
                for i in range(self.vertexCount):
                    nx, ny, nz = struct.unpack(
                        endarg + 'fff', vnormals[i * 0x10:i * 0x10 + 12])
                    self.normals.append((nx, ny, nz))

            if elem._usage == 4:  # IG_VERTEX_USAGE_COLOR
                vcolors = elem.unpack(stream, streamSize, packData, endarg)

                # Extract colors
                for i in range(self.vertexCount):
                    r, g, b, a = struct.unpack(
                        endarg + 'ffff', vcolors[i * 0x10:i * 0x10 + 16])
                    self.colors.append((r, g, b, a))

            if elem._usage == 5 and elem._usageIndex == 0:  # IG_VERTEX_USAGE_TEXCOORD
                vtexcoords = elem.unpack(stream, streamSize, packData, endarg)

                # Extract UVs
                for i in range(self.vertexCount):
                    u = struct.unpack(
                        endarg + 'f', vtexcoords[i * 0x10:i * 0x10 + 4])[0]
                    v = struct.unpack(
                        endarg + 'f', vtexcoords[i * 0x10 + 4:i * 0x10 + 8])[0]
                    self.uvs.append((u, v))

            if elem._usage == 6 and elem._usageIndex == 0 and constants.dBuildBones:  # IG_VERTEX_USAGE_BLENDWEIGHTS
                vblendweights = elem.unpack(
                    stream, streamSize, packData, endarg)

                # Extract weights
                for i in range(self.vertexCount):
                    weights = []
                    for j in range(elem._count):
                        weight = struct.unpack(
                            endarg + 'f', vblendweights[i * 0x10 + j * 4:i * 0x10 + (j + 1) * 4])[0]
                        weights.append(weight)
                    # Pad with zeros if needed
                    while len(weights) < 4:
                        weights.append(0.0)
                    self.weights.append(weights)

            if elem._usage == 8 and elem._usageIndex == 0 and constants.dBuildBones:  # IG_VERTEX_USAGE_BLENDINDICES
                vfblendindices = elem.unpack(
                    stream, streamSize, packData, endarg)

                # Extract bone indices
                for i in range(self.vertexCount):
                    indices = []
                    for j in range(elem._count):
                        index = int(struct.unpack(
                            endarg + 'f', vfblendindices[i * 0x10 + j * 4:i * 0x10 + (j + 1) * 4])[0])
                        indices.append(index)
                    # Pad with zeros if needed
                    while len(indices) < 4:
                        indices.append(0)
                    self.boneIndices.append(indices)

        # Process index data
        if constants.dBuildFaces and self.primType != constants.PrimitiveType.TRIANGLE_STRIP:
            if self.vertexCount <= 0xFFFF:
                # Extract triangles from 16-bit indices
                for i in range(0, self.indexCount, 3):
                    if i + 2 < self.indexCount:
                        idx1 = struct.unpack(
                            endarg + 'H', self.indexBuffer[i * 2:(i + 1) * 2])[0]
                        idx2 = struct.unpack(
                            endarg + 'H', self.indexBuffer[(i + 1) * 2:(i + 2) * 2])[0]
                        idx3 = struct.unpack(
                            endarg + 'H', self.indexBuffer[(i + 2) * 2:(i + 3) * 2])[0]
                        self.faces.append((idx1, idx2, idx3))
            else:
                # Extract triangles from 32-bit indices
                for i in range(0, self.indexCount, 3):
                    if i + 2 < self.indexCount:
                        idx1 = struct.unpack(
                            endarg + 'I', self.indexBuffer[i * 4:(i + 1) * 4])[0]
                        idx2 = struct.unpack(
                            endarg + 'I', self.indexBuffer[(i + 1) * 4:(i + 2) * 4])[0]
                        idx3 = struct.unpack(
                            endarg + 'I', self.indexBuffer[(i + 2) * 4:(i + 3) * 4])[0]
                        self.faces.append((idx1, idx2, idx3))
        elif constants.dBuildFaces and self.primType == constants.PrimitiveType.TRIANGLE_STRIP:
            # Handle triangle strips - simplified for now
            # A proper implementation would convert strips to triangles
            pass

    def buildPs3MeshNew(self, boneMapList, version):
        # Simplified PS3 mesh processing for Blender
        print(f"Building PS3 mesh {self.name}")

        # Get position buffer
        vPositions = self.buildBatchedPS3VertexBuffer(1)
        if vPositions:
            # Extract positions
            for i in range(len(vPositions) // 16):
                x, y, z = struct.unpack('>fff', vPositions[i * 16:i * 16 + 12])
                self.vertices.append((x, y, z))

        # Get UVs if available
        vUV0 = self.buildBatchedPS3VertexBuffer(5)
        if vUV0:
            for i in range(len(vUV0) // 16):
                u, v = struct.unpack('>ff', vUV0[i * 16:i * 16 + 8])
                self.uvs.append((u, v))

        # Get colors if available
        vColor = self.buildBatchedPS3VertexBuffer(9)
        if vColor:
            for i in range(len(vColor) // 16):
                r, g, b, a = struct.unpack('>ffff', vColor[i * 16:i * 16 + 16])
                self.colors.append((r, g, b, a))

        # Handle bones if available
        if constants.dBuildBones and len(boneMapList) > 0 and len(boneMapList[self.boneMapIndex]) > 0:
            boneBuffers = self.buildBatchedPs3BoneBuffers()
            if boneBuffers:
                weight_data = boneBuffers[0]
                index_data = boneBuffers[1]

                # Extract weights and indices
                for i in range(len(weight_data) // 4):
                    weights = []
                    indices = []

                    for j in range(4):
                        w = weight_data[i*4+j] / 255.0
                        weights.append(w)
                        idx = index_data[i*4+j]
                        indices.append(idx)

                    # Normalize weights if they don't sum to 1
                    weight_sum = sum(weights)
                    if weight_sum > 0 and abs(weight_sum - 1.0) > 0.001:
                        weights = [w / weight_sum for w in weights]

                    self.weights.append(weights)
                    self.boneIndices.append(indices)

        # Extract faces
        indexBuffer = self.buildBatchedPS3IndexBuffer()
        if indexBuffer:
            index_data = indexBuffer[0]
            index_count = indexBuffer[1]

            for i in range(0, index_count, 3):
                if i + 2 < index_count:
                    idx1 = struct.unpack(
                        '>I', index_data[i * 4:(i + 1) * 4])[0]
                    idx2 = struct.unpack(
                        '>I', index_data[(i + 1) * 4:(i + 2) * 4])[0]
                    idx3 = struct.unpack(
                        '>I', index_data[(i + 2) * 4:(i + 3) * 4])[0]
                    self.faces.append((idx1, idx2, idx3))

    def buildBatchedPS3VertexBuffer(self, attributeId):
        batchedBuffer = []
        valid = False
        for segment in self.ps3Segments:
            unpackedBuffer = segment.getBufferForAttribute(attributeId)
            if unpackedBuffer is None:
                for i in range(segment.vertexCount):
                    batchedBuffer.extend(
                        bytes(struct.pack('>ffff', 0.0, 0.0, 0.0, 1.0)))
            else:
                valid = True
                batchedBuffer.extend(unpackedBuffer)
        if valid:
            return bytes(batchedBuffer)
        else:
            return None

    def buildBatchedPS3IndexBuffer(self):
        batchedBuffer = []
        currentIndex = 0
        indexCount = 0
        for segment in self.ps3Segments:
            for i in range(segment.indexCount):
                index = struct.unpack('>H', segment.indexBuffer[i*2:i*2+2])[0]
                batchedBuffer.extend(
                    bytes(struct.pack('>I', index + currentIndex)))
            currentIndex += segment.vertexCount
            indexCount += segment.indexCount
        return (bytes(batchedBuffer), indexCount)

    def buildBatchedPs3BoneBuffers(self):
        bwBuffer = []
        biBuffer = []

        for segment in self.ps3Segments:
            buffers = segment.getPs3BoneStuff()
            if buffers:
                bw, bi = buffers
                # Check if we have valid data
                if len(bw) > 0 and len(bi) > 0:
                    bwBuffer.extend(bw)
                    biBuffer.extend(bi)

        # If we have data, return it as bytes
        if len(bwBuffer) > 0 and len(biBuffer) > 0:
            return (bytes(bwBuffer), bytes(biBuffer))

        return None

    def superchargersFunkiness(self, endarg):
        fVBuf = []
        for i in range(self.vertexCount):
            coord = struct.unpack(
                f"{endarg}hhh", self.vertexBuffers[0][i * self.vertexStrides[0]+0:i * self.vertexStrides[0]+6])
            scale = struct.unpack(
                f"{endarg}h", self.vertexBuffers[0][i * self.vertexStrides[0]+6:i * self.vertexStrides[0]+8])[0]
            fVBuf.extend(bytes(struct.pack(f"{endarg}f", coord[0] / scale)))
            fVBuf.extend(bytes(struct.pack(f"{endarg}f", coord[1] / scale)))
            fVBuf.extend(bytes(struct.pack(f"{endarg}f", coord[2] / scale)))
        return bytes(fVBuf)

    def handlePackData(self, vertexBuff, stride):
        fVBuf = []
        for i in range(self.vertexCount):
            coord = struct.unpack(
                '>hhh', vertexBuff[i * stride+0:i * stride+6])
            fVBuf.extend(bytes(struct.pack(">f", coord[0] / 1024)))
            fVBuf.extend(bytes(struct.pack(">f", coord[1] / 1024)))
            fVBuf.extend(bytes(struct.pack(">f", coord[2] / 1024)))
        return bytes(fVBuf)

    def transform(self, mtx):
        self.transformation = mtx

    def createBlenderMesh(self, name="Mesh"):
        """Create a Blender mesh from the extracted data"""
        mesh = bpy.data.meshes.new(name)

        # Create mesh from vertices and faces
        mesh.from_pydata(self.vertices, [], self.faces)
        mesh.update()

        # Create UV coordinates if available
        if self.uvs:
            uv_layer = mesh.uv_layers.new(name="UVMap")
            for i, loop in enumerate(mesh.loops):
                try:
                    vidx = loop.vertex_index
                    uv_layer.data[i].uv = (
                        self.uvs[vidx][0], 1.0 - self.uvs[vidx][1])
                except IndexError:
                    # Handle cases where UV data might be missing
                    uv_layer.data[i].uv = (0.0, 0.0)

        # Create vertex colors if available
        if self.colors:
            color_layer = mesh.vertex_colors.new(name="Col")
            for i, loop in enumerate(mesh.loops):
                try:
                    vidx = loop.vertex_index
                    color_layer.data[i].color = (
                        self.colors[vidx][0],
                        self.colors[vidx][1],
                        self.colors[vidx][2],
                        self.colors[vidx][3]
                    )
                except IndexError:
                    # Handle cases where color data might be missing
                    color_layer.data[i].color = (1.0, 1.0, 1.0, 1.0)

        return mesh


class ModelObject:
    def __init__(self, id=0):
        self.meshes = []
        self.boneList = []
        self.boneMatrices = []
        self.boneIdList = []
        self.boneMapList = []
        self.anims = []
        self.id = id

    def build(self, igz, modelIndex):
        """Build Blender objects from the parsed data"""
        index = 0

        if len(self.meshes) == 0:
            print("No meshes found in model")
            return None

        # Create armature if we have bones
        armature = None
        if constants.dBuildBones and len(self.boneList) > 0:
            armature_name = f"Armature_{modelIndex}"
            armature = bpy.data.armatures.new(armature_name)
            armature_obj = bpy.data.objects.new(armature_name, armature)
            bpy.context.scene.collection.objects.link(armature_obj)

            # Enter edit mode to add bones
            bpy.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode='EDIT')

            # Create Blender bones from the bone list
            edit_bones = {}
            for bone in self.boneList:
                edit_bone = armature.edit_bones.new(bone.name)
                edit_bones[bone.index] = edit_bone

                # Set up parent relationships
                if bone.parentIndex != -1 and bone.parentIndex in edit_bones:
                    edit_bone.parent = edit_bones[bone.parentIndex]

                # Set bone positions (head, tail, roll)
                position = bone.getPosition()
                edit_bone.head = position

                # Basic tail calculation (point down Y-axis by default)
                if hasattr(bone, 'children') and bone.children:
                    # Point to first child
                    child_position = bone.children[0].getPosition()
                    edit_bone.tail = child_position
                else:
                    # No children, create a small tail offset
                    edit_bone.tail = (
                        position[0], position[1], position[2] + 5)

            # Exit edit mode
            bpy.ops.object.mode_set(mode='OBJECT')

        # Process each mesh
        for mesh_obj in self.meshes:
            print(f"Building mesh {index} of {len(self.meshes)}")
            mesh_name = f"Mesh_{modelIndex}_{index}"

            if len(mesh_obj.vertices) == 0:
                # Extract mesh data if not already processed
                if mesh_obj.isPs3:
                    mesh_obj.buildPs3MeshNew(self.boneMapList, igz.version)
                else:
                    mesh_obj.buildMesh(
                        self.boneMapList, igz.endianness, igz.version, igz.platform)

            if len(mesh_obj.vertices) > 0:
                # Create the Blender mesh
                mesh = mesh_obj.createBlenderMesh(mesh_name)
                blender_obj = bpy.data.objects.new(mesh_name, mesh)
                bpy.context.scene.collection.objects.link(blender_obj)

                # If we have an armature, parent and add vertex groups
                if armature and constants.dBuildBones:
                    # Parent mesh to armature
                    blender_obj.parent = armature_obj

                    # Add armature modifier
                    modifier = blender_obj.modifiers.new(
                        name="Armature", type='ARMATURE')
                    modifier.object = armature_obj

                    # Create vertex groups for skinning
                    if mesh_obj.weights and mesh_obj.boneIndices:
                        # Create groups for each bone
                        bone_map = self.boneMapList[mesh_obj.boneMapIndex] if len(
                            self.boneMapList) > mesh_obj.boneMapIndex else []

                        # Pre-create all needed vertex groups
                        for bone_idx in range(len(bone_map)):
                            mapped_bone = bone_map[bone_idx]
                            if mapped_bone < len(self.boneList):
                                bone_name = self.boneList[mapped_bone].name
                            else:
                                bone_name = f"Bone_{mapped_bone}"

                            if bone_name not in blender_obj.vertex_groups:
                                blender_obj.vertex_groups.new(name=bone_name)

                        # Assign weights to vertex groups
                        for vertex_idx in range(len(mesh_obj.weights)):
                            # Normalize weights if needed
                            weights = mesh_obj.weights[vertex_idx]
                            weight_sum = sum(weights)
                            if weight_sum > 0.001 and abs(weight_sum - 1.0) > 0.01:
                                weights = [w / weight_sum for w in weights]

                            # Assign weights that are significant
                            for i in range(4):
                                weight = weights[i]
                                if weight > 0.001:  # Skip near-zero weights
                                    bone_idx = mesh_obj.boneIndices[vertex_idx][i]
                                    if bone_idx < len(bone_map):
                                        mapped_bone = bone_map[bone_idx]
                                        bone_name = self.boneList[mapped_bone].name if mapped_bone < len(
                                            self.boneList) else f"Bone_{mapped_bone}"

                                        if bone_name in blender_obj.vertex_groups:
                                            blender_obj.vertex_groups[bone_name].add(
                                                [vertex_idx], weight, 'ADD')

            index += 1

        return True

    def __init__(self):
        self.name = ""
        self.boneList = []
        self.boneMatrices = []
        self.boneIdList = []
        self.boneMapList = []
        self.anims = []
        self.id = 0
        self.meshes = []
        self.meshCount = 0
        self.name = ""
        self.vertexBuffers = []
        self.vertexStrides = []
        self.vertexCount = 0
        self.indexBuffer = None
        self.isPs3 = False
        self.ps3Segments = []
        self.spuConfigInfo = False  # PS3 Exclusive
        self.skipBuild = False
        self.vertexElements = []
        self.vertexStreams = []
        self.primType = constants.PrimitiveType.TRIANGLE
        self.indexCount = 0
        self.boneMapIndex = 0
        self.transformation = None
        self.packData = None
        self.platform = 0
        self.platformData = None

        # For Blender mesh construction
        self.vertices = []
        self.faces = []
        self.normals = []
        self.uvs = []
        self.colors = []
        self.weights = []
        self.boneIndices = []

    def buildMesh(self, boneMapList, endianness, version, platform):
        if self.vertexCount == 0:
            return

        endarg = '>' if endianness == "BE" else '<'

        print(f"vertex count    {hex(self.vertexCount)}")
        print(f"vertex stride   {hex(self.vertexStrides[0])}")
        print(f"index count     {hex(self.indexCount)}")
        print(f"name:           {self.name}")
        print(f"bone map index: {hex(self.boneMapIndex)}")

        # Process vertex data
        if platform == 2 and struct.unpack(">H", self.vertexBuffers[0][0:2])[0] == 0x9F:
            self.vertexBuffers[0] = bytes(self.vertexBuffers[0][4:])

        if version >= 6:
            packData = self.packData[2] if self.packData is not None else None
        else:
            packDataOffset = 0
            for elem in self.vertexElements:
                if elem._type == 0x2C:
                    continue
                print(f"processing... {elem._packDataOffset}")
                if (elem._packTypeAndFracHint & 7) == 2:
                    if packDataOffset < elem._packDataOffset:
                        packDataOffset = elem._packDataOffset
            packData = bytes(self.vertexBuffers[0][len(
                self.vertexBuffers[0]) - packDataOffset - 4:])

        for elem in self.vertexElements:
            if elem._type == 0x2C:
                continue

            streamOffset = 0
            for i in range(elem._stream):
                streamOffset += (((self.vertexStreams[i] *
                                 self.vertexCount) + 0x1F) // 0x20) * 0x20
            print(
                f"Getting bytes for stream from {hex(streamOffset)} to {hex(streamOffset + self.vertexCount * self.vertexStreams[elem._stream])}")
            stream = bytes(self.vertexBuffers[0][streamOffset:streamOffset +
                           self.vertexCount * self.vertexStreams[elem._stream]])
            streamSize = self.vertexStreams[elem._stream]

            print(f"usage: {hex(elem._usage)}; offset: {hex(elem._offset)}; stream: {hex(elem._stream)}; count: {hex(elem._count)}; type: {hex(elem._type)}; mapToElement: {hex(elem._mapToElement)}; usageIndex: {hex(elem._usageIndex)}; packDataOffset: {hex(elem._packDataOffset)}; packTypeAndFracHint: {hex(elem._packTypeAndFracHint)}; freq: {hex(elem._freq)}; streamOffset: {hex(streamOffset)}")

            if elem._usage == 0:  # IG_VERTEX_USAGE_POSITION
                stride = 0x10
                if elem._type == 0x23:
                    fakeVertexBuffer = self.superchargersFunkiness(endarg)
                    stride = 0x0C
                else:
                    fakeVertexBuffer = elem.unpack(
                        stream, streamSize, packData, endarg)

                # Extract positions
                for i in range(self.vertexCount):
                    x, y, z = struct.unpack(
                        endarg + 'fff', fakeVertexBuffer[i * stride:i * stride + 12])
                    self.vertices.append((x, y, z))

            if elem._usage == 1:  # IG_VERTEX_USAGE_NORMAL
                vnormals = elem.unpack(stream, streamSize, packData, endarg)

                # Extract normals
                for i in range(self.vertexCount):
                    nx, ny, nz = struct.unpack(
                        endarg + 'fff', vnormals[i * 0x10:i * 0x10 + 12])
                    self.normals.append((nx, ny, nz))

            if elem._usage == 4:  # IG_VERTEX_USAGE_COLOR
                vcolors = elem.unpack(stream, streamSize, packData, endarg)

                # Extract colors
                for i in range(self.vertexCount):
                    r, g, b, a = struct.unpack(
                        endarg + 'ffff', vcolors[i * 0x10:i * 0x10 + 16])
                    self.colors.append((r, g, b, a))

            if elem._usage == 5 and elem._usageIndex == 0:  # IG_VERTEX_USAGE_TEXCOORD
                vtexcoords = elem.unpack(stream, streamSize, packData, endarg)

                # Extract UVs
                for i in range(self.vertexCount):
                    u = struct.unpack(
                        endarg + 'f', vtexcoords[i * 0x10:i * 0x10 + 4])[0]
                    v = struct.unpack(
                        endarg + 'f', vtexcoords[i * 0x10 + 4:i * 0x10 + 8])[0]
                    self.uvs.append((u, v))

            if elem._usage == 6 and elem._usageIndex == 0 and constants.dBuildBones:  # IG_VERTEX_USAGE_BLENDWEIGHTS
                vblendweights = elem.unpack(
                    stream, streamSize, packData, endarg)

                # Extract weights
                for i in range(self.vertexCount):
                    weights = []
                    for j in range(elem._count):
                        weight = struct.unpack(
                            endarg + 'f', vblendweights[i * 0x10 + j * 4:i * 0x10 + (j + 1) * 4])[0]
                        weights.append(weight)
                    # Pad with zeros if needed
                    while len(weights) < 4:
                        weights.append(0.0)
                    self.weights.append(weights)

            if elem._usage == 8 and elem._usageIndex == 0 and constants.dBuildBones:  # IG_VERTEX_USAGE_BLENDINDICES
                vfblendindices = elem.unpack(
                    stream, streamSize, packData, endarg)

                # Extract bone indices
                for i in range(self.vertexCount):
                    indices = []
                    for j in range(elem._count):
                        index = int(struct.unpack(
                            endarg + 'f', vfblendindices[i * 0x10 + j * 4:i * 0x10 + (j + 1) * 4])[0])
                        indices.append(index)
                    # Pad with zeros if needed
                    while len(indices) < 4:
                        indices.append(0)
                    self.boneIndices.append(indices)

        # Process index data
        if constants.dBuildFaces and self.primType != constants.PrimitiveType.TRIANGLE_STRIP:
            if self.vertexCount <= 0xFFFF:
                # Extract triangles from 16-bit indices
                for i in range(0, self.indexCount, 3):
                    if i + 2 < self.indexCount:
                        idx1 = struct.unpack(
                            endarg + 'H', self.indexBuffer[i * 2:(i + 1) * 2])[0]
                        idx2 = struct.unpack(
                            endarg + 'H', self.indexBuffer[(i + 1) * 2:(i + 2) * 2])[0]
                        idx3 = struct.unpack(
                            endarg + 'H', self.indexBuffer[(i + 2) * 2:(i + 3) * 2])[0]
                        self.faces.append((idx1, idx2, idx3))
            else:
                # Extract triangles from 32-bit indices
                for i in range(0, self.indexCount, 3):
                    if i + 2 < self.indexCount:
                        idx1 = struct.unpack(
                            endarg + 'I', self.indexBuffer[i * 4:(i + 1) * 4])[0]
                        idx2 = struct.unpack(
                            endarg + 'I', self.indexBuffer[(i + 1) * 4:(i + 2) * 4])[0]
                        idx3 = struct.unpack(
                            endarg + 'I', self.indexBuffer[(i + 2) * 4:(i + 3) * 4])[0]
                        self.faces.append((idx1, idx2, idx3))
        elif constants.dBuildFaces and self.primType == constants.PrimitiveType.TRIANGLE_STRIP:
            # Handle triangle strips - simplified for now
            # A proper implementation would convert strips to triangles
            pass
