from enum import Enum
from typing import List

dBuildMeshes = True      # Whether to build the meshes or just parse the file
dBuildBones = True       # Whether to build the bones
dBuildFaces = True       # Whether to build the index buffer
dAllowWii = True         # Whether to allow Wii models
# Offset of the first object to process, -1 means just loop through every object
dFirstObjectOffset = -1
# The highest number of models to extract before the user is prompted
dModelThreshold = 50


class Endianness(str, Enum):
    BIG = ">"
    LITTLE = "<"


class SeekMode(int, Enum):
    ABS = 0  # os.SEEK_SET
    REL = 1  # os.SEEK_CUR

# Edge Geometry Skin types as Enum for clarity


class EdgeGeomSkinType(int, Enum):
    NONE = 0
    NO_SCALING = 1
    UNIFORM_SCALING = 2
    NON_UNIFORM_SCALING = 3
    SINGLE_BONE_NO_SCALING = 4
    SINGLE_BONE_UNIFORM_SCALING = 5
    SINGLE_BONE_NON_UNIFORM_SCALING = 6

# Primitive types as Enum


class PrimitiveType(int, Enum):
    POINTS = 0
    TRIANGLE = 3
    TRIANGLE_STRIP = 4
    TRIANGLE_FAN = 5
    TRIANGLE_QUADS = 6


# Vertex maximum magnitude values
vertexMaxMags: List[int] = [
    1,          # FLOAT1
    1,          # FLOAT2
    1,          # FLOAT3
    1,          # FLOAT4
    1,          # UBYTE4N_COLOR
    1,          # UBYTE4N_COLOR_ARGB
    1,          # UBYTE4N_COLOR_RGBA
    1,          # UNDEFINED_0
    1,          # UBYTE2N_COLOR_5650
    1,          # UBYTE2N_COLOR_5551
    1,          # UBYTE2N_COLOR_4444
    0x7FFFFFFF,  # INT1
    0x7FFFFFFF,  # INT2
    0x7FFFFFFF,  # INT4
    0xFFFFFFFF,  # UINT1
    0xFFFFFFFF,  # UINT2
    0xFFFFFFFF,  # UINT4
    1,          # INT1N
    1,          # INT2N
    1,          # INT4N
    1,          # UINT1N
    1,          # UINT2N
    1,          # UINT4N
    0xFF,       # UBYTE4
    0xFF,       # UBYTE4_X4
    0x7F,       # BYTE4
    1,          # UBYTE4N
    1,          # UNDEFINED_1
    1,          # BYTE4N
    0x3FFF,     # SHORT2
    0x3FFF,     # SHORT4
    0xFFFF,     # USHORT2
    0xFFFF,     # USHORT4
    1,          # SHORT2N
    1,          # SHORT3N
    1,          # SHORT4N
    1,          # USHORT2N
    1,          # USHORT3N
    1,          # USHORT4N
    1,          # UDEC3
    1,          # DEC3N
    1,          # DEC3N_S11_11_10
    1,          # HALF2
    1,          # HALF4
    1,          # UNUSED
    1,          # BYTE3N
    0x7FFF,     # SHORT3
    0xFFFF,     # USHORT3
    0xFF,       # UBYTE4_ENDIAN
    0xFF,       # UBYTE4_COLOR
    0x7F,       # BYTE3
    1,          # UBYTE2N_COLOR_5650_RGB
    1,          # UDEC3_OES
    1,          # DEC3N_OES
    1,          # SHORT4N_EDGE
]
