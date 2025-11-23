"""
Utility classes and functions for the Skylanders importer
"""

import struct
from . import constants
import bpy
from typing import Any


class NoeBitStream:
    """
    Reimplementation of Noesis's stream functionality for reading binary data
    """

    data: bytes
    endian: str
    offset: int

    def __init__(self, data: bytes, endian: str = constants.Endianness.LITTLE) -> None:
        self.data = data
        self.endian = endian
        self.offset = 0

    def _require(self, size: int) -> None:
        remaining = len(self.data) - self.offset
        if self.offset < 0 or remaining < size:
            raise ValueError(
                f"Unexpected end of IGZ stream at {self.offset} (wanted {size} bytes, have {max(remaining, 0)})"
            )

    def seek(self, offset: int, whence: int = constants.SeekMode.ABS) -> int:
        if whence == constants.SeekMode.ABS:
            self.offset = offset
        elif whence == constants.SeekMode.REL:
            self.offset += offset
        return self.offset

    def tell(self) -> int:
        return self.offset

    def readBytes(self, size: int) -> bytes:
        self._require(size)
        bytes_data = self.data[self.offset : self.offset + size]
        self.offset += size
        return bytes_data

    def readUInt(self) -> int:
        endian = self.endian.value if hasattr(self.endian, "value") else self.endian
        self._require(4)
        val = struct.unpack(f"{endian}I", self.data[self.offset : self.offset + 4])[0]
        self.offset += 4
        return val

    def readUInt64(self) -> int:
        endian = self.endian.value if hasattr(self.endian, "value") else self.endian
        self._require(8)
        val = struct.unpack(f"{endian}Q", self.data[self.offset : self.offset + 8])[0]
        self.offset += 8
        return val

    def readInt(self) -> int:
        endian = self.endian.value if hasattr(self.endian, "value") else self.endian
        self._require(4)
        val = struct.unpack(f"{endian}i", self.data[self.offset : self.offset + 4])[0]
        self.offset += 4
        return val

    def readUShort(self) -> int:
        endian = self.endian.value if hasattr(self.endian, "value") else self.endian
        self._require(2)
        val = struct.unpack(endian + "H", self.data[self.offset : self.offset + 2])[0]
        self.offset += 2
        return val

    def readUByte(self) -> int:
        self._require(1)
        val = self.data[self.offset]
        self.offset += 1
        return val

    def readString(self) -> str:
        start = self.offset
        while self.offset < len(self.data) and self.data[self.offset] != 0:
            self.offset += 1
        result = self.data[start : self.offset].decode("utf-8")
        # Move past the null terminator
        if self.offset < len(self.data):
            self.offset += 1
        return result

    def readFloat(self) -> float:
        endian = self.endian.value if hasattr(self.endian, "value") else self.endian
        self._require(4)
        val = struct.unpack(endian + "f", self.data[self.offset : self.offset + 4])[0]
        self.offset += 4
        return val

    def readDouble(self) -> float:
        endian = self.endian.value if hasattr(self.endian, "value") else self.endian
        self._require(8)
        val = struct.unpack(endian + "d", self.data[self.offset : self.offset + 8])[0]
        self.offset += 8
        return val

    def readShort(self) -> int:
        endian = self.endian.value if hasattr(self.endian, "value") else self.endian
        self._require(2)
        val = struct.unpack(endian + "h", self.data[self.offset : self.offset + 2])[0]
        self.offset += 2
        return val

    def readHalfFloat(self) -> float:
        """Convert half precision (16-bit) float to regular float"""
        self._require(2)
        half = struct.unpack(
            self.endian + "H", self.data[self.offset : self.offset + 2]
        )[0]
        self.offset += 2

        # Extract sign, exponent, and mantissa
        sign = (half >> 15) & 0x1
        exponent = (half >> 10) & 0x1F
        mantissa = half & 0x3FF

        # Handle special cases
        if exponent == 0:
            if mantissa == 0:
                return -0.0 if sign else 0.0
            else:
                # Denormalized number
                return ((-1) ** sign) * (mantissa / 1024.0) * (2**-14)
        elif exponent == 31:
            if mantissa == 0:
                return float("-inf") if sign else float("inf")
            else:
                return float("nan")

        # Normalized number
        return ((-1) ** sign) * (1 + mantissa / 1024.0) * (2 ** (exponent - 15))


def decompressEdgeIndices(indexBuffer: bytes, indexCount: int) -> bytes:
    """
    Simplified placeholder for the edge index decompression
    In a full implementation, this would properly handle the triangle strips format
    """
    # This is a very basic implementation, just returning transformed indices
    # A proper implementation would need to follow the original algorithm
    result = bytearray(indexCount * 2)

    for i in range(indexCount):
        # Placeholder - in real implementation we would properly decompress
        idx = i % 256
        result[i * 2] = idx
        result[i * 2 + 1] = 0

    return bytes(result)


class Bone:
    """Helper class for bone data"""

    index: int
    name: str
    parentIndex: int
    position: Any
    matrix: Any
    children: list
    size_multiplier: float
    blender_bone: Any

    def __init__(
        self,
        index: int,
        name: str,
        parentIndex: int,
        translation: Any,
        size_multiplier: float = 1.5,
    ) -> None:
        self.index = index
        self.name = name if name else f"bone_{index}"
        self.parentIndex = parentIndex
        self.position = translation
        self.matrix = None
        self.children = []
        self.size_multiplier = size_multiplier
        self.blender_bone = None  # Store reference to created Blender bone

    def setMatrix(self, matrix_data: bytes, endian: str) -> None:
        """Parse matrix data from the file to create a bone matrix"""
        from mathutils import Matrix

        # Create a 4x4 matrix from the raw data - ensure correct ordering
        mtx = []
        for i in range(0, 64, 4):
            value = struct.unpack(endian + "f", matrix_data[i : i + 4])[0]
            mtx.append(value)

        # Create matrix with correct column/row order for Blender
        self.matrix = Matrix(
            (
                (mtx[0], mtx[4], mtx[8], mtx[12]),
                (mtx[1], mtx[5], mtx[9], mtx[13]),
                (mtx[2], mtx[6], mtx[10], mtx[14]),
                (mtx[3], mtx[7], mtx[11], mtx[15]),
            )
        )

        # Matrix is stored inverted in the file, invert it
        self.matrix.invert()

        # Extract position from matrix
        self.position = self.matrix.to_translation()

    def getPosition(self) -> Any:
        """Get the bone position, either from translation or matrix"""
        if self.matrix:
            # Extract position from matrix
            return (self.matrix[0][3], self.matrix[1][3], self.matrix[2][3])
        else:
            # Use the translation directly
            return self.position

    def create_in_blender(self, armature: Any, bone_map: dict = None) -> Any:
        """Create this bone in a Blender armature"""
        # Switch to edit mode to add bones
        bpy.ops.object.mode_set(mode="EDIT")

        # Create a new bone
        edit_bone = armature.edit_bones.new(self.name)

        # Set bone position using head and tail
        position = self.getPosition()
        edit_bone.head = position

        # Set a default tail position (offset in Z direction)
        # This can be adjusted based on your skeletal structure
        tail_offset = self.size_multiplier
        edit_bone.tail = (position[0], position[1], position[2] + tail_offset)

        # Store reference to parent if available
        if self.parentIndex >= 0:
            parent_name = f"bone_{self.parentIndex}"
            if bone_map and self.parentIndex in bone_map:
                parent_name = bone_map[self.parentIndex].name

            if parent_name in armature.edit_bones:
                edit_bone.parent = armature.edit_bones[parent_name]

                # Optionally connect bones if they're close enough
                # edit_bone.use_connect = True

        # Keep reference to the created bone
        self.blender_bone = edit_bone
        return edit_bone

    def apply_transform(self, armature_obj: Any) -> None:
        """Apply the bone's transformation matrix in pose mode"""
        if not self.matrix:
            return

        # Switch to pose mode
        bpy.ops.object.mode_set(mode="POSE")

        # Get the pose bone
        if self.name in armature_obj.pose.bones:
            pose_bone = armature_obj.pose.bones[self.name]

            # Apply the matrix as a pose transformation
            # Might need conversion from global to local space
            from mathutils import Matrix

            if self.matrix:
                # Calculate local transformation
                if pose_bone.parent:
                    local_matrix = pose_bone.parent.matrix.inverted() @ self.matrix
                else:
                    local_matrix = self.matrix

                pose_bone.matrix = local_matrix


def create_armature_from_bones(bone_list: list, name: str = "Armature") -> Any:
    """Create a Blender armature from a list of Bone objects"""
    import bpy

    # Create a new armature data object
    armature = bpy.data.armatures.new(name)

    # Create a new object with the armature data
    armature_obj = bpy.data.objects.new(name, armature)

    # Add the armature to the scene
    bpy.context.collection.objects.link(armature_obj)

    # Select the armature object
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)

    # Enter edit mode
    bpy.ops.object.mode_set(mode="EDIT")

    # Create map for parent lookup
    bone_map = {bone.index: bone for bone in bone_list}

    # First create all bones
    for bone in bone_list:
        edit_bone = armature.edit_bones.new(bone.name)

        # Set position from translation or matrix
        if bone.position:
            edit_bone.head = bone.position
            # Set a default length for the bone
            edit_bone.tail = (
                bone.position[0],
                bone.position[1],
                bone.position[2] + bone.size_multiplier,
            )

        # Set parent if available
        if bone.parentIndex >= 0 and bone.parentIndex in bone_map:
            parent_name = bone_map[bone.parentIndex].name
            if parent_name in armature.edit_bones:
                edit_bone.parent = armature.edit_bones[parent_name]

    # Return to object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    return armature_obj
