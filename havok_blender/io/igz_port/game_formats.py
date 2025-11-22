from typing import Any
from . import igz_file
from . import formats
from . import utils
from . import constants

# ------------------------------------------------------------------------------
# Trap Team implementation
# ------------------------------------------------------------------------------


class sttIgzFile(igz_file.igzFile):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        # On trap team, IG_CORE_PLATFORM_MARMALADE was turned into IG_CORE_PLATFORM_DEPRECATED
        self.is64Bit = ssfIgzFile.is64BitCall
        self.arkRegisteredTypes = sttarkRegisteredTypes

    def process_tfbSpriteInfo(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0xD8)
        _contextDataInfo = self.process_igObject(bs, self.readPointer(bs))

    def process_tfbPhysicsModel(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0x14)
        _tfbBody = self.process_igObject(bs, self.readPointer(bs))

    def process_tfbPhysicsBody(self, bs: Any, offset: int) -> None:
        isModelNew = self.addModel(offset)
        if isModelNew:
            self.bitAwareSeek(bs, offset, 0x00, 0x28)
            _combinerPrototype = self.process_igObject(
                bs, self.readPointer(bs))
            if self.platform == 0x0B or self.platform == 0x04:
                bs.seek(offset + 0x20, constants.SeekMode.ABS)
            else:
                self.bitAwareSeek(bs, offset, 0x00, 0x30)
            _entityInfo = self.process_igObject(bs, self.readPointer(bs))

    def process_tfbBodyEntityInfo(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0x24)
        _blendMatrixIndexLists = self.process_igObject(
            bs, self.readPointer(bs))
        if _blendMatrixIndexLists is not None:
            print(f"boneMpaList length is {hex(len(_blendMatrixIndexLists))}")
            self.models[-1].boneMapList.extend(_blendMatrixIndexLists)
        sttIgzFile.process_tfbEntityInfo(self, bs, offset)

    def process_tfbEntityInfo(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0x14)
        _drawables = self.process_igObject(bs, self.readPointer(bs))

    def process_Drawable(self, bs: Any, offset: int) -> None:
        self.models[-1].meshes.append(formats.MeshObject())
        self.bitAwareSeek(bs, offset, 0x00, 0x16)
        _blendMatrixSet = bs.readUShort()
        self.models[-1].meshes[-1].boneMapIndex = _blendMatrixSet

        self.bitAwareSeek(bs, offset, 0x00, 0x0C)
        _geometry = self.process_igObject(bs, self.readPointer(bs))

    def process_tfbPhysicsWorld(self, bs: Any, offset: int) -> None:
        self.addModel(offset)
        self.bitAwareSeek(bs, offset, 0x00, 0x28)
        _entityInfo = self.process_igObject(bs, self.readPointer(bs))

    def process_tfbPhysicsCombinerLink(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0x0C)
        _skeleton = self.process_igObject(bs, self.readPointer(bs))

    def process_tfbActorInfo(self, bs: Any, offset: int) -> None:
        if self.platform != 0x04:
            self.bitAwareSeek(bs, offset, 0x00, 0xEC)
        else:
            self.bitAwareSeek(bs, offset, 0x00, 0xEC)
        _model = self.process_igObject(bs, self.readPointer(bs))

    def process_tfbMobileLodGeometry(self, bs: Any, offset: int) -> None:
        ssfIgzFile.process_igGeometry(self, bs, offset)
        self.bitAwareSeek(bs, offset, 0x00, 0x2C)
        _lodAttrs = self.process_igObject(bs, self.readPointer(bs))


# ------------------------------------------------------------------------------
# Giants implementation
# ------------------------------------------------------------------------------
class sgIgzFile(igz_file.igzFile):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.is64Bit = sgIgzFile.is64BitCall
        self.arkRegisteredTypes = sgarkRegisteredTypes

    def is64BitCall(self) -> bool:
        platformbittness = [
            False,  # IG_CORE_PLATFORM_DEFAULT
            False,  # IG_CORE_PLATFORM_WIN32
            False,  # IG_CORE_PLATFORM_WII
            True,   # IG_CORE_PLATFORM_DEPRECATED
            False,  # IG_CORE_PLATFORM_ASPEN
            False,  # IG_CORE_PLATFORM_XENON
            False,  # IG_CORE_PLATFORM_PS3
            False,  # IG_CORE_PLATFORM_OSX
            True,   # IG_CORE_PLATFORM_WIN64
            False,  # IG_CORE_PLATFORM_CAFE
            False,  # IG_CORE_PLATFORM_NGP
            False,  # IG_CORE_PLATFORM_ANDROID
            False,  # IG_CORE_PLATFORM_MARMALADE
            False,  # IG_CORE_PLATFORM_MAX
        ]
        return platformbittness[self.platform]

    def process_tfbSpriteInfo(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0xD0)
        _contextDataInfo = self.process_igObject(bs, self.readPointer(bs))

    def process_tfbPhysicsBody(self, bs: Any, offset: int) -> None:
        shouldAddModel = self.addModel(offset)
        if shouldAddModel:
            self.bitAwareSeek(bs, offset, 0x00, 0x24)
            _combinerPrototype = self.process_igObject(
                bs, self.readPointer(bs))
            self.bitAwareSeek(bs, offset, 0x00, 0x20)
            _node = self.process_igObject(bs, self.readPointer(bs))

    def process_tfbRuntimeTechniqueInstance(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0x28)
        _geomAttr = self.process_igObject(bs, self.readPointer(bs))

    def process_igEdgeGeometryAttr(self, bs: Any, offset: int) -> None:
        self.models[-1].meshes.append(formats.MeshObject())
        ssfIgzFile.process_igEdgeGeometryAttr(self, bs, offset)

    def process_tfbActorInfo(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0xDC)
        _model = self.process_igObject(bs, self.readPointer(bs))

    def process_tfbPhysicsWorld(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0x20)
        _sceneInfo = self.process_igObject(bs, self.readPointer(bs))


# ------------------------------------------------------------------------------
# Spyro's Adventure implementation
# ------------------------------------------------------------------------------
class ssaIgzFile(igz_file.igzFile):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.is64Bit = ssaIgzFile.is64BitCall
        self.arkRegisteredTypes = ssaarkRegisteredTypes

    def is64BitCall(self) -> bool:
        platformbittness = [
            False,  # IG_CORE_PLATFORM_DEFAULT
            False,  # IG_CORE_PLATFORM_WIN32
            False,  # IG_CORE_PLATFORM_WII
            True,   # IG_CORE_PLATFORM_DEPRECATED
            False,  # IG_CORE_PLATFORM_ASPEN
            False,  # IG_CORE_PLATFORM_XENON
            False,  # IG_CORE_PLATFORM_PS3
            False,  # IG_CORE_PLATFORM_OSX
            True,   # IG_CORE_PLATFORM_WIN64
            False,  # IG_CORE_PLATFORM_CAFE
            False,  # IG_CORE_PLATFORM_NGP
            False,  # IG_CORE_PLATFORM_ANDROID
            False,  # IG_CORE_PLATFORM_MARMALADE
            False,  # IG_CORE_PLATFORM_MAX
        ]
        return platformbittness[self.platform]

    def process_igBlendMatrixSelect(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0xB0)
        self.models[-1].boneMapList.append(
            self.process_igObject(bs, self.readPointer(bs)))
        ssfIgzFile.process_igAttrSet(self, bs, offset)


# ------------------------------------------------------------------------------
# SuperChargers implementation
# ------------------------------------------------------------------------------
class sscIgzFile(igz_file.igzFile):
    """SuperChargers implementation"""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.is64Bit = sscIgzFile.is64BitCall
        self.arkRegisteredTypes = sscarkRegisteredTypes

    def is64BitCall(self) -> bool:
        platformbittness = [
            False,  # IG_CORE_PLATFORM_DEFAULT
            False,  # IG_CORE_PLATFORM_WIN32
            False,  # IG_CORE_PLATFORM_WII
            True,   # IG_CORE_PLATFORM_DURANGO
            False,  # IG_CORE_PLATFORM_ASPEN
            False,  # IG_CORE_PLATFORM_XENON
            False,  # IG_CORE_PLATFORM_PS3
            False,  # IG_CORE_PLATFORM_OSX
            True,   # IG_CORE_PLATFORM_WIN64
            False,  # IG_CORE_PLATFORM_CAFE
            False,  # IG_CORE_PLATFORM_RASPI
            False,  # IG_CORE_PLATFORM_ANDROID
            True,   # IG_CORE_PLATFORM_ASPEN64
            False,  # IG_CORE_PLATFORM_LGTV
            True,   # IG_CORE_PLATFORM_PS4
            False,  # IG_CORE_PLATFORM_WP8
            False,  # IG_CORE_PLATFORM_LINUX
            False,  # IG_CORE_PLATFORM_MAX
        ]
        return platformbittness[self.platform]

    def process_CGraphicsSkinInfo(self, bs: Any, offset: int) -> None:
        self.models.append(formats.ModelObject())
        self.bitAwareSeek(bs, offset, 0x28, 0x14)
        _skeleton = self.process_igObject(bs, self.readPointer(bs))
        self.bitAwareSeek(bs, offset, 0x30, 0x18)
        _skin = self.process_igObject(bs, self.readPointer(bs))

    def process_igSkeleton2(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x20, 0x10)
        _inverseJointArray = self.readMemoryRef(bs)
        print(f"_inverseJointArray offset: {hex(_inverseJointArray[1])}")
        print(f"_inverseJointArray size: {hex(_inverseJointArray[0])}")
        self.models[-1].boneMatrices = _inverseJointArray[2]
        self.bitAwareSeek(bs, offset, 0x18, 0x0C)
        _boneList = self.process_igObject(bs, self.readPointer(bs))

    def process_igSkeletonBoneList(self, bs: Any, offset: int) -> None:
        bones = self.process_igObjectList(bs, offset)
        endarg = constants.Endianness.BIG if self.endianness == "BE" else constants.Endianness.LITTLE
        index = 0

        # Check if there are any corrupted bones
        for bone in bones:
            if bone[2] == -1:
                bones.remove(bone)

        mtxStream = utils.NoeBitStream(self.models[-1].boneMatrices, endarg)

        for bone in bones:
            print(f"bone_{index}_{bone[2]}_{bone[1]}::{bone[0]}::{bone[3]}")

            # Create a Blender-compatible bone
            bone_obj = utils.Bone(bone[2], bone[0], bone[1]-1, bone[3])

            if bone[2] != -1:
                mtxStream.seek(bone[2] * 0x40, constants.SeekMode.ABS)
                bone_matrix_data = mtxStream.readBytes(0x40)
                bone_obj.setMatrix(bone_matrix_data, endarg)

            self.models[-1].boneList.append(bone_obj)
            index += 1

    def process_igSkeletonBone(self, bs: Any, offset: int) -> tuple:
        _name = self.process_igNamedObject(bs, offset)
        self.bitAwareSeek(bs, offset, 0x18, 0x0C)
        _parentIndex = bs.readInt()
        self.bitAwareSeek(bs, offset, 0x1C, 0x10)
        _blendMatrixIndex = bs.readInt()
        self.bitAwareSeek(bs, offset, 0x20, 0x14)
        _translation = self.readVector3(bs)
        return (_name, _parentIndex, _blendMatrixIndex, _translation)

    def process_igModelInfo(self, bs: Any, offset: int) -> None:
        self.models.append(formats.ModelObject())
        self.bitAwareSeek(bs, offset, 0x28, 0x14)
        _modelData = self.process_igObject(bs, self.readPointer(bs))

    def process_igModelData(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x40, 0x30)
        _transforms = self.readObjectVector(bs)
        self.bitAwareSeek(bs, offset, 0x58, 0x3C)
        _transformHeirarchy = self.readIntVector(bs)
        self.bitAwareSeek(bs, offset, 0x70, 0x48)
        _drawCalls = self.readObjectVector(bs)
        self.bitAwareSeek(bs, offset, 0x88, 0x54)
        _drawCallTransformIndices = self.readIntVector(bs)
        self.bitAwareSeek(bs, offset, 0xB8, 0x6C)
        _blendMatrixIndices = self.readIntVector(bs)
        self.models[-1].boneIdList = _blendMatrixIndices
        print(
            f"igModelData._drawCalls.count(): {hex(len(_drawCalls))}; transforms: {hex(len(_transforms))}")
        for i in range(len(_drawCalls)):
            mesh = formats.MeshObject()
            mesh.boneMapIndex = len(self.models[-1].boneMapList)
            self.models[-1].meshes.append(mesh)
            self.process_igObject(bs, _drawCalls[i])

    def process_igModelDrawCallData(self, bs: Any, offset: int) -> None:
        _name = self.process_igNamedObject(bs, offset)
        self.bitAwareSeek(bs, offset, 0x48, 0x34)
        _graphicsVertexBuffer = self.process_igObject(bs, self.readPointer(bs))
        self.bitAwareSeek(bs, offset, 0x50, 0x38)
        _graphicsIndexBuffer = self.process_igObject(bs, self.readPointer(bs))
        self.bitAwareSeek(bs, offset, 0x58, 0x3C)
        _platformData = self.process_igObject(bs, self.readPointer(bs))
        self.bitAwareSeek(bs, offset, 0x60, 0x40)
        _blendVectorOffset = bs.readUShort()
        self.bitAwareSeek(bs, offset, 0x62, 0x42)
        _blendVectorCount = bs.readUShort()

        print(f"_blendVectorOffset: {hex(_blendVectorOffset)}")
        print(f"_blendVectorCount: {hex(_blendVectorCount)}")

        self.models[-1].boneMapList.append(
            self.models[-1].boneIdList[_blendVectorOffset:_blendVectorOffset + _blendVectorCount])
        self.models[-1].meshes[-1].name = _name

    def process_igGraphicsVertexBuffer(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x10, 0x0C)
        _vertexBuffer = self.process_igObject(bs, self.readPointer(bs))

    def process_igGraphicsIndexBuffer(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x10, 0x0C)
        _indexBuffer = self.process_igObject(bs, self.readPointer(bs))

    def process_igVertexBuffer(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x0C, 0x08)
        self.models[-1].meshes[-1].vertexCount = bs.readUInt()
        self.bitAwareSeek(bs, offset, 0x20, 0x14)
        _data = self.readMemoryRefHandle(bs)
        self.bitAwareSeek(bs, offset, 0x28, 0x18)
        _format = self.process_igObject(bs, self.readPointer(bs))
        self.models[-1].meshes[-1].vertexBuffers.append(_data[2])
        self.models[-1].meshes[-1].vertexStrides.append(_format)

        if self.version >= 0x06:
            self.bitAwareSeek(bs, offset, 0x30, 0x20)
            _packData = self.readMemoryRef(bs)

            if _packData[0] > 0:
                self.models[-1].meshes[-1].packData = _packData
                print(f"packData offset: {hex(_packData[1])}")
                print(f"packData size: {hex(_packData[0])}")

        print(f"vertexCount:  {hex(self.models[-1].meshes[-1].vertexCount)}")
        print(f"vertex offset: {hex(_data[1])}")
        print(f"vertex buf size: {hex(_data[0])}")

    def process_igVertexFormat(self, bs: Any, offset: int) -> int:
        self.bitAwareSeek(bs, offset, 0x0C, 0x08)
        _vertexSize = bs.readUInt()
        self.bitAwareSeek(bs, offset, 0x30, 0x1C)
        self.models[-1].meshes[-1].platform = bs.readUInt()
        self.bitAwareSeek(bs, offset, 0x20, 0x14)
        self.models[-1].meshes[-1].platformData = self.readMemoryRef(bs)
        self.bitAwareSeek(bs, offset, 0x10, 0x0C)
        _elements = self.readMemoryRef(bs)
        elementCount = _elements[0] // 0x0C
        self.bitAwareSeek(bs, offset, 0x58, 0x30)
        _streams = self.readMemoryRef(bs)
        if _streams[1] != 0:
            bs.seek(_streams[1])
            for i in range(0, _streams[0], 4):
                self.models[-1].meshes[-1].vertexStreams.append(bs.readUInt())
            print(
                f"{hex(len(self.models[-1].meshes[-1].vertexStreams))} streams at {hex(_streams[1])}")
        else:
            self.models[-1].meshes[-1].vertexStreams.append(_vertexSize)

        if self.models[-1].meshes[-1].platformData[0] > 0:
            print(
                f"platformData offset: {hex(self.models[-1].meshes[-1].platformData[1])}")
            print(
                f"platformData size: {hex(self.models[-1].meshes[-1].platformData[0])}")

        endarg = '>' if self.endianness == "BE" else '<'
        for i in range(elementCount):
            self.models[-1].meshes[-1].vertexElements.append(
                formats.igVertexElement(_elements[2][i * 0x0C: (i + 1) * 0x0C], endarg))
        return _vertexSize

    def process_igIndexBuffer(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x0C, 0x08)
        self.models[-1].meshes[-1].indexCount = bs.readUInt()
        self.bitAwareSeek(bs, offset, 0x20, 0x14)
        _data = self.readMemoryRefHandle(bs)
        self.models[-1].meshes[-1].indexBuffer = _data[2]
        self.bitAwareSeek(bs, offset, 0x30, 0x1C)
        primType = bs.readInt()
        if primType == 0:
            primType = constants.PrimitiveType.POINTS
        elif primType == 3:
            primType = constants.PrimitiveType.TRIANGLE
        elif primType == 4:
            primType = constants.PrimitiveType.TRIANGLE_STRIP
        elif primType == 5:
            primType = constants.PrimitiveType.TRIANGLE_FAN
        elif primType == 6:
            primType = constants.PrimitiveType.TRIANGLE_QUADS
        else:
            raise NotImplementedError(
                f"primitive type {hex(primType)} is not supported.")
        self.models[-1].meshes[-1].primType = primType

        print(f"indexCount:   {hex(self.models[-1].meshes[-1].indexCount)}")
        print(f"index offset: {hex(_data[1])}")
        print(f"index buf size: {hex(_data[0])}")

    def process_igPS3EdgeGeometry(self, bs: Any, offset: int) -> None:
        # igPS3EdgeGeometry inherits from igPS3EdgeGeometrySegmentList which inherits from igObjectList<igPS3EdgeGeometrySegment>
        geometries = self.process_igObjectList(bs, offset)
        bs.seek(offset + 0x19, constants.SeekMode.ABS)
        _isSkinned = bs.readUByte()

        index = 0
        self.models[-1].meshes[-1].isPs3 = True

        for geom in geometries:
            spuConfigInfo = geom[0]

            print(f"indexCount:   {hex(spuConfigInfo.numIndexes)}")
            print(f"index offset: {hex(geom[1][1])}")
            print(f"index buf size: {hex(geom[2][0])}")
            print(f"vertexCount:  {hex(spuConfigInfo.numVertexes)}")
            print(f"vertex offset: {hex(geom[2][1])}")
            print(f"vertex buf size: {hex(geom[2][0])}")

            # For Blender, we'll need to implement a proper edge decompression
            # This is a placeholder - we'd need to implement decompressEdgeIndices
            edgeDecomp = utils.decompressEdgeIndices(
                geom[1][2], spuConfigInfo.numIndexes)

            segment = formats.PS3MeshObject()

            segment.spuConfigInfo = spuConfigInfo
            segment.vertexBuffers.extend(
                [geom[2][2], geom[3][2], geom[4][2], geom[8][2]])
            segment.vertexCount = spuConfigInfo.numVertexes
            segment.vertexStrides.extend(
                [geom[5].vertexStride, geom[6].vertexStride, geom[7].vertexStride])
            segment.indexBuffer = edgeDecomp
            segment.indexCount = spuConfigInfo.numIndexes
            segment.vertexElements.extend([geom[5], geom[6], geom[7]])
            self.models[-1].meshes[-1].ps3Segments.append(segment)
            index += 1

    def process_igPS3EdgeGeometrySegment(self, bs: Any, offset: int) -> tuple:
        # PS3 likes to have sub sub meshes for some reason so we merge them into one submesh
        bs.seek(offset + 0x08, constants.SeekMode.ABS)
        _spuConfigInfo = self.readMemoryRef(bs)
        bs.seek(offset + 0x10, constants.SeekMode.ABS)
        _indexes = self.readMemoryRef(bs)
        bs.seek(offset + 0x1C, constants.SeekMode.ABS)
        _spuVertexes0 = self.readMemoryRef(bs)
        bs.seek(offset + 0x24, constants.SeekMode.ABS)
        _spuVertexes1 = self.readMemoryRef(bs)
        bs.seek(offset + 0x38, constants.SeekMode.ABS)
        _rsxOnlyVertexes = self.readMemoryRef(bs)
        bs.seek(offset + 0x44, constants.SeekMode.ABS)
        _skinMatrixByteOffsets0 = bs.readUShort()
        _skinMatrixByteOffsets1 = bs.readUShort()
        _skinMatricesSizes0 = bs.readUShort()
        _skinMatricesSizes1 = bs.readUShort()
        bs.seek(offset + 0x50, constants.SeekMode.ABS)
        _skinIndexesAndWeights = self.readMemoryRef(bs)
        print(
            f"_skinIndexesAndWeights Buffer @ {hex(_skinIndexesAndWeights[1])}")
        print(f"_spuConfigInfo Buffer @ {hex(_spuConfigInfo[1])}")
        bs.seek(offset + 0x60, constants.SeekMode.ABS)
        _spuInputStreamDescs0 = self.readMemoryRef(bs)
        bs.seek(offset + 0x68, constants.SeekMode.ABS)
        _spuInputStreamDescs1 = self.readMemoryRef(bs)
        bs.seek(offset + 0x78, constants.SeekMode.ABS)
        _rsxOnlyStreamDesc = self.readMemoryRef(bs)
        spuConfigInfoObject = formats.EdgeGeomSpuConfigInfo(_spuConfigInfo[2])
        spuConfigInfoObject.skinMatrixOffset0 = _skinMatrixByteOffsets0
        spuConfigInfoObject.skinMatrixOffset1 = _skinMatrixByteOffsets1
        spuConfigInfoObject.skinMatrixSize0 = _skinMatricesSizes0
        spuConfigInfoObject.skinMatrixSize1 = _skinMatricesSizes1
        # Return values in order:
        # 0=spuConfigInfo, 1=indexes, 2=spuVertexes0, 3=spuVertexes1, 4=rsxOnlyVertexes,
        # 5=spuInputStreamDescs0, 6=spuInputStreamDescs1, 7=rsxOnlyStreamDesc, 8=skinIndexesAndWeights
        return (spuConfigInfoObject, _indexes, _spuVertexes0, _spuVertexes1, _rsxOnlyVertexes,
                formats.EdgeGeometryVertexDescriptor(_spuInputStreamDescs0[2]),
                formats.EdgeGeometryVertexDescriptor(_spuInputStreamDescs1[2]),
                formats.EdgeGeometryVertexDescriptor(_rsxOnlyStreamDesc[2]),
                _skinIndexesAndWeights)


# ------------------------------------------------------------------------------
# Swap Force implementation
# ------------------------------------------------------------------------------
class ssfIgzFile(igz_file.igzFile):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.is64Bit = ssfIgzFile.is64BitCall
        self.arkRegisteredTypes = ssfarkRegisteredTypes

    def is64BitCall(self) -> bool:
        platformbittness = [
            False,  # IG_CORE_PLATFORM_DEFAULT
            False,  # IG_CORE_PLATFORM_WIN32
            False,  # IG_CORE_PLATFORM_WII
            True,   # IG_CORE_PLATFORM_DURANGO
            False,  # IG_CORE_PLATFORM_ASPEN
            False,  # IG_CORE_PLATFORM_XENON
            False,  # IG_CORE_PLATFORM_PS3
            False,  # IG_CORE_PLATFORM_OSX
            True,   # IG_CORE_PLATFORM_WIN64
            False,  # IG_CORE_PLATFORM_CAFE
            False,  # IG_CORE_PLATFORM_RASPI
            False,  # IG_CORE_PLATFORM_ANDROID
            False,  # IG_CORE_PLATFORM_MARMALADE
            False,  # IG_CORE_PLATFORM_LGTV
            True,   # IG_CORE_PLATFORM_PS4
            False,  # IG_CORE_PLATFORM_WP8
            False,  # IG_CORE_PLATFORM_LINUX
            False,  # IG_CORE_PLATFORM_MAX
        ]
        return platformbittness[self.platform]

    def process_igSceneInfo(self, bs: Any, offset: int) -> None:
        self.models.append(formats.ModelObject())
        self.bitAwareSeek(bs, offset, 0x00, 0x14)
        _sceneGraph = self.process_igObject(bs, self.readPointer(bs))

    def process_igGroup(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0x20)
        _childList = self.process_igObject(bs, self.readPointer(bs))

    def process_igTransform(self, bs: Any, offset: int) -> None:
        self.process_igGroup(bs, offset)

    def process_igFxMaterialNode(self, bs: Any, offset: int) -> None:
        self.process_igGroup(bs, offset)

    def process_igGeometry(self, bs: Any, offset: int) -> None:
        ssfIgzFile.process_igGroup(self, bs, offset)
        self.bitAwareSeek(bs, offset, 0x00, 0x24)
        mesh = formats.MeshObject()
        if self.models[-1].boneMapList is not None and len(self.models[-1].boneMapList) > 0:
            mesh.boneMapIndex = len(self.models[-1].boneMapList)-1
        self.models[-1].meshes.append(mesh)
        _attrList = self.process_igObject(bs, self.readPointer(bs))

    def process_igEdgeGeometryAttr(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0x10)
        _geometry = self.process_igObject(bs, self.readPointer(bs))

    def process_igGeometryAttr(self, bs: Any, offset: int) -> None:
        self.models[-1].meshes.append(formats.MeshObject())
        self.bitAwareSeek(bs, offset, 0x00, 0x10)
        _vertexBuffer = self.process_igObject(bs, self.readPointer(bs))
        self.bitAwareSeek(bs, offset, 0x00, 0x14)
        print("I'M GOING TO READ THE INDEX BUFFER NOW")
        _indexBuffer = self.process_igObject(bs, self.readPointer(bs))

    def process_asAnimationDatabase(self, bs: Any, offset: int) -> None:
        self.models.append(formats.ModelObject())
        self.bitAwareSeek(bs, offset, 0x00, 0x14)
        _skeleton = self.process_igObject(bs, self.readPointer(bs))
        self.bitAwareSeek(bs, offset, 0x00, 0x18)
        _skin = self.process_igObject(bs, self.readPointer(bs))

    def process_igAttrSet(self, bs: Any, offset: int) -> None:
        ssfIgzFile.process_igGroup(self, bs, offset)
        self.bitAwareSeek(bs, offset, 0x00, 0x24)
        _attributes = self.process_igObject(bs, self.readPointer(bs))

    def process_igBlendMatrixSelect(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0xB4)
        self.models[-1].boneMapList.append(
            self.process_igObject(bs, self.readPointer(bs)))
        ssfIgzFile.process_igAttrSet(self, bs, offset)

    def process_igAnimation2Info(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0x14)
        _animationList = self.process_igObject(bs, self.readPointer(bs))

    def process_igSkeleton2Info(self, bs: Any, offset: int) -> None:
        self.bitAwareSeek(bs, offset, 0x00, 0x14)
        _skeletonList = self.process_igObject(bs, self.readPointer(bs))

# SSA Wii U "forward declarations"
    def process_tfbSpriteInfo(self, bs: Any, offset: int) -> None:
        pass

    def process_tfbPhysicsModel(self, bs: Any, offset: int) -> None:
        pass

    def process_tfbPhysicsBody(self, bs: Any, offset: int) -> None:
        pass

    def process_tfbEntityInfo(self, bs: Any, offset: int) -> None:
        pass

    def process_Drawable(self, bs: Any, offset: int) -> None:
        pass

    def process_tfbPhysicsWorld(self, bs: Any, offset: int) -> None:
        pass

    def process_tfbPhysicsCombinerLink(self, bs: Any, offset: int) -> None:
        pass

    def process_tfbActorInfo(self, bs: Any, offset: int) -> None:
        pass

    def process_tfbRuntimeTechniqueInstance(self, bs: Any, offset: int) -> None:
        pass


 # Registry for the SSF (Swap Force) format
ssfarkRegisteredTypes = {
    "igDataList": igz_file.igzFile.process_igDataList,
    "igNamedObject": igz_file.igzFile.process_igNamedObject,
    "igObjectList": igz_file.igzFile.process_igObjectList,
    "igSkeleton2": sscIgzFile.process_igSkeleton2,
    "igSkeletonBoneList": sscIgzFile.process_igSkeletonBoneList,
    "igSkeletonBone": sscIgzFile.process_igSkeletonBone,
    "igGraphicsVertexBuffer": sscIgzFile.process_igGraphicsVertexBuffer,
    "igGraphicsIndexBuffer": sscIgzFile.process_igGraphicsIndexBuffer,
    "igVertexBuffer": sscIgzFile.process_igVertexBuffer,
    "igVertexFormat": sscIgzFile.process_igVertexFormat,
    "igIndexBuffer": sscIgzFile.process_igIndexBuffer,
    "igPS3EdgeGeometry": sscIgzFile.process_igPS3EdgeGeometry,
    "igPS3EdgeGeometrySegment": sscIgzFile.process_igPS3EdgeGeometrySegment,
    "igSceneInfo": ssfIgzFile.process_igSceneInfo,
    "igGroup": ssfIgzFile.process_igGroup,
    "igActor2": ssfIgzFile.process_igGroup,
    "igTransform": ssfIgzFile.process_igTransform,
    "igFxMaterialNode": ssfIgzFile.process_igFxMaterialNode,
    "igGeometry": ssfIgzFile.process_igGeometry,
    "igWiiGeometry": ssfIgzFile.process_igGeometry,
    "igNodeList": ssfIgzFile.process_igObjectList,
    "igAttrList": ssfIgzFile.process_igObjectList,
    "igEdgeGeometryAttr": ssfIgzFile.process_igEdgeGeometryAttr,
    "igGeometryAttr": ssfIgzFile.process_igGeometryAttr,
    "igWiiGeometryAttr": ssfIgzFile.process_igGeometryAttr,
    "asAnimationDatabase": ssfIgzFile.process_asAnimationDatabase,
    "igAttrSet": ssfIgzFile.process_igAttrSet,
    "igBlendMatrixSelect": ssfIgzFile.process_igBlendMatrixSelect,
    "igIntList": igz_file.igzFile.process_igIntList,

    # Lost Islands Exclusive Types
    "igSkeleton2Info": ssfIgzFile.process_igSkeleton2Info,
    "igSkeleton2List": ssfIgzFile.process_igObjectList,
    "igAnimation2Info": ssfIgzFile.process_igAnimation2Info,
    "igAnimation2List": ssfIgzFile.process_igObjectList,

    # SSA Wii U
    "tfbSpriteInfo": ssfIgzFile.process_tfbSpriteInfo,
    "tfbPhysicsModel": ssfIgzFile.process_tfbPhysicsModel,
    "tfbPhysicsBody": ssfIgzFile.process_tfbPhysicsBody,
    "tfbBodyEntityInfo": ssfIgzFile.process_tfbEntityInfo
}

# Registry for the STT (Trap Team) format
sttarkRegisteredTypes = {
    "igDataList": igz_file.igzFile.process_igDataList,
    "igNamedObject": igz_file.igzFile.process_igNamedObject,
    "igObjectList": igz_file.igzFile.process_igObjectList,
    "igSkeleton2": sscIgzFile.process_igSkeleton2,
    "igSkeletonBoneList": sscIgzFile.process_igSkeletonBoneList,
    "igSkeletonBone": sscIgzFile.process_igSkeletonBone,
    "igGraphicsVertexBuffer": sscIgzFile.process_igGraphicsVertexBuffer,
    "igGraphicsIndexBuffer": sscIgzFile.process_igGraphicsIndexBuffer,
    "igVertexBuffer": sscIgzFile.process_igVertexBuffer,
    "igVertexFormat": sscIgzFile.process_igVertexFormat,
    "igIndexBuffer": sscIgzFile.process_igIndexBuffer,
    "igPS3EdgeGeometry": sscIgzFile.process_igPS3EdgeGeometry,
    "igPS3EdgeGeometrySegment": sscIgzFile.process_igPS3EdgeGeometrySegment,
    "igEdgeGeometryAttr": ssfIgzFile.process_igEdgeGeometryAttr,
    "igGeometryAttr": ssfIgzFile.process_igGeometryAttr,
    "igWiiGeometryAttr": ssfIgzFile.process_igGeometryAttr,
    "igFxMaterialNode": ssfIgzFile.process_igGroup,
    "igNodeList": igz_file.igzFile.process_igObjectList,
    "igIntListList": igz_file.igzFile.process_igObjectList,
    "igIntList": igz_file.igzFile.process_igIntList,
    "tfbSpriteInfo": sttIgzFile.process_tfbSpriteInfo,
    "tfbPhysicsModel": sttIgzFile.process_tfbPhysicsModel,
    "tfbPhysicsBody": sttIgzFile.process_tfbPhysicsBody,
    "tfbBodyEntityInfo": sttIgzFile.process_tfbBodyEntityInfo,
    "DrawableList": sttIgzFile.process_igObjectList,
    "Drawable": sttIgzFile.process_Drawable,
    "tfbPhysicsWorld": sttIgzFile.process_tfbPhysicsWorld,
    "tfbPhysicsCombinerLink": sttIgzFile.process_tfbPhysicsCombinerLink,
    "tfbWorldEntityInfo": sttIgzFile.process_tfbEntityInfo,
    "tfbActorInfo": sttIgzFile.process_tfbActorInfo,

    # STT iOS exclusive types
    "igActor2": ssfIgzFile.process_igGroup,
    "tfbPointLightPicker": ssfIgzFile.process_igGroup,
    "igBlendMatrixSelect": ssfIgzFile.process_igBlendMatrixSelect,
    "tfbMobileLodGeometry": sttIgzFile.process_tfbMobileLodGeometry,
    "igAttrList": igz_file.igzFile.process_igObjectList,
    "igGroup": ssfIgzFile.process_igGroup,
    "igGeometry": ssfIgzFile.process_igGeometry
}

# Registry for the SG (Giants) format
sgarkRegisteredTypes = {
    "igDataList": igz_file.igzFile.process_igDataList,
    "igNamedObject": igz_file.igzFile.process_igNamedObject,
    "igObjectList": igz_file.igzFile.process_igObjectList,
    "igSkeleton2": sscIgzFile.process_igSkeleton2,
    "igSkeletonBoneList": sscIgzFile.process_igSkeletonBoneList,
    "igSkeletonBone": sscIgzFile.process_igSkeletonBone,
    "igGraphicsVertexBuffer": sscIgzFile.process_igGraphicsVertexBuffer,
    "igGraphicsIndexBuffer": sscIgzFile.process_igGraphicsIndexBuffer,
    "igVertexBuffer": sscIgzFile.process_igVertexBuffer,
    "igVertexFormat": sscIgzFile.process_igVertexFormat,
    "igIndexBuffer": sscIgzFile.process_igIndexBuffer,
    "igPS3EdgeGeometry": sscIgzFile.process_igPS3EdgeGeometry,
    "igPS3EdgeGeometrySegment": sscIgzFile.process_igPS3EdgeGeometrySegment,
    "igEdgeGeometryAttr": sgIgzFile.process_igEdgeGeometryAttr,
    "igGeometryAttr": ssfIgzFile.process_igGeometryAttr,
    "igWiiGeometryAttr": ssfIgzFile.process_igGeometryAttr,
    "igFxMaterialNode": ssfIgzFile.process_igGroup,
    "igActor2": ssfIgzFile.process_igGroup,
    "igGroup": ssfIgzFile.process_igGroup,
    "igNodeList": ssfIgzFile.process_igObjectList,
    "tfbSpriteInfo": sgIgzFile.process_tfbSpriteInfo,
    "tfbPhysicsModel": sttIgzFile.process_tfbPhysicsModel,
    "tfbPhysicsBody": sgIgzFile.process_tfbPhysicsBody,
    "tfbBodyEntityInfo": sttIgzFile.process_tfbEntityInfo,
    "DrawableList": sttIgzFile.process_igObjectList,
    "Drawable": sttIgzFile.process_Drawable,
    "tfbPhysicsWorld": sgIgzFile.process_tfbPhysicsWorld,
    "igSceneInfo": ssfIgzFile.process_igSceneInfo,
    "igSpatialNode": ssfIgzFile.process_igGroup,
    "tfbPhysicsCombinerLink": sttIgzFile.process_tfbPhysicsCombinerLink,
    "tfbWorldEntityInfo": sttIgzFile.process_tfbEntityInfo,
    "tfbActorInfo": sgIgzFile.process_tfbActorInfo,
    "igBlendMatrixSelect": ssfIgzFile.process_igBlendMatrixSelect,
    "igIntList": igz_file.igzFile.process_igIntList,
    "tfbRuntimeTechniqueInstance": sgIgzFile.process_tfbRuntimeTechniqueInstance
}

# Registry for the SSA (Spyro's Adventure) format
ssaarkRegisteredTypes = {
    "igDataList": igz_file.igzFile.process_igDataList,
    "igNamedObject": igz_file.igzFile.process_igNamedObject,
    "igObjectList": igz_file.igzFile.process_igObjectList,
    "igSkeleton2": sscIgzFile.process_igSkeleton2,
    "igSkeletonBoneList": sscIgzFile.process_igSkeletonBoneList,
    "igSkeletonBone": sscIgzFile.process_igSkeletonBone,
    "igGraphicsVertexBuffer": sscIgzFile.process_igGraphicsVertexBuffer,
    "igGraphicsIndexBuffer": sscIgzFile.process_igGraphicsIndexBuffer,
    "igVertexBuffer": sscIgzFile.process_igVertexBuffer,
    "igVertexFormat": sscIgzFile.process_igVertexFormat,
    "igIndexBuffer": sscIgzFile.process_igIndexBuffer,
    "igPS3EdgeGeometry": sscIgzFile.process_igPS3EdgeGeometry,
    "igPS3EdgeGeometrySegment": sscIgzFile.process_igPS3EdgeGeometrySegment,
    "igEdgeGeometryAttr": sgIgzFile.process_igEdgeGeometryAttr,
    "igGeometryAttr": ssfIgzFile.process_igGeometryAttr,
    "igWiiGeometryAttr": ssfIgzFile.process_igGeometryAttr,
    "igFxMaterialNode": ssfIgzFile.process_igGroup,
    "igActor2": ssfIgzFile.process_igGroup,
    "igGroup": ssfIgzFile.process_igGroup,
    "igNodeList": ssfIgzFile.process_igObjectList,
    "tfbSpriteInfo": sgIgzFile.process_tfbSpriteInfo,
    "tfbPhysicsModel": sttIgzFile.process_tfbPhysicsModel,
    "tfbPhysicsBody": sgIgzFile.process_tfbPhysicsBody,
    "tfbBodyEntityInfo": sttIgzFile.process_tfbEntityInfo,
    "DrawableList": sttIgzFile.process_igObjectList,
    "Drawable": sttIgzFile.process_Drawable,
    "tfbPhysicsWorld": sgIgzFile.process_tfbPhysicsWorld,
    "igSceneInfo": ssfIgzFile.process_igSceneInfo,
    "igSpatialNode": ssfIgzFile.process_igGroup,
    "tfbPhysicsCombinerLink": sttIgzFile.process_tfbPhysicsCombinerLink,
    "tfbWorldEntityInfo": sttIgzFile.process_tfbEntityInfo,
    "tfbActorInfo": sgIgzFile.process_tfbActorInfo,
    "igBlendMatrixSelect": ssaIgzFile.process_igBlendMatrixSelect,
    "igIntList": igz_file.igzFile.process_igIntList,
    "tfbRuntimeTechniqueInstance": sgIgzFile.process_tfbRuntimeTechniqueInstance
}

# Registry for the SSC (SuperChargers) format
sscarkRegisteredTypes = {
    "igDataList": igz_file.igzFile.process_igDataList,
    "igNamedObject": igz_file.igzFile.process_igNamedObject,
    "igObjectList": igz_file.igzFile.process_igObjectList,
    "CGraphicsSkinInfo": sscIgzFile.process_CGraphicsSkinInfo,
    "igSkeleton2": sscIgzFile.process_igSkeleton2,
    "igSkeletonBoneList": sscIgzFile.process_igSkeletonBoneList,
    "igSkeletonBone": sscIgzFile.process_igSkeletonBone,
    "igModelInfo": sscIgzFile.process_igModelInfo,
    "igModelData": sscIgzFile.process_igModelData,
    "igModelDrawCallData": sscIgzFile.process_igModelDrawCallData,
    "igGraphicsVertexBuffer": sscIgzFile.process_igGraphicsVertexBuffer,
    "igGraphicsIndexBuffer": sscIgzFile.process_igGraphicsIndexBuffer,
    "igVertexBuffer": sscIgzFile.process_igVertexBuffer,
    "igVertexFormat": sscIgzFile.process_igVertexFormat,
    "igIndexBuffer": sscIgzFile.process_igIndexBuffer,
    "igPS3EdgeGeometry": sscIgzFile.process_igPS3EdgeGeometry,
    "igPS3EdgeGeometrySegment": sscIgzFile.process_igPS3EdgeGeometrySegment,
}
