import struct
import io
from .spline_decompressor import SplineDecompressor


class BinaryReader:
    def __init__(self, data):
        self.data = data
        self.offset = 0
        self.endian = "<"  # Default to little endian

    def set_endian(self, little_endian):
        self.endian = "<" if little_endian else ">"

    def read(self, fmt):
        size = struct.calcsize(fmt)
        val = struct.unpack(
            self.endian + fmt, self.data[self.offset : self.offset + size]
        )
        self.offset += size
        return val

    def read_struct(self, fmt):
        return self.read(fmt)

    def read_bytes(self, size):
        val = self.data[self.offset : self.offset + size]
        self.offset += size
        return val

    def read_string(self, size):
        b = self.read_bytes(size)
        # Find first null byte
        end = b.find(b"\0")
        if end != -1:
            b = b[:end]
        return b.decode("ascii", errors="ignore")

    def seek(self, pos, whence=0):
        if whence == 0:
            self.offset = pos
        elif whence == 1:
            self.offset += pos
        elif whence == 2:
            self.offset = len(self.data) + pos

    def tell(self):
        return self.offset


class hkxHeaderLayout:
    def __init__(
        self, bytes_in_pointer, little_endian, reuse_padding_opt, empty_base_class_opt
    ):
        self.bytes_in_pointer = bytes_in_pointer
        self.little_endian = little_endian
        self.reuse_padding_opt = reuse_padding_opt
        self.empty_base_class_opt = empty_base_class_opt


class hkxHeader:
    def __init__(self):
        self.magic1 = 0
        self.magic2 = 0
        self.user_tag = 0
        self.version = 0
        self.layout = None
        self.num_sections = 0
        self.contents_section_index = 0
        self.contents_section_offset = 0
        self.contents_class_name_section_index = 0
        self.contents_class_name_section_offset = 0
        self.contents_version = ""
        self.flags = 0
        self.max_predicate = 0
        self.predicate_array_size_plus_padding = 0
        self.sections = []

    def load(self, reader):
        # Read hkxHeaderData
        (self.magic1, self.magic2, self.user_tag, self.version) = reader.read("IIII")

        layout_bytes = reader.read("BBBB")
        self.layout = hkxHeaderLayout(*layout_bytes)

        (
            self.num_sections,
            self.contents_section_index,
            self.contents_section_offset,
            self.contents_class_name_section_index,
            self.contents_class_name_section_offset,
        ) = reader.read("iiiii")

        self.contents_version = reader.read_string(16)
        (self.flags,) = reader.read("I")
        (self.max_predicate, self.predicate_array_size_plus_padding) = reader.read("hh")

        # Check magic
        if self.magic1 != 0x57E0E057:
            # Try swapping endianness if magic doesn't match
            # But wait, we read as little endian by default.
            # If magic is wrong, maybe it's big endian?
            # The C++ code checks magic AFTER reading.
            pass

        # Update reader endianness based on layout
        reader.set_endian(self.layout.little_endian)

        # If we needed to swap endianness of the header data we just read, we should do it here.
        # But for now let's assume we can re-read or just proceed if it matches.
        # Actually, if layout.littleEndian is 0, we need to swap what we just read.
        if not self.layout.little_endian:
            # Re-read header with big endian? Or just swap the values we have.
            # For simplicity, let's assume we might need to re-parse if we detect wrong endianness early on.
            pass

        if self.max_predicate != -1:
            reader.seek(self.predicate_array_size_plus_padding, 1)

        # Read sections
        for i in range(self.num_sections):
            section = hkxSectionHeader(self)
            section.section_id = i
            section.load_header_data(reader)
            self.sections.append(section)

        # Handle version > 9 padding
        if self.version > 9:
            # The C++ code says: if (version > 9) rd.Seek(16, std::ios_base::cur);
            # But it does this INSIDE the loop in C++?
            # "for (auto &s : sections) { ... if (version > 9) rd.Seek(16); ... }"
            # Yes, it seems there is padding after EACH section header.
            pass

        # Load section data
        for section in self.sections:
            section.load_data(reader)

        # Link buffers
        for section in self.sections:
            section.link_buffer()

        # Debug: print root class name
        root_class_name = self.read_string_at(
            self.contents_class_name_section_index,
            self.contents_class_name_section_offset,
        )

    def get_section(self, index):
        if 0 <= index < len(self.sections):
            return self.sections[index]
        return None

    def read_pointer(self, section_index, offset):
        section = self.get_section(section_index)
        if not section:
            return None

        # Check if there is a fixup at this offset
        if offset in section.pointer_map:
            target_section_id, target_offset = section.pointer_map[offset]
            return (target_section_id, target_offset)

        # If no fixup, read the value at the offset (it might be a null pointer or relative offset)
        # But wait, if it's a pointer, it MUST have a fixup if it points to something valid?
        # Or maybe it's just an offset?
        # In Havok binary, pointers are stored as offsets relative to the section start.
        # But if it points to another section, it MUST have a global fixup.
        # If it points within the same section, it MUST have a local fixup?
        # Not necessarily. If it's 0, it's null.
        # If it's not 0, it's an offset.

        # Let's read the raw value
        ptr_size = self.layout.bytes_in_pointer
        if offset + ptr_size > len(section.data):
            return None

        raw_val = int.from_bytes(
            section.data[offset : offset + ptr_size],
            "little" if self.layout.little_endian else "big",
        )

        if raw_val == 0:  # Null pointer
            return None

        # If it's not 0 and no fixup, it's a local offset?
        # The C++ code says:
        # for (auto &lf : localFixups) { *ptr = sectionBuffer + lf.destination; }
        # So the value in the file is IGNORED and replaced by destination.
        # So if there is no fixup, it's likely NULL or invalid?
        # Wait, "lf.pointer" is the location OF the pointer.
        # So if we are reading a pointer at "offset", we check if "offset" is in localFixups.

        return (section_index, raw_val)  # Assume local offset if no fixup?

    def read_hkarray(self, section_index, offset):
        ptr_size = self.layout.bytes_in_pointer

        # data (ptr)
        data_ptr = self.read_pointer(section_index, offset)
        offset += ptr_size

        # count (int32)
        section = self.get_section(section_index)
        count = int.from_bytes(
            section.data[offset : offset + 4],
            "little" if self.layout.little_endian else "big",
        )
        offset += 4

        # capacityAndFlags (int32)
        capacity = int.from_bytes(
            section.data[offset : offset + 4],
            "little" if self.layout.little_endian else "big",
        )
        offset += 4

        # Handle potential swapped count/capacity (observed in some PS4 files)
        if count == 0 and capacity > 0:
            # Heuristic: if count is 0 but capacity is set, and capacity looks like a valid count/size
            # (e.g. matches num_blocks or data size), assume swapped.
            # For now, just swap if count is 0.
            count, capacity = capacity, count

        # Handle masked pointer (observed 0x80000000 base)
        if data_ptr:
            sid, soff = data_ptr
            if soff & 0x80000000:
                soff = soff & 0x7FFFFFFF
                data_ptr = (sid, soff)

        return data_ptr, count, capacity

    def read_string_ptr(self, section_index, offset):
        ptr = self.read_pointer(section_index, offset)
        if not ptr:
            return ""

        sid, soff = ptr
        section = self.get_section(sid)
        if not section:
            return ""

        # Read string until null terminator
        end = section.data.find(b"\0", soff)
        if end == -1:
            return section.data[soff:].decode("ascii", errors="ignore")
        return section.data[soff:end].decode("ascii", errors="ignore")

    def read_string_at(self, section_index, offset):
        section = self.get_section(section_index)
        if not section:
            return ""
        end = section.data.find(b"\0", offset)
        if end == -1:
            return section.data[offset:].decode("ascii", errors="ignore")
        return section.data[offset:end].decode("ascii", errors="ignore")

    def get_root_level_container(self):
        # Root level container is at contents_section_index + contents_section_offset
        # It appears to be just an hkArray<hkNamedVariant> without hkReferencedObject header
        # in this file version/platform.

        sid = self.contents_section_index
        soff = self.contents_section_offset

        variants_ptr, variants_size, _ = self.read_hkarray(sid, soff)

        variants = []
        if variants_ptr:
            vsid, vsoff = variants_ptr
            ptr_size = self.layout.bytes_in_pointer
            # hkNamedVariant2_t size = 3 * ptr_size
            variant_size = 3 * ptr_size

            for i in range(variants_size):
                offset = vsoff + i * variant_size

                name = self.read_string_ptr(vsid, offset)
                class_name = self.read_string_ptr(vsid, offset + ptr_size)
                variant_ptr = self.read_pointer(vsid, offset + 2 * ptr_size)

                variants.append(
                    {"name": name, "class_name": class_name, "variant_ptr": variant_ptr}
                )

        return variants

    def read_hka_animation_container(self, section_index, offset):
        ptr_size = self.layout.bytes_in_pointer

        # Skip hkReferenceObject
        # vtable (ptr) + memSizeAndFlags (2) + referenceCount (2) + padding (4 on 64-bit)
        # On 64-bit: 8 + 2 + 2 + 4 = 16 bytes
        # On 32-bit: 4 + 2 + 2 = 8 bytes
        header_size = 16 if ptr_size == 8 else 8
        current_offset = offset + header_size

        # Skeletons (hkArray)
        skeletons_ptr, skeletons_size, _ = self.read_hkarray(
            section_index, current_offset
        )
        current_offset += 16 if ptr_size == 8 else 12

        # Animations (hkArray)
        animations_ptr, animations_size, _ = self.read_hkarray(
            section_index, current_offset
        )
        current_offset += 16 if ptr_size == 8 else 12

        # Bindings (hkArray)
        bindings_ptr, bindings_size, _ = self.read_hkarray(
            section_index, current_offset
        )
        current_offset += 16 if ptr_size == 8 else 12

        # Attachments (hkArray)
        attachments_ptr, attachments_size, _ = self.read_hkarray(
            section_index, current_offset
        )
        current_offset += 16 if ptr_size == 8 else 12

        # Skins (hkArray)
        skins_ptr, skins_size, _ = self.read_hkarray(section_index, current_offset)

        return {
            "skeletons": (skeletons_ptr, skeletons_size),
            "animations": (animations_ptr, animations_size),
            "bindings": (bindings_ptr, bindings_size),
            "attachments": (attachments_ptr, attachments_size),
            "skins": (skins_ptr, skins_size),
        }

    def read_hka_skeleton(self, section_index, offset):
        ptr_size = self.layout.bytes_in_pointer

        # Skip hkReferencedObject
        header_size = 16 if ptr_size == 8 else 8
        current_offset = offset + header_size

        # name (string ptr)
        name = self.read_string_ptr(section_index, current_offset)
        current_offset += ptr_size

        # parentIndices (hkArray<int16>)
        parent_indices_ptr, parent_indices_size, _ = self.read_hkarray(
            section_index, current_offset
        )
        current_offset += 16 if ptr_size == 8 else 12

        # bones (hkArray<hkaBone>)
        bones_ptr, bones_size, _ = self.read_hkarray(section_index, current_offset)
        current_offset += 16 if ptr_size == 8 else 12

        # referencePose (hkArray<hkQTransform>)
        ref_pose_ptr, ref_pose_size, _ = self.read_hkarray(
            section_index, current_offset
        )

        # Read parent indices
        parent_indices = []
        if parent_indices_ptr:
            sid, soff = parent_indices_ptr
            section = self.get_section(sid)
            for i in range(parent_indices_size):
                val = int.from_bytes(
                    section.data[soff + i * 2 : soff + i * 2 + 2],
                    "little" if self.layout.little_endian else "big",
                    signed=True,
                )
                parent_indices.append(val)

        # Read bones
        bones = []
        if bones_ptr:
            sid, soff = bones_ptr
            # hkaBone size?
            # name (ptr) + lockTranslation (1) + padding?
            # On 64-bit: 8 + 1 + 7 padding = 16 bytes?
            # On 32-bit: 4 + 1 + 3 padding = 8 bytes?
            bone_struct_size = 16 if ptr_size == 8 else 8

            for i in range(bones_size):
                bone_offset = soff + i * bone_struct_size
                bone_name = self.read_string_ptr(sid, bone_offset)

                bones.append(
                    {
                        "name": bone_name,
                        "parent": parent_indices[i] if i < len(parent_indices) else -1,
                    }
                )

        # Read reference pose (transforms)
        ref_poses = []
        if ref_pose_ptr:
            sid, soff = ref_pose_ptr
            transform_size = 48  # hkQTransform is 48 bytes (T, R, S)
            section = self.get_section(sid)

            for i in range(ref_pose_size):
                t_off = soff + i * transform_size
                # Read 12 floats
                floats = []
                for j in range(12):
                    f_val = struct.unpack(
                        "<f" if self.layout.little_endian else ">f",
                        section.data[t_off + j * 4 : t_off + j * 4 + 4],
                    )[0]
                    floats.append(f_val)

                translation = floats[0:3]  # Ignore W
                rotation = floats[4:8]  # x,y,z,w
                scale = floats[8:11]  # Ignore W

                ref_poses.append(
                    {"translation": translation, "rotation": rotation, "scale": scale}
                )

        return {"name": name, "bones": bones, "ref_poses": ref_poses}

    def read_hka_animation(self, section_index, offset):
        ptr_size = self.layout.bytes_in_pointer

        # Skip hkReferencedObject
        header_size = 16 if ptr_size == 8 else 8
        current_offset = offset + header_size

        # Note: In some versions/platforms (like PS4/Havok 2014), m_type seems to be missing
        # or the layout is different. We observe duration (float) as the first field.

        section = self.get_section(section_index)

        # duration (float)
        duration = struct.unpack(
            "<f" if self.layout.little_endian else ">f",
            section.data[current_offset : current_offset + 4],
        )[0]
        current_offset += 4

        # numberOfTransformTracks (int32)
        num_transform_tracks = int.from_bytes(
            section.data[current_offset : current_offset + 4],
            "little" if self.layout.little_endian else "big",
        )
        current_offset += 4

        # numberOfFloatTracks (int32)
        num_float_tracks = int.from_bytes(
            section.data[current_offset : current_offset + 4],
            "little" if self.layout.little_endian else "big",
        )
        current_offset += 4

        # extractedMotion (ptr)
        extracted_motion_ptr = self.read_pointer(section_index, current_offset)
        current_offset += ptr_size

        # annotationTracks (hkArray)
        annotation_tracks_ptr, annotation_tracks_size, _ = self.read_hkarray(
            section_index, current_offset
        )
        current_offset += 16 if ptr_size == 8 else 12

        tracks = []
        num_frames = 0

        # Check for Spline Compressed Animation fields
        # We observed an unknown field (possibly numTransformTracks | 0x80000000) before numFrames

        # Peek at next value
        v1 = int.from_bytes(
            section.data[current_offset : current_offset + 4],
            "little" if self.layout.little_endian else "big",
        )

        # If v1 looks like a flag/count, skip it
        if v1 & 0x80000000:
            current_offset += 4

        # Read Spline fields
        num_frames = int.from_bytes(
            section.data[current_offset : current_offset + 4],
            "little" if self.layout.little_endian else "big",
        )
        current_offset += 4
        num_blocks = int.from_bytes(
            section.data[current_offset : current_offset + 4],
            "little" if self.layout.little_endian else "big",
        )
        current_offset += 4
        max_frames_per_block = int.from_bytes(
            section.data[current_offset : current_offset + 4],
            "little" if self.layout.little_endian else "big",
        )
        current_offset += 4
        mask_and_quantization_size = int.from_bytes(
            section.data[current_offset : current_offset + 4],
            "little" if self.layout.little_endian else "big",
        )
        current_offset += 4
        block_duration = struct.unpack(
            "<f" if self.layout.little_endian else ">f",
            section.data[current_offset : current_offset + 4],
        )[0]
        current_offset += 4
        block_inverse_duration = struct.unpack(
            "<f" if self.layout.little_endian else ">f",
            section.data[current_offset : current_offset + 4],
        )[0]
        current_offset += 4
        frame_duration = struct.unpack(
            "<f" if self.layout.little_endian else ">f",
            section.data[current_offset : current_offset + 4],
        )[0]
        current_offset += 4

        # Align to ptr_size before reading arrays
        if current_offset % ptr_size != 0:
            current_offset += ptr_size - (current_offset % ptr_size)

        # Read arrays
        arrays_info = []
        for i in range(5):
            ptr, size, cap = self.read_hkarray(section_index, current_offset)
            arrays_info.append((ptr, size, cap))
            current_offset += 16 if ptr_size == 8 else 12

        block_offsets_ptr, block_offsets_size, _ = arrays_info[0]
        float_block_offsets_ptr, float_block_offsets_size, _ = arrays_info[1]
        transform_offsets_ptr, transform_offsets_size, _ = arrays_info[2]
        float_offsets_ptr, float_offsets_size, _ = arrays_info[3]
        data_ptr, data_size, _ = arrays_info[4]

        # endian (int32)
        endian = int.from_bytes(
            section.data[current_offset : current_offset + 4],
            "little" if self.layout.little_endian else "big",
        )
        current_offset += 4

        # Read block offsets
        block_offsets = []
        if block_offsets_ptr:
            sid, soff = block_offsets_ptr
            b_section = self.get_section(sid)
            for i in range(block_offsets_size):
                val = int.from_bytes(
                    b_section.data[soff + i * 4 : soff + i * 4 + 4],
                    "little" if self.layout.little_endian else "big",
                )
                block_offsets.append(val)

        if not block_offsets and num_blocks > 0:
            if num_blocks == 1:
                block_offsets = [0]
            else:
                print(f"WARNING: num_blocks={num_blocks} but block_offsets is empty!")

        # Read data
        data_bytes = b""
        if data_ptr:
            sid, soff = data_ptr
            d_section = self.get_section(sid)
            data_bytes = d_section.data[soff : soff + data_size]

        decompressor = SplineDecompressor()
        try:
            decompressor.decompress(
                data_bytes,
                block_offsets,
                num_transform_tracks,
                num_float_tracks,
                block_duration,
                little_endian=self.layout.little_endian,
            )
            tracks = decompressor.sample_all_tracks(num_frames, duration)
        except Exception as e:
            print(f"WARNING: Failed to decompress spline animation: {e}")
            import traceback

            traceback.print_exc()
            tracks = [[] for _ in range(num_transform_tracks)]

        return {
            "name": "Animation",
            "duration": duration,
            "tracks": tracks,
            "num_frames": num_frames,
        }

    def read_hka_animation_binding(self, section_index, offset):
        ptr_size = self.layout.bytes_in_pointer

        # Skip hkReferencedObject
        header_size = 16 if ptr_size == 8 else 8
        current_offset = offset + header_size

        # originalSkeletonName (string ptr)
        original_skeleton_name = self.read_string_ptr(section_index, current_offset)
        current_offset += ptr_size

        # animation (ptr)
        animation_ptr = self.read_pointer(section_index, current_offset)
        current_offset += ptr_size

        # transformTrackToBoneIndices (hkArray<int16>)
        track_to_bone_ptr, track_to_bone_size, _ = self.read_hkarray(
            section_index, current_offset
        )
        current_offset += 16 if ptr_size == 8 else 12

        # floatTrackToFloatSlotIndices (hkArray<int16>)
        float_to_slot_ptr, float_to_slot_size, _ = self.read_hkarray(
            section_index, current_offset
        )
        current_offset += 16 if ptr_size == 8 else 12

        # blendHint (int8)
        section = self.get_section(section_index)
        blend_hint = section.data[current_offset]

        # Read track to bone indices
        track_to_bone = []
        if track_to_bone_ptr:
            sid, soff = track_to_bone_ptr
            section = self.get_section(sid)
            for i in range(track_to_bone_size):
                val = int.from_bytes(
                    section.data[soff + i * 2 : soff + i * 2 + 2],
                    "little" if self.layout.little_endian else "big",
                    signed=True,
                )
                track_to_bone.append(val)

        return {
            "original_skeleton_name": original_skeleton_name,
            "animation_ptr": animation_ptr,
            "track_to_bone": track_to_bone,
            "blend_hint": blend_hint,
        }


class hkxSectionHeader:
    def __init__(self, header):
        self.header = header
        self.section_tag = ""
        self.absolute_data_start = 0
        self.local_fixups_offset = 0
        self.global_fixups_offset = 0
        self.virtual_fixups_offset = 0
        self.exports_offset = 0
        self.imports_offset = 0
        self.buffer_size = 0
        self.section_id = 0

        self.data = bytearray()
        self.local_fixups = []
        self.global_fixups = []
        self.virtual_fixups = []
        self.virtual_classes = []
        self.pointer_map = {}

    def load_header_data(self, reader):
        self.section_tag = reader.read_string(20)
        (
            self.absolute_data_start,
            self.local_fixups_offset,
            self.global_fixups_offset,
            self.virtual_fixups_offset,
            self.exports_offset,
            self.imports_offset,
            self.buffer_size,
        ) = reader.read("IIIIIII")

        if self.header.version > 9:
            reader.seek(16, 1)

    def load_data(self, reader):
        if self.buffer_size == 0:
            return

        # Read buffer
        reader.seek(self.absolute_data_start)
        self.data = bytearray(reader.read_bytes(self.local_fixups_offset))

        # Read fixups
        virtual_eof = (
            self.imports_offset
            if self.exports_offset == 0xFFFFFFFF
            else self.exports_offset
        )

        num_local = (self.global_fixups_offset - self.local_fixups_offset) // 8
        num_global = (self.virtual_fixups_offset - self.global_fixups_offset) // 12
        num_virtual = (virtual_eof - self.virtual_fixups_offset) // 12

        reader.seek(self.absolute_data_start + self.local_fixups_offset)
        for _ in range(num_local):
            self.local_fixups.append(reader.read("ii"))  # pointer, destination

        reader.seek(self.absolute_data_start + self.global_fixups_offset)
        for _ in range(num_global):
            self.global_fixups.append(
                reader.read("iii")
            )  # pointer, sectionid, destination

        reader.seek(self.absolute_data_start + self.virtual_fixups_offset)
        for _ in range(num_virtual):
            self.virtual_fixups.append(
                reader.read("iii")
            )  # dataoffset, sectionid, classnameoffset

    def link_buffer(self):
        self.pointer_map = {}
        for p, d in self.local_fixups:
            if p != -1:
                self.pointer_map[p] = (self.section_id, d)

        for p, sid, d in self.global_fixups:
            if p != -1:
                self.pointer_map[p] = (sid, d)

    def get_pointer_at(self, offset, ptr_size, endian):
        # Check if there is a fixup at this offset
        if offset in self.pointer_map:
            return self.pointer_map[offset]

        # Read raw value
        if offset + ptr_size > len(self.data):
            return None

        raw_val = int.from_bytes(
            self.data[offset : offset + ptr_size], "little" if endian else "big"
        )

        if raw_val == 0:
            return None

        # If no fixup, it might be a local offset (if raw_val is valid offset in this section)
        # But usually pointers have fixups.
        # For now, assume local offset if within bounds?
        if 0 < raw_val < len(self.data):
            return (self.section_id, raw_val)

        return None
