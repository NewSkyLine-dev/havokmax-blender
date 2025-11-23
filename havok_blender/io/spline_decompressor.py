import struct
import math

# Enums
STT_DYNAMIC = 0
STT_STATIC = 1
STT_IDENTITY = 2

QT_8bit = 0
QT_16bit = 1
QT_32bit = 2
QT_40bit = 3
QT_48bit = 4

ttPosX = 0
ttPosY = 1
ttPosZ = 2
ttRotation = 3
ttScaleX = 4
ttScaleY = 5
ttScaleZ = 6


class TransformMask:
    def __init__(self, data):
        # data is 4 bytes
        self.quantizationTypes = data[0]
        self.positionTypes = data[1]
        self.rotationTypes = data[2]
        self.scaleTypes = data[3]

    def get_pos_quantization_type(self):
        return self.quantizationTypes & 3

    def get_rot_quantization_type(self):
        return ((self.quantizationTypes >> 2) & 0xF) + 2

    def get_scale_quantization_type(self):
        return (self.quantizationTypes >> 6) & 3

    def get_sub_track_type(self, type_idx):
        if type_idx == ttPosX:
            if self.positionTypes & (1 << 0):
                return STT_STATIC
            if self.positionTypes & (1 << 4):
                return STT_DYNAMIC
            return STT_IDENTITY
        elif type_idx == ttPosY:
            if self.positionTypes & (1 << 1):
                return STT_STATIC
            if self.positionTypes & (1 << 5):
                return STT_DYNAMIC
            return STT_IDENTITY
        elif type_idx == ttPosZ:
            if self.positionTypes & (1 << 2):
                return STT_STATIC
            if self.positionTypes & (1 << 6):
                return STT_DYNAMIC
            return STT_IDENTITY
        elif type_idx == ttRotation:
            if self.rotationTypes & 0xF0:
                return STT_DYNAMIC
            if self.rotationTypes & 0xF:
                return STT_STATIC
            return STT_IDENTITY
        elif type_idx == ttScaleX:
            if self.scaleTypes & (1 << 0):
                return STT_STATIC
            if self.scaleTypes & (1 << 4):
                return STT_DYNAMIC
            return STT_IDENTITY
        elif type_idx == ttScaleY:
            if self.scaleTypes & (1 << 1):
                return STT_STATIC
            if self.scaleTypes & (1 << 5):
                return STT_DYNAMIC
            return STT_IDENTITY
        elif type_idx == ttScaleZ:
            if self.scaleTypes & (1 << 2):
                return STT_STATIC
            if self.scaleTypes & (1 << 6):
                return STT_DYNAMIC
            return STT_IDENTITY
        return STT_IDENTITY


def apply_padding(offset, alignment=4):
    result = offset & (alignment - 1)
    if result:
        return offset + (alignment - result)
    return offset


def read_32_quat(data, offset, endian="<"):
    # 4 bytes
    cVal = struct.unpack_from(endian + "I", data, offset)[0]

    rMask = (1 << 10) - 1
    rFrac = 1.0 / 1023.0

    fPI = 3.14159265
    fPI2 = 0.5 * fPI
    fPI4 = 0.5 * fPI2
    phiFrac = fPI2 / 511.0

    R = float((cVal >> 18) & rMask) * rFrac
    R = 1.0 - (R * R)

    phiTheta = float(cVal & 0x3FFFF)

    phi = math.floor(math.sqrt(phiTheta))
    theta = 0.0

    if phi > 0.0:
        theta = fPI4 * (phiTheta - (phi * phi)) / phi
        phi = phiFrac * phi

    magnitude = math.sqrt(max(0.0, 1.0 - R * R))
    sPhi = math.sin(phi)
    cPhi = math.cos(phi)
    sTheta = math.sin(theta)
    cTheta = math.cos(theta)

    x = sPhi * cTheta * magnitude
    y = sPhi * sTheta * magnitude
    z = cPhi * magnitude
    w = float(((cVal >> 18) & rMask)) * rFrac
    w = 1.0 - (w * w)

    if (cVal & 0x10000000) == 0x10000000:
        x = -x
    if (cVal & 0x20000000) == 0x20000000:
        y = -y
    if (cVal & 0x40000000) == 0x40000000:
        z = -z
    if (cVal & 0x80000000) == 0x80000000:
        w = -w

    return (x, y, z, w), offset + 4


def read_40_quat(data, offset, endian="<"):
    # 5 bytes
    if offset + 8 > len(data):
        chunk = data[offset:] + b"\x00" * (8 - (len(data) - offset))
        cVal0 = struct.unpack(endian + "Q", chunk)[0]
    else:
        cVal0 = struct.unpack_from(endian + "Q", data, offset)[0]

    fractal = 0.000345436

    x_raw = (cVal0) & 0xFFF
    y_raw = (cVal0 >> 12) & 0xFFF
    z_raw = (cVal0 >> 24) & 0xFFF

    # C++ reference uses 2049 subtraction (IVector4A16(tmpVal) - (1 << 11) - 1)
    x = (float(x_raw) - 2049.0) * fractal
    y = (float(y_raw) - 2049.0) * fractal
    z = (float(z_raw) - 2049.0) * fractal

    w_sq = 1.0 - (x * x + y * y + z * z)
    w = math.sqrt(max(0.0, w_sq))

    if (cVal0 >> 38) & 1:
        w = -w

    resultShift = (cVal0 >> 36) & 3
    res = [x, y, z, w]

    if resultShift == 0:
        return (res[3], res[0], res[1], res[2]), offset + 5
    elif resultShift == 1:
        return (res[0], res[3], res[1], res[2]), offset + 5
    elif resultShift == 2:
        return (res[0], res[1], res[3], res[2]), offset + 5
    else:
        return (res[0], res[1], res[2], res[3]), offset + 5


def read_48_quat(data, offset, endian="<"):
    # 6 bytes
    vals = struct.unpack_from(endian + "hhh", data, offset)
    vx, vy, vz = vals

    resultShift = ((vy >> 14) & 2) | ((vx >> 15) & 1)
    rSign = (vz >> 15) != 0

    mask = 0x7FFF
    fractal = 0.000043161

    x = (float(vx & mask) - 16383.0) * fractal
    y = (float(vy & mask) - 16383.0) * fractal
    z = (float(vz & mask) - 16383.0) * fractal

    w_sq = 1.0 - (x * x + y * y + z * z)
    w = math.sqrt(max(0.0, w_sq))

    if rSign:
        w = -w

    res = [x, y, z, w]

    if resultShift == 0:
        return (res[3], res[0], res[1], res[2]), offset + 6
    elif resultShift == 1:
        return (res[0], res[3], res[1], res[2]), offset + 6
    elif resultShift == 2:
        return (res[0], res[1], res[3], res[2]), offset + 6
    else:
        return (res[0], res[1], res[2], res[3]), offset + 6


def read_quat(q_type, data, offset, endian="<"):
    if q_type == QT_32bit:
        return read_32_quat(data, offset, endian)
    elif q_type == QT_40bit:
        return read_40_quat(data, offset, endian)
    elif q_type == QT_48bit:
        return read_48_quat(data, offset, endian)
    return (0.0, 0.0, 0.0, 1.0), offset


def find_knot_span(degree, value, c_points_size, knots):
    if value >= knots[c_points_size]:
        return c_points_size - 1

    low = degree
    high = c_points_size
    mid = (low + high) // 2

    while value < knots[mid] or value >= knots[mid + 1]:
        if value < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2

    return mid


def get_single_point(knot_span_index, degree, frame, knots, c_points):
    N = [0.0] * 6
    N[0] = 1.0

    for i in range(1, degree + 1):
        for j in range(i - 1, -1, -1):
            denom = knots[knot_span_index + i - j] - knots[knot_span_index - j]
            if denom == 0:
                A = 0
            else:
                A = (frame - knots[knot_span_index - j]) / denom

            tmp = N[j] * A
            N[j + 1] += N[j] - tmp
            N[j] = tmp

    is_vector = isinstance(c_points[0], (list, tuple))
    if is_vector:
        dim = len(c_points[0])
        ret_val = [0.0] * dim
        for i in range(degree + 1):
            weight = N[i]
            pt = c_points[knot_span_index - i]
            for k in range(dim):
                ret_val[k] += pt[k] * weight
        return tuple(ret_val)
    else:
        ret_val = 0.0
        for i in range(degree + 1):
            ret_val += c_points[knot_span_index - i] * N[i]
        return ret_val


class SplineDecompressor:
    def __init__(self):
        self.tracks = []
        self.block_duration = 0.0
        self.endian = "<"

    def decompress(
        self,
        data,
        block_offsets,
        num_transform_tracks,
        num_float_tracks,
        block_duration,
        little_endian=True,
    ):
        self.endian = "<" if little_endian else ">"
        self.block_duration = block_duration
        for block_offset in block_offsets:
            self.parse_block(data, block_offset, num_transform_tracks, num_float_tracks)

    def parse_block(self, data, offset, num_transform_tracks, num_float_tracks):
        masks = []
        curr_offset = offset
        has_dynamic = False
        for i in range(num_transform_tracks):
            mask_data = data[curr_offset : curr_offset + 4]
            mask = TransformMask(mask_data)
            masks.append(mask)

            # Check for dynamic (simple check)
            is_dyn = False
            if mask.positionTypes & 0x70:
                is_dyn = True  # Bits 4,5,6
            if mask.rotationTypes & 0xF0:
                is_dyn = True
            if mask.scaleTypes & 0x70:
                is_dyn = True

            if is_dyn:
                has_dynamic = True
            curr_offset += 4

        curr_offset += num_float_tracks

        curr_offset = apply_padding(curr_offset, 4)

        for i in range(num_transform_tracks):
            mask = masks[i]

            pos_track = self.parse_vector_track(
                data, curr_offset, mask, ttPosX, mask.get_pos_quantization_type(), 0.0
            )
            curr_offset = pos_track["next_offset"]

            rot_track = self.parse_rotation_track(data, curr_offset, mask)
            curr_offset = rot_track["next_offset"]

            scale_track = self.parse_vector_track(
                data,
                curr_offset,
                mask,
                ttScaleX,
                mask.get_scale_quantization_type(),
                1.0,
            )
            curr_offset = scale_track["next_offset"]

            self.tracks.append(
                {"pos": pos_track, "rot": rot_track, "scale": scale_track}
            )

    def parse_vector_track(self, data, offset, mask, type_start, q_type, def_val):
        is_dynamic = False
        for i in range(3):
            if mask.get_sub_track_type(type_start + i) == STT_DYNAMIC:
                is_dynamic = True
                break

        if is_dynamic:
            num_items = struct.unpack_from(self.endian + "H", data, offset)[0]
            offset += 2
            degree = struct.unpack_from(self.endian + "B", data, offset)[0]
            offset += 1

            buffer_skip = num_items + degree + 2
            knots = struct.unpack_from(f"{self.endian}{buffer_skip}B", data, offset)
            offset += buffer_skip

            offset = apply_padding(offset, 4)

            extremes = []
            tracks = [[], [], []]

            for i in range(3):
                ttype = mask.get_sub_track_type(type_start + i)
                if ttype == STT_DYNAMIC:
                    emin, emax = struct.unpack_from(self.endian + "ff", data, offset)
                    offset += 8
                    extremes.append((emin, emax))
                    tracks[i] = [0.0] * (num_items + 1)
                elif ttype == STT_STATIC:
                    val = struct.unpack_from(self.endian + "f", data, offset)[0]
                    offset += 4
                    extremes.append(None)
                    tracks[i] = [val]
                else:
                    extremes.append(None)
                    tracks[i] = [def_val]

            fractal8 = 1.0 / 255.0
            fractal16 = 1.0 / 65535.0

            for t in range(num_items + 1):
                for i in range(3):
                    ttype = mask.get_sub_track_type(type_start + i)
                    if ttype == STT_DYNAMIC:
                        if q_type == QT_8bit:
                            val = struct.unpack_from(self.endian + "B", data, offset)[0]
                            offset += 1
                            dVar = float(val) * fractal8
                        else:
                            val = struct.unpack_from(self.endian + "H", data, offset)[0]
                            offset += 2
                            dVar = float(val) * fractal16

                        emin, emax = extremes[i]
                        tracks[i][t] = emin + (emax - emin) * dVar

            offset = apply_padding(offset, 4)

            return {
                "type": "spline",
                "degree": degree,
                "knots": knots,
                "tracks": tracks,
                "next_offset": offset,
            }
        else:
            vals = []
            for i in range(3):
                ttype = mask.get_sub_track_type(type_start + i)
                if ttype == STT_STATIC:
                    val = struct.unpack_from(self.endian + "f", data, offset)[0]
                    offset += 4
                    vals.append(val)
                else:
                    vals.append(def_val)

            return {"type": "static", "value": tuple(vals), "next_offset": offset}

    def parse_rotation_track(self, data, offset, mask):
        if mask.get_sub_track_type(ttRotation) == STT_DYNAMIC:
            num_items = struct.unpack_from(self.endian + "H", data, offset)[0]
            offset += 2
            degree = struct.unpack_from(self.endian + "B", data, offset)[0]
            offset += 1

            buffer_skip = num_items + degree + 2
            knots = struct.unpack_from(f"{self.endian}{buffer_skip}B", data, offset)
            offset += buffer_skip

            q_type = mask.get_rot_quantization_type()
            if q_type == QT_48bit:
                offset = apply_padding(offset, 2)
            elif q_type == QT_32bit:
                offset = apply_padding(offset, 4)

            points = []
            for _ in range(num_items + 1):
                pt, offset = read_quat(q_type, data, offset, self.endian)
                points.append(pt)

            offset = apply_padding(offset, 4)

            return {
                "type": "spline",
                "degree": degree,
                "knots": knots,
                "points": points,
                "next_offset": offset,
            }
        else:
            if mask.get_sub_track_type(ttRotation) == STT_STATIC:
                q_type = mask.get_rot_quantization_type()
                pt, offset = read_quat(q_type, data, offset, self.endian)
                offset = apply_padding(offset, 4)
                return {"type": "static", "value": pt, "next_offset": offset}
            else:
                offset = apply_padding(offset, 4)
                return {
                    "type": "static",
                    "value": (0.0, 0.0, 0.0, 1.0),
                    "next_offset": offset,
                }

    def sample_all_tracks(self, num_frames, duration):
        sampled_tracks = []
        frame_duration = duration / (num_frames - 1) if num_frames > 1 else 0

        for track in self.tracks:
            frames = []
            for i in range(num_frames):
                time = i * frame_duration
                # Clamp time to duration?
                if time > duration:
                    time = duration

                # Scale time to knot space (0-255)
                knot_time = time
                if self.block_duration > 0:
                    knot_time = time * (255.0 / self.block_duration)

                pos = sample_spline_track(track["pos"], knot_time)
                rot = sample_spline_track(track["rot"], knot_time)
                scale = sample_spline_track(track["scale"], knot_time)

                frames.append((pos, rot, scale))
            sampled_tracks.append(frames)
        return sampled_tracks


def sample_spline_track(track_data, time):
    if track_data["type"] == "static":
        return track_data["value"]

    # Spline
    degree = track_data["degree"]
    knots = track_data["knots"]

    if "tracks" in track_data:
        # Vector track (3 components)
        c_points = list(zip(*track_data["tracks"]))  # Transpose to list of (x,y,z)
    else:
        # Quaternion track
        c_points = track_data["points"]

    c_points_size = len(c_points)

    if c_points_size <= degree:
        # Fallback for insufficient points: just return the first point or interpolate linearly?
        # If we have at least 1 point, return it.
        if c_points_size > 0:
            return c_points[0]
        return (0.0, 0.0, 0.0) if isinstance(c_points, list) else 0.0

    knot_span = find_knot_span(degree, time, c_points_size, knots)
    return get_single_point(knot_span, degree, time, knots, c_points)
