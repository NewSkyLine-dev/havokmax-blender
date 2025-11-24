"""Spline compressed Havok animation decompression.

This module reimplements the spline decompression routines used by
HavokLib/HavokMax in pure Python so Blender can import HKA/HKX animation
tracks without relying on the legacy logic that previously lived here.

The implementation mirrors the structure of ``hka_spline_decompressor``
from HavokLib:
* Transform masks drive whether each component is dynamic, static, or
  implicitly identity.
* Dynamic components are stored as B-spline control points with shared
  knot vectors per transform component.
* Static components are decoded once per block and reused for all frames
  inside the block.
* Rotations support the 32/40/48-bit compressed quaternion encodings
  alongside uncompressed values.
"""

from __future__ import annotations

import math
import struct
import logging
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple


_base_logger = logging.getLogger("havok_blender")
if not _base_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[HavokIO] %(levelname)s %(name)s: %(message)s")
    )
    _base_logger.addHandler(handler)
    _base_logger.setLevel(logging.INFO)

logger = _base_logger.getChild("io.spline_decompressor")


class SplineTrackType:
    DYNAMIC = 0
    STATIC = 1
    IDENTITY = 2


class QuantizationType:
    QT_8BIT = 0
    QT_16BIT = 1
    QT_32BIT = 2
    QT_40BIT = 3
    QT_48BIT = 4
    QT_24BIT = 5
    QT_16BIT_QUAT = 6
    QT_UNCOMPRESSED = 7


class TransformType:
    POS_X = 0
    POS_Y = 1
    POS_Z = 2
    ROTATION = 3
    SCALE_X = 4
    SCALE_Y = 5
    SCALE_Z = 6


@dataclass(frozen=True)
class TransformMask:
    quantization_types: int
    position_types: int
    rotation_types: int
    scale_types: int

    @classmethod
    def from_bytes(cls, payload: bytes) -> "TransformMask":
        if len(payload) != 4:
            raise ValueError("TransformMask requires exactly 4 bytes")
        return cls(payload[0], payload[1], payload[2], payload[3])

    def _flag(self, flags: int, static_bit: int, dynamic_bit: int) -> int:
        if flags & (1 << static_bit):
            return SplineTrackType.STATIC
        if flags & (1 << dynamic_bit):
            return SplineTrackType.DYNAMIC
        return SplineTrackType.IDENTITY

    def sub_track_type(self, transform: int) -> int:
        if transform == TransformType.POS_X:
            return self._flag(self.position_types, 0, 4)
        if transform == TransformType.POS_Y:
            return self._flag(self.position_types, 1, 5)
        if transform == TransformType.POS_Z:
            return self._flag(self.position_types, 2, 6)
        if transform == TransformType.ROTATION:
            if self.rotation_types & 0xF0:
                return SplineTrackType.DYNAMIC
            if self.rotation_types & 0x0F:
                return SplineTrackType.STATIC
            return SplineTrackType.IDENTITY
        if transform == TransformType.SCALE_X:
            return self._flag(self.scale_types, 0, 4)
        if transform == TransformType.SCALE_Y:
            return self._flag(self.scale_types, 1, 5)
        if transform == TransformType.SCALE_Z:
            return self._flag(self.scale_types, 2, 6)
        return SplineTrackType.IDENTITY

    def pos_quantization(self) -> int:
        return self.quantization_types & 0x3

    def rot_quantization(self) -> int:
        return ((self.quantization_types >> 2) & 0xF) + 2

    def scale_quantization(self) -> int:
        return (self.quantization_types >> 6) & 0x3


def _apply_padding(offset: int, alignment: int = 4) -> int:
    remainder = offset & (alignment - 1)
    if remainder:
        return offset + (alignment - remainder)
    return offset


def _read_f32(data: bytes, offset: int, little_endian: bool) -> float:
    fmt = "<f" if little_endian else ">f"
    return struct.unpack_from(fmt, data, offset)[0]


def _read_u16(data: bytes, offset: int, little_endian: bool) -> int:
    fmt = "<H" if little_endian else ">H"
    return struct.unpack_from(fmt, data, offset)[0]


def _read_u8(data: bytes, offset: int) -> int:
    return data[offset]


def _read32_quat(
    data: bytes, offset: int, little_endian: bool
) -> Tuple[float, float, float, float]:
    fmt = "<I" if little_endian else ">I"
    c_val = struct.unpack_from(fmt, data, offset)[0]
    r_mask = (1 << 10) - 1
    r_frac = 1.0 / 1023.0
    f_pi = math.pi
    f_pi2 = 0.5 * f_pi
    f_pi4 = 0.5 * f_pi2
    phi_frac = f_pi2 / 511.0

    r = float((c_val >> 18) & r_mask) * r_frac
    r = 1.0 - (r * r)

    phi_theta = float(c_val & 0x3FFFF)
    phi = math.floor(math.sqrt(phi_theta))
    theta = 0.0
    if phi > 0.0:
        theta = f_pi4 * (phi_theta - (phi * phi)) / phi
        phi = phi_frac * phi

    magnitude = math.sqrt(max(0.0, 1.0 - r * r))
    s_phi = math.sin(phi)
    c_phi = math.cos(phi)
    s_theta = math.sin(theta)
    c_theta = math.cos(theta)

    x = s_phi * c_theta * magnitude
    y = s_phi * s_theta * magnitude
    z = c_phi * magnitude
    w = float((c_val >> 18) & r_mask) * r_frac
    w = 1.0 - (w * w)

    if c_val & 0x10000000:
        x = -x
    if c_val & 0x20000000:
        y = -y
    if c_val & 0x40000000:
        z = -z
    if c_val & 0x80000000:
        w = -w

    return (x, y, z, w)


def _read40_quat(
    data: bytes, offset: int, little_endian: bool
) -> Tuple[float, float, float, float]:
    fmt = "<Q" if little_endian else ">Q"
    remaining = len(data) - offset
    if remaining >= 8:
        raw = struct.unpack_from(fmt, data, offset)[0]
    else:
        if remaining <= 0:
            chunk = b"\0" * 8
        else:
            chunk = data[offset : offset + remaining] + b"\0" * (8 - remaining)
        raw = struct.unpack(fmt, chunk)[0]

    fractal = 0.000345436
    x_raw = raw & 0xFFF
    y_raw = (raw >> 12) & 0xFFF
    z_raw = (raw >> 24) & 0xFFF

    x = (float(x_raw) - 2049.0) * fractal
    y = (float(y_raw) - 2049.0) * fractal
    z = (float(z_raw) - 2049.0) * fractal

    w_sq = 1.0 - (x * x + y * y + z * z)
    w = math.sqrt(max(0.0, w_sq))

    if (raw >> 38) & 1:
        w = -w

    shift = (raw >> 36) & 3
    values = [x, y, z, w]
    if shift == 0:
        return (values[3], values[0], values[1], values[2])
    if shift == 1:
        return (values[0], values[3], values[1], values[2])
    if shift == 2:
        return (values[0], values[1], values[3], values[2])
    return (values[0], values[1], values[2], values[3])


def _read48_quat(
    data: bytes, offset: int, little_endian: bool
) -> Tuple[float, float, float, float]:
    fmt = "<hhh" if little_endian else ">hhh"
    vx, vy, vz = struct.unpack_from(fmt, data, offset)
    result_shift = ((vy >> 14) & 2) | ((vx >> 15) & 1)
    r_sign = (vz >> 15) != 0
    mask = 0x7FFF
    fractal = 0.000043161
    x = (float(vx & mask) - 16383.0) * fractal
    y = (float(vy & mask) - 16383.0) * fractal
    z = (float(vz & mask) - 16383.0) * fractal
    w_sq = 1.0 - (x * x + y * y + z * z)
    w = math.sqrt(max(0.0, w_sq))
    if r_sign:
        w = -w
    values = [x, y, z, w]
    if result_shift == 0:
        return (values[3], values[0], values[1], values[2])
    if result_shift == 1:
        return (values[0], values[3], values[1], values[2])
    if result_shift == 2:
        return (values[0], values[1], values[3], values[2])
    return (values[0], values[1], values[2], values[3])


def read_quaternion(
    data: bytes, offset: int, quantization: int, little_endian: bool
) -> Tuple[Tuple[float, float, float, float], int]:
    if quantization == QuantizationType.QT_32BIT:
        return _read32_quat(data, offset, little_endian), offset + 4
    if quantization == QuantizationType.QT_40BIT:
        return _read40_quat(data, offset, little_endian), offset + 5
    if quantization in (QuantizationType.QT_48BIT, QuantizationType.QT_16BIT_QUAT):
        return _read48_quat(data, offset, little_endian), offset + 6
    if quantization == QuantizationType.QT_UNCOMPRESSED:
        fmt = "<ffff" if little_endian else ">ffff"
        quat = struct.unpack_from(fmt, data, offset)
        return quat, offset + 16
    return (0.0, 0.0, 0.0, 1.0), offset


@dataclass
class SplineDynamicVectorTrack:
    control_points: List[List[float]]
    knots: List[int]
    degree: int

    def value_at(self, local_frame: float) -> Tuple[float, float, float]:
        return (
            self._value_for_axis(0, local_frame),
            self._value_for_axis(1, local_frame),
            self._value_for_axis(2, local_frame),
        )

    def _value_for_axis(self, axis: int, local_frame: float) -> float:
        points = self.control_points[axis]
        if len(points) == 1:
            return points[0]
        knot_span = _find_knot_span(self.degree, local_frame, len(points), self.knots)
        return _get_single_point(knot_span, self.degree, local_frame, self.knots, points)  # type: ignore[return-value]


@dataclass
class SplineDynamicQuatTrack:
    control_points: List[Tuple[float, float, float, float]]
    knots: List[int]
    degree: int

    def value_at(self, local_frame: float) -> Tuple[float, float, float, float]:
        knot_span = _find_knot_span(
            self.degree, local_frame, len(self.control_points), self.knots
        )
        return _get_single_point(knot_span, self.degree, local_frame, self.knots, self.control_points)  # type: ignore[return-value]


@dataclass
class TransformTrack:
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]
    scale: Tuple[float, float, float]

    def __iter__(self):
        yield self.position
        yield self.rotation
        yield self.scale


@dataclass
class TransformSplineBlock:
    masks: List[TransformMask]
    pos_tracks: List[SplineDynamicVectorTrack | None]
    rot_tracks: List[SplineDynamicQuatTrack | None]
    pos_static: List[Tuple[float, float, float]]
    rot_static: List[Tuple[float, float, float, float]]
    scale_tracks: List[SplineDynamicVectorTrack | None]
    scale_static: List[Tuple[float, float, float]]

    def sample(self, track_index: int, frame: float) -> TransformTrack:
        mask = self.masks[track_index]
        position = self._sample_vector(
            mask,
            track_index,
            frame,
            TransformType.POS_X,
            self.pos_tracks,
            self.pos_static,
            default=(0.0, 0.0, 0.0),
        )
        rotation = self._sample_quaternion(mask, track_index, frame)
        scale = self._sample_vector(
            mask,
            track_index,
            frame,
            TransformType.SCALE_X,
            self.scale_tracks,
            self.scale_static,
            default=(1.0, 1.0, 1.0),
        )
        return TransformTrack(position, rotation, scale)

    def _sample_vector(
        self,
        mask: TransformMask,
        track_index: int,
        frame: float,
        base_type: int,
        dynamic_tracks: List[SplineDynamicVectorTrack | None],
        static_tracks: List[Tuple[float, float, float]],
        default: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        track_type = mask.sub_track_type(base_type)
        if track_type == SplineTrackType.DYNAMIC:
            track = dynamic_tracks[track_index]
            if track is None:
                return default
            return track.value_at(frame)
        if track_type == SplineTrackType.STATIC:
            return static_tracks[track_index]
        return default

    def _sample_quaternion(
        self, mask: TransformMask, track_index: int, frame: float
    ) -> Tuple[float, float, float, float]:
        track_type = mask.sub_track_type(TransformType.ROTATION)
        if track_type == SplineTrackType.DYNAMIC:
            track = self.rot_tracks[track_index]
            if track is None:
                return (0.0, 0.0, 0.0, 1.0)
            return track.value_at(frame)
        if track_type == SplineTrackType.STATIC:
            return self.rot_static[track_index]
        return (0.0, 0.0, 0.0, 1.0)


def _find_knot_span(
    degree: int, value: float, num_points: int, knots: Sequence[int]
) -> int:
    if value >= knots[num_points]:
        return num_points - 1
    low = degree
    high = num_points
    mid = (low + high) // 2
    while value < knots[mid] or value >= knots[mid + 1]:
        if value < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def _get_single_point(
    knot_span: int,
    degree: int,
    frame: float,
    knots: Sequence[int],
    control_points: Sequence[Tuple[float, ...] | float],
) -> Tuple[float, ...] | float:
    n_vals = [0.0] * (degree + 1)
    n_vals[0] = 1.0
    for i in range(1, degree + 1):
        for j in range(i - 1, -1, -1):
            denom = knots[knot_span + i - j] - knots[knot_span - j]
            a = 0.0 if denom == 0 else (frame - knots[knot_span - j]) / denom
            tmp = n_vals[j] * a
            n_vals[j + 1] += n_vals[j] - tmp
            n_vals[j] = tmp

    accum = None
    for i in range(degree + 1):
        weight = n_vals[i]
        point = control_points[knot_span - i]
        if accum is None:
            if isinstance(point, tuple):
                accum = tuple(component * weight for component in point)
            else:
                accum = point * weight
        else:
            if isinstance(point, tuple):
                accum = tuple(accum[idx] + component * weight for idx, component in enumerate(point))  # type: ignore[index]
            else:
                accum = accum + point * weight  # type: ignore[assignment]
    assert accum is not None
    return accum


class SplineDecompressor:
    def __init__(self, little_endian: bool = True):
        self.blocks: List[TransformSplineBlock] = []
        self.little_endian = little_endian

    def decompress(
        self,
        data_buffer: bytes,
        block_offsets: Sequence[int],
        num_transform_tracks: int,
        num_float_tracks: int,
    ) -> None:
        logger.info(
            "SplineDecompressor: decompressing %d blocks (tracks=%d floats=%d)",
            len(block_offsets),
            num_transform_tracks,
            num_float_tracks,
        )
        self.blocks = []
        for idx, offset in enumerate(block_offsets):
            if offset >= len(data_buffer):
                logger.warning(
                    "SplineDecompressor: block %d offset %d beyond buffer size %d",
                    idx,
                    offset,
                    len(data_buffer),
                )
                continue
            block_start = time.perf_counter()
            block = self._parse_block(
                data_buffer, offset, num_transform_tracks, num_float_tracks
            )
            self.blocks.append(block)
            logger.debug(
                "SplineDecompressor: parsed block %d/%d at offset %d in %.3fs",
                idx + 1,
                len(block_offsets),
                offset,
                time.perf_counter() - block_start,
            )

    def _parse_block(
        self,
        data: bytes,
        offset: int,
        num_transform_tracks: int,
        num_float_tracks: int,
    ) -> TransformSplineBlock:
        masks: List[TransformMask] = []
        pos_tracks: List[SplineDynamicVectorTrack | None] = []
        pos_static: List[Tuple[float, float, float]] = []
        rot_tracks: List[SplineDynamicQuatTrack | None] = []
        rot_static: List[Tuple[float, float, float, float]] = []
        scale_tracks: List[SplineDynamicVectorTrack | None] = []
        scale_static: List[Tuple[float, float, float]] = []

        cursor = offset
        for _ in range(num_transform_tracks):
            masks.append(TransformMask.from_bytes(data[cursor : cursor + 4]))
            cursor += 4
        cursor += num_float_tracks
        cursor = _apply_padding(cursor)

        for mask in masks:
            pos_track, pos_static_value, cursor = self._parse_vector_track(
                data,
                cursor,
                mask,
                mask.pos_quantization(),
                TransformType.POS_X,
                default_value=0.0,
            )
            pos_tracks.append(pos_track)
            pos_static.append(pos_static_value)

            rot_track, rot_static_value, cursor = self._parse_rotation_track(
                data, cursor, mask
            )
            rot_tracks.append(rot_track)
            rot_static.append(rot_static_value)

            scale_track, scale_static_value, cursor = self._parse_vector_track(
                data,
                cursor,
                mask,
                mask.scale_quantization(),
                TransformType.SCALE_X,
                default_value=1.0,
            )
            scale_tracks.append(scale_track)
            scale_static.append(scale_static_value)

            cursor = _apply_padding(cursor)

        return TransformSplineBlock(
            masks=masks,
            pos_tracks=pos_tracks,
            rot_tracks=rot_tracks,
            pos_static=pos_static,
            rot_static=rot_static,
            scale_tracks=scale_tracks,
            scale_static=scale_static,
        )

    def _parse_vector_track(
        self,
        data: bytes,
        offset: int,
        mask: TransformMask,
        quantization_type: int,
        base_type: int,
        default_value: float,
    ) -> Tuple[SplineDynamicVectorTrack | None, Tuple[float, float, float], int]:
        if any(
            mask.sub_track_type(base_type + axis) == SplineTrackType.DYNAMIC
            for axis in range(3)
        ):
            try:
                num_items = _read_u16(data, offset, self.little_endian)
                offset += 2
                degree = _read_u8(data, offset)
                offset += 1
                knot_count = num_items + degree + 2
                knots = list(data[offset : offset + knot_count])
                if len(knots) != knot_count:
                    raise IndexError("vector knot buffer truncated")
                offset += knot_count
                offset = _apply_padding(offset)

                extremes: List[Tuple[float, float]] = []
                control_points = [[0.0] * (num_items + 1) for _ in range(3)]
                static_defaults: List[float] = []

                for axis in range(3):
                    comp_type = mask.sub_track_type(base_type + axis)
                    if comp_type == SplineTrackType.DYNAMIC:
                        min_val = _read_f32(data, offset, self.little_endian)
                        max_val = _read_f32(data, offset + 4, self.little_endian)
                        extremes.append((min_val, max_val))
                        offset += 8
                    elif comp_type == SplineTrackType.STATIC:
                        static_defaults.append(
                            _read_f32(data, offset, self.little_endian)
                        )
                        extremes.append((0.0, 0.0))
                        offset += 4
                    else:
                        static_defaults.append(default_value)
                        extremes.append((0.0, 0.0))

                def unpack_point(axis_index: int, idx: int, value: int) -> None:
                    min_val, max_val = extremes[axis_index]
                    if max_val == min_val:
                        control_points[axis_index][idx] = min_val
                    else:
                        fraction = value / (
                            255.0
                            if quantization_type == QuantizationType.QT_8BIT
                            else 65535.0
                        )
                        control_points[axis_index][idx] = (
                            min_val + (max_val - min_val) * fraction
                        )

                for idx in range(num_items + 1):
                    for axis in range(3):
                        comp_type = mask.sub_track_type(base_type + axis)
                        if comp_type != SplineTrackType.DYNAMIC:
                            continue
                        if quantization_type == QuantizationType.QT_8BIT:
                            raw = _read_u8(data, offset)
                            offset += 1
                            unpack_point(axis, idx, raw)
                        else:
                            raw = _read_u16(data, offset, self.little_endian)
                            offset += 2
                            unpack_point(axis, idx, raw)
                    if quantization_type == QuantizationType.QT_16BIT:
                        offset = _apply_padding(offset, alignment=2)

                offset = _apply_padding(offset)
                static_value = tuple(
                    (
                        static_defaults[axis]
                        if mask.sub_track_type(base_type + axis)
                        != SplineTrackType.DYNAMIC
                        else control_points[axis][0]
                    )
                    for axis in range(3)
                )
                track = SplineDynamicVectorTrack(control_points, knots, degree)
                return track, static_value, offset
            except (IndexError, struct.error):
                return None, (default_value, default_value, default_value), len(data)

        static_values = []
        for axis in range(3):
            comp_type = mask.sub_track_type(base_type + axis)
            if comp_type == SplineTrackType.STATIC:
                try:
                    static_values.append(
                        _read_f32(data, offset + axis * 4, self.little_endian)
                    )
                except (IndexError, struct.error):
                    static_values.append(default_value)
            else:
                static_values.append(default_value)
        return None, tuple(static_values), offset + 12

    def _parse_rotation_track(
        self, data: bytes, offset: int, mask: TransformMask
    ) -> Tuple[SplineDynamicQuatTrack | None, Tuple[float, float, float, float], int]:
        track_type = mask.sub_track_type(TransformType.ROTATION)
        quantization = mask.rot_quantization()
        if track_type == SplineTrackType.DYNAMIC:
            try:
                num_items = _read_u16(data, offset, self.little_endian)
                offset += 2
                degree = _read_u8(data, offset)
                offset += 1
                knot_count = num_items + degree + 2
                knots = list(data[offset : offset + knot_count])
                if len(knots) != knot_count:
                    raise IndexError("rotation knot buffer truncated")
                offset += knot_count
                if quantization in (
                    QuantizationType.QT_48BIT,
                    QuantizationType.QT_16BIT_QUAT,
                ):
                    offset = _apply_padding(offset, alignment=2)
                elif quantization in (
                    QuantizationType.QT_32BIT,
                    QuantizationType.QT_UNCOMPRESSED,
                ):
                    offset = _apply_padding(offset)

                control_points: List[Tuple[float, float, float, float]] = []
                for _ in range(num_items + 1):
                    quat, offset = read_quaternion(
                        data, offset, quantization, self.little_endian
                    )
                    control_points.append(quat)
                track = SplineDynamicQuatTrack(control_points, knots, degree)
                return track, control_points[0], offset
            except (IndexError, struct.error, ValueError):
                return None, (0.0, 0.0, 0.0, 1.0), len(data)

        if track_type == SplineTrackType.STATIC:
            quat, offset = read_quaternion(
                data, offset, quantization, self.little_endian
            )
            return None, quat, offset

        return None, (0.0, 0.0, 0.0, 1.0), offset

    def sample_all_tracks(
        self, num_frames: int, frame_duration: float
    ) -> List[List[TransformTrack]]:
        if not self.blocks:
            return []
        track_count = len(self.blocks[0].masks)
        result: List[List[TransformTrack]] = [[] for _ in range(track_count)]
        if num_frames == 0:
            return result

        total_duration = frame_duration * num_frames
        block_duration = total_duration / max(len(self.blocks), 1)

        logger.info(
            "SplineDecompressor: sampling %d tracks across %d frames (frame_duration=%.5f block_duration=%.5f)",
            track_count,
            num_frames,
            frame_duration,
            block_duration,
        )

        frame_log_interval = max(1, num_frames // 10 or 1)

        for frame_idx in range(num_frames):
            time = frame_idx * frame_duration
            block_index = min(
                int(time // max(block_duration, frame_duration)), len(self.blocks) - 1
            )
            local_time = time - block_index * block_duration
            normalized_frame = (
                0.0 if frame_duration == 0 else local_time / frame_duration
            )
            block = self.blocks[block_index]
            for track_id in range(track_count):
                result[track_id].append(block.sample(track_id, normalized_frame))
            if frame_idx % frame_log_interval == 0 or frame_idx == num_frames - 1:
                logger.debug(
                    "SplineDecompressor: sampled frame %d/%d (block %d)",
                    frame_idx + 1,
                    num_frames,
                    block_index,
                )
        return result
