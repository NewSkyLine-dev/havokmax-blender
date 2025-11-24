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

from havok_blender.io.spline_decompressor import (
    SplineDecompressor,
    SplineTrackType,
    TransformMask,
    TransformType,
)


def test_identity_block_sampling():
    # mask declaring all identity tracks
    mask = TransformMask(0, 0, 0, 0)
    # block layout: mask + padding bytes for position/scale statics (ignored)
    data_buffer = bytes([mask.quantization_types, mask.position_types, mask.rotation_types, mask.scale_types])
    data_buffer += b"\x00" * 12  # position static placeholder
    data_buffer += b"\x00" * 12  # scale static placeholder

    decompressor = SplineDecompressor(little_endian=True)
    decompressor.decompress(data_buffer, [0], num_transform_tracks=1, num_float_tracks=0)
    tracks = decompressor.sample_all_tracks(num_frames=2, frame_duration=1.0)

    assert len(tracks) == 1
    assert len(tracks[0]) == 2
    for frame in tracks[0]:
        assert frame.position == (0.0, 0.0, 0.0)
        assert frame.rotation == (0.0, 0.0, 0.0, 1.0)
        assert frame.scale == (1.0, 1.0, 1.0)


def test_transform_mask_sub_track_types():
    mask = TransformMask(quantization_types=0b01000000, position_types=0b00110001, rotation_types=0x10, scale_types=0)
    assert mask.sub_track_type(TransformType.POS_X) == SplineTrackType.STATIC
    assert mask.sub_track_type(TransformType.POS_Y) == SplineTrackType.DYNAMIC
    assert mask.sub_track_type(TransformType.POS_Z) == SplineTrackType.IDENTITY
    assert mask.sub_track_type(TransformType.ROTATION) == SplineTrackType.DYNAMIC
    assert mask.sub_track_type(TransformType.SCALE_X) == SplineTrackType.IDENTITY
