"""Addon preferences mirroring the legacy HavokMax presets."""
from __future__ import annotations

import bpy


class HAVOK_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__.split(".")[0]

    scale: bpy.props.FloatProperty(
        name="Scale",
        default=1.0,
        min=0.0001,
        description="Multiplier applied to imported meshes and animations",
    )
    up_axis: bpy.props.EnumProperty(
        name="Up Axis",
        description="Coordinate system up-axis for conversions",
        items=[
            ("Z", "Z", "Use Blender's default Z-up system"),
            ("Y", "Y", "Convert from Y-up scenes"),
            ("X", "X", "Convert from X-up scenes"),
        ],
        default="Z",
    )
    forward_axis: bpy.props.EnumProperty(
        name="Forward Axis",
        description="Forward axis to align imported skeletons",
        items=[
            ("Y", "Y", "Positive Y"),
            ("-Y", "-Y", "Negative Y"),
            ("Z", "Z", "Positive Z"),
            ("-Z", "-Z", "Negative Z"),
        ],
        default="Y",
    )

    def draw(self, context: bpy.types.Context) -> None:  # pragma: no cover - UI
        layout = self.layout
        layout.label(text="Havok IO Presets")
        layout.prop(self, "scale")
        layout.prop(self, "up_axis")
        layout.prop(self, "forward_axis")
