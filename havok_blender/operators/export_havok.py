"""Simple Havok exporter placeholder.

The exporter records scene object transforms into a JSON structure placed next
to the requested HKX/HKT/HKA file name. This mirrors the original tool's
geometry/animation intent while keeping the implementation dependency-light.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import bpy
from bpy_extras.io_utils import ExportHelper


class HAVOK_OT_export(bpy.types.Operator, ExportHelper):
    """Export placeholder HKX/HKT/HKA data."""

    bl_idname = "havok.export_hkx"
    bl_label = "Export Havok (.hkx/.hkt/.hka)"
    bl_options = {"UNDO"}

    filename_ext = ".hkx"
    filter_glob: bpy.props.StringProperty(
        default="*.hkx;*.hkt;*.hka",
        options={"HIDDEN"},
    )
    apply_modifiers: bpy.props.BoolProperty(
        name="Apply Modifiers",
        description="Apply object modifiers before sampling transforms",
        default=True,
    )

    def execute(self, context: bpy.types.Context):
        filepath = Path(self.filepath)
        prefs = context.preferences.addons[__package__.split(".")[0]].preferences

        payload = {
            "scene": context.scene.name,
            "scale": prefs.scale,
            "up_axis": prefs.up_axis,
            "forward_axis": prefs.forward_axis,
            "include_animations": prefs.include_animations,
            "objects": _serialize_objects(context.selected_objects),
        }

        json_path = filepath.with_suffix(filepath.suffix + ".json")
        json_path.write_text(json.dumps(payload, indent=2))

        self.report({"INFO"}, f"Wrote Havok placeholder: {json_path.name}")
        return {"FINISHED"}


def _serialize_objects(objs: Iterable[bpy.types.Object]):
    items = []
    for obj in objs:
        item = {
            "name": obj.name,
            "type": obj.type,
            "location": tuple(obj.location),
            "rotation": tuple(obj.rotation_euler),
            "scale": tuple(obj.scale),
        }

        if obj.type == "ARMATURE":
            item["bones"] = [
                {"name": bone.name, "head": tuple(bone.head_local), "tail": tuple(bone.tail_local)}
                for bone in obj.data.bones
            ]

        items.append(item)
    return items


def menu_func_export(self, _context):
    self.layout.operator(HAVOK_OT_export.bl_idname, text="Havok (.hkx/.hkt/.hka)")
