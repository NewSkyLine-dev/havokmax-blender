"""Havok importer for HKX/HKT/HKA/IGZ/PAK packfiles."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import bpy
from bpy_extras.io_utils import ImportHelper, axis_conversion
from mathutils import Vector

from ..io import parsers
from ..io.parsers import HavokPack, load_from_path, SUPPORTED_EXTENSIONS


class HavokPakEntry(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    size: bpy.props.IntProperty()
    mode: bpy.props.StringProperty()


class HAVOK_UL_pak_entries(bpy.types.UIList):
    bl_idname = "HAVOK_UL_pak_entries"

    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname):  # pragma: no cover - UI
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            layout.label(text=item.name, icon="FILE_ARCHIVE")
            layout.label(text=f"{item.size} bytes")
            layout.label(text=item.mode)
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=item.name)


class HAVOK_OT_import(bpy.types.Operator, ImportHelper):
    """Import Havok HKX/HKT/HKA/IGZ/PAK data into the current scene."""

    bl_idname = "havok.import_hkx"
    bl_label = "Import Havok (.hkx/.hkt/.hka/.igz/.pak)"
    bl_options = {"UNDO"}

    filename_ext = ".hkx"
    filter_glob: bpy.props.StringProperty(
        default="*.hkx;*.hkt;*.hka;*.igz;*.pak",
        options={"HIDDEN"},
    )

    pak_entries: bpy.props.CollectionProperty(type=HavokPakEntry)
    pak_active_index: bpy.props.IntProperty()
    last_pak_path: bpy.props.StringProperty(options={"HIDDEN"})

    archive_entry: bpy.props.StringProperty(
        name="Archive entry",
        description=(
            "Optional entry name inside PAK/ZIP archives. Leave empty to auto-select the "
            "first Havok payload."
        ),
        default="",
    )

    import_animation: bpy.props.BoolProperty(
        name="Import animation",
        default=True,
        description="Generate Blender actions for each Havok animation track",
    )

    import_skeleton: bpy.props.BoolProperty(
        name="Import skeleton",
        default=True,
        description="Create an armature from the Havok skeleton definition",
    )

    def check(self, _context):  # pragma: no cover - UI callback
        if self.filepath.lower().endswith(".pak") and self.filepath != self.last_pak_path:
            self._load_pak_entries()
            self.last_pak_path = self.filepath
            return True
        return False

    def execute(self, context: bpy.types.Context):
        filepath = Path(self.filepath)
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            self.report({"ERROR"}, f"Unsupported extension: {filepath.suffix}")
            return {"CANCELLED"}

        if filepath.suffix.lower() == ".pak" and self.pak_entries:
            if 0 <= self.pak_active_index < len(self.pak_entries):
                self.archive_entry = self.pak_entries[self.pak_active_index].name

        try:
            pack = load_from_path(filepath, entry=self.archive_entry or None)
        except Exception as exc:  # pragma: no cover - Blender reports the error
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        prefs = context.preferences.addons[__package__.split(".")[0]].preferences
        armature_obj: Optional[bpy.types.Object] = None
        if self.import_skeleton and pack.skeleton:
            armature_obj = self._build_armature(
                context, pack, prefs.scale, prefs.forward_axis, prefs.up_axis
            )
        if self.import_animation and pack.animations:
            self._build_animations(
                context, pack, armature_obj, prefs.scale, prefs.forward_axis, prefs.up_axis
            )

        self.report({"INFO"}, f"Imported {filepath.name}")
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "import_skeleton")
        layout.prop(self, "import_animation")
        layout.prop(self, "archive_entry")
        if self.filepath.lower().endswith(".pak"):
            row = layout.row()
            row.template_list(
                "HAVOK_UL_pak_entries",
                "pak_entries",
                self,
                "pak_entries",
                self,
                "pak_active_index",
                rows=5,
            )

    def _build_armature(
        self,
        context: bpy.types.Context,
        pack: HavokPack,
        scale: float,
        forward: str,
        up: str,
    ) -> bpy.types.Object:
        assert pack.skeleton
        skel = pack.skeleton
        armature = bpy.data.armatures.new(skel.name)
        armature_obj = bpy.data.objects.new(skel.name, armature)

        collection = _get_or_create_collection(context, "Havok Imports")
        collection.objects.link(armature_obj)
        bpy.context.view_layer.objects.active = armature_obj

        axis_mat = axis_conversion(from_forward="-Y", from_up="Z", to_forward=forward, to_up=up).to_4x4()
        armature_obj.matrix_world = axis_mat @ armature_obj.matrix_world

        bpy.ops.object.mode_set(mode="EDIT")
        for idx, bone in enumerate(skel.bones):
            edit_bone = armature.edit_bones.new(bone.name)
            edit_bone.head = bone.translation * scale
            edit_bone.tail = bone.translation * scale + bone.rotation @ Vector((0.0, 0.1 * scale, 0.0))
            if bone.parent >= 0 and bone.parent < len(skel.bones):
                parent_edit_bone = armature.edit_bones[skel.bones[bone.parent].name]
                edit_bone.parent = parent_edit_bone
        bpy.ops.object.mode_set(mode="OBJECT")

        # Apply rest pose rotations
        for bone, pose_bone in zip(skel.bones, armature_obj.pose.bones):
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.rotation_quaternion = bone.rotation

        return armature_obj

    def _load_pak_entries(self) -> None:
        self.pak_entries.clear()
        if not self.filepath:
            return
        try:
            entries = parsers.enumerate_pak_entries(Path(self.filepath))
        except Exception:
            return

        for entry in entries:
            item = self.pak_entries.add()
            item.name = entry.name
            item.size = entry.size
            item.mode = hex(entry.mode)
        if entries:
            self.archive_entry = entries[0].name
            self.pak_active_index = 0

    def _build_animations(
        self,
        context: bpy.types.Context,
        pack: HavokPack,
        armature_obj: Optional[bpy.types.Object],
        scale: float,
        forward: str,
        up: str,
    ) -> None:
        if armature_obj is None:
            armature_obj = (
                self._build_armature(context, pack, scale, forward, up) if pack.skeleton else None
            )
        if armature_obj is None:
            return

        for animation in pack.animations:
            action = bpy.data.actions.new(animation.name)
            armature_obj.animation_data_create()
            armature_obj.animation_data.action = action

            for track_idx, track in enumerate(animation.tracks):
                if not track:
                    continue
                bone_idx = animation.track_to_bone[track_idx] if track_idx < len(animation.track_to_bone) else track_idx
                if bone_idx < 0 or bone_idx >= len(pack.skeleton.bones):
                    continue
                bone_name = pack.skeleton.bones[bone_idx].name
                pose_bone = armature_obj.pose.bones.get(bone_name)
                if not pose_bone:
                    continue

                data_path_loc = pose_bone.path_from_id("location")
                data_path_rot = pose_bone.path_from_id("rotation_quaternion")
                fcurves_loc = [action.fcurves.new(data_path_loc, index=i) for i in range(3)]
                fcurves_rot = [action.fcurves.new(data_path_rot, index=i) for i in range(4)]

                frame_count = len(track)
                frame_rate = (animation.duration / max(frame_count - 1, 1)) if animation.duration > 0 else 1.0
                for frame_idx, (trans, quat) in enumerate(track):
                    frame = frame_idx * frame_rate * context.scene.render.fps / context.scene.render.fps_base
                    for axis, curve in enumerate(fcurves_loc):
                        curve.keyframe_points.insert(frame, (trans[axis] * scale), options={'FAST'}).interpolation = 'LINEAR'
                    for axis, curve in enumerate(fcurves_rot):
                        curve.keyframe_points.insert(frame, quat[axis], options={'FAST'}).interpolation = 'LINEAR'

            # Keep last action applied
            armature_obj.animation_data.action = action


def _get_or_create_collection(context: bpy.types.Context, name: str):
    root = context.scene.collection
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    collection = bpy.data.collections.new(name)
    root.children.link(collection)
    return collection


def menu_func_import(self, _context):
    self.layout.operator(HAVOK_OT_import.bl_idname, text="Havok (.hkx/.hkt/.hka/.igz/.pak)")
