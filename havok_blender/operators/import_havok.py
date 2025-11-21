"""Havok importer for HKX/HKT/HKA/IGZ/PAK packfiles."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import bpy
from bpy_extras.io_utils import ImportHelper, axis_conversion
from mathutils import Vector

from ..io import parsers
from ..io.parsers import (
    HavokPack,
    load_from_path,
    SUPPORTED_EXTENSIONS,
    PAK_PROFILE_NAMES,
    PAK_PLATFORM_ENDIANNESS,
)


class HavokPakEntry(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    size: bpy.props.IntProperty()
    mode: bpy.props.StringProperty()


class HavokPakTreeNode(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    path: bpy.props.StringProperty()
    size: bpy.props.IntProperty()
    mode: bpy.props.StringProperty()
    level: bpy.props.IntProperty()
    is_directory: bpy.props.BoolProperty()
    expanded: bpy.props.BoolProperty(default=True)
    parent_index: bpy.props.IntProperty(default=-1)
    entry_index: bpy.props.IntProperty(default=-1)


def _load_pak_entries(operator) -> None:
    operator.pak_entries.clear()
    operator.pak_tree_nodes.clear()
    if not operator.filepath:
        return
    try:
        entries = parsers.enumerate_pak_entries(
            Path(operator.filepath), operator.pak_profile, operator.pak_platform
        )
    except Exception:
        return

    for entry in entries:
        item = operator.pak_entries.add()
        item.name = entry.name
        item.size = entry.size
        item.mode = hex(entry.mode)

    _rebuild_pak_tree(operator)

    if entries:
        _set_active_entry(operator, 0)
    else:
        operator.archive_entry = ""
        operator.pak_active_index = 0
        operator.pak_tree_active_index = -1


def _rebuild_pak_tree(operator) -> None:
    operator.pak_tree_nodes.clear()
    tree = {}

    for idx, entry in enumerate(operator.pak_entries):
        parts = [part for part in entry.name.split("/") if part] or [entry.name]
        cursor = tree
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {"children": {}, "entry_index": -1})["children"]
        leaf = parts[-1]
        leaf_node = cursor.setdefault(leaf, {"children": {}, "entry_index": -1})
        leaf_node["entry_index"] = idx

    def _flatten(branch, parent_idx, level, prefix):
        for name in sorted(branch.keys()):
            data = branch[name]
            node = operator.pak_tree_nodes.add()
            node.name = name
            node.path = f"{prefix}{name}" if prefix else name
            node.level = level
            node.parent_index = parent_idx
            has_children = bool(data["children"])
            node.is_directory = has_children
            node.entry_index = data["entry_index"] if not has_children else -1
            node.expanded = True
            if not has_children and data["entry_index"] >= 0:
                entry_data = operator.pak_entries[data["entry_index"]]
                node.size = entry_data.size
                node.mode = entry_data.mode
            else:
                node.size = 0
                node.mode = ""
            current_idx = len(operator.pak_tree_nodes) - 1
            if has_children:
                _flatten(data["children"], current_idx, level + 1, f"{node.path}/")

    _flatten(tree, -1, 0, "")


def _find_tree_node_for_entry(operator, entry_index: int) -> int:
    for idx, node in enumerate(operator.pak_tree_nodes):
        if not node.is_directory and node.entry_index == entry_index:
            return idx
    return -1


def _set_active_entry(operator, entry_index: int) -> None:
    if 0 <= entry_index < len(operator.pak_entries):
        operator.pak_active_index = entry_index
        operator.archive_entry = operator.pak_entries[entry_index].name
        operator.pak_tree_active_index = _find_tree_node_for_entry(operator, entry_index)
    else:
        operator.pak_active_index = -1
        operator.pak_tree_active_index = -1


def _on_tree_selection_changed(self, _context):
    if not self.pak_tree_nodes:
        return
    index = self.pak_tree_active_index
    if 0 <= index < len(self.pak_tree_nodes):
        node = self.pak_tree_nodes[index]
        if not node.is_directory and 0 <= node.entry_index < len(self.pak_entries):
            self.pak_active_index = node.entry_index
            self.archive_entry = self.pak_entries[node.entry_index].name



def _refresh_pak_entries(self, _context):  # pragma: no cover - UI callback
    if self.filepath.lower().endswith(".pak"):
        _load_pak_entries(self)


class HAVOK_UL_pak_tree(bpy.types.UIList):
    bl_idname = "HAVOK_UL_pak_tree"

    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname):  # pragma: no cover - UI
        row = layout.row(align=True)
        if item.level > 0:
            indent = row.row(align=True)
            indent.enabled = False
            indent.label(text="    " * item.level)
        if item.is_directory:
            icon = "TRIA_DOWN" if item.expanded else "TRIA_RIGHT"
            row.prop(item, "expanded", text="", icon=icon, emboss=False)
            row.label(text=item.name, icon="FILE_FOLDER")
            row.label(text="", icon="BLANK1")
            row.label(text="")
        else:
            row.label(text="", icon="BLANK1")
            row.label(text=item.name, icon="FILE")
            row.label(text=f"{item.size} bytes")
            row.label(text=item.mode)

    def filter_items(self, _context, data, propname):
        collection = getattr(data, propname)
        flags = [self.bitflag_filter_item] * len(collection)
        for idx, node in enumerate(collection):
            parent_idx = node.parent_index
            visible = True
            while parent_idx >= 0:
                parent = collection[parent_idx]
                if parent.is_directory and not parent.expanded:
                    visible = False
                    break
                parent_idx = parent.parent_index
            if not visible:
                flags[idx] = 0
        return flags, []


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
    pak_tree_nodes: bpy.props.CollectionProperty(type=HavokPakTreeNode)
    pak_tree_active_index: bpy.props.IntProperty(default=-1, update=_on_tree_selection_changed)
    pak_active_index: bpy.props.IntProperty()
    last_pak_path: bpy.props.StringProperty(options={"HIDDEN"})

    pak_profile: bpy.props.EnumProperty(
        name="Game version",
        description="Choose the exact PAK layout for the game you are importing",
        items=[
            (name, name.replace("_", " "), f"Use the {name} PAK layout")
            for name in PAK_PROFILE_NAMES
        ],
        default=PAK_PROFILE_NAMES[0] if PAK_PROFILE_NAMES else "",
        update=_refresh_pak_entries,
    )
    pak_platform: bpy.props.EnumProperty(
        name="Platform",
        description="Pick the platform endianness that matches the dump you are importing",
        items=[
            (key, label, label)
            for key, label in PAK_PLATFORM_ENDIANNESS.items()
        ],
        default="little",
        update=_refresh_pak_entries,
    )

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
            _load_pak_entries(self)
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

        pak_profile = self.pak_profile if filepath.suffix.lower() == ".pak" else None
        pak_platform = self.pak_platform if filepath.suffix.lower() == ".pak" else None

        try:
            pack = load_from_path(
                filepath,
                entry=self.archive_entry or None,
                pak_profile=pak_profile,
                pak_platform=pak_platform,
            )
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
            layout.prop(self, "pak_profile")
            layout.prop(self, "pak_platform")
            row = layout.row()
            row.template_list(
                "HAVOK_UL_pak_tree",
                "pak_tree",
                self,
                "pak_tree_nodes",
                self,
                "pak_tree_active_index",
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
