"""Havok importer for HKX/HKT/HKA/IGZ/PAK packfiles."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import bpy
from bpy_extras.io_utils import ImportHelper, axis_conversion
from mathutils import Matrix, Vector

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
    path: bpy.props.StringProperty()
    size: bpy.props.IntProperty()
    mode: bpy.props.StringProperty()
    is_dir: bpy.props.BoolProperty()
    depth: bpy.props.IntProperty()


def _refresh_pak_entries(self, _context):  # pragma: no cover - UI callback
    if not self.filepath.lower().endswith(".pak"):
        return

    # Some Blender versions construct RNA proxy objects that may not expose
    # Python-only helpers until the operator is fully instantiated. Be
    # defensive so the UI callback never raises while the operator is being
    # created or refreshed.
    loader = getattr(self, "_load_pak_entries", None)
    if callable(loader):
        loader()


def _on_active_pak_changed(self, _context):  # pragma: no cover - UI callback
    if not getattr(self, "pak_entries", None):
        return
    if self.pak_active_index < 0 or self.pak_active_index >= len(self.pak_entries):
        return
    item = self.pak_entries[self.pak_active_index]
    if not item.is_dir:
        self.archive_entry = item.path or item.name


def _build_pak_tree(entries: List[parsers.PakEntry]) -> List[Dict[str, object]]:
    """Flattened directory tree for UI presentation."""

    root: Dict[str, object] = {"children": {}, "depth": -1}

    for entry in entries:
        parts = [p for p in entry.name.replace("\\", "/").split("/") if p]
        if not parts:
            parts = [entry.name]

        node = root
        for depth, part in enumerate(parts):
            children: Dict[str, Dict[str, object]] = node.setdefault("children", {})  # type: ignore[assignment]
            if part not in children:
                children[part] = {
                    "name": part,
                    "path": "/".join(parts[: depth + 1]),
                    "children": {},
                    "depth": depth,
                    "is_dir": True,
                }
            node = children[part]

        node.update({
            "is_dir": False,
            "size": entry.size,
            "mode": hex(entry.mode),
        })

    ordered: List[Dict[str, object]] = []

    def walk(current: Dict[str, object]) -> None:
        for key in sorted(current.get("children", {}).keys()):
            child: Dict[str, object] = current["children"][key]  # type: ignore[index]
            ordered.append(child)
            walk(child)

    walk(root)
    return ordered


class HAVOK_UL_pak_entries(bpy.types.UIList):
    bl_idname = "HAVOK_UL_pak_entries"

    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname):  # pragma: no cover - UI
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row()
            indent_row = row.row()
            for _ in range(max(item.depth, 0)):
                indent_row.separator_spacer()
            indent_row.label(text=item.name, icon="FILE_FOLDER" if item.is_dir else "FILE_ARCHIVE")
            if not item.is_dir:
                row.label(text=f"{item.size} bytes")
                row.label(text=item.mode)
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
    pak_active_index: bpy.props.IntProperty(update=_on_active_pak_changed)
    last_pak_path: bpy.props.StringProperty(options={"HIDDEN"})
    last_pak_profile: bpy.props.StringProperty(options={"HIDDEN"})
    last_pak_platform: bpy.props.StringProperty(options={"HIDDEN"})

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

    import_meshes: bpy.props.BoolProperty(
        name="Import static meshes",
        default=True,
        description="Build Blender meshes from Havok geometry when available",
    )

    import_skeleton: bpy.props.BoolProperty(
        name="Import skeleton",
        default=True,
        description="Create an armature from the Havok skeleton definition",
    )

    def check(self, _context):  # pragma: no cover - UI callback
        if self.filepath.lower().endswith(".pak"):
            if (
                self.filepath != self.last_pak_path
                or self.pak_profile != self.last_pak_profile
                or self.pak_platform != self.last_pak_platform
            ):
                self._load_pak_entries()
                self.last_pak_path = self.filepath
                self.last_pak_profile = self.pak_profile
                self.last_pak_platform = self.pak_platform
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
        axis_mat: Matrix = axis_conversion(
            from_forward="-Y", from_up="Z", to_forward=prefs.forward_axis, to_up=prefs.up_axis
        ).to_4x4()
        armature_obj: Optional[bpy.types.Object] = None
        if self.import_skeleton and pack.skeleton:
            armature_obj = self._build_armature(context, pack, prefs.scale, axis_mat)
        if self.import_meshes and pack.meshes:
            self._build_meshes(context, pack, prefs.scale, axis_mat, armature_obj)
        if self.import_animation and pack.animations:
            self._build_animations(context, pack, armature_obj, prefs.scale, axis_mat)

        self.report({"INFO"}, f"Imported {filepath.name}")
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "import_meshes")
        layout.prop(self, "import_skeleton")
        layout.prop(self, "import_animation")
        layout.prop(self, "archive_entry")
        if self.filepath.lower().endswith(".pak"):
            # Lazy-refresh in case the file picker did not trigger `check`.
            if not self.pak_entries:
                self._load_pak_entries()
            layout.prop(self, "pak_profile")
            layout.prop(self, "pak_platform")
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
        axis_mat: Matrix,
    ) -> bpy.types.Object:
        assert pack.skeleton
        skel = pack.skeleton
        armature = bpy.data.armatures.new(skel.name)
        armature_obj = bpy.data.objects.new(skel.name, armature)

        collection = _get_or_create_collection(context, "Havok Imports")
        collection.objects.link(armature_obj)
        bpy.context.view_layer.objects.active = armature_obj

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
        previous_path = self.archive_entry

        self.pak_entries.clear()
        if not self.filepath:
            return
        try:
            entries = parsers.enumerate_pak_entries(
                Path(self.filepath), self.pak_profile, self.pak_platform
            )
        except Exception:
            return

        tree = _build_pak_tree(entries)
        for node in tree:
            item = self.pak_entries.add()
            item.name = node["name"]
            item.path = node["path"]
            item.size = node.get("size", 0)
            item.mode = node.get("mode", "")
            item.is_dir = node.get("is_dir", False)
            item.depth = node.get("depth", 0)

        preferred = next((n for n in tree if n.get("path") == previous_path and not n["is_dir"]), None)
        if preferred:
            self.archive_entry = preferred["path"]
            self.pak_active_index = tree.index(preferred)
        else:
            first_leaf = next((n for n in tree if not n["is_dir"]), None)
            if first_leaf:
                self.archive_entry = first_leaf["path"]
                self.pak_active_index = tree.index(first_leaf)
            else:
                self.archive_entry = ""
                self.pak_active_index = 0

        self.last_pak_path = self.filepath
        self.last_pak_profile = self.pak_profile
        self.last_pak_platform = self.pak_platform

    def _build_animations(
        self,
        context: bpy.types.Context,
        pack: HavokPack,
        armature_obj: Optional[bpy.types.Object],
        scale: float,
        axis_mat: Matrix,
    ) -> None:
        if armature_obj is None:
            armature_obj = (
                self._build_armature(context, pack, scale, axis_mat) if pack.skeleton else None
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

    def _build_meshes(
        self,
        context: bpy.types.Context,
        pack: HavokPack,
        scale: float,
        axis_mat: Matrix,
        armature_obj: Optional[bpy.types.Object],
    ) -> None:
        collection = _get_or_create_collection(context, "Havok Imports")
        for mesh_data in pack.meshes:
            mesh = bpy.data.meshes.new(mesh_data.name)
            transformed_verts = [
                (axis_mat @ (v * scale).to_4d()).to_3d() for v in mesh_data.vertices
            ]
            mesh.from_pydata(transformed_verts, [], mesh_data.faces)
            mesh.update()

            obj = bpy.data.objects.new(mesh_data.name, mesh)
            collection.objects.link(obj)
            if armature_obj:
                obj.parent = armature_obj


def _get_or_create_collection(context: bpy.types.Context, name: str):
    root = context.scene.collection
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    collection = bpy.data.collections.new(name)
    root.children.link(collection)
    return collection


def menu_func_import(self, _context):
    self.layout.operator(HAVOK_OT_import.bl_idname, text="Havok (.hkx/.hkt/.hka/.igz/.pak)")
