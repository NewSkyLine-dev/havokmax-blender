from __future__ import annotations

"""Havok importer for HKX/HKT/HKA/IGZ/PAK packfiles."""

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import bpy
import mathutils
from bpy_extras.io_utils import ImportHelper, axis_conversion
from mathutils import Matrix, Vector

from ..io import parsers
from ..io.igz_port import constants as igz_constants
from ..io.parsers import (
    HavokPack,
    load_from_path,
    load_igz_bytes,
    SUPPORTED_EXTENSIONS,
    PAK_PROFILE_NAMES,
    PAK_PLATFORM_ENDIANNESS,
)

_base_logger = logging.getLogger("havok_blender")
if not _base_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[HavokIO] %(levelname)s %(name)s: %(message)s")
    )
    _base_logger.addHandler(handler)
    _base_logger.setLevel(logging.INFO)

logger = _base_logger.getChild("operators.import_havok")


class HavokPakEntry(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    path: bpy.props.StringProperty()
    size: bpy.props.IntProperty()
    mode: bpy.props.StringProperty()
    is_dir: bpy.props.BoolProperty()
    depth: bpy.props.IntProperty()


class HavokAnimationEntry(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    index: bpy.props.IntProperty()
    duration: bpy.props.FloatProperty()
    num_tracks: bpy.props.IntProperty()


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
        refresher = getattr(self, "_refresh_animation_metadata", None)
        if callable(refresher):
            refresher()


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

        node.update(
            {
                "is_dir": False,
                "size": entry.size,
                "mode": hex(entry.mode),
            }
        )

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

    def draw_item(
        self, _context, layout, _data, item, _icon, _active_data, _active_propname
    ):  # pragma: no cover - UI
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row()
            indent_row = row.row()
            for _ in range(max(item.depth, 0)):
                indent_row.separator_spacer()
            indent_row.label(
                text=item.name, icon="FILE_FOLDER" if item.is_dir else "FILE_ARCHIVE"
            )
            if not item.is_dir:
                row.label(text=f"{item.size} bytes")
                row.label(text=item.mode)
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text=item.name)


class HAVOK_UL_animation_entries(bpy.types.UIList):
    bl_idname = "HAVOK_UL_animation_entries"

    def draw_item(
        self, _context, layout, _data, item, _icon, _active_data, _active_propname
    ):  # pragma: no cover - UI
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row()
            row.label(text=item.name, icon="ACTION")
            row.label(text=f"{item.duration:.2f}s")
            row.label(text=f"{item.num_tracks} tracks")
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
    animation_count: bpy.props.IntProperty(options={"HIDDEN"}, default=0)
    animation_names: bpy.props.StringProperty(options={"HIDDEN"}, default="")
    animation_metadata_key: bpy.props.StringProperty(options={"HIDDEN"}, default="")

    animation_entries: bpy.props.CollectionProperty(type=HavokAnimationEntry)
    animation_active_index: bpy.props.IntProperty()

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
        items=[(key, label, label) for key, label in PAK_PLATFORM_ENDIANNESS.items()],
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

    animation_index: bpy.props.IntProperty(
        name="Animation index",
        description=(
            "Choose which animation to import when the source contains multiple motions. "
            "Use -1 to import every animation."
        ),
        default=-1,
        min=-1,
    )

    igz_build_meshes: bpy.props.BoolProperty(
        name="Build meshes",
        description="Generate meshes when importing IGZ payloads",
        default=igz_constants.dBuildMeshes,
    )
    igz_build_bones: bpy.props.BoolProperty(
        name="Build bones",
        description="Create an armature from IGZ skeleton data",
        default=igz_constants.dBuildBones,
    )
    igz_build_faces: bpy.props.BoolProperty(
        name="Build faces",
        description="Include index buffers when creating IGZ meshes",
        default=igz_constants.dBuildFaces,
    )
    igz_allow_wii: bpy.props.BoolProperty(
        name="Allow Wii models",
        description="Permit Wii IGZ assets that may be unstable",
        default=igz_constants.dAllowWii,
    )
    igz_model_threshold: bpy.props.IntProperty(
        name="Model threshold",
        description="Limit the number of IGZ models imported before prompting",
        default=igz_constants.dModelThreshold,
        min=1,
    )
    igz_first_object_offset: bpy.props.IntProperty(
        name="First object offset",
        description=(
            "Override the offset of the first IGObject. Use -1 to import all "
            "objects without skipping."
        ),
        default=igz_constants.dFirstObjectOffset,
        min=-1,
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

    animation_scale: bpy.props.FloatProperty(
        name="Animation Scale",
        description="Additional scaling factor for animation translation tracks",
        default=1.0,
        min=0.0001,
    )

    anim_flip_x: bpy.props.BoolProperty(name="Flip X Translation")
    anim_flip_y: bpy.props.BoolProperty(name="Flip Y Translation")
    anim_flip_z: bpy.props.BoolProperty(name="Flip Z Translation")
    anim_flip_quat_x: bpy.props.BoolProperty(name="Flip Quat X")
    anim_flip_quat_y: bpy.props.BoolProperty(name="Flip Quat Y")
    anim_flip_quat_z: bpy.props.BoolProperty(name="Flip Quat Z")
    anim_flip_quat_w: bpy.props.BoolProperty(name="Flip Quat W")
    anim_swap_yz: bpy.props.BoolProperty(name="Swap Y/Z Axes")

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context: bpy.types.Context):
        filepath = Path(self.filepath)
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            self.report({"ERROR"}, f"Unsupported extension: {filepath.suffix}")
            return {"CANCELLED"}

        import_start = time.perf_counter()
        logger.info(
            "Import requested path=%s entry=%s mode=%s",
            filepath,
            self.archive_entry,
            filepath.suffix.lower(),
        )

        if filepath.suffix.lower() == ".pak" and self.pak_entries:
            if 0 <= self.pak_active_index < len(self.pak_entries):
                active_item = self.pak_entries[self.pak_active_index]
                self.archive_entry = active_item.path or active_item.name

        pak_profile = self.pak_profile if filepath.suffix.lower() == ".pak" else None
        pak_platform = self.pak_platform if filepath.suffix.lower() == ".pak" else None

        target_ext = Path(self.archive_entry or filepath.name).suffix.lower()
        if target_ext == ".igz":
            self._apply_igz_settings()
            try:
                logger.info(
                    "Loading IGZ bytes from %s entry=%s", filepath, self.archive_entry
                )
                igz_bytes = load_igz_bytes(
                    filepath,
                    entry=self.archive_entry or None,
                    pak_profile=pak_profile,
                    pak_platform=pak_platform,
                )
                logger.info("IGZ payload size=%d", len(igz_bytes))
                self._import_igz_blob(igz_bytes)
            except Exception as exc:  # pragma: no cover - Blender reports the error
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}

            self.report({"INFO"}, f"Imported {filepath.name}")
            logger.info(
                "IGZ import finished in %.2fs", time.perf_counter() - import_start
            )
            return {"FINISHED"}

        try:
            logger.info(
                "Loading Havok packfile entry=%s profile=%s platform=%s",
                self.archive_entry,
                pak_profile,
                pak_platform,
            )
            pack_start = time.perf_counter()
            pack = load_from_path(
                filepath,
                entry=self.archive_entry or None,
                pak_profile=pak_profile,
                pak_platform=pak_platform,
            )
            logger.info(
                "Pack loaded in %.2fs (skeleton=%s animations=%d meshes=%d)",
                time.perf_counter() - pack_start,
                bool(pack.skeleton),
                len(pack.animations),
                len(pack.meshes),
            )
        except Exception as exc:  # pragma: no cover - Blender reports the error
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        prefs = context.preferences.addons[__package__.split(".")[0]].preferences
        axis_mat: Matrix = axis_conversion(
            from_forward="-Y",
            from_up="Z",
            to_forward=prefs.forward_axis,
            to_up=prefs.up_axis,
        ).to_4x4()

        armature_obj = (
            context.active_object
            if context.active_object and context.active_object.type == "ARMATURE"
            else None
        )

        if armature_obj is None and self.import_skeleton and pack.skeleton:
            logger.info("Building new armature from skeleton '%s'", pack.skeleton.name)
            armature_obj = self._build_armature(context, pack, prefs.scale, axis_mat)

        if self.import_meshes and pack.meshes:
            logger.info("Building %d mesh(es)", len(pack.meshes))
            self._build_meshes(context, pack, prefs.scale, axis_mat, armature_obj)

        selected_animations = self._resolve_animations(pack)
        logger.info("Resolved %d animation(s) for import", len(selected_animations))
        if self._should_import_animation(target_ext, selected_animations, armature_obj):
            logger.info("Starting animation build")
            self._build_animations(
                context,
                pack,
                selected_animations,
                armature_obj,
                prefs.scale * self.animation_scale,
                axis_mat,
            )

        self.report({"INFO"}, f"Imported {filepath.name}")
        logger.info("Import finished in %.2fs", time.perf_counter() - import_start)
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "import_meshes")
        layout.prop(self, "import_skeleton")
        target_ext = Path(self.archive_entry or self.filepath).suffix.lower()
        self._refresh_animation_metadata()
        layout.prop(self, "archive_entry")
        if target_ext in {".hka", ".hkx"}:
            layout.separator()
            layout.label(text="Animation options:")
            layout.prop(self, "animation_scale")

            row = layout.row()
            row.prop(self, "anim_swap_yz")

            col = layout.column(align=True)
            col.label(text="Flip Translation:")
            row = col.row(align=True)
            row.prop(self, "anim_flip_x", text="X")
            row.prop(self, "anim_flip_y", text="Y")
            row.prop(self, "anim_flip_z", text="Z")

            col = layout.column(align=True)
            col.label(text="Flip Quaternion:")
            row = col.row(align=True)
            row.prop(self, "anim_flip_quat_x", text="X")
            row.prop(self, "anim_flip_quat_y", text="Y")
            row.prop(self, "anim_flip_quat_z", text="Z")
            row.prop(self, "anim_flip_quat_w", text="W")

            if self.animation_count:
                layout.label(text=f"Available animations: {self.animation_count}")
                layout.template_list(
                    "HAVOK_UL_animation_entries",
                    "animation_entries",
                    self,
                    "animation_entries",
                    self,
                    "animation_active_index",
                    rows=5,
                )
            else:
                layout.prop(self, "animation_index")
        if target_ext == ".igz":
            layout.use_property_split = True
            layout.separator()
            layout.label(text="IGZ options:")
            layout.prop(self, "igz_build_meshes")
            layout.prop(self, "igz_build_faces")
            layout.prop(self, "igz_build_bones")
            layout.prop(self, "igz_allow_wii")
            layout.prop(self, "igz_model_threshold")
            layout.prop(self, "igz_first_object_offset")
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

        # Calculate global transforms for all bones
        # Havok stores bones in local (parent-relative) space, so we must accumulate them
        global_transforms = []
        for idx, bone in enumerate(skel.bones):
            local_t = Matrix.Translation(bone.translation)
            local_r = bone.rotation.to_matrix().to_4x4()
            local_s = (
                Matrix.Scale(bone.scale[0], 4, (1, 0, 0))
                @ Matrix.Scale(bone.scale[1], 4, (0, 1, 0))
                @ Matrix.Scale(bone.scale[2], 4, (0, 0, 1))
            )
            local_mat = local_t @ local_r @ local_s

            if bone.parent >= 0 and bone.parent < len(global_transforms):
                parent_mat = global_transforms[bone.parent]
                global_mat = parent_mat @ local_mat
            else:
                global_mat = local_mat

            global_transforms.append(global_mat)

        bpy.ops.object.mode_set(mode="EDIT")
        for idx, bone in enumerate(skel.bones):
            edit_bone = armature.edit_bones.new(bone.name)
            global_mat = global_transforms[idx]

            head = global_mat.to_translation() * scale
            rot = global_mat.to_quaternion()

            edit_bone.head = head
            # Align bone Y-axis with the rotation
            edit_bone.tail = head + rot @ Vector((0.0, 0.1 * scale, 0.0))

            if bone.parent >= 0 and bone.parent < len(skel.bones):
                parent_edit_bone = armature.edit_bones[skel.bones[bone.parent].name]
                edit_bone.parent = parent_edit_bone

        bpy.ops.object.mode_set(mode="OBJECT")

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

        preferred = next(
            (n for n in tree if n.get("path") == previous_path and not n["is_dir"]),
            None,
        )
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

        self._refresh_animation_metadata()

    def _build_animations(
        self,
        context: bpy.types.Context,
        pack: HavokPack,
        animations: List[parsers.HavokAnimation],
        armature_obj: Optional[bpy.types.Object],
        scale: float,
        axis_mat: Matrix,
    ) -> None:
        if armature_obj is None:
            logger.info("Skipping animation build: no armature available")
            return

        build_start = time.perf_counter()
        logger.info(
            "Building %d animation(s) on armature '%s'",
            len(animations),
            armature_obj.name,
        )

        def _write_curve_samples(
            curve, frames: List[float], values: List[float]
        ) -> None:
            if not frames or not values or len(frames) != len(values):
                if frames and values:
                    logger.warning(
                        "Keyframe mismatch frames=%d values=%d",
                        len(frames),
                        len(values),
                    )
                return
            keyframes = curve.keyframe_points
            keyframes.add(len(frames))
            for idx, (frame_val, value) in enumerate(zip(frames, values)):
                key = keyframes[idx]
                key.co = (frame_val, value)
                key.interpolation = "LINEAR"
                if hasattr(key, "handle_left_type"):
                    key.handle_left_type = "AUTO"
                if hasattr(key, "handle_right_type"):
                    key.handle_right_type = "AUTO"

        # 1. Reconstruct Havok Global Rest Poses
        # We need this to calculate the mapping between Havok bone space and Blender bone space
        havok_global_rest = []
        if pack.skeleton:
            for idx, bone in enumerate(pack.skeleton.bones):
                # HavokBone stores the reference pose in local space
                local_t = Matrix.Translation(bone.translation)
                local_r = bone.rotation.to_matrix().to_4x4()
                local_s = (
                    Matrix.Scale(bone.scale[0], 4, (1, 0, 0))
                    @ Matrix.Scale(bone.scale[1], 4, (0, 1, 0))
                    @ Matrix.Scale(bone.scale[2], 4, (0, 0, 1))
                )

                local_mat = local_t @ local_r @ local_s

                if bone.parent >= 0 and bone.parent < len(havok_global_rest):
                    parent_mat = havok_global_rest[bone.parent]
                    global_mat = parent_mat @ local_mat
                else:
                    global_mat = local_mat

                havok_global_rest.append(global_mat)

        # 2. Calculate R_fix for each bone
        # R_fix = M_havok_rest.inverted() @ M_blender_rest
        # M_blender_rest is bone.matrix_local
        bone_r_fix = {}
        blender_rest_globals = {}

        bpy.ops.object.mode_set(
            mode="EDIT"
        )  # Ensure we are reading edit bones for rest pose?
        # No, object mode is fine for reading data.bones
        bpy.ops.object.mode_set(mode="OBJECT")

        for bone in armature_obj.data.bones:
            # Find corresponding Havok bone index
            h_idx = -1
            if pack.skeleton:
                for i, b in enumerate(pack.skeleton.bones):
                    if b.name == bone.name:
                        h_idx = i
                        break

            if h_idx != -1 and h_idx < len(havok_global_rest):
                m_havok = havok_global_rest[h_idx]
                m_blender = bone.matrix_local  # Armature space

                # R_fix maps Havok vector to Blender vector in Global space
                # v_blender = v_havok @ R_fix ? No.
                # M_blender = M_havok @ R_fix
                # R_fix = M_havok.inverted() @ M_blender
                r_fix = m_havok.inverted() @ m_blender
                bone_r_fix[bone.name] = r_fix
                blender_rest_globals[bone.name] = m_blender
            else:
                bone_r_fix[bone.name] = Matrix.Identity(4)
                blender_rest_globals[bone.name] = bone.matrix_local

        # Pre-calculate rest local matrices for all bones (Blender space)
        rest_locals = {}
        for bone in armature_obj.data.bones:
            if bone.parent:
                rest_locals[bone.name] = (
                    bone.parent.matrix_local.inverted() @ bone.matrix_local
                )
            else:
                rest_locals[bone.name] = bone.matrix_local

        for animation in animations:
            anim_start = time.perf_counter()
            logger.info(
                "Creating animation '%s' duration=%.2fs tracks=%d",
                animation.name,
                animation.duration,
                len(animation.tracks),
            )
            action = bpy.data.actions.new(animation.name)
            armature_obj.animation_data_create()
            armature_obj.animation_data.action = action

            # Blender 5.0+ compatibility: Action.fcurves replaced by Layered Action API
            fcurves_collection = None
            if hasattr(action, "fcurves"):
                fcurves_collection = action.fcurves
            else:
                # Create necessary hierarchy for Layered Action
                slot = action.slots.new("OBJECT", "Slot")
                layer = action.layers.new("Layer")
                strip = layer.strips.new(type="KEYFRAME")
                bag = strip.channelbags.new(slot)
                fcurves_collection = bag.fcurves

                # Assign the slot to the object's animation data
                if hasattr(armature_obj.animation_data, "action_slot_handle"):
                    armature_obj.animation_data.action_slot_handle = slot.handle

            is_additive = animation.blend_hint in ("ADDITIVE", "ADDITIVE_DEPRECATED")

            for track_idx, track in enumerate(animation.tracks):
                if not track:
                    logger.info(
                        "Animation '%s': track %d is empty, skipping",
                        animation.name,
                        track_idx,
                    )
                    continue

                bone_name = None

                # Try to find bone by annotation track name first
                if track_idx < len(animation.annotation_tracks):
                    annot_name = animation.annotation_tracks[track_idx]
                    if annot_name and armature_obj.pose.bones.get(annot_name):
                        bone_name = annot_name

                # Fallback to binding indices
                if not bone_name:
                    bone_idx = (
                        animation.track_to_bone[track_idx]
                        if track_idx < len(animation.track_to_bone)
                        else track_idx
                    )
                    if pack.skeleton and 0 <= bone_idx < len(pack.skeleton.bones):
                        bone_name = pack.skeleton.bones[bone_idx].name

                if not bone_name:
                    logger.warning(
                        "Animation '%s': track %d has no bone binding, skipping",
                        animation.name,
                        track_idx,
                    )
                    continue

                pose_bone = armature_obj.pose.bones.get(bone_name)
                if not pose_bone:
                    logger.warning(
                        "Animation '%s': bone '%s' missing on armature, skipping track %d",
                        animation.name,
                        bone_name,
                        track_idx,
                    )
                    continue

                logger.info(
                    "Animation '%s': processing track %d -> bone '%s' (%d frames)",
                    animation.name,
                    track_idx,
                    bone_name,
                    len(track),
                )

                data_path_loc = pose_bone.path_from_id("location")
                data_path_rot = pose_bone.path_from_id("rotation_quaternion")
                data_path_scale = pose_bone.path_from_id("scale")

                fcurves_loc = [
                    fcurves_collection.new(data_path_loc, index=i) for i in range(3)
                ]
                fcurves_rot = [
                    fcurves_collection.new(data_path_rot, index=i) for i in range(4)
                ]
                fcurves_scale = [
                    fcurves_collection.new(data_path_scale, index=i) for i in range(3)
                ]

                frame_count = len(track)
                frame_rate = (
                    (animation.duration / max(frame_count - 1, 1))
                    if animation.duration > 0
                    else 1.0
                )
                fps_ratio = (
                    context.scene.render.fps / context.scene.render.fps_base
                    if context.scene.render.fps_base
                    else context.scene.render.fps
                )
                frame_scale = frame_rate * fps_ratio

                rest_local = rest_locals.get(bone_name, Matrix.Identity(4))
                r_fix = bone_r_fix.get(bone_name, Matrix.Identity(4))

                # Parent info for global calculation
                parent_bone = pose_bone.parent
                parent_name = parent_bone.name if parent_bone else None

                # We need to cache calculated global matrices for the current frame
                # But tracks are processed independently. This is a problem for global calculation.
                # However, we can assume that we only need the PARENT's global matrix.
                # But we don't have the parent's animated global matrix yet!
                # This approach requires processing frames time-slice by time-slice, not track by track.

                # Alternative:
                # If we assume the animation is purely local (which it is),
                # and we want to apply it to the Blender bone.
                # M_blender_local = M_blender_parent_global.inverted() @ M_blender_global
                # M_blender_global = M_havok_global @ R_fix
                # M_havok_global = M_havok_parent_global @ M_havok_local

                # This dependency chain is hard to resolve track-by-track.

                # SIMPLIFICATION:
                # Assume R_fix is only a rotation (it usually is).
                # And assume parent's R_fix is similar? No.

                # Let's go back to the delta idea.
                # D_havok = M_havok_rest_local.inverted() @ M_havok_anim_local
                # This is the local deformation in Havok space.
                # We want to apply this deformation to the Blender bone.
                # But the Blender bone has a different coordinate system.
                # If we map D_havok to Blender space:
                # D_blender = R_fix_local.inverted() @ D_havok @ R_fix_local ?
                # Where R_fix_local is the rotation from Havok Bone Local Space to Blender Bone Local Space.

                # Is R_fix (Global) the same as R_fix_local?
                # M_blender = M_havok @ R_fix.
                # M_blender_local = M_blender_parent.inverted() @ M_blender
                # = (M_havok_parent @ R_fix_parent).inverted() @ (M_havok @ R_fix)
                # = R_fix_parent.inverted() @ M_havok_parent.inverted() @ M_havok @ R_fix
                # = R_fix_parent.inverted() @ M_havok_local @ R_fix

                # So M_blender_local depends on Parent's R_fix and Own R_fix.

                # We can pre-calculate R_fix for all bones.
                # And we can pre-calculate R_fix_parent for all bones.

                r_fix_parent = Matrix.Identity(4)
                if parent_name:
                    r_fix_parent = bone_r_fix.get(parent_name, Matrix.Identity(4))

                # Now we can calculate M_blender_local for each frame WITHOUT full global simulation.
                # M_blender_local = R_fix_parent.inverted() @ M_havok_local @ R_fix

                # This is it! This formula converts a Havok Local Transform to a Blender Local Transform
                # preserving the global pose, accounting for axis changes in both parent and child.

                r_fix_parent_inv = r_fix_parent.inverted()

                track_log_interval = max(1, len(track) // 10 or 1)

                frame_numbers: List[float] = []
                loc_samples = [[], [], []]
                rot_samples = [[], [], [], []]
                scale_samples = [[], [], []]

                for frame_idx, (trans, quat, scale_vec) in enumerate(track):
                    # Apply user overrides
                    t_x, t_y, t_z = trans
                    q_w, q_x, q_y, q_z = quat.w, quat.x, quat.y, quat.z

                    if self.anim_swap_yz:
                        t_y, t_z = t_z, t_y
                        q_y, q_z = q_z, q_y

                    if self.anim_flip_x:
                        t_x = -t_x
                    if self.anim_flip_y:
                        t_y = -t_y
                    if self.anim_flip_z:
                        t_z = -t_z

                    if self.anim_flip_quat_x:
                        q_x = -q_x
                    if self.anim_flip_quat_y:
                        q_y = -q_y
                    if self.anim_flip_quat_z:
                        q_z = -q_z
                    if self.anim_flip_quat_w:
                        q_w = -q_w

                    trans = Vector((t_x, t_y, t_z))
                    quat = mathutils.Quaternion((q_w, q_x, q_y, q_z))

                    # Construct M_havok_local
                    m_havok_local = (
                        Matrix.Translation(trans)
                        @ quat.to_matrix().to_4x4()
                        @ Matrix.Scale(scale_vec[0], 4, (1, 0, 0))
                        @ Matrix.Scale(scale_vec[1], 4, (0, 1, 0))
                        @ Matrix.Scale(scale_vec[2], 4, (0, 0, 1))
                    )

                    # Calculate M_blender_local
                    # M_blender_local = R_fix_parent.inverted() @ M_havok_local @ R_fix
                    m_blender_local = r_fix_parent_inv @ m_havok_local @ r_fix

                    # Calculate M_basis (relative to rest pose)
                    # M_basis = rest_local.inverted() @ m_blender_local
                    matrix_basis = rest_local.inverted() @ m_blender_local

                    loc, rot, sca = matrix_basis.decompose()

                    frame_value = frame_idx * frame_scale
                    frame_numbers.append(frame_value)

                    for axis in range(3):
                        loc_samples[axis].append(loc[axis])
                        scale_samples[axis].append(sca[axis])
                    for axis in range(4):
                        rot_samples[axis].append(rot[axis])

                    if (
                        frame_idx % track_log_interval == 0
                        or frame_idx == len(track) - 1
                    ):
                        logger.debug(
                            "Animation '%s': track %d frame %d/%d applied",
                            animation.name,
                            track_idx,
                            frame_idx + 1,
                            len(track),
                        )

                for axis, curve in enumerate(fcurves_loc):
                    _write_curve_samples(curve, frame_numbers, loc_samples[axis])
                for axis, curve in enumerate(fcurves_rot):
                    _write_curve_samples(curve, frame_numbers, rot_samples[axis])
                for axis, curve in enumerate(fcurves_scale):
                    _write_curve_samples(curve, frame_numbers, scale_samples[axis])

            logger.info(
                "Finished animation '%s' in %.2fs",
                animation.name,
                time.perf_counter() - anim_start,
            )

            # Keep last action applied
            armature_obj.animation_data.action = action

        logger.info(
            "Finished building animations in %.2fs",
            time.perf_counter() - build_start,
        )

    def _get_selected_armature(
        self, context: bpy.types.Context
    ) -> Optional[bpy.types.Object]:
        for obj in context.selected_objects:
            if obj.type == "ARMATURE":
                return obj
        return None

    def _should_import_animation(
        self,
        target_ext: str,
        animations: List[parsers.HavokAnimation],
        armature_obj: Optional[bpy.types.Object],
    ) -> bool:
        if not animations:
            return False
        if target_ext == ".hka":
            return armature_obj is not None
        return True

    def _resolve_animations(self, pack: HavokPack) -> List[parsers.HavokAnimation]:
        # Use animation_active_index if available and valid
        if self.animation_entries and 0 <= self.animation_active_index < len(
            self.animation_entries
        ):
            entry = self.animation_entries[self.animation_active_index]
            if entry.index == -1:
                return list(pack.animations)
            if 0 <= entry.index < len(pack.animations):
                return [pack.animations[entry.index]]

        # Fallback to legacy animation_index
        if self.animation_index < 0:
            return list(pack.animations)

        if self.animation_index >= len(pack.animations):
            self.report(
                {"WARNING"},
                f"Animation index {self.animation_index} is out of range; no animations imported",
            )
            return []

        return [pack.animations[self.animation_index]]

    def _refresh_animation_metadata(self) -> None:
        filepath = Path(self.filepath) if self.filepath else None
        if filepath is None or not filepath.suffix:
            return

        target_ext = Path(self.archive_entry or filepath.name).suffix.lower()
        if target_ext not in {".hka", ".hkx"}:
            if self.animation_metadata_key:
                self.animation_metadata_key = ""
                self.animation_count = 0
                self.animation_names = ""
            return

        key = "|".join(
            [
                str(filepath),
                self.archive_entry,
                self.pak_profile,
                self.pak_platform,
            ]
        )
        if key == self.animation_metadata_key:
            return

        self.animation_metadata_key = key
        self.animation_count = 0
        self.animation_names = ""

        try:
            pack = load_from_path(
                filepath,
                entry=self.archive_entry or None,
                pak_profile=(
                    self.pak_profile if filepath.suffix.lower() == ".pak" else None
                ),
                pak_platform=(
                    self.pak_platform if filepath.suffix.lower() == ".pak" else None
                ),
                metadata_only=True,
            )
        except Exception:
            return

        self.animation_count = len(pack.animations)
        self.animation_names = "\n".join(
            anim.name or f"Animation {idx}" for idx, anim in enumerate(pack.animations)
        )

        self.animation_entries.clear()

        # Add "All Animations" option
        item = self.animation_entries.add()
        item.name = "All Animations"
        item.index = -1
        item.duration = 0.0
        item.num_tracks = 0

        for idx, anim in enumerate(pack.animations):
            item = self.animation_entries.add()
            item.name = anim.name or f"Animation {idx}"
            item.index = idx
            item.duration = anim.duration
            item.num_tracks = len(anim.tracks)

        if self.animation_index >= self.animation_count and self.animation_count:
            self.animation_index = 0

    def _apply_igz_settings(self) -> None:
        igz_constants.dBuildMeshes = self.igz_build_meshes
        igz_constants.dBuildBones = self.igz_build_bones
        igz_constants.dBuildFaces = self.igz_build_faces
        igz_constants.dAllowWii = self.igz_allow_wii
        igz_constants.dModelThreshold = self.igz_model_threshold
        igz_constants.dFirstObjectOffset = self.igz_first_object_offset

    def _import_igz_blob(self, data: bytes):
        # Mirror the io_scene_igz import path by letting its parser build Blender
        # objects directly from the binary IGZ stream.
        from ..io.igz_port import game_formats as igz_formats

        profile_to_parser: Dict[str, Callable[[bytes], object]] = {
            "IMAGINATORS": igz_formats.sscIgzFile,
            "SSA_WII": igz_formats.ssaIgzFile,
            "SSA_WIIU": igz_formats.ssaIgzFile,
            "SWAP_FORCE": igz_formats.ssfIgzFile,
            "TRAP_TEAM": igz_formats.sttIgzFile,
            "SUPER_CHARGERS": igz_formats.sscIgzFile,
        }

        parser_factory = profile_to_parser.get(self.pak_profile, igz_formats.sscIgzFile)
        parser = parser_factory(data)
        parser.loadFile()
        parser.buildMeshes()

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
    self.layout.operator(
        HAVOK_OT_import.bl_idname, text="Havok (.hkx/.hkt/.hka/.igz/.pak)"
    )
