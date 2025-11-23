# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from . import constants, game_formats, utils


_VERSION_PARSERS: Dict[int, Callable[[bytes], Any]] = {
    0x05: game_formats.ssaIgzFile,
    0x06: game_formats.sgIgzFile,
    0x07: game_formats.ssfIgzFile,
    0x08: game_formats.sttIgzFile,
    0x09: game_formats.sscIgzFile,
    0x0A: game_formats.nstIgzFile,
}


class ImportSkylandersIGZ(bpy.types.Operator, ImportHelper):
    """Import Skylanders IGZ/BLD models."""

    bl_idname = "import_mesh.skylanders_igz"
    bl_label = "Import Skylanders IGZ/BLD"
    bl_options = {"PRESET", "UNDO"}

    filename_ext: str = ".igz;.bld"
    filter_glob: StringProperty = StringProperty(
        default="*.igz;*.bld",
        options={"HIDDEN"},
    )

    build_meshes: BoolProperty = BoolProperty(
        name="Build Meshes",
        description="Whether to build the meshes or just parse the file",
        default=True,
    )

    build_bones: BoolProperty = BoolProperty(
        name="Build Bones",
        description="Whether to build the bones",
        default=True,
    )

    build_faces: BoolProperty = BoolProperty(
        name="Build Faces",
        description="Whether to build the faces",
        default=True,
    )

    allow_wii: BoolProperty = BoolProperty(
        name="Allow Wii Models",
        description="Whether to allow Wii models (may be buggy)",
        default=True,
    )

    model_threshold: IntProperty = IntProperty(
        name="Model Threshold",
        description=(
            "Maximum number of models to import before limiting to a subset. "
            "This mirrors the legacy Noesis importer safety prompt."
        ),
        default=constants.dModelThreshold,
        min=1,
    )

    first_object_offset: IntProperty = IntProperty(
        name="First Object Offset",
        description=(
            "Override the offset of the first IGObject. Use -1 to let the parser "
            "walk all objects automatically."
        ),
        default=constants.dFirstObjectOffset,
        min=-1,
    )

    def draw(self, _context: bpy.types.Context) -> None:  # pragma: no cover - UI
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "build_meshes")
        layout.prop(self, "build_faces")
        layout.prop(self, "build_bones")
        layout.prop(self, "allow_wii")
        layout.separator()
        layout.prop(self, "model_threshold")
        layout.prop(self, "first_object_offset")

    def execute(self, _context: Any) -> set:
        self._apply_import_settings()

        try:
            data = Path(self.filepath).read_bytes()
        except OSError as exc:  # pragma: no cover - Blender handles reporting
            self.report({"ERROR"}, f"Failed to read file: {exc}")
            return {"CANCELLED"}

        header = self._validate_magic_and_version(data)
        if header is None:
            self.report({"ERROR"}, "Invalid or unsupported IGZ/BLD file")
            return {"CANCELLED"}

        version, _endian = header
        parser = self._build_parser(version, data)
        if parser is None:
            self.report({"ERROR"}, f"Version {hex(version)} is unsupported.")
            return {"CANCELLED"}

        parser.loadFile()

        if parser.version < 0x0A and parser.platform == 2 and not constants.dAllowWii:
            self.report(
                {"ERROR"},
                "Wii Models are not allowed as they are buggy. Enable 'Allow Wii Models' in import options to try anyway.",
            )
            return {"CANCELLED"}

        if constants.dBuildMeshes:
            parser.buildMeshes()

        self.report({"INFO"}, f"Successfully imported {len(parser.models)} models")
        return {"FINISHED"}

    def _apply_import_settings(self) -> None:
        constants.dBuildMeshes = self.build_meshes
        constants.dBuildBones = self.build_bones
        constants.dBuildFaces = self.build_faces
        constants.dAllowWii = self.allow_wii
        constants.dModelThreshold = self.model_threshold
        constants.dFirstObjectOffset = self.first_object_offset

    def _validate_magic_and_version(self, data: bytes) -> Optional[Tuple[int, constants.Endianness]]:
        magic = int.from_bytes(data[:4], byteorder="little", signed=False)
        endian = None
        if magic == 0x015A4749:
            endian = constants.Endianness.LITTLE
        elif magic == 0x49475A01:
            endian = constants.Endianness.BIG
        else:
            return None

        bs = utils.NoeBitStream(data, endian)
        bs.readUInt()  # skip magic
        version = bs.readUInt()
        return version, endian

    def _build_parser(self, version: int, data: bytes) -> Optional[Any]:
        factory = _VERSION_PARSERS.get(version)
        return factory(data) if factory else None


# ------------------------------------------------------------------------------
# Register/Unregister functionality
# ------------------------------------------------------------------------------
def menu_func_import(self, _context):
    self.layout.operator(
        ImportSkylandersIGZ.bl_idname, text="Skylanders IGZ/BLD (.igz/.bld)"
    )


def register():
    bpy.utils.register_class(ImportSkylandersIGZ)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.utils.unregister_class(ImportSkylandersIGZ)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)


if __name__ == "__main__":
    register()
