"""UI panels for the Havok IO add-on."""
from __future__ import annotations

import bpy


class HAVOK_PT_tools(bpy.types.Panel):
    bl_label = "Havok IO"
    bl_idname = "HAVOK_PT_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Havok"

    def draw(self, context):  # pragma: no cover - UI
        layout = self.layout
        prefs = context.preferences.addons[__package__.split(".")[0]].preferences

        layout.label(text="Import")
        import_op = layout.operator("havok.import_hkx", icon="IMPORT")
        import_op.import_animation = True
        import_op.import_skeleton = True

        layout.separator()
        layout.label(text="Presets")
        layout.prop(prefs, "scale")
        layout.prop(prefs, "up_axis")
        layout.prop(prefs, "forward_axis")


classes = (HAVOK_PT_tools,)
