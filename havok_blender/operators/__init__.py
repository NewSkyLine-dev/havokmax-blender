"""Operators exposed by the Havok IO add-on."""
from . import import_havok, preferences

classes = (
    preferences.HAVOK_AddonPreferences,
    import_havok.HAVOK_OT_import,
)


def add_menu_items():
    import bpy

    bpy.types.TOPBAR_MT_file_import.append(import_havok.menu_func_import)


def remove_menu_items():
    import bpy

    bpy.types.TOPBAR_MT_file_import.remove(import_havok.menu_func_import)
