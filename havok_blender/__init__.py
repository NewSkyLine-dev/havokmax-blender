"""Havok IO add-on for Blender 5.0."""
from importlib import reload
from typing import Iterable

import bpy

from . import operators, ui


bl_info = {
    "name": "Havok IO",
    "author": "PredatorCZ & Contributors",
    "version": (0, 1, 0),
    "blender": (5, 0, 0),
    "location": "File > Import/Export",
    "description": "Import Havok HKX/HKT/HKA/IGZ/PAK assets",
    "warning": "Early Blender port – format coverage is limited",
    "doc_url": "https://github.com/PredatorCZ/HavokMax",
    "tracker_url": "https://github.com/PredatorCZ/HavokMax/issues",
    "category": "Import-Export",
}


# Utilities
_modules: Iterable = (operators, ui)


def reload_modules():
    for module in _modules:
        reload(module)


classes = tuple(operators.classes + ui.classes)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    operators.add_menu_items()


def unregister():
    operators.remove_menu_items()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":  # pragma: no cover - convenience for reloading
    reload_modules()
