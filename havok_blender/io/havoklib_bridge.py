"""Lightweight bridge to HavokLib's Python module (`havokpy`).

The legacy HavokMax plugin relied on HavokLib to unwrap binary HKX/HKA/HKT
packfiles. Blender add-ons cannot ship compiled C++ extensions by default, but
users can still build the `havokpy` module from the original HavokLib project.
When present, this helper converts binary packfiles to XML so the rest of the
importer can parse skeletons and animations without placeholders.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from typing import Optional


def _load_havokpy():
    module = sys.modules.get("havokpy")
    if module is not None:
        return module

    try:
        return importlib.import_module("havokpy")
    except ModuleNotFoundError:
        pass

    addon_root = Path(__file__).resolve().parent.parent
    search_roots = [addon_root / "3rd_party" / "HavokLib", addon_root / "3rd_party"]

    for root in search_roots:
        if not root.exists():
            continue

        for candidate in root.rglob("havokpy.*"):
            if not candidate.is_file():
                continue

            parent_str = str(candidate.parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)

            try:
                return importlib.import_module("havokpy")
            except Exception:
                continue

    return None


def convert_packfile_to_xml(path: Path) -> Optional[bytes]:
    """Use HavokLib's Python bindings to turn a binary packfile into XML.

    Args:
        path: Path to an HKX/HKA/HKT binary packfile.

    Returns:
        XML bytes when the conversion succeeds, otherwise ``None`` so callers
        can fall back to other heuristics.
    """

    havokpy = _load_havokpy()
    if havokpy is None:
        return None

    try:
        pack = havokpy.hkPackfile(str(path))
    except Exception:
        return None

    # Prefer the highest toolset version available, mirroring the legacy
    # HavokMax behavior when exporting XML.
    toolset = getattr(havokpy, "HK2014", None)
    if toolset is None:
        toolset = getattr(havokpy, "HKX2014", None)
    if toolset is None:
        toolset = getattr(havokpy, "HK2011", None)

    try:
        xml_text = pack.to_xml(toolset) if toolset is not None else pack.to_xml()
    except Exception:
        return None

    if isinstance(xml_text, str):
        return xml_text.encode("utf-8")
    return None


def convert_bytes_to_xml(data: bytes) -> Optional[bytes]:
    """Temporarily persist bytes to allow `havokpy` to convert them."""

    havokpy = _load_havokpy()
    if havokpy is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".hkx", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        return convert_packfile_to_xml(Path(tmp.name))

