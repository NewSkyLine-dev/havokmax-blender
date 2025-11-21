# Havok IO (Blender 5.0)

A Blender add-on that replaces the archived HavokMax 3ds Max plugin. The port
adds Blender-native import entries for Havok `.hkx`, `.hkt`, `.hka`, `.igz`,
and `.pak` containers, plus configuration options that mirror the original
tool's presets.

> **Status:** The importer now reads Havok XML packfiles directly and mirrors the
> igArchiveExtractor logic to unwrap PAK/IGZ archives while building real
> armatures and keyframed actions when animation data is present. IGZ files are
> parsed with the io_scene_igz-style fixup walker so NHMT memory blocks are
> resolved before scanning for Havok XML or compressed payloads.

## Installation

1. Download or clone this repository.
2. From Blender **Edit → Preferences → Add-ons → Install…**, select the project
   directory (the folder containing `manifest.toml`).
3. Enable **Havok IO** in the add-on list.

## Usage

- **Import:** `File → Import → Havok (.hkx/.hkt/.hka/.igz/.pak)` will unpack
  compressed IGZ payloads, scan PAK containers (zip, tar, or raw blobs) for
  Havok XML, and build an armature plus animation actions. Selecting a `.pak`
  shows a tree of contained entries parsed with the igArchiveExtractor chunk
  rules; pick an entry to import or type a path manually in **Archive entry**.
- **Presets:** Configure scale and up/forward axes in **Preferences → Add-ons →
  Havok IO** or in the 3D View **Havok** side panel.

### HKX/HKT/HKA support

The add-on expects Havok XML packfiles. If you encounter binary Havok assets,
convert them to XML first (for example with hkxpack) and then import the
resulting XML or IGZ/PAK wrapper.

## Development

The add-on uses Blender's new `manifest.toml` packaging format (introduced for
Blender 4.2+ and forward-compatible with Blender 5.0). Source files live in the
`havok_blender/` package:

- `operators/` contains the import operator and shared preferences.
- `ui/` defines the 3D View toolbar panel.

To validate basic syntax without Blender:

```bash
python -m compileall havok_blender
```

## License

Havok IO is available under the GPL v3 license (see `LICENSE`).
