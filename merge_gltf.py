#!/usr/bin/env python3
"""Merge all GLTF files in a directory into one GLTF scene."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def _aligned_length(length: int) -> int:
    """Return the next 4-byte-aligned length required by GLTF."""
    return (length + 3) & ~3


def _read_buffer(path: Path, buffer: dict[str, Any]) -> bytes:
    uri = buffer.get("uri")
    if not uri or uri.startswith("data:"):
        raise ValueError(f"{path}: embedded or missing buffer URI is not supported")
    buffer_path = path.parent / uri
    try:
        return buffer_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{path}: cannot read buffer {buffer_path}") from exc


def _remap_texture_indices(value: Any, texture_offset: int) -> Any:
    """Remap all glTF texture-info indices in a material."""
    if isinstance(value, list):
        return [_remap_texture_indices(item, texture_offset) for item in value]
    if not isinstance(value, dict):
        return value

    result = {}
    for key, item in value.items():
        if key.endswith("Texture") and isinstance(item, dict):
            item = copy.deepcopy(item)
            if "index" in item:
                item["index"] += texture_offset
            result[key] = _remap_texture_indices(item, texture_offset)
        else:
            result[key] = _remap_texture_indices(item, texture_offset)
    return result


def merge_gltf(parts_dir: Path, output_path: Path) -> None:
    files = sorted(parts_dir.glob("*.gltf"))
    if not files:
        raise ValueError(f"No .gltf files found in {parts_dir}")

    merged: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "merge_gltf.py"},
        "scene": 0,
        "scenes": [{"nodes": []}],
        "nodes": [],
        "meshes": [],
        "accessors": [],
        "bufferViews": [],
        "buffers": [{"uri": output_path.with_suffix(".bin").name, "byteLength": 0}],
    }
    binary = bytearray()
    extensions_used: set[str] = set()
    extensions_required: set[str] = set()

    resource_names = ("samplers", "images", "textures", "materials")
    for name in resource_names:
        merged[name] = []

    for source_path in files:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source_buffers = source.get("buffers", [])
        source_data = [_read_buffer(source_path, buffer) for buffer in source_buffers]

        while len(binary) % 4:
            binary.append(0)
        buffer_base = len(binary)
        for data in source_data:
            binary.extend(data)
            while len(binary) % 4:
                binary.append(0)

        accessor_offset = len(merged["accessors"])
        buffer_view_offset = len(merged["bufferViews"])
        mesh_offset = len(merged["meshes"])
        node_offset = len(merged["nodes"])
        sampler_offset = len(merged["samplers"])
        image_offset = len(merged["images"])
        texture_offset = len(merged["textures"])
        material_offset = len(merged["materials"])

        source_buffer_bases: list[int] = []
        current = buffer_base
        for data in source_data:
            source_buffer_bases.append(current)
            current += _aligned_length(len(data))

        for view in source.get("bufferViews", []):
            view = copy.deepcopy(view)
            source_buffer = view.get("buffer", 0)
            view["buffer"] = 0
            view["byteOffset"] = source_buffer_bases[source_buffer] + view.get(
                "byteOffset", 0
            )
            merged["bufferViews"].append(view)

        for accessor in source.get("accessors", []):
            accessor = copy.deepcopy(accessor)
            accessor["bufferView"] = accessor["bufferView"] + buffer_view_offset
            merged["accessors"].append(accessor)

        for sampler in source.get("samplers", []):
            merged["samplers"].append(copy.deepcopy(sampler))

        for image in source.get("images", []):
            image = copy.deepcopy(image)
            if "bufferView" in image:
                image["bufferView"] += buffer_view_offset
            merged["images"].append(image)

        for texture in source.get("textures", []):
            texture = copy.deepcopy(texture)
            if "sampler" in texture:
                texture["sampler"] += sampler_offset
            if "source" in texture:
                texture["source"] += image_offset
            merged["textures"].append(texture)

        for material in source.get("materials", []):
            merged["materials"].append(
                _remap_texture_indices(material, texture_offset)
            )

        for mesh in source.get("meshes", []):
            mesh = copy.deepcopy(mesh)
            for primitive in mesh.get("primitives", []):
                primitive["attributes"] = {
                    key: value + accessor_offset
                    for key, value in primitive.get("attributes", {}).items()
                }
                if "indices" in primitive:
                    primitive["indices"] += accessor_offset
                if "material" in primitive:
                    primitive["material"] += material_offset
            merged["meshes"].append(mesh)

        for node in source.get("nodes", []):
            node = copy.deepcopy(node)
            if "mesh" in node:
                node["mesh"] += mesh_offset
            if "children" in node:
                node["children"] = [child + node_offset for child in node["children"]]
            if "camera" in node:
                node["camera"] += len(merged.get("cameras", []))
            if "skin" in node:
                node["skin"] += len(merged.get("skins", []))
            merged["nodes"].append(node)

        scene_nodes = []
        for scene in source.get("scenes", []):
            scene_nodes.extend(node + node_offset for node in scene.get("nodes", []))
        merged["scenes"][0]["nodes"].extend(scene_nodes)

        extensions_used.update(source.get("extensionsUsed", []))
        extensions_required.update(source.get("extensionsRequired", []))

    merged["buffers"][0]["byteLength"] = len(binary)
    if extensions_used:
        merged["extensionsUsed"] = sorted(extensions_used)
    if extensions_required:
        merged["extensionsRequired"] = sorted(extensions_required)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    output_path.with_suffix(".bin").write_bytes(binary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parts_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("parts"),
        help="directory containing source .gltf files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name("merged.gltf"),
        help="output GLTF path (the binary is written beside it)",
    )
    args = parser.parse_args()
    merge_gltf(args.parts_dir, args.output)
    print(f"Merged GLTF written to {args.output}")


if __name__ == "__main__":
    main()