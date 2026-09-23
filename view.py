#!/usr/bin/env python3
"""Display a textured GLTF model with mouse orbit, pan, and zoom controls."""

from __future__ import annotations

import argparse
import base64
import copy
import ctypes
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyglet
from PIL import Image
from pyglet import gl

pyglet.options["debug_gl"] = False


COMPONENT_DTYPES = {
    5121: np.dtype("<u1"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
TYPE_SHAPES = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}


def read_accessor(
    model: dict[str, Any], buffers: list[bytes], accessor_index: int
) -> np.ndarray:
    accessor = model["accessors"][accessor_index]
    dtype = COMPONENT_DTYPES[accessor["componentType"]]
    item_size = TYPE_SHAPES[accessor["type"]]
    count = accessor["count"]
    if "bufferView" not in accessor:
        return np.zeros((count, item_size), dtype=dtype)

    view = model["bufferViews"][accessor["bufferView"]]
    buffer = buffers[view.get("buffer", 0)]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * item_size)
    if stride == dtype.itemsize * item_size:
        values = np.frombuffer(
            buffer, dtype=dtype, count=count * item_size, offset=start
        ).reshape(count, item_size)
    else:
        values = np.ndarray(
            (count, item_size),
            dtype=dtype,
            buffer=buffer,
            offset=start,
            strides=(stride, dtype.itemsize),
        ).copy()
    if accessor.get("normalized"):
        if np.issubdtype(values.dtype, np.unsignedinteger):
            values = values.astype(np.float32) / np.iinfo(values.dtype).max
        else:
            values = np.maximum(
                values.astype(np.float32) / np.iinfo(values.dtype).max, -1.0
            )
    return values


def node_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float32).reshape((4, 4), order="F")

    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    scale = np.eye(4, dtype=np.float32)
    scale[range(3), range(3)] = node.get("scale", [1, 1, 1])
    translation = np.eye(4, dtype=np.float32)
    translation[:3, 3] = node.get("translation", [0, 0, 0])
    return translation @ rotation @ scale


def load_buffers(model_path: Path, model: dict[str, Any]) -> list[bytes]:
    result = []
    for buffer in model.get("buffers", []):
        uri = buffer.get("uri", "")
        if uri.startswith("data:"):
            result.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            result.append((model_path.parent / uri).read_bytes())
    return result


def image_bytes(
    model_path: Path, model: dict[str, Any], buffers: list[bytes], image: dict[str, Any]
) -> bytes:
    if "uri" in image:
        uri = image["uri"]
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        return (model_path.parent / uri).read_bytes()
    view = model["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    end = start + view["byteLength"]
    return buffers[view.get("buffer", 0)][start:end]


def create_textures(
    model_path: Path, model: dict[str, Any], buffers: list[bytes]
) -> list[int]:
    image_data: list[tuple[int, int, bytes]] = []
    for image in model.get("images", []):
        with Image.open(io.BytesIO(image_bytes(model_path, model, buffers, image))) as source:
            rgba = source.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            image_data.append((rgba.width, rgba.height, rgba.tobytes()))

    texture_ids: list[int] = []
    for texture in model.get("textures", []):
        source_index = texture.get("source")
        if source_index is None or source_index >= len(image_data):
            texture_ids.append(0)
            continue

        width, height, pixels = image_data[source_index]
        sampler_index = texture.get("sampler")
        sampler = (
            model.get("samplers", [])[sampler_index]
            if sampler_index is not None
            and sampler_index < len(model.get("samplers", []))
            else {}
        )
        texture_id = gl.GLuint()
        gl.glGenTextures(1, texture_id)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        min_filter = {
            9728: gl.GL_NEAREST,
            9729: gl.GL_LINEAR,
            9984: gl.GL_NEAREST_MIPMAP_NEAREST,
            9985: gl.GL_LINEAR_MIPMAP_NEAREST,
            9986: gl.GL_NEAREST_MIPMAP_LINEAR,
            9987: gl.GL_LINEAR_MIPMAP_LINEAR,
        }.get(sampler.get("minFilter", 9729), gl.GL_LINEAR)
        mag_filter = (
            gl.GL_NEAREST
            if sampler.get("magFilter", 9729) == 9728
            else gl.GL_LINEAR
        )
        wrap_s = (
            gl.GL_CLAMP_TO_EDGE
            if sampler.get("wrapS", 10497) == 33071
            else gl.GL_REPEAT
        )
        wrap_t = (
            gl.GL_CLAMP_TO_EDGE
            if sampler.get("wrapT", 10497) == 33071
            else gl.GL_REPEAT
        )
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, min_filter)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, mag_filter)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, wrap_s)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, wrap_t)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            pixels,
        )
        if min_filter in (
            gl.GL_NEAREST_MIPMAP_NEAREST,
            gl.GL_LINEAR_MIPMAP_NEAREST,
            gl.GL_NEAREST_MIPMAP_LINEAR,
            gl.GL_LINEAR_MIPMAP_LINEAR,
        ):
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        texture_ids.append(texture_id.value)
    return texture_ids


def material_info(
    model: dict[str, Any], materials: list[dict[str, Any]], material_index: int | None
) -> tuple[tuple[float, float, float, float], int | None, int]:
    if material_index is None or material_index >= len(materials):
        return (0.8, 0.8, 0.8, 1.0), None, 0
    material = materials[material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    color = tuple(pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0]))
    texture_info = pbr.get("baseColorTexture", {})
    texture = texture_info.get("index")
    texcoord = texture_info.get("texCoord", 0)
    return color, texture, texcoord


def collect_primitives(
    model: dict[str, Any], buffers: list[bytes]
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[float, ...], int | None]]:
    result = []
    materials = model.get("materials", [])

    def visit(node_index: int, parent: np.ndarray) -> None:
        node = model["nodes"][node_index]
        transform = parent @ node_matrix(node)
        if "mesh" in node:
            for primitive in model["meshes"][node["mesh"]].get("primitives", []):
                if primitive.get("mode", 4) != 4:
                    continue
                attributes = primitive.get("attributes", {})
                if "POSITION" not in attributes:
                    continue
                vertices = read_accessor(model, buffers, attributes["POSITION"])[:, :3]
                color, texture, texcoord = material_info(
                    model, materials, primitive.get("material")
                )
                uv = (
                    read_accessor(
                        model, buffers, attributes[f"TEXCOORD_{texcoord}"]
                    )[:, :2]
                    if f"TEXCOORD_{texcoord}" in attributes
                    else None
                )
                if "indices" in primitive:
                    indices = read_accessor(model, buffers, primitive["indices"]).ravel()
                else:
                    indices = np.arange(len(vertices))
                indices = indices.astype(np.int64)
                indices = indices[: len(indices) - len(indices) % 3].reshape(-1, 3)
                vertices = (transform[:3, :3] @ vertices.T).T + transform[:3, 3]
                result.append((vertices, indices, uv, color, texture))
        for child in node.get("children", []):
            visit(child, transform)

    scene = model.get("scenes", [{}])[model.get("scene", 0)]
    for root in scene.get("nodes", []):
        visit(root, np.eye(4, dtype=np.float32))
    return result


class GPUPrimitive:
    """Holds GPU-resident vertex/index buffers for a single primitive.

    Building interleaved VBOs (position + uv) and index/edge-index buffers once
    at load time lets on_draw issue a couple of glDrawElements calls instead of
    a Python-level glVertex3f/glTexCoord2f loop per vertex every frame.
    """

    VERTEX_STRIDE = 5 * 4  # 3 floats position + 2 floats uv, all float32

    def __init__(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        uv: np.ndarray | None,
        color: tuple[float, ...],
        texture_index: int | None,
    ):
        self.color = color
        self.texture_index = texture_index
        self.has_uv = uv is not None
        self.index_count = indices.size

        interleaved = np.zeros((len(vertices), 5), dtype=np.float32)
        interleaved[:, :3] = vertices
        if uv is not None:
            interleaved[:, 3:5] = uv

        vbo = gl.GLuint()
        gl.glGenBuffers(1, vbo)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            interleaved.nbytes,
            interleaved.tobytes(),
            gl.GL_STATIC_DRAW,
        )
        self.vbo = vbo

        tri_indices = indices.astype(np.uint32)
        ebo = gl.GLuint()
        gl.glGenBuffers(1, ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            tri_indices.nbytes,
            tri_indices.tobytes(),
            gl.GL_STATIC_DRAW,
        )
        self.ebo = ebo

        edges = np.concatenate(
            [tri_indices[:, [0, 1, 1, 2, 2, 0]]], axis=0
        ).ravel().astype(np.uint32)
        self.edge_count = edges.size
        edge_ebo = gl.GLuint()
        gl.glGenBuffers(1, edge_ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, edge_ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            edges.nbytes,
            edges.tobytes(),
            gl.GL_STATIC_DRAW,
        )
        self.edge_ebo = edge_ebo

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    def bind_vertex_arrays(self) -> None:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glVertexPointer(3, gl.GL_FLOAT, self.VERTEX_STRIDE, 0)
        gl.glTexCoordPointer(
            2, gl.GL_FLOAT, self.VERTEX_STRIDE, ctypes.c_void_p(3 * 4)
        )

    def draw_triangles(self) -> None:
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glDrawElements(
            gl.GL_TRIANGLES, self.index_count, gl.GL_UNSIGNED_INT, 0
        )

    def draw_edges(self) -> None:
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.edge_ebo)
        gl.glDrawElements(gl.GL_LINES, self.edge_count, gl.GL_UNSIGNED_INT, 0)


class Viewer(pyglet.window.Window):
    def __init__(self, model_path: Path):
        config = gl.Config(double_buffer=True, depth_size=24, major_version=2, minor_version=1)
        super().__init__(
            width=1280,
            height=800,
            caption=f"GLTF Viewer - {model_path.name}",
            resizable=True,
            config=config,
        )
        self.model_path = model_path
        self.model = json.loads(model_path.read_text(encoding="utf-8"))
        buffers = load_buffers(model_path, self.model)
        primitives = collect_primitives(self.model, buffers)
        self.textures = create_textures(model_path, self.model, buffers)
        points = np.concatenate([item[0] for item in primitives])
        self.primitives = [
            GPUPrimitive(vertices, indices, uv, color, texture)
            for vertices, indices, uv, color, texture in primitives
        ]
        self.center = points.mean(axis=0)
        self.radius = max(float(np.ptp(points, axis=0).max()) / 2, 0.01)
        self.distance = self.radius * 2.5
        self.azimuth = -45.0
        self.elevation = 25.0
        self.drag_button = None
        self.last_mouse = None
        self.textures_enabled = True
        self.edges_enabled = False
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glClearColor(0.08, 0.08, 0.1, 1.0)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_TEXTURE_COORD_ARRAY)

    def on_resize(self, width: int, height: int) -> None:
        gl.glViewport(0, 0, width, height)

    def on_draw(self) -> None:
        self.clear()
        aspect = self.width / max(self.height, 1)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        near = max(self.radius / 1000, 0.001)
        far = self.distance * 100
        top = near * math.tan(math.radians(45.0) / 2)
        right = top * aspect
        gl.glFrustum(-right, right, -top, top, near, far)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glTranslatef(0, 0, -self.distance)
        gl.glRotatef(self.elevation, 1, 0, 0)
        gl.glRotatef(self.azimuth, 0, 1, 0)
        gl.glTranslatef(-self.center[0], -self.center[1], -self.center[2])

        for primitive in self.primitives:
            texture_id = (
                self.textures[primitive.texture_index]
                if (
                    self.textures_enabled
                    and primitive.texture_index is not None
                    and primitive.texture_index < len(self.textures)
                )
                else None
            )
            if texture_id is not None and primitive.has_uv:
                gl.glEnable(gl.GL_TEXTURE_2D)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            else:
                gl.glDisable(gl.GL_TEXTURE_2D)
            gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_REPLACE)
            gl.glColor4f(
                *primitive.color if texture_id is None else (1.0, 1.0, 1.0, 1.0)
            )
            primitive.bind_vertex_arrays()
            primitive.draw_triangles()

        if self.edges_enabled:
            gl.glDisable(gl.GL_TEXTURE_2D)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glColor4f(0.02, 0.02, 0.02, 1.0)
            gl.glLineWidth(1.0)
            for primitive in self.primitives:
                primitive.bind_vertex_arrays()
                primitive.draw_edges()
            gl.glEnable(gl.GL_CULL_FACE)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glDisable(gl.GL_TEXTURE_2D)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        self.drag_button = button
        self.last_mouse = (x, y)

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        self.drag_button = None
        self.last_mouse = None

    def on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
    ) -> None:
        if self.last_mouse is None:
            return
        if self.drag_button == pyglet.window.mouse.LEFT:
            self.azimuth += dx * 0.5
            self.elevation = max(-89, min(89, self.elevation + dy * 0.5))
        elif self.drag_button == pyglet.window.mouse.RIGHT:
            scale = self.radius * 0.002
            self.center += np.array([-dx * scale, dy * scale, 0])
        self.last_mouse = (x, y)

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self.distance *= math.pow(0.85, scroll_y)
        self.distance = max(self.radius * 0.05, min(self.radius * 100, self.distance))

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == pyglet.window.key.T:
            self.textures_enabled = not self.textures_enabled
        elif symbol == pyglet.window.key.E:
            self.edges_enabled = not self.edges_enabled
        else:
            return
        textures = "textures on" if self.textures_enabled else "textures off"
        edges = "edges on" if self.edges_enabled else "edges off"
        self.set_caption(f"GLTF Viewer - {self.model_path.name} ({textures}, {edges})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("merged.gltf"),
        help="GLTF file to display",
    )
    args = parser.parse_args()
    Viewer(args.model)
    pyglet.app.run()


if __name__ == "__main__":
    main()
