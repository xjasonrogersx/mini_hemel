"""View-projected texture application for mesh scenes."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

from mesh_refinement import project_points, project_scene_faces, visible_face_keys


def apply_view_projected_texture(
    loaded: trimesh.Scene,
    texture_path: Path,
    camera_pose: dict[str, Any],
    capture_size: dict[str, Any],
) -> tuple[list[Any], int]:
    """Project a full-resolution image onto visible scene faces.

    Occluded faces retain their original mesh material. Visible faces receive
    camera-projected UVs and the supplied source image as their base texture.
    """
    texture = Image.open(texture_path).convert("RGB")
    projection_records = project_scene_faces(loaded, camera_pose, capture_size)
    visible_keys = visible_face_keys(projection_records, capture_size)
    meshes = loaded.dump(concatenate=False)
    projected_meshes = []
    for mesh_index, mesh in enumerate(meshes):
        visible_faces = [
            face_index for face_index in range(len(mesh.faces))
            if (mesh_index, face_index) in visible_keys
        ]
        occluded_faces = [
            face_index for face_index in range(len(mesh.faces))
            if (mesh_index, face_index) not in visible_keys
        ]
        if occluded_faces:
            projected_meshes.append(mesh.submesh([occluded_faces], append=True, repair=False))
        if not visible_faces:
            continue
        visible_mesh = mesh.submesh([visible_faces], append=True, repair=False)
        pixels, _depths = project_points(
            np.asarray(visible_mesh.vertices), camera_pose["matrix"], capture_size
        )
        visible_mesh.visual.uv = np.column_stack((
            np.clip(pixels[:, 0] / float(capture_size["width"]), 0.0, 1.0),
            np.clip(1.0 - pixels[:, 1] / float(capture_size["height"]), 0.0, 1.0),
        ))
        material = getattr(visible_mesh.visual, "material", None)
        if material is not None and hasattr(material, "baseColorTexture"):
            visible_mesh.visual.material = copy.deepcopy(material)
            visible_mesh.visual.material.baseColorTexture = texture
        projected_meshes.append(visible_mesh)
    return projected_meshes, len(visible_keys)
