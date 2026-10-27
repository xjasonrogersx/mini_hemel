"""Global mesh refinement by grid quantization and redundant-vertex removal."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def refine_meshes(meshes: list[Any], grid_size_m: float) -> tuple[list[Any], dict[str, int | float]]:
    """Quantize vertices, remove degenerate faces, and deduplicate triangles globally."""
    grid_size_m = float(np.clip(grid_size_m, 0.0, 5.0))
    stats: dict[str, int | float] = {
        "grid_size_m": grid_size_m,
        "meshes_changed": 0,
        "vertices_before": 0,
        "vertices_after_quantization": 0,
        "vertices_removed": 0,
        "triangles_before": 0,
        "degenerate_triangles_removed": 0,
        "overlapping_triangles_removed": 0,
        "triangles_after": 0,
    }
    output = []
    seen_triangles: set[tuple[tuple[float, float, float], ...]] = set()
    for mesh_index, source_mesh in enumerate(meshes):
        mesh = source_mesh.copy()
        before = len(mesh.vertices)
        stats["vertices_before"] += before
        if grid_size_m > 0 and before:
            mesh.vertices = np.round(np.asarray(mesh.vertices) / grid_size_m) * grid_size_m
        stats["vertices_after_quantization"] += len(mesh.vertices)
        mesh.merge_vertices()
        after = len(mesh.vertices)
        stats["vertices_removed"] += before - after
        faces = np.asarray(mesh.faces, dtype=np.int64)
        stats["triangles_before"] += len(faces)
        keep_faces = []
        keep_indices = []
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        for face_index, face in enumerate(faces):
            points = vertices[face]
            area_vector = np.cross(points[1] - points[0], points[2] - points[0])
            if np.linalg.norm(area_vector) <= 1e-10:
                stats["degenerate_triangles_removed"] += 1
                continue
            triangle_key = tuple(sorted(tuple(float(value) for value in point) for point in points))
            if triangle_key in seen_triangles:
                stats["overlapping_triangles_removed"] += 1
                continue
            seen_triangles.add(triangle_key)
            keep_faces.append(face)
            keep_indices.append(face_index)
        mesh.update_faces(np.asarray(keep_indices, dtype=np.int64))
        stats["triangles_after"] += len(mesh.faces)
        if len(keep_faces) != len(faces):
            stats["meshes_changed"] += 1
        if after != before:
            stats["meshes_changed"] += 1
            LOGGER.info("Global refinement mesh=%d vertices=%d->%d", mesh_index, before, after)
        if len(keep_faces) != len(faces):
            LOGGER.info(
                "Global refinement triangles mesh=%d %d->%d",
                mesh_index,
                len(faces),
                len(keep_faces),
            )
        output.append(mesh)
    LOGGER.info("Global refinement complete: %s", stats)
    return output, stats
