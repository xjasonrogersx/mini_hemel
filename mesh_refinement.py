"""Pure mesh-to-image helpers used by the view-guided refinement pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def project_points(
    points: np.ndarray,
    camera_matrix: Any,
    capture_size: dict[str, Any],
    yfov_degrees: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space points into image pixels using a pyrender camera pose."""
    points = np.asarray(points, dtype=np.float64)
    pose = np.asarray(camera_matrix, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("camera matrix must be 4x4")
    width = int(capture_size["width"])
    height = int(capture_size["height"])
    if width <= 0 or height <= 0:
        raise ValueError("capture dimensions must be positive")
    camera_points = (np.linalg.inv(pose) @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    depth = -camera_points[:, 2]
    focal_y = height / (2.0 * np.tan(np.deg2rad(yfov_degrees) / 2.0))
    focal_x = focal_y / (width / height)
    pixels = np.column_stack((
        focal_x * camera_points[:, 0] / np.maximum(depth, 1e-12) + width / 2.0,
        height / 2.0 - focal_y * camera_points[:, 1] / np.maximum(depth, 1e-12),
    ))
    return pixels, depth


def project_scene_faces(
    scene: Any,
    camera_pose: dict[str, Any],
    capture_size: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return face-centroid projections for every dumped mesh in a scene."""
    records: list[dict[str, Any]] = []
    meshes = scene.dump(concatenate=False)
    for mesh_index, mesh in enumerate(meshes):
        faces = np.asarray(mesh.faces, dtype=np.int64)
        triangle_points = np.asarray(mesh.vertices)[faces]
        pixels, depths = project_points(
            triangle_points.reshape(-1, 3), camera_pose["matrix"], capture_size
        )
        pixels = pixels.reshape(-1, 3, 2)
        depths = depths.reshape(-1, 3)
        for face_index, (triangle, triangle_depths) in enumerate(zip(pixels, depths)):
            pixel = triangle.mean(axis=0)
            depth = float(triangle_depths.mean())
            x, y = pixel
            triangle_min = triangle.min(axis=0)
            triangle_max = triangle.max(axis=0)
            records.append({
                "mesh_index": mesh_index,
                "face_index": face_index,
                "pixel": [float(x), float(y)],
                "triangle_pixels": triangle.tolist(),
                "depth": depth,
                "in_front": bool(np.all(triangle_depths > 0)),
                "in_frame": bool(
                    triangle_max[0] >= 0 and triangle_min[0] < capture_size["width"]
                    and triangle_max[1] >= 0 and triangle_min[1] < capture_size["height"]
                ),
            })
    return records


def _triangle_samples(triangle: np.ndarray) -> np.ndarray:
    """Return centroid and edge/interior samples for a projected triangle."""
    first, second, third = triangle
    return np.asarray([
        (first + second + third) / 3.0,
        first * 0.6 + second * 0.2 + third * 0.2,
        first * 0.2 + second * 0.6 + third * 0.2,
        first * 0.2 + second * 0.2 + third * 0.6,
        first * 0.5 + second * 0.5,
        second * 0.5 + third * 0.5,
        third * 0.5 + first * 0.5,
    ])


def sample_mask(mask_path: Path, face_records: list[dict[str, Any]], capture_size: dict[str, Any]) -> dict[str, Any]:
    """Associate projected triangle samples with nonzero pixels in a mask image."""
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L"))
    width = int(capture_size["width"])
    height = int(capture_size["height"])
    if mask.shape[1] != width or mask.shape[0] != height:
        raise ValueError(f"mask dimensions {mask.shape[1]}x{mask.shape[0]} do not match {width}x{height}")
    selected = []
    coverage_by_face = []
    for record in face_records:
        if not record["in_front"] or not record["in_frame"]:
            continue
        triangle = np.asarray(record.get("triangle_pixels", [record["pixel"]] * 3))
        samples = _triangle_samples(triangle)
        in_frame = (
            (samples[:, 0] >= 0) & (samples[:, 0] < width) &
            (samples[:, 1] >= 0) & (samples[:, 1] < height)
        )
        values = np.zeros(len(samples), dtype=bool)
        valid_samples = samples[in_frame].astype(int)
        values[in_frame] = mask[valid_samples[:, 1], valid_samples[:, 0]] > 0
        coverage = float(values.mean())
        coverage_by_face.append({
            "mesh_index": record["mesh_index"],
            "face_index": record["face_index"],
            "coverage": coverage,
        })
        if coverage >= 0.25:
            selected.append([record["mesh_index"], record["face_index"]])
    return {
        "file": mask_path.name,
        "selected_faces": selected,
        "selected_count": len(selected),
        "coverage_threshold": 0.25,
        "coverage_by_face": coverage_by_face,
    }
