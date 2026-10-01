"""Pure mesh-to-image helpers used by the view-guided refinement pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


VISIBILITY_SAMPLE_DIVISIONS = 8
VISIBILITY_MIN_IN_FRAME_RATIO = 0.80
VISIBILITY_MIN_VISIBLE_RATIO = 0.80
VISIBILITY_DEPTH_TOLERANCE = 0.005


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
        camera_position = np.asarray(camera_pose["matrix"], dtype=float)[:3, 3]
        for face_index, (world_triangle, triangle, triangle_depths) in enumerate(
            zip(triangle_points, pixels, depths)
        ):
            pixel = triangle.mean(axis=0)
            depth = float(triangle_depths.mean())
            x, y = pixel
            triangle_min = triangle.min(axis=0)
            triangle_max = triangle.max(axis=0)
            normal = np.cross(world_triangle[1] - world_triangle[0], world_triangle[2] - world_triangle[0])
            front_facing = bool(np.dot(normal, camera_position - world_triangle.mean(axis=0)) > 0)
            records.append({
                "mesh_index": mesh_index,
                "face_index": face_index,
                "pixel": [float(x), float(y)],
                "triangle_pixels": triangle.tolist(),
                "triangle_depths": triangle_depths.tolist(),
                "depth": depth,
                "in_front": bool(np.all(triangle_depths > 0)),
                "front_facing": front_facing,
                "in_frame": bool(
                    triangle_max[0] >= 0 and triangle_min[0] < capture_size["width"]
                    and triangle_max[1] >= 0 and triangle_min[1] < capture_size["height"]
                ),
            })
    return records


def visible_face_keys(face_records: list[dict[str, Any]], capture_size: dict[str, Any]) -> set[tuple[int, int]]:
    """Keep front-facing faces that are mostly visible at projected samples."""
    width = int(capture_size["width"])
    height = int(capture_size["height"])
    nearest: dict[tuple[int, int], float] = {}
    sample_data = []
    for record in face_records:
        if not record["in_front"] or not record["in_frame"] or not record.get("front_facing", True):
            continue
        triangle = np.asarray(record["triangle_pixels"], dtype=float)
        triangle_depths = np.asarray(record["triangle_depths"], dtype=float)
        samples = _triangle_samples(triangle)
        sample_depth = _triangle_sample_weights() @ triangle_depths
        key = (int(record["mesh_index"]), int(record["face_index"]))
        projected_samples = []
        for sample, depth in zip(samples, sample_depth):
            x, y = np.rint(sample).astype(int)
            if not (0 <= x < width and 0 <= y < height):
                continue
            pixel_key = (x, y)
            projected_samples.append((pixel_key, float(depth)))
            nearest[pixel_key] = min(nearest.get(pixel_key, float("inf")), float(depth))
        if projected_samples:
            sample_data.append((key, projected_samples))
    visible = set()
    for key, projected_samples in sample_data:
        total_samples = len(_triangle_sample_weights())
        in_frame_ratio = len(projected_samples) / total_samples
        if in_frame_ratio < VISIBILITY_MIN_IN_FRAME_RATIO:
            continue
        visible_samples = sum(
            depth <= nearest[pixel_key] + max(VISIBILITY_DEPTH_TOLERANCE, depth * VISIBILITY_DEPTH_TOLERANCE)
            for pixel_key, depth in projected_samples
        )
        if visible_samples / len(projected_samples) >= VISIBILITY_MIN_VISIBLE_RATIO:
            visible.add(key)
    return visible


def _triangle_samples(triangle: np.ndarray) -> np.ndarray:
    """Return a dense barycentric grid across the complete projected triangle."""
    return _triangle_sample_weights() @ np.asarray(triangle, dtype=float)


def _triangle_sample_weights() -> np.ndarray:
    divisions = VISIBILITY_SAMPLE_DIVISIONS
    weights = []
    for first in range(divisions + 1):
        for second in range(divisions + 1 - first):
            third = divisions - first - second
            weights.append([first / divisions, second / divisions, third / divisions])
    return np.asarray(weights, dtype=float)


def sample_mask(mask_path: Path, face_records: list[dict[str, Any]], capture_size: dict[str, Any]) -> dict[str, Any]:
    """Associate projected triangle samples with a mask at its native resolution."""
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L"))
    capture_width = int(capture_size["width"])
    capture_height = int(capture_size["height"])
    height, width = mask.shape[:2]
    scale_x = width / capture_width
    scale_y = height / capture_height
    selected = []
    coverage_by_face = []
    for record in face_records:
        if not record["in_front"] or not record["in_frame"]:
            continue
        triangle_values = record.get("triangle_pixels")
        if triangle_values is None:
            triangle_values = [record["pixel"]] * 3
        triangle = np.asarray(triangle_values, dtype=float)
        samples = _triangle_samples(triangle)
        samples[:, 0] *= scale_x
        samples[:, 1] *= scale_y
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
        "mask_dimensions": {"width": width, "height": height},
        "projection_dimensions": {"width": capture_width, "height": capture_height},
        "pixel_scale": {"x": scale_x, "y": scale_y},
        "selected_faces": selected,
        "selected_count": len(selected),
        "coverage_threshold": 0.25,
        "coverage_by_face": coverage_by_face,
    }
