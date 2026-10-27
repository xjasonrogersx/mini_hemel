"""Building-only view-guided geometry tightening."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


def project_points(points: np.ndarray, camera_matrix: Any, capture_size: dict[str, Any], yfov_degrees: float = 45.0) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    pose = np.asarray(camera_matrix, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("camera matrix must be 4x4")
    width, height = int(capture_size["width"]), int(capture_size["height"])
    camera_points = (np.linalg.inv(pose) @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    depth = -camera_points[:, 2]
    focal = height / (2.0 * np.tan(np.deg2rad(yfov_degrees) / 2.0))
    pixels = np.column_stack((
        focal * camera_points[:, 0] / np.maximum(depth, 1e-12) + width / 2.0,
        height / 2.0 - focal * camera_points[:, 1] / np.maximum(depth, 1e-12),
    ))
    return pixels, depth


def _load_building_mask(mask_paths: list[Path], capture_size: dict[str, Any]) -> np.ndarray:
    width, height = int(capture_size["width"]), int(capture_size["height"])
    combined = np.zeros((height, width), dtype=np.uint8)
    for path in mask_paths:
        with Image.open(path) as image:
            source = np.asarray(image.convert("L"))
        combined |= (cv2.resize(source, (width, height), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    return combined


def _sample_mask(mask: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    coordinates = np.rint(pixels).astype(np.int64)
    valid = ((coordinates[:, 0] >= 0) & (coordinates[:, 0] < width) & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < height))
    values = np.zeros(len(coordinates), dtype=bool)
    values[valid] = mask[coordinates[valid, 1], coordinates[valid, 0]] > 0
    return values


def _fit_plane_ransac(
    points: np.ndarray,
    threshold: float = 0.08,
    iterations: int = 64,
    prefer_axis_aligned: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return np.zeros(3), np.zeros(3), 0
    rng = np.random.default_rng(0)
    best_normal = np.zeros(3)
    best_center = points.mean(axis=0)
    best_inliers = np.zeros(len(points), dtype=bool)
    best_score = -1.0
    for _ in range(min(iterations, max(1, len(points) * 2))):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        distances = np.abs((points - sample[0]) @ normal)
        inliers = distances <= threshold
        axis_alignment = max(abs(normal[2]), 1.0 - abs(normal[2]))
        score = float(inliers.sum()) * (1.0 + 0.75 * (axis_alignment - 0.5)) if prefer_axis_aligned else float(inliers.sum())
        if score > best_score:
            best_normal, best_center, best_inliers = normal, points[inliers].mean(axis=0), inliers
            best_score = score
    if best_inliers.sum() >= 3:
        _, _, basis = np.linalg.svd(points[best_inliers] - best_center, full_matrices=False)
        best_normal = basis[-1]
    return best_normal, best_center, int(best_inliers.sum())


def tighten_building_meshes(
    meshes: list[Any],
    mask_paths: list[Path],
    camera_matrix: Any,
    capture_size: dict[str, Any],
    aggressiveness: float = 0.65,
    prefer_axis_aligned: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """Tighten visible building surfaces while retaining original topology/materials."""
    aggressiveness = float(np.clip(aggressiveness, 0.0, 2.0))
    pull_strength = aggressiveness
    max_displacement = 0.25 * (0.5 + aggressiveness / 0.65)
    mask = _load_building_mask(mask_paths, capture_size)
    stats: dict[str, Any] = {
        "mask_files": [path.name for path in mask_paths],
        "aggressiveness": aggressiveness,
        "prefer_axis_aligned": prefer_axis_aligned,
        "meshes_changed": 0,
        "building_faces": 0,
        "planes_fitted": 0,
        "vertices_moved": 0,
        "ransac_inliers": 0,
        "max_displacement": 0.0,
    }
    output = []
    for mesh_index, mesh in enumerate(meshes):
        mesh = mesh.copy()
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if not len(vertices) or not len(faces):
            output.append(mesh)
            continue
        pixels, depths = project_points(vertices, camera_matrix, capture_size)
        vertex_building = _sample_mask(mask, pixels) & (depths > 0)
        face_building = vertex_building[faces].sum(axis=1) >= 2
        selected_indices = np.unique(faces[face_building]) if np.any(face_building) else np.empty(0, dtype=np.int64)
        if len(selected_indices) < 3:
            output.append(mesh)
            continue
        selected_points = vertices[selected_indices]
        normal, center, inlier_count = _fit_plane_ransac(
            selected_points, prefer_axis_aligned=prefer_axis_aligned
        )
        if inlier_count < 3:
            output.append(mesh)
            continue
        distances = (selected_points - center) @ normal
        displacement = np.clip(-distances * pull_strength, -max_displacement, max_displacement)
        updated = selected_points + displacement[:, None] * normal
        moved = np.linalg.norm(updated - selected_points, axis=1) > 1e-8
        vertices[selected_indices] = updated
        mesh.vertices = vertices
        stats["meshes_changed"] += 1
        stats["building_faces"] += int(face_building.sum())
        stats["planes_fitted"] += 1
        stats["vertices_moved"] += int(moved.sum())
        stats["ransac_inliers"] += inlier_count
        stats["max_displacement"] = max(stats["max_displacement"], float(np.max(np.linalg.norm(updated - selected_points, axis=1))))
        LOGGER.info("Building mesh tightened: mesh=%d faces=%d vertices=%d inliers=%d", mesh_index, int(face_building.sum()), len(selected_indices), inlier_count)
        output.append(mesh)
    LOGGER.info("Building geometry complete: %s", stats)
    return output, stats
