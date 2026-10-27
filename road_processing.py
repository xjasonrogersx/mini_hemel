"""Projection and flatten-and-retile helpers for segmented road surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import time

import mapbox_earcut
import numpy as np
import cv2
import trimesh
from PIL import Image


LOGGER = logging.getLogger(__name__)


def project_points(
    points: np.ndarray,
    camera_matrix: Any,
    capture_size: dict[str, Any],
    yfov_degrees: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    pose = np.asarray(camera_matrix, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("camera matrix must be 4x4")
    width = int(capture_size["width"])
    height = int(capture_size["height"])
    camera_points = (np.linalg.inv(pose) @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    depth = -camera_points[:, 2]
    focal = height / (2.0 * np.tan(np.deg2rad(yfov_degrees) / 2.0))
    pixels = np.column_stack((
        focal * camera_points[:, 0] / np.maximum(depth, 1e-12) + width / 2.0,
        height / 2.0 - focal * camera_points[:, 1] / np.maximum(depth, 1e-12),
    ))
    return pixels, depth


def _load_mask_triangles(mask_paths: list[Path], capture_size: dict[str, Any]) -> list[np.ndarray]:
    """Convert raster mask regions into capture-space triangles."""
    started = time.perf_counter()
    triangles: list[np.ndarray] = []
    for mask_path in mask_paths:
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"))
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        LOGGER.info(
            "Road mask loaded: file=%s dimensions=%sx%s contours=%d foreground_pixels=%d",
            mask_path.name, mask.shape[1], mask.shape[0], len(contours), int(np.count_nonzero(mask)),
        )
        scale = np.array([capture_size["width"] / mask.shape[1], capture_size["height"] / mask.shape[0]])
        for contour in contours:
            if cv2.contourArea(contour) < 1.0:
                continue
            polygon = contour[:, 0, :].astype(np.float64) * scale
            indices = mapbox_earcut.triangulate_float64(polygon, np.asarray([len(polygon)], dtype=np.uint32))
            if len(indices):
                triangles.extend(polygon[np.asarray(indices, dtype=np.int64)].reshape(-1, 3, 2))
    LOGGER.info(
        "Road mask triangulated: files=%d contour_triangles=%d elapsed=%.2fs",
        len(mask_paths), len(triangles), time.perf_counter() - started,
    )
    return triangles


def _load_mask_raster(mask_paths: list[Path], capture_size: dict[str, Any]) -> np.ndarray:
    """Combine mask files into a capture-resolution binary raster."""
    width = int(capture_size["width"])
    height = int(capture_size["height"])
    combined = np.zeros((height, width), dtype=np.uint8)
    for mask_path in mask_paths:
        with Image.open(mask_path) as image:
            source = np.asarray(image.convert("L"))
        resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_NEAREST)
        combined |= (resized > 0).astype(np.uint8)
    return combined


def _sample_mask_raster(raster: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    height, width = raster.shape
    coordinates = np.rint(pixels).astype(np.int64)
    valid = (
        (coordinates[:, 0] >= 0) & (coordinates[:, 0] < width)
        & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < height)
    )
    values = np.zeros(len(coordinates), dtype=bool)
    values[valid] = raster[coordinates[valid, 1], coordinates[valid, 0]] > 0
    return values


def _clip_convex_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    clip = np.asarray(clip, dtype=np.float64)
    if _cross_2d(clip[1] - clip[0], clip[2] - clip[0]) < 0:
        clip = clip[::-1]
    result = [point for point in np.asarray(subject, dtype=np.float64)]
    for index, endpoint in enumerate(clip):
        start = clip[index - 1]
        if not result:
            break
        output: list[np.ndarray] = []
        for current, following in zip(result, result[1:] + result[:1]):
            current_inside = _cross_2d(endpoint - start, current - start) >= -1e-7
            following_inside = _cross_2d(endpoint - start, following - start) >= -1e-7
            if current_inside != following_inside:
                direction = following - current
                denominator = _cross_2d(endpoint - start, direction)
                if abs(denominator) > 1e-12:
                    distance = _cross_2d(endpoint - start, start - current) / denominator
                    output.append(current + distance * direction)
            if following_inside:
                output.append(following)
        result = output
    return np.asarray(result, dtype=np.float64)


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _barycentric_to_world(points: np.ndarray, triangle_pixels: np.ndarray, triangle_world: np.ndarray) -> np.ndarray:
    first, second, third = triangle_pixels
    vectors = np.column_stack((second - first, third - first))
    weights = np.linalg.lstsq(vectors, (points - first).T, rcond=None)[0].T
    return triangle_world[0] + weights[:, 0, None] * (triangle_world[1] - triangle_world[0]) + weights[:, 1, None] * (triangle_world[2] - triangle_world[0])


def _build_triangle_index(
    triangles: list[np.ndarray], cell_size: int = 32,
) -> tuple[dict[tuple[int, int], list[int]], np.ndarray]:
    """Build a coarse pixel-grid index for contour triangles."""
    bounds = np.asarray(
        [[triangle[:, 0].min(), triangle[:, 1].min(), triangle[:, 0].max(), triangle[:, 1].max()]
         for triangle in triangles],
        dtype=np.float64,
    ) if triangles else np.empty((0, 4), dtype=np.float64)
    index: dict[tuple[int, int], list[int]] = {}
    for triangle_index, (min_x, min_y, max_x, max_y) in enumerate(bounds):
        for cell_y in range(max(0, int(min_y) // cell_size), max(0, int(max_y) // cell_size) + 1):
            for cell_x in range(max(0, int(min_x) // cell_size), max(0, int(max_x) // cell_size) + 1):
                index.setdefault((cell_x, cell_y), []).append(triangle_index)
    return index, bounds


def _candidate_triangle_indices(
    bounds: np.ndarray,
    index: dict[tuple[int, int], list[int]],
    pixel_triangle: np.ndarray,
    cell_size: int = 32,
) -> list[int]:
    min_x, min_y = np.floor(pixel_triangle.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(pixel_triangle.max(axis=0)).astype(int)
    candidates: set[int] = set()
    for cell_y in range(max(0, min_y // cell_size), max(0, max_y // cell_size) + 1):
        for cell_x in range(max(0, min_x // cell_size), max(0, max_x // cell_size) + 1):
            candidates.update(index.get((cell_x, cell_y), ()))
    if not candidates:
        return []
    face_min = pixel_triangle.min(axis=0)
    face_max = pixel_triangle.max(axis=0)
    return [
        triangle_index for triangle_index in candidates
        if bounds[triangle_index, 2] >= face_min[0]
        and bounds[triangle_index, 0] <= face_max[0]
        and bounds[triangle_index, 3] >= face_min[1]
        and bounds[triangle_index, 1] <= face_max[1]
    ]


def _flatten_patch(vertices: np.ndarray) -> np.ndarray:
    center = vertices.mean(axis=0)
    _, _, basis = np.linalg.svd(vertices - center, full_matrices=False)
    normal = basis[-1]
    return vertices - ((vertices - center) @ normal)[:, None] * normal


def flatten_and_retile_meshes(
    meshes: list[Any],
    mask_groups: dict[str, list[Path]],
    camera_matrix: Any,
    capture_size: dict[str, Any],
) -> tuple[list[Any], dict[str, int]]:
    """Clip mesh faces against each semantic mask and create separate flat patches."""
    started = time.perf_counter()
    LOGGER.info(
        "Road processing started: meshes=%d categories=%s capture=%sx%s",
        len(meshes), {category: len(paths) for category, paths in mask_groups.items()},
        capture_size.get("width"), capture_size.get("height"),
    )
    patches_by_category: dict[str, list[Any]] = {}
    faces_removed = 0
    triangles_added = 0
    category_stats: dict[str, dict[str, int]] = {}
    mask_triangles = {
        category: _load_mask_triangles(paths, capture_size)
        for category, paths in mask_groups.items()
        if paths
    }
    mask_rasters = {
        category: _load_mask_raster(paths, capture_size)
        for category, paths in mask_groups.items()
        if paths
    }
    triangle_indexes = {
        category: _build_triangle_index(triangles)
        for category, triangles in mask_triangles.items()
    }
    LOGGER.info(
        "Road clipping index ready: categories=%s index_cells=%s",
        {category: len(triangles) for category, triangles in mask_triangles.items()},
        {category: len(index_bounds[0]) for category, index_bounds in triangle_indexes.items()},
    )
    for mesh_index, mesh in enumerate(meshes):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if not len(faces):
            continue
        mesh_started = time.perf_counter()
        LOGGER.info("Road mesh started: mesh=%d vertices=%d faces=%d", mesh_index, len(vertices), len(faces))
        projected, depths = project_points(vertices, camera_matrix, capture_size)
        kept_faces: list[np.ndarray] = []
        intersection_tests = 0
        candidate_tests = 0
        boundary_faces = 0
        direct_faces = 0
        for face_index, face in enumerate(faces):
            if face_index and face_index % 256 == 0:
                time.sleep(0)
            if face_index and face_index % 2048 == 0:
                LOGGER.info(
                    "Road mesh progress: mesh=%d faces=%d/%d candidates=%d",
                    mesh_index, face_index, len(faces), candidate_tests,
                )
            world_triangle = vertices[face]
            pixel_triangle = projected[face]
            if np.any(depths[face] <= 0):
                kept_faces.append(face)
                continue
            face_patches = 0
            covered_area = 0.0
            face_categories: set[str] = set()
            face_area = abs(_cross_2d(pixel_triangle[1] - pixel_triangle[0], pixel_triangle[2] - pixel_triangle[0])) * 0.5
            for category, contour_triangles in mask_triangles.items():
                category_stats.setdefault(category, {"triangles": 0, "faces": 0})
                sample_pixels = np.vstack((pixel_triangle, pixel_triangle.mean(axis=0)))
                sample_values = _sample_mask_raster(mask_rasters[category], sample_pixels)
                if not np.any(sample_values):
                    continue
                if np.all(sample_values):
                    flattened = _flatten_patch(world_triangle)
                    patches_by_category.setdefault(category, []).append(
                        trimesh.Trimesh(vertices=flattened, faces=np.asarray([[0, 1, 2]], dtype=np.int64), process=False)
                    )
                    triangles_added += 1
                    category_stats[category]["triangles"] += 1
                    category_stats[category]["faces"] += 1
                    face_patches += 1
                    covered_area += face_area
                    direct_faces += 1
                    continue
                boundary_faces += 1
                category_index, category_bounds = triangle_indexes[category]
                candidates = _candidate_triangle_indices(category_bounds, category_index, pixel_triangle)
                candidate_tests += len(candidates)
                intersection_tests += len(contour_triangles)
                for triangle_index in candidates:
                    contour_triangle = contour_triangles[triangle_index]
                    clipped = _clip_convex_polygon(pixel_triangle, contour_triangle)
                    if len(clipped) < 3 or abs(cv2.contourArea(clipped.astype(np.float32))) < 0.01:
                        continue
                    covered_area += abs(cv2.contourArea(clipped.astype(np.float32)))
                    face_categories.add(category)
                    world_polygon = _barycentric_to_world(clipped, pixel_triangle, world_triangle)
                    patch_faces = np.asarray(mapbox_earcut.triangulate_float64(clipped, np.asarray([len(clipped)], dtype=np.uint32)), dtype=np.int64).reshape(-1, 3)
                    if not len(patch_faces):
                        continue
                    patches_by_category.setdefault(category, []).append(
                        trimesh.Trimesh(vertices=_flatten_patch(world_polygon), faces=patch_faces, process=False)
                    )
                    added = len(patch_faces)
                    triangles_added += added
                    category_stats[category]["triangles"] += added
                    face_patches += added
            if face_patches:
                if covered_area < face_area * 0.98:
                    kept_faces.append(face)
                else:
                    faces_removed += 1
                for category in face_categories:
                    category_stats[category]["faces"] += 1
            else:
                kept_faces.append(face)
        mesh.faces = np.asarray(kept_faces, dtype=np.int64).reshape(-1, 3)
        LOGGER.info(
            "Road mesh finished: mesh=%d kept_faces=%d removed_faces=%d direct_faces=%d boundary_faces=%d candidate_tests=%d full_scan_tests=%d reduction=%.2f%% elapsed=%.2fs",
            mesh_index, len(mesh.faces), len(faces) - len(mesh.faces), direct_faces, boundary_faces,
            candidate_tests, intersection_tests, 100.0 * (1.0 - candidate_tests / max(intersection_tests, 1)),
            time.perf_counter() - mesh_started,
        )
    patches = [trimesh.util.concatenate(category_patches) for category_patches in patches_by_category.values()]
    stats = {
        "triangles_removed": faces_removed,
        "triangles_added": triangles_added,
        "categories": category_stats,
    }
    LOGGER.info(
        "Road processing finished: output_meshes=%d patches=%d stats=%s elapsed=%.2fs",
        len(meshes) + len(patches), len(patches), stats, time.perf_counter() - started,
    )
    return meshes + patches, stats
