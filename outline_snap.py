"""Snap front-facing mesh edges toward detected image outline lines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


def _project_points(points: np.ndarray, camera_matrix: Any, capture_size: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(camera_matrix, dtype=np.float64)
    width, height = int(capture_size["width"]), int(capture_size["height"])
    camera_points = (np.linalg.inv(pose) @ np.column_stack((points, np.ones(len(points)))).T).T[:, :3]
    depth = -camera_points[:, 2]
    focal = height / (2.0 * np.tan(np.deg2rad(45.0) / 2.0))
    pixels = np.column_stack((
        focal * camera_points[:, 0] / np.maximum(depth, 1e-9) + width / 2.0,
        height / 2.0 - focal * camera_points[:, 1] / np.maximum(depth, 1e-9),
    ))
    return pixels, depth


def _load_outline_lines(lines_data: Any) -> np.ndarray:
    lines = np.asarray(lines_data, dtype=np.float64)
    if lines.ndim != 2 or lines.shape[1] != 4 or len(lines) == 0:
        raise ValueError("outline metadata contains no line coordinates; rerun M-LSD")
    return lines


def _load_outline_lines_from_image(path: Path, capture_size: dict[str, Any]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"outline image does not exist or is unreadable: {path}")
    width, height = int(capture_size["width"]), int(capture_size["height"])
    if image.shape[::-1] != (width, height):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180.0, threshold=24, minLineLength=18, maxLineGap=8)
    if lines is None:
        raise ValueError("outline map contains no detectable line segments")
    return lines[:, 0].astype(np.float64)


def _nearest_line(point: np.ndarray, lines: np.ndarray) -> tuple[np.ndarray, float]:
    starts = lines[:, :2]
    vectors = lines[:, 2:4] - starts
    lengths = np.maximum(np.sum(vectors * vectors, axis=1), 1e-9)
    factors = np.clip(np.sum((point - starts) * vectors, axis=1) / lengths, 0.0, 1.0)
    projections = starts + vectors * factors[:, None]
    distances = np.linalg.norm(projections - point[None, :], axis=1)
    index = int(np.argmin(distances))
    return projections[index], float(distances[index])


def snap_mesh_edges_to_outline(
    meshes: list[Any],
    outline_path: Path,
    camera_matrix: Any,
    capture_size: dict[str, Any],
    aggressiveness_m: float = 0.5,
    line_data: Any = None,
) -> tuple[list[Any], dict[str, Any]]:
    aggressiveness_m = float(np.clip(aggressiveness_m, 0.0, 1.0))
    lines = _load_outline_lines(line_data) if line_data is not None else _load_outline_lines_from_image(outline_path, capture_size)
    pose = np.asarray(camera_matrix, dtype=np.float64)
    camera_position = pose[:3, 3]
    stats: dict[str, Any] = {
        "outline_file": outline_path.name,
        "aggressiveness_m": aggressiveness_m,
        "line_count": int(len(lines)),
        "front_facing_edges": 0,
        "edges_snapped": 0,
        "vertices_snapped": 0,
        "max_screen_distance": 0.0,
        "rejected_out_of_range": 0,
        "rejected_out_of_view": 0,
    }
    output = []
    for mesh_index, source_mesh in enumerate(meshes):
        mesh = source_mesh.copy()
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(vertices) == 0 or len(faces) == 0:
            output.append(mesh)
            continue
        pixels, depths = _project_points(vertices, camera_matrix, capture_size)
        proposals: list[list[np.ndarray]] = [[] for _ in range(len(vertices))]
        face_normals = np.asarray(mesh.face_normals)
        visible_edges: set[tuple[int, int]] = set()
        for face_index, face in enumerate(faces):
            center = vertices[face].mean(axis=0)
            if np.dot(face_normals[face_index], camera_position - center) <= 0:
                continue
            for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                visible_edges.add(tuple(sorted((int(edge[0]), int(edge[1])))))
        stats["front_facing_edges"] = len(visible_edges)
        focal = int(capture_size["height"]) / (2.0 * np.tan(np.deg2rad(45.0) / 2.0))
        width, height = int(capture_size["width"]), int(capture_size["height"])
        for first, second in visible_edges:
            if depths[first] <= 0 or depths[second] <= 0:
                continue
            edge_pixels = pixels[[first, second]]
            if (edge_pixels[:, 0].max() < 0 or edge_pixels[:, 0].min() >= width or
                    edge_pixels[:, 1].max() < 0 or edge_pixels[:, 1].min() >= height):
                stats["rejected_out_of_view"] += 1
                continue
            midpoint = edge_pixels.mean(axis=0)
            target, screen_distance = _nearest_line(midpoint, lines)
            depth = max((depths[first] + depths[second]) * 0.5, 1e-6)
            if screen_distance > aggressiveness_m * focal / depth:
                continue
            delta = target - midpoint
            corrected_points = []
            for vertex_index in (first, second):
                camera_point = (np.linalg.inv(pose) @ np.append(vertices[vertex_index], 1.0))[:3]
                camera_point[0] += delta[0] * depths[vertex_index] / focal
                camera_point[1] -= delta[1] * depths[vertex_index] / focal
                corrected = (pose @ np.append(camera_point, 1.0))[:3]
                corrected_points.append(corrected)
            if any(np.linalg.norm(corrected - vertices[index]) > aggressiveness_m for corrected, index in zip(corrected_points, (first, second))):
                stats["rejected_out_of_range"] += 1
                continue
            for vertex_index, corrected in zip((first, second), corrected_points):
                proposals[vertex_index].append(corrected)
            stats["edges_snapped"] += 1
            stats["max_screen_distance"] = max(stats["max_screen_distance"], screen_distance)
        for vertex_index, candidates in enumerate(proposals):
            if candidates:
                vertices[vertex_index] = np.mean(candidates, axis=0)
                stats["vertices_snapped"] += 1
        mesh.vertices = vertices
        if stats["vertices_snapped"]:
            LOGGER.info("Outline snap mesh=%d edges=%d vertices=%d", mesh_index, stats["edges_snapped"], stats["vertices_snapped"])
        output.append(mesh)
    LOGGER.info("Outline snap complete: %s", stats)
    return output, stats
