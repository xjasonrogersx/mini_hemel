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
DEPTH_EDGE_PERCENTILE = 90.0


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
    focal_x = focal_y
    pixels = np.column_stack((
        focal_x * camera_points[:, 0] / np.maximum(depth, 1e-12) + width / 2.0,
        height / 2.0 - focal_y * camera_points[:, 1] / np.maximum(depth, 1e-12),
    ))
    return pixels, depth


def load_relative_depth(
    depth_path: Path,
    capture_size: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load Depth Anything's normalized map and resize it to the capture frame."""
    with Image.open(depth_path) as image:
        source = np.asarray(image.convert("I"), dtype=np.float32)
    source_min = float(np.nanmin(source))
    source_max = float(np.nanmax(source))
    if source_max > source_min:
        source = (source - source_min) / (source_max - source_min)
    else:
        source.fill(0.0)
    target_size = (int(capture_size["width"]), int(capture_size["height"]))
    resized = Image.fromarray(source, mode="F").resize(target_size, Image.Resampling.BILINEAR)
    depth = np.asarray(resized, dtype=np.float32)
    gradient_y, gradient_x = np.gradient(depth)
    gradient = np.hypot(gradient_x, gradient_y)
    metadata = {
        "file": depth_path.name,
        "source_dimensions": {"width": int(source.shape[1]), "height": int(source.shape[0])},
        "projection_dimensions": {"width": target_size[0], "height": target_size[1]},
        "value_range": {"min": float(depth.min()), "max": float(depth.max())},
        "gradient_percentile": float(np.percentile(gradient, DEPTH_EDGE_PERCENTILE)),
        "relative_only": True,
    }
    return depth, metadata


def sample_relative_depth(
    depth: np.ndarray,
    face_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach relative depth and gradient evidence to projected face records."""
    height, width = depth.shape
    gradient_y, gradient_x = np.gradient(depth)
    gradient = np.hypot(gradient_x, gradient_y)
    evidence = []
    for record in face_records:
        triangle_values = record.get("triangle_pixels")
        if triangle_values is None:
            triangle_values = [record["pixel"]] * 3
        samples = _triangle_samples(np.asarray(triangle_values, dtype=float))
        valid = (
            (samples[:, 0] >= 0) & (samples[:, 0] < width) &
            (samples[:, 1] >= 0) & (samples[:, 1] < height)
        )
        if not np.any(valid):
            continue
        pixels = samples[valid].astype(int)
        values = depth[pixels[:, 1], pixels[:, 0]]
        gradients = gradient[pixels[:, 1], pixels[:, 0]]
        evidence.append({
            "mesh_index": int(record["mesh_index"]),
            "face_index": int(record["face_index"]),
            "relative_depth_mean": float(values.mean()),
            "relative_depth_range": float(values.max() - values.min()),
            "gradient_mean": float(gradients.mean()),
            "gradient_max": float(gradients.max()),
            "sample_count": int(len(values)),
        })
    return {
        "face_count": len(evidence),
        "gradient_edge_threshold": float(np.percentile(gradient, DEPTH_EDGE_PERCENTILE)),
        "faces": evidence,
        "relative_only": True,
    }


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


def apply_conservative_geometry_refinement(
    meshes: list[Any],
    face_records: list[dict[str, Any]],
    semantic_masks: list[dict[str, Any]],
    depth_guidance: dict[str, Any],
    visible_faces: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    """Smooth bounded road and facade regions without treating relative depth as metric."""
    records_by_key = {
        (int(record["mesh_index"]), int(record["face_index"])): record
        for record in face_records
    }
    depth_by_key = {
        (int(item["mesh_index"]), int(item["face_index"])): item
        for item in depth_guidance.get("faces", [])
    }
    edge_threshold = float(depth_guidance.get("gradient_edge_threshold", float("inf")))
    selected_by_region: dict[str, set[tuple[int, int]]] = {"ground": set(), "facade": set()}
    for mask in semantic_masks:
        name = str(mask.get("file", "")).lower()
        if "road" in name or "sidewalk" in name:
            selected_by_region["ground"].update(tuple(face) for face in mask.get("selected_faces", []))
    for key in visible_faces:
        record = records_by_key.get(key)
        if record is None or key in selected_by_region["ground"]:
            continue
        triangle = np.asarray(record["triangle_pixels"], dtype=float)
        world_triangle = np.asarray(meshes[key[0]].vertices)[meshes[key[0]].faces[key[1]]]
        normal = np.cross(world_triangle[1] - world_triangle[0], world_triangle[2] - world_triangle[0])
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1e-9:
            continue
        normal /= normal_length
        depth_item = depth_by_key.get(key, {})
        if abs(float(normal[1])) < 0.35 and float(depth_item.get("gradient_max", 0.0)) <= edge_threshold:
            selected_by_region["facade"].add(key)

    changes = []
    for region, selected in selected_by_region.items():
        for mesh_index, face_index in _face_components(meshes, selected):
            mesh = meshes[mesh_index]
            faces = np.asarray(mesh.faces, dtype=np.int64)
            component_faces = [faces[index] for index in face_index]
            vertex_indices = np.unique(np.asarray(component_faces, dtype=np.int64).reshape(-1))
            if len(vertex_indices) < 3:
                continue
            original = np.asarray(mesh.vertices, dtype=np.float64).copy()
            points = original[vertex_indices]
            center = points.mean(axis=0)
            _, _, vh = np.linalg.svd(points - center, full_matrices=False)
            plane_normal = vh[-1]
            if region == "ground" and abs(float(plane_normal[1])) < 0.55:
                continue
            if region == "facade" and abs(float(plane_normal[1])) >= 0.55:
                continue
            signed_distance = (points - center) @ plane_normal
            extent = float(np.max(np.ptp(points, axis=0)))
            max_displacement = max(extent * 0.01, 1e-4)
            displacement = np.clip(-signed_distance * 0.35, -max_displacement, max_displacement)
            updated = points + displacement[:, None] * plane_normal
            mesh.vertices[vertex_indices] = updated
            moved = np.linalg.norm(updated - points, axis=1)
            moved_mask = moved > 1e-8
            if not np.any(moved_mask):
                continue
            changes.append({
                "region": region,
                "mesh_index": int(mesh_index),
                "face_count": len(face_index),
                "vertex_count": int(len(vertex_indices)),
                "moved_vertex_count": int(np.count_nonzero(moved_mask)),
                "max_displacement": float(moved.max()),
                "plane_normal": plane_normal.tolist(),
                "relative_depth_used_as_metric": False,
            })
    return changes


def smooth_masked_road(
    meshes: list[Any],
    road_masks: list[dict[str, Any]],
    iterations: int = 3,
    strength: float = 0.45,
) -> list[dict[str, Any]]:
    """Laplacian-smooth Mask2Former road vertices while preserving road boundaries."""
    selected = set()
    for mask in road_masks:
        if "mask2former_road" not in str(mask.get("file", "")).lower():
            continue
        selected.update(
            (int(face[0]), int(face[1]))
            for face in mask.get("selected_faces", [])
            if isinstance(face, (list, tuple)) and len(face) == 2
        )
    if not selected:
        return []

    changes = []
    for mesh_index, face_indices in _face_components(meshes, selected):
        mesh = meshes[mesh_index]
        faces = np.asarray(mesh.faces, dtype=np.int64)
        component_faces = np.asarray([faces[index] for index in face_indices], dtype=np.int64)
        road_vertices = set(np.unique(component_faces).tolist())
        selected_face_set = set(face_indices)
        vertex_faces: dict[int, set[int]] = {}
        all_vertex_faces: dict[int, set[int]] = {}
        selected_edge_counts: dict[tuple[int, int], int] = {}
        for face_index, face in enumerate(faces):
            for vertex_index in face:
                all_vertex_faces.setdefault(int(vertex_index), set()).add(face_index)
        for face_index in face_indices:
            face = faces[face_index]
            for vertex_index in face:
                vertex_faces.setdefault(int(vertex_index), set()).add(int(face_index))
            for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = tuple(sorted((int(first), int(second))))
                selected_edge_counts[edge] = selected_edge_counts.get(edge, 0) + 1
            perimeter_vertices = {
                vertex_index
                for edge, count in selected_edge_counts.items()
                if count == 1
                for vertex_index in edge
            }
        boundary_vertices = {
            vertex_index
            for vertex_index, adjacent_faces in vertex_faces.items()
            if vertex_index in perimeter_vertices
            or bool(all_vertex_faces[vertex_index] - selected_face_set)
        }
        # Boundary vertices are fixed so the road remains attached to its neighbors.
        interior_vertices = sorted(road_vertices - boundary_vertices)
        if not interior_vertices:
            continue
        neighbors: dict[int, set[int]] = {vertex_index: set() for vertex_index in interior_vertices}
        for triangle in component_faces:
            for vertex_index in triangle:
                if int(vertex_index) not in neighbors:
                    continue
                neighbors[int(vertex_index)].update(
                    int(other) for other in triangle if int(other) != int(vertex_index)
                )
        original = np.asarray(mesh.vertices, dtype=np.float64).copy()
        updated = original.copy()
        for _ in range(max(1, int(iterations))):
            previous = updated.copy()
            for vertex_index in interior_vertices:
                adjacent = sorted(neighbors[vertex_index])
                if adjacent:
                    average = previous[adjacent].mean(axis=0)
                    updated[vertex_index] = previous[vertex_index] * (1.0 - strength) + average * strength
        mesh.vertices[:] = updated
        moved = np.linalg.norm(updated - original, axis=1)
        moved_mask = np.asarray([moved[index] > 1e-8 for index in interior_vertices])
        if np.any(moved_mask):
            changes.append({
                "region": "road",
                "mesh_index": int(mesh_index),
                "face_count": len(face_indices),
                "vertex_count": len(road_vertices),
                "interior_vertex_count": len(interior_vertices),
                "boundary_vertex_count": len(boundary_vertices),
                "moved_vertex_count": int(np.count_nonzero(moved_mask)),
                "max_displacement": float(moved[interior_vertices].max()),
                "iterations": int(iterations),
                "boundary_vertices_preserved": True,
            })
    return changes


def canny_edges(image_path: Path, low_ratio: float = 0.10, high_ratio: float = 0.25) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a binary Canny edge map without requiring an external image package."""
    with Image.open(image_path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    gaussian = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
    blurred = _convolve_axis(_convolve_axis(gray, gaussian, axis=0), gaussian, axis=1)
    gradient_x = _convolve_axis(blurred, np.asarray([-1.0, 0.0, 1.0]), axis=1)
    gradient_y = _convolve_axis(blurred, np.asarray([-1.0, 0.0, 1.0]), axis=0)
    magnitude = np.hypot(gradient_x, gradient_y)
    angle = (np.rad2deg(np.arctan2(gradient_y, gradient_x)) + 180.0) % 180.0
    nms = np.zeros_like(magnitude)
    for first, second, lower, upper in (
        (0, 22.5, 0, 1), (22.5, 67.5, 1, 2), (67.5, 112.5, 2, 3),
        (112.5, 157.5, 3, 2), (157.5, 180, 0, 1),
    ):
        selected = (angle >= first) & (angle < second)
        first_neighbor = np.roll(magnitude, lower, axis=1)
        second_neighbor = np.roll(magnitude, -lower, axis=1)
        if upper != lower:
            first_neighbor = np.roll(first_neighbor, 1, axis=0)
            second_neighbor = np.roll(second_neighbor, -1, axis=0)
        nms[selected & (magnitude >= first_neighbor) & (magnitude >= second_neighbor)] = magnitude[selected & (magnitude >= first_neighbor) & (magnitude >= second_neighbor)]
    nonzero = nms[nms > 0]
    if len(nonzero) == 0:
        return np.zeros_like(nms, dtype=np.uint8), {"width": int(gray.shape[1]), "height": int(gray.shape[0]), "edge_pixels": 0}
    low = float(np.quantile(nonzero, low_ratio))
    high = float(np.quantile(nonzero, high_ratio))
    strong = nms >= high
    weak = (nms >= low) & ~strong
    edges = strong.copy()
    while True:
        connected = np.zeros_like(edges)
        for row_shift in (-1, 0, 1):
            for column_shift in (-1, 0, 1):
                if row_shift or column_shift:
                    connected |= np.roll(np.roll(edges, row_shift, axis=0), column_shift, axis=1)
        new_edges = edges | (weak & connected)
        if np.array_equal(new_edges, edges):
            break
        edges = new_edges
    edges[[0, -1], :] = False
    edges[:, [0, -1]] = False
    return (edges.astype(np.uint8) * 255), {
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "edge_pixels": int(np.count_nonzero(edges)),
        "low_threshold": low,
        "high_threshold": high,
    }


def _convolve_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    padded = np.pad(values, [(pad, pad) if current_axis == axis else (0, 0) for current_axis in range(values.ndim)], mode="edge")
    result = np.zeros_like(values, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(offset, offset + values.shape[axis])
        result += padded[tuple(slices)] * weight
    return result


def regularize_masked_buildings(
    meshes: list[Any],
    building_faces: set[tuple[int, int]],
    face_records: list[dict[str, Any]],
    edge_map: np.ndarray,
    aggression: float = 0.5,
) -> list[dict[str, Any]]:
    """Regularize the nearest visible building component and reduce its interior triangles."""
    aggression = float(np.clip(aggression, 0.0, 1.0))
    if not building_faces:
        return []
    records = {(int(item["mesh_index"]), int(item["face_index"])): item for item in face_records}
    components = _face_components(meshes, building_faces)
    if not components:
        return []
    component_scores = []
    for mesh_index, face_indices in components:
        component_mesh_faces = np.asarray(meshes[mesh_index].faces, dtype=np.int64)
        component_triangles = component_mesh_faces[face_indices]
        component_vertices = set(np.unique(component_triangles).tolist())
        component_edges: dict[tuple[int, int], int] = {}
        for triangle in component_triangles:
            for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
                edge = tuple(sorted((int(first), int(second))))
                component_edges[edge] = component_edges.get(edge, 0) + 1
        perimeter = {vertex for edge, count in component_edges.items() if count == 1 for vertex in edge}
        if len(component_vertices - perimeter) < 3:
            continue
        depths = [float(records[(mesh_index, face_index)]["depth"]) for face_index in face_indices if (mesh_index, face_index) in records]
        component_scores.append((float(np.mean(depths)) if depths else float("inf"), mesh_index, face_indices))
    if not component_scores:
        return []
    _, mesh_index, face_indices = min(component_scores)
    mesh = meshes[mesh_index]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    component_vertices = set(np.unique(faces[face_indices]).tolist())
    selected_edge_counts: dict[tuple[int, int], int] = {}
    for face_index in face_indices:
        face = faces[face_index]
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(first), int(second))))
            selected_edge_counts[edge] = selected_edge_counts.get(edge, 0) + 1
    perimeter = {vertex for edge, count in selected_edge_counts.items() if count == 1 for vertex in edge}
    interior = sorted(component_vertices - perimeter)
    if len(interior) < 3:
        return []

    original = np.asarray(mesh.vertices, dtype=np.float64).copy()
    points = original[sorted(component_vertices)]
    center = points.mean(axis=0)
    horizontal = points[:, [0, 2]] - center[[0, 2]]
    _, _, vectors = np.linalg.svd(horizontal, full_matrices=False)
    basis = np.column_stack((vectors[0], vectors[1]))
    coordinates = (original[:, [0, 2]] - center[[0, 2]]) @ basis
    updated = original.copy()
    for vertex_index in interior:
        adjacent = np.unique(faces[[face_index for face_index in face_indices if vertex_index in faces[face_index]]])
        adjacent = [int(index) for index in adjacent if int(index) in component_vertices and int(index) != vertex_index]
        if not adjacent:
            continue
        neighbor_coordinates = coordinates[adjacent]
        neighbor_heights = original[adjacent, 1]
        blend = 0.35 - aggression * 0.30
        coordinates[vertex_index] = coordinates[vertex_index] * blend + np.median(neighbor_coordinates, axis=0) * (1.0 - blend)
        updated[vertex_index, 1] = original[vertex_index, 1] * blend + float(np.median(neighbor_heights)) * (1.0 - blend)
        updated[vertex_index, [0, 2]] = coordinates[vertex_index] @ basis.T + center[[0, 2]]

    mesh.vertices[:] = updated
    component_extent = max(float(np.max(np.ptp(points, axis=0))), 1e-5)
    cluster_size = component_extent * (0.08 + aggression * 0.22)
    cluster_keys: dict[tuple[int, int, int], int] = {}
    vertex_map = {index: index for index in range(len(original))}
    for vertex_index in interior:
        key = tuple(np.rint(updated[vertex_index] / cluster_size).astype(int).tolist())
        representative = cluster_keys.setdefault(key, vertex_index)
        vertex_map[vertex_index] = representative
    mapped_faces = faces.copy()
    for face_index in face_indices:
        mapped_faces[face_index] = [vertex_map[int(index)] for index in faces[face_index]]
    keep = np.ones(len(faces), dtype=bool)
    seen: set[tuple[int, int, int]] = set()
    for face_index in face_indices:
        mapped = tuple(int(index) for index in mapped_faces[face_index])
        if len(set(mapped)) < 3 or tuple(sorted(mapped)) in seen:
            keep[face_index] = False
        else:
            seen.add(tuple(sorted(mapped)))
    mesh.faces = mapped_faces
    mesh.update_faces(keep)
    edge_score = 0.0
    for face_index in face_indices:
        record = records.get((mesh_index, face_index))
        if record is None:
            continue
        pixels = np.asarray(record.get("triangle_pixels", [record["pixel"]]), dtype=float).astype(int)
        valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < edge_map.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < edge_map.shape[0])
        if np.any(valid):
            edge_score += float(np.mean(edge_map[pixels[valid, 1], pixels[valid, 0]] > 0))
    return [{
        "region": "building",
        "mesh_index": int(mesh_index),
        "selected_face_count": len(face_indices),
        "remaining_face_count": int(np.count_nonzero(keep[face_indices])),
        "polygon_reduction": int(len(face_indices) - np.count_nonzero(keep[face_indices])),
        "vertex_count": len(component_vertices),
        "interior_vertex_count": len(interior),
        "boundary_vertex_count": len(perimeter),
        "boundary_vertices_preserved": True,
        "canny_edge_score": edge_score / max(len(face_indices), 1),
        "axis_alignment": "dominant horizontal PCA axes with median-height horizontal bands",
        "aggression": aggression,
    }]


def _face_components(meshes: list[Any], selected: set[tuple[int, int]]) -> list[tuple[int, list[int]]]:
    """Return connected face components while keeping mesh boundaries isolated."""
    components = []
    for mesh_index in sorted({key[0] for key in selected}):
        mesh_faces = {face_index for current_mesh, face_index in selected if current_mesh == mesh_index}
        faces = np.asarray(meshes[mesh_index].faces, dtype=np.int64)
        vertex_to_faces: dict[int, set[int]] = {}
        for face_index in mesh_faces:
            for vertex_index in faces[face_index]:
                vertex_to_faces.setdefault(int(vertex_index), set()).add(face_index)
        remaining = set(mesh_faces)
        while remaining:
            component = []
            pending = [remaining.pop()]
            while pending:
                face_index = pending.pop()
                component.append(face_index)
                for vertex_index in faces[face_index]:
                    neighbors = vertex_to_faces[int(vertex_index)] & remaining
                    pending.extend(neighbors)
                    remaining.difference_update(neighbors)
            components.append((mesh_index, component))
    return components
