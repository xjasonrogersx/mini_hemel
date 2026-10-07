"""Open3D-backed building regularization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import open3d as o3d


def _face_components(meshes: list[Any], selected: set[tuple[int, int]]) -> list[tuple[int, list[int]]]:
    """Return connected selected face components, isolated per source mesh."""
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


def _boundary_vertices(faces: np.ndarray, face_indices: list[int]) -> set[int]:
    edge_counts: dict[tuple[int, int], int] = {}
    for face_index in face_indices:
        face = faces[face_index]
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return {vertex for edge, count in edge_counts.items() if count == 1 for vertex in edge}


def _open3d_smooth_component(
    original: np.ndarray,
    faces: np.ndarray,
    face_indices: list[int],
    aggression: float,
) -> np.ndarray:
    """Smooth a component with Open3D while keeping its boundary fixed."""
    component_vertices = sorted(np.unique(faces[face_indices]).tolist())
    vertex_remap = {vertex_index: local_index for local_index, vertex_index in enumerate(component_vertices)}
    local_faces = np.asarray(
        [[vertex_remap[int(vertex)] for vertex in faces[face_index]] for face_index in face_indices],
        dtype=np.int32,
    )
    component = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(original[component_vertices]),
        o3d.utility.Vector3iVector(local_faces),
    )
    component.remove_degenerate_triangles()
    component.remove_unreferenced_vertices()
    iterations = max(1, int(round(1 + aggression * 5)))
    smoothed = component.filter_smooth_taubin(number_of_iterations=iterations)
    smoothed_vertices = np.asarray(smoothed.vertices)
    boundary = _boundary_vertices(faces, face_indices)
    for vertex_index in boundary:
        local_index = vertex_remap.get(vertex_index)
        if local_index is not None and local_index < len(smoothed_vertices):
            smoothed_vertices[local_index] = original[vertex_index]
    updated = original.copy()
    updated[component_vertices] = (
        original[component_vertices] * (1.0 - aggression)
        + smoothed_vertices * aggression
    )
    updated[list(boundary)] = original[list(boundary)]
    return updated


def regularize_masked_buildings(
    meshes: list[Any],
    building_faces: set[tuple[int, int]],
    face_records: list[dict[str, Any]],
    line_map: np.ndarray,
    aggression: float = 0.5,
) -> list[dict[str, Any]]:
    """Regularize the nearest visible building component with Open3D."""
    aggression = float(np.clip(aggression, 0.0, 1.0))
    if not building_faces:
        return []
    records = {(int(item["mesh_index"]), int(item["face_index"])): item for item in face_records}
    components = _face_components(meshes, building_faces)
    scored_components = []
    for mesh_index, face_indices in components:
        faces = np.asarray(meshes[mesh_index].faces, dtype=np.int64)
        vertices = set(np.unique(faces[face_indices]).tolist())
        boundary = _boundary_vertices(faces, face_indices)
        if len(vertices - boundary) < 3:
            continue
        depths = [float(records[key]["depth"]) for key in ((mesh_index, face_index) for face_index in face_indices) if key in records]
        scored_components.append((float(np.mean(depths)) if depths else float("inf"), mesh_index, face_indices))
    if not scored_components:
        return []
    _, mesh_index, face_indices = min(scored_components)
    mesh = meshes[mesh_index]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    component_vertices = set(np.unique(faces[face_indices]).tolist())
    boundary = _boundary_vertices(faces, face_indices)
    interior = sorted(component_vertices - boundary)
    if len(interior) < 3:
        return []

    original = np.asarray(mesh.vertices, dtype=np.float64).copy()
    updated = _open3d_smooth_component(original, faces, face_indices, aggression)
    mesh.vertices[:] = updated

    extent = max(float(np.max(np.ptp(original[sorted(component_vertices)], axis=0))), 1e-5)
    cluster_size = extent * (0.08 + aggression * 0.22)
    cluster_keys: dict[tuple[int, int, int], int] = {}
    vertex_map = {index: index for index in range(len(original))}
    for vertex_index in interior:
        key = tuple(np.rint(updated[vertex_index] / cluster_size).astype(int).tolist())
        vertex_map[vertex_index] = cluster_keys.setdefault(key, vertex_index)
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
        valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < line_map.shape[1]) &
                 (pixels[:, 1] >= 0) & (pixels[:, 1] < line_map.shape[0]))
        if np.any(valid):
            edge_score += float(np.mean(line_map[pixels[valid, 1], pixels[valid, 0]] > 0))
    return [{
        "region": "building",
        "mesh_index": int(mesh_index),
        "selected_face_count": len(face_indices),
        "remaining_face_count": int(np.count_nonzero(keep[face_indices])),
        "polygon_reduction": int(len(face_indices) - np.count_nonzero(keep[face_indices])),
        "vertex_count": len(component_vertices),
        "interior_vertex_count": len(interior),
        "boundary_vertex_count": len(boundary),
        "boundary_vertices_preserved": True,
        "mlsd_line_score": edge_score / max(len(face_indices), 1),
        "axis_alignment": "Open3D Taubin smoothing with preserved component boundary",
        "aggression": aggression,
    }]
