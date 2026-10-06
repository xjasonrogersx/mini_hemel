"""Open3D-backed road smoothing helpers."""

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


def smooth_masked_road(
    meshes: list[Any],
    road_masks: list[dict[str, Any]],
    iterations: int = 3,
    strength: float = 0.45,
) -> list[dict[str, Any]]:
    """Taubin-smooth selected road components with Open3D and pin their boundaries."""
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
        interior_vertices = sorted(road_vertices - boundary_vertices)
        if not interior_vertices:
            continue
        original = np.asarray(mesh.vertices, dtype=np.float64).copy()
        component_vertex_indices = np.asarray(sorted(road_vertices), dtype=np.int64)
        component_remap = {
            int(vertex_index): local_index
            for local_index, vertex_index in enumerate(component_vertex_indices)
        }
        local_faces = np.asarray([
            [component_remap[int(vertex)] for vertex in face]
            for face in component_faces
        ], dtype=np.int32)
        open3d_component = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(original[component_vertex_indices]),
            o3d.utility.Vector3iVector(local_faces),
        )
        open3d_component.compute_vertex_normals()
        open3d_component = open3d_component.filter_smooth_taubin(
            number_of_iterations=max(1, int(iterations)),
            lambda_filter=float(np.clip(strength, 0.05, 0.95)),
            mu=-float(np.clip(strength * 0.98, 0.05, 0.95)),
        )
        smoothed = np.asarray(open3d_component.vertices)
        for vertex_index in boundary_vertices:
            smoothed[component_remap[int(vertex_index)]] = original[int(vertex_index)]
        component_points = original[component_vertex_indices]
        plane_center = component_points.mean(axis=0)
        _, _, plane_basis = np.linalg.svd(component_points - plane_center, full_matrices=False)
        plane_normal = plane_basis[-1]
        interior_local = [component_remap[index] for index in interior_vertices]
        if interior_local:
            distances = (smoothed[interior_local] - plane_center) @ plane_normal
            smoothed[interior_local] -= distances[:, None] * plane_normal * 0.65
        updated = original.copy()
        updated[component_vertex_indices] = smoothed
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
