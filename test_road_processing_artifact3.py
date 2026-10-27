"""Focused regression test for road processing on Artifact 3."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import trimesh

from road_processing import (
    _build_triangle_index,
    _candidate_triangle_indices,
    _load_mask_triangles,
    flatten_and_retile_meshes,
    project_points,
)


ROOT = Path(__file__).resolve().parent


class Artifact3RoadProcessingTest(unittest.TestCase):
    def test_segformer_road_and_sidewalk_follow_artifact_3_masks(self) -> None:
        artifacts = json.loads((ROOT / "artifacts.json").read_text(encoding="utf-8"))
        artifact = artifacts[2]
        camera_pose = artifact["camera_pose"]
        capture_size = camera_pose["capture_size"]
        mask_groups = {
            category: [
                ROOT / "captures" / name
                for name in artifact["segmentation_masks"]
                if f"_{category}_" in f"_{name.lower()}_"
            ]
            for category in ("road", "sidewalk")
        }
        self.assertEqual(artifact["road_processing_segmentation_source"], "segformer")
        self.assertTrue(all(path.is_file() for paths in mask_groups.values() for path in paths))

        source_scene = trimesh.load(ROOT / "merged.gltf", file_type="gltf", force="scene")
        source_meshes = source_scene.dump(concatenate=False)
        selected_meshes = [source_meshes[22], source_meshes[26]]
        mask_triangles = {
            category: _load_mask_triangles(paths, capture_size)
            for category, paths in mask_groups.items()
        }
        indexed_triangles = {
            category: _build_triangle_index(triangles)
            for category, triangles in mask_triangles.items()
        }
        candidate_count = 0
        for source_mesh in selected_meshes:
            projected, depths = project_points(source_mesh.vertices, camera_pose["matrix"], capture_size)
            candidate_count += sum(
                len(_candidate_triangle_indices(bounds, index, projected[face]))
                for face in source_mesh.faces
                if (depths[face] > 0).all()
                for index, bounds in indexed_triangles.values()
            )
        self.assertGreater(candidate_count, 0)

        processed_meshes, stats = flatten_and_retile_meshes(
            [source_mesh.copy() for source_mesh in selected_meshes],
            mask_groups,
            camera_pose["matrix"],
            capture_size,
        )

        self.assertEqual(len(processed_meshes), 4)
        self.assertGreater(stats["triangles_added"], 0)
        self.assertEqual(set(stats["categories"]), {"road", "sidewalk"})
        self.assertGreater(stats["categories"]["road"]["triangles"], 0)
        self.assertGreater(stats["categories"]["sidewalk"]["triangles"], 0)
        self.assertGreater(len(processed_meshes[1].faces), 0)
        self.assertGreater(len(processed_meshes[2].faces), 0)


if __name__ == "__main__":
    unittest.main()