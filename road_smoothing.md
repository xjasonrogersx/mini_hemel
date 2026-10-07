# Road Smoothing

The road smoothing workflow uses Open3D Taubin smoothing on road faces selected by Mask2Former. It smooths only the interior of each connected road region, preserves the region boundary, and flattens the smoothed interior toward the component's best-fit plane.

## Process

1. The viewer loads the source scene and the artifact's road segmentation masks.
2. Road masks whose filename contains `mask2former_road` contribute their selected `(mesh_index, face_index)` pairs.
3. Selected faces are grouped into connected components within each mesh.
4. For each component, boundary vertices are detected from the component perimeter and from vertices shared with unselected faces. These vertices remain fixed so roads stay connected to neighboring surfaces.
5. The component is converted to an Open3D `TriangleMesh` and processed with Taubin smoothing.
6. The smoothed interior is moved partway toward the component's best-fit plane to reduce bumps and local height variation.
7. The updated vertices are written back to the original `trimesh` mesh.
8. Change records are returned to the viewer worker, which exports the smoothed scene and stores metadata in `artifacts.json`.
9. **Deploy mesh to viewer** loads the generated road-smoothed mesh. Deployment does not repeat smoothing.

## Inputs

`smooth_masked_road()` accepts:

- `meshes`: a list of `trimesh.Trimesh` objects from the source scene.
- `road_masks`: mask dictionaries with a `file` name and `selected_faces`. Only masks containing `mask2former_road` in the filename are used.
- `iterations`: the number of Open3D Taubin iterations. The viewer's default is `3`.
- `strength`: the Taubin lambda strength, clipped to `0.05` through `0.95`. The viewer's default is `0.45`; the negative mu value is derived from it.

Each `selected_faces` entry must contain a two-item `[mesh_index, face_index]` pair. Mesh and face indices refer to the decomposed scene used to create the mask records.

## Outputs

The function returns one change record per component that moved. Records include:

- `mesh_index`
- total road `face_count` and `vertex_count`
- interior and boundary vertex counts
- moved vertex count and maximum displacement
- smoothing iteration count
- `boundary_vertices_preserved`

The worker uses these records for logging and artifact metadata. The generated file is consumed by the viewer deployment action; texture reprojection and road-hole replacement are separate workflows.

## Implementation

The implementation is in `road_smoothing.py`. The viewer orchestration remains in `viewer.py`; projection, visibility, and mask sampling remain in `mesh_refinement.py`. Open3D is provided by the `open3d` dependency in `requirements.txt`.
