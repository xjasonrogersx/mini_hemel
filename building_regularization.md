# Building Regularisation

The building regularisation workflow selects visible building faces from a Mask2Former building mask, groups connected faces into components, chooses the nearest component, and regularises its interior geometry with Open3D.

## Process

1. The viewer loads the source scene and decomposes it into meshes.
2. `project_scene_faces()` projects every mesh face into the artifact camera view.
3. Mask2Former building masks are sampled against those projected faces.
4. Only faces that are both selected by the mask and visible in the camera view are retained.
5. Connected selected faces are found per mesh. Components without at least three interior vertices are skipped.
6. The nearest usable component is smoothed with Open3D Taubin smoothing. Boundary vertices are restored so the building remains attached to surrounding geometry.
7. Interior vertices are clustered to reduce duplicate or overly dense polygons. Degenerate and duplicate faces are removed.
8. M-LSD line pixels are sampled for a diagnostic line score; the lines guide reporting but do not replace the geometry mask.
9. The modified scene is exported as `captures/buildings_regularized_<artifact-index>.gltf` and its metadata is written to `artifacts.json`.
10. **Deploy mesh to viewer** loads that generated mesh. Deployment does not run regularisation again.

## Inputs

The worker receives:

- `meshes`: a list of `trimesh.Trimesh` objects from the source GLTF scene.
- `building_faces`: a set of `(mesh_index, face_index)` pairs selected by the building mask and visibility test.
- `face_records`: projected face records containing `mesh_index`, `face_index`, `depth`, `pixel`, and optionally `triangle_pixels`.
- `line_map`: a 2D grayscale NumPy array containing the M-LSD line image.
- `aggression`: a float from `0.0` to `1.0`. Higher values apply stronger Open3D smoothing and larger vertex clustering.

The artifact must also contain a camera matrix and capture dimensions, a generated render image, and an existing Mask2Former building mask. M-LSD is run against the generated render when its line image is missing.

## Outputs

The function returns a list of change records. Each record includes the selected mesh, selected and remaining face counts, polygon reduction, interior and boundary vertex counts, whether the boundary was preserved, the M-LSD line score, and the aggression value.

The worker stores the output filename, M-LSD filename and metadata, aggression, status, and change records in the artifact entry.

## Implementation

The implementation is in `building_regularization.py`. The viewer orchestration remains in `viewer.py`, while projection and mask sampling remain in `mesh_refinement.py`.
