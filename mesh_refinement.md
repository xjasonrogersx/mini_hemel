# `mesh_refinement.py`

`mesh_refinement.py` contains the shared view-guided mesh utilities used by the viewer. It converts camera images and segmentation masks into mesh face selections, then applies optional geometry changes to selected regions.

The module is not one single refinement algorithm. It provides several independent options that can be combined into a pipeline.

## Typical Pipeline

```text
source scene
    |
    +--> project scene faces
    |       |
    |       +--> visibility filtering
    |       +--> segmentation-mask sampling
    |       +--> relative-depth sampling
    |
    +--> remove selected vertices/faces
    |       |
    |       +--> retile removed regions
    |
    +--> conservative geometry refinement
```

Building regularisation is a separate module:

- `building_regularization.py` owns Open3D building regularisation.

## Options

### 1. Project 3D points

`project_points(points, camera_matrix, capture_size, yfov_degrees=45.0)`

Projects world-space points into the camera image.

Inputs:

- `points`: NumPy array shaped `(N, 3)`.
- `camera_matrix`: 4x4 camera pose matrix.
- `capture_size`: dictionary containing positive `width` and `height`.
- `yfov_degrees`: vertical field of view used for the projection.

Returns:

- Pixel coordinates shaped `(N, 2)`.
- Camera-relative depth values.

This is the base operation used by all image-guided options.

### 2. Load relative depth

`load_relative_depth(depth_path, capture_size)`

Loads a Depth Anything image, normalises it to `0..1`, resizes it to the capture resolution, and calculates an image-gradient threshold.

Important limitation: the result is relative depth, not metric world distance. It can identify depth changes and edges but must not be treated as a scale-accurate measurement.

Returns:

- A 2D floating-point depth array.
- Metadata containing source dimensions, projection dimensions, value range, gradient percentile, and `relative_only=True`.

### 3. Sample relative depth on faces

`sample_relative_depth(depth, face_records)`

Samples each projected triangle over a barycentric grid and attaches depth statistics to each face.

Per-face evidence includes:

- Mean relative depth.
- Relative depth range.
- Mean and maximum image gradient.
- Number of valid samples.

This option is useful for detecting depth discontinuities or filtering geometry near strong image edges. It does not recover metric 3D depth.

### 4. Project every scene face

`project_scene_faces(scene, camera_pose, capture_size)`

Dumps each scene mesh and records every face's projected triangle, centroid pixel, depth, front-facing status, and in-frame status.

The returned `face_records` are the common input for visibility tests and segmentation-mask sampling.

### 5. Keep visible faces

`visible_face_keys(face_records, capture_size)`

Uses dense triangle samples and a nearest-depth buffer to keep faces that are:

- In front of the camera.
- In the image frame.
- Front-facing.
- Mostly visible rather than hidden behind another face.

The thresholds are module constants:

- `VISIBILITY_SAMPLE_DIVISIONS = 8`
- `VISIBILITY_MIN_IN_FRAME_RATIO = 0.80`
- `VISIBILITY_MIN_VISIBLE_RATIO = 0.80`
- `VISIBILITY_DEPTH_TOLERANCE = 0.005`

This option prevents a mask from modifying geometry that is projected in the image but actually occluded.

### 6. Sample a segmentation mask

`sample_mask(mask_path, face_records, capture_size)`

Samples a raster mask over each projected triangle and returns the faces whose coverage meets the mask-selection rules.

The result contains the mask filename, selected face pairs, selected count, and sampling metadata. Face pairs use this format:

```text
(mesh_index, face_index)
```

This is the normal bridge from a Mask2Former or other image mask to mesh geometry.

### 7. Remove geometry inside masks

`remove_vertices_inside_masks(meshes, selected_faces, ...)`

Removes vertices and faces associated with selected mask regions while preserving enough information to repair the resulting holes.

The function also preserves or rebuilds visual data where possible:

- UV coordinates and texture material when the mesh has texture visuals.
- Face colours when the mesh uses face colours.
- Boundary-region information for later retile operations.

The removed-patch data is attached to the mesh as private attributes, including `_mask_hole_removed_patches` and boundary-region data.

Use this option for removal workflows such as car, road-object, or masked-object removal.

### 8. Retile removed regions

`retile_mesh_holes(meshes, flat_color=(180, 180, 180, 255))`

Creates replacement patch meshes for regions recorded by `remove_vertices_inside_masks`.

There are two paths:

- Preferred removed-patch path: rebuilds every saved removed triangle region, flattens it to a best-fit plane with Open3D, and preserves the removed/replacement region structure.
- Boundary fallback path: builds a 2D boundary hull and triangulates it with `mapbox_earcut`.

Returns:

- The original meshes plus generated patch meshes.
- The number of added triangles.

This option fills holes; it does not smooth the surrounding source mesh or generate a realistic texture.

### 9. Conservative geometry refinement

`apply_conservative_geometry_refinement(meshes, face_records, semantic_masks, depth_guidance, visible_faces)`

Applies small planar corrections to selected ground and facade regions.

Selection rules:

- Road and sidewalk masks select the `ground` region.
- Other visible, mostly vertical faces can be selected as `facade`.
- Relative-depth gradients are used as an edge-safety signal, not as metric depth.
- Ground and facade planes are checked using their normals before modification.
- Vertex displacement is capped at approximately one percent of the component extent.

This is the least aggressive geometry option. It is intended for bounded cleanup, not major remodelling.

## Private Helpers

- `_triangle_samples(triangle)`: creates a dense barycentric sampling grid over a triangle.
- `_triangle_sample_weights()`: returns the barycentric weights used by visibility sampling.
- `_face_components(meshes, selected)`: groups selected faces by connected vertices while keeping source meshes separate.

These helpers support the public operations and are not intended to be called by the web UI directly.

## Building Regularisation Note

The current file contains duplicate `regularize_masked_buildings()` definitions from earlier edits. Python uses the later definition, so the first definition is shadowed. The active viewer import comes from `building_regularization.py`, which is the Open3D implementation and should be treated as the canonical building path.

## Choosing an Option

| Goal | Recommended operation |
| --- | --- |
| Convert 3D geometry to image coordinates | `project_points` or `project_scene_faces` |
| Ignore hidden geometry | `visible_face_keys` |
| Select geometry from an image mask | `sample_mask` |
| Use monocular depth as an edge clue | `load_relative_depth` + `sample_relative_depth` |
| Remove masked geometry | `remove_vertices_inside_masks` |
| Fill holes after removal | `retile_mesh_holes` |
| Make small plane-safe corrections | `apply_conservative_geometry_refinement` |
| Smooth roads | `road_smoothing.py` |
| Regularise buildings | `building_regularization.py` |

## Dependencies

The module uses:

- NumPy for arrays and geometry calculations.
- Pillow for depth and mask images.
- Trimesh for scene and mesh data.
- Open3D for mesh smoothing and patch processing.
- Mapbox Earcut for fallback polygon triangulation.
