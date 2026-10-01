# View-Guided Mesh Refinement Plan

## Goal

Use the generated texture, depth maps, and segmentation results to create a refined mesh without modifying the original `merged.gltf`. The refined output should have better road geometry, more local polygons, discrete window and door regions, and semantic attributes for road, sidewalk, trees, windows, and doors.

## Current Evidence

- `merge_gltf.py` creates `merged.gltf` and `merged.bin` from the source parts.
- The current GLTF contains indexed triangle primitives, UVs, textures, materials, and node transforms.
- `viewer.py` loads the scene with trimesh and renders it with pyrender.
- `artifacts.json` stores camera poses, result images, depth images, semantic masks, Grounding DINO boxes, and SAM2 masks.
- Depth Anything provides normalized relative depth, not calibrated metric depth.
- SegFormer and Mask2Former provide semantic label maps and masks for building, wall, roof, window, door, tree, street, and sidewalk.
- Grounding DINO provides image-space boxes; SAM2 provides image-space binary masks.

## Design Decisions

- Keep `merged.gltf`, source parts, captures, and original artifacts unchanged.
- Create a new refined GLTF and a versioned refinement sidecar.
- Use `result_render` as the generated texture target.
- Use the saved camera matrix and capture dimensions to associate image masks with mesh faces.
- Prefer renderer z-buffer depth for visibility. Use Depth Anything only as relative-depth evidence.
- Start with sidecar face/vertex labels before adding custom GLTF attributes.
- Use local adaptive refinement instead of globally multiplying every polygon.

## Phases

### 1. Asset Inventory and Sidecar Schema

1. Inventory nodes, primitives, transforms, vertex/index counts, UVs, materials, textures, and bounds.
2. Define a sidecar containing:
   - source and refined GLTF paths
   - source asset hash
   - artifact index
   - camera matrix and capture size
   - source image, result image, depth, and mask files
   - node, mesh, primitive, and face IDs
   - labels, confidence, source model, and thresholds
   - geometry operations and before/after counts
3. Confirm image and mask dimensions match the recorded camera capture size.

**Exit check:** one image annotation can be linked reproducibly to a source primitive and face set without changing the asset.

### 2. Apply the Generated Texture

1. Resolve the artifact's `result_render` image.
2. Preserve existing UVs, materials, node transforms, and unaffected primitives.
3. Write a copied GLTF/material set referencing the generated image.
4. Keep the original texture for comparison.
5. Render from the recorded camera pose to check alignment, orientation, and stretching.

**Exit check:** the refined asset displays the generated texture with unchanged geometry.

### 3. Project Image Masks onto the Mesh

1. Implement one world-to-pixel projection using the saved camera matrix and pyrender camera conventions.
2. Transform each primitive's vertices into world space.
3. Project triangle vertices into image coordinates.
4. Select faces using mask overlap or rasterized triangle coverage, not only triangle centroids.
5. Reject triangles outside the frame or behind the camera.
6. Compare projected triangle depth against the renderer depth buffer with a configurable tolerance.
7. Store selected faces and coverage statistics in the sidecar.

Use road, sidewalk, and tree semantic masks; window and door masks; SAM2 masks; and explicit car-removal regions.

**Exit check:** tests cover visible, partial, off-screen, behind-camera, and occluded triangles.

### 4. Remove Road Bumps and Car Geometry

1. Identify road faces with SegFormer or Mask2Former.
2. Identify regions where cars were removed using explicit detections or masks.
3. Fit a robust local plane or low-order surface from nearby road vertices.
4. Flatten or smooth only selected road vertices.
5. Apply boundary falloff so the change does not create hard steps.
6. Preserve sidewalk, building, tree, and road boundaries unless their labels support modification.
7. Record moved vertices, fit parameters, mask coverage, and confidence.

First produce a preview with highlighted candidate faces. Apply geometry changes only after visual review.

#### 4A. Depth-guided conservative edit

The first applied implementation slice is complete:

1. Load `generated_depth_render` as normalized relative depth.
2. Resize depth maps to the artifact camera frame and record source/projection dimensions.
3. Compute per-face relative-depth mean/range and gradient evidence.
4. Mark the evidence as `relative_only`; values are never interpreted as metres.
5. Fit local planes independently per connected candidate component.
6. Blend toward the fitted plane and clamp each vertex displacement to 1% of the component extent.
7. Export and reload a separate GLTF; keep the source asset unchanged.

Depth Anything remains relative-only: it gates smooth interior regions and
provides ordering/edge evidence, but is never converted into metres. Road and
sidewalk masks drive ground candidates; vertical front-facing regions drive
facade candidates while tree and high-gradient boundaries are protected.

### 5. Increase Polygon Count

1. Subdivide selected road, sidewalk, facade, window, and door regions.
2. Use depth gradients to identify surfaces that need additional geometry.
3. Treat Depth Anything gradients as relative evidence, not metric measurements.
4. Preserve UV interpolation and material boundaries.
5. Regenerate normals only when required by the renderer or exporter.
6. Cap subdivision levels and compare vertex, face, memory, and render-time increases.

Use adaptive local subdivision before considering global remeshing.

### 6. Create Discrete Window and Door Regions

1. Combine semantic window/door masks with Grounding DINO and SAM2 masks.
2. Project those masks onto candidate mesh faces.
3. Split faces along region boundaries where needed.
4. Create separate primitives or material groups for windows and doors.
5. Preserve the facade as backing geometry unless actual openings are explicitly required.
6. Assign stable IDs such as `window_id` and `door_id` in the sidecar.

**Exit check:** individual window and door regions can be selected and remain separate after GLTF export/reload.

### 7. Add Semantic Attributes

Start with sidecar labels because the current asset does not consistently contain custom GLTF vertex attributes.

Assign labels and confidence for:

- `road`
- `sidewalk`
- `tree`
- `window`
- `door`
- `building`
- `wall`
- `roof`

Store the source mask, model, threshold, confidence, and projected coverage. Optionally export `COLOR_0` as a debug visualization after GLTF round-trip tests prove that the target tools preserve it.

**Exit check:** semantic debug colors can be reproduced after reloading the refined asset and sidecar.

### 8. Add Viewer and Web Controls

1. Add refinement commands for preview, texture-only, semantic-only, and full refinement modes.
2. Add artifact-page controls for previewing and building a refined mesh.
3. Display selected face counts, skipped/occluded faces, displaced vertices, polygon counts, and confidence warnings.
4. Add links to the refined GLTF and sidecar.
5. Keep original images, masks, and source GLTF available for comparison.

## Implementation Surfaces

- `merge_gltf.py`: GLTF/resource preservation and refined asset writing.
- `viewer.py`: scene loading, camera state, rendering, capture metadata, and artifact linkage.
- `model_server/depth_anything_worker.py`: relative-depth response format.
- `model_server/segformer_worker.py`: semantic segmentation outputs.
- `model_server/mask2former_worker.py`: semantic segmentation outputs.
- `model_server/grounding_dino_worker.py`: bounding-box outputs.
- `model_server/sam2_worker.py`: prompted mask outputs.
- `web_server.py`: refinement controls and refined-asset routes.
- `artifacts.json`: image, camera, and annotation references.

## Verification

1. Compile the refinement module, viewer, web server, and GLTF tooling.
2. Compare source/refined node, primitive, material, texture, UV, bounds, and resource counts.
3. Render the recorded camera pose and verify generated texture alignment.
4. Test projection with visible, partial, off-screen, behind-camera, and occluded triangles.
5. Compare renderer depth with Depth Anything and document scale/direction assumptions.
6. Confirm road cleanup changes only road/car candidate regions.
7. Verify local subdivision raises polygon counts within configured limits.
8. Verify window and door IDs survive GLTF export/reload.
9. Reload sidecar labels and reproduce semantic debug colors.
10. Reload the refined asset in the viewer and test orbit, walk mode, hit testing, RGB/depth modes, and source-camera overlays.

## Scope Boundaries and Risks

Deferred initially:

- New model training
- Calibrated multi-view reconstruction
- Unseen-surface completion
- UV unwrapping
- Full global remeshing
- Metric reconstruction from monocular Depth Anything alone

Risks:

- Generated textures can hallucinate or shift object boundaries.
- Normalized Depth Anything maps cannot establish reliable metric vertex positions without calibration.
- A single view cannot classify hidden or back-facing geometry.
- GLTF round-tripping may split primitives or alter resources.
- Global polygon increases can cause large memory and render-time costs.
