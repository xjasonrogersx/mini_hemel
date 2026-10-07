# Triangle Visibility

The refined viewer decides which triangles may receive the generated texture in
`mesh_refinement.visible_face_keys`. Triangles that fail any required test keep
their original material and UVs.

## Projection

Each mesh face is projected into the saved artifact camera using the camera
matrix, a 45-degree vertical field of view, and the recorded capture size. The
projected triangle is sampled on a barycentric grid with 8 divisions, producing
45 samples distributed across the interior and edges.

## Required Tests

A face must:

- Have all three vertices in front of the camera.
- Be front-facing according to its world-space normal and camera position.
- Intersect the image bounds.
- Have at least 80% of its 45 projected samples inside the image.
- Win the projected depth test for at least 80% of its in-frame samples.

The depth test compares each face sample with the nearest face sample rounded
to the same image pixel. A relative tolerance of 0.5%, with an absolute floor
of 0.005 scene units, treats nearly coincident surfaces as visible rather than
rejecting them because of rasterization or floating-point noise.

The rule is intentionally less pessimistic than requiring every sample to be
visible. A triangle may cross a viewport boundary or have a small number of
samples collide with a neighboring surface and still receive the generated
texture. A genuinely hidden or back-facing triangle remains excluded.

## Texture Application

The visible face IDs are split from each mesh. Visible submeshes receive UVs
projected into the generated image. Non-visible submeshes retain their original
UVs and materials. The generated image is center-fitted to the artifact capture
size before upload, and pyrender handles the OpenGL image-row flip.

## Tuning

The constants are defined at the top of `mesh_refinement.py`:

- `VISIBILITY_SAMPLE_DIVISIONS`: sampling density.
- `VISIBILITY_MIN_IN_FRAME_RATIO`: allowed viewport-boundary tolerance.
- `VISIBILITY_MIN_VISIBLE_RATIO`: allowed partial-occlusion tolerance.
- `VISIBILITY_DEPTH_TOLERANCE`: depth comparison tolerance.

Increasing either 80% threshold makes the result more conservative. Lowering a
threshold applies the generated texture to more triangles but increases the
risk of texturing partially occluded surfaces.
