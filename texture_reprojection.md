# Texture Reprojection

`texture_reprojection.py` applies a captured or generated RGB image to the visible faces of a mesh using the recorded camera view. It is a view-projected texture operation: it does not unwrap the mesh or create a new global UV layout.

## Process

1. Load the source RGB image at its original resolution.
2. Project every scene face into the artifact camera using `project_scene_faces()`.
3. Run `visible_face_keys()` to reject faces that are hidden, behind another face, back-facing, or mostly outside the capture frame.
4. Split each mesh into visible and occluded face subsets.
5. Keep occluded faces in their original mesh/material subset.
6. Project every visible vertex into image pixels with `project_points()`.
7. Convert pixel coordinates into UV coordinates:

   - `u = pixel_x / capture_width`
   - `v = 1 - pixel_y / capture_height`

8. Attach the full-resolution source image as the visible subset's base-color texture.
9. Return the split meshes to `load_scene()`, which sends them to Pyrender.

## Main Function

```python
apply_view_projected_texture(
    loaded,
    texture_path,
    camera_pose,
    capture_size,
)
```

### Inputs

- `loaded`: a `trimesh.Scene` containing the source mesh geometry.
- `texture_path`: path to an RGB image. The image is loaded without resizing.
- `camera_pose`: artifact camera metadata containing a 4x4 `matrix`.
- `capture_size`: dictionary containing positive `width` and `height` values matching the camera capture.

### Outputs

Returns:

```text
(projected_meshes, visible_face_count)
```

`projected_meshes` contains:

- Original-material mesh subsets for occluded faces.
- Camera-UV mesh subsets for visible faces.

`visible_face_count` is the number of visible face records used for reprojection.

## Why Faces Are Split

A single mesh cannot safely use one camera-projected texture for faces that were not visible in the source view. The function therefore keeps occluded faces separate with their original material and applies the generated image only to visible faces.

This avoids painting the source image onto hidden geometry whose image-space location is unrelated to the captured surface.

## UV Convention

The projected UVs use normalized image coordinates. The horizontal axis is unchanged, while the vertical axis is flipped because image pixels start at the top and texture UVs start at the bottom:

```text
u = x / width
v = 1 - y / height
```

Coordinates are clipped to `0..1`.

## Integration

`viewer.load_scene()` validates that the camera pose and capture size exist, then calls `apply_view_projected_texture()`. It records:

- `_projected_visible_faces`
- `_projected_total_faces`

on the Pyrender scene for renderer diagnostics.

The workflow is used when the viewer deploys a refined, road-smoothed, or road-retiling asset together with an artifact `result_render` texture.

## Limitations

- The method is correct for the recorded camera view, not arbitrary viewpoints.
- It does not solve occluded-surface texturing.
- It does not unwrap or optimize mesh UVs.
- It depends on accurate camera pose and capture dimensions.
- Generated image content may not align perfectly with geometry if the image model changes object boundaries.
- Meshes without a compatible material may receive UV coordinates without displaying the assigned texture until a texture-capable material is present.

## Implementation Location

The implementation is in `texture_reprojection.py`. Camera projection and visibility calculations remain shared utilities in `mesh_refinement.py`; scene loading and renderer setup remain in `viewer.py`.
