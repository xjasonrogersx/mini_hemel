# GLTF Viewer Work Completed

## Viewers

- `view.py` remains the custom pyglet GLTF viewer.
- `view2.py` uses trimesh to load and display GLTF/GLB scenes.
- The trimesh viewer uses a black background.
- GLTF texture and UV handling was improved in `view.py`.
- Texture remapping in `merge_gltf.py` now handles texture entries without
  optional `texCoord` fields.

## `view2.py` keyboard controls

- **G**: Toggle a Y-up bird's-eye/top-down camera view.
- **Q**: Capture the current view, run Mask2Former COCO panoptic
  segmentation, identify `car` segments, and highlight the corresponding
  3D triangles.
- **W**: Capture the current view and run SegFormer semantic segmentation
  using the configured Cityscapes checkpoint.
- **Y**: Capture the current view and run the local VisDrone YOLO checkpoint
  (`best.pt`) for car detection.
- **R**: After a detection, remove the red detection geometry, hide the
  original affected mesh nodes, preserve the non-car mesh remainder, and
  replace detected car triangles with grey untextured geometry flattened to a
  shared road plane. The road plane is calculated from the lowest world-Y
  value across the detected car triangles.
- **S**: Save the current view as a timestamped PNG.

## Segmentation models

Default model options:

```text
Mask2Former:
  facebook/mask2former-swin-small-coco-panoptic

SegFormer:
  nvidia/segformer-b2-finetuned-cityscapes-1024-1024

VisDrone YOLO:
  best.pt
```

The VisDrone checkpoint supplied in the project is a detection model rather
than a segmentation model. The Y pipeline therefore uses true masks when
available and otherwise converts detected car bounding boxes into highlight
masks.

## Screenshots and logging

Screenshots are saved under `captures/` by default. The output directory can
be changed with:

```bash
python3 view2.py model.gltf --screenshot-dir screenshots
```

Runtime logs report model loading, inference start and completion, detected
car counts, screenshot paths, fallback from segmentation masks to detection
boxes, and flattening road levels.

## Command-line options

```bash
python3 view2.py [model.gltf] \
  --segmentation-model MODEL \
  --segformer-model MODEL \
  --visdrone-model MODEL \
  --confidence 0.25 \
  --screenshot-dir captures
```

## Dependencies

`requirements.txt` includes:

- NumPy
- Pillow
- pyglet
- trimesh
- PyTorch
- Transformers
- Ultralytics

## Validation completed

- Python syntax checks for the viewer and merger.
- CLI help checks.
- GLTF scene loading through trimesh.
- Mask2Former car-label filtering checks.
- SegFormer car-mask filtering checks.
- VisDrone `best.pt` detection fallback checks.
- Top-down camera toggle checks.
- Screenshot saving checks.
- Grey replacement and shared road-plane flattening checks.
