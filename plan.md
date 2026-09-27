# mini_hemel implementation plan

## Goal

Convert the viewer to a pyrender-based rendering flow while keeping the existing trimesh scene logic, interaction tools, and segmentation/export actions intact.

## Completed implementation

### 1. Rendering backend
- Kept trimesh as the source of geometry and hit-testing logic.
- Switched the live render path and screenshot export path to use pyrender.
- Ensured RGB and depth exports come from the same render call for consistency.

### 2. Scene and camera sync
- Reused the existing trimesh scene camera transform as the authoritative camera state.
- Synced the pyrender camera node and viewer trackball state with the trimesh camera pose.
- Added helper logic to update both scene and renderer state together.

### 3. Camera modes
- Orbit mode is the default viewer behavior for broad model inspection.
- Walk mode is the street-level navigation mode used for walking through the scene.
- Preserved top-down toggle behavior with a dedicated camera pose restore path.

#### Orbit mode controls
- Left drag: orbit around the scene center.
- Middle drag: pan the camera.
- Right drag: zoom / dolly in or out.
- Mouse wheel: zoom.
- Shift + left drag: pan.
- Ctrl + left drag: roll.
- Ctrl + Shift + left drag: zoom.

#### Walk mode controls
- Double-click on the ground: enter walk mode.
- Mouse drag: look around.
- Up / Down arrows: move forward and backward.
- Left / Right arrows: turn left and right.
- H / L: move vertically up and down.
- G: leave walk mode and toggle top-down view.

### 4. App actions
- Kept the segmentation shortcuts working:
  - Q: Mask2Former
  - W: SegFormer
  - Y: VisDrone
  - R: flatten detected car geometry
  - S: save RGB + depth image
  - G: toggle top-down view
- Kept the viewer caption and message text behavior working with pyrender's update flow.

### 5. Compatibility fixes
- Added NumPy 2 compatibility shims for older pyrender assumptions.
- Avoided assigning into pyrender's internal `scene` property and instead stored the trimesh scene separately.
- Patched the pyglet event loop to iterate over a snapshot of the window set to avoid the weak-set mutation crash.
- Set the viewer environment to use a proper visible X11 display path rather than a forced headless context.

## Verification

- `python3 -m py_compile view2.py` succeeded.
- Runtime launch no longer crashes in the container with the pyglet weak-set mutation issue.
- Depth exports were successfully saved to the captures folder during render validation.

## Notes

- The viewer is intended to run with a real desktop/X11 display so the window is visible to the user.
- In a headless-only environment, rendering may continue without a visible GUI unless a desktop session or X11 forwarding is configured.
