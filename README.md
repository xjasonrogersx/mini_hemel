# mini_hemel

---
Reality itself is not fully defined until it is observed.

The idea that the world is not concrete until observed lies at the heart of the famous thought experiment known as Schrödinger's cat. Until the box is opened, quantum mechanics suggests the system exists in a superposition of possibilities, with the cat both alive and dead in a mathematical sense. The act of observation appears to transform these possibilities into a single reality, raising the profound question of whether the world is fully defined before it is observed. This idea echoes the philosophy of George Berkeley, who argued that "to be is to be perceived," and resonates with John Archibald Wheeler's notion of a "participatory universe," in which observers play an active role in bringing reality into concrete existence. Whether observation creates reality or merely reveals it remains one of the deepest mysteries in both physics and philosophy
---

This project loads and annotates GLTF/GLB street scenes with a pyrender-based viewer and segmentation/export pipeline.

## Rendering model

The live viewer and the export path both render through pyrender instead of relying on a custom trimesh pixel loop. This keeps RGB screenshots and depth captures consistent because both come from the same render call.

The geometry and hit-testing remain in trimesh, but the final render output is produced by pyrender so the camera, depth buffer, screenshot path, and scene view stay aligned.

## Controls

- Orbit mode: left-drag orbit, middle-drag pan, right-drag/scroll zoom. This is the default viewer navigation behavior provided by pyrender.
- Walk mode: double-click on the ground to enter, then use the mouse to look and arrow keys to move forward/back/turn. This keeps the camera grounded on the scene surface while preserving the viewer’s street-level navigation behavior.
- G: toggle top-down view.
- Q: run Mask2Former segmentation on the current view.
- W: run SegFormer semantic segmentation.
- Y: run VisDrone detection on the current view.
- R: flatten detected car geometry back onto the road plane.
- S: save a timestamped RGB PNG and corresponding depth PNG.

## Environment setup

For headless or containerized use, set:

```bash
export PYOPENGL_PLATFORM=egl
```

This is required in the current dev container to initialize the OpenGL context reliably for pyrender offscreen rendering.

```bash
xhost +local:docker
docker run -it --name god -v /home/jason/work:/workspace  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --device /dev/dri  ubuntu:24.04

apt-get update
apt-get install -y libgl1 libglu1-mesa libx11-6 libxext6 libxrender1 mesa-utils
```

![alt text](images/image.png)

## Streat level

![alt text](images/street_level.pngimage.png)


![alt text](images/view1.png)
![alt text](images/view1-improved.png)
![alt text](images/view1-with-context.png)



## Geomenty form imags

https://huggingface.co/spaces/microsoft/TRELLIS.2


## 

RabbitMQ is a good fit for this asynchronous, potentially slow GPU job. For
large images or many concurrent clients, store images in object storage and
send only an object key through RabbitMQ; base64-encoded PNG messages are kept
here because the current viewer sends one screenshot at a time.

