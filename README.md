# mini_hemel

---
Reality itself is not fully defined until it is observed.

The idea that the world is not concrete until observed lies at the heart of the famous thought experiment known as Schrödinger's cat. Until the box is opened, quantum mechanics suggests the system exists in a superposition of possibilities, with the cat both alive and dead in a mathematical sense. The act of observation appears to transform these possibilities into a single reality, raising the profound question of whether the world is fully defined before it is observed. This idea echoes the philosophy of George Berkeley, who argued that "to be is to be perceived," and resonates with John Archibald Wheeler's notion of a "participatory universe," in which observers play an active role in bringing reality into concrete existence. Whether observation creates reality or merely reveals it remains one of the deepest mysteries in both physics and philosophy

---

## Setup 

```bash
# On the host
xhost +local:
docker run -it --name god \
  -v /home/jason/work:/workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --device /dev/dri \
  ubuntu:24.04

# Inside the container
apt-get update && apt-get install -y \
  python3 python3-pip \
  libgl1 libglx0 libegl1 libgl1-mesa-dri \
  libglu1-mesa \
  libx11-6 libxext6 libxi6 libxfixes3 libxrandr2 libxxf86vm1
pip config set global.break-system-packages true
cd /workspace/3d_test
pip install -r requirements.txt
# pyrender 0.1.45 pins the incompatible PyOpenGL 3.1.0 release.
pip install --upgrade --force-reinstall \
 'PyOpenGL>=3.1.10' 'PyOpenGL_accelerate>=3.1.10'
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


## Orbit mode navigation

The viewer currently uses **Google Earth-style orbit controls**.

| Action | Control |
| --- | --- |
| Orbit / tilt | Shift + left mouse button drag |
| New texture render | Right click, then choose a worker |
| Zoom | Scroll wheel |

### Comparison matrix

| Action | SketchUp | Blender (Default) | Google Earth |
| --- | --- | --- | --- |
| Orbit / tilt | Middle mouse button drag | Middle mouse button drag | Shift + left mouse button drag (or middle mouse button drag) |
| Pan | Shift + middle mouse button drag | Shift + middle mouse button drag | Left mouse button drag |
| Zoom | Scroll wheel | Scroll wheel | Scroll wheel (or right mouse button drag up/down) |

Press `G` to return to orbit mode and `F` to toggle fullscreen. Press `Q` to
quit the viewer.

## Walk mode navigation

Walk mode uses basic Quake-style movement controls:

| Action | Control |
| --- | --- |
| Move forward | Up arrow |
| Move backward | Down arrow |
| Strafe left | Left arrow |
| Strafe right | Right arrow |
| Turn left/right in windowed mode | `Z` / `X` |
| Look around in fullscreen mode | Move the mouse freely |
| Look around in windowed mode | Hold the left mouse button and drag |
| Raise/lower camera height | `H` / `L` |
| Return to orbit mode | `G` |
| Toggle fullscreen | `F` |

The window title shows the current walk height in metres.

## Texture worker picker

Right-click in orbit mode and choose Stable Diffusion 1.5 ControlNet, SDXL
ControlNet, or Flux depth ControlNet. The viewer saves the current RGB/depth
pair and sends it to the selected RabbitMQ worker.

For the Stable Diffusion 1.5 worker:

```bash
python3 model_server/controlnet_worker.py \
  --model runwayml/stable-diffusion-v1-5 \
  --controlnet lllyasviel/sd-controlnet-depth
python3 viewer.py
```

The generated image is saved in `captures/` as `view_controlnet_<timestamp>.png`.

## SDXL textured renders

Press `R` in the viewer for a direct SDXL render, or choose SDXL from the
worker picker. This generates a higher quality textured render with SDXL and
the SDXL depth ControlNet. Start RabbitMQ and the worker before launching the
viewer:

```bash
python3 model_server/sdxl_worker.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --controlnet diffusers/controlnet-depth-sdxl-1.0
python3 viewer.py
```

The generated image is saved in `captures/` as `view_sdxl_<timestamp>.png`.

## Configured texture render

Press `T` to read `config.json` and use the configured texture generator. For
`runpod_nano_banana_2`, the viewer captures matching 4:3 RGB and depth images,
uploads them temporarily to the configured Cloudflare R2 bucket, calls the
RunPod edit endpoint, downloads the result to `captures/`, and deletes the
temporary bucket objects. The R2 public base URL must be reachable by RunPod.


<img width="1261" height="740" alt="image" src="https://github.com/user-attachments/assets/15857e07-fdf6-4675-97c6-6f0e44e6054e" />

<img width="1251" height="710" alt="image" src="https://github.com/user-attachments/assets/fd1af397-f8ae-4721-8f4c-25e22fe8bc34" />

<img width="1272" height="871" alt="image" src="https://github.com/user-attachments/assets/41105b23-0898-4469-a7eb-3b093284c871" />


python3 model_server/depth_anything_worker.py   --model depth-anything/Depth-Anything-V2-Small-hf   --rabbitmq-url amqp://guest:guest@192.168.1.220:5672/%2F

python3 model_server/segformer_test.py    --rabbitmq-url amqp://guest:guest@192.168.1.220:5672/%2F

