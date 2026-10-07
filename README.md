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
| New texture render | Press `R` for SDXL or `T` for the configured generator |
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

## Texture workers

Press `R` in orbit mode for SDXL ControlNet, or `T` for the configured texture
generator. The viewer saves the current RGB/depth pair and sends it to the
selected worker.

For the Stable Diffusion 1.5 worker:

```bash
python3 model_server/controlnet_worker.py \
  --model runwayml/stable-diffusion-v1-5 \
  --controlnet lllyasviel/sd-controlnet-depth
python3 viewer.py
```

The generated image is saved in `captures/` as `view_controlnet_<timestamp>.png`.

## SDXL textured renders

Press `R` in the viewer for a direct SDXL render. This generates a higher
quality textured render with SDXL and
the SDXL depth ControlNet. Start RabbitMQ and the worker before launching the
viewer:

```bash
python3 model_server/sdxl_worker.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --controlnet diffusers/controlnet-depth-sdxl-1.0
python3 viewer.py
```

The generated image is saved in `captures/` as `view_sdxl_<timestamp>.png`.

## Web control panel

Starting `viewer.py` also starts a local control panel at
`http://127.0.0.1:8765/`. It exposes display and camera modes, configured
texture options, generation controls, and the entries in `artifacts.json`.
Each artifact shows its captured and generated images; **Navigate display**
restores the saved camera pose. Set `VIEWER_WEB_PORT` to use another port.

## Configured texture render

Press `T` to read `config.json` and use the configured texture generator. For
`runpod_nano_banana_2`, the viewer captures matching 4:3 RGB and depth images,
uploads them temporarily to the configured Cloudflare R2 bucket, calls the
RunPod edit endpoint, downloads the result to `captures/`, and deletes the
temporary bucket objects. The R2 public base URL must be reachable by RunPod.


<img width="1261" height="740" alt="image" src="https://github.com/user-attachments/assets/15857e07-fdf6-4675-97c6-6f0e44e6054e" />

<img width="1251" height="710" alt="image" src="https://github.com/user-attachments/assets/fd1af397-f8ae-4721-8f4c-25e22fe8bc34" />

<img width="1272" height="871" alt="image" src="https://github.com/user-attachments/assets/41105b23-0898-4469-a7eb-3b093284c871" />


```
python3 model_server/depth_anything_worker.py   --model depth-anything/Depth-Anything-V2-Small-hf   --rabbitmq-url amqp://guest:guest@192.168.1.220:5672/%2F
```
Quick and runs on laptop



```
python3 model_server/segformer_worker.py   --model nvidia/segformer-b0-finetuned-ade-512-512   --queue segformer   --rabbitmq-url amqp://guest:guest@192.168.1.220:5672/%2F
```
Quick and runs on laptop
It can segment categories such as:

Buildings: building, house, wall, windowpane, door, roof, skyscraper
Roads and terrain: road, sidewalk, grass, earth, sand, mountain, rock
Vegetation: tree, plant, palm, flower, bush
Vehicles: car, truck, bus, train, boat, airplane, bicycle, motorcycle
Furniture/interior: chair, table, bed, sofa, cabinet, desk, shelf
Objects: person, animal, sign, pole, lamp, fence, bridge, stairs


```
python3 model_server/mask2former_worker.py \
  --model facebook/mask2former-swin-small-ade-semantic \
  --queue mask2former \
  --rabbitmq-url amqp://guest:guest@192.168.1.220:5672/%2F
  ```

Grounding DINO detection and per-box SAM2 masks:

```bash
python3 model_server/grounding_dino_worker.py \
  --model IDEA-Research/grounding-dino-tiny \
  --queue grounding-dino \
  --rabbitmq-url amqp://guest:guest@192.168.1.220:5672/%2F



pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-headless
python3 model_server/sam2_worker.py \
  --model sam2_b.pt \
  --queue sam2 \
  --rabbitmq-url amqp://guest:guest@192.168.1.220:5672/%2F
```


```bash

cd /workspace
git clone https://github.com/lhwcv/mlsd_pytorch.git
mkdir -p /workspace/mlsd_pytorch/models
wget -O /workspace/mlsd_pytorch/models/mlsd_large_512_fp32.pth \
  https://github.com/lhwcv/mlsd_pytorch/raw/main/models/mlsd_large_512_fp32.pth


cd /workspace/mini_hemel
python3 model_server/mlsd_worker.py \
  --model /workspace/mlsd_pytorch/models/mlsd_large_512_fp32.pth \
  --source-dir /workspace/mlsd_pytorch \
  --queue mlsd \
  --input-size 512 \
  --rabbitmq-url 'amqp://guest:guest@192.168.1.220:5672/%2F'
```
# Semantic Reality Refinement
## Materialising a World Through Observation

### Origin

The project starts with a low-detail mesh extracted from Google Earth.

Traditionally, such a mesh would be treated as an approximation of a fixed ground truth world.

This proposal takes a different view:

> The LowLOD mesh is not the truth.
>
> It is a hint towards reality.

The mesh provides:

- Approximate geometry
- Building locations
- Road layouts
- Terrain structure
- Initial textures

Everything else remains open to interpretation.

---

# Philosophical Inspiration

## Schrödinger's Cat

In the famous thought experiment, the cat exists in a superposition of states until observed.

Applied to a semantic city model:

```text
LowLOD Building
    =
Many Possible Detailed Buildings
```

Observation gradually collapses these possibilities into a specific interpretation.

---

## Berkeley

George Berkeley proposed:

> "To be is to be perceived."

In this system:

- Unobserved regions remain loosely defined.
- Observation increases detail.
- Repeated observation increases certainty.

The city becomes progressively more concrete as it is explored.

---

## Wheeler's Participatory Universe

John Archibald Wheeler suggested that observers play a role in bringing reality into existence.

In the semantic city:

```text
Observer
    ->
Observation
    ->
Inference
    ->
Materialisation
```

Reality emerges through participation.

---

# Core Principle

Traditional Reconstruction:

```text
Reality
    ->
Images
    ->
Reconstruction
```

Semantic Reality Refinement:

```text
LowLOD World
     ->
Observation
     ->
Inference
     ->
World Update
     ->
Refined Reality
```

The goal is not:

> What really exists?

The goal is:

> What is the most believable and self-consistent world?

---

# The LowLOD Mesh

The LowLOD mesh acts as:

```text
Scaffold
```

rather than:

```text
Ground Truth
```

The mesh defines:

- A building probably exists here
- A road probably exists here
- A tree probably exists here

It does **not** define:

- Exact doors
- Exact windows
- Architectural details
- Shop displays
- Interior structures

These emerge over time.

---

# Semantic World Model

Instead of a world being represented as:

```text
Mesh
 └─ Triangles
```

Represent it as:

```text
World
 ├─ Entities
 ├─ Observations
 ├─ Hypotheses
 ├─ Canonical Facts
 └─ Geometry
```

---

# Entity Example

```yaml
entity:
  id: building_001

  class: building
  subtype: sweet_shop

  attributes:
    style: victorian
    occupancy: retail

  geometry:
    lod0_mesh: building.glb

  observations: [...]
  hypotheses: [...]
```

Geometry becomes a consequence of semantic understanding.

---

# Semantic Attributes

Objects can carry semantic information:

```yaml
type: building
subtype: sweet_shop

style: victorian
condition: maintained
era: 1890s
```

These attributes constrain future generation.

For example:

```text
Sweet Shop
```

supports:

- Display windows
- Signage
- Shelving
- Glass storefronts

while discouraging:

- Industrial equipment
- Warehousing structures
- Factory chimneys

---

# Observation Driven Reality

Observers do not reveal detail.

Observers create detail.

Example:

## At Long Range

The system sees:

```text
Retail Unit
```

Only coarse representation is required.

---

## At Medium Range

The system can infer:

```text
Sweet Shop
```

Additional semantic details become available.

---

## At Close Range

The system may generate:

```text
Window displays
Posters
Shelves
Product jars
```

Information density increases with observation distance.

---

# Example: Windows and Doors

Initially:

```text
Flat textured façade
```

After observation:

```text
AI texture enhancement
       +
semantic segmentation
```

detects:

```text
door
window
window
window
```

These detections become hypotheses.

Eventually:

```text
Window
```

is promoted into:

```text
Actual geometry
```

creating:

- Recesses
- Frames
- Window sills

---

# Unobserved Geometry

Traditional systems:

```text
Back of building
    =
Unknown
```

Proposed system:

```text
Back of building
    =
Field of possibilities
```

Potential details can be generated from:

- Architectural style
- Nearby buildings
- Semantic labels
- Previous observations

Observation resolves uncertainty.

---

# Reality States

The world exists at multiple certainty levels.

```text
Observed
Inference
Potential
```

or alternatively:

```text
L0  Seed Reality
L1  Semantic Reality
L2  Generated Reality
L3  Confirmed Reality
L4  Canonical Reality
```

Example:

```text
Building exists                L0
Sweet shop                     L1
Display window                 L2
Window frame geometry          L3
Interior shelves               L4
```

---

# Observation Objects

Observations are immutable evidence.

```yaml
observation:
  id: 123

  source: observer_a

  confidence: 0.87

  detects:
    - window
    - door
```

Observations are never edited.

They are historical records.

---

# Hypotheses

Observations generate hypotheses.

```yaml
hypothesis:
  type: chimney

  confidence: 0.61

  support:
    observations: 4

  state: candidate
```

Hypotheses are beliefs.

Not facts.

---

# Hypothesis Promotion

## State Machine

```text
Potential
    ↓
Candidate
    ↓
Probable
    ↓
Accepted
    ↓
Canonical
```

---

## Rule 1: Multi View Agreement

A feature observed from multiple viewpoints gains confidence.

```text
1 View
   -> Candidate

3 Views
   -> Probable

5 Views
   -> Accepted

Repeated Observation
   -> Canonical
```

---

## Rule 2: Semantic Consistency

Generated content must align with known semantic attributes.

Example:

```text
Sweet Shop
```

supports:

```text
display windows
shop signs
glass frontage
```

---

## Rule 3: Geometric Consistency

Observations projected back into 3D should agree spatially.

If multiple observations intersect:

```text
Promote confidence
```

If they disagree:

```text
Conflict
```

---

## Rule 4: Temporal Stability

Features repeatedly observed over time become increasingly trusted.

---

# Drift

The greatest challenge is:

```text
Drift
```

Example:

Iteration 1

```text
Small chimney
```

Iteration 5

```text
Large chimney
```

Iteration 10

```text
Church tower
```

The AI gradually moves away from plausibility.

---

# Preventing Drift

Never directly bake generated geometry.

Avoid:

```text
Generate
    ->
Bake Mesh
```

Prefer:

```text
Generate
    ->
Hypothesis
    ->
Validation
    ->
Promotion
    ->
Mesh Update
```

Reality emerges gradually.

---

# Multi Observer Conflicts

Observer A:

```text
Chimney
```

Observer B:

```text
Skylight
```

Both plausible.

Both fit the LowLOD mesh.

Store both.

```yaml
chimney:
  confidence: 0.51

skylight:
  confidence: 0.49
```

The world remains unresolved until additional evidence arrives.

---

# Consensus Reality

The mesh is not truth.

The mesh is consensus.

Reality becomes:

```text
Most Self Consistent Explanation
```

rather than:

```text
Absolute Truth
```

---

# The Matrix and Déjà Vu

In *The Matrix*, the repeated black cat represented a change in the underlying simulation.

Applied here:

Suppose a feature becomes canonical.

```text
Chimney
```

Later observations prove:

```text
Skylight
```

The accepted world model changes.

This can be recorded as:

```text
Reality Revision Event
```

or:

```text
Déjà Vu Event
```

A signal that consensus reality has been rewritten.

---

# The Backrooms and the Green Glow

The Backrooms often imply that large regions exist only partially defined.

The persistent greenish illumination is interesting because:

```text
Lighting exists
before geometry exists
```

The observer experiences:

- Atmosphere
- Scale
- Mood
- Illumination

before the space is fully materialised.

Similarly:

```text
Semantic Constraints
```

can exist before:

```text
Detailed Geometry
```

---

# Resolution Rather Than Revelation

The key insight is:

Objects are not necessarily revealed.

They are resolved.

Like progressive texture streaming:

```text
Low Detail
    ->
Medium Detail
    ->
High Detail
```

Reality itself gains fidelity as observation increases.

---

# The Living World Model

The final architecture becomes:

```text
LowLOD Mesh
      +
Semantic Knowledge
      +
Observation History
      +
Hypothesis Graph
      +
Generated Detail
      +
Canonical Consensus
```

where:

```text
Geometry is not truth.

Observations are evidence.

Hypotheses are belief.

Consensus becomes reality.
```

The city is never completely defined.

It continuously materialises around observers, refining itself toward the most believable and self-consistent interpretation of the world.
