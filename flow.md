# Mini Hemel Model Flow

This document describes the current viewer, model, RabbitMQ, capture, and artifact flow.

## Overview

The viewer creates or loads a 3D scene, captures an image from the current camera, and stores generated outputs under `captures/`. Model operations are started from the viewer or browser control panel. RabbitMQ-backed operations use the global `rabbitmq_url` in `config.json` when their section has `"flow": "rabbitmq"`.

```mermaid
flowchart TD
    Viewer[viewer.py] --> Capture[RGB/depth capture]
    Capture --> Texture[Texture generation]
    Texture --> Result[result_render in captures/]
    Result --> Depth[Depth Anything]
    Result --> Seg[SegFormer]
    Result --> Mask[Mask2Former]
    Result --> DINO[Grounding DINO]
    DINO --> Boxes[Prompt-grouped bounding boxes]
    Boxes --> SAM[SAM2 per box or all boxes]
    Depth --> Artifacts[artifacts.json]
    Seg --> Artifacts
    Mask --> Artifacts
    DINO --> Artifacts
    SAM --> Artifacts
    Texture --> Artifacts
    RabbitMQ[(RabbitMQ)] --> Depth
    RabbitMQ --> Seg
    RabbitMQ --> Mask
    RabbitMQ --> DINO
    RabbitMQ --> SAM
    R2[(Cloudflare R2)] --> Texture
    Texture --> RunPod[RunPod texture model]
```

## Configuration

The global broker is configured at the top level:

```json
{
  "rabbitmq_url": "amqp://...",
  "depth_generator": {"flow": "rabbitmq", "queue": "depth-anything"},
  "segmentation_generator": {"flow": "rabbitmq", "queue": "segformer"},
  "mask2former_generator": {"flow": "rabbitmq", "queue": "mask2former"},
  "grounding_dino_generator": {"flow": "rabbitmq", "queue": "grounding-dino"},
  "sam2_generator": {"flow": "rabbitmq", "queue": "sam2"}
}
```

`texture_generator` is the exception. It does not use RabbitMQ in the current viewer flow. It uses the configured RunPod model and temporary R2 objects.

The real `config.json` contains credentials. Do not copy those values into documentation or commit them to a public repository.

## 1. Texture Generation

Configured section:

- Model: `runpod_nano_banana_2`
- Resolution: `1k`
- Aspect ratio: `4:3`
- Prompt: configured in `texture_generator.prompt`
- Storage: Cloudflare R2 for temporary RGB/depth inputs
- Remote service: RunPod edit endpoint

Flow:

1. The viewer captures the current scene as RGB and depth images.
2. The viewer uploads temporary inputs to R2.
3. The viewer calls the RunPod texture generator.
4. The output image is downloaded into `captures/`.
5. Temporary R2 objects are deleted.
6. The output is stored in the artifact as `result_render`.

The configured texture generator is started with `T` in the viewer or through the configured generation control.

## 2. Depth Anything

Configured section:

- Flow: `rabbitmq`
- Model: `depth-anything/Depth-Anything-V2`
- Queue: `depth-anything`

The viewer sends the captured/generated image as base64 through RabbitMQ. The depth worker returns an encoded depth image. The viewer stores the result in `generated_depth_render` or the corresponding depth output field used by the current worker path.

This stage is also used as part of the texture-generation input flow when a matching depth image is required.

## 3. SegFormer

Configured section:

- Flow: `rabbitmq`
- Model: `nvidia/segformer-b0-finetuned-ade-512-512`
- Queue: `segformer`
- Requested classes:
  - `building`
  - `wall`
  - `house`
  - `roof`
  - `window`
  - `door`
  - `tree`
  - `street`
  - `sidewalk`

The viewer sends the artifact image and requested classes to the SegFormer worker. The response contains a color map, label map, and class masks. These are saved into `captures/` and referenced from the artifact record as:

- `segmentation_color_map`
- `segmentation_label_map`
- `segmentation_masks`

The operation is available as **Run / rerun SegFormer** on an artifact page.

## 4. Mask2Former

Configured section:

- Flow: `rabbitmq`
- Model: `facebook/mask2former-swin-small-ade-semantic`
- Queue: `mask2former`
- Requested classes: the same architectural and outdoor class list used by SegFormer

The viewer sends the artifact image and classes to the Mask2Former worker. The response is saved to `captures/` and referenced by:

- `mask2former_color_map`
- `mask2former_label_map`
- `mask2former_masks`

The operation is available as **Run / rerun Mask2Former** on an artifact page.

## 5. Grounding DINO

Configured section:

- Flow: `rabbitmq`
- Model: `IDEA-Research/grounding-dino-tiny`
- Queue: `grounding-dino`
- Detection threshold: `0.35`
- Text threshold: `0.25`

The browser supplies a text prompt such as `windows` or `doors`. The viewer sends the generated result image and prompt to Grounding DINO. The worker returns bounding boxes, labels, and confidence scores.

The viewer stores detections grouped by normalized prompt:

```json
"grounding_dino_detections": {
  "windows": [
    {
      "prompt": "windows.",
      "label": "windows",
      "score": 0.62,
      "box_xyxy": [80.3, 478.5, 130.6, 540.9]
    }
  ],
  "doors": []
}
```

Running the same prompt replaces only that prompt group. Running another prompt preserves the existing groups.

The artifact page displays every stored box over the artifact's `result_render` image. It also supports deleting an individual detection.

## 6. SAM2

Configured section:

- Flow: `rabbitmq`
- Model: local `sam2_b.pt`
- Queue: `sam2`

SAM2 uses the generated result image and Grounding DINO boxes as prompts. There are two current modes:

### Per-box SAM2

Each detection row has **Run / rerun SAM2**. The viewer sends one bounding box to the SAM2 worker and stores the returned mask filename on the matching detection:

```json
{
  "sam2_mask": "sam2_dino_box_0_YYYYMMDD_HHMMSS_microseconds.png"
}
```

### All-boxes SAM2

The Grounding DINO section also has **Run SAM2 on all boxes**. The viewer sends all stored boxes in one SAM2 request, then maps returned masks back to their prompt group and detection index.

The all-boxes action is useful when a prompt produced many detections and each box needs a segmentation mask.

## Artifact Storage

`artifacts.json` stores metadata and references to files in `captures/`. Model images and masks are stored as filenames rather than embedded base64 data.

Typical artifact fields include:

- `texture_render`: original/generated texture input or render
- `result_render`: final generated result image used for DINO and SAM2
- `depth_render` or `generated_depth_render`: depth output
- `segmentation_color_map`
- `segmentation_label_map`
- `segmentation_masks`
- `mask2former_color_map`
- `mask2former_label_map`
- `mask2former_masks`
- `grounding_dino_detections`: prompt-grouped detections
- `sam2_mask`: per-detection mask filename
- `camera_pose`: saved camera state for navigation

## Browser Control Flow

The viewer starts the local web server at `http://127.0.0.1:8765/` by default. Artifact pages expose these operations:

- Navigate to the saved camera pose
- Run SegFormer
- Run Mask2Former
- Run Grounding DINO with a text prompt
- Run SAM2 for one Grounding DINO box
- Run SAM2 for all Grounding DINO boxes
- Delete a Grounding DINO detection

The browser sends commands to `/api/control`. The viewer queues commands onto the render thread, starts model work in background threads, and updates `artifacts.json` when results are saved.

## Logging

The viewer logs outgoing RabbitMQ work with the queue, correlation ID, and request field names. The Grounding DINO and SAM2 workers log when work is received.

SAM2 additionally logs these phases:

1. Processing started
2. Image decoded
3. Inference started
4. Inference returned with duration
5. Processing completed with detection count and duration

These logs distinguish a RabbitMQ delivery problem from a slow model inference.

## Worker Commands

Use the global RabbitMQ URL when starting workers. Examples:

```bash
python3 model_server/depth_anything_worker.py \
  --model depth-anything/Depth-Anything-V2-Small-hf \
  --rabbitmq-url amqp://guest:guest@192.168.1.220:5672/%2F

python3 model_server/segformer_worker.py \
  --model nvidia/segformer-b0-finetuned-ade-512-512 \
  --queue segformer \
  --rabbitmq-url amqp://guest:guest@192.168.1.220:5672/%2F

python3 model_server/mask2former_worker.py \
  --model facebook/mask2former-swin-small-ade-semantic \
  --queue mask2former \
  --rabbitmq-url amqp://guest:guest@192.168.1.220:5672/%2F

python3 model_server/grounding_dino_worker.py \
  --model IDEA-Research/grounding-dino-tiny \
  --queue grounding-dino \
  --rabbitmq-url amqp://guest:guest@192.168.1.220:5672/%2F

python3 model_server/sam2_worker.py \
  --model sam2_b.pt \
  --queue sam2 \
  --rabbitmq-url amqp://guest:guest@192.168.1.220:5672/%2F
```

For the SAM2 worker, use `opencv-python-headless` in headless containers instead of the GUI-enabled OpenCV package.
