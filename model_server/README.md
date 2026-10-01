# Model Server

These workers run on the machine with the larger memory/GPU. The viewer sends
jobs through RabbitMQ and receives results through RabbitMQ RPC replies.

The workers use separate queues so they can run independently:

| Worker | Script | Queue | Result |
| --- | --- | --- | --- |
| Stable Diffusion | `diffusion_worker.py` | `stable-diffusion` | Generated PNG |
| Stable Diffusion 1.5 | `d1.5_worker.py` | `stable-diffusion-15` | Generated PNG |
| Qwen Image 2.1 | `qwenimage_worker.py` | `qwen-image` | Generated PNG |
| ControlNet (SD 1.5 + depth) | `controlnet_worker.py` | `stable-diffusion-controlnet` | Generated PNG |
| SDXL ControlNet (SDXL + depth) | `sdxl_worker.py` | `stable-diffusion-controlnet-sdxl` | Generated PNG |
| Flux ControlNet (Flux + depth) | `flux_controlnet_worker.py` | `flux-controlnet-depth` | Generated PNG |
| Depth Anything V2 | `depth_anything_worker.py` | `depth-anything` | 16-bit depth PNG |
| SegFormer ADE20K | `segformer_worker.py` | `segformer` | Semantic label map and masks |
| Ultralytics SAM3 | `sam3_worker.py` | `sam3` | Detection metadata and masks |
| Ultralytics SAM2 | `sam2_worker.py` | `sam2` | Detection metadata and masks |

### Flux depth ControlNet

`flux_controlnet_worker.py` uses `black-forest-labs/FLUX.1-dev` with the XLabs
depth adapter. The XLabs repository contains the raw
`flux-depth-controlnet-v3.safetensors` adapter rather than a Diffusers
`FluxControlNetModel` directory. The worker therefore expects `--controlnet`
to point to a converted Diffusers directory:

```bash
python3 flux_controlnet_worker.py \
	--model black-forest-labs/FLUX.1-dev \
	--controlnet /opt/models/flux-depth-controlnet-v3-diffusers \
	--queue flux-controlnet-depth \
	--rabbitmq-url amqp://guest:guest@HOST:5672/%2F
```

The raw XLabs checkpoint cannot be passed directly to Diffusers 0.40.0 because
that version has no single-file loader for `FluxControlNetModel`.

### SegFormer semantic segmentation

`segformer_worker.py` uses `nvidia/segformer-b0-finetuned-ade-512-512`, whose
ADE20K labels include `tree`, `windowpane`, `door`, `road`, and `sidewalk`.
The worker returns a complete label map and color visualization, plus binary
masks for requested classes. The request can use `street`; it is mapped to the
ADE20K `road` class.

```bash
python3 segformer_worker.py \
	--model nvidia/segformer-b0-finetuned-ade-512-512 \
	--queue segformer \
	--rabbitmq-url amqp://guest:guest@HOST:5672/%2F
```

Request body:

```json
{
	"image_base64": "<PNG bytes encoded as base64>",
	"classes": ["tree", "window", "door", "street", "sidewalk"]
}
```

Each returned mask includes `class_id`, `pixel_count`, `bbox_xyxy`, and a
`mask_base64` grayscale PNG. The response also includes
`label_map_base64`, `color_map_base64`, `class_map`, and `present_classes`.

Run the bundled test using the `segmentation_generator` settings from the
root `config.json`:

```bash
python3 segformer_test.py
```

Results are written to `model_server/test/segformer_output/`. Override the
input image or requested classes with `--image` and `--classes`.

### Depth Anything V2

`depth_anything_worker.py` uses the Transformers implementation of Depth
Anything V2. The default checkpoint is the small Hugging Face model, which is
appropriate for a shared GPU worker. Start it with:

```bash
python3 depth_anything_worker.py \
	--model depth-anything/Depth-Anything-V2-Small-hf \
	--queue depth-anything \
	--rabbitmq-url amqp://guest:guest@HOST:5672/%2F
```

With the worker running, send the bundled test image through RabbitMQ:

```bash
python3 depth_anything_test.py \
	--rabbitmq-url amqp://guest:guest@HOST:5672/%2F
```

The request contains an RGB image:

```json
{
	"image_base64": "<PNG bytes encoded as base64>"
}
```

Successful responses contain a normalized 16-bit PNG depth image. Larger
values represent larger model-predicted depth values:

```json
{
	"ok": true,
	"width": 800,
	"height": 600,
	"depth_image_base64": "<16-bit PNG bytes encoded as base64>",
	"depth_min": 0,
	"depth_max": 65535
}
```

When the viewer uses the configured `runpod_nano_banana_2` texture generator,
it automatically sends the returned texture image to this worker. The generated
depth image is saved in `captures/` and referenced by `generated_depth_render`
in `artifacts.json`.

## RabbitMQ message format

Both workers receive a UTF-8 JSON request message. The AMQP message properties
must include `reply_to` and a unique `correlation_id`; the response copies the
correlation ID. The request is acknowledged only after processing finishes.

### Stable Diffusion request

```json
{
	"image_base64": "<PNG bytes encoded as base64>",
	"model": "optional model identifier",
	"prompt": "high quality photorealistic street-level 3D reconstruction",
	"negative_prompt": "changed camera angle, warped geometry",
	"steps": 30,
	"strength": 0.2,
	"guidance_scale": 4.5,
	"seed": 0
}
```

Successful response:

```json
{
	"ok": true,
	"image_base64": "<generated PNG bytes encoded as base64>"
}
```

### Stable Diffusion 1.5 request

The SD 1.5 worker uses the same image-to-image JSON format, but listens on the
separate `stable-diffusion-15` queue. It uses `float16` on CUDA, which is more
suitable for GPUs such as the RTX 2060 or RTX 3060 with limited VRAM.

```json
{
	"image_base64": "<PNG bytes encoded as base64>",
	"prompt": "high quality photorealistic street-level 3D reconstruction",
	"negative_prompt": "changed camera angle, warped geometry",
	"steps": 30,
	"strength": 0.2,
	"guidance_scale": 7.5,
	"seed": 0
}
```

The successful response is the same generated-PNG format shown above.

### Qwen Image 2.1 request

Qwen requests use the same RabbitMQ RPC properties and image encoding as Stable
Diffusion. The worker accepts a local Diffusers model directory or a model ID.
The worker maps `guidance_scale` to Qwen's `true_cfg_scale` argument. Qwen
Image 2.1 is text-to-image; its pipeline does not use the input image,
`strength`, or image-to-image editing arguments.

```json
{
	"image_base64": "<PNG bytes encoded as base64>",
	"prompt": "high quality photorealistic street-level 3D reconstruction",
	"negative_prompt": "changed camera angle, warped geometry",
	"steps": 30,
	"strength": 0.2,
	"guidance_scale": 4.0,
	"seed": 0
}
```

Successful response:

```json
{
	"ok": true,
	"image_base64": "<generated PNG bytes encoded as base64>"
}
```

### SAM3 request

`image_base64` is required. The prompt fields are optional and should match
the Ultralytics SAM3 prediction API. Use points and labels together for point
prompting, or use bounding boxes for box prompting.

```json
{
	"image_base64": "<PNG bytes encoded as base64>",
	"points": [[320, 240]],
	"labels": [1],
	"bboxes": [[100, 80, 500, 400]],
	"text": "car",
	"conf": 0.25
}
```

### SAM2 request

The SAM2 request uses the same image and detection response format as SAM3.
`image_base64` is required. `points` and `labels` are used together for point
prompting; `bboxes` is used for box prompting. All prompt fields are optional.

```json
{
	"image_base64": "<PNG bytes encoded as base64>",
	"points": [[320, 240]],
	"labels": [1],
	"bboxes": [[100, 80, 500, 400]],
	"conf": 0.25
}
```

Successful responses contain `width`, `height`, and one detection entry per
returned mask:

```json
{
	"ok": true,
	"width": 800,
	"height": 600,
	"detections": [
		{
			"mask_base64": "<binary PNG mask, white is inside the mask>",
			"box_xyxy": [100.0, 80.0, 500.0, 400.0],
			"confidence": 0.98,
			"class_id": 0
		}
	]
}
```

Successful response:

```json
{
	"ok": true,
	"width": 800,
	"height": 600,
	"detections": [
		{
			"mask_base64": "<binary PNG mask, white is inside the mask>",
			"box_xyxy": [100.0, 80.0, 500.0, 400.0],
			"confidence": 0.98,
			"class_id": 0
		}
	]
}
```

For either worker, failures use:

```json
{"ok": false, "error": "description of the failure"}
```

## 1. Copy and extract the model

Copy `models--stabilityai--stable-diffusion-3.5-medium.tar.gz` to this machine.
For example, from the machine that has the archive:

```bash
scp /root/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3.5-medium.tar.gz \
	user@gpu-machine:/opt/models/
```

On the model server, extract it into `/opt/models`:

```bash
sudo mkdir -p /opt/models
sudo tar -xzf /opt/models/models--stabilityai--stable-diffusion-3.5-medium.tar.gz \
	-C /opt/models
```

The archive contains the Hugging Face cache layout. Locate the actual model
snapshot with:

```bash
find /opt/models/models--stabilityai--stable-diffusion-3.5-medium/snapshots \
	-mindepth 1 -maxdepth 1 -type d -print
```

Set `MODEL_PATH` to the printed snapshot directory. It should contain files
such as `config.json`, model component directories, and tokenizer files:

```bash
export MODEL_PATH=/opt/models/models--stabilityai--stable-diffusion-3.5-medium/snapshots/<snapshot-id>
test -f "$MODEL_PATH/model_index.json" || test -f "$MODEL_PATH/config.json"
```

Do not pass the `.tar.gz` file or the cache directory itself to `--model`.

## 2. Install the worker

From this `model_server` directory, create a virtual environment and install
the worker dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch transformers accelerate pika Pillow
# Qwen Image 2.1 requires the current Diffusers source version.
python3 -m pip install --upgrade git+https://github.com/huggingface/diffusers.git
```

Install a CUDA-enabled PyTorch build if this machine has an NVIDIA GPU. Verify
that PyTorch can see it:

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 3. Start RabbitMQ

RabbitMQ must be reachable from both the model server and the viewer. A quick
Docker setup for a local RabbitMQ instance is:

```bash
docker run -d --name rabbitmq \
	-p 5672:5672 -p 15672:15672 \
	rabbitmq:3-management
```

The default worker URL is:

```text
amqp://guest:guest@localhost:5672/%2F
```

The default `guest` account is normally restricted to local connections. For a
worker and viewer on different machines, create a RabbitMQ user and use its
host address in both commands instead.

### ControlNet worker

The viewer's `Q` key uses the `stable-diffusion-controlnet` queue. Install the
additional worker packages in the model-server environment:

```bash
source .venv/bin/activate
python3 -m pip install diffusers accelerate torchvision
```

When the viewer or worker is running inside a Docker container and RabbitMQ
was published by Docker on the host, `localhost` points to the container, not
the host. Use the Docker host gateway address. In this development container,
the gateway is `172.17.0.1`:

```bash
python3 controlnet_worker.py \
	--rabbitmq-url amqp://guest:guest@172.17.0.1:5672/%2F \
	--queue stable-diffusion-controlnet
```

Start the viewer with the same broker URL:

```bash
RABBITMQ_URL=amqp://guest:guest@172.17.0.1:5672/%2F python3 ../viewer.py
```

Alternatively, run the worker and viewer on the same host as RabbitMQ and use
the default `localhost` URL. Right-click in the viewer and choose ControlNet
after the worker reports that it is waiting for requests. The generated image is saved in `captures/`. The worker
logs the device, request settings, input modes, effective/requested iterations,
and inference duration. A successful response includes the same metadata:

```json
{
	"ok": true,
	"image_base64": "<generated PNG bytes encoded as base64>",
	"iterations": 6,
	"requested_iterations": 30,
	"duration_seconds": 18.2
}
```

To compare the worker with the Hugging Face depth example, use the standalone
test client. It defaults to depth-only text-to-image, so the RGB image is not
used as the generation source:

```bash
RABBITMQ_URL=amqp://guest:guest@192.168.1.252:5672/%2F \
python3 controlnet_test.py
```

Use `--mode img2img` to start from the RGB input instead. The worker must be
restarted after changing its code.

### SDXL ControlNet worker

The viewer's `R` key, or the SDXL option in the left-click worker picker, uses
the `stable-diffusion-controlnet-sdxl` queue. This
worker upgrades the ControlNet worker above to
`stabilityai/stable-diffusion-xl-base-1.0` with
`diffusers/controlnet-depth-sdxl-1.0`, uses a lower default denoising
strength (`0.18`) to better preserve mesh geometry, and enables additional
memory optimizations (`enable_model_cpu_offload`, `enable_attention_slicing`,
`enable_vae_slicing`, `enable_vae_tiling`) to fit GPUs with as little as 8GB
of VRAM, such as an RTX 3060 Ti. Install the same additional worker packages
used by the ControlNet worker:

```bash
source .venv/bin/activate
python3 -m pip install diffusers accelerate torchvision
```

Start the worker:

```bash
python3 sdxl_worker.py \
	--rabbitmq-url ******172.17.0.1:5672/%2F \
	--queue stable-diffusion-controlnet-sdxl
```

The worker prints:

```text
Waiting for SDXL requests on stable-diffusion-controlnet-sdxl
```

Press `R` after the worker reports that it is waiting for requests. The
generated image is saved in `captures/` as `view_sdxl_<timestamp>.png`. The
response format is identical to the ControlNet worker's response shown above.

Use the standalone test client to compare the worker directly through
RabbitMQ, without going through the viewer:

```bash
RABBITMQ_URL=******192.168.1.252:5672/%2F \
python3 sdxl_test.py
```

Use `--mode img2img` to start from the RGB input instead. The worker must be
restarted after changing its code.

## 4. Start the worker

Use the snapshot path discovered above:

```bash
source .venv/bin/activate
python3 diffusion_worker.py \
	--model "$MODEL_PATH" \
	--rabbitmq-url amqp://guest:guest@localhost:5672/%2F \
	--queue stable-diffusion
```

The first startup loads the model and can take several minutes. A successful
worker prints a message similar to:

```text
Waiting for diffusion requests on queue stable-diffusion
```

Keep this process running. It loads the model once and then handles requests
one at a time.

## Start the Stable Diffusion 1.5 worker

Install the same Diffusers dependencies, then start the smaller img2img worker:

```bash
source .venv/bin/activate
python3 -m pip install torch diffusers transformers accelerate pika Pillow
python3 d1.5_worker.py \
	--rabbitmq-url amqp://guest:guest@LXP-J-ROGERS2:5672/%2F \
	--queue stable-diffusion-15
```

Add `--cpu` to force CPU inference, for example when GPU memory is exhausted:

```bash
python3 d1.5_worker.py \
	--cpu \
	--rabbitmq-url amqp://guest:guest@LXP-J-ROGERS2:5672/%2F \
	--queue stable-diffusion-15
```

The default model is `runwayml/stable-diffusion-v1-5`. To use a downloaded
local Diffusers directory, pass it with `--model`:

```bash
python3 d1.5_worker.py \
	--model /opt/models/stable-diffusion-v1-5 \
	--rabbitmq-url amqp://guest:guest@LXP-J-ROGERS2:5672/%2F \
	--queue stable-diffusion-15
```

The worker prints:

```text
Waiting for Stable Diffusion 1.5 requests on queue stable-diffusion-15
```

## Test Stable Diffusion 1.5 through RabbitMQ

With `d1.5_worker.py` running, send an input image through the same broker:

```bash
source .venv/bin/activate
python3 sd1.5_test.py /path/to/test-image.jpg \
	--rabbitmq-url amqp://guest:guest@LXP-J-ROGERS2:5672/%2F \
	--queue stable-diffusion-15 \
	--prompt "photorealistic street scene with sharp natural details" \
	--output sd15_test_output.png \
	--display
```

`--output` and `--display` are optional. The test waits for the RPC response,
writes the generated PNG when requested, and can open it in the system image
viewer.

## Start the Qwen Image worker

Install the same base dependencies used by the diffusion worker:

```bash
source .venv/bin/activate
python3 -m pip install torch transformers accelerate pika Pillow
# Qwen Image 2.1 requires the current Diffusers source version.
python3 -m pip install --upgrade git+https://github.com/huggingface/diffusers.git
```

Start with the Qwen Image 2.1 model ID:

```bash
python3 qwenimage_worker.py \
	--model Qwen/Qwen-Image-2.1 \
	--rabbitmq-url amqp://guest:guest@LXP-J-ROGERS2:5672/%2F \
	--queue qwen-image
```

For an already downloaded model, replace the model ID with its local Diffusers
directory. The worker prints:

```text
Waiting for Qwen Image requests on queue qwen-image
```

Use a Qwen-compatible client that publishes to `qwen-image`; the response is
the same generated-PNG format documented above.

## Test Qwen Image through RabbitMQ

Keep `qwenimage_worker.py` running, then send a test image through RabbitMQ:

```bash
source .venv/bin/activate
python3 qwenimage_test.py /path/to/test-image.jpg \
	--rabbitmq-url amqp://guest:guest@LXP-J-ROGERS2:5672/%2F \
	--queue qwen-image \
	--prompt "photorealistic street scene with sharp natural details" \
	--output qwenimage_test_output.png \
	--display
```

The test sends the image and generation settings as JSON, waits for the RPC
response, and optionally writes the returned PNG to `--output`. Use
`--display` to open the result in the system image viewer. Both options are
optional; omit both when only checking that the RabbitMQ request succeeds.

## Start the SAM3 worker

Install Ultralytics in the same virtual environment. Use a CUDA-enabled
PyTorch build on the GPU server when appropriate:

```bash
source .venv/bin/activate
python3 -m pip install ultralytics
```

Start the worker with a local SAM3 checkpoint. The checkpoint can be named
`sam3.pt`, or you can provide its full path:

```bash
python3 sam3_worker.py \
	--model /opt/models/sam3.pt \
	--rabbitmq-url amqp://guest:guest@LXP-J-ROGERS2:5672/%2F \
	--queue sam3
```

The worker prints:

```text
Waiting for SAM3 requests on queue sam3
```

Do not run both workers on the same queue. Stable Diffusion uses
`stable-diffusion`; SAM3 uses `sam3`; SAM2 uses `sam2`.

## Start the SAM2 worker

Install Ultralytics in the worker virtual environment:

```bash
source .venv/bin/activate
python3 -m pip install ultralytics
```

Start the worker with a SAM2 checkpoint. Ultralytics can download
`sam2_b.pt` on first use, or you can provide a local checkpoint path:

```bash
python3 sam2_worker.py \
	--model /opt/models/sam2_b.pt \
	--rabbitmq-url amqp://guest:guest@LXP-J-ROGERS2:5672/%2F \
	--queue sam2
```

The worker prints:

```text
Waiting for SAM2 requests on queue sam2
```

Use the `sam2` queue for SAM2 clients and the `sam3` queue for SAM3 clients.

## Test SAM2 through RabbitMQ

Keep `sam2_worker.py` running, then run the RPC test from another terminal in
this directory. It sends a center-point prompt using the input image:

```bash
source .venv/bin/activate
python3 sam2_test.py /path/to/test-image.jpg \
	--rabbitmq-url amqp://guest:guest@LXP-J-ROGERS2:5672/%2F \
	--queue sam2
```

The test writes `sam2_overlay.png` and one `sam2_mask_*.png` per returned mask
under `sam2_test_output/`. Use `--output-dir` to change that location.
This is an end-to-end test of the RabbitMQ connection, queue, RPC properties,
SAM2 inference, and response decoding.

## 5. Connect the viewer

On the viewer machine, install `pika` in its Python environment:

```bash
python3 -m pip install pika
```

Start `view2.py` with the same RabbitMQ URL and queue. Replace
`gpu-machine` with the model server hostname or IP address:

```bash
python3 view2.py merged.gltf \
	--rabbitmq-url amqp://guest:guest@gpu-machine:5672/%2F \
	--diffusion-queue stable-diffusion
```

Press `A` in the viewer to capture the current view and submit it. The input
and generated images are saved in the viewer's `captures/` directory.

## Troubleshooting

- `Connection refused`: RabbitMQ is not running, or port `5672` is blocked by
	the server firewall.
- `Model ... does not appear to have a file named config.json`: pass the
	directory inside `snapshots/`, not the cache root.
- `CUDA out of memory`: use a GPU with more VRAM, reduce the image resolution,
	or use a smaller diffusion model.
- The viewer times out: check that the worker reached the waiting message and
	that both sides use the same queue and RabbitMQ URL.
add