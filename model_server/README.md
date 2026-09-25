# Model Server

These workers run on the machine with the larger memory/GPU. The viewer sends
jobs through RabbitMQ and receives results through RabbitMQ RPC replies.

The workers use separate queues so they can run independently:

| Worker | Script | Queue | Result |
| --- | --- | --- | --- |
| Stable Diffusion | `diffusion_worker.py` | `stable-diffusion` | Generated PNG |
| Stable Diffusion 1.5 | `d1.5_worker.py` | `stable-diffusion-15` | Generated PNG |
| Qwen Image 2.1 | `qwenimage_worker.py` | `qwen-image` | Generated PNG |
| Ultralytics SAM3 | `sam3_worker.py` | `sam3` | Detection metadata and masks |
| Ultralytics SAM2 | `sam2_worker.py` | `sam2` | Detection metadata and masks |

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