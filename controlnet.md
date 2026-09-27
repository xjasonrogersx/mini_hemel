# Stable Diffusion ControlNet

The viewer uses Stable Diffusion with a depth ControlNet to turn the current RGB render into a realistic textured image.

## How it works

When `Q` is pressed in `viewer.py`:

1. The current RGB render is saved in `captures/`.
2. A 16-bit depth render is saved beside it.
3. Both images are sent to RabbitMQ.
4. `controlnet_worker.py` generates a textured image using the RGB image and depth map.
5. The result is saved as `captures/view_controlnet_<timestamp>.png`.

## Requirements

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The ControlNet worker also needs PyTorch and Diffusers:

```bash
pip install torch diffusers transformers accelerate
```

For GPU inference, install the PyTorch build matching the installed CUDA version from the official PyTorch installation instructions.

RabbitMQ must be running and accessible at the default URL:

```text
amqp://guest:guest@localhost:5672/%2F
```

Override it with `--rabbitmq-url` if RabbitMQ uses another host, port, or account.

## Start the worker

From the project directory:

```bash
python3 model_server/controlnet_worker.py \
  --model runwayml/stable-diffusion-v1-5 \
  --controlnet lllyasviel/sd-controlnet-depth
```

The first run downloads the Stable Diffusion and ControlNet model files from Hugging Face. Use `--cpu` to force CPU inference:

```bash
python3 model_server/controlnet_worker.py --cpu
```

Available worker options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--model` | `runwayml/stable-diffusion-v1-5` | Stable Diffusion model ID or local path |
| `--controlnet` | `lllyasviel/sd-controlnet-depth` | Depth ControlNet model ID or local path |
| `--queue` | `stable-diffusion-controlnet` | RabbitMQ request queue |
| `--rabbitmq-url` | `amqp://guest:guest@localhost:5672/%2F` | RabbitMQ connection URL |
| `--cpu` | disabled | Force CPU inference |

## Start the viewer

In another terminal, start the viewer:

```bash
python3 viewer.py
```

Double-click the scene to enter walk mode, position the camera, and press `Q`. The viewer displays progress in the window title. Generated images appear in `captures/` when processing completes.

## Tuning

The current request uses:

- 30 inference steps
- Img2img strength `0.35`
- Guidance scale `7.5`
- ControlNet conditioning scale `1.0`
- Seed `0`

The RGB image preserves the camera composition while the depth image provides structural guidance. If the generated result changes the geometry too much, lower the img2img strength or increase the ControlNet conditioning scale in `viewer.py`.

## Troubleshooting

- `Connection refused`: start RabbitMQ and confirm the URL.
- `No module named pika`: install the project requirements.
- `No module named diffusers` or `torch`: install the worker dependencies.
- CUDA out of memory: use a smaller model, enable CPU mode, or reduce the render resolution.
- No generated image: check the worker terminal for model download or inference errors.
