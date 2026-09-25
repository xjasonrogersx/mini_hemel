# Stable Diffusion Model Server

This service runs Stable Diffusion on the machine with the larger memory/GPU.
The viewer sends image-generation jobs through RabbitMQ and receives the
generated PNG in response.

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
python3 -m pip install torch diffusers transformers accelerate pika Pillow
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
