# mini_hemel

Reality itself is not fully defined until it is observed.

The idea that the world is not concrete until observed lies at the heart of the famous thought experiment known as Schrödinger's cat. Until the box is opened, quantum mechanics suggests the system exists in a superposition of possibilities, with the cat both alive and dead in a mathematical sense. The act of observation appears to transform these possibilities into a single reality, raising the profound question of whether the world is fully defined before it is observed. This idea echoes the philosophy of George Berkeley, who argued that "to be is to be perceived," and resonates with John Archibald Wheeler's notion of a "participatory universe," in which observers play an active role in bringing reality into concrete existence. Whether observation creates reality or merely reveals it remains one of the deepest mysteries in both physics and philosophy

```
xhost +local:docker
docker run -it --name god -v /home/jason/work:/workspace  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --device /dev/dri  ubuntu:24.04


apt-get update
apt-get install -y libgl1 libglu1-mesa libx11-6 libxext6 libxrender1 mesa-utils


```

## Remote Stable Diffusion worker

Stable Diffusion runs in a separate process on the machine with the GPU and
memory. The viewer sends the captured PNG and generation settings through
RabbitMQ; the worker returns the generated PNG. RabbitMQ must be reachable from
both machines, and the same queue name must be used on both sides.

On the worker machine, install the dependencies and start RabbitMQ, then run:

```bash
python3 diffusion_worker.py \
  --rabbitmq-url amqp://user:password@rabbitmq-host:5672/%2F \
  --queue stable-diffusion
```

On the viewer machine, install `pika` and start the viewer with the worker's
RabbitMQ URL:

```bash
python3 view2.py merged.gltf \
  --rabbitmq-url amqp://user:password@rabbitmq-host:5672/%2F \
  --diffusion-queue stable-diffusion
```

Press `A` to submit the current view. The viewer keeps the input and output
captures locally. Requests are acknowledged only after inference completes,
and the viewer reports a timeout or worker error in the window caption.

RabbitMQ is a good fit for this asynchronous, potentially slow GPU job. For
large images or many concurrent clients, store images in object storage and
send only an object key through RabbitMQ; base64-encoded PNG messages are kept
here because the current viewer sends one screenshot at a time.

![alt text](image.png)