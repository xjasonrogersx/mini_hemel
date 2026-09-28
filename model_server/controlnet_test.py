#!/usr/bin/env python3
"""Send a depth ControlNet text-to-image or img2img request through RabbitMQ."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import time
import uuid

import pika
from PIL import Image


DEFAULT_RABBITMQ_URL = "amqp://guest:guest@192.168.1.252:5672/%2F"
DEFAULT_QUEUE = "stable-diffusion-controlnet"
DEFAULT_PROMPT = (
    "a highly detailed photorealistic stormtrooper, realistic white armor, "
    "studio lighting, sharp details, preserve the exact pose and composition"
)
DEFAULT_NEGATIVE_PROMPT = (
    "changed pose, changed camera angle, changed composition, warped geometry, "
    "deformed armor, extra limbs, extra objects, text, watermark, blur"
)


def encode_png(path: Path, kind: str) -> str:
    """Encode color as RGB while preserving the supplied depth PNG."""
    with Image.open(path) as image:
        print(f"Input {path}: mode={image.mode} size={image.size}")
        if kind == "color":
            encoded = io.BytesIO()
            image.convert("RGB").save(encoded, format="PNG")
            return base64.b64encode(encoded.getvalue()).decode("ascii")
        if image.mode not in {"L", "RGB", "I;16", "I;16B", "I"}:
            raise ValueError(f"unsupported depth image mode {image.mode} for {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def request_controlnet(
    image_path: Path,
    depth_path: Path,
    output_path: Path,
    rabbitmq_url: str,
    queue: str,
    timeout: float,
    prompt: str,
    negative_prompt: str,
    steps: int,
    strength: float,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    seed: int,
    mode: str,
) -> None:
    request = {
        "control_image_base64": encode_png(depth_path, "depth"),
        "mode": mode,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "seed": seed,
    }
    if mode == "img2img":
        request["image_base64"] = encode_png(image_path, "color")
    print(f"Positive prompt: {prompt}")
    print(f"Negative prompt: {negative_prompt}")
    correlation_id = str(uuid.uuid4())
    parameters = pika.URLParameters(rabbitmq_url)
    parameters.heartbeat = 0
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue=queue, durable=True)
    reply_queue = channel.queue_declare(queue="", exclusive=True).method.queue
    response: bytes | None = None

    def on_response(_channel, _method, properties, body: bytes) -> None:
        nonlocal response
        if properties.correlation_id == correlation_id:
            response = body

    consumer_tag = channel.basic_consume(
        queue=reply_queue, on_message_callback=on_response, auto_ack=True
    )
    channel.basic_publish(
        exchange="",
        routing_key=queue,
        body=json.dumps(request).encode("utf-8"),
        properties=pika.BasicProperties(
            content_type="application/json",
            correlation_id=correlation_id,
            reply_to=reply_queue,
            delivery_mode=2,
        ),
    )
    print(
        f"Sent ControlNet request to {queue}: steps={steps}, strength={strength}, "
        f"guidance={guidance_scale}, control_scale={controlnet_conditioning_scale}"
    )

    deadline = time.monotonic() + timeout
    try:
        while response is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"No response received within {timeout:g} seconds")
            connection.process_data_events(time_limit=min(1.0, remaining))
    finally:
        channel.basic_cancel(consumer_tag)
        connection.close()

    result = json.loads(response.decode("utf-8"))
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "ControlNet worker returned an error"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(result["image_base64"]))
    print(f"Saved generated image: {output_path}")
    print(
        "Worker result: iterations=%s/%s duration=%ss"
        % (
            result.get("iterations", "?"),
            result.get("requested_iterations", steps),
            result.get("duration_seconds", "?"),
        )
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, default=script_dir / "test/stormtrooper.png"
    )
    parser.add_argument(
        "--depth", type=Path, default=script_dir / "test/stormtrooper_depth.png"
    )
    parser.add_argument(
        "--output", type=Path, default=script_dir / "test/stormtrooper_controlnet.png"
    )
    parser.add_argument(
        "--rabbitmq-url",
        default=os.getenv("RABBITMQ_URL", DEFAULT_RABBITMQ_URL),
    )
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.20)
    parser.add_argument("--guidance-scale", type=float, default=5.5)
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("text2img", "img2img"),
        default="text2img",
        help="Use depth-only text-to-image like the Hugging Face example, or RGB img2img",
    )
    args = parser.parse_args()

    for path in (args.image, args.depth):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    request_controlnet(
        args.image,
        args.depth,
        args.output,
        args.rabbitmq_url,
        args.queue,
        args.timeout,
        args.prompt,
        args.negative_prompt,
        args.steps,
        args.strength,
        args.guidance_scale,
        args.controlnet_conditioning_scale,
        args.seed,
        args.mode,
    )


if __name__ == "__main__":
    main()
