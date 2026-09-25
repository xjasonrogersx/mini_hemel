#!/usr/bin/env python3
"""Send one Qwen Image request through RabbitMQ and save the generated PNG."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import time
import uuid

import pika
from PIL import Image


def encode_image(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode("ascii")


def request_qwen(
    image_path: Path,
    output_path: Path | None,
    display: bool,
    prompt: str,
    rabbitmq_url: str,
    queue: str,
    timeout: float,
    steps: int,
    strength: float,
    guidance_scale: float,
    seed: int,
) -> None:
    request = {
        "image_base64": encode_image(image_path),
        "prompt": prompt,
        "steps": steps,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }
    correlation_id = str(uuid.uuid4())
    connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
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
    print(f"Sent Qwen Image request to queue {queue}; waiting for response...")
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
        raise RuntimeError(result.get("error", "Qwen Image worker returned an error"))

    output_bytes = base64.b64decode(result["image_base64"])
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(output_bytes)
        print(f"Saved generated image: {output_path}")
    if display:
        Image.open(io.BytesIO(output_bytes)).show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image to send to Qwen Image")
    parser.add_argument(
        "--prompt",
        default=(
            "high quality photorealistic street-level 3D reconstruction, "
            "sharp details, natural lighting, preserve the exact geometry "
            "and composition of the input image"
        ),
        help="Image generation prompt",
    )
    parser.add_argument(
        "--rabbitmq-url",
        default="amqp://guest:guest@localhost:5672/%2F",
        help="RabbitMQ connection URL",
    )
    parser.add_argument("--queue", default="qwen-image", help="Qwen request queue")
    parser.add_argument(
        "--timeout", type=float, default=900.0, help="Response timeout in seconds"
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.2)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional generated PNG output path",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Open the generated image using the system image viewer",
    )
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    request_qwen(
        args.image,
        args.output,
        args.display,
        args.prompt,
        args.rabbitmq_url,
        args.queue,
        args.timeout,
        args.steps,
        args.strength,
        args.guidance_scale,
        args.seed,
    )


if __name__ == "__main__":
    main()