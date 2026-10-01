#!/usr/bin/env python3
"""Send one Depth Anything V2 request through RabbitMQ and save the depth map."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import pika
from PIL import Image


DEFAULT_RABBITMQ_URL = "amqp://guest:guest@192.168.1.252:5672/%2F"
DEFAULT_QUEUE = "depth-anything"


def encode_image(path: Path) -> str:
    with Image.open(path) as image:
        encoded = io.BytesIO()
        image.convert("RGB").save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode("ascii")


def request_depth(
    image_path: Path,
    output_path: Path,
    rabbitmq_url: str,
    queue: str,
    timeout: float,
) -> None:
    request = {"image_base64": encode_image(image_path)}
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
    print(f"Sent Depth Anything request to {queue}; waiting for response...")
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
        raise RuntimeError(result.get("error", "Depth Anything worker returned an error"))
    if "depth_image_base64" not in result:
        raise RuntimeError("Depth Anything response did not contain depth_image_base64")

    depth_bytes = base64.b64decode(result["depth_image_base64"])
    with Image.open(io.BytesIO(depth_bytes)) as depth_image:
        depth = np.asarray(depth_image)
        if depth.ndim != 2:
            raise RuntimeError(f"Expected a single-channel depth image, got shape {depth.shape}")
        if depth.size == 0:
            raise RuntimeError("Depth Anything returned an empty depth image")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        depth_image.save(output_path)
        print(
            f"Saved depth map: {output_path} mode={depth_image.mode} "
            f"size={depth_image.size} range={int(depth.min())}..{int(depth.max())}"
        )
    print(
        "Worker metadata: size=%sx%s range=%s..%s"
        % (
            result.get("width", "?"),
            result.get("height", "?"),
            result.get("depth_min", "?"),
            result.get("depth_max", "?"),
        )
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_image = script_dir / "test/view_20260929_204726_630090.png"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=default_image)
    parser.add_argument(
        "--output", type=Path, default=script_dir / "test/depth_anything.png"
    )
    parser.add_argument(
        "--rabbitmq-url",
        default=os.getenv("RABBITMQ_URL", DEFAULT_RABBITMQ_URL),
    )
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    request_depth(args.image, args.output, args.rabbitmq_url, args.queue, args.timeout)


if __name__ == "__main__":
    main()
