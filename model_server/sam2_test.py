#!/usr/bin/env python3
"""Send one SAM2 segmentation request through RabbitMQ and save the result."""

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


def encode_image(path: Path) -> tuple[str, Image.Image]:
    image = Image.open(path).convert("RGB")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode("ascii"), image


def request_sam2(
    image_path: Path,
    rabbitmq_url: str,
    queue: str,
    timeout: float,
    output_dir: Path,
) -> None:
    image_base64, image = encode_image(image_path)
    request = {
        "image_base64": image_base64,
        "points": [[image.width // 2, image.height // 2]],
        "labels": [1],
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
    print(f"Sent SAM2 request to queue {queue}; waiting for response...")
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
        raise RuntimeError(result.get("error", "SAM2 worker returned an error"))

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy().convert("RGBA")
    detections = result.get("detections", [])
    for index, detection in enumerate(detections):
        mask_bytes = base64.b64decode(detection["mask_base64"])
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_path = output_dir / f"sam2_mask_{index:03d}.png"
        mask.save(mask_path)
        color = Image.new("RGBA", image.size, (255, 32, 32, 0))
        color.putalpha(mask.point(lambda value: min(150, value)))
        overlay = Image.alpha_composite(overlay, color)

    overlay_path = output_dir / "sam2_overlay.png"
    overlay.convert("RGB").save(overlay_path)
    print(f"Received {len(detections)} detection(s)")
    print(f"Saved overlay: {overlay_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image to segment")
    parser.add_argument(
        "--rabbitmq-url",
        default="amqp://guest:guest@localhost:5672/%2F",
        help="RabbitMQ connection URL",
    )
    parser.add_argument("--queue", default="sam2", help="SAM2 request queue")
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="Response timeout in seconds"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("sam2_test_output")
    )
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    request_sam2(
        args.image,
        args.rabbitmq_url,
        args.queue,
        args.timeout,
        args.output_dir,
    )


if __name__ == "__main__":
    main()