#!/usr/bin/env python3
"""Send one SegFormer semantic-segmentation request through RabbitMQ."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
import time
import uuid

import pika
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_IMAGE = SCRIPT_DIR / "test/view_20260929_204726_630090.png"
DEFAULT_RABBITMQ_URL = "amqp://guest:guest@localhost:5672/%2F"
DEFAULT_QUEUE = "segformer"
DEFAULT_CLASSES = ["tree", "window", "door", "street", "sidewalk"]


def config_defaults() -> tuple[str, str, list[str]]:
    config_path = ROOT_DIR / "config.json"
    if not config_path.is_file():
        return DEFAULT_RABBITMQ_URL, DEFAULT_QUEUE, DEFAULT_CLASSES
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    settings = config.get("segmentation_generator", {})
    return (
        settings.get("rabbitmq_url", DEFAULT_RABBITMQ_URL),
        settings.get("queue", DEFAULT_QUEUE),
        settings.get("classes", DEFAULT_CLASSES),
    )


def encode_image(path: Path) -> str:
    with Image.open(path) as image:
        encoded = io.BytesIO()
        image.convert("RGB").save(encoded, format="PNG")
    return base64.b64encode(encoded.getvalue()).decode("ascii")


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_")


def request_segmentation(
    image_path: Path,
    output_dir: Path,
    rabbitmq_url: str,
    queue: str,
    classes: list[str],
    timeout: float,
) -> None:
    request = {"image_base64": encode_image(image_path), "classes": classes}
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
    print(f"Sent SegFormer request to {queue}; classes={', '.join(classes)}")
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
        raise RuntimeError(result.get("error", "SegFormer worker returned an error"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for field, filename in (
        ("color_map_base64", "segformer_color_map.png"),
        ("label_map_base64", "segformer_label_map.png"),
    ):
        encoded = result.get(field)
        if not encoded:
            raise RuntimeError(f"SegFormer response did not contain {field}")
        (output_dir / filename).write_bytes(base64.b64decode(encoded))

    masks = result.get("masks", [])
    for mask in masks:
        filename = f"segformer_{safe_name(mask['label'])}_mask.png"
        (output_dir / filename).write_bytes(base64.b64decode(mask["mask_base64"]))
        print(
            f"Saved {mask['label']} mask: {filename} "
            f"pixels={mask['pixel_count']} bbox={mask['bbox_xyxy']}"
        )

    present = ", ".join(item["label"] for item in result.get("present_classes", []))
    print(f"Saved maps to {output_dir}")
    print(f"Image size: {result.get('width', '?')}x{result.get('height', '?')}")
    print(f"Present classes: {present}")


def main() -> None:
    config_url, config_queue, config_classes = config_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "test/segformer_output")
    parser.add_argument(
        "--rabbitmq-url",
        default=os.getenv("RABBITMQ_URL", config_url),
    )
    parser.add_argument("--queue", default=config_queue)
    parser.add_argument("--classes", nargs="+", default=config_classes)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    request_segmentation(
        args.image,
        args.output_dir,
        args.rabbitmq_url,
        args.queue,
        args.classes,
        args.timeout,
    )


if __name__ == "__main__":
    main()
