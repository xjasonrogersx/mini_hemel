#!/usr/bin/env python3
"""Run an Ultralytics SAM3 segmentation worker behind RabbitMQ."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from typing import Any

import numpy as np
import pika
from PIL import Image
from ultralytics import SAM


LOGGER = logging.getLogger(__name__)


class SAM3Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        LOGGER.info("Loading Ultralytics SAM3 model: %s", args.model)
        self.model = SAM(args.model)
        self.queue = args.queue
        self.rabbitmq_url = args.rabbitmq_url
        LOGGER.info("SAM3 model loaded")

    @staticmethod
    def _decode_image(image_base64: str) -> np.ndarray:
        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
        return np.asarray(image)

    @staticmethod
    def _encode_mask(mask: np.ndarray) -> str:
        output = io.BytesIO()
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(
            output, format="PNG"
        )
        return base64.b64encode(output.getvalue()).decode("ascii")

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        image = self._decode_image(request["image_base64"])
        prediction_args: dict[str, Any] = {"source": image, "verbose": False}
        for field in ("points", "labels", "bboxes", "text", "conf"):
            if field in request:
                prediction_args[field] = request[field]

        results = self.model.predict(**prediction_args)
        if not results:
            return {
                "width": image.shape[1],
                "height": image.shape[0],
                "detections": [],
            }

        result = results[0]
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        boxes = result.boxes
        detections = []
        for index, mask in enumerate(masks):
            detection: dict[str, Any] = {"mask_base64": self._encode_mask(mask)}
            if boxes is not None and index < len(boxes):
                detection["box_xyxy"] = boxes.xyxy[index].cpu().tolist()
                if boxes.conf is not None:
                    detection["confidence"] = float(boxes.conf[index].cpu())
                if boxes.cls is not None:
                    detection["class_id"] = int(boxes.cls[index].cpu())
            detections.append(detection)

        return {
            "width": image.shape[1],
            "height": image.shape[0],
            "detections": detections,
        }

    def run(self) -> None:
        connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue, durable=True)
        channel.basic_qos(prefetch_count=1)

        def on_request(
            channel: Any, method: Any, properties: Any, body: bytes
        ) -> None:
            try:
                request = json.loads(body.decode("utf-8"))
                response = {"ok": True, **self.process(request)}
            except Exception as exc:
                LOGGER.exception("SAM3 request failed")
                response = {"ok": False, "error": str(exc)}
            channel.basic_publish(
                exchange="",
                routing_key=properties.reply_to,
                body=json.dumps(response).encode("utf-8"),
                properties=pika.BasicProperties(
                    content_type="application/json",
                    correlation_id=properties.correlation_id,
                ),
            )
            channel.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_consume(queue=self.queue, on_message_callback=on_request)
        LOGGER.info("Waiting for SAM3 requests on queue %s", self.queue)
        channel.start_consuming()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="sam3.pt",
        help="Ultralytics SAM3 checkpoint or local model path",
    )
    parser.add_argument(
        "--queue",
        default="sam3",
        help="RabbitMQ request queue (default: sam3)",
    )
    parser.add_argument(
        "--rabbitmq-url",
        default="amqp://guest:guest@localhost:5672/%2F",
        help="RabbitMQ connection URL",
    )
    args = parser.parse_args()
    SAM3Worker(args).run()


if __name__ == "__main__":
    main()