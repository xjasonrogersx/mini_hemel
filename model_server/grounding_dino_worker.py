#!/usr/bin/env python3
"""Run Grounding DINO text-prompted object detection behind RabbitMQ."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import time
from typing import Any

import numpy as np
import pika
import torch
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection


LOGGER = logging.getLogger(__name__)


class GroundingDinoWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        LOGGER.info("Loading Grounding DINO model: %s", args.model)
        self.processor = AutoProcessor.from_pretrained(args.model)
        self.model = GroundingDinoForObjectDetection.from_pretrained(args.model).to(self.device)
        self.model.eval()
        self.queue = args.queue
        self.rabbitmq_url = args.rabbitmq_url
        self.default_threshold = args.threshold
        self.default_text_threshold = args.text_threshold
        LOGGER.info(
            "Grounding DINO loaded: device=%s queue=%s broker=%s",
            self.device, self.queue, args.rabbitmq_url.split("@")[-1],
        )

    @staticmethod
    def _decode_image(value: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        image = self._decode_image(request["image_base64"])
        prompt = str(request.get("text_prompt", "door"))
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("text_prompt must not be empty")
        if not prompt.endswith("."):
            prompt += "."
        threshold = float(request.get("threshold", self.default_threshold))
        text_threshold = float(request.get("text_threshold", self.default_text_threshold))
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        boxes = results.get("boxes", []).detach().cpu().tolist()
        scores = results.get("scores", []).detach().cpu().tolist()
        labels = results.get("text_labels", results.get("labels", []))
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            detections.append({
                "prompt": prompt,
                "label": str(label),
                "score": float(score),
                "box_xyxy": [float(value) for value in box],
            })
        LOGGER.info(
            "Grounding DINO completed: prompt=%r image=%sx%s detections=%d threshold=%.2f duration=%.2fs",
            prompt, image.width, image.height, len(detections), threshold, time.monotonic() - started,
        )
        return {
            "width": image.width,
            "height": image.height,
            "prompt": prompt,
            "threshold": threshold,
            "text_threshold": text_threshold,
            "detections": detections,
        }

    def run(self) -> None:
        connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue, durable=True)
        channel.basic_qos(prefetch_count=1)

        def on_request(channel: Any, method: Any, properties: Any, body: bytes) -> None:
            try:
                request = json.loads(body.decode("utf-8"))
                LOGGER.info(
                    "Grounding DINO work received: correlation_id=%s prompt=%r",
                    properties.correlation_id,
                    request.get("text_prompt", "door"),
                )
                response = {"ok": True, **self.process(request)}
            except Exception as exc:
                LOGGER.exception("Grounding DINO request failed")
                response = {"ok": False, "error": str(exc)}
            channel.basic_publish(
                exchange="", routing_key=properties.reply_to,
                body=json.dumps(response).encode("utf-8"),
                properties=pika.BasicProperties(
                    content_type="application/json", correlation_id=properties.correlation_id,
                ),
            )
            channel.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_consume(queue=self.queue, on_message_callback=on_request)
        LOGGER.info("Waiting for Grounding DINO requests on queue %s", self.queue)
        channel.start_consuming()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--queue", default="grounding-dino")
    parser.add_argument("--rabbitmq-url", default="amqp://guest:guest@localhost:5672/%2F")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    GroundingDinoWorker(args).run()


if __name__ == "__main__":
    main()
