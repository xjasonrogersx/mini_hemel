#!/usr/bin/env python3
"""Run Mask2Former ADE20K semantic segmentation behind RabbitMQ."""

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
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


LOGGER = logging.getLogger(__name__)
DEFAULT_CLASSES = ("building", "wall", "house", "roof", "window", "door", "tree", "street", "sidewalk")
CLASS_ALIASES = {"street": "road", "window": "windowpane", "windows": "windowpane"}


class Mask2FormerWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(args.model)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model).to(self.device)
        self.model.eval()
        self.queue = args.queue
        self.rabbitmq_url = args.rabbitmq_url
        self.id_to_label = {
            int(class_id): str(label).lower()
            for class_id, label in self.model.config.id2label.items()
        }
        LOGGER.info(
            "Mask2Former loaded: model=%s device=%s labels=%d queue=%s broker=%s",
            args.model, self.device, len(self.id_to_label), self.queue,
            args.rabbitmq_url.split("@")[-1],
        )

    @staticmethod
    def _decode_image(value: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")

    @staticmethod
    def _encode_png(image: Image.Image) -> str:
        output = io.BytesIO()
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    def _color_map(self, labels: np.ndarray) -> str:
        colors = np.zeros((len(self.id_to_label), 3), dtype=np.uint8)
        for class_id in self.id_to_label:
            colors[class_id] = (
                (class_id * 67 + 41) % 256,
                (class_id * 131 + 83) % 256,
                (class_id * 197 + 149) % 256,
            )
        return self._encode_png(Image.fromarray(colors[labels], mode="RGB"))

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        image = self._decode_image(request["image_base64"])
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = self.model(**inputs)
        segmentation = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]
        labels = segmentation.detach().cpu().numpy().astype(np.uint8)
        requested = request.get("classes", list(DEFAULT_CLASSES))
        if isinstance(requested, str):
            requested = [requested]
        masks = []
        for requested_label in requested:
            normalized = CLASS_ALIASES.get(str(requested_label).lower(), str(requested_label).lower())
            for class_id, label in self.id_to_label.items():
                if label != normalized:
                    continue
                mask = labels == class_id
                if not mask.any():
                    continue
                rows, columns = np.where(mask)
                masks.append({
                    "label": normalized,
                    "requested_label": str(requested_label),
                    "class_id": class_id,
                    "pixel_count": int(mask.sum()),
                    "bbox_xyxy": [int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1],
                    "mask_base64": self._encode_png(Image.fromarray((mask * 255).astype(np.uint8), mode="L")),
                })
        result = {
            "width": image.width,
            "height": image.height,
            "class_map": {str(class_id): label for class_id, label in self.id_to_label.items()},
            "label_map_base64": self._encode_png(Image.fromarray(labels, mode="L")),
            "color_map_base64": self._color_map(labels),
            "masks": masks,
            "present_classes": [
                {"class_id": int(class_id), "label": self.id_to_label.get(int(class_id), "unknown")}
                for class_id in np.unique(labels)
            ],
        }
        LOGGER.info(
            "Mask2Former completed: image=%sx%s present_classes=%d masks=%d duration=%.2fs",
            image.width, image.height, len(result["present_classes"]), len(masks), time.monotonic() - started,
        )
        return result

    def run(self) -> None:
        connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue, durable=True)
        channel.basic_qos(prefetch_count=1)

        def on_request(channel: Any, method: Any, properties: Any, body: bytes) -> None:
            try:
                response = {"ok": True, **self.process(json.loads(body.decode("utf-8")))}
            except Exception as exc:
                LOGGER.exception("Mask2Former request failed")
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
        LOGGER.info("Waiting for Mask2Former requests on queue %s", self.queue)
        channel.start_consuming()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/mask2former-swin-small-ade-semantic")
    parser.add_argument("--queue", default="mask2former")
    parser.add_argument("--rabbitmq-url", default="amqp://guest:guest@localhost:5672/%2F")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    Mask2FormerWorker(args).run()


if __name__ == "__main__":
    main()