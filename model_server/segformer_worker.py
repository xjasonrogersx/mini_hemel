#!/usr/bin/env python3
"""Run SegFormer ADE20K semantic segmentation behind RabbitMQ."""

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
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


LOGGER = logging.getLogger(__name__)
DEFAULT_CLASSES = ("building", "wall", "house", "roof", "window", "door", "tree", "street", "sidewalk")
CLASS_ALIASES = {"street": "road", "windows": "windowpane", "window": "windowpane"}


class SegFormerWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        )
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        LOGGER.info("Loading SegFormer model: %s", args.model)
        self.processor = AutoImageProcessor.from_pretrained(args.model)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.queue = args.queue
        self.rabbitmq_url = args.rabbitmq_url
        self.id_to_label = {
            int(class_id): str(label).lower()
            for class_id, label in self.model.config.id2label.items()
        }
        LOGGER.info(
            "SegFormer loaded on %s using %s with %d labels",
            self.device,
            self.dtype,
            len(self.id_to_label),
        )
        LOGGER.info(
            "SegFormer configuration: queue=%s rabbitmq_host=%s model=%s",
            self.queue,
            args.rabbitmq_url.split("@")[-1],
            args.model,
        )

    @staticmethod
    def _decode_image(image_base64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")

    @staticmethod
    def _encode_png(image: Image.Image) -> str:
        output = io.BytesIO()
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    def _encode_label_map(self, labels: np.ndarray) -> str:
        label_image = Image.fromarray(labels.astype(np.uint8), mode="L")
        return self._encode_png(label_image)

    def _encode_color_map(self, labels: np.ndarray) -> str:
        colors = np.zeros((len(self.id_to_label), 3), dtype=np.uint8)
        for class_id in self.id_to_label:
            # Stable, distinct colors derived from the ADE20K class ID.
            colors[class_id] = (
                (class_id * 67 + 41) % 256,
                (class_id * 131 + 83) % 256,
                (class_id * 197 + 149) % 256,
            )
        return self._encode_png(Image.fromarray(colors[labels], mode="RGB"))

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        started_at = time.monotonic()
        LOGGER.info(
            "SegFormer request started: keys=%s requested_classes=%s",
            sorted(request),
            request.get("classes", list(DEFAULT_CLASSES)),
        )
        image = self._decode_image(request["image_base64"])
        LOGGER.info("SegFormer image decoded: size=%sx%s mode=%s", image.width, image.height, image.mode)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(device=self.device, dtype=self.dtype)
            if value.is_floating_point()
            else value.to(self.device)
            for key, value in inputs.items()
        }
        LOGGER.info(
            "SegFormer inputs prepared: tensors=%s device=%s",
            {key: tuple(value.shape) for key, value in inputs.items()},
            self.device,
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)
        LOGGER.info("SegFormer model inference completed")
        segmentation = self.processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = segmentation.detach().cpu().numpy().astype(np.uint8)
        LOGGER.info(
            "SegFormer map postprocessed: shape=%s unique_labels=%d",
            labels.shape,
            len(np.unique(labels)),
        )

        requested = request.get("classes", list(DEFAULT_CLASSES))
        if isinstance(requested, str):
            requested = [requested]
        masks = []
        for requested_label in requested:
            normalized_label = CLASS_ALIASES.get(str(requested_label).lower(), str(requested_label).lower())
            matching_ids = [
                class_id for class_id, label in self.id_to_label.items()
                if label == normalized_label
            ]
            for class_id in matching_ids:
                mask = labels == class_id
                if not mask.any():
                    continue
                rows, columns = np.where(mask)
                masks.append(
                    {
                        "label": normalized_label,
                        "requested_label": str(requested_label),
                        "class_id": class_id,
                        "pixel_count": int(mask.sum()),
                        "bbox_xyxy": [
                            int(columns.min()), int(rows.min()),
                            int(columns.max()) + 1, int(rows.max()) + 1,
                        ],
                        "mask_base64": self._encode_png(
                            Image.fromarray((mask * 255).astype(np.uint8), mode="L")
                        ),
                    }
                )

        present_ids = np.unique(labels)
        result = {
            "width": image.width,
            "height": image.height,
            "class_map": {str(class_id): label for class_id, label in self.id_to_label.items()},
            "label_map_base64": self._encode_label_map(labels),
            "color_map_base64": self._encode_color_map(labels),
            "masks": masks,
            "present_classes": [
                {"class_id": int(class_id), "label": self.id_to_label.get(int(class_id), "unknown")}
                for class_id in present_ids
            ],
        }
        LOGGER.info(
            "SegFormer request completed: present_classes=%s masks=%d duration=%.2fs",
            [item["label"] for item in result["present_classes"]],
            len(result["masks"]),
            time.monotonic() - started_at,
        )
        return result

    def run(self) -> None:
        LOGGER.info(
            "Connecting to RabbitMQ: host=%s queue=%s",
            self.rabbitmq_url.split("@")[-1],
            self.queue,
        )
        connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue, durable=True)
        channel.basic_qos(prefetch_count=1)
        LOGGER.info("RabbitMQ connection ready; prefetch_count=1")

        def on_request(
            channel: Any, method: Any, properties: Any, body: bytes
        ) -> None:
            request_started = time.monotonic()
            LOGGER.info(
                "RabbitMQ request received: delivery_tag=%s correlation_id=%s reply_to=%s bytes=%d",
                method.delivery_tag,
                properties.correlation_id,
                properties.reply_to,
                len(body),
            )
            try:
                request = json.loads(body.decode("utf-8"))
                response = {"ok": True, **self.process(request)}
            except Exception as exc:
                LOGGER.exception("SegFormer request failed")
                response = {"ok": False, "error": str(exc)}
            LOGGER.info(
                "RabbitMQ response sent: correlation_id=%s ok=%s bytes=%d duration=%.2fs",
                properties.correlation_id,
                response.get("ok"),
                len(json.dumps(response).encode("utf-8")),
                time.monotonic() - request_started,
            )
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
        LOGGER.info("Waiting for SegFormer requests on queue %s", self.queue)
        try:
            channel.start_consuming()
        finally:
            LOGGER.info("Stopping SegFormer worker")
            if not connection.is_closed:
                connection.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="SegFormer semantic-segmentation model ID or local directory",
    )
    parser.add_argument(
        "--queue",
        default="segformer",
        help="RabbitMQ request queue (default: segformer)",
    )
    parser.add_argument(
        "--rabbitmq-url",
        default="amqp://guest:guest@localhost:5672/%2F",
        help="RabbitMQ connection URL",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even when CUDA is available",
    )
    args = parser.parse_args()
    SegFormerWorker(args).run()


if __name__ == "__main__":
    main()
