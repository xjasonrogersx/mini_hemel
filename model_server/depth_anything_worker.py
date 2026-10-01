#!/usr/bin/env python3
"""Run Depth Anything V2 monocular depth estimation behind RabbitMQ."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from typing import Any

import numpy as np
import pika
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


LOGGER = logging.getLogger(__name__)


class DepthAnythingWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        )
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        LOGGER.info("Loading Depth Anything V2 model: %s", args.model)
        self.processor = AutoImageProcessor.from_pretrained(args.model)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.queue = args.queue
        self.rabbitmq_url = args.rabbitmq_url
        LOGGER.info("Depth Anything V2 loaded on %s using %s", self.device, self.dtype)

    @staticmethod
    def _decode_image(image_base64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")

    @staticmethod
    def _encode_depth(depth: np.ndarray) -> str:
        output = io.BytesIO()
        Image.fromarray(depth, mode="I;16").save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _normalize_depth(depth: torch.Tensor) -> np.ndarray:
        values = depth.detach().float().cpu().numpy()
        valid = np.isfinite(values)
        if not np.any(valid):
            return np.zeros(values.shape, dtype=np.uint16)
        low, high = np.percentile(values[valid], [2.0, 98.0])
        if high <= low:
            normalized = np.zeros(values.shape, dtype=np.float32)
        else:
            normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
        return np.asarray(normalized * np.iinfo(np.uint16).max, dtype=np.uint16)

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        image = self._decode_image(request["image_base64"])
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(device=self.device, dtype=self.dtype)
            if value.is_floating_point()
            else value.to(self.device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            prediction = self.model(**inputs).predicted_depth
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1).squeeze(0)
        depth = self._normalize_depth(prediction)
        return {
            "width": image.width,
            "height": image.height,
            "depth_image_base64": self._encode_depth(depth),
            "depth_min": int(depth.min()),
            "depth_max": int(depth.max()),
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
                LOGGER.exception("Depth Anything request failed")
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
        LOGGER.info("Waiting for Depth Anything requests on queue %s", self.queue)
        channel.start_consuming()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Depth Anything V2 Transformers model ID or local directory",
    )
    parser.add_argument(
        "--queue",
        default="depth-anything",
        help="RabbitMQ request queue (default: depth-anything)",
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
    DepthAnythingWorker(args).run()


if __name__ == "__main__":
    main()
