#!/usr/bin/env python3
"""Run the upstream PyTorch M-LSD line detector behind RabbitMQ."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pika
from PIL import Image
import cv2
import torch

LOGGER = logging.getLogger(__name__)


class MLSDWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.source_dir:
            sys.path.insert(0, str(args.source_dir))
        try:
            from models.mbv2_mlsd_large import MobileV2_MLSD_Large
        except ImportError as exc:
            raise RuntimeError(
                "M-LSD source checkout is required; pass --source-dir pointing to lhwcv/mlsd_pytorch"
            ) from exc
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        self.net = MobileV2_MLSD_Large().to(self.device).eval()
        checkpoint = torch.load(args.model, map_location=self.device, weights_only=True)
        self.net.load_state_dict(checkpoint, strict=True)
        self.input_size = int(args.input_size)
        self.queue = args.queue
        self.rabbitmq_url = args.rabbitmq_url
        LOGGER.info(
            "M-LSD loaded: model=%s input=%d device=%s queue=%s broker=%s",
            args.model, self.input_size, self.device, self.queue,
            args.rabbitmq_url.split("@")[-1],
        )

    @staticmethod
    def _decode_image(value: str) -> np.ndarray:
        image = Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _encode_png(image: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("M-LSD could not encode line map")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        image = self._decode_image(request["image_base64"])
        height, width = image.shape[:2]
        threshold = float(request.get("score_threshold", 0.20))
        resized = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        rgba = np.concatenate([cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), np.ones((*resized.shape[:2], 1), dtype=np.uint8)], axis=2)
        tensor = torch.from_numpy(rgba.transpose(2, 0, 1)).float().div(127.5).sub(1.0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.net(tensor)
            center = torch.sigmoid(output[:, 0:1])
            map_height, map_width = output.shape[-2:]
            pooled = torch.nn.functional.max_pool2d(center, kernel_size=3, stride=1, padding=1)
            keep = (pooled == center).float()
            scores, indices = torch.topk((center * keep).reshape(-1), 200)
            y = torch.div(indices, map_width, rounding_mode="floor")
            x = torch.remainder(indices, map_width)
            displacement = output[0, 1:5].permute(1, 2, 0)
        lines = []
        for xx, yy, score in zip(x.cpu().numpy(), y.cpu().numpy(), scores.cpu().numpy()):
            if float(score) < threshold:
                continue
            dx1, dy1, dx2, dy2 = displacement[int(yy), int(xx)].cpu().numpy()
            x1, y1, x2, y2 = (self.input_size / map_width) * np.array(
                [xx + dx1, yy + dy1, xx + dx2, yy + dy2], dtype=float
            )
            if np.hypot(x2 - x1, y2 - y1) < 8.0:
                continue
            lines.append((x1 * width / self.input_size, y1 * height / self.input_size,
                          x2 * width / self.input_size, y2 * height / self.input_size, float(score)))
        line_map = np.zeros((height, width), dtype=np.uint8)
        for x1, y1, x2, y2, _score in lines:
            cv2.line(line_map, (round(x1), round(y1)), (round(x2), round(y2)), 255, 2, cv2.LINE_AA)
        metadata = {
            "width": width,
            "height": height,
            "line_count": len(lines),
            "line_pixels": int(np.count_nonzero(line_map)),
            "score_threshold": threshold,
            "model": "M-LSD",
            "duration_seconds": time.monotonic() - started,
        }
        LOGGER.info(
            "M-LSD completed: image=%sx%s lines=%d pixels=%d duration=%.2fs",
            width, height, len(lines), metadata["line_pixels"], metadata["duration_seconds"],
        )
        return {"width": width, "height": height, "line_map_base64": self._encode_png(line_map), "metadata": metadata}

    def run(self) -> None:
        connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
        channel = connection.channel()
        channel.queue_declare(queue=self.queue, durable=True)
        channel.basic_qos(prefetch_count=1)

        def on_request(channel: Any, method: Any, properties: Any, body: bytes) -> None:
            try:
                response = {"ok": True, **self.process(json.loads(body.decode("utf-8")))}
            except Exception as exc:
                LOGGER.exception("M-LSD request failed")
                response = {"ok": False, "error": str(exc)}
            channel.basic_publish(
                exchange="", routing_key=properties.reply_to, body=json.dumps(response).encode("utf-8"),
                properties=pika.BasicProperties(content_type="application/json", correlation_id=properties.correlation_id),
            )
            channel.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_consume(queue=self.queue, on_message_callback=on_request)
        LOGGER.info("Waiting for M-LSD requests on queue %s", self.queue)
        channel.start_consuming()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="M-LSD PyTorch checkpoint (.pth)")
    parser.add_argument("--source-dir", type=Path, required=True, help="Root of the lhwcv/mlsd_pytorch checkout")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--queue", default="mlsd")
    parser.add_argument("--rabbitmq-url", default="amqp://guest:guest@localhost:5672/%2F")
    parser.add_argument("--cuda", action="store_true")
    MLSDWorker(parser.parse_args()).run()


if __name__ == "__main__":
    main()
