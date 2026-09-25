#!/usr/bin/env python3
"""Run the Stable Diffusion image-to-image worker behind RabbitMQ."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from typing import Any

import pika
import torch
from diffusers import StableDiffusion3Img2ImgPipeline
from PIL import Image


LOGGER = logging.getLogger(__name__)


class DiffusionWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        LOGGER.info("Loading Stable Diffusion model: %s", args.model)
        self.pipeline = StableDiffusion3Img2ImgPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to("cpu")
            self.pipeline.enable_attention_slicing()
        LOGGER.info("Stable Diffusion loaded on %s", "cuda" if torch.cuda.is_available() else "cpu")

        self.queue = args.queue
        self.rabbitmq_url = args.rabbitmq_url

    def process(self, request: dict[str, Any]) -> str:
        image = Image.open(
            io.BytesIO(base64.b64decode(request["image_base64"]))
        ).convert("RGB")
        generator = torch.Generator(device="cpu").manual_seed(int(request.get("seed", 0)))
        with torch.inference_mode():
            result = self.pipeline(
                prompt=request["prompt"],
                negative_prompt=request.get("negative_prompt"),
                image=image,
                strength=float(request.get("strength", 0.2)),
                num_inference_steps=int(request.get("steps", 30)),
                guidance_scale=float(request.get("guidance_scale", 4.5)),
                generator=generator,
            ).images[0]
        output = io.BytesIO()
        result.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

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
                response = {"ok": True, "image_base64": self.process(request)}
            except Exception as exc:
                LOGGER.exception("Diffusion request failed")
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
        LOGGER.info("Waiting for diffusion requests on queue %s", self.queue)
        channel.start_consuming()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Stable Diffusion model checkpoint",
    )
    parser.add_argument(
        "--queue",
        default="stable-diffusion",
        help="RabbitMQ request queue (default: stable-diffusion)",
    )
    parser.add_argument(
        "--rabbitmq-url",
        default="amqp://guest:guest@localhost:5672/%2F",
        help="RabbitMQ connection URL",
    )
    args = parser.parse_args()
    DiffusionWorker(args).run()


if __name__ == "__main__":
    main()