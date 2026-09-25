#!/usr/bin/env python3
"""Run a Qwen Image 2.1 worker behind RabbitMQ."""

from __future__ import annotations

import argparse
import base64
import io
import inspect
import json
import logging
from typing import Any

import pika
import torch
from diffusers import DiffusionPipeline
from PIL import Image


LOGGER = logging.getLogger(__name__)


class QwenImageWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        LOGGER.info("Loading Qwen Image model: %s", args.model)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            self.pipeline = DiffusionPipeline.from_pretrained(
                args.model,
                torch_dtype=dtype,
            )
        except AttributeError as exc:
            if "QwenImage21Pipeline" in str(exc):
                raise RuntimeError(
                    "This Qwen Image 2.1 checkpoint requires a newer Diffusers "
                    "build. Install it with: python3 -m pip install --upgrade "
                    "git+https://github.com/huggingface/diffusers.git"
                ) from exc
            raise
        if torch.cuda.is_available():
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to("cpu")
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
        self.queue = args.queue
        self.rabbitmq_url = args.rabbitmq_url
        LOGGER.info(
            "Qwen Image model loaded on %s",
            "cuda" if torch.cuda.is_available() else "cpu",
        )

    def process(self, request: dict[str, Any]) -> str:
        image = Image.open(
            io.BytesIO(base64.b64decode(request["image_base64"]))
        ).convert("RGB")
        prompt = request.get(
            "prompt",
            "high quality photorealistic street-level 3D reconstruction, "
            "sharp details, natural lighting, preserve the exact geometry "
            "and composition of the input image",
        )
        generator = torch.Generator(device="cpu").manual_seed(
            int(request.get("seed", 0))
        )
        pipeline_args: dict[str, Any] = {
            "prompt": prompt,
            "image": image,
            "num_inference_steps": int(request.get("steps", 30)),
            "true_cfg_scale": float(request.get("guidance_scale", 4.0)),
            "generator": generator,
        }
        if request.get("negative_prompt") is not None:
            pipeline_args["negative_prompt"] = request["negative_prompt"]
        if request.get("strength") is not None:
            pipeline_args["strength"] = float(request["strength"])

        supported_args = set(inspect.signature(self.pipeline.__call__).parameters)
        unsupported_args = set(pipeline_args) - supported_args
        if unsupported_args:
            LOGGER.warning(
                "Qwen pipeline does not support %s; ignoring these request fields",
                ", ".join(sorted(unsupported_args)),
            )
            for argument in unsupported_args:
                pipeline_args.pop(argument)

        with torch.inference_mode():
            result = self.pipeline(**pipeline_args).images[0]

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
                LOGGER.exception("Qwen Image request failed")
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
        LOGGER.info("Waiting for Qwen Image requests on queue %s", self.queue)
        channel.start_consuming()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-2.1",
        help="Qwen Image model ID or local Diffusers directory",
    )
    parser.add_argument(
        "--queue",
        default="qwen-image",
        help="RabbitMQ request queue (default: qwen-image)",
    )
    parser.add_argument(
        "--rabbitmq-url",
        default="amqp://guest:guest@localhost:5672/%2F",
        help="RabbitMQ connection URL",
    )
    args = parser.parse_args()
    QwenImageWorker(args).run()


if __name__ == "__main__":
    main()