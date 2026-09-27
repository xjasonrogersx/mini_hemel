#!/usr/bin/env python3
"""Generate a textured render from an RGB image and depth ControlNet input."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from typing import Any

import pika
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from PIL import Image


LOGGER = logging.getLogger(__name__)


class ControlNetWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        dtype = torch.float16 if torch.cuda.is_available() and not args.cpu else torch.float32
        controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
        self.pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            args.model, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
        )
        if dtype == torch.float16:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to("cpu")
            self.pipeline.enable_attention_slicing()
        self.queue = args.queue
        self.rabbitmq_url = args.rabbitmq_url

    def process(self, request: dict[str, Any]) -> str:
        image = Image.open(io.BytesIO(base64.b64decode(request["image_base64"]))).convert("RGB")
        control = Image.open(io.BytesIO(base64.b64decode(request["control_image_base64"]))).convert("L").convert("RGB")
        generator = torch.Generator(device="cpu").manual_seed(int(request.get("seed", 0)))
        with torch.inference_mode():
            result = self.pipeline(
                prompt=request.get("prompt", "photorealistic textured reconstruction, realistic materials and lighting"),
                negative_prompt=request.get("negative_prompt"),
                image=image,
                control_image=control,
                strength=float(request.get("strength", 0.35)),
                num_inference_steps=int(request.get("steps", 30)),
                guidance_scale=float(request.get("guidance_scale", 7.5)),
                controlnet_conditioning_scale=float(request.get("controlnet_conditioning_scale", 1.0)),
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
        def on_request(channel: Any, method: Any, properties: Any, body: bytes) -> None:
            try:
                response = {"ok": True, "image_base64": self.process(json.loads(body.decode("utf-8")))}
            except Exception as exc:
                LOGGER.exception("ControlNet request failed")
                response = {"ok": False, "error": str(exc)}
            channel.basic_publish(exchange="", routing_key=properties.reply_to, body=json.dumps(response).encode("utf-8"), properties=pika.BasicProperties(content_type="application/json", correlation_id=properties.correlation_id))
            channel.basic_ack(delivery_tag=method.delivery_tag)
        channel.basic_consume(queue=self.queue, on_message_callback=on_request)
        LOGGER.info("Waiting for ControlNet requests on %s", self.queue)
        channel.start_consuming()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet", default="lllyasviel/sd-controlnet-depth")
    parser.add_argument("--queue", default="stable-diffusion-controlnet")
    parser.add_argument("--rabbitmq-url", default="amqp://guest:guest@localhost:5672/%2F")
    parser.add_argument("--cpu", action="store_true")
    ControlNetWorker(parser.parse_args()).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()