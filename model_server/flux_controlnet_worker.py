#!/usr/bin/env python3
"""Run a Diffusers Flux depth-ControlNet worker behind RabbitMQ."""

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
from diffusers import (
    FluxControlNetImg2ImgPipeline,
    FluxControlNetModel,
    FluxControlNetPipeline,
)
from PIL import Image


LOGGER = logging.getLogger(__name__)
DEFAULT_PROMPT = (
    "photorealistic aerial 3D town reconstruction, detailed roof and facade materials, "
    "crisp architectural edges, realistic vegetation, accurate building layout, "
    "sharp focus, natural daylight"
)
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, blurry, low quality, warped geometry, "
    "distorted structures, duplicate buildings, text, watermark"
)


class FluxControlNetWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        use_cuda = torch.cuda.is_available() and not args.cpu
        dtype = torch.bfloat16 if use_cuda else torch.float32
        self.device_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
        LOGGER.info("Loading Flux ControlNet: %s", args.controlnet)
        try:
            controlnet = FluxControlNetModel.from_pretrained(
                args.controlnet, torch_dtype=dtype
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "The XLabs checkpoint is a single .safetensors adapter and is not "
                "a Diffusers directory. Convert it to a Diffusers FluxControlNetModel "
                "directory first, then pass that directory with --controlnet. "
                "The raw XLabs repository cannot be loaded by Diffusers 0.40.0 directly."
            ) from exc
        self.pipeline = FluxControlNetImg2ImgPipeline.from_pretrained(
            args.model, controlnet=controlnet, torch_dtype=dtype
        )
        if use_cuda:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.to("cpu")
            if hasattr(self.pipeline, "enable_attention_slicing"):
                self.pipeline.enable_attention_slicing()
        self.pipeline.enable_attention_slicing()
        self.text_pipeline: FluxControlNetPipeline | None = None
        self.queue = args.queue
        self.rabbitmq_url = args.rabbitmq_url
        LOGGER.info("Flux ControlNet loaded on %s", self.device_name)

    @staticmethod
    def _decode_depth(encoded: str) -> Image.Image:
        source = Image.open(io.BytesIO(base64.b64decode(encoded)))
        if source.mode in {"I;16", "I;16B", "I"}:
            values = np.asarray(source, dtype=np.uint16)
            source = Image.fromarray((values / 257.0).astype(np.uint8), mode="L")
        return source.convert("RGB")

    def process(self, request: dict[str, Any]) -> tuple[str, int, int, float]:
        requested_steps = max(1, int(request.get("steps", 25)))
        strength = float(request.get("strength", 0.35))
        mode = request.get("mode", "img2img")
        started_at = time.monotonic()
        control = self._decode_depth(request["control_image_base64"])
        prompt = request.get("prompt", DEFAULT_PROMPT)
        LOGGER.info(
            "Flux request started: mode=%s steps=%d strength=%.3f control_size=%s device=%s",
            mode,
            requested_steps,
            strength,
            control.size,
            self.device_name,
        )
        pipeline_args: dict[str, Any] = {
            "prompt": prompt,
            "image": control,
            "control_image": control,
            "num_inference_steps": requested_steps,
            "strength": strength,
            "guidance_scale": float(request.get("guidance_scale", 3.5)),
            "controlnet_conditioning_scale": float(
                request.get("controlnet_conditioning_scale", 1.0)
            ),
            "generator": torch.Generator(device="cpu").manual_seed(
                int(request.get("seed", 0))
            ),
        }
        with torch.inference_mode():
            if mode == "text2img":
                if self.text_pipeline is None:
                    self.text_pipeline = FluxControlNetPipeline.from_pipe(self.pipeline)
                pipeline_args.pop("strength")
                pipeline_args.pop("image")
                result = self.text_pipeline(**pipeline_args).images[0]
            elif mode == "img2img":
                pipeline_args["image"] = Image.open(
                    io.BytesIO(base64.b64decode(request["image_base64"]))
                ).convert("RGB")
                result = self.pipeline(**pipeline_args).images[0]
            else:
                raise ValueError(f"unsupported Flux mode: {mode}")
        output = io.BytesIO()
        result.save(output, format="PNG")
        duration = time.monotonic() - started_at
        LOGGER.info("Flux request completed: steps=%d duration=%.2fs", requested_steps, duration)
        return base64.b64encode(output.getvalue()).decode("ascii"), requested_steps, requested_steps, duration

    def run(self) -> None:
        parameters = pika.URLParameters(self.rabbitmq_url)
        parameters.heartbeat = 0
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue=self.queue, durable=True)
        channel.basic_qos(prefetch_count=1)

        def on_request(channel: Any, method: Any, properties: Any, body: bytes) -> None:
            try:
                image, requested_steps, steps, duration = self.process(json.loads(body.decode("utf-8")))
                response = {
                    "ok": True,
                    "image_base64": image,
                    "iterations": steps,
                    "requested_iterations": requested_steps,
                    "duration_seconds": round(duration, 2),
                }
            except Exception as exc:
                LOGGER.exception("Flux ControlNet request failed")
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
        LOGGER.info("Waiting for Flux ControlNet requests on %s", self.queue)
        channel.start_consuming()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument(
        "--controlnet", default="XLabs-AI/flux-controlnet-depth-v3",
        help="A converted Diffusers FluxControlNetModel directory",
    )
    parser.add_argument("--queue", default="flux-controlnet-depth")
    parser.add_argument("--rabbitmq-url", default="amqp://guest:guest@localhost:5672/%2F")
    parser.add_argument("--cpu", action="store_true")
    FluxControlNetWorker(parser.parse_args()).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()