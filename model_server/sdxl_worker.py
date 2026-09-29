#!/usr/bin/env python3
"""Generate a textured render from an RGB image using SDXL + depth ControlNet.

This worker implements the SDXL texture-enhancement upgrade: it replaces the
SD 1.5 + depth ControlNet pipeline used by ``controlnet_worker.py`` with
``stabilityai/stable-diffusion-xl-base-1.0`` and
``diffusers/controlnet-depth-sdxl-1.0`` for sharper textures, better material
quality, and more realistic building details, while keeping the same
RabbitMQ RPC request/response format on a dedicated queue.
"""

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
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
)
from PIL import Image


LOGGER = logging.getLogger(__name__)

# Phase 4 prompts: favor sharp, photogrammetry-style detail and discourage
# painterly or structurally inconsistent output.
DEFAULT_PROMPT = (
    "high resolution aerial photogrammetry texture, realistic building materials, "
    "detailed rooftops, clean facade textures, sharp roads, realistic vegetation, "
    "survey grade reconstruction, high frequency detail, consistent lighting"
)
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, watermark, logo, text, duplicate buildings, "
    "warped geometry, distorted structures, deformed roofs, blurry, low quality"
)
# Phase 3: a lower denoising strength preserves mesh geometry and produces
# cleaner reprojection than the SD 1.5 ControlNet worker's default.
DEFAULT_STRENGTH = 0.18


class SDXLWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        dtype = torch.float16 if torch.cuda.is_available() and not args.cpu else torch.float32
        self.device_name = torch.cuda.get_device_name(0) if dtype == torch.float16 else "cpu"
        controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
        self.pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            args.model, controlnet=controlnet, torch_dtype=dtype
        )
        # Phase 2: memory optimizations to fit an 8GB GPU such as an RTX 3060 Ti.
        if dtype == torch.float16:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to("cpu")
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        self.queue = args.queue
        self.rabbitmq_url = args.rabbitmq_url
        self.text_pipeline: StableDiffusionXLControlNetPipeline | None = None

    def process(self, request: dict[str, Any]) -> tuple[str, int, int, float]:
        requested_iterations = int(request.get("steps", 30))
        strength = float(request.get("strength", DEFAULT_STRENGTH))
        iterations = max(1, min(requested_iterations, int(requested_iterations * strength)))
        started_at = time.monotonic()
        mode = request.get("mode", "img2img")
        LOGGER.info(
            "SDXL request started: mode=%s requested_iterations=%d effective_iterations=%d strength=%.3f guidance=%.2f control_scale=%.2f device=%s",
            mode,
            requested_iterations,
            iterations,
            strength,
            float(request.get("guidance_scale", 5.5)),
            float(request.get("controlnet_conditioning_scale", 1.0)),
            self.device_name,
        )
        control_source = Image.open(
            io.BytesIO(base64.b64decode(request["control_image_base64"]))
        )
        if control_source.mode == "I;16":
            control16 = np.asarray(control_source, dtype=np.uint16)
            control_source = Image.fromarray((control16 / 257.0).astype(np.uint8), mode="L")
        control = control_source.convert("RGB")
        LOGGER.info(
            "SDXL inputs mode=%s control_mode=%s control_size=%s",
            mode,
            control_source.mode,
            control.size,
        )
        generator = torch.Generator(device="cpu").manual_seed(int(request.get("seed", 0)))
        prompt = request.get("prompt", DEFAULT_PROMPT)
        negative_prompt = request.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        with torch.inference_mode():
            pipeline_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": requested_iterations,
                "guidance_scale": float(request.get("guidance_scale", 5.5)),
                "controlnet_conditioning_scale": float(
                    request.get("controlnet_conditioning_scale", 1.0)
                ),
                "generator": generator,
            }
            if mode == "text2img":
                if self.text_pipeline is None:
                    self.text_pipeline = StableDiffusionXLControlNetPipeline.from_pipe(
                        self.pipeline
                    )
                result = self.text_pipeline(
                    image=control, **pipeline_kwargs
                ).images[0]
            elif mode == "img2img":
                image = Image.open(
                    io.BytesIO(base64.b64decode(request["image_base64"]))
                ).convert("RGB")
                result = self.pipeline(
                    image=image,
                    strength=strength,
                    control_image=control,
                    **pipeline_kwargs,
                ).images[0]
            else:
                raise ValueError(f"unsupported SDXL mode: {mode}")
        output = io.BytesIO()
        result.save(output, format="PNG")
        duration = time.monotonic() - started_at
        LOGGER.info(
            "SDXL request completed: effective_iterations=%d requested_iterations=%d duration=%.2fs",
            iterations,
            requested_iterations,
            duration,
        )
        return base64.b64encode(output.getvalue()).decode("ascii"), requested_iterations, iterations, duration

    def run(self) -> None:
        parameters = pika.URLParameters(self.rabbitmq_url)
        # Inference can take longer than RabbitMQ's default heartbeat interval.
        parameters.heartbeat = 0
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue=self.queue, durable=True)
        channel.basic_qos(prefetch_count=1)

        def on_request(channel: Any, method: Any, properties: Any, body: bytes) -> None:
            try:
                image_base64, requested_iterations, iterations, duration = self.process(json.loads(body.decode("utf-8")))
                response = {
                    "ok": True,
                    "image_base64": image_base64,
                    "iterations": iterations,
                    "requested_iterations": requested_iterations,
                    "duration_seconds": round(duration, 2),
                }
            except Exception as exc:
                LOGGER.exception("SDXL request failed")
                response = {"ok": False, "error": str(exc)}
            try:
                channel.basic_publish(exchange="", routing_key=properties.reply_to, body=json.dumps(response).encode("utf-8"), properties=pika.BasicProperties(content_type="application/json", correlation_id=properties.correlation_id))
            except pika.exceptions.AMQPError:
                LOGGER.warning(
                    "SDXL client disconnected before the result could be returned",
                    exc_info=True,
                )
            channel.basic_ack(delivery_tag=method.delivery_tag)
        channel.basic_consume(queue=self.queue, on_message_callback=on_request)
        LOGGER.info("Waiting for SDXL requests on %s", self.queue)
        channel.start_consuming()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--controlnet", default="diffusers/controlnet-depth-sdxl-1.0")
    parser.add_argument("--queue", default="stable-diffusion-controlnet-sdxl")
    parser.add_argument("--rabbitmq-url", default="******localhost:5672/%2F")
    parser.add_argument("--cpu", action="store_true")
    SDXLWorker(parser.parse_args()).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
