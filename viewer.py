"""Display merged.gltf with orbit and Quake-style walk controls."""

from pathlib import Path
from datetime import datetime
import base64
import boto3
import io
import json
import logging
import math
import os
import queue
import requests
import threading
import time
import uuid

import numpy as np
import pyglet
from pyglet import gl
import pyrender
import trimesh
from botocore.config import Config
from PIL import Image, ImageOps
from web_server import ViewerWebServer


ASSET_PATH = Path(__file__).with_name("merged.gltf")
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
CAPTURES_PATH = Path(__file__).with_name("captures")
CAMERA_NEAR_RATIO = 0.005
CAMERA_FAR_RATIO = 20.0
CONTROLNET_QUEUE = "stable-diffusion-controlnet"
SDXL_CONTROLNET_QUEUE = "stable-diffusion-controlnet-sdxl"
FLUX_CONTROLNET_QUEUE = "flux-controlnet-depth"
CONFIG_PATH = Path(__file__).with_name("config.json")
R2_PUBLIC_BASE_URL = os.getenv(
	"R2_PUBLIC_BASE_URL", "https://pub-e615b9910ad849b2a11f2ca22ba7869b.r2.dev"
).rstrip("/")
ARTIFACTS_PATH = Path(__file__).with_name("artifacts.json")
ARTIFACTS_LOCK = threading.Lock()
# SDXL prompt: favor crisp architectural materials while preserving the captured
# camera view and depth geometry.
SDXL_PROMPT = (
	"photorealistic aerial 3D town reconstruction, finely detailed roof tiles and facade materials, "
	"crisp architectural edges, detailed windows and vegetation, accurate roads and building layout, "
	"sharp focus, natural daylight, high quality"
)
SDXL_NEGATIVE_PROMPT = (
	"cartoon, illustration, painting, watermark, logo, text, duplicate buildings, "
	"warped geometry, distorted structures, deformed roofs, blurry, low quality"
)
# Moderate denoising improves texture detail while retaining the captured view.
SDXL_STRENGTH = 0.45
RABBITMQ_URL = os.getenv(
	"RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F"
)
WEB_HOST = os.getenv("VIEWER_WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("VIEWER_WEB_PORT", "8765"))
LOGGER = logging.getLogger(__name__)


def load_scene(asset_path: Path) -> tuple[pyrender.Scene, pyrender.PerspectiveCamera]:
	loaded = trimesh.load(asset_path, file_type="gltf", force="scene")
	if not isinstance(loaded, trimesh.Scene):
		loaded = trimesh.Scene(loaded)

	bounds = loaded.bounds
	center = (bounds[0] + bounds[1]) * 0.5
	extent = float(np.max(bounds[1] - bounds[0]))
	extent = max(extent, 0.001)

	render_scene = pyrender.Scene(
		bg_color=[0.035, 0.045, 0.06, 1.0],
		ambient_light=[1.0, 1.0, 1.0],
	)
	meshes = loaded.dump(concatenate=False)
	render_scene.add(pyrender.Mesh.from_trimesh(meshes, smooth=False))
	render_scene._walk_meshes = meshes

	camera = pyrender.PerspectiveCamera(
		yfov=math.radians(45.0),
		aspectRatio=WINDOW_WIDTH / WINDOW_HEIGHT,
		znear=max(extent * CAMERA_NEAR_RATIO, 0.001),
		zfar=extent * CAMERA_FAR_RATIO,
	)
	camera_node = render_scene.add(camera, name="orbit_camera")

	render_scene._orbit_center = np.asarray(center, dtype=np.float32)
	render_scene._orbit_extent = extent
	render_scene._orbit_camera_node = camera_node
	return render_scene, camera


class ModelWindow(pyglet.window.Window):
	def __init__(self, render_scene: pyrender.Scene, camera: pyrender.PerspectiveCamera):
		super().__init__(
			width=WINDOW_WIDTH,
			height=WINDOW_HEIGHT,
			caption="merged.gltf | textured",
			resizable=True,
			vsync=True,
		)
		self.render_scene = render_scene
		self.camera = camera
		self.renderer = pyrender.OffscreenRenderer(WINDOW_WIDTH, WINDOW_HEIGHT)
		self.mode = "textured"
		self.yaw = math.radians(35.0)
		self.pitch = math.radians(18.0)
		self.distance = render_scene._orbit_extent * 2.4
		self.orbit_target = render_scene._orbit_center.copy()
		self.orbit_button = None
		self.walk_mode = False
		self.walk_keys: set[int] = set()
		self.walk_look_drag = False
		self.walk_height = 0.0
		self.last_click: tuple[float, int, int] | None = None
		self.color_buffer = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 4), dtype=np.uint8)
		self.web_commands: queue.Queue[dict[str, object]] = queue.Queue()
		self.web_server: ViewerWebServer | None = None
		self.update_camera()
		pyglet.clock.schedule_interval(self.render_frame, 1.0 / 60.0)
		pyglet.clock.schedule_interval(self.update_walk, 1.0 / 60.0)
		pyglet.clock.schedule_interval(self.process_web_commands, 1.0 / 30.0)

	def web_state(self) -> dict[str, object]:
		with CONFIG_PATH.open(encoding="utf-8") as config_file:
			config = json.load(config_file)
		texture_config = dict(config.get("texture_generator", {}))
		texture_config.pop("key", None)
		return {
			"mode": self.mode,
			"walk_mode": self.walk_mode,
			"yaw": self.yaw,
			"pitch": self.pitch,
			"distance": self.distance,
			"orbit_target": self.orbit_target.astype(float).tolist(),
			"texture_generator": texture_config,
		}

	def enqueue_web_command(self, command: dict[str, object]) -> dict[str, object]:
		completion = threading.Event()
		result: dict[str, object] = {}
		self.web_commands.put({"command": command, "completion": completion, "result": result})
		if not completion.wait(timeout=5.0):
			return {"ok": False, "error": "viewer did not process the command in time"}
		return result

	def process_web_commands(self, _delta_time: float) -> None:
		while True:
			try:
				queued = self.web_commands.get_nowait()
			except queue.Empty:
				return
			command = queued["command"]
			completion = queued["completion"]
			result = queued["result"]
			try:
				self.apply_web_command(command)
				result.update({"ok": True})
			except Exception as exc:
				LOGGER.exception("Web viewer command failed: %s", command)
				self.set_caption(f"Web command failed: {exc}")
				result.update({"ok": False, "error": str(exc)})
			finally:
				completion.set()

	def apply_web_command(self, command: dict[str, object]) -> None:
		action = command.get("action")
		if action == "set_mode":
			mode = command.get("mode")
			if mode not in {"textured", "depth"}:
				raise ValueError("mode must be textured or depth")
			self.mode = str(mode)
			self.set_caption(f"merged.gltf | {self.mode}")
		elif action == "set_navigation":
			navigation_mode = command.get("mode")
			if navigation_mode == "walk" and not self.walk_mode:
				self.enter_walk_mode(self.width // 2, self.height // 2)
			elif navigation_mode == "orbit" and self.walk_mode:
				self.walk_mode = False
				self.walk_keys.clear()
				self.walk_look_drag = False
				self.set_exclusive_mouse(False)
				self.update_camera()
				self.set_caption(f"merged.gltf | orbit | {self.mode}")
		elif action == "generate":
			generator = command.get("generator")
			if generator == "sdxl":
				self.generate_sdxl_view()
			elif generator == "configured":
				self.generate_configured_texture_view()
			else:
				raise ValueError("unsupported generator")
		elif action == "save_view":
			self.save_current_view()
		elif action == "navigate":
			self.navigate_to_artifact(int(command["index"]))
		elif action == "set_options":
			self.update_texture_options(command.get("options", {}))
		else:
			raise ValueError(f"unsupported web action: {action}")

	def navigate_to_artifact(self, index: int) -> None:
		with ARTIFACTS_PATH.open(encoding="utf-8") as artifacts_file:
			artifacts = json.load(artifacts_file)
		if not isinstance(artifacts, list) or index < 0 or index >= len(artifacts):
			raise ValueError("artifact index is out of range")
		camera_pose = artifacts[index].get("camera_pose", {})
		self.yaw = float(camera_pose["yaw"])
		self.pitch = float(camera_pose["pitch"])
		self.distance = float(camera_pose["distance"])
		self.orbit_target = np.asarray(camera_pose["orbit_target"], dtype=np.float32)
		if self.walk_mode:
			self.walk_mode = False
			self.walk_keys.clear()
			self.walk_look_drag = False
			self.set_exclusive_mouse(False)
		self.update_camera()
		stored_matrix = camera_pose.get("matrix")
		if stored_matrix is not None:
			pose = np.asarray(stored_matrix, dtype=np.float32)
			if pose.shape != (4, 4):
				raise ValueError("artifact camera matrix must be 4x4")
			self.render_scene.set_pose(self.render_scene._orbit_camera_node, pose)
		LOGGER.info(
			"Navigated to artifact index=%d yaw=%.6f pitch=%.6f distance=%.6f target=%s pose=%s",
			index,
			self.yaw,
			self.pitch,
			self.distance,
			self.orbit_target.tolist(),
			self.render_scene.get_pose(self.render_scene._orbit_camera_node).tolist(),
		)
		self.set_caption(f"Navigated to artifact {index + 1}")

	def update_texture_options(self, options: object) -> None:
		if not isinstance(options, dict):
			raise ValueError("options must be an object")
		with CONFIG_PATH.open(encoding="utf-8") as config_file:
			config = json.load(config_file)
		texture_config = config.setdefault("texture_generator", {})
		for key in ("prompt", "resolution", "aspect_ratio"):
			if key in options:
				texture_config[key] = str(options[key])
		temporary_path = CONFIG_PATH.with_suffix(".json.tmp")
		with temporary_path.open("w", encoding="utf-8") as config_file:
			json.dump(config, config_file, indent=4)
			config_file.write("\n")
		temporary_path.replace(CONFIG_PATH)
		self.set_caption("Texture options saved")

	def update_camera(self) -> None:
		center = self.orbit_target
		distance = self.distance
		camera_position = np.array(
			[
				center[0] + distance * math.cos(self.pitch) * math.sin(self.yaw),
				center[1] + distance * math.sin(self.pitch),
				center[2] + distance * math.cos(self.pitch) * math.cos(self.yaw),
			],
			dtype=np.float32,
		)
		forward = center - camera_position
		forward /= np.linalg.norm(forward)
		right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
		right /= np.linalg.norm(right)
		up = np.cross(right, forward)
		pose = np.eye(4, dtype=np.float32)
		pose[:3, :3] = np.column_stack((right, up, -forward))
		pose[:3, 3] = camera_position
		self.render_scene.set_pose(self.render_scene._orbit_camera_node, pose)

	def render_frame(self, _delta_time: float) -> None:
		flags = pyrender.RenderFlags.RGBA
		if self.mode == "depth":
			depth = self.renderer.render(self.render_scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
			far = self.camera.zfar
			normalized = np.clip(1.0 - depth / far, 0.0, 1.0)
			grayscale = (normalized * 255).astype(np.uint8)
			self.color_buffer[:, :, :3] = grayscale[:, :, None]
			self.color_buffer[:, :, 3] = 255
		else:
			self.color_buffer, _ = self.renderer.render(self.render_scene, flags=flags)
			self.color_buffer = self.color_buffer.copy()
		self.invalid = True

	def on_draw(self) -> None:
		self.clear()
		image = pyglet.image.ImageData(
			self.width,
			self.height,
			"RGBA",
			self.color_buffer.tobytes(),
			pitch=-self.width * 4,
		)
		image.blit(0, 0, width=self.width, height=self.height)
	def on_resize(self, width: int, height: int) -> None:
		self.camera.aspectRatio = width / max(height, 1)
		self.renderer.delete()
		self.renderer = pyrender.OffscreenRenderer(width, height)
		self.color_buffer = np.zeros((height, width, 4), dtype=np.uint8)

	def on_key_press(self, symbol: int, _modifiers: int) -> None:
		if symbol == pyglet.window.key.Q:
			self.close()
			return
		if symbol == pyglet.window.key.S:
			self.save_current_view()
			return
		if symbol == pyglet.window.key.R and not self.walk_mode:
			self.generate_sdxl_view()
			return
		if symbol == pyglet.window.key.T and not self.walk_mode:
			LOGGER.info("T pressed: starting configured texture generation")
			self.generate_configured_texture_view()
			return
		if symbol == pyglet.window.key.F:
			self.set_fullscreen(not self.fullscreen)
			self.set_exclusive_mouse(self.fullscreen and self.walk_mode)
			return
		if symbol == pyglet.window.key.G and self.walk_mode:
			self.walk_mode = False
			self.walk_keys.clear()
			self.walk_look_drag = False
			self.set_exclusive_mouse(False)
			self.update_camera()
			self.set_caption("merged.gltf | orbit | %s" % self.mode)
			return
		if self.walk_mode:
			self.walk_keys.add(symbol)
			return
		if symbol == pyglet.window.key.SPACE:
			self.mode = "depth" if self.mode == "textured" else "textured"
			self.set_caption(f"merged.gltf | {self.mode}")

	def save_current_view(self) -> None:
		CAPTURES_PATH.mkdir(parents=True, exist_ok=True)
		stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		view_path = CAPTURES_PATH / f"view_{stamp}.png"
		depth_path = CAPTURES_PATH / f"depth_{stamp}.png"
		color, depth = self.renderer.render(self.render_scene, flags=pyrender.RenderFlags.RGBA)
		Image.fromarray(color[:, :, :3], mode="RGB").save(view_path)
		depth = self.renderer.render(
			self.render_scene, flags=pyrender.RenderFlags.DEPTH_ONLY
		)
		valid = np.isfinite(depth) & (depth > 0.0)
		depth_image = np.zeros(depth.shape, dtype=np.uint16)
		if valid.any():
			nearest, farthest = np.percentile(depth[valid], [2.0, 98.0])
			if farthest > nearest:
				depth_image[valid] = np.asarray(
					(farthest - depth[valid])
					/ (farthest - nearest)
					* np.iinfo(np.uint16).max,
					dtype=np.uint16,
				)
			else:
				depth_image[valid] = np.iinfo(np.uint16).max
		Image.fromarray(depth_image, mode="I;16").save(depth_path)
		self.set_caption(f"Saved {view_path.name} and {depth_path.name}")

	def generate_controlnet_view(self) -> None:
		try:
			view_path, depth_path = self.save_current_view_files()
		except Exception as exc:
			self.set_caption(f"ControlNet capture failed: {exc}")
			return
		self.set_caption("Sending RGB/depth view to ControlNet...")
		threading.Thread(
			target=self._request_controlnet,
			args=(view_path, depth_path),
			daemon=True,
			name="controlnet-request",
		).start()

	def generate_sdxl_view(self) -> None:
		try:
			view_path, depth_path = self.save_current_view_files()
		except Exception as exc:
			self.set_caption(f"SDXL capture failed: {exc}")
			return
		self.set_caption("Sending RGB/depth view to SDXL...")
		threading.Thread(
			target=self._request_sdxl,
			args=(view_path, depth_path),
			daemon=True,
			name="sdxl-request",
		).start()

	def generate_flux_view(self) -> None:
		try:
			view_path, depth_path = self.save_current_view_files()
		except Exception as exc:
			self.set_caption(f"Flux capture failed: {exc}")
			return
		self.set_caption("Sending RGB/depth view to Flux...")
		threading.Thread(
			target=self._request_flux,
			args=(view_path, depth_path),
			daemon=True,
			name="flux-request",
		).start()

	def generate_configured_texture_view(self) -> None:
		try:
			texture_config = self.load_texture_config()
			LOGGER.info(
				"Configured texture settings: model=%s resolution=%s aspect_ratio=%s prompt=%s",
				texture_config.get("model"),
				texture_config.get("resolution"),
				texture_config.get("aspect_ratio"),
				texture_config.get("prompt"),
			)
			view_path, depth_path, camera_pose = self.save_texture_view_files(
				texture_config.get("aspect_ratio", "4:3")
			)
			LOGGER.info(
				"Configured texture captures ready: rgb=%s (%d bytes) depth=%s (%d bytes)",
				view_path,
				view_path.stat().st_size,
				depth_path,
				depth_path.stat().st_size,
			)
		except Exception as exc:
			self.set_caption(f"Configured texture capture failed: {exc}")
			return
		self.set_caption(
			f"Sending {texture_config.get('model', 'configured')} texture request..."
		)
		threading.Thread(
			target=self._request_configured_texture,
			args=(view_path, depth_path, camera_pose, texture_config),
			daemon=True,
			name="configured-texture-request",
		).start()

	@staticmethod
	def load_texture_config() -> dict[str, str]:
		with CONFIG_PATH.open(encoding="utf-8") as config_file:
			config = json.load(config_file)
		texture_config = config.get("texture_generator")
		if not isinstance(texture_config, dict):
			raise ValueError("config.json is missing texture_generator")
		return texture_config

	def save_texture_view_files(self, aspect_ratio: str) -> tuple[Path, Path, dict[str, object]]:
		try:
			aspect_width, aspect_height = (int(value) for value in aspect_ratio.split(":", 1))
			if aspect_width <= 0 or aspect_height <= 0:
				raise ValueError
		except (ValueError, TypeError):
			raise ValueError(f"invalid texture aspect ratio: {aspect_ratio}")
		output_size = (1024, max(1, round(1024 * aspect_height / aspect_width)))
		LOGGER.info(
			"Rendering configured texture inputs: source=%dx%d target=%dx%d aspect=%s",
			self.width,
			self.height,
			output_size[0],
			output_size[1],
			aspect_ratio,
		)
		CAPTURES_PATH.mkdir(parents=True, exist_ok=True)
		stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		view_path = CAPTURES_PATH / f"texture_view_{stamp}.png"
		depth_path = CAPTURES_PATH / f"texture_depth_{stamp}.png"
		color, depth = self.renderer.render(self.render_scene, flags=pyrender.RenderFlags.RGBA)
		view_image = ImageOps.fit(
			Image.fromarray(color[:, :, :3], mode="RGB"), output_size, method=Image.Resampling.LANCZOS
		)
		depth_image = ImageOps.fit(
			Image.fromarray(depth, mode="F"), output_size, method=Image.Resampling.NEAREST
		)
		depth_array = np.asarray(depth_image, dtype=np.float32)
		valid = np.isfinite(depth_array) & (depth_array > 0.0)
		depth_output = np.zeros(depth_array.shape, dtype=np.uint16)
		if valid.any():
			nearest, farthest = np.percentile(depth_array[valid], [2.0, 98.0])
			LOGGER.info(
				"Configured depth capture: valid_pixels=%d/%d range=%.4f..%.4f",
				int(valid.sum()),
				valid.size,
				float(nearest),
				float(farthest),
			)
			if farthest > nearest:
				depth_output[valid] = np.asarray(
					(farthest - depth_array[valid]) / (farthest - nearest) * np.iinfo(np.uint16).max,
					dtype=np.uint16,
				)
		view_image.save(view_path)
		Image.fromarray(depth_output, mode="I;16").save(depth_path)
		pose = self.render_scene.get_pose(self.render_scene._orbit_camera_node)
		camera_pose = {
			"matrix": pose.astype(float).tolist(),
			"yaw": self.yaw,
			"pitch": self.pitch,
			"distance": self.distance,
			"orbit_target": self.orbit_target.astype(float).tolist(),
			"capture_size": {"width": output_size[0], "height": output_size[1]},
		}
		LOGGER.info("Configured camera pose captured: yaw=%.4f pitch=%.4f distance=%.4f", self.yaw, self.pitch, self.distance)
		return view_path, depth_path, camera_pose

	def _request_configured_texture(
		self,
		view_path: Path,
		depth_path: Path,
		camera_pose: dict[str, object],
		texture_config: dict[str, str],
	) -> None:
		try:
			model = texture_config.get("model")
			LOGGER.info("Configured texture dispatch selected model=%s", model)
			if model != "runpod_nano_banana_2":
				raise ValueError(f"unsupported texture_generator.model: {model}")
			output_path = self._request_runpod_nano_banana(
				view_path, depth_path, texture_config
			)
			generated_depth_path = self._request_configured_depth(output_path)
			self.append_artifact(
				view_path,
				depth_path,
				output_path,
				generated_depth_path,
				camera_pose,
				texture_config,
			)
			LOGGER.info("Configured texture completed: output=%s bytes=%d", output_path, output_path.stat().st_size)
			self.set_caption(
				f"Configured texture and depth saved: {output_path.name}, {generated_depth_path.name}"
			)
		except Exception as exc:
			LOGGER.exception("Configured texture request failed")
			self.set_caption(f"Configured texture failed: {exc}")

	@staticmethod
	def _request_runpod_nano_banana(
		view_path: Path, depth_path: Path, texture_config: dict[str, str]
	) -> Path:
		with CONFIG_PATH.open(encoding="utf-8") as config_file:
			config = json.load(config_file)
		r2_config = config["r2"]
		s3 = boto3.client(
			"s3",
			endpoint_url=f"https://{r2_config['account_id']}.r2.cloudflarestorage.com",
			aws_access_key_id=r2_config["access_key"],
			aws_secret_access_key=r2_config["secret_key"],
			region_name="auto",
			config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
		)
		stamp = uuid.uuid4().hex
		keys = [f"images/texture-{stamp}-view.png", f"images/texture-{stamp}-depth.png"]
		LOGGER.info(
			"RunPod staging started: bucket=%s keys=%s endpoint=%s",
			r2_config["bucket"],
			keys,
			f"https://{r2_config['account_id']}.r2.cloudflarestorage.com",
		)
		try:
			for path, key in zip((view_path, depth_path), keys):
				LOGGER.info("Uploading R2 object: local=%s key=%s bytes=%d", path, key, path.stat().st_size)
				with path.open("rb") as image_file:
					s3.upload_fileobj(
						image_file,
						r2_config["bucket"],
						key,
						ExtraArgs={"ContentType": "image/png"},
					)
				LOGGER.info("R2 upload complete: key=%s", key)
			payload = {
				"input": {
					"images": [f"{R2_PUBLIC_BASE_URL}/{key}" for key in keys],
					"prompt": texture_config.get("prompt", "improve texture quality and enhance details"),
					"resolution": texture_config.get("resolution", "1k"),
					"aspect_ratio": texture_config.get("aspect_ratio", "4:3"),
					"output_format": "png",
				}
			}
			LOGGER.info(
				"Calling RunPod Nano Banana 2: endpoint=%s images=%d resolution=%s aspect_ratio=%s prompt=%s",
				"google-nano-banana-2-edit/runsync",
				len(payload["input"]["images"]),
				payload["input"]["resolution"],
				payload["input"]["aspect_ratio"],
				payload["input"]["prompt"],
			)
			request_started = time.monotonic()
			response = requests.post(
				"https://api.runpod.ai/v2/google-nano-banana-2-edit/runsync",
				headers={
					"Authorization": f"Bearer {texture_config['key']}",
					"Content-Type": "application/json",
				},
				json=payload,
				timeout=300,
			)
			LOGGER.info(
				"RunPod response received: status=%d duration=%.2fs",
				response.status_code,
				time.monotonic() - request_started,
			)
			response.raise_for_status()
			data = response.json()
			output_data = data.get("output", {})
			image_url = output_data.get("image_url") or output_data.get("result")
			if not image_url:
				raise RuntimeError(f"RunPod response did not contain output image URL: {data}")
			LOGGER.info("Downloading RunPod output image: url=%s", image_url)
			image_response = requests.get(image_url, timeout=120)
			image_response.raise_for_status()
			output_path = CAPTURES_PATH / f"view_runpod_nano_banana_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
			output_path.write_bytes(image_response.content)
			LOGGER.info("RunPod output saved: path=%s bytes=%d", output_path, len(image_response.content))
			return output_path
		finally:
			for key in keys:
				try:
					LOGGER.info("Deleting temporary R2 object: key=%s", key)
					s3.delete_object(Bucket=r2_config["bucket"], Key=key)
					LOGGER.info("Temporary R2 object deleted: key=%s", key)
				except Exception:
					LOGGER.warning("Could not delete temporary R2 object %s", key, exc_info=True)

	@staticmethod
	def _request_configured_depth(image_path: Path) -> Path:
		with CONFIG_PATH.open(encoding="utf-8") as config_file:
			config = json.load(config_file)
		depth_config = config.get("depth_generator")
		if not isinstance(depth_config, dict):
			raise ValueError("config.json is missing depth_generator")
		if not depth_config.get("rabbitmq_url"):
			raise ValueError("depth_generator.rabbitmq_url is required")
		queue = depth_config.get("queue", "depth-anything")
		model = depth_config.get("model", "depth-anything/Depth-Anything-V2")
		request = {
			"image_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
		}
		correlation_id = str(uuid.uuid4())
		LOGGER.info(
			"Sending generated texture to depth worker: model=%s queue=%s image=%s",
			model,
			queue,
			image_path.name,
		)
		import pika
		parameters = pika.URLParameters(depth_config["rabbitmq_url"])
		parameters.heartbeat = 0
		connection = pika.BlockingConnection(parameters)
		channel = connection.channel()
		channel.queue_declare(queue=queue, durable=True)
		reply_queue = channel.queue_declare(queue="", exclusive=True).method.queue
		response: bytes | None = None

		def on_response(_channel, _method, properties, body: bytes) -> None:
			nonlocal response
			if properties.correlation_id == correlation_id:
				response = body

		consumer_tag = channel.basic_consume(
			queue=reply_queue, on_message_callback=on_response, auto_ack=True
		)
		channel.basic_publish(
			exchange="",
			routing_key=queue,
			body=json.dumps(request).encode("utf-8"),
			properties=pika.BasicProperties(
				content_type="application/json",
				correlation_id=correlation_id,
				reply_to=reply_queue,
			),
		)
		try:
			deadline = time.monotonic() + float(depth_config.get("timeout", 300))
			while response is None:
				remaining = deadline - time.monotonic()
				if remaining <= 0:
					raise TimeoutError("Depth Anything worker response timed out")
				connection.process_data_events(time_limit=min(1.0, remaining))
		finally:
			channel.basic_cancel(consumer_tag)
			connection.close()

		result = json.loads(response.decode("utf-8"))
		if not result.get("ok"):
			raise RuntimeError(result.get("error", "Depth Anything worker failed"))
		depth_base64 = result.get("depth_image_base64")
		if not depth_base64:
			raise RuntimeError("Depth Anything response did not contain depth_image_base64")
		depth_path = CAPTURES_PATH / (
			f"depth_anything_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
		)
		depth_path.write_bytes(base64.b64decode(depth_base64))
		LOGGER.info(
			"Generated depth saved: path=%s bytes=%d size=%sx%s range=%s..%s",
			depth_path,
			depth_path.stat().st_size,
			result.get("width", "?"),
			result.get("height", "?"),
			result.get("depth_min", "?"),
			result.get("depth_max", "?"),
		)
		return depth_path

	def append_artifact(
		self,
		view_path: Path,
		depth_path: Path,
		output_path: Path,
		generated_depth_path: Path,
		camera_pose: dict[str, object],
		texture_config: dict[str, str],
	) -> None:
		try:
			with ARTIFACTS_LOCK:
				if ARTIFACTS_PATH.exists():
					with ARTIFACTS_PATH.open(encoding="utf-8") as artifacts_file:
						artifacts = json.load(artifacts_file)
				else:
					artifacts = []
				if not isinstance(artifacts, list):
					raise ValueError("artifacts.json must contain a JSON array")
				artifact = {
					"created_at": datetime.now().astimezone().isoformat(),
					"generator": texture_config.get("model"),
					"depth_render": depth_path.name,
					"generated_depth_render": generated_depth_path.name,
					"texture_render": view_path.name,
					"result_render": output_path.name,
					"camera_pose": camera_pose,
				}
				artifacts.append(artifact)
				temporary_path = ARTIFACTS_PATH.with_suffix(".json.tmp")
				with temporary_path.open("w", encoding="utf-8") as artifacts_file:
					json.dump(artifacts, artifacts_file, indent=2)
					artifacts_file.write("\n")
				temporary_path.replace(ARTIFACTS_PATH)
			LOGGER.info("Artifact recorded: file=%s result=%s", ARTIFACTS_PATH, output_path.name)
		except Exception:
			LOGGER.exception("Could not record texture artifact")

	def save_current_view_files(self) -> tuple[Path, Path]:
		CAPTURES_PATH.mkdir(parents=True, exist_ok=True)
		stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		view_path = CAPTURES_PATH / f"view_{stamp}.png"
		depth_path = CAPTURES_PATH / f"depth_{stamp}.png"
		color, depth = self.renderer.render(self.render_scene, flags=pyrender.RenderFlags.RGBA)
		Image.fromarray(color[:, :, :3], mode="RGB").save(view_path)
		valid = np.isfinite(depth) & (depth > 0.0)
		depth_image = np.zeros(depth.shape, dtype=np.uint16)
		if valid.any():
			nearest = depth[valid].min()
			farthest = depth[valid].max()
			if farthest > nearest:
				depth_image[valid] = np.asarray(
					(farthest - depth[valid]) / (farthest - nearest) * np.iinfo(np.uint16).max,
					dtype=np.uint16,
				)
			else:
				depth_image[valid] = np.iinfo(np.uint16).max
		Image.fromarray(depth_image, mode="I;16").save(depth_path)
		LOGGER.info(
			"Saved ControlNet capture rgb=%s depth=%s valid_depth_range=%.3f..%.3f",
			view_path.name,
			depth_path.name,
			float(nearest) if valid.any() else 0.0,
			float(farthest) if valid.any() else 0.0,
		)
		return view_path, depth_path

	def _request_controlnet(self, view_path: Path, depth_path: Path) -> None:
		request = {
			"prompt": "photorealistic textured reconstruction, natural materials, realistic lighting, preserve exact geometry and composition",
			"negative_prompt": "changed camera angle, changed geometry, warped structures, extra objects, text, watermark, blur",
			"steps": 30,
			"strength": 0.20,
			"guidance_scale": 5.5,
			"controlnet_conditioning_scale": 1.25,
			"seed": 0,
		}
		self._request_controlnet_worker(
			view_path, depth_path, queue=CONTROLNET_QUEUE, worker_label="ControlNet", request=request
		)

	def _request_sdxl(self, view_path: Path, depth_path: Path) -> None:
		request = {
			"prompt": SDXL_PROMPT,
			"negative_prompt": SDXL_NEGATIVE_PROMPT,
			"steps": 40,
			"strength": SDXL_STRENGTH,
			"guidance_scale": 6.0,
			"controlnet_conditioning_scale": 1.0,
			"seed": 0,
		}
		self._request_controlnet_worker(
			view_path, depth_path, queue=SDXL_CONTROLNET_QUEUE, worker_label="SDXL", request=request
		)

	def _request_flux(self, view_path: Path, depth_path: Path) -> None:
		request = {
			"mode": "img2img",
			"prompt": SDXL_PROMPT,
			"negative_prompt": SDXL_NEGATIVE_PROMPT,
			"steps": 25,
			"strength": 0.35,
			"guidance_scale": 3.5,
			"controlnet_conditioning_scale": 1.0,
			"seed": 0,
		}
		self._request_controlnet_worker(
			view_path, depth_path, queue=FLUX_CONTROLNET_QUEUE, worker_label="Flux", request=request
		)

	def _request_controlnet_worker(
		self, view_path: Path, depth_path: Path, *, queue: str, worker_label: str, request: dict
	) -> None:
		try:
			import pika
			def encode(path: Path) -> str:
				return base64.b64encode(path.read_bytes()).decode("ascii")
			request = {
				"image_base64": encode(view_path),
				"control_image_base64": encode(depth_path),
				**request,
			}
			LOGGER.info(
				"Sending %s request steps=%d strength=%.2f guidance=%.2f control_scale=%.2f",
				worker_label,
				request["steps"],
				request["strength"],
				request["guidance_scale"],
				request["controlnet_conditioning_scale"],
			)
			correlation_id = str(uuid.uuid4())
			parameters = pika.URLParameters(RABBITMQ_URL)
			parameters.heartbeat = 0
			connection = pika.BlockingConnection(parameters)
			channel = connection.channel()
			channel.queue_declare(queue=queue, durable=True)
			reply_queue = channel.queue_declare(queue="", exclusive=True).method.queue
			response: bytes | None = None
			def on_response(_channel, _method, properties, body: bytes) -> None:
				nonlocal response
				if properties.correlation_id == correlation_id:
					response = body
			consumer_tag = channel.basic_consume(queue=reply_queue, on_message_callback=on_response, auto_ack=True)
			channel.basic_publish(
				exchange="", routing_key=queue, body=json.dumps(request).encode("utf-8"),
				properties=pika.BasicProperties(content_type="application/json", correlation_id=correlation_id, reply_to=reply_queue),
			)
			while response is None:
				connection.process_data_events(time_limit=1.0)
			channel.basic_cancel(consumer_tag)
			connection.close()
			result = json.loads(response.decode("utf-8"))
			if not result.get("ok"):
				raise RuntimeError(result.get("error", f"{worker_label} worker failed"))
			output_path = (
				CAPTURES_PATH
				/ f"view_{worker_label.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
			)
			output_path.write_bytes(base64.b64decode(result["image_base64"]))
			LOGGER.info(
				"%s response received iterations=%s/%s duration=%ss output=%s",
				worker_label,
				result.get("iterations", "?"),
				result.get("requested_iterations", "?"),
				result.get("duration_seconds", "?"),
				output_path.name,
			)
			iterations = result.get("iterations", request["steps"])
			requested_iterations = result.get("requested_iterations", request["steps"])
			duration = result.get("duration_seconds")
			self.set_caption(
				f"{worker_label} render saved: {output_path.name} | "
				f"iterations: {iterations}/{requested_iterations} | duration: {duration}s"
			)
		except Exception as exc:
			self.set_caption(f"{worker_label} request failed: {exc}")

	def on_key_release(self, symbol: int, _modifiers: int) -> None:
		self.walk_keys.discard(symbol)

	def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
		if button == pyglet.window.mouse.LEFT:
			if self.walk_mode:
				self.walk_look_drag = not self.fullscreen
				return
			self.orbit_button = "orbit" if modifiers & pyglet.window.key.MOD_SHIFT else "pan"
		elif button == pyglet.window.mouse.MIDDLE:
			if self.walk_mode:
				return
			self.orbit_button = "orbit"
	def on_mouse_release(self, _x: int, _y: int, button: int, _modifiers: int) -> None:
		if button == pyglet.window.mouse.MIDDLE and self.walk_mode:
			return
		if button == pyglet.window.mouse.LEFT:
			self.walk_look_drag = False
		if button in (
			pyglet.window.mouse.LEFT,
			pyglet.window.mouse.MIDDLE,
		):
			self.orbit_button = None

	def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, _buttons: int, _modifiers: int) -> None:
		if self.walk_mode and (self.fullscreen or self.walk_look_drag):
			self.turn_camera(-dx * 0.004, dy * 0.004)
		elif self.orbit_button == "orbit" and not self.walk_mode:
			self.yaw -= dx * 0.01
			self.pitch = np.clip(self.pitch - dy * 0.01, -1.45, 1.45)
			self.update_camera()
		elif self.orbit_button == "pan" and not self.walk_mode:
			self.pan_orbit(dx, dy)
		elif self.orbit_button == "zoom" and not self.walk_mode:
			self.distance *= math.exp(-dy * 0.01)
			extent = self.render_scene._orbit_extent
			self.distance = float(np.clip(self.distance, extent * 0.03, extent * 20.0))
			self.update_camera()

	def on_mouse_motion(self, _x: int, _y: int, dx: int, dy: int) -> None:
		if self.walk_mode and self.fullscreen:
			self.turn_camera(-dx * 0.004, dy * 0.004)

	def enter_walk_mode(self, x: int, y: int) -> None:
		hit = self.scene_hit(x, y)
		if hit is None:
			return
		camera_position, forward = hit
		forward[1] = 0.0
		if np.linalg.norm(forward) < 1e-6:
			forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
		forward /= np.linalg.norm(forward)
		eye_height = 1.5
		camera_position = camera_position + np.array([0.0, eye_height, 0.0], dtype=np.float32)
		self.set_walk_pose(camera_position, forward)
		self.walk_mode = True
		self.walk_height = float(eye_height)
		self.set_exclusive_mouse(self.fullscreen)
		self.update_walk_caption()

	def update_walk_caption(self) -> None:
		self.set_caption(
			"merged.gltf | walk | height %.2f m | F fullscreen | G orbit"
			% self.walk_height
		)

	def scene_hit(self, x: int, y: int) -> tuple[np.ndarray, np.ndarray] | None:
		width, height = self.width, self.height
		horizontal_tangent = math.tan(self.camera.yfov / 2.0) * self.camera.aspectRatio
		local_x = (2.0 * x / max(width, 1) - 1.0) * horizontal_tangent
		local_y = (2.0 * y / max(height, 1) - 1.0) * math.tan(self.camera.yfov / 2.0)
		pose = self.render_scene.get_pose(self.render_scene._orbit_camera_node)
		direction = pose[:3, :3] @ np.array([local_x, local_y, -1.0], dtype=np.float32)
		direction /= np.linalg.norm(direction)
		origin = pose[:3, 3]
		closest_distance = float("inf")
		closest_point = None
		for mesh in self.render_scene._walk_meshes:
			triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
			edge_one = triangles[:, 1] - triangles[:, 0]
			edge_two = triangles[:, 2] - triangles[:, 0]
			cross_direction = np.cross(direction, edge_two)
			determinant = np.einsum("ij,ij->i", edge_one, cross_direction)
			valid = np.abs(determinant) > 1e-8
			if not valid.any():
				continue
			inverse = 1.0 / determinant[valid]
			offset = origin - triangles[valid, 0]
			first = inverse * np.einsum("ij,ij->i", offset, cross_direction[valid])
			second = inverse * np.einsum("j,ij->i", direction, np.cross(offset, edge_one[valid]))
			distance = inverse * np.einsum("ij,ij->i", edge_two[valid], np.cross(offset, edge_one[valid]))
			inside = (first >= 0.0) & (second >= 0.0) & (first + second <= 1.0) & (distance > 1e-8)
			if inside.any():
				candidate = float(np.min(distance[inside]))
				if candidate < closest_distance:
					closest_distance = candidate
					closest_point = origin + direction * candidate
		if closest_point is None:
			return None
		return closest_point.astype(np.float32), (-pose[:3, 2]).astype(np.float32)

	def lowest_triangle_y(self, x: float, z: float) -> float | None:
		lowest_y = None
		point = np.array([x, z], dtype=np.float32)
		for mesh in self.render_scene._walk_meshes:
			triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
			projected = triangles[:, :, (0, 2)]
			edge_one = projected[:, 1] - projected[:, 0]
			edge_two = projected[:, 2] - projected[:, 0]
			denominator = edge_one[:, 0] * edge_two[:, 1] - edge_two[:, 0] * edge_one[:, 1]
			valid = np.abs(denominator) > 1e-8
			if not valid.any():
				continue
			relative = point - projected[valid, 0]
			edge_one_valid = edge_one[valid]
			edge_two_valid = edge_two[valid]
			denominator_valid = denominator[valid]
			first = (
				relative[:, 0] * edge_two_valid[:, 1]
				- edge_two_valid[:, 0] * relative[:, 1]
			) / denominator_valid
			second = (
				edge_one_valid[:, 0] * relative[:, 1]
				- relative[:, 0] * edge_one_valid[:, 1]
			) / denominator_valid
			inside = (
				(first >= -1e-6)
				& (second >= -1e-6)
				& (first + second <= 1.0 + 1e-6)
			)
			if not inside.any():
				continue
			valid_triangles = triangles[valid][inside]
			first_inside = first[inside]
			second_inside = second[inside]
			heights = (
				valid_triangles[:, 0, 1]
				+ first_inside * (valid_triangles[:, 1, 1] - valid_triangles[:, 0, 1])
				+ second_inside * (valid_triangles[:, 2, 1] - valid_triangles[:, 0, 1])
			)
			candidate = float(np.min(heights))
			if lowest_y is None or candidate < lowest_y:
				lowest_y = candidate
		return lowest_y

	def set_walk_pose(self, position: np.ndarray, forward: np.ndarray) -> None:
		world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
		right = np.cross(forward, world_up)
		right /= np.linalg.norm(right)
		up = np.cross(right, forward)
		pose = np.eye(4, dtype=np.float32)
		pose[:3, :3] = np.column_stack((right, up, -forward))
		pose[:3, 3] = position
		self.render_scene.set_pose(self.render_scene._orbit_camera_node, pose)

	def turn_camera(self, yaw: float, pitch: float) -> None:
		pose = self.render_scene.get_pose(self.render_scene._orbit_camera_node)
		forward = -pose[:3, 2].copy()
		world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
		forward = self.rotate_vector(forward, world_up, yaw)
		right = np.cross(forward, world_up)
		right /= np.linalg.norm(right)
		candidate = self.rotate_vector(forward, right, pitch)
		if abs(candidate[1]) < 0.996:
			forward = candidate
		self.set_walk_pose(pose[:3, 3], forward)

	@staticmethod
	def rotate_vector(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
		axis = axis / np.linalg.norm(axis)
		return vector * math.cos(angle) + np.cross(axis, vector) * math.sin(angle) + axis * np.dot(axis, vector) * (1.0 - math.cos(angle))

	def update_walk(self, delta_time: float) -> None:
		if not self.walk_mode:
			return
		pose = self.render_scene.get_pose(self.render_scene._orbit_camera_node)
		forward = -pose[:3, 2].copy()
		forward[1] = 0.0
		forward /= max(np.linalg.norm(forward), 1e-8)
		right = pose[:3, 0].copy()
		right[1] = 0.0
		right /= max(np.linalg.norm(right), 1e-8)
		movement = np.zeros(3, dtype=np.float32)
		if pyglet.window.key.UP in self.walk_keys:
			movement += forward
		if pyglet.window.key.DOWN in self.walk_keys:
			movement -= forward
		if self.fullscreen:
			if pyglet.window.key.LEFT in self.walk_keys:
				movement -= right
			if pyglet.window.key.RIGHT in self.walk_keys:
				movement += right
		if np.linalg.norm(movement) > 0:
			movement /= np.linalg.norm(movement)
			current_ground_y = self.lowest_triangle_y(pose[0, 3], pose[2, 3])
			next_position = pose[:3, 3] + movement * float(self.render_scene._orbit_extent) * 0.06 * delta_time
			next_ground_y = self.lowest_triangle_y(next_position[0], next_position[2])
			step_up = (
				next_ground_y - current_ground_y
				if current_ground_y is not None and next_ground_y is not None
				else 0.0
			)
			if step_up <= 2.0:
				pose[:3, 3] = next_position
				if next_ground_y is not None:
					pose[1, 3] = next_ground_y + self.walk_height
				self.render_scene.set_pose(self.render_scene._orbit_camera_node, pose)
		if pyglet.window.key.Z in self.walk_keys:
			self.turn_camera(-1.8 * delta_time, 0.0)
		if pyglet.window.key.X in self.walk_keys:
			self.turn_camera(1.8 * delta_time, 0.0)
		if not self.fullscreen:
			if pyglet.window.key.LEFT in self.walk_keys:
				self.turn_camera(1.8 * delta_time, 0.0)
			if pyglet.window.key.RIGHT in self.walk_keys:
				self.turn_camera(-1.8 * delta_time, 0.0)
		height_change = 0.0
		if pyglet.window.key.H in self.walk_keys:
			height_change += 1.0
		if pyglet.window.key.L in self.walk_keys:
			height_change -= 1.0
		if height_change:
			step = max(float(self.render_scene._orbit_extent) * 0.12, 0.05)
			old_height = self.walk_height
			self.walk_height = max(0.1, old_height + height_change * step * delta_time)
			delta_height = self.walk_height - old_height
			if delta_height:
				pose = self.render_scene.get_pose(self.render_scene._orbit_camera_node)
				pose[1, 3] += delta_height
				self.render_scene.set_pose(self.render_scene._orbit_camera_node, pose)
				self.update_walk_caption()

	def on_mouse_scroll(self, _x: int, _y: int, _scroll_x: int, scroll_y: int) -> None:
		if self.walk_mode:
			return
		self.distance *= 0.88 ** scroll_y
		extent = self.render_scene._orbit_extent
		self.distance = float(np.clip(self.distance, extent * 0.03, extent * 20.0))
		self.update_camera()

	def pan_orbit(self, dx: int, dy: int) -> None:
		forward = np.array(
			[
				math.cos(self.pitch) * math.sin(self.yaw),
				math.sin(self.pitch),
				math.cos(self.pitch) * math.cos(self.yaw),
			],
			dtype=np.float32,
		)
		right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
		right /= np.linalg.norm(right)
		up = np.cross(right, forward)
		pan_scale = self.distance * 0.0015
		self.orbit_target += (dx * right - dy * up) * pan_scale
		self.update_camera()

	def on_close(self) -> None:
		self.renderer.delete()
		super().on_close()


def main() -> None:
	render_scene, camera = load_scene(ASSET_PATH)
	window = ModelWindow(render_scene, camera)
	web_server = ViewerWebServer(
		CAPTURES_PATH,
		ARTIFACTS_PATH,
		window.web_state,
		window.enqueue_web_command,
		window.enqueue_web_command,
		WEB_HOST,
		WEB_PORT,
	)
	window.web_server = web_server
	web_server.start()
	LOGGER.info("Viewer control panel listening at http://%s:%d", WEB_HOST, WEB_PORT)
	try:
		pyglet.app.run()
	finally:
		web_server.stop()


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
	main()
