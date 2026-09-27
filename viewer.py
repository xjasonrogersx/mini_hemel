"""Display merged.gltf with orbit and Quake-style walk controls."""

from pathlib import Path
from datetime import datetime
import base64
import io
import json
import math
import os
import threading
import time
import uuid

import numpy as np
import pyglet
from pyglet import gl
import pyrender
import trimesh
from PIL import Image


ASSET_PATH = Path(__file__).with_name("merged.gltf")
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
CAPTURES_PATH = Path(__file__).with_name("captures")
CAMERA_NEAR_RATIO = 0.005
CAMERA_FAR_RATIO = 20.0
CONTROLNET_QUEUE = "stable-diffusion-controlnet"
RABBITMQ_URL = os.getenv(
	"RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F"
)


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
		self.update_camera()
		pyglet.clock.schedule_interval(self.render_frame, 1.0 / 60.0)
		pyglet.clock.schedule_interval(self.update_walk, 1.0 / 60.0)

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
		if symbol == pyglet.window.key.S:
			self.save_current_view()
			return
		if symbol == pyglet.window.key.Q:
			self.generate_controlnet_view()
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
			nearest = depth[valid].min()
			farthest = depth[valid].max()
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
		return view_path, depth_path

	def _request_controlnet(self, view_path: Path, depth_path: Path) -> None:
		try:
			import pika
			def encode(path: Path) -> str:
				return base64.b64encode(path.read_bytes()).decode("ascii")
			request = {
				"image_base64": encode(view_path),
				"control_image_base64": encode(depth_path),
				"prompt": "photorealistic textured reconstruction, natural materials, realistic lighting, preserve exact geometry and composition",
				"negative_prompt": "changed camera angle, changed geometry, warped structures, extra objects, text, watermark, blur",
				"steps": 30,
				"strength": 0.35,
				"guidance_scale": 7.5,
				"controlnet_conditioning_scale": 1.0,
				"seed": 0,
			}
			correlation_id = str(uuid.uuid4())
			connection = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL))
			channel = connection.channel()
			channel.queue_declare(queue=CONTROLNET_QUEUE, durable=True)
			reply_queue = channel.queue_declare(queue="", exclusive=True).method.queue
			response: bytes | None = None
			def on_response(_channel, _method, properties, body: bytes) -> None:
				nonlocal response
				if properties.correlation_id == correlation_id:
					response = body
			consumer_tag = channel.basic_consume(queue=reply_queue, on_message_callback=on_response, auto_ack=True)
			channel.basic_publish(
				exchange="", routing_key=CONTROLNET_QUEUE, body=json.dumps(request).encode("utf-8"),
				properties=pika.BasicProperties(content_type="application/json", correlation_id=correlation_id, reply_to=reply_queue),
			)
			deadline = time.monotonic() + 900.0
			while response is None:
				if time.monotonic() >= deadline:
					raise TimeoutError("ControlNet request timed out")
				connection.process_data_events(time_limit=1.0)
			channel.basic_cancel(consumer_tag)
			connection.close()
			result = json.loads(response.decode("utf-8"))
			if not result.get("ok"):
				raise RuntimeError(result.get("error", "ControlNet worker failed"))
			output_path = CAPTURES_PATH / f"view_controlnet_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
			output_path.write_bytes(base64.b64decode(result["image_base64"]))
			self.set_caption(f"ControlNet render saved: {output_path.name}")
		except Exception as exc:
			self.set_caption(f"ControlNet request failed: {exc}")

	def on_key_release(self, symbol: int, _modifiers: int) -> None:
		self.walk_keys.discard(symbol)

	def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
		if button == pyglet.window.mouse.LEFT:
			if self.walk_mode:
				self.walk_look_drag = not self.fullscreen
				return
			now = time.monotonic()
			previous = self.last_click
			self.last_click = (now, x, y)
			if (
				not self.walk_mode
				and previous is not None
				and now - previous[0] <= 0.35
				and (x - previous[1]) ** 2 + (y - previous[2]) ** 2 <= 64
			):
				self.enter_walk_mode(x, y)
				return
			self.orbit_button = "orbit" if modifiers & pyglet.window.key.MOD_SHIFT else "pan"
		elif button == pyglet.window.mouse.MIDDLE:
			if self.walk_mode:
				return
			self.orbit_button = "orbit"
		elif button == pyglet.window.mouse.RIGHT:
			self.orbit_button = "zoom"

	def on_mouse_release(self, _x: int, _y: int, button: int, _modifiers: int) -> None:
		if button == pyglet.window.mouse.MIDDLE and self.walk_mode:
			return
		if button == pyglet.window.mouse.LEFT:
			self.walk_look_drag = False
		if button in (
			pyglet.window.mouse.LEFT,
			pyglet.window.mouse.MIDDLE,
			pyglet.window.mouse.RIGHT,
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
			self.distance = float(np.clip(self.distance, extent * 0.25, extent * 20.0))
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
			next_position = pose[:3, 3] + movement * float(self.render_scene._orbit_extent) * 0.12 * delta_time
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
		self.distance = float(np.clip(self.distance, extent * 0.25, extent * 20.0))
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
	ModelWindow(render_scene, camera)
	pyglet.app.run()


if __name__ == "__main__":
	main()
