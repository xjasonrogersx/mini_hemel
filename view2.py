#!/usr/bin/env python3
"""Display a GLTF/GLB model with trimesh's interactive viewer."""

from __future__ import annotations

import argparse
from datetime import datetime
import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import trimesh
from trimesh.viewer import SceneViewer
from trimesh.transformations import transform_points


LOGGER = logging.getLogger(__name__)


class CarSegmentationViewer(SceneViewer):
    """Trimesh viewer with on-demand car segmentation."""

    def __init__(
        self,
        scene: trimesh.Scene,
        model_path: str,
        segformer_model_path: str,
        visdrone_model_path: str,
        confidence: float,
        screenshot_dir: Path,
    ) -> None:
        self.segmentation_model_path = model_path
        self.segmentation_confidence = confidence
        self.screenshot_dir = screenshot_dir
        self.segmentation_model: Any = None
        self.segmentation_processor: Any = None
        self.segmentation_device: Any = None
        self.segformer_model_path = segformer_model_path
        self.segformer_model: Any = None
        self.segformer_processor: Any = None
        self.segformer_device: Any = None
        self.visdrone_model_path = visdrone_model_path
        self.visdrone_model: Any = None
        self.highlight_geometry: list[str] = []
        self.detected_car_triangles: list[np.ndarray] = []
        self.detected_face_records: list[tuple[str, str, np.ndarray]] = []
        self._top_down = False
        self._normal_camera_transform = scene.camera_transform.copy()
        super().__init__(
            scene,
            background=(0, 0, 0, 255),
            caption=(
                "Trimesh SceneViewer "
                "(G: top-down, Q: Mask2Former, W: SegFormer, Y: VisDrone YOLO)"
            ),
        )

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if symbol == self._key("Q"):
            self.segment_cars()
            return
        if symbol == self._key("S"):
            self.save_current_view()
            return
        if symbol == self._key("G"):
            self.toggle_top_down_view()
            return
        if symbol == self._key("W"):
            self.segment_cars_with_segformer()
            return
        if symbol == self._key("Y"):
            self.segment_cars_with_visdrone()
            return
        if symbol == self._key("R"):
            self.flatten_detected_cars()
            return
        super().on_key_press(symbol, modifiers)

    @staticmethod
    def _key(name: str) -> int:
        import pyglet

        return getattr(pyglet.window.key, name)

    def _capture_view(self) -> np.ndarray:
        self.on_draw()
        output = io.BytesIO()
        self.save_image(output)
        output.seek(0)
        with Image.open(output) as image:
            return np.asarray(image.convert("RGB"))

    def save_current_view(self) -> None:
        try:
            image = self._capture_view()
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = self.screenshot_dir / f"view_{timestamp}.png"
            Image.fromarray(image).save(output_path)
            LOGGER.info("Saved current view to %s", output_path)
            self.set_caption(f"Trimesh SceneViewer (saved: {output_path.name})")
        except Exception:
            LOGGER.exception("Failed to save current view")

    def _load_segmentation_model(self) -> Any:
        if self.segmentation_model is None:
            try:
                import torch
                from transformers import (
                    AutoImageProcessor,
                    Mask2FormerForUniversalSegmentation,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Car segmentation requires PyTorch and transformers. "
                    "Install them with: pip install torch transformers"
                ) from exc
            LOGGER.info(
                "Loading Mask2Former segmentation model: %s",
                self.segmentation_model_path,
            )
            self.segmentation_processor = AutoImageProcessor.from_pretrained(
                self.segmentation_model_path
            )
            self.segmentation_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.segmentation_model_path
            )
            self.segmentation_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.segmentation_model.to(self.segmentation_device)
            self.segmentation_model.eval()
            LOGGER.info(
                "Mask2Former segmentation model loaded on %s",
                self.segmentation_device,
            )
        return self.segmentation_model

    def _remove_highlights(self) -> None:
        for geometry_name in self.highlight_geometry:
            if geometry_name in self.scene.geometry:
                self.scene.delete_geometry(geometry_name)
        self.highlight_geometry.clear()
        self.cleanup_geometries()

    def _load_segformer_model(self) -> Any:
        if self.segformer_model is None:
            try:
                import torch
                from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
            except ImportError as exc:
                raise RuntimeError(
                    "SegFormer car segmentation requires PyTorch and transformers. "
                    "Install them with: pip install torch transformers"
                ) from exc
            LOGGER.info("Loading SegFormer model: %s", self.segformer_model_path)
            self.segformer_processor = AutoImageProcessor.from_pretrained(
                self.segformer_model_path
            )
            self.segformer_model = SegformerForSemanticSegmentation.from_pretrained(
                self.segformer_model_path
            )
            self.segformer_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.segformer_model.to(self.segformer_device)
            self.segformer_model.eval()
            LOGGER.info(
                "SegFormer model loaded on %s",
                self.segformer_device,
            )
        return self.segformer_model

    def _segformer_car_mask(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        model = self._load_segformer_model()
        LOGGER.info(
            "Running SegFormer car segmentation (confidence >= %.2f)",
            self.segmentation_confidence,
        )
        import torch

        inputs = self.segformer_processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self.segformer_device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        segmentation = self.segformer_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.shape[:2]]
        )[0]
        if hasattr(segmentation, "cpu"):
            segmentation = segmentation.cpu().numpy()
        segmentation = np.asarray(segmentation)
        labels = getattr(model.config, "id2label", {})
        car_label_ids = {
            int(label_id)
            for label_id, label in labels.items()
            if str(label).lower() == "car"
        }
        mask = np.isin(segmentation, list(car_label_ids))
        car_count = int(mask.any())
        LOGGER.info(
            "SegFormer car segmentation completed: car mask %s (%d pixels)",
            "found" if car_count else "empty",
            int(mask.sum()),
        )
        return mask, car_count

    def _car_mask(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        model = self._load_segmentation_model()
        LOGGER.info(
            "Running Mask2Former car segmentation (confidence >= %.2f)",
            self.segmentation_confidence,
        )
        import torch

        inputs = self.segmentation_processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self.segmentation_device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = model(**inputs)
        result = self.segmentation_processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image.shape[:2]]
        )[0]
        segmentation = result["segmentation"]
        if hasattr(segmentation, "cpu"):
            segmentation = segmentation.cpu().numpy()
        segmentation = np.asarray(segmentation)
        labels = getattr(model.config, "id2label", {})
        car_label_ids = {
            int(label_id)
            for label_id, label in labels.items()
            if str(label).lower() == "car"
        }
        mask = np.zeros(image.shape[:2], dtype=bool)
        car_count = 0
        for segment in result["segments_info"]:
            if (
                segment["label_id"] in car_label_ids
                and segment.get("score", 1.0) >= self.segmentation_confidence
            ):
                mask |= segmentation == segment["id"]
                car_count += 1
        LOGGER.info(
            "Mask2Former car segmentation completed: %d cars detected",
            car_count,
        )
        return mask, car_count

    def toggle_top_down_view(self) -> None:
        if self._top_down:
            self.scene.camera_transform = self._normal_camera_transform.copy()
            self.view["ball"]._pose = self.scene.camera_transform
            self.view["ball"]._n_pose = self.scene.camera_transform
            self._top_down = False
            self.set_caption(
                "Trimesh SceneViewer "
                "(G: top-down, Q: Mask2Former, W: SegFormer, Y: VisDrone YOLO)"
            )
            self._redraw()
            return

        rotation = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        top_down = self.scene.camera.look_at(
            self.scene.bounds,
            rotation=rotation,
            center=self.scene.centroid,
            pad=1.2,
        )
        self.scene.camera_transform = top_down
        self.view["ball"]._pose = top_down
        self.view["ball"]._n_pose = top_down
        self._top_down = True
        self.set_caption(
            "Trimesh SceneViewer "
            "(top-down, Q: Mask2Former, W: SegFormer, Y: VisDrone YOLO)"
        )
        self._redraw()

    def _project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera_points = transform_points(
            points, np.linalg.inv(self.scene.camera_transform)
        )
        z = -camera_points[:, 2]
        visible = z > 0
        pixels = np.zeros((len(points), 2), dtype=np.float64)
        focal = self.scene.camera.focal
        center = self.scene.camera.resolution / 2.0
        pixels[:, 0] = focal[0] * camera_points[:, 0] / np.maximum(z, 1e-8) + center[0]
        pixels[:, 1] = center[1] - (
            focal[1] * camera_points[:, 1] / np.maximum(z, 1e-8)
        )
        return pixels, visible

    def _highlight_faces(self, mask: np.ndarray) -> None:
        width, height = mask.shape[1], mask.shape[0]
        scale = max(float(self.scene.scale), 1.0)
        self.detected_car_triangles.clear()
        self.detected_face_records.clear()
        for node_name in list(self.scene.graph.nodes_geometry):
            transform, geometry_name = self.scene.graph.get(node_name)
            if geometry_name is None or geometry_name not in self.scene.geometry:
                continue
            geometry = self.scene.geometry[geometry_name]
            if not isinstance(geometry, trimesh.Trimesh) or geometry.faces.size == 0:
                continue

            world_vertices = transform_points(geometry.vertices, transform)
            faces = geometry.faces
            triangles = world_vertices[faces]
            centers = triangles.mean(axis=1)
            pixels, visible = self._project(centers)
            x = np.rint(pixels[:, 0]).astype(np.int64)
            y = np.rint(pixels[:, 1]).astype(np.int64)
            inside = (
                visible
                & (x >= 0)
                & (x < width)
                & (y >= 0)
                & (y < height)
            )
            selected = np.zeros(len(faces), dtype=bool)
            selected[inside] = mask[y[inside], x[inside]]
            if not selected.any():
                continue

            selected_triangles = triangles[selected]
            self.detected_car_triangles.append(selected_triangles.copy())
            self.detected_face_records.append(
                (node_name, geometry_name, selected.copy())
            )
            normals = np.cross(
                selected_triangles[:, 1] - selected_triangles[:, 0],
                selected_triangles[:, 2] - selected_triangles[:, 0],
            )
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            normals /= np.maximum(lengths, 1e-8)
            selected_triangles = selected_triangles + normals[:, None, :] * scale * 1e-4
            overlay = trimesh.Trimesh(
                vertices=selected_triangles.reshape(-1, 3),
                faces=np.arange(len(selected_triangles) * 3).reshape(-1, 3),
                process=False,
            )
            overlay.visual.face_colors = np.tile(
                np.array([255, 40, 40, 230], dtype=np.uint8),
                (len(selected_triangles), 1),
            )
            overlay_name = f"__car_highlight_{len(self.highlight_geometry)}"
            self.scene.add_geometry(overlay, geom_name=overlay_name)
            self.highlight_geometry.append(overlay_name)

    def flatten_detected_cars(self) -> None:
        if not self.detected_face_records:
            LOGGER.warning("R pressed, but no detected car triangles are available")
            self.set_caption("No detected car triangles to flatten")
            return

        self._remove_highlights()
        selected_world_triangles: list[np.ndarray] = []
        for node_name, geometry_name, selected in self.detected_face_records:
            if geometry_name not in self.scene.geometry:
                continue
            transform, current_geometry_name = self.scene.graph.get(node_name)
            geometry = self.scene.geometry[current_geometry_name]
            if isinstance(geometry, trimesh.Trimesh):
                world_vertices = transform_points(geometry.vertices, transform)
                selected_world_triangles.append(
                    world_vertices[geometry.faces[selected]].copy()
                )
        if not selected_world_triangles:
            LOGGER.warning("Detected car records contain no valid mesh faces")
            self.set_caption("No valid car triangles to flatten")
            return

        road_y = min(
            float(triangles[:, :, 1].min())
            for triangles in selected_world_triangles
        )
        flattened_count = 0
        for record_index, (
            node_name,
            geometry_name,
            selected,
        ) in enumerate(self.detected_face_records):
            if geometry_name not in self.scene.geometry:
                continue
            transform, current_geometry_name = self.scene.graph.get(node_name)
            geometry = self.scene.geometry[current_geometry_name]
            if not isinstance(geometry, trimesh.Trimesh):
                continue

            unselected_indices = np.flatnonzero(~selected)
            if len(unselected_indices):
                remainder = geometry.submesh(
                    [unselected_indices], append=True, repair=False
                )
                remainder_name = f"__car_remainder_{record_index}"
                remainder_node = f"__car_remainder_node_{record_index}"
                self.scene.add_geometry(
                    remainder,
                    node_name=remainder_node,
                    geom_name=remainder_name,
                    transform=transform,
                )
                self.highlight_geometry.append(remainder_name)
            world_vertices = transform_points(geometry.vertices, transform)
            selected_triangles = world_vertices[geometry.faces[selected]].copy()
            selected_triangles = world_vertices[geometry.faces[selected]].copy()
            selected_triangles[:, :, 1] = road_y
            flattened = trimesh.Trimesh(
                vertices=selected_triangles.reshape(-1, 3),
                faces=np.arange(len(selected_triangles) * 3).reshape(-1, 3),
                process=False,
            )
            flattened.visual.face_colors = np.tile(
                np.array([128, 128, 128, 255], dtype=np.uint8),
                (len(selected_triangles), 1),
            )
            flattened_name = f"__flattened_car_geometry_{record_index}"
            self.scene.add_geometry(flattened, geom_name=flattened_name)
            self.highlight_geometry.append(flattened_name)
            self.hide_geometry(node_name)
            flattened_count += len(selected_triangles)

        self._update_vertex_list()
        self._redraw()
        LOGGER.info(
            "Replaced %d detected car triangles with grey geometry flattened "
            "to shared road level Y=%.6f",
            flattened_count,
            road_y,
        )
        self.set_caption(f"Flattened grey cars: {flattened_count} triangles")

    def segment_cars(self) -> None:
        try:
            image = self._capture_view()
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = self.screenshot_dir / f"view_{timestamp}.png"
            Image.fromarray(image).save(output_path)
            LOGGER.info("Saved Q capture to %s", output_path)
            mask, car_count = self._car_mask(image)
            self._remove_highlights()
            self._highlight_faces(mask)
            self._update_vertex_list()
            self._redraw()
            self.set_caption(f"Trimesh SceneViewer (cars detected: {car_count})")
        except Exception as exc:
            self.set_caption(f"Car segmentation failed: {exc}")
            LOGGER.exception("Car segmentation failed")

    def segment_cars_with_segformer(self) -> None:
        try:
            image = self._capture_view()
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = self.screenshot_dir / f"view_segformer_{timestamp}.png"
            Image.fromarray(image).save(output_path)
            LOGGER.info("Saved W capture to %s", output_path)
            mask, car_count = self._segformer_car_mask(image)
            self._remove_highlights()
            self._highlight_faces(mask)
            self._update_vertex_list()
            self._redraw()
            self.set_caption(f"Trimesh SceneViewer (SegFormer car mask: {car_count})")
        except Exception:
            self.set_caption("SegFormer car segmentation failed")
            LOGGER.exception("SegFormer car segmentation failed")

    def _load_visdrone_model(self) -> Any:
        if self.visdrone_model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "VisDrone YOLO segmentation requires ultralytics. "
                    "Install it with: pip install ultralytics"
                ) from exc
            LOGGER.info(
                "Loading VisDrone YOLOv11-seg model: %s",
                self.visdrone_model_path,
            )
            self.visdrone_model = YOLO(self.visdrone_model_path)
            LOGGER.info(
                "VisDrone YOLO model loaded (task=%s)",
                getattr(self.visdrone_model, "task", "unknown"),
            )
        return self.visdrone_model

    def _visdrone_car_mask(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        model = self._load_visdrone_model()
        names = model.names
        car_ids = [
            int(class_id)
            for class_id, label in names.items()
            if str(label).strip().lower() == "car"
        ]
        if not car_ids:
            raise RuntimeError(
                "The VisDrone YOLO model does not contain a 'car' class. "
                f"Available classes: {names}"
            )
        LOGGER.info(
            "Running VisDrone YOLOv11-seg car segmentation (confidence >= %.2f)",
            self.segmentation_confidence,
        )
        results = model.predict(
            source=image,
            classes=car_ids,
            conf=self.segmentation_confidence,
            verbose=False,
        )
        if not results:
            LOGGER.info("VisDrone YOLOv11-seg completed: 0 cars detected")
            return np.zeros(image.shape[:2], dtype=bool), 0
        detections = results[0].boxes
        car_count = len(detections) if detections is not None else 0
        if results[0].masks is not None:
            masks = results[0].masks.data
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            mask = np.asarray(masks, dtype=bool).any(axis=0)
            if mask.shape != image.shape[:2]:
                mask = np.asarray(
                    Image.fromarray(mask).resize(
                        (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
                    ),
                    dtype=bool,
                )
        else:
            LOGGER.warning(
                "VisDrone checkpoint is detection-only; using car bounding boxes "
                "as highlight masks"
            )
            mask = np.zeros(image.shape[:2], dtype=bool)
            boxes = detections.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in boxes:
                left = max(0, int(np.floor(x1)))
                top = max(0, int(np.floor(y1)))
                right = min(image.shape[1], int(np.ceil(x2)))
                bottom = min(image.shape[0], int(np.ceil(y2)))
                if right > left and bottom > top:
                    mask[top:bottom, left:right] = True
        LOGGER.info(
            "VisDrone YOLO car detection completed: %d cars detected",
            car_count,
        )
        return mask, car_count

    def segment_cars_with_visdrone(self) -> None:
        try:
            image = self._capture_view()
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = self.screenshot_dir / f"view_visdrone_{timestamp}.png"
            Image.fromarray(image).save(output_path)
            LOGGER.info("Saved Y capture to %s", output_path)
            mask, car_count = self._visdrone_car_mask(image)
            self._remove_highlights()
            self._highlight_faces(mask)
            self._update_vertex_list()
            self._redraw()
            self.set_caption(
                f"Trimesh SceneViewer (VisDrone cars detected: {car_count})"
            )
        except Exception:
            self.set_caption("VisDrone YOLO car segmentation failed")
            LOGGER.exception("VisDrone YOLO car segmentation failed")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("merged.gltf"),
        help="GLTF or GLB file to display",
    )
    parser.add_argument(
        "--segmentation-model",
        default="facebook/mask2former-swin-small-coco-panoptic",
        help=(
            "Mask2Former checkpoint "
            "(default: facebook/mask2former-swin-small-coco-panoptic)"
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="minimum car detection confidence (default: 0.25)",
    )
    parser.add_argument(
        "--segformer-model",
        default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        help=(
            "SegFormer checkpoint "
            "(default: nvidia/segformer-b2-finetuned-cityscapes-1024-1024)"
        ),
    )
    parser.add_argument(
        "--visdrone-model",
        default=Path(__file__).with_name("best.pt"),
        help=(
            "YOLOv11-seg VisDrone checkpoint "
            "(default: best.pt beside view2.py)"
        ),
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=Path("captures"),
        help="directory for S and Q screenshots (default: captures)",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"model file does not exist: {args.model}")

    loaded = trimesh.load(args.model, force="scene", process=False)
    if not isinstance(loaded, trimesh.Scene):
        loaded = trimesh.Scene(loaded)
    CarSegmentationViewer(
        loaded,
        model_path=args.segmentation_model,
        segformer_model_path=args.segformer_model,
        visdrone_model_path=args.visdrone_model,
        confidence=args.confidence,
        screenshot_dir=args.screenshot_dir,
    )


if __name__ == "__main__":
    main()
