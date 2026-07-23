from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2

from visionsort.core.paths import ROOT_DIR
from visionsort.database.db import VisionSortDB


class BaseAutoAnnotator:
    task = "detection"

    def __init__(
        self,
        db: VisionSortDB,
        model_row: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ):
        self.db = db
        self.model_row = model_row
        self.model_id = str(model_row["id"])
        self.backend = str(model_row["backend"])
        self.checkpoint = str(model_row.get("weights_path") or "")
        self.config = dict(config or {})
        self.model = None
        self.demo_by_source: dict[str, dict[int, list[dict[str, Any]]]] = {}
        if self.backend == "demo":
            self._load_demo_sidecars()
        elif self.backend == "ultralytics":
            os.environ.setdefault(
                "YOLO_CONFIG_DIR", str(ROOT_DIR / "data" / "ultralytics")
            )
            try:
                from ultralytics import YOLO
            except Exception as exc:  # pragma: no cover - optional environment
                raise RuntimeError("Ultralytics indisponible.") from exc
            self.model = YOLO(self.checkpoint)
        else:
            raise RuntimeError(f"Backend d'annotation non supporté: {self.backend}")

    def _load_demo_sidecars(self) -> None:
        for source in self.db.fetch_all("SELECT id, uri FROM sources"):
            sidecar = Path(str(source["uri"])).with_suffix(".jsonl")
            frames: dict[int, list[dict[str, Any]]] = {}
            if sidecar.exists():
                for line in sidecar.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    frames.setdefault(int(item["frame_index"]), []).append(item)
            self.demo_by_source[str(source["id"])] = frames

    def predict(
        self, *, source_id: str, frame_index: int, image_path: Path
    ) -> list[dict[str, Any]]:
        if self.backend == "demo":
            return [
                dict(item)
                for item in self.demo_by_source.get(source_id, {}).get(frame_index, [])
            ]
        results = self.model.predict(str(image_path), verbose=False)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = getattr(result, "names", {}) or {}
        masks = getattr(result, "masks", None)
        keypoints = getattr(result, "keypoints", None)
        detections: list[dict[str, Any]] = []
        for index in range(len(boxes)):
            cls_index = int(boxes.cls[index].item())
            item: dict[str, Any] = {
                "class_name": str(names.get(cls_index, cls_index)),
                "confidence": float(boxes.conf[index].item()),
                "bbox": [float(value) for value in boxes.xyxy[index].tolist()],
                "model_id": self.model_id,
                "attributes": {},
            }
            if (
                masks is not None
                and getattr(masks, "xy", None) is not None
                and index < len(masks.xy)
            ):
                item["mask"] = [
                    [float(value) for value in point]
                    for point in masks.xy[index].tolist()
                ]
            if (
                keypoints is not None
                and getattr(keypoints, "data", None) is not None
                and index < len(keypoints.data)
            ):
                item["keypoints"] = [
                    [float(value) for value in point]
                    for point in keypoints.data[index].tolist()
                ]
            detections.append(item)
        return detections

    def write_labels(
        self,
        *,
        image_path: Path,
        label_path: Path,
        detections: list[dict[str, Any]],
        names: dict[str, int],
    ) -> int:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Image illisible: {image_path}")
        height, width = image.shape[:2]
        lines = [
            line
            for item in detections
            if (line := self._label_line(item, names, width, height)) is not None
        ]
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
        )
        return len(lines)

    def _label_line(
        self,
        detection: dict[str, Any],
        names: dict[str, int],
        width: int,
        height: int,
    ) -> str | None:
        class_name = str(detection["class_name"])
        if class_name not in names:
            return None
        x1, y1, x2, y2 = [float(value) for value in detection["bbox"]]
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        box_width = abs(x2 - x1) / width
        box_height = abs(y2 - y1) / height
        return (
            f"{names[class_name]} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    def provenance(
        self,
        *,
        session_id: str,
        camera_id: str,
        timestamp_global: float,
        quality_score: float,
    ) -> dict[str, Any]:
        return {
            "task": self.task,
            "model_id": self.model_id,
            "model_version": self.checkpoint or self.model_id,
            "checkpoint": self.checkpoint,
            "configuration": self.config,
            "session_id": session_id,
            "camera_id": camera_id,
            "timestamp_global": timestamp_global,
            "quality_score": quality_score,
        }


class DetectionAutoAnnotator(BaseAutoAnnotator):
    task = "detection"


class SegmentationAutoAnnotator(BaseAutoAnnotator):
    task = "segmentation"

    def _label_line(
        self,
        detection: dict[str, Any],
        names: dict[str, int],
        width: int,
        height: int,
    ) -> str | None:
        class_name = str(detection["class_name"])
        mask = detection.get("mask") or []
        if class_name not in names or len(mask) < 3:
            return None
        coordinates = " ".join(
            f"{float(x) / width:.6f} {float(y) / height:.6f}" for x, y in mask
        )
        return f"{names[class_name]} {coordinates}"


class PoseAutoAnnotator(BaseAutoAnnotator):
    task = "pose"

    def _label_line(
        self,
        detection: dict[str, Any],
        names: dict[str, int],
        width: int,
        height: int,
    ) -> str | None:
        base = super()._label_line(detection, names, width, height)
        keypoints = detection.get("keypoints") or []
        if base is None or not keypoints:
            return None
        values: list[str] = []
        for point in keypoints:
            x, y = float(point[0]), float(point[1])
            confidence = float(point[2]) if len(point) > 2 else 1.0
            visibility = 2 if confidence > 0.5 else 1 if confidence > 0.0 else 0
            values.extend((f"{x / width:.6f}", f"{y / height:.6f}", str(visibility)))
        return f"{base} {' '.join(values)}"


class LocalTrackingExporter:
    def export(self, items: list[dict[str, Any]], output_path: Path) -> int:
        rows: list[dict[str, Any]] = []
        for item in items:
            metadata = json.loads(item.get("metadata_json") or "{}")
            for observation in metadata.get("observations", []):
                if observation.get("local_track_id") is None:
                    continue
                rows.append(
                    {
                        "dataset_item_id": item["id"],
                        "camera_id": item.get("source_id"),
                        "frame_index": item.get("frame_index"),
                        "track_identity": [
                            item.get("source_id"),
                            observation["local_track_id"],
                        ],
                        "bbox": observation.get("bbox"),
                        "class_name": observation.get("class_name"),
                    }
                )
        output_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        return len(rows)


class MultiCameraReIDExporter:
    def export(self, items: list[dict[str, Any]], output_path: Path) -> int:
        rows: list[dict[str, Any]] = []
        for item in items:
            metadata = json.loads(item.get("metadata_json") or "{}")
            for observation in metadata.get("observations", []):
                global_id = observation.get("global_parcel_id")
                if not global_id:
                    continue
                rows.append(
                    {
                        "dataset_item_id": item["id"],
                        "global_parcel_id": global_id,
                        "camera_id": item.get("source_id"),
                        "local_track_id": observation.get("local_track_id"),
                        "timestamp_global": item.get("timestamp_global"),
                        "embedding": observation.get("appearance_hint"),
                    }
                )
        output_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        return len(rows)


def build_auto_annotator(
    db: VisionSortDB,
    model_row: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> BaseAutoAnnotator:
    task = str(model_row.get("task") or "detection")
    annotator_class = {
        "detection": DetectionAutoAnnotator,
        "segmentation": SegmentationAutoAnnotator,
        "pose": PoseAutoAnnotator,
    }.get(task)
    if annotator_class is None:
        raise RuntimeError(f"Tâche d'annotation non supportée: {task}")
    return annotator_class(db, model_row, config=config)
