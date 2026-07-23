from __future__ import annotations

import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2

from visionsort.core.config import relative_to_root
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
        points = [
            (float(point[0]), float(point[1]))
            for point in mask
            if len(point) >= 2
            and math.isfinite(float(point[0]))
            and math.isfinite(float(point[1]))
        ]
        area = abs(
            sum(
                points[index - 1][0] * points[index][1]
                - points[index][0] * points[index - 1][1]
                for index in range(len(points))
            )
        ) / 2.0
        if len(points) < 3 or area <= 1.0:
            return None
        coordinates = " ".join(
            f"{x / width:.6f} {y / height:.6f}" for x, y in points
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
                        "timestamp_global": item.get("timestamp_global"),
                        "session_id": item.get("session_id"),
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
    def export(
        self,
        items: list[dict[str, Any]],
        output_path: Path,
        *,
        crops_dir: Path | None = None,
    ) -> dict[str, Any]:
        crops_dir = crops_dir or output_path.parent / "reid_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        ambiguous = 0
        missing_identity = 0
        for item in items:
            metadata = json.loads(item.get("metadata_json") or "{}")
            image_path = Path(str(item["image_path"]))
            if not image_path.is_absolute():
                image_path = ROOT_DIR / image_path
            image = cv2.imread(str(image_path))
            for observation_index, observation in enumerate(
                metadata.get("observations", [])
            ):
                if observation.get("class_name") != "parcel":
                    continue
                if observation.get("match_result") == "AMBIGUOUS":
                    ambiguous += 1
                    continue
                global_id = observation.get("global_parcel_id")
                if not global_id:
                    missing_identity += 1
                    continue
                bbox = observation.get("bbox") or []
                if image is None or len(bbox) != 4:
                    missing_identity += 1
                    continue
                height, width = image.shape[:2]
                x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    missing_identity += 1
                    continue
                crop_path = crops_dir / f"{item['id']}_{observation_index}.jpg"
                if not cv2.imwrite(str(crop_path), image[y1:y2, x1:x2]):
                    missing_identity += 1
                    continue
                rows.append(
                    {
                        "dataset_item_id": item["id"],
                        "global_parcel_id": global_id,
                        "camera_id": item.get("source_id"),
                        "session_id": item.get("session_id"),
                        "split": item.get("split"),
                        "local_track_id": observation.get("local_track_id"),
                        "timestamp_global": item.get("timestamp_global"),
                        "bbox": bbox,
                        "crop_path": relative_to_root(crop_path),
                        "embedding": observation.get("appearance_hint"),
                    }
                )
        output_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        positive_pairs: list[dict[str, Any]] = []
        negative_pairs: list[dict[str, Any]] = []
        by_session: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_session.setdefault(str(row["session_id"]), []).append(row)
        for session_id, session_rows in by_session.items():
            for left, right in combinations(session_rows, 2):
                if left["split"] != right["split"]:
                    continue
                pair = {
                    "session_id": session_id,
                    "left_crop": left["crop_path"],
                    "right_crop": right["crop_path"],
                    "left_camera": left["camera_id"],
                    "right_camera": right["camera_id"],
                }
                if (
                    left["global_parcel_id"] == right["global_parcel_id"]
                    and left["camera_id"] != right["camera_id"]
                ):
                    positive_pairs.append({**pair, "label": 1})
                elif left["global_parcel_id"] != right["global_parcel_id"]:
                    negative_pairs.append({**pair, "label": 0})
        pairs_path = output_path.with_name("reid_pairs.jsonl")
        pairs = [*positive_pairs, *negative_pairs]
        pairs_path.write_text(
            "\n".join(json.dumps(pair) for pair in pairs)
            + ("\n" if pairs else ""),
            encoding="utf-8",
        )
        if ambiguous:
            status = "NEEDS_REVIEW"
        elif not positive_pairs or not negative_pairs or missing_identity:
            status = "NOT_READY"
        else:
            status = "READY"
        return {
            "status": status,
            "crops": len(rows),
            "positive_pairs": len(positive_pairs),
            "negative_pairs": len(negative_pairs),
            "ambiguous_observations": ambiguous,
            "missing_global_identity": missing_identity,
            "manifest_path": relative_to_root(output_path),
            "pairs_path": relative_to_root(pairs_path),
        }


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
