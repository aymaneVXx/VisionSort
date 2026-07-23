from __future__ import annotations

import json
import queue
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from visionsort.core.config import AppConfig
from visionsort.core.types import Observation
from visionsort.database.db import VisionSortDB

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - dépend de l'environnement
    YOLO = None


class DemoDetectionBackend:
    def __init__(self):
        self.sidecars: dict[str, dict[int, list[dict[str, Any]]]] = {}

    def register_sidecar(self, camera_id: str, source_uri: str) -> None:
        sidecar = Path(source_uri).with_suffix(".jsonl")
        data: dict[int, list[dict[str, Any]]] = {}
        if sidecar.exists():
            for line in sidecar.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                data.setdefault(int(item["frame_index"]), []).append(item)
        self.sidecars[camera_id] = data

    def predict(self, camera_id: str, frame_index: int, image: np.ndarray) -> list[Observation]:
        rows = self.sidecars.get(camera_id, {}).get(frame_index, [])
        return [
            Observation(
                class_name=row["class_name"],
                confidence=float(row.get("confidence", 1.0)),
                bbox=tuple(row["bbox"]),
                mask=row.get("mask"),
                keypoints=row.get("keypoints"),
                embedding=row.get("embedding"),
                attributes=row.get("attributes", {}),
            )
            for row in rows
        ]


class UltralyticsBackend:
    def __init__(self, weights_path: str, device: str = "auto"):
        if YOLO is None:
            raise RuntimeError("Ultralytics n'est pas disponible dans cet environnement.")
        self.model = YOLO(weights_path)
        self.device = device

    def predict(self, camera_id: str, frame_index: int, image: np.ndarray) -> list[Observation]:
        _ = camera_id, frame_index
        results = self.model.predict(image, verbose=False, device=self.device)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}
        keypoints = getattr(result, "keypoints", None)
        masks = getattr(result, "masks", None)
        output: list[Observation] = []
        if boxes is None:
            return output
        for idx in range(len(boxes)):
            cls_idx = int(boxes.cls[idx].item())
            bbox_xyxy = boxes.xyxy[idx].tolist()
            cls_name = names.get(cls_idx, str(cls_idx))
            kp_value = None
            if keypoints is not None and getattr(keypoints, "data", None) is not None and idx < len(keypoints.data):
                kp_value = [tuple(map(float, item)) for item in keypoints.data[idx].tolist()]
            mask_value = None
            if masks is not None and getattr(masks, "xy", None) is not None and idx < len(masks.xy):
                mask_value = [[float(v) for v in point] for point in masks.xy[idx].tolist()]
            output.append(
                Observation(
                    class_name=str(cls_name),
                    confidence=float(boxes.conf[idx].item()),
                    bbox=tuple(float(v) for v in bbox_xyxy),
                    mask=mask_value,
                    keypoints=kp_value,
                )
            )
        return output


class SharedInferenceEngine:
    def __init__(self, db: VisionSortDB, config: AppConfig):
        self.db = db
        self.config = config
        self.backend: DemoDetectionBackend | UltralyticsBackend | None = None
        self.backend_info: dict[str, Any] = {}
        self.model_id: str | None = None
        self.model_version: str | None = None

    def load_model(self, model_id: str, source_map: dict[str, dict[str, Any]]) -> None:
        row = self.db.fetch_one("SELECT * FROM model_registry WHERE id = ?", (model_id,))
        if row is None:
            raise RuntimeError(f"Modèle introuvable: {model_id}")
        backend = row["backend"]
        if backend == "demo":
            if not self.config.demo_mode:
                raise RuntimeError("Le backend de démonstration exige DEMO_MODE=1.")
            demo = DemoDetectionBackend()
            for camera_id, source in source_map.items():
                demo.register_sidecar(camera_id, source["uri"])
            self.backend = demo
        elif backend == "ultralytics":
            self.backend = UltralyticsBackend(
                weights_path=row["weights_path"],
                device=self.config.get("gpu", "device", default="auto"),
            )
        else:
            raise RuntimeError(f"Backend de modèle non supporté: {backend}")
        self.model_id = model_id
        self.model_version = row["weights_path"] or model_id
        self.backend_info = {"model_id": model_id, "backend": backend, "model_version": self.model_version, "loaded_at": time.time()}

    def predict(self, camera_id: str, frame_index: int, image: np.ndarray) -> list[Observation]:
        if self.backend is None:
            return []
        output = self.backend.predict(camera_id, frame_index, image)
        for obs in output:
            obs.model_id = self.model_id
            obs.model_version = self.model_version
        return output


def inference_worker_loop(
    request_queue,
    result_queue,
    stop_event,
    db_path: str,
    config_values: dict[str, Any],
) -> None:
    db = VisionSortDB(Path(db_path))
    config = AppConfig(values=config_values)
    engine = SharedInferenceEngine(db, config)
    loaded_model_id: str | None = None
    source_map: dict[str, dict[str, Any]] = {}
    while not stop_event.is_set():
        try:
            message = request_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        kind = message.get("kind")
        if kind == "SYNC_SOURCES":
            source_map = message["source_map"]
        elif kind == "LOAD_MODEL":
            model_id = message["model_id"]
            if model_id != loaded_model_id:
                engine.load_model(model_id, source_map)
                loaded_model_id = model_id
            result_queue.put({"kind": "MODEL_READY", "model_id": model_id, "backend_info": engine.backend_info})
        elif kind == "INFER":
            try:
                result_queue.put(
                    {
                        "kind": "INFER_RESULT",
                        "request_id": message["request_id"],
                        "session_id": message["session_id"],
                        "source_id": message["source_id"],
                        "camera_id": message["camera_id"],
                        "camera_role": message.get("camera_role"),
                        "stream_epoch": message["stream_epoch"],
                        "frame_index": message["frame_index"],
                        "timestamp_local": message["timestamp_local"],
                        "timestamp_global": message["timestamp_global"],
                        "created_at": message["created_at"],
                        "expires_at": message["expires_at"],
                        "observations": [
                            asdict(obs)
                            for obs in engine.predict(
                                message["camera_id"],
                                message["frame_index"],
                                message["image"],
                            )
                        ],
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime
                result_queue.put(
                    {
                        "kind": "INFER_ERROR",
                        "request_id": message.get("request_id"),
                        "session_id": message.get("session_id"),
                        "source_id": message.get("source_id"),
                        "camera_id": message["camera_id"],
                        "stream_epoch": message.get("stream_epoch"),
                        "frame_index": message["frame_index"],
                        "created_at": message.get("created_at"),
                        "expires_at": message.get("expires_at"),
                        "error": str(exc),
                    }
                )
