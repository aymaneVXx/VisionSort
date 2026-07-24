from __future__ import annotations

import gc
import hashlib
import json
import queue
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from visionsort.core.config import AppConfig
from visionsort.core.paths import ROOT_DIR
from visionsort.core.types import Observation
from visionsort.database.db import VisionSortDB

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - dépend de l'environnement
    YOLO = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_model_artifact(model_row: dict[str, Any]) -> tuple[Path, str]:
    value = str(model_row.get("weights_path") or "")
    if not value:
        raise RuntimeError("Chemin de poids vide.")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    path = path.resolve()
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"Poids de modèle introuvables: {path}")
    actual_hash = _sha256(path)
    notes = json.loads(model_row.get("notes_json") or "{}")
    metrics = json.loads(model_row.get("metrics_json") or "{}")
    expected_hash = notes.get("artifact_sha256") or metrics.get(
        "artifact_sha256"
    )
    if expected_hash and str(expected_hash) != actual_hash:
        raise RuntimeError(
            f"Hash du modèle invalide: attendu {expected_hash}, obtenu {actual_hash}."
        )
    return path, actual_hash


def release_model_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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
    def __init__(self, weights_path: Path, device: str = "auto"):
        if YOLO is None:
            raise RuntimeError("Ultralytics n'est pas disponible dans cet environnement.")
        self.model = YOLO(str(weights_path))
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

    def load_model(
        self, model_id: str, source_map: dict[str, dict[str, Any]]
    ) -> None:
        started = time.perf_counter()
        previous_model_id = self.model_id
        self.backend = None
        release_model_memory()
        try:
            backend, info = self._build_backend(model_id, source_map)
        except Exception as exc:
            rollback_error = None
            if previous_model_id:
                try:
                    self.backend, previous_info = self._build_backend(
                        previous_model_id, source_map
                    )
                    self.model_id = previous_model_id
                    self.model_version = str(previous_info["model_version"])
                    self.backend_info = {
                        **previous_info,
                        "rollback_after_failed_model_id": model_id,
                        "reload_error": str(exc),
                    }
                except Exception as rollback_exc:  # pragma: no cover
                    rollback_error = str(rollback_exc)
                    self.backend = None
                    self.model_id = None
                    self.model_version = None
            rollback_status = (
                "ok" if previous_model_id and not rollback_error else "indisponible"
            )
            suffix = f" ({rollback_error})" if rollback_error else ""
            raise RuntimeError(
                f"Échec du rechargement {model_id}; "
                f"rollback={rollback_status}{suffix}: {exc}"
            ) from exc
        self.backend = backend
        self.model_id = model_id
        self.model_version = str(info["model_version"])
        self.backend_info = {
            **info,
            "loaded_at": time.time(),
            "reload_duration_seconds": time.perf_counter() - started,
            "previous_model_id": previous_model_id,
        }

    def _build_backend(
        self, model_id: str, source_map: dict[str, dict[str, Any]]
    ) -> tuple[DemoDetectionBackend | UltralyticsBackend, dict[str, Any]]:
        raw_row = self.db.fetch_one(
            "SELECT * FROM model_registry WHERE id = ?", (model_id,)
        )
        if raw_row is None:
            raise RuntimeError(f"Modèle introuvable: {model_id}")
        row = dict(raw_row)
        backend = str(row["backend"])
        artifact_hash = None
        resolved_path = None
        if backend == "demo":
            if not self.config.demo_mode:
                raise RuntimeError(
                    "Le backend de démonstration exige DEMO_MODE=1."
                )
            demo = DemoDetectionBackend()
            for camera_id, source in source_map.items():
                demo.register_sidecar(camera_id, source["uri"])
            built: DemoDetectionBackend | UltralyticsBackend = demo
            if row.get("weights_path"):
                resolved_path, artifact_hash = resolve_model_artifact(row)
        elif backend == "ultralytics":
            resolved_path, artifact_hash = resolve_model_artifact(row)
            built = UltralyticsBackend(
                weights_path=resolved_path,
                device=self.config.get("gpu", "device", default="auto"),
            )
        else:
            raise RuntimeError(f"Backend de modèle non supporté: {backend}")
        return built, {
            "model_id": model_id,
            "task": str(row["task"]),
            "backend": backend,
            "model_version": (
                str(resolved_path) if resolved_path is not None else model_id
            ),
            "resolved_weights_path": (
                str(resolved_path) if resolved_path is not None else None
            ),
            "artifact_sha256": artifact_hash,
        }

    def predict(self, camera_id: str, frame_index: int, image: np.ndarray) -> list[Observation]:
        if self.backend is None:
            return []
        output = self.backend.predict(camera_id, frame_index, image)
        for obs in output:
            obs.model_id = self.model_id
            obs.model_version = self.model_version
        return output

    def sync_sources(
        self, source_map: dict[str, dict[str, Any]]
    ) -> None:
        if isinstance(self.backend, DemoDetectionBackend):
            for camera_id, source in source_map.items():
                self.backend.register_sidecar(camera_id, source["uri"])


def inference_worker_loop(
    request_queue,
    result_queue,
    stop_event,
    db_path: str,
    config_values: dict[str, Any],
) -> None:
    db = VisionSortDB(Path(db_path))
    config = AppConfig(values=config_values)
    engines: dict[str, SharedInferenceEngine] = {}
    load_counts: dict[str, int] = {}
    inference_metrics: dict[str, dict[str, float | int]] = {}
    source_map: dict[str, dict[str, Any]] = {}
    while not stop_event.is_set():
        try:
            message = request_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        kind = message.get("kind")
        if kind == "SYNC_SOURCES":
            source_map = message["source_map"]
            for engine in engines.values():
                engine.sync_sources(source_map)
        elif kind == "LOAD_MODEL":
            model_id = str(message["model_id"])
            started = time.perf_counter()
            try:
                if model_id not in engines or bool(message.get("reload")):
                    previous_engine = engines.get(model_id)
                    replacement = SharedInferenceEngine(db, config)
                    replacement.load_model(model_id, source_map)
                    engines[model_id] = replacement
                    if previous_engine is not None:
                        del previous_engine
                        release_model_memory()
                    load_counts[model_id] = (
                        int(load_counts.get(model_id, 0)) + 1
                    )
                engine = engines[model_id]
                result_queue.put(
                    {
                        "kind": "MODEL_READY",
                        "model_id": model_id,
                        "task": engine.backend_info.get("task"),
                        "backend_info": engine.backend_info,
                        "duration_seconds": time.perf_counter() - started,
                        "load_count": load_counts[model_id],
                        "loaded_model_ids": sorted(engines),
                    }
                )
            except Exception as exc:
                result_queue.put(
                    {
                        "kind": "MODEL_LOAD_FAILED",
                        "model_id": model_id,
                        "task": message.get("task"),
                        "active_model_id": (
                            engines[model_id].model_id
                            if model_id in engines
                            else None
                        ),
                        "error": str(exc),
                        "duration_seconds": time.perf_counter() - started,
                        "rollback_succeeded": model_id in engines,
                        "loaded_model_ids": sorted(engines),
                    }
                )
        elif kind == "UNLOAD_MODEL":
            model_id = str(message["model_id"])
            engines.pop(model_id, None)
            release_model_memory()
            result_queue.put(
                {
                    "kind": "MODEL_UNLOADED",
                    "model_id": model_id,
                    "loaded_model_ids": sorted(engines),
                }
            )
        elif kind == "INFER":
            model_id = str(
                message.get("model_id") or next(iter(engines), "")
            )
            task = str(message.get("task") or "")
            started = time.perf_counter()
            try:
                if model_id not in engines:
                    raise RuntimeError(
                        f"Modèle non chargé pour l'inférence: {model_id}"
                    )
                engine = engines[model_id]
                if not task:
                    task = str(engine.backend_info.get("task") or "")
                observations = engine.predict(
                    message["camera_id"],
                    message["frame_index"],
                    message["image"],
                )
                elapsed = time.perf_counter() - started
                metrics = inference_metrics.setdefault(
                    model_id,
                    {
                        "requests": 0,
                        "errors": 0,
                        "total_duration_seconds": 0.0,
                    },
                )
                metrics["requests"] = int(metrics["requests"]) + 1
                metrics["total_duration_seconds"] = (
                    float(metrics["total_duration_seconds"]) + elapsed
                )
                result_queue.put(
                    {
                        "kind": "INFER_RESULT",
                        "request_id": message["request_id"],
                        "model_id": model_id,
                        "task": task,
                        "pipeline_role": message.get("pipeline_role"),
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
                        "duration_seconds": elapsed,
                        "model_metrics": dict(metrics),
                        "observations": [
                            asdict(obs) for obs in observations
                        ],
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime
                metrics = inference_metrics.setdefault(
                    model_id,
                    {
                        "requests": 0,
                        "errors": 0,
                        "total_duration_seconds": 0.0,
                    },
                )
                metrics["errors"] = int(metrics["errors"]) + 1
                result_queue.put(
                    {
                        "kind": "INFER_ERROR",
                        "request_id": message.get("request_id"),
                        "model_id": model_id,
                        "task": task,
                        "pipeline_role": message.get("pipeline_role"),
                        "session_id": message.get("session_id"),
                        "source_id": message.get("source_id"),
                        "camera_id": message["camera_id"],
                        "stream_epoch": message.get("stream_epoch"),
                        "frame_index": message["frame_index"],
                        "created_at": message.get("created_at"),
                        "expires_at": message.get("expires_at"),
                        "model_metrics": dict(metrics),
                        "error": str(exc),
                    }
                )
