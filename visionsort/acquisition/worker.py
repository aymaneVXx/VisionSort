from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from visionsort.calibration.geometry import WorldGeometry, runtime_calibration_diagnostic
from visionsort.calibration.models import CalibrationProfile
from visionsort.core.config import AppConfig
from visionsort.core.config import relative_to_root
from visionsort.core.enums import SourceStatus
from visionsort.core.paths import OBSERVATIONS_DIR, PREVIEWS_DIR, RECORDINGS_DIR
from visionsort.core.types import Observation
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import ControlRepository
from visionsort.events.engine import ParcelEventEngine
from visionsort.reid.encoder import ParcelReIDEncoder
from visionsort.reid.keyframes import HandoffKeyframeSelector
from visionsort.sources.frame_sources import build_source
from visionsort.tracking.engine import build_tracker


def build_inference_request(
    *,
    frame,
    source_id: str,
    ttl_seconds: float,
    model_id: str | None = None,
    task: str | None = None,
    pipeline_role: str | None = None,
    routing_generation: int = 0,
) -> dict[str, Any]:
    now = time.time()
    return {
        "kind": "INFER",
        "request_id": str(uuid.uuid4()),
        "session_id": frame.session_id,
        "source_id": source_id,
        "model_id": model_id,
        "task": task,
        "pipeline_role": pipeline_role,
        "routing_generation": int(routing_generation),
        "camera_id": frame.camera_id,
        "camera_role": frame.camera_role,
        "stream_epoch": int(frame.stream_epoch),
        "frame_index": int(frame.frame_index),
        "timestamp_local": float(frame.timestamp_local),
        "timestamp_global": float(frame.timestamp_global),
        "created_at": now,
        "expires_at": now + max(0.1, float(ttl_seconds)),
        "image": frame.image,
    }


def resolve_pipeline_model(
    pipeline: dict[str, Any],
    active_runtime_models_by_task,
) -> tuple[str, int]:
    """Resolve a pipeline without querying SQLite from the camera worker."""
    configured_model_id = str(
        pipeline.get("configured_model_id")
        or pipeline.get("model_id")
        or ""
    )
    if not bool(pipeline.get("use_active", True)):
        if not configured_model_id:
            raise RuntimeError(
                f"Modèle fixe absent pour {pipeline.get('pipeline_role')}."
            )
        return configured_model_id, 0
    task = str(pipeline.get("task") or "")
    route = dict(active_runtime_models_by_task.get(task, {}) or {})
    model_id = str(route.get("model_id") or pipeline.get("model_id") or "")
    if not model_id:
        raise RuntimeError(f"Aucun modèle runtime actif pour la tâche {task}.")
    return model_id, int(route.get("generation") or 0)


def _increment_inference_metric(
    inference_result_store,
    source_id: str,
    metric: str,
) -> None:
    key = f"__inference_metrics__:{source_id}"
    current = dict(inference_result_store.get(key, {}))
    current[metric] = int(current.get(metric, 0)) + 1
    inference_result_store[key] = current


def _inference_metrics(inference_result_store, source_id: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in dict(
            inference_result_store.get(
                f"__inference_metrics__:{source_id}", {}
            )
        ).items()
    }


class LatestFrameBuffer:
    """Continuously acquire frames while consumers process only the freshest one."""

    def __init__(
        self,
        source,
        *,
        capacity: int = 3,
        source_type: str = "REPLAY",
        reconnect_delay_s: float = 1.0,
    ):
        self.source = source
        self.capacity = max(1, int(capacity))
        self.source_type = str(source_type).upper()
        self.reconnect_delay_s = max(0.0, float(reconnect_delay_s))
        self._frames: deque = deque(maxlen=self.capacity)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.finished = False
        self.error: Exception | None = None
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.reconnections = 0

    def start(self) -> None:
        self.source.open()
        self._thread = threading.Thread(
            target=self._acquire,
            daemon=True,
            name="visionsort-frame-acquisition",
        )
        self._thread.start()

    def _acquire(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    frame = self.source.read()
                except Exception as exc:
                    if self.source_type != "RTSP":
                        raise
                    self.reconnections += 1
                    self.source.close()
                    if self._stop.wait(self.reconnect_delay_s):
                        break
                    self.source.open()
                    continue
                if frame is None:
                    if self.source_type != "RTSP":
                        break
                    self.reconnections += 1
                    if self._stop.wait(self.reconnect_delay_s):
                        break
                    continue
                with self._condition:
                    self.frames_received += 1
                    if len(self._frames) == self.capacity:
                        self.frames_dropped += 1
                    self._frames.append(frame)
                    self._condition.notify()
        except Exception as exc:  # pragma: no cover - exercised by runtime integration
            self.error = exc
        finally:
            with self._condition:
                self.finished = True
                self._condition.notify_all()

    def take_latest(self, timeout: float = 0.5):
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while not self._frames and not self.finished and self.error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if not self._frames:
                return None
            latest = self._frames.pop()
            if self._frames:
                self.frames_dropped += len(self._frames)
                self._frames.clear()
            return latest

    def mark_processed(self) -> None:
        with self._condition:
            self.frames_processed += 1

    def mark_dropped(self) -> None:
        with self._condition:
            self.frames_dropped += 1

    def metrics(self) -> dict[str, int]:
        with self._condition:
            return {
                "frames_received": self.frames_received,
                "frames_processed": self.frames_processed,
                "frames_dropped": self.frames_dropped,
                "reconnections": self.reconnections,
                "buffered_frames": len(self._frames),
            }

    def stop(self) -> None:
        self._stop.set()
        self.source.close()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


def _annotate_frame(image, observations: list[Observation], camera_id: str, fps: float, status: str):
    output = image.copy()
    cv2.putText(output, f"{camera_id} | {status} | {fps:.1f} FPS", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for obs in observations:
        x1, y1, x2, y2 = [int(v) for v in obs.bbox]
        color = (0, 220, 220) if obs.class_name == "parcel" else (220, 100, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{obs.class_name} {obs.confidence:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return output


class _SegmentRecorder:
    def __init__(
        self,
        *,
        source_id: str,
        session_id: str,
        camera_role: str,
        segment_seconds: int,
        recordings_dir: Path = RECORDINGS_DIR,
    ):
        self.source_id = source_id
        self.session_id = session_id
        self.camera_role = camera_role
        self.segment_seconds = segment_seconds
        self.recordings_dir = Path(recordings_dir)
        self.writer = None
        self.segment_started_at = 0.0
        self.segment_path: Path | None = None
        self.recording_id: str | None = None
        self.stream_epoch: int | None = None
        self.segment_index = -1
        self.segment_fps = 0.0
        self.last_timestamp_global = 0.0
        self.frames: list[dict[str, Any]] = []
        self.frame_count = 0
        self.frames_archived_total = 0
        self.segments_produced = 0
        self.segments_corrupted = 0
        self.bytes_used = 0

    def _open(self, frame, fps: float) -> None:
        self.segment_index += 1
        self.recording_id = str(uuid.uuid4())
        self.stream_epoch = int(frame.stream_epoch)
        self.segment_started_at = float(frame.timestamp_global)
        self.last_timestamp_global = float(frame.timestamp_global)
        self.segment_fps = max(1.0, float(fps))
        self.segment_path = (
            self.recordings_dir
            / self.session_id
            / self.source_id
            / (
                f"segment-{self.segment_index:06d}-"
                f"epoch-{self.stream_epoch}-{self.recording_id[:8]}.mp4"
            )
        )
        self.segment_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.image.shape[:2]
        self.writer = cv2.VideoWriter(
            str(self.segment_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.segment_fps,
            (width, height),
        )
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            raise RuntimeError(
                f"Impossible d'ouvrir le writer d'archive: {self.segment_path}"
            )
        self.frames = []
        self.frame_count = 0

    def write(self, frame, fps: float) -> dict[str, Any] | None:
        now = float(frame.timestamp_global)
        rotate = (
            self.writer is not None
            and (
                (now - self.segment_started_at) >= float(self.segment_seconds)
                or int(frame.stream_epoch) != int(self.stream_epoch or 0)
            )
        )
        finished = self.close() if rotate else None
        if self.writer is None:
            self._open(frame, fps)
        if self.writer is None:
            raise RuntimeError("Writer d'archive indisponible.")
        segment_frame_index = self.frame_count
        self.writer.write(frame.image)
        self.frames.append(
            {
                "camera_role": self.camera_role,
                "stream_epoch": int(frame.stream_epoch),
                "frame_index": int(frame.frame_index),
                "timestamp_local": float(frame.timestamp_local),
                "timestamp_global": float(frame.timestamp_global),
                "segment_frame_index": segment_frame_index,
            }
        )
        self.frame_count += 1
        self.frames_archived_total += 1
        self.last_timestamp_global = now
        return finished

    def close(self, ended_at: float | None = None) -> dict[str, Any] | None:
        if (
            self.writer is None
            or self.segment_path is None
            or self.recording_id is None
        ):
            return None
        self.writer.release()
        path = self.segment_path
        size_bytes = path.stat().st_size if path.exists() else 0
        corrupted = size_bytes <= 0
        if not corrupted:
            capture = cv2.VideoCapture(str(path))
            ok, _ = capture.read()
            capture.release()
            corrupted = not ok
        digest = None
        if path.is_file() and size_bytes > 0:
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(
                    lambda: handle.read(1024 * 1024), b""
                ):
                    hasher.update(block)
            digest = hasher.hexdigest()
        self.segments_produced += 1
        self.segments_corrupted += int(corrupted)
        self.bytes_used += size_bytes
        payload = {
            "recording_id": self.recording_id,
            "source_id": self.source_id,
            "session_id": self.session_id,
            "camera_role": self.camera_role,
            "stream_epoch": int(self.stream_epoch or 0),
            "segment_index": self.segment_index,
            "segment_path": relative_to_root(path),
            "started_at": self.segment_started_at,
            "ended_at": float(
                ended_at
                if ended_at is not None
                else self.last_timestamp_global
            ),
            "frame_count": self.frame_count,
            "size_bytes": size_bytes,
            "fps": self.segment_fps,
            "codec": "mp4v",
            "sha256": digest,
            "corrupted": corrupted,
            "immutable": True,
            "metadata": {
                "archive_kind": "dataset_traceability",
                "validated_on_site": False,
            },
            "frames": list(self.frames),
        }
        self.writer = None
        self.segment_path = None
        self.recording_id = None
        self.stream_epoch = None
        self.frames = []
        self.frame_count = 0
        return payload


def camera_worker_loop(
    source_config: dict[str, Any],
    db_path: str,
    config_values: dict[str, Any],
    inference_request_queue,
    inference_result_store,
    runtime_queue,
    stop_event,
    control_flags,
    active_runtime_models_by_task,
) -> None:
    db = VisionSortDB(Path(db_path))
    config = AppConfig(values=config_values)
    repo = ControlRepository(db)
    source_id = source_config["id"]
    session_id = source_config["session_id"]
    camera_role = source_config["role"]
    camera_id = source_id
    zones_by_role = config.get("tracking", "zones", default={})
    tracker_id = source_config["tracker_id"]
    tracker = build_tracker(
        tracker_id=tracker_id,
        session_id=session_id,
        source_id=source_id,
        camera_id=camera_id,
        camera_role=camera_role,
        zones=zones_by_role.get(camera_role, []),
        integrity_config=config.get("tracking", "integrity", default={}),
    )
    reid_config = config.get("tracking", "reid", default={}) or {}
    keyframe_selector: HandoffKeyframeSelector | None = None
    reid_error: str | None = None
    if bool(reid_config.get("enabled", True)):
        try:
            reid_encoder = ParcelReIDEncoder(
                str(
                    reid_config.get("model")
                    or "data/models/reid/mobilenet_v3_small-047dcff4.pth"
                ),
                device=str(reid_config.get("device") or "cpu"),
            )
            keyframe_selector = HandoffKeyframeSelector(
                reid_encoder,
                max_views=int(reid_config.get("max_views") or 5),
                min_views=int(reid_config.get("min_views") or 3),
                priority_zone_ids={
                    str(zone.get("zone_id"))
                    for zone in zones_by_role.get(camera_role, [])
                    if str(zone.get("kind") or "").lower()
                    in {"entry", "exit"}
                },
            )
        except Exception as exc:
            # Tracking remains operational; the supervisor exposes the local
            # artifact/configuration problem instead of losing the camera.
            reid_error = str(exc)
    event_engine = ParcelEventEngine(
        zones_by_role=zones_by_role,
        source_roles={camera_id: camera_role},
        confirmation_seconds=config.get(
            "tracking", "event_confirmation_seconds", default={}
        ),
    )
    recorder = _SegmentRecorder(
        source_id=source_id,
        session_id=session_id,
        camera_role=camera_role,
        segment_seconds=int(config.get("runtime", "recording_segment_seconds", default=10)),
    )
    archive_required = bool(source_config.get("archive_required", False))
    model_pipeline = list(source_config.get("model_pipeline") or [])
    if not model_pipeline:
        model_pipeline = [
            {
                "pipeline_role": "parcel_detection",
                "task": str(source_config.get("model_task") or "detection"),
                "model_id": source_config["model_id"],
            }
        ]
    session_obs_dir = OBSERVATIONS_DIR / session_id
    session_obs_dir.mkdir(parents=True, exist_ok=True)
    observations_path = session_obs_dir / f"{source_id}.jsonl"
    source = build_source(
        source_type=source_config["source_type"],
        camera_id=camera_id,
        camera_role=camera_role,
        session_id=session_id,
        uri=source_config["uri"],
        session_start_global=float(source_config["session_start_global"]),
        replay_fps=source_config.get("replay_fps", 8.0),
        replay_offset_ms=source_config.get("replay_offset_ms", 0.0),
        loop=bool(source_config.get("replay_loop", False)),
    )
    preview_path = PREVIEWS_DIR / f"{source_id}.jpg"
    calibration_frame_path = PREVIEWS_DIR / f"{source_id}-calibration.jpg"
    calibration_payload = source_config.get("calibration_profile_json") or {}
    if isinstance(calibration_payload, str):
        calibration_payload = json.loads(calibration_payload or "{}")
    calibration_profile = (
        CalibrationProfile.from_dict(calibration_payload)
        if calibration_payload
        else None
    )
    world_geometry: WorldGeometry | None = None
    calibration_diagnostic: dict[str, Any] = {
        "status": "NOT_CHECKED",
        "applicable": False,
    }
    calibration_image_size: tuple[int, int] | None = None
    last_tracking_model_id: str | None = None
    status = SourceStatus.CONNECTING.value
    frame_counter = 0
    fps_window_started = time.time()
    acquisition = LatestFrameBuffer(
        source,
        capacity=int(config.get("runtime", "max_buffer_size", default=3)),
        source_type=source_config["source_type"],
        reconnect_delay_s=1.0,
    )
    repo.update_source_state(source_id, status=status, fps=0.0, metrics=acquisition.metrics())
    try:
        acquisition.start()
        status = SourceStatus.REPLAY.value if source_config["source_type"] == "REPLAY" else SourceStatus.LIVE.value
        repo.update_source_state(
            source_id,
            status=status,
            fps=0.0,
            preview_path=str(preview_path),
            calibration_frame_path=str(calibration_frame_path),
        )
        while not stop_event.is_set():
            frame = acquisition.take_latest(timeout=0.5)
            if frame is None:
                if acquisition.error is not None:
                    raise acquisition.error
                if acquisition.finished:
                    break
                if source_config["source_type"] == "RTSP" and acquisition.reconnections:
                    status = SourceStatus.RECONNECTING.value
                    repo.update_source_state(
                        source_id,
                        status=status,
                        fps=0.0,
                        preview_path=str(preview_path),
                        metrics=acquisition.metrics(),
                    )
                continue

            while (
                bool(control_flags.get("__inference_paused__", False))
                and not stop_event.is_set()
            ):
                time.sleep(0.01)
            if stop_event.is_set():
                break
            current_image_size = (
                int(frame.image.shape[1]),
                int(frame.image.shape[0]),
            )
            if current_image_size != calibration_image_size:
                calibration_image_size = current_image_size
                calibration_diagnostic = runtime_calibration_diagnostic(
                    calibration_profile,
                    current_image_size,
                    source_id=source_id,
                    optical_configuration={
                        "source_id": source_id,
                        "source_type": str(source_config.get("source_type") or ""),
                        "source_uri": str(source_config.get("uri") or ""),
                        "camera_role": str(
                            source_config.get("role")
                            or source_config.get("camera_role")
                            or ""
                        ),
                        "optical_setup_id": str(
                            source_config.get("optical_setup_id") or "default"
                        ),
                    },
                )
                world_geometry = (
                    WorldGeometry(calibration_profile)
                    if calibration_profile is not None
                    and bool(calibration_diagnostic.get("applicable"))
                    else None
                )
            if acquisition.frames_processed % 5 == 0:
                cv2.imwrite(
                    str(calibration_frame_path),
                    frame.image,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        int(
                            config.get(
                                "runtime",
                                "preview_jpeg_quality",
                                default=82,
                            )
                        ),
                    ],
                )
            optional_recording = bool(
                control_flags.get(source_id, {}).get("recording")
            )
            if archive_required or optional_recording:
                finished = recorder.write(
                    frame,
                    max(float(frame.source_fps or 0.0), 1.0),
                )
                if finished:
                    runtime_queue.put(
                        {"kind": "RECORDING", **finished}
                    )
            elif recorder.writer is not None:
                finished = recorder.close(frame.timestamp_global)
                if finished:
                    runtime_queue.put(
                        {"kind": "RECORDING", **finished}
                    )
            ttl_seconds = float(
                config.get(
                    "runtime", "inference_result_ttl_seconds", default=5.0
                )
            )
            observations: list[Observation] = []
            frame_results: list[dict[str, Any]] = []
            pipeline_errors: list[str] = []
            has_pose_pipeline = any(
                str(item.get("task")) == "pose"
                or str(item.get("pipeline_role")) == "operator_pose"
                for item in model_pipeline
            )
            for pipeline in model_pipeline:
                task = str(pipeline["task"])
                pipeline_role = str(pipeline["pipeline_role"])
                model_id, routing_generation = resolve_pipeline_model(
                    pipeline,
                    active_runtime_models_by_task,
                )
                while (
                    bool(
                        control_flags.get(
                            f"__inference_paused__:{model_id}", False
                        )
                    )
                    and not stop_event.is_set()
                ):
                    time.sleep(0.01)
                message = build_inference_request(
                    frame=frame,
                    source_id=source_id,
                    ttl_seconds=ttl_seconds,
                    model_id=model_id,
                    task=task,
                    pipeline_role=pipeline_role,
                    routing_generation=routing_generation,
                )
                inflight_key = (
                    f"__inflight__:{message['request_id']}"
                )
                control_flags[inflight_key] = {
                    "source_id": source_id,
                    "model_id": model_id,
                    "task": task,
                    "routing_generation": routing_generation,
                }
                try:
                    inference_request_queue.put_nowait(message)
                except queue.Full:
                    control_flags.pop(inflight_key, None)
                    pipeline_errors.append(
                        f"{pipeline_role}: queue d'inférence pleine"
                    )
                    continue

                request_id = str(message["request_id"])
                while (
                    request_id not in inference_result_store
                    and time.time() < float(message["expires_at"])
                    and not stop_event.is_set()
                ):
                    time.sleep(0.01)
                result = inference_result_store.pop(request_id, None)
                control_flags.pop(inflight_key, None)
                if result is None:
                    _increment_inference_metric(
                        inference_result_store,
                        source_id,
                        f"timed_out:{model_id}",
                    )
                    pipeline_errors.append(
                        f"{pipeline_role}: réponse expirée"
                    )
                    continue
                expected_context = (
                    session_id,
                    source_id,
                    camera_id,
                    int(frame.stream_epoch),
                    int(frame.frame_index),
                    model_id,
                    task,
                    routing_generation,
                )
                result_context = (
                    result.get("session_id"),
                    result.get("source_id"),
                    result.get("camera_id"),
                    int(result.get("stream_epoch", -1)),
                    int(result.get("frame_index", -1)),
                    str(result.get("model_id") or ""),
                    str(result.get("task") or ""),
                    int(result.get("routing_generation") or 0),
                )
                if result_context != expected_context:
                    _increment_inference_metric(
                        inference_result_store,
                        source_id,
                        f"ignored:{model_id}",
                    )
                    pipeline_errors.append(
                        f"{pipeline_role}: contexte de réponse invalide"
                    )
                    continue
                if "error" in result:
                    pipeline_errors.append(
                        f"{pipeline_role}: {result['error']}"
                    )
                    continue
                frame_results.append(result)
                pipeline_observations = [
                    Observation(**row)
                    for row in result["observations"]
                ]
                if has_pose_pipeline and pipeline_role in {
                    "parcel_detection",
                    "parcel_segmentation",
                }:
                    pipeline_observations = [
                        item
                        for item in pipeline_observations
                        if item.class_name == "parcel"
                    ]
                elif task == "pose" or pipeline_role == "operator_pose":
                    pipeline_observations = [
                        item
                        for item in pipeline_observations
                        if item.class_name != "parcel"
                    ]
                for observation in pipeline_observations:
                    observation.attributes["_stream_epoch"] = int(
                        frame.stream_epoch
                    )
                    observation.attributes["inference_task"] = task
                    observation.attributes[
                        "pipeline_role"
                    ] = pipeline_role
                    observation.attributes[
                        "routing_generation"
                    ] = routing_generation
                observations.extend(pipeline_observations)
            if not frame_results:
                acquisition.mark_dropped()
                repo.update_source_state(
                    source_id,
                    status=SourceStatus.DEGRADED.value,
                    fps=0.0,
                    last_error="; ".join(pipeline_errors),
                    preview_path=str(preview_path),
                    calibration_frame_path=relative_to_root(
                        calibration_frame_path
                    ),
                    last_frame_ts=frame.timestamp_global,
                    metrics={
                        **acquisition.metrics(),
                        "calibration": calibration_diagnostic,
                    },
                )
                continue
            with observations_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "source_id": source_id,
                            "camera_id": camera_id,
                            "camera_role": camera_role,
                            "request_ids": [
                                item["request_id"]
                                for item in frame_results
                            ],
                            "stream_epoch": frame.stream_epoch,
                            "frame_index": frame.frame_index,
                            "timestamp_local": frame.timestamp_local,
                            "timestamp_global": frame.timestamp_global,
                            "model_id": observations[0].model_id if observations else None,
                            "model_ids": sorted(
                                {
                                    str(item["model_id"])
                                    for item in frame_results
                                }
                            ),
                            "tasks": sorted(
                                {
                                    str(item["task"])
                                    for item in frame_results
                                }
                            ),
                            "routing_generations": {
                                str(item["task"]): int(
                                    item.get("routing_generation") or 0
                                )
                                for item in frame_results
                            },
                            "pipeline_results": [
                                {
                                    "request_id": item["request_id"],
                                    "model_id": item["model_id"],
                                    "task": item["task"],
                                    "pipeline_role": item.get(
                                        "pipeline_role"
                                    ),
                                    "routing_generation": int(
                                        item.get("routing_generation") or 0
                                    ),
                                    "model_metrics": item.get(
                                        "model_metrics", {}
                                    ),
                                }
                                for item in frame_results
                            ],
                            "pipeline_errors": pipeline_errors,
                            "tracker_id": tracker_id,
                            "observations": [asdict(obs) for obs in observations],
                            "validated_on_site": False,
                        }
                    )
                    + "\n"
                )
            track_obs, finalized = tracker.update(
                frame_index=frame.frame_index,
                timestamp_local=frame.timestamp_local,
                timestamp_global=frame.timestamp_global,
                image_size=(int(frame.image.shape[1]), int(frame.image.shape[0])),
                observations=observations,
                image=frame.image,
                stream_epoch=int(frame.stream_epoch),
                world_geometry=world_geometry,
            )
            if keyframe_selector is not None:
                keyframe_selector.observe(track_obs, frame.image)
            last_tracking_model_id = observations[0].model_id if observations else None
            pop_integrity_events = getattr(tracker, "pop_integrity_events", None)
            if callable(pop_integrity_events):
                for integrity_event in pop_integrity_events():
                    runtime_queue.put(
                        {
                            "kind": "EVENT",
                            "session_id": session_id,
                            "source_id": source_id,
                            "model_id": last_tracking_model_id,
                            "tracker_id": tracker_id,
                            **integrity_event,
                        }
                    )
            parcel_tracks = [item for item in track_obs if item.class_name == "parcel"]
            context_tracks = [item for item in track_obs if item.class_name != "parcel"]
            for event in event_engine.update(
                camera_id,
                parcel_tracks,
                context_tracks,
                timestamp_global=frame.timestamp_global,
            ):
                local_parcel_key = str(event.pop("parcel_id"))
                runtime_queue.put(
                    {
                        "kind": "EVENT",
                        "session_id": session_id,
                        "source_id": source_id,
                        "parcel_id": None,
                        "local_parcel_key": local_parcel_key,
                        **event,
                    }
                )
            for tracklet in finalized:
                if keyframe_selector is not None:
                    tracklet = keyframe_selector.attach(tracklet)
                runtime_queue.put({"kind": "TRACKLET", "tracklet": asdict(tracklet)})

            now = time.time()
            acquisition.mark_processed()
            frame_counter += 1
            fps = frame_counter / max(now - fps_window_started, 1e-6)
            if frame_counter % 5 == 0:
                annotated = _annotate_frame(frame.image, observations, camera_id, fps, status)
                cv2.imwrite(str(preview_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), int(config.get("runtime", "preview_jpeg_quality", default=82))])
            repo.update_source_state(
                source_id,
                status=status,
                fps=fps,
                last_error=(
                    "; ".join(pipeline_errors)
                    if pipeline_errors
                    else None
                ),
                last_frame_ts=frame.timestamp_global,
                preview_path=relative_to_root(preview_path),
                calibration_frame_path=relative_to_root(
                    calibration_frame_path
                ),
                details_path=relative_to_root(observations_path),
                recording_enabled=archive_required
                or bool(
                    control_flags.get(source_id, {}).get("recording")
                ),
                metrics={
                    **acquisition.metrics(),
                    **_inference_metrics(inference_result_store, source_id),
                    "archive_required": archive_required,
                    "frames_archived": recorder.frames_archived_total,
                    "segments_produced": recorder.segments_produced,
                    "segments_corrupted": recorder.segments_corrupted,
                    "archive_bytes": recorder.bytes_used,
                    "model_pipeline": model_pipeline,
                    "model_metrics": {
                        str(item["model_id"]): item.get(
                            "model_metrics", {}
                        )
                        for item in frame_results
                    },
                    "calibration": calibration_diagnostic,
                    "tracking_integrity": (
                        tracker.integrity_metrics()
                        if callable(getattr(tracker, "integrity_metrics", None))
                        else {}
                    ),
                    "parcel_reid": {
                        "enabled": keyframe_selector is not None,
                        "model_version": (
                            keyframe_selector.encoder.model_version
                            if keyframe_selector is not None
                            else None
                        ),
                        "error": reid_error,
                    },
                    "validated_on_site": False,
                },
            )
        for tracklet in tracker.flush():
            if keyframe_selector is not None:
                tracklet = keyframe_selector.attach(tracklet)
            runtime_queue.put({"kind": "TRACKLET", "tracklet": asdict(tracklet)})
        pop_integrity_events = getattr(tracker, "pop_integrity_events", None)
        if callable(pop_integrity_events):
            for integrity_event in pop_integrity_events():
                runtime_queue.put(
                    {
                        "kind": "EVENT",
                        "session_id": session_id,
                        "source_id": source_id,
                        "model_id": last_tracking_model_id,
                        "tracker_id": tracker_id,
                        **integrity_event,
                    }
                )
        repo.update_source_state(
            source_id,
            status=SourceStatus.OFFLINE.value,
            fps=0.0,
            preview_path=relative_to_root(preview_path),
            calibration_frame_path=relative_to_root(calibration_frame_path),
        )
    except Exception as exc:  # pragma: no cover - runtime
        repo.update_source_state(
            source_id,
            status=SourceStatus.ERROR.value,
            fps=0.0,
            last_error=str(exc),
            preview_path=relative_to_root(preview_path),
            calibration_frame_path=relative_to_root(calibration_frame_path),
        )
    finally:
        finished = recorder.close()
        if finished:
            runtime_queue.put({"kind": "RECORDING", **finished})
        for key in list(control_flags.keys()):
            inflight = control_flags.get(key)
            if (
                str(key).startswith("__inflight__:")
                and (
                    inflight == source_id
                    or (
                        isinstance(inflight, dict)
                        and inflight.get("source_id") == source_id
                    )
                )
            ):
                control_flags.pop(key, None)
        acquisition.stop()
        metrics = acquisition.metrics()
        runtime_queue.put(
            {
                "kind": "MEDIA_COVERAGE",
                "session_id": session_id,
                "source_id": source_id,
                "archive_required": archive_required,
                "frames_acquired": metrics["frames_received"],
                "frames_processed": metrics["frames_processed"],
                "frames_archived": recorder.frames_archived_total,
                "segments_produced": recorder.segments_produced,
                "segments_corrupted": recorder.segments_corrupted,
                "bytes_used": recorder.bytes_used,
                "details": {
                    **metrics,
                    "source_type": source_config["source_type"],
                    "camera_role": camera_role,
                    "automatic_archive": archive_required,
                },
            }
        )
