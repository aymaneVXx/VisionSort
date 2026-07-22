from __future__ import annotations

import json
import queue
import time
from pathlib import Path
from typing import Any

import cv2

from visionsort.core.config import AppConfig
from visionsort.core.enums import SourceStatus
from visionsort.core.paths import PREVIEWS_DIR, RECORDINGS_DIR
from visionsort.core.types import Observation
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import ControlRepository
from visionsort.events.engine import ParcelEventEngine
from visionsort.sources.frame_sources import build_source
from visionsort.tracking.engine import GreedyIOUTracker


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
    def __init__(self, source_id: str, segment_seconds: int):
        self.source_id = source_id
        self.segment_seconds = segment_seconds
        self.writer = None
        self.segment_started_at = 0.0
        self.segment_path: Path | None = None
        self.frame_count = 0

    def write(self, frame, fps: float) -> dict[str, Any] | None:
        now = time.time()
        if self.writer is None or (now - self.segment_started_at) >= self.segment_seconds:
            finished = self.close()
            self.segment_started_at = now
            self.segment_path = RECORDINGS_DIR / self.source_id / f"{int(now)}.mp4"
            self.segment_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.segment_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, fps),
                (width, height),
            )
            self.frame_count = 0
            if finished:
                return finished
        self.writer.write(frame)
        self.frame_count += 1
        return None

    def close(self) -> dict[str, Any] | None:
        if self.writer is None or self.segment_path is None:
            return None
        self.writer.release()
        payload = {
            "source_id": self.source_id,
            "segment_path": str(self.segment_path),
            "started_at": self.segment_started_at,
            "ended_at": time.time(),
            "frame_count": self.frame_count,
            "size_bytes": self.segment_path.stat().st_size if self.segment_path.exists() else 0,
        }
        self.writer = None
        self.segment_path = None
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
) -> None:
    db = VisionSortDB(Path(db_path))
    config = AppConfig(values=config_values)
    repo = ControlRepository(db)
    source_id = source_config["id"]
    camera_id = source_config["role"]
    zones_by_role = config.get("tracking", "zones", default={})
    tracker = GreedyIOUTracker(camera_id=camera_id, zones=zones_by_role.get(camera_id, []))
    event_engine = ParcelEventEngine(zones_by_role=zones_by_role, source_roles={camera_id: camera_id})
    recorder = _SegmentRecorder(
        source_id=source_id,
        segment_seconds=int(config.get("runtime", "recording_segment_seconds", default=10)),
    )
    source = build_source(
        source_type=source_config["source_type"],
        camera_id=camera_id,
        uri=source_config["uri"],
        replay_fps=source_config.get("replay_fps", 8.0),
    )
    preview_path = PREVIEWS_DIR / f"{source_id}.jpg"
    status = SourceStatus.CONNECTING.value
    frame_counter = 0
    fps_window_started = time.time()
    dropped_frames = 0
    repo.update_source_state(source_id, status=status, fps=0.0, metrics={"dropped_frames": dropped_frames})
    try:
        source.open()
        status = SourceStatus.REPLAY.value if source_config["source_type"] == "REPLAY" else SourceStatus.LIVE.value
        repo.update_source_state(source_id, status=status, fps=0.0, preview_path=str(preview_path))
        while not stop_event.is_set():
            frame = source.read()
            if frame is None:
                if source_config["source_type"] == "RTSP":
                    status = SourceStatus.RECONNECTING.value
                    repo.update_source_state(source_id, status=status, fps=0.0, preview_path=str(preview_path))
                    time.sleep(1.0)
                    continue
                break

            message = {
                "kind": "INFER",
                "camera_id": camera_id,
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "image": frame.image,
            }
            try:
                inference_request_queue.put_nowait(message)
            except queue.Full:
                dropped_frames += 1
                repo.update_source_state(
                    source_id,
                    status=SourceStatus.DEGRADED.value,
                    fps=0.0,
                    preview_path=str(preview_path),
                    last_frame_ts=frame.timestamp,
                    metrics={"dropped_frames": dropped_frames},
                )
                continue

            result_key = f"{camera_id}:{frame.frame_index}"
            started_wait = time.time()
            while result_key not in inference_result_store and (time.time() - started_wait) < 3.0 and not stop_event.is_set():
                time.sleep(0.01)
            result = inference_result_store.pop(result_key, None)
            if result is None:
                dropped_frames += 1
                continue
            if "error" in result:
                repo.update_source_state(source_id, status=SourceStatus.ERROR.value, last_error=result["error"], preview_path=str(preview_path))
                continue
            observations = [Observation(**row) for row in result["observations"]]
            track_obs, finalized = tracker.update(frame.frame_index, frame.timestamp, observations)
            parcel_tracks = [item for item in track_obs if item.class_name == "parcel"]
            context_tracks = [item for item in track_obs if item.class_name != "parcel"]
            for event in event_engine.update(camera_id, parcel_tracks, context_tracks):
                runtime_queue.put({"kind": "EVENT", **event})
            for tracklet in finalized:
                runtime_queue.put({"kind": "TRACKLET", "tracklet": tracklet.__dict__})

            now = time.time()
            frame_counter += 1
            fps = frame_counter / max(now - fps_window_started, 1e-6)
            if frame_counter % 5 == 0:
                annotated = _annotate_frame(frame.image, observations, camera_id, fps, status)
                cv2.imwrite(str(preview_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), int(config.get("runtime", "preview_jpeg_quality", default=82))])
            if control_flags.get(source_id, {}).get("recording"):
                finished = recorder.write(frame.image, max(fps, 1.0))
                if finished:
                    runtime_queue.put({"kind": "RECORDING", **finished})
            elif recorder.writer is not None:
                finished = recorder.close()
                if finished:
                    runtime_queue.put({"kind": "RECORDING", **finished})

            repo.update_source_state(
                source_id,
                status=status,
                fps=fps,
                last_error=None,
                last_frame_ts=frame.timestamp,
                preview_path=str(preview_path),
                recording_enabled=bool(control_flags.get(source_id, {}).get("recording")),
                metrics={"dropped_frames": dropped_frames, "validated_on_site": False},
            )
        for tracklet in tracker.flush():
            runtime_queue.put({"kind": "TRACKLET", "tracklet": tracklet.__dict__})
        finished = recorder.close()
        if finished:
            runtime_queue.put({"kind": "RECORDING", **finished})
        repo.update_source_state(source_id, status=SourceStatus.OFFLINE.value, fps=0.0, preview_path=str(preview_path))
    except Exception as exc:  # pragma: no cover - runtime
        repo.update_source_state(source_id, status=SourceStatus.ERROR.value, fps=0.0, last_error=str(exc), preview_path=str(preview_path))
    finally:
        source.close()
