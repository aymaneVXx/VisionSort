from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable

from visionsort.core.types import TrackObservation
from visionsort.tracking.integrity import TrackIntegrityManager


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    frame_index: int
    timestamp_global: float
    stream_epoch: int
    tracks: tuple[dict[str, Any], ...]


def load_replay(path: str | Path) -> list[ReplayFrame]:
    frames: list[ReplayFrame] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        tracks = payload.get("tracks", [])
        if not isinstance(tracks, list):
            raise ValueError(f"Ligne {line_number}: tracks doit etre une liste.")
        frames.append(
            ReplayFrame(
                frame_index=int(payload.get("frame_index", len(frames))),
                timestamp_global=float(payload["timestamp_global"]),
                stream_epoch=int(payload.get("stream_epoch", 0)),
                tracks=tuple(dict(item) for item in tracks),
            )
        )
    return frames


def _observation(frame: ReplayFrame, payload: dict[str, Any]) -> TrackObservation:
    backend_id = int(payload["backend_track_id"])
    return TrackObservation(
        session_id=str(payload.get("session_id") or "integrity-benchmark"),
        source_id=str(payload.get("source_id") or "benchmark-camera"),
        camera_id=str(payload.get("camera_id") or "benchmark-camera"),
        camera_role=str(payload.get("camera_role") or "C1"),
        local_track_id=backend_id,
        backend_track_id=backend_id,
        frame_index=frame.frame_index,
        timestamp_local=float(payload.get("timestamp_local", frame.timestamp_global)),
        timestamp_global=frame.timestamp_global,
        class_name=str(payload.get("class_name") or "parcel"),
        confidence=float(payload.get("confidence", 1.0)),
        bbox=tuple(float(value) for value in payload["bbox"]),
        velocity=(0.0, 0.0),
        model_id=payload.get("model_id"),
        tracker_id="bytetrack_cpu",
        extra={"mask": payload["mask"]} if payload.get("mask") else {},
    )


def _identity_metrics(records: Iterable[tuple[str, str]]) -> dict[str, int]:
    identities_by_truth: dict[str, list[str]] = {}
    for truth_id, identity in records:
        identities_by_truth.setdefault(truth_id, []).append(identity)
    fragmentations = sum(
        max(0, len(set(identities)) - 1)
        for identities in identities_by_truth.values()
    )
    switches = 0
    for identities in identities_by_truth.values():
        switches += sum(
            current != previous
            for previous, current in zip(identities, identities[1:])
        )
    return {"fragmentations": fragmentations, "id_switches": switches}


def benchmark_replay(
    frames: Iterable[ReplayFrame],
    *,
    image_size: tuple[int, int],
    integrity_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manager = TrackIntegrityManager(
        session_id="integrity-benchmark",
        source_id="benchmark-camera",
        camera_id="benchmark-camera",
        camera_role="C1",
        tracker_id="bytetrack_cpu",
        config=integrity_config,
    )
    raw_truth_records: list[tuple[str, str]] = []
    canonical_truth_records: list[tuple[str, str]] = []
    raw_identities: set[str] = set()
    canonical_identities: set[int] = set()
    event_counts: dict[str, int] = {}
    started = time.perf_counter()
    for frame in frames:
        observations = [_observation(frame, payload) for payload in frame.tracks]
        canonical, _ = manager.update(
            observations,
            frame_index=frame.frame_index,
            timestamp_global=frame.timestamp_global,
            image_size=image_size,
            stream_epoch=frame.stream_epoch,
        )
        canonical_by_backend = {
            int(item.backend_track_id): item.local_track_id
            for item in canonical
            if item.backend_track_id is not None
        }
        for payload in frame.tracks:
            backend_id = int(payload["backend_track_id"])
            raw_identity = f"{frame.stream_epoch}:{backend_id}"
            raw_identities.add(raw_identity)
            canonical_id = canonical_by_backend.get(backend_id)
            if canonical_id is not None:
                canonical_identities.add(canonical_id)
            truth_id = payload.get("ground_truth_id")
            if truth_id is not None:
                raw_truth_records.append((str(truth_id), raw_identity))
                if canonical_id is not None:
                    canonical_truth_records.append((str(truth_id), str(canonical_id)))
        for event in manager.pop_events():
            event_type = str(event["event_type"])
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
    wall_seconds = time.perf_counter() - started
    runtime = manager.metrics()
    return {
        "raw_bytetrack": {
            "tracks_created": len(raw_identities),
            **_identity_metrics(raw_truth_records),
        },
        "bytetrack_plus_integrity": {
            "canonical_tracks_created": len(canonical_identities),
            **_identity_metrics(canonical_truth_records),
            "relinks": int(runtime["relinks"]),
            "ambiguous_or_refused": int(runtime["ambiguous_refusals"]),
        },
        "integrity_runtime": {
            "average_ms_per_frame": float(runtime["runtime_avg_ms"]),
            "maximum_ms_per_frame": float(runtime["runtime_max_ms"]),
            "lapjv_subproblems": int(runtime["lapjv_subproblems"]),
            "benchmark_wall_seconds": wall_seconds,
        },
        "events": event_counts,
        "ground_truth_available": bool(raw_truth_records),
        "validated_on_site": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare les IDs ByteTrack bruts et les IDs canoniques VisionSort."
    )
    parser.add_argument("replay_jsonl", type=Path)
    parser.add_argument("--image-width", type=int, required=True)
    parser.add_argument("--image-height", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark_replay(
        load_replay(args.replay_jsonl),
        image_size=(args.image_width, args.image_height),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
