from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from visionsort.acquisition.worker import _SegmentRecorder
from visionsort.core.paths import ROOT_DIR
from visionsort.core.types import Frame
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import (
    ArtifactRepository,
    ControlRepository,
)
from visionsort.datasets.pipeline import build_dataset
from visionsort.media.archive import (
    FrameArchiveError,
    FrameArchiveKey,
    FrameArchiveResolver,
    build_session_media_report,
)


def _write_source_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (96, 64)
    )
    assert writer.isOpened()
    for index in range(4):
        image = np.full((64, 96, 3), 20 + index * 30, dtype=np.uint8)
        writer.write(image)
    writer.release()


def _persist_segment(
    repository: ArtifactRepository, payload: dict
) -> None:
    repository.add_recording(
        recording_id=payload["recording_id"],
        source_id=payload["source_id"],
        session_id=payload["session_id"],
        camera_role=payload["camera_role"],
        stream_epoch=payload["stream_epoch"],
        segment_index=payload["segment_index"],
        segment_path=payload["segment_path"],
        started_at=payload["started_at"],
        ended_at=payload["ended_at"],
        frame_count=payload["frame_count"],
        size_bytes=payload["size_bytes"],
        fps=payload["fps"],
        codec=payload["codec"],
        sha256=payload["sha256"],
        corrupted=payload["corrupted"],
        immutable=payload["immutable"],
        metadata=payload["metadata"],
        frames=payload["frames"],
    )


def _insert_tracklet(
    db: VisionSortDB,
    *,
    tmp_path: Path,
    session_id: str,
    source_id: str,
    frame_index: int,
) -> None:
    details = tmp_path / f"{session_id}-{source_id}.jsonl"
    details.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "source_id": source_id,
                "camera_id": source_id,
                "camera_role": "C1",
                "local_track_id": 1,
                "frame_index": frame_index,
                "timestamp_local": frame_index / 8.0,
                "timestamp_global": 1000.0 + frame_index / 8.0,
                "class_name": "parcel",
                "confidence": 0.95,
                "bbox": [10, 10, 40, 40],
                "velocity": [0, 0],
                "zone_id": None,
                "appearance_hint": None,
                "model_id": "demo_synth_det",
                "tracker_id": "greedy_iou",
                "extra": {"_stream_epoch": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db.execute(
        """
        INSERT INTO tracklets
        (tracklet_id, parcel_id, session_id, source_id, camera_id,
         camera_role, local_track_id, started_at_local, ended_at_local,
         started_at_global, ended_at_global, class_name, last_zone_id,
         frame_count, avg_speed, observation_path, summary_json,
         match_result, model_id, tracker_id)
        VALUES (?, NULL, ?, ?, ?, 'C1', 1, ?, ?, ?, ?, 'parcel', NULL,
                1, 0.0, ?, '{}', 'UNMATCHED', 'demo_synth_det',
                'greedy_iou')
        """,
        (
            f"tracklet-{session_id}",
            session_id,
            source_id,
            source_id,
            frame_index / 8.0,
            frame_index / 8.0,
            1000.0 + frame_index / 8.0,
            1000.0 + frame_index / 8.0,
            str(details),
        ),
    )


def _archived_session(tmp_path: Path):
    db = VisionSortDB(tmp_path / "visionsort.db")
    db.initialize()
    control = ControlRepository(db)
    artifacts = ArtifactRepository(db)
    source_video = tmp_path / "source.mp4"
    _write_source_video(source_video)
    source_id = control.upsert_source(
        {
            "id": "video-c1",
            "name": "Video C1",
            "role": "C1",
            "source_type": "VIDEO_FILE",
            "uri": str(source_video),
            "model_id": "demo_synth_det",
            "tracker_id": "greedy_iou",
            "enabled": True,
        }
    )
    session_id = control.create_capture_session(
        name="Archived video",
        demo_mode=True,
        sources=[{"source_id": source_id, "camera_role": "C1"}],
        config={"validated_on_site": False},
    )
    recorder = _SegmentRecorder(
        source_id=source_id,
        session_id=session_id,
        camera_role="C1",
        segment_seconds=30,
        recordings_dir=tmp_path / "storage" / "recordings",
    )
    for index in range(4):
        frame = Frame(
            session_id=session_id,
            camera_id=source_id,
            camera_role="C1",
            frame_index=10 + index,
            timestamp_local=(10 + index) / 8.0,
            timestamp_global=1000.0 + (10 + index) / 8.0,
            image=np.full(
                (64, 96, 3), 40 + index * 40, dtype=np.uint8
            ),
            source_fps=8.0,
            stream_epoch=0,
        )
        assert recorder.write(frame, 8.0) is None
    payload = recorder.close()
    assert payload is not None
    _persist_segment(artifacts, payload)
    artifacts.upsert_media_coverage(
        session_id=session_id,
        source_id=source_id,
        archive_required=True,
        frames_acquired=4,
        frames_processed=4,
        frames_archived=4,
        segments_produced=1,
        segments_corrupted=0,
        bytes_used=payload["size_bytes"],
    )
    control.update_capture_session(session_id, ended_at=1002.0)
    return db, control, session_id, source_id, payload


def test_segment_archive_is_scoped_to_session_and_resolves_frame(tmp_path):
    db, _, session_id, source_id, payload = _archived_session(tmp_path)

    assert session_id in Path(payload["segment_path"]).parts
    assert source_id in Path(payload["segment_path"]).parts
    assert payload["frame_count"] == 4
    assert payload["sha256"]
    assert payload["corrupted"] is False

    image, provenance = FrameArchiveResolver(db).resolve(
        FrameArchiveKey(
            session_id=session_id,
            source_id=source_id,
            stream_epoch=0,
            frame_index=11,
            timestamp_global=1000.0 + 11 / 8.0,
        )
    )
    assert image is not None
    assert provenance["origin"] == "session_archive"
    assert provenance["segment_frame_index"] == 1


def test_dataset_uses_archive_after_source_uri_changes(tmp_path):
    db, control, session_id, source_id, _ = _archived_session(tmp_path)
    _insert_tracklet(
        db,
        tmp_path=tmp_path,
        session_id=session_id,
        source_id=source_id,
        frame_index=11,
    )
    control.upsert_source(
        {
            "id": source_id,
            "name": "Video C1 changed",
            "role": "C1",
            "source_type": "VIDEO_FILE",
            "uri": str(tmp_path / "missing-after-capture.mp4"),
            "model_id": "demo_synth_det",
            "tracker_id": "greedy_iou",
            "enabled": True,
        }
    )

    result = build_dataset(
        db, session_id=session_id, name="archive-backed"
    )
    item = db.fetch_one(
        "SELECT * FROM dataset_items WHERE dataset_id = ?",
        (result["dataset_id"],),
    )
    metadata = json.loads(item["metadata_json"])
    image_path = ROOT_DIR / item["image_path"]

    assert image_path.exists()
    assert metadata["media_provenance"]["origin"] == "session_archive"
    assert metadata["media_provenance"]["recording_id"]


def test_required_archive_with_insufficient_coverage_is_refused(tmp_path):
    db = VisionSortDB(tmp_path / "insufficient.db")
    db.initialize()
    control = ControlRepository(db)
    source_path = tmp_path / "source.mp4"
    _write_source_video(source_path)
    source_id = control.upsert_source(
        {
            "id": "video-missing",
            "name": "Missing archive",
            "role": "C1",
            "source_type": "VIDEO_FILE",
            "uri": str(source_path),
            "model_id": "demo_synth_det",
            "tracker_id": "greedy_iou",
            "enabled": True,
        }
    )
    session_id = control.create_capture_session(
        name="Missing archive",
        demo_mode=True,
        sources=[{"source_id": source_id, "camera_role": "C1"}],
        config={},
    )
    control.update_capture_session(session_id, ended_at=1.0)
    _insert_tracklet(
        db,
        tmp_path=tmp_path,
        session_id=session_id,
        source_id=source_id,
        frame_index=1,
    )

    report = build_session_media_report(db, session_id)
    assert report["valid"] is False
    assert report["status"] == "INSUFFICIENT"
    with pytest.raises(FrameArchiveError, match="Couverture média"):
        build_dataset(db, session_id=session_id, name="must-fail")
