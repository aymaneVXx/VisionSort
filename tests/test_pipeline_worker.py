import json
from pathlib import Path

from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import (
    ArtifactRepository,
    ControlRepository,
)
from visionsort.core.paths import OBSERVATIONS_DIR
from visionsort.runtime.demo_assets import ensure_demo_assets
from visionsort.runtime.pipeline_worker import pipeline_worker_loop


def test_pipeline_sample_autoannotate_finalize(tmp_path):
    db_path = tmp_path / "visionsort.db"
    db = VisionSortDB(db_path)
    db.initialize()
    repo = ControlRepository(db)

    assets = ensure_demo_assets()
    now = utc_now()
    db.execute(
        """
        INSERT INTO sources (id, name, role, source_type, uri, model_id, tracker_id, enabled, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        """,
        ("src1", "Replay C1", "C1", "REPLAY", assets["C1"], "demo_synth_det", "greedy_iou", now, now),
    )
    db.execute(
        """
        INSERT INTO sources (id, name, role, source_type, uri, model_id, tracker_id, enabled, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        """,
        ("src2", "Replay C2", "C2", "REPLAY", assets["C2"], "demo_synth_det", "greedy_iou", now, now),
    )

    session_id = repo.create_capture_session(
        name="pytest session",
        demo_mode=True,
        sources=[
            {"source_id": "src1", "camera_role": "C1", "time_offset_ms": 0.0},
            {"source_id": "src2", "camera_role": "C2", "time_offset_ms": 0.0},
        ],
        config={},
    )

    details = tmp_path / "tracklet.jsonl"
    details.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "source_id": "src1",
                "camera_id": "src1",
                "camera_role": "C1",
                "local_track_id": 1,
                "frame_index": 5,
                "timestamp_local": 0.5,
                "timestamp_global": 0.5,
                "class_name": "parcel",
                "confidence": 0.95,
                "bbox": [100, 160, 160, 200],
                "velocity": [0, 0],
                "zone_id": "c1_exit",
                "appearance_hint": None,
                "model_id": "demo_synth_det",
                "tracker_id": "greedy_iou",
                "extra": {"parcel_hint": "P1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db.execute(
        """
        INSERT INTO tracklets
        (tracklet_id, parcel_id, session_id, source_id, camera_id, camera_role, local_track_id,
         started_at_local, ended_at_local, started_at_global, ended_at_global,
         class_name, last_zone_id, frame_count, avg_speed, observation_path, summary_json, match_result, model_id, tracker_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "t1",
            "P1",
            session_id,
            "src1",
            "src1",
            "C1",
            1,
            0.0,
            1.0,
            0.0,
            1.0,
            "parcel",
            "c1_exit",
            1,
            0.0,
            str(details),
            '{"parcel_hint":"P1"}',
            "UNMATCHED",
            "demo_synth_det",
            "greedy_iou",
        ),
    )

    pipeline_worker_loop(str(db_path), session_id, "SAMPLE", {"name": "pytest_ds"})
    sess = repo.get_capture_session(session_id)
    assert sess is not None
    assert sess["pipeline_state"] == "SAMPLED"
    assert sess["last_dataset_id"]
    dataset_id = sess["last_dataset_id"]

    pipeline_worker_loop(str(db_path), session_id, "AUTO_ANNOTATE", {"dataset_id": dataset_id, "fallback_model_id": "demo_synth_det"})
    items = db.fetch_all("SELECT * FROM dataset_items WHERE dataset_id = ?", (dataset_id,))
    assert len(items) >= 1
    assert any(row["label_path"] for row in items)

    pipeline_worker_loop(str(db_path), session_id, "FINALIZE_DATASET", {"dataset_id": dataset_id})
    sess2 = repo.get_capture_session(session_id)
    assert sess2 is not None
    assert sess2["pipeline_state"] in {"REVIEW_PENDING", "DATASET_READY"}


def test_pipeline_finalize_becomes_dataset_ready_after_review(tmp_path):
    db_path = tmp_path / "visionsort.db"
    db = VisionSortDB(db_path)
    db.initialize()
    repo = ControlRepository(db)

    assets = ensure_demo_assets()
    now = utc_now()
    db.execute(
        """
        INSERT INTO sources (id, name, role, source_type, uri, model_id, tracker_id, enabled, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        """,
        ("src1", "Replay C1", "C1", "REPLAY", assets["C1"], "demo_synth_det", "greedy_iou", now, now),
    )
    session_id = repo.create_capture_session(
        name="pytest review session",
        demo_mode=True,
        sources=[{"source_id": "src1", "camera_role": "C1", "time_offset_ms": 0.0}],
        config={},
    )

    details = tmp_path / "tracklet_review.jsonl"
    details.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "source_id": "src1",
                "camera_id": "src1",
                "camera_role": "C1",
                "local_track_id": 1,
                "frame_index": 3,
                "timestamp_local": 0.3,
                "timestamp_global": 0.3,
                "class_name": "parcel",
                "confidence": 0.4,
                "bbox": [80, 90, 140, 150],
                "velocity": [0, 0],
                "zone_id": "c1_exit",
                "appearance_hint": None,
                "model_id": "demo_synth_det",
                "tracker_id": "greedy_iou",
                "extra": {"parcel_hint": "P2"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db.execute(
        """
        INSERT INTO tracklets
        (tracklet_id, parcel_id, session_id, source_id, camera_id, camera_role, local_track_id,
         started_at_local, ended_at_local, started_at_global, ended_at_global,
         class_name, last_zone_id, frame_count, avg_speed, observation_path, summary_json, match_result, model_id, tracker_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "t-review",
            "P2",
            session_id,
            "src1",
            "src1",
            "C1",
            1,
            0.0,
            1.0,
            0.0,
            1.0,
            "parcel",
            "c1_exit",
            1,
            0.0,
            str(details),
            '{"parcel_hint":"P2"}',
            "UNMATCHED",
            "demo_synth_det",
            "greedy_iou",
        ),
    )

    pipeline_worker_loop(str(db_path), session_id, "SAMPLE", {"name": "pytest_review_ds"})
    session = repo.get_capture_session(session_id)
    assert session is not None
    dataset_id = session["last_dataset_id"]
    pipeline_worker_loop(str(db_path), session_id, "AUTO_ANNOTATE", {"dataset_id": dataset_id, "fallback_model_id": "demo_synth_det"})
    pipeline_worker_loop(str(db_path), session_id, "FINALIZE_DATASET", {"dataset_id": dataset_id})

    item = db.fetch_one("SELECT * FROM dataset_items WHERE dataset_id = ? LIMIT 1", (dataset_id,))
    assert item is not None
    label_path = Path(str(item["label_path"]))
    if not label_path.is_absolute():
        from visionsort.core.paths import ROOT_DIR

        label_path = ROOT_DIR / label_path
    parcel_line = next(
        line
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("0 ")
    )
    label_path.write_text(parcel_line + "\n", encoding="utf-8")
    ArtifactRepository(db).update_dataset_item(
        str(item["id"]), annotation_status="HUMAN_VALIDATED"
    )

    pipeline_worker_loop(str(db_path), session_id, "FINALIZE_DATASET", {"dataset_id": dataset_id})
    dataset = db.fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    session2 = repo.get_capture_session(session_id)
    assert dataset is not None
    assert session2 is not None
    assert dataset["status"] == "DATASET_READY"
    assert session2["pipeline_state"] == "DATASET_READY"


def test_pipeline_process_session_sets_processed(tmp_path):
    db_path = tmp_path / "visionsort.db"
    db = VisionSortDB(db_path)
    db.initialize()
    repo = ControlRepository(db)
    session_id = repo.create_capture_session(name="pytest process", demo_mode=True, sources=[], config={})
    repo.update_capture_session(session_id, ended_at=123.4)
    db.execute(
        "INSERT INTO events (id, event_type, parcel_id, camera_id, severity, payload_json, session_id, source_id, frame_index, timestamp_global, model_id, tracker_id, created_at) VALUES (?, ?, NULL, NULL, 'info', '{}', ?, NULL, NULL, NULL, NULL, NULL, ?)",
        ("evt1", "session_stopped", session_id, utc_now()),
    )
    db.execute(
        "INSERT INTO recordings (id, source_id, session_id, segment_path, started_at, ended_at, frame_count, size_bytes, created_at) VALUES (?, ?, ?, ?, 0.0, 1.0, 1, 0, ?)",
        ("rec1", "srcX", session_id, "data/recordings/x.mp4", utc_now()),
    )
    pipeline_worker_loop(str(db_path), session_id, "PROCESS_SESSION", {})
    sess = repo.get_capture_session(session_id)
    assert sess is not None
    assert sess["pipeline_state"] == "PROCESSED"
    assert sess["report_path"]


def test_pipeline_export_observations_parquet(tmp_path):
    db_path = tmp_path / "visionsort.db"
    db = VisionSortDB(db_path)
    db.initialize()
    repo = ControlRepository(db)
    session_id = repo.create_capture_session(name="pytest export", demo_mode=True, sources=[], config={})
    session_dir = OBSERVATIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    jsonl = session_dir / "src1.jsonl"
    jsonl.write_text('{"session_id":"%s","frame_index":1}\n{"session_id":"%s","frame_index":2}\n' % (session_id, session_id), encoding="utf-8")
    pipeline_worker_loop(str(db_path), session_id, "EXPORT_OBSERVATIONS_PARQUET", {})
    step = db.fetch_one(
        "SELECT * FROM pipeline_step_runs WHERE session_id = ? AND step = ? ORDER BY created_at DESC LIMIT 1",
        (session_id, "EXPORT_OBSERVATIONS_PARQUET"),
    )
    assert step is not None
    if step["status"] == "COMPLETED":
        assert (session_dir / "src1.parquet").exists()
    else:
        assert step["status"] == "FAILED"
