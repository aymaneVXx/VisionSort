from pathlib import Path
import json

from visionsort.database.db import VisionSortDB, utc_now
from visionsort.datasets.pipeline import build_dataset
from visionsort.runtime.demo_assets import ensure_demo_assets


def test_build_dataset_from_tracklet_jsonl(tmp_path):
    assets = ensure_demo_assets()
    db = VisionSortDB(tmp_path / "visionsort.db")
    db.initialize()
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
        INSERT INTO capture_sessions (id, name, pipeline_state, demo_mode, site_validated, config_json, report_path, started_at, ended_at, created_at, updated_at)
        VALUES (?, ?, 'CAPTURED', 1, 0, '{}', NULL, NULL, NULL, ?, ?)
        """,
        ("session-test", "pytest session", now, now),
    )
    db.execute(
        """
        INSERT INTO capture_session_sources (id, session_id, source_id, camera_role, time_offset_ms, replay_fps, created_at, updated_at)
        VALUES (?, ?, ?, ?, 0, 8.0, ?, ?)
        """,
        ("sesssrc1", "session-test", "src1", "C1", now, now),
    )
    details = tmp_path / "tracklet.jsonl"
    details.write_text(
        '{"session_id":"session-test","source_id":"src1","camera_id":"src1","camera_role":"C1","local_track_id":1,"frame_index":5,"timestamp_local":0.5,"timestamp_global":0.5,"class_name":"parcel","confidence":0.95,"bbox":[100,160,160,200],"velocity":[0,0],"zone_id":"c1_exit","appearance_hint":null,"model_id":"demo_synth_det","tracker_id":"greedy_iou","extra":{"parcel_hint":"P1"}}\n',
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
            "session-test",
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
    result = build_dataset(db, session_id="session-test", name="pytest_dataset")
    assert result["manifest_rows"] >= 1
    assert Path(result["root"]).exists()
    dataset = db.fetch_one("SELECT * FROM datasets WHERE id = ?", (result["dataset_id"],))
    assert dataset is not None
    summary = json.loads(dataset["summary_json"] or "{}")
    assert summary["dataset_version"] == result["dataset_id"]
    assert summary["manifest_sha256"]
