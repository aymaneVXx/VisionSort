from pathlib import Path

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
    details = tmp_path / "tracklet.jsonl"
    details.write_text(
        '{"camera_id":"C1","local_track_id":1,"frame_index":5,"timestamp":0.5,"class_name":"parcel","confidence":0.95,"bbox":[100,160,160,200],"velocity":[0,0],"zone_id":"c1_exit","appearance_hint":null,"extra":{"parcel_hint":"P1"}}\n',
        encoding="utf-8",
    )
    db.execute(
        """
        INSERT INTO tracklets (tracklet_id, parcel_id, camera_id, local_track_id, started_at, ended_at, class_name, last_zone_id, frame_count, avg_speed, observation_path, summary_json, match_result)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("t1", "P1", "C1", 1, 0.0, 1.0, "parcel", "c1_exit", 1, 0.0, str(details), '{"parcel_hint":"P1"}', "UNMATCHED"),
    )
    result = build_dataset(db, name="pytest_dataset")
    assert result["manifest_rows"] >= 1
    assert Path(result["root"]).exists()
