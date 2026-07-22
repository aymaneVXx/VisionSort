import json

from visionsort.database.db import VisionSortDB, utc_now
from visionsort.runtime.pipeline_worker import pipeline_worker_loop


def test_auto_annotate_refuses_champion_without_explicit_override(tmp_path):
    db = VisionSortDB(tmp_path / "visionsort.db")
    db.initialize()
    now = utc_now()

    db.execute(
        """
        INSERT INTO capture_sessions (id, name, pipeline_state, demo_mode, site_validated, config_json, report_path, last_dataset_id, started_at, ended_at, created_at, updated_at)
        VALUES (?, ?, 'SAMPLED', 1, 0, '{}', NULL, ?, NULL, NULL, ?, ?)
        """,
        ("session-guard", "Session Guard", "dataset-guard", now, now),
    )
    db.execute(
        """
        INSERT INTO datasets (id, name, root_path, status, manifest_path, data_yaml_path, summary_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "dataset-guard",
            "Dataset Guard",
            "data/datasets/dataset-guard",
            "SAMPLED",
            "data/datasets/dataset-guard/manifest.csv",
            "data/datasets/dataset-guard/data.yaml",
            json.dumps({"session_id": "session-guard"}),
            now,
            now,
        ),
    )
    db.execute(
        """
        INSERT INTO dataset_items (id, dataset_id, session_id, sample_group_id, split, source_id, camera_role, frame_index, timestamp_global, image_path, label_path, annotation_status, reason, score, metadata_json, created_at)
        VALUES (?, ?, ?, NULL, 'train', 'src1', 'C1', 1, 1.0, 'missing.jpg', NULL, 'NEEDS_REVIEW', 'manual', 0.5, '{}', ?)
        """,
        ("item-guard", "dataset-guard", "session-guard", now),
    )
    db.execute(
        """
        UPDATE model_registry SET status = ?, is_active = 1 WHERE id = ?
        """,
        ("CHAMPION", "demo_synth_det"),
    )

    pipeline_worker_loop(
        str(tmp_path / "visionsort.db"),
        "session-guard",
        "AUTO_ANNOTATE",
        {"dataset_id": "dataset-guard", "model_id": "demo_synth_det"},
    )
    step = db.fetch_one(
        "SELECT * FROM pipeline_step_runs WHERE session_id = ? AND step = ? ORDER BY created_at DESC LIMIT 1",
        ("session-guard", "AUTO_ANNOTATE"),
    )
    assert step is not None
    assert step["status"] == "FAILED"
    assert "CHAMPION" in str(step["error_text"])
