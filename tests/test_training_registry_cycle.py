from visionsort.core.enums import ModelStatus, PipelineState
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.datasets.pipeline import compute_dataset_fingerprint
from visionsort.deployment.registry import promote_model, rollback_to_previous_active
from visionsort.training.pipeline import create_training_job, training_worker_loop


def test_training_then_promote_then_rollback_cycle(tmp_path):
    db_path = tmp_path / "visionsort.db"
    db = VisionSortDB(db_path)
    db.initialize()
    now = utc_now()

    db.execute(
        """
        INSERT INTO capture_sessions (id, name, pipeline_state, demo_mode, site_validated, config_json, report_path, started_at, ended_at, created_at, updated_at)
        VALUES (?, ?, ?, 1, 0, '{}', NULL, NULL, NULL, ?, ?)
        """,
        ("session-cycle", "Session Cycle", PipelineState.DATASET_READY.value, now, now),
    )
    db.execute(
        """
        INSERT INTO datasets (id, name, root_path, status, manifest_path, data_yaml_path, summary_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "ds-cycle",
            "Dataset Cycle",
            "data/datasets/ds-cycle",
            "DATASET_READY",
            "data/datasets/ds-cycle/manifest.csv",
            "data/datasets/ds-cycle/data.yaml",
            '{"session_id":"session-cycle"}',
            now,
            now,
        ),
    )
    db.execute(
        "UPDATE datasets SET dataset_fingerprint = ? WHERE id = ?",
        (compute_dataset_fingerprint(db, "ds-cycle"), "ds-cycle"),
    )

    recipe = {
        "dataset_id": "ds-cycle",
        "model_id": "demo_synth_det",
        "task": "detection",
        "architecture": "yolo11n",
        "imgsz": 640,
        "epochs": 1,
        "batch": 2,
        "device": "cpu",
        "patience": 1,
        "mode": "demo",
    }
    job_id = create_training_job(db, "ds-cycle", "demo_synth_det", recipe)
    training_worker_loop(str(db_path), job_id, recipe, True)

    candidate_id = f"candidate-{job_id[:8]}"
    candidate = db.fetch_one("SELECT * FROM model_registry WHERE id = ?", (candidate_id,))
    assert candidate is not None
    assert candidate["status"] == ModelStatus.CANDIDATE.value

    promote_model(db, candidate_id)
    promoted = db.fetch_one("SELECT * FROM model_registry WHERE id = ?", (candidate_id,))
    session = db.fetch_one("SELECT * FROM capture_sessions WHERE id = ?", ("session-cycle",))
    assert promoted is not None
    assert session is not None
    assert promoted["status"] == ModelStatus.CHAMPION.value
    assert promoted["is_active"] == 1
    assert session["pipeline_state"] == PipelineState.DEPLOYED.value

    rolled_back_id = rollback_to_previous_active(db)
    rolled_back = db.fetch_one("SELECT * FROM model_registry WHERE id = ?", (rolled_back_id,))
    assert rolled_back_id is not None
    assert rolled_back is not None
    assert rolled_back["is_active"] == 1
