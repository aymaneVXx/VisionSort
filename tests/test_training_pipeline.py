import json
from pathlib import Path

from visionsort.core.enums import PipelineState
from visionsort.core.paths import ROOT_DIR
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.training.pipeline import create_training_job, training_worker_loop


def test_training_worker_demo_updates_session_and_report(tmp_path):
    db_path = tmp_path / "visionsort.db"
    db = VisionSortDB(db_path)
    db.initialize()
    now = utc_now()

    db.execute(
        """
        INSERT INTO capture_sessions (id, name, pipeline_state, demo_mode, site_validated, config_json, report_path, started_at, ended_at, created_at, updated_at)
        VALUES (?, ?, ?, 1, 0, '{}', NULL, NULL, NULL, ?, ?)
        """,
        ("session-train", "Session Train", PipelineState.DATASET_READY.value, now, now),
    )
    db.execute(
        """
        INSERT INTO datasets (id, name, root_path, status, manifest_path, data_yaml_path, summary_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "ds-train",
            "Dataset Train",
            "data/datasets/ds-train",
            "DATASET_READY",
            "data/datasets/ds-train/manifest.csv",
            "data/datasets/ds-train/data.yaml",
            json.dumps({"session_id": "session-train"}),
            now,
            now,
        ),
    )

    recipe = {
        "dataset_id": "ds-train",
        "model_id": "demo_synth_det",
        "task": "detection",
        "architecture": "yolo11n",
        "imgsz": 640,
        "epochs": 2,
        "batch": 2,
        "device": "cpu",
        "patience": 2,
        "mode": "demo",
    }
    job_id = create_training_job(db, "ds-train", "demo_synth_det", recipe)

    training_worker_loop(str(db_path), job_id, recipe, True)

    job = db.fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    session = db.fetch_one("SELECT * FROM capture_sessions WHERE id = ?", ("session-train",))
    candidate_id = f"candidate-{job_id[:8]}"
    candidate = db.fetch_one("SELECT * FROM model_registry WHERE id = ?", (candidate_id,))

    assert job is not None
    assert session is not None
    assert candidate is not None
    assert job["status"] == "COMPLETED"
    assert session["pipeline_state"] == PipelineState.CANDIDATE.value
    assert session["last_candidate_model_id"] == candidate_id

    metrics = json.loads(job["metrics_json"] or "{}")
    assert metrics["evaluation_status"] == PipelineState.EVALUATED.value
    assert metrics["candidate_status"] == PipelineState.CANDIDATE.value
    assert metrics["candidate_model_id"] == candidate_id
    assert "comparison" in metrics
    assert "benchmark" in metrics
    assert metrics["report_path"]

    report_path = ROOT_DIR / metrics["report_path"]
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "COMPLETED"
    assert report["candidate_model_id"] == candidate_id
    assert report["evaluation"]["evaluation_status"] == PipelineState.EVALUATED.value
