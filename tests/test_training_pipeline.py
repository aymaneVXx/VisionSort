import json
from pathlib import Path
from types import SimpleNamespace

from visionsort.core.enums import PipelineState
from visionsort.core.paths import ROOT_DIR
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.datasets.pipeline import compute_dataset_fingerprint
from visionsort.training.pipeline import create_training_job, training_worker_loop
import visionsort.training.pipeline as training_pipeline


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
    db.execute(
        "UPDATE datasets SET dataset_fingerprint = ? WHERE id = ?",
        (compute_dataset_fingerprint(db, "ds-train"), "ds-train"),
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
    assert metrics["test"]["status"] == "COMPLETED"
    assert metrics["test"]["frozen"] is True
    assert metrics["count_accuracy"] == 1.0
    assert metrics["merge_rate"] == 0.0
    assert metrics["artifact_sha256"]
    assert metrics["promotion_eligible"] is True

    report_path = ROOT_DIR / metrics["report_path"]
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "COMPLETED"
    assert report["candidate_model_id"] == candidate_id
    assert report["evaluation"]["evaluation_status"] == PipelineState.EVALUATED.value
    best_path = ROOT_DIR / report["outputs"]["weights_path"]
    assert best_path.name == "best.pt"
    assert best_path.exists()
    assert candidate["weights_path"] == report["outputs"]["weights_path"]


def test_training_job_refuses_non_ready_dataset(tmp_path):
    db = VisionSortDB(tmp_path / "not-ready.db")
    db.initialize()
    now = utc_now()
    db.execute(
        """
        INSERT INTO datasets
        (id, name, root_path, status, manifest_path, data_yaml_path, summary_json,
         created_at, updated_at)
        VALUES ('ds-not-ready', 'Not ready', 'data/datasets/not-ready', 'SAMPLED',
                'manifest.csv', 'data.yaml', '{}', ?, ?)
        """,
        (now, now),
    )

    try:
        create_training_job(db, "ds-not-ready", "demo_synth_det", {"mode": "demo"})
    except RuntimeError as exc:
        assert "DATASET_READY" in str(exc)
    else:
        raise AssertionError("Un dataset non prêt ne doit jamais être entraîné.")


def test_ultralytics_training_copies_real_best_pt_to_immutable_version(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "real-training.db"
    db = VisionSortDB(db_path)
    db.initialize()
    now = utc_now()
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\n", encoding="utf-8")
    db.execute(
        """
        INSERT INTO datasets
        (id, name, root_path, status, manifest_path, data_yaml_path, summary_json,
         created_at, updated_at)
        VALUES (?, ?, ?, 'DATASET_READY', ?, ?, ?, ?, ?)
        """,
        (
            "ds-real",
            "Real",
            str(tmp_path),
            str(tmp_path / "manifest.csv"),
            str(data_yaml),
            json.dumps(
                {
                    "split_integrity": {
                        "test_frozen": True,
                        "test_items": 1,
                        "frozen_test_sha256": "frozen-sha",
                    }
                }
            ),
            now,
            now,
        ),
    )
    db.execute(
        "UPDATE datasets SET dataset_fingerprint = ? WHERE id = ?",
        (compute_dataset_fingerprint(db, "ds-real"), "ds-real"),
    )
    run_dir = tmp_path / "fake-run"

    class FakeYOLO:
        def __init__(self, weights):
            self.weights = weights

        def train(self, **kwargs):
            best = run_dir / "weights" / "best.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            best.write_bytes(b"real-trained-weights")
            return SimpleNamespace(save_dir=run_dir, results_dict={})

        def val(self, **kwargs):
            return SimpleNamespace(
                results_dict={
                    "metrics/precision(B)": 0.8,
                    "metrics/recall(B)": 0.7,
                    "metrics/mAP50(B)": 0.75,
                    "metrics/mAP50-95(B)": 0.5,
                }
            )

        def predict(self, image, verbose=False):
            return []

    monkeypatch.setattr(training_pipeline, "YOLO", FakeYOLO)
    recipe = {
        "mode": "ultralytics",
        "imgsz": 320,
        "epochs": 1,
        "batch": 1,
        "device": "cpu",
    }
    job_id = create_training_job(db, "ds-real", "yolo11n_det", recipe)
    training_worker_loop(str(db_path), job_id, recipe, False)

    candidate = db.fetch_one(
        "SELECT * FROM model_registry WHERE created_from_job_id = ?", (job_id,)
    )
    assert candidate is not None
    immutable_best = ROOT_DIR / candidate["weights_path"]
    assert immutable_best.exists()
    assert immutable_best.read_bytes() == b"real-trained-weights"
    assert "versions" in immutable_best.parts
    assert run_dir not in immutable_best.parents
