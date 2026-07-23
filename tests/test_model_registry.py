import json

from visionsort.core.enums import ModelStatus, PipelineState
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.deployment.registry import promote_model, rollback_to_previous_active, set_model_status


def test_promote_and_rollback_model_registry(tmp_path):
    db = VisionSortDB(tmp_path / "visionsort.db")
    db.initialize()
    now = utc_now()

    db.execute(
        """
        INSERT INTO capture_sessions (id, name, pipeline_state, demo_mode, site_validated, config_json, report_path, started_at, ended_at, created_at, updated_at)
        VALUES (?, ?, ?, 1, 0, '{}', NULL, NULL, NULL, ?, ?)
        """,
        ("session-a", "Session A", PipelineState.CANDIDATE.value, now, now),
    )
    db.execute(
        """
        INSERT INTO datasets (id, name, root_path, status, manifest_path, data_yaml_path, summary_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("ds-a", "Dataset A", "data/datasets/ds-a", "DATASET_READY", "manifest.csv", "data.yaml", '{"session_id":"session-a"}', now, now),
    )
    db.execute(
        """
        INSERT INTO training_jobs (id, dataset_id, model_id, status, recipe_json, log_path, metrics_json, error_text, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
        """,
        ("job-a", "ds-a", "demo_synth_det", "COMPLETED", "{}", "logs/train.log", "{}", now, now),
    )
    db.execute(
        """
        INSERT INTO model_registry
        (id, name, task, backend, weights_path, status, is_active, notes_json, metrics_json, parent_model_id, created_from_job_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "archived-base",
            "Archived Base",
            "detection",
            "demo",
            "",
            ModelStatus.ARCHIVED.value,
            0,
            "{}",
            "{}",
            None,
            None,
            now,
            now,
        ),
    )
    db.execute(
        """
        INSERT INTO model_registry
        (id, name, task, backend, weights_path, status, is_active, notes_json, metrics_json, parent_model_id, created_from_job_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "candidate-a",
            "Candidate A",
            "detection",
            "demo",
            "",
            ModelStatus.CANDIDATE.value,
            0,
            "{}",
            json.dumps(
                {
                    "precision": 0.9,
                    "promotion_eligible": True,
                    "promotion_criteria": {"precision_min": 0.5},
                    "test": {"status": "COMPLETED", "frozen": True},
                }
            ),
            "demo_synth_det",
            "job-a",
            now,
            now,
        ),
    )

    promote_model(db, "candidate-a")

    promoted = db.fetch_one("SELECT status, is_active FROM model_registry WHERE id = ?", ("candidate-a",))
    session = db.fetch_one("SELECT pipeline_state, last_candidate_model_id FROM capture_sessions WHERE id = ?", ("session-a",))
    assert promoted is not None
    assert session is not None
    assert promoted["status"] == ModelStatus.CHAMPION.value
    assert promoted["is_active"] == 1
    assert session["pipeline_state"] == PipelineState.DEPLOYED.value
    assert session["last_candidate_model_id"] == "candidate-a"

    set_model_status(db, "candidate-a", ModelStatus.REJECTED.value)
    rejected = db.fetch_one("SELECT status, is_active FROM model_registry WHERE id = ?", ("candidate-a",))
    assert rejected is not None
    assert rejected["status"] == ModelStatus.REJECTED.value
    assert rejected["is_active"] == 0

    rolled_back = rollback_to_previous_active(db)
    archived = db.fetch_one("SELECT is_active FROM model_registry WHERE id = ?", ("archived-base",))
    assert rolled_back == "archived-base"
    assert archived is not None
    assert archived["is_active"] == 1


def test_promotion_refuses_candidate_without_frozen_test_and_criteria(tmp_path):
    db = VisionSortDB(tmp_path / "guard.db")
    db.initialize()
    now = utc_now()
    db.execute(
        """
        INSERT INTO model_registry
        (id, name, task, backend, weights_path, status, is_active, notes_json,
         metrics_json, parent_model_id, created_from_job_id, created_at, updated_at)
        VALUES ('unsafe-candidate', 'Unsafe', 'detection', 'demo', '', 'CANDIDATE',
                0, '{}', '{"precision":0.99}', NULL, NULL, ?, ?)
        """,
        (now, now),
    )

    try:
        promote_model(db, "unsafe-candidate")
    except RuntimeError as exc:
        assert "test figé" in str(exc)
    else:
        raise AssertionError("La promotion aurait dû être refusée.")
