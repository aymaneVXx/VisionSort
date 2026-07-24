import json

import cv2
import numpy as np

from visionsort.core.enums import ModelStatus, PipelineState
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.deployment.registry import promote_model, rollback_to_previous_active, set_model_status
from visionsort.datasets.pipeline import (
    compute_dataset_fingerprint,
    rewrite_training_manifest,
)


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
    root = tmp_path / "dataset-a"
    root.mkdir()
    image = root / "image.jpg"
    label = root / "label.txt"
    manifest = root / "manifest.csv"
    data_yaml = root / "data.yaml"
    cv2.imwrite(
        str(image), np.zeros((32, 32, 3), dtype=np.uint8)
    )
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        "test: images/test\ntask: detection\nnames:\n  0: parcel\n",
        encoding="utf-8",
    )
    db.execute(
        """
        UPDATE datasets SET root_path = ?, manifest_path = ?,
                            data_yaml_path = ?
        WHERE id = 'ds-a'
        """,
        (str(root), str(manifest), str(data_yaml)),
    )
    db.execute(
        """
        INSERT INTO dataset_sessions
        (dataset_id, session_id, split, created_at)
        VALUES ('ds-a', 'session-a', 'train', ?)
        """,
        (now,),
    )
    db.execute(
        """
        INSERT INTO dataset_items
        (id, dataset_id, session_id, sample_group_id, split, source_id,
         camera_role, frame_index, timestamp_global, image_path, label_path,
         annotation_status, reason, score, metadata_json, created_at)
        VALUES ('item-a', 'ds-a', 'session-a', 'group-a', 'train',
                'source-a', 'C1', 1, 1.0, ?, ?, 'HUMAN_VALIDATED',
                'ready', 1.0, '{"instance_count":1}', ?)
        """,
        (str(image), str(label), now),
    )
    rewrite_training_manifest(db, "ds-a", manifest)
    db.execute(
        """
        UPDATE datasets SET dataset_fingerprint = ?
        WHERE id = 'ds-a'
        """,
        (compute_dataset_fingerprint(db, "ds-a"),),
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
                    "recall": 0.9,
                    "mAP50": 0.9,
                    "count_accuracy": 0.9,
                    "merge_rate": 0.01,
                    "fps": 20.0,
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
        assert "Promotion refusée" in str(exc)
    else:
        raise AssertionError("La promotion aurait dû être refusée.")
