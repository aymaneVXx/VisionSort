import json

import cv2
import numpy as np
import pytest
import yaml

from visionsort.database.db import VisionSortDB, utc_now
from visionsort.core.enums import ModelStatus
from visionsort.deployment.registry import promote_model
from visionsort.datasets.integrity import DatasetIntegrityValidator
from visionsort.datasets.pipeline import (
    compute_dataset_fingerprint,
    rewrite_training_manifest,
    verify_dataset_fingerprint,
)
from visionsort.training.pipeline import create_training_job


def _label_for(task: str) -> str:
    if task == "segmentation":
        return "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
    if task == "pose":
        return (
            "0 0.5 0.5 0.6 0.8 "
            + " ".join(["0.5 0.5 2"] * 17)
            + "\n"
        )
    return "0 0.5 0.5 0.4 0.4\n"


def _create_dataset(tmp_path, task: str = "detection"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = VisionSortDB(tmp_path / f"{task}.db")
    db.initialize()
    now = utc_now()
    session_id = f"session-{task}"
    dataset_id = f"dataset-{task}"
    root = tmp_path / f"root-{task}"
    root.mkdir()
    image = root / "image.jpg"
    label = root / "label.txt"
    manifest = root / "manifest.csv"
    data_yaml = root / "data.yaml"
    cv2.imwrite(
        str(image), np.zeros((64, 64, 3), dtype=np.uint8)
    )
    label.write_text(_label_for(task), encoding="utf-8")
    data = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "task": task,
        "names": {0: "person" if task == "pose" else "parcel"},
    }
    if task == "pose":
        data["kpt_shape"] = [17, 3]
    data_yaml.write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    db.execute(
        """
        INSERT INTO capture_sessions
        (id, name, pipeline_state, demo_mode, site_validated, config_json,
         started_at, ended_at, created_at, updated_at)
        VALUES (?, ?, 'PROCESSED', 1, 0, '{}', 1.0, 2.0, ?, ?)
        """,
        (session_id, session_id, now, now),
    )
    db.execute(
        """
        INSERT INTO datasets
        (id, name, root_path, task, status, manifest_path, data_yaml_path,
         generation_config_json, summary_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'REVIEW_PENDING', ?, ?, '{"seed":17}', '{}',
                ?, ?)
        """,
        (
            dataset_id,
            dataset_id,
            str(root),
            task,
            str(manifest),
            str(data_yaml),
            now,
            now,
        ),
    )
    db.execute(
        """
        INSERT INTO dataset_sessions
        (dataset_id, session_id, split, created_at)
        VALUES (?, ?, 'train', ?)
        """,
        (dataset_id, session_id, now),
    )
    metadata = {
        "instance_count": 1,
        "observations": [
            {
                "class_name": "parcel",
                "camera_id": "camera-1",
                "local_track_id": 1,
                "global_parcel_id": "parcel-1",
            }
        ],
    }
    db.execute(
        """
        INSERT INTO dataset_items
        (id, dataset_id, session_id, sample_group_id, split, source_id,
         camera_role, frame_index, timestamp_global, image_path, label_path,
         annotation_status, reason, score, metadata_json, created_at)
        VALUES ('item-1', ?, ?, 'group-1', 'train', 'camera-1', 'C1',
                1, 1.0, ?, ?, 'HUMAN_VALIDATED', 'ready', 1.0, ?, ?)
        """,
        (
            dataset_id,
            session_id,
            str(image),
            str(label),
            json.dumps(metadata),
            now,
        ),
    )
    if task == "local_tracking":
        (root / "tracking_manifest.jsonl").write_text(
            json.dumps(
                {
                    "dataset_item_id": "item-1",
                    "camera_id": "camera-1",
                    "track_identity": ["camera-1", 1],
                }
            )
            + "\n",
            encoding="utf-8",
        )
    if task == "reid_multicamera":
        crops = []
        for index, (identity, camera) in enumerate(
            (("p1", "C1"), ("p1", "C2"), ("p2", "C1"))
        ):
            crop = root / f"crop-{index}.jpg"
            cv2.imwrite(
                str(crop), np.zeros((8, 8, 3), dtype=np.uint8)
            )
            crops.append((crop, identity, camera))
        (root / "reid_manifest.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "dataset_item_id": "item-1",
                        "global_parcel_id": identity,
                        "camera_id": camera,
                        "crop_path": str(crop),
                    }
                )
                for crop, identity, camera in crops
            )
            + "\n",
            encoding="utf-8",
        )
        (root / "reid_pairs.jsonl").write_text(
            "\n".join(
                (
                    json.dumps(
                        {
                            "left_crop": str(crops[0][0]),
                            "right_crop": str(crops[1][0]),
                            "label": 1,
                        }
                    ),
                    json.dumps(
                        {
                            "left_crop": str(crops[0][0]),
                            "right_crop": str(crops[2][0]),
                            "label": 0,
                        }
                    ),
                )
            )
            + "\n",
            encoding="utf-8",
        )
    rewrite_training_manifest(db, dataset_id, manifest)
    return {
        "db": db,
        "dataset_id": dataset_id,
        "root": root,
        "image": image,
        "label": label,
        "manifest": manifest,
        "data_yaml": data_yaml,
    }


@pytest.mark.parametrize(
    "task",
    [
        "detection",
        "segmentation",
        "pose",
        "local_tracking",
        "reid_multicamera",
    ],
)
def test_integrity_validator_accepts_complete_task_artifacts(
    tmp_path, task,
):
    fixture = _create_dataset(tmp_path, task)

    report = DatasetIntegrityValidator(
        fixture["db"], fixture["dataset_id"]
    ).validate()

    assert report["valid"] is True
    assert report["checked_items"] == 1
    assert report["split_counts"]["train"] == 1


def test_integrity_rejects_missing_image_and_label(tmp_path):
    fixture = _create_dataset(tmp_path)
    fixture["image"].unlink()
    report = DatasetIntegrityValidator(
        fixture["db"], fixture["dataset_id"]
    ).validate()
    assert not report["valid"]
    assert any("image absente" in error for error in report["errors"])

    cv2.imwrite(
        str(fixture["image"]),
        np.zeros((64, 64, 3), dtype=np.uint8),
    )
    fixture["label"].unlink()
    report = DatasetIntegrityValidator(
        fixture["db"], fixture["dataset_id"]
    ).validate()
    assert not report["valid"]
    assert any("fichier label absent" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("task", "invalid_label", "expected_error"),
    [
        ("detection", "0 0.5 0.5 -0.2 0.2\n", "bbox"),
        (
            "segmentation",
            "0 0.1 0.1 0.2 0.2 0.3 0.3\n",
            "aire du polygone nulle",
        ),
        (
            "pose",
            "0 0.5 0.5 0.4 0.4 "
            + " ".join(["0.5 0.5 2"] * 16)
            + "\n",
            "kpt_shape",
        ),
    ],
)
def test_integrity_rejects_invalid_task_labels(
    tmp_path, task, invalid_label, expected_error
):
    fixture = _create_dataset(tmp_path, task)
    fixture["label"].write_text(invalid_label, encoding="utf-8")

    report = DatasetIntegrityValidator(
        fixture["db"], fixture["dataset_id"]
    ).validate()

    assert not report["valid"]
    assert any(
        expected_error in error for error in report["errors"]
    )


def test_integrity_rejects_empty_label_manifest_and_yaml_corruption(
    tmp_path,
):
    fixture = _create_dataset(tmp_path)
    fixture["label"].write_text("", encoding="utf-8")
    report = DatasetIntegrityValidator(
        fixture["db"], fixture["dataset_id"]
    ).validate()
    assert any("label est vide" in error for error in report["errors"])

    fixture = _create_dataset(tmp_path / "manifest-case")
    fixture["manifest"].write_text(
        "bad,column\nvalue,value\n", encoding="utf-8"
    )
    report = DatasetIntegrityValidator(
        fixture["db"], fixture["dataset_id"]
    ).validate()
    assert any("colonnes absentes" in error for error in report["errors"])

    fixture = _create_dataset(tmp_path / "yaml-case")
    fixture["data_yaml"].write_text("names: [parcel]\n", encoding="utf-8")
    report = DatasetIntegrityValidator(
        fixture["db"], fixture["dataset_id"]
    ).validate()
    assert any("train" in error for error in report["errors"])


def test_fingerprint_is_strict_and_detects_tampering(tmp_path):
    fixture = _create_dataset(tmp_path)
    db = fixture["db"]
    dataset_id = fixture["dataset_id"]
    fingerprint = compute_dataset_fingerprint(db, dataset_id)
    db.execute(
        """
        UPDATE datasets SET status = 'DATASET_READY',
                            dataset_fingerprint = ?
        WHERE id = ?
        """,
        (fingerprint, dataset_id),
    )
    assert verify_dataset_fingerprint(db, dataset_id)["valid"]

    fixture["label"].write_text(
        "0 0.5 0.5 0.3 0.3\n", encoding="utf-8"
    )
    assert not verify_dataset_fingerprint(db, dataset_id)["valid"]

    fixture = _create_dataset(tmp_path / "missing-case")
    fixture["image"].unlink()
    with pytest.raises(RuntimeError, match="Fingerprint refusé"):
        compute_dataset_fingerprint(
            fixture["db"], fixture["dataset_id"]
        )


def test_training_and_promotion_refuse_tampered_dataset(tmp_path):
    fixture = _create_dataset(tmp_path)
    db = fixture["db"]
    dataset_id = fixture["dataset_id"]
    fingerprint = compute_dataset_fingerprint(db, dataset_id)
    db.execute(
        """
        UPDATE datasets SET status = 'DATASET_READY',
                            dataset_fingerprint = ?
        WHERE id = ?
        """,
        (fingerprint, dataset_id),
    )
    fixture["label"].write_text(
        "0 0.5 0.5 0.3 0.3\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="fingerprint"):
        create_training_job(
            db,
            dataset_id,
            "demo_synth_det",
            {"mode": "demo"},
        )

    now = utc_now()
    db.execute(
        """
        INSERT INTO training_jobs
        (id, dataset_id, model_id, status, recipe_json, log_path,
         metrics_json, error_text, created_at, updated_at)
        VALUES ('job-tampered', ?, 'demo_synth_det', 'COMPLETED', '{}',
                'training.log', '{}', NULL, ?, ?)
        """,
        (dataset_id, now, now),
    )
    metrics = {
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
    db.execute(
        """
        INSERT INTO model_registry
        (id, name, task, backend, weights_path, status, is_active,
         notes_json, metrics_json, parent_model_id, created_from_job_id,
         created_at, updated_at)
        VALUES ('candidate-tampered', 'Candidate', 'detection', 'demo', '',
                ?, 0, '{}', ?, 'demo_synth_det', 'job-tampered', ?, ?)
        """,
        (
            ModelStatus.CANDIDATE.value,
            json.dumps(metrics),
            now,
            now,
        ),
    )
    with pytest.raises(RuntimeError, match="fingerprint"):
        promote_model(db, "candidate-tampered")
