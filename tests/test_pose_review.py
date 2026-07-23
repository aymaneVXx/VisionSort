import json

import cv2
import numpy as np
import pytest
import yaml

from visionsort.annotations.review import (
    export_review_cases,
    import_review_annotations,
)
from visionsort.annotations.validators import (
    COCO_KEYPOINT_NAMES,
    PoseLabelValidator,
)
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import ArtifactRepository


def _pose_item(item_id: str, image_path) -> dict:
    keypoints = [
        [20.0 + index * 5.0, 20.0 + index * 2.0, 0.9]
        for index in range(17)
    ]
    return {
        "id": item_id,
        "dataset_id": "pose-dataset",
        "session_id": "pose-session",
        "source_id": "camera-pose",
        "split": "train",
        "image_path": str(image_path),
        "reason": "pose_review",
        "metadata_json": json.dumps(
            {
                "instance_count": 1,
                "pseudo_label_count": 1,
                "annotation_task": "pose",
                "pseudo_labels": [
                    {
                        "class_name": "person",
                        "confidence": 0.9,
                        "bbox": [10.0, 10.0, 120.0, 90.0],
                        "keypoints": keypoints,
                    }
                ],
            }
        ),
    }


def _pose_database(tmp_path, item: dict) -> tuple[VisionSortDB, object]:
    db = VisionSortDB(tmp_path / f"{item['id']}.db")
    db.initialize()
    dataset_root = tmp_path / f"dataset-{item['id']}"
    data_yaml_path = dataset_root / "data.yaml"
    dataset_root.mkdir()
    data_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "task": "pose",
                "names": {0: "person"},
                "kpt_shape": [17, 3],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    now = utc_now()
    db.execute(
        """
        INSERT INTO datasets
        (id, name, root_path, task, status, manifest_path, data_yaml_path,
         summary_json, created_at, updated_at)
        VALUES ('pose-dataset', 'Pose', ?, 'pose', 'REVIEW_PENDING',
                ?, ?, '{}', ?, ?)
        """,
        (
            str(dataset_root),
            str(dataset_root / "manifest.csv"),
            str(data_yaml_path),
            now,
            now,
        ),
    )
    db.execute(
        """
        INSERT INTO dataset_items
        (id, dataset_id, session_id, sample_group_id, split, source_id,
         camera_role, frame_index, timestamp_global, image_path, label_path,
         annotation_status, reason, score, metadata_json, created_at)
        VALUES (?, 'pose-dataset', NULL, 'group-pose', 'train',
                'camera-pose', 'C2', 1, 1.0, ?, NULL, 'NEEDS_REVIEW',
                'pose_review', 0.9, ?, ?)
        """,
        (
            item["id"],
            item["image_path"],
            item["metadata_json"],
            now,
        ),
    )
    return db, data_yaml_path


def test_label_studio_pose_export_and_import_preserve_coco_keypoints(
    tmp_path,
):
    image_path = tmp_path / "pose-ls.jpg"
    cv2.imwrite(
        str(image_path), np.zeros((100, 200, 3), dtype=np.uint8)
    )
    item = _pose_item("pose-ls", image_path)
    exported_content = export_review_cases(
        [item], export_format="label_studio"
    )
    exported = json.loads(exported_content)
    results = exported[0]["predictions"][0]["result"]
    keypoints = [
        result
        for result in results
        if result["type"] == "keypointlabels"
    ]

    assert len(keypoints) == 17
    assert [
        result["value"]["keypointlabels"][0] for result in keypoints
    ] == list(COCO_KEYPOINT_NAMES)
    assert all(result["original_width"] == 200 for result in keypoints)
    assert all(result["original_height"] == 100 for result in keypoints)

    db, data_yaml_path = _pose_database(tmp_path, item)
    imported = import_review_annotations(
        db,
        ArtifactRepository(db),
        dataset_id="pose-dataset",
        content=exported_content,
        filename="pose.json",
    )
    updated = db.fetch_one(
        "SELECT * FROM dataset_items WHERE id = 'pose-ls'"
    )
    assert imported["updated_items"] == 1
    assert updated["annotation_status"] == "HUMAN_VALIDATED"
    label_content = (
        tmp_path / "dataset-pose-ls" / "labels" / "train" / "pose-ls.txt"
    ).read_text(encoding="utf-8")
    assert len(label_content.split()) == 56
    assert PoseLabelValidator(data_yaml_path).validate_content(
        label_content, expected_instances=1
    ).valid


def test_cvat_pose_export_and_import_preserve_all_points(tmp_path):
    image_path = tmp_path / "pose-cvat.jpg"
    cv2.imwrite(
        str(image_path), np.zeros((100, 200, 3), dtype=np.uint8)
    )
    item = _pose_item("pose-cvat", image_path)
    exported = export_review_cases([item], export_format="cvat")

    assert exported.count(b"<points ") == 17
    assert b"<kpt_shape>17,3</kpt_shape>" in exported

    db, data_yaml_path = _pose_database(tmp_path, item)
    result = import_review_annotations(
        db,
        ArtifactRepository(db),
        dataset_id="pose-dataset",
        content=exported,
        filename="pose.xml",
    )
    updated = db.fetch_one(
        "SELECT * FROM dataset_items WHERE id = 'pose-cvat'"
    )
    assert result["updated_items"] == 1
    assert updated["annotation_status"] == "HUMAN_VALIDATED"
    assert PoseLabelValidator(data_yaml_path).validate(
        updated["label_path"], expected_instances=1
    ).valid


def test_incomplete_pose_import_and_empty_label_are_never_validated(
    tmp_path,
):
    image_path = tmp_path / "pose-invalid.jpg"
    cv2.imwrite(
        str(image_path), np.zeros((100, 200, 3), dtype=np.uint8)
    )
    item = _pose_item("pose-invalid", image_path)
    exported = json.loads(
        export_review_cases([item], export_format="label_studio")
    )
    results = exported[0]["predictions"][0]["result"]
    exported[0]["predictions"][0]["result"] = [
        result
        for result in results
        if result.get("id") != "keypoint-0-16"
    ]
    db, data_yaml_path = _pose_database(tmp_path, item)
    repository = ArtifactRepository(db)

    with pytest.raises(RuntimeError, match="16 keypoint"):
        import_review_annotations(
            db,
            repository,
            dataset_id="pose-dataset",
            content=json.dumps(exported).encode("utf-8"),
            filename="incomplete.json",
        )
    unchanged = db.fetch_one(
        "SELECT * FROM dataset_items WHERE id = 'pose-invalid'"
    )
    assert unchanged["annotation_status"] == "NEEDS_REVIEW"
    assert unchanged["label_path"] is None

    empty_label = tmp_path / "empty.txt"
    empty_label.write_text("", encoding="utf-8")
    db.execute(
        "UPDATE dataset_items SET label_path = ? WHERE id = ?",
        (str(empty_label), "pose-invalid"),
    )
    with pytest.raises(RuntimeError, match="fichier label Pose est vide"):
        repository.update_dataset_item(
            "pose-invalid", annotation_status="HUMAN_VALIDATED"
        )
    assert not PoseLabelValidator(data_yaml_path).validate(
        empty_label
    ).valid


def test_pose_validator_rejects_wrong_kpt_shape(tmp_path):
    data_yaml = tmp_path / "bad-data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {"names": {0: "person"}, "kpt_shape": [16, 3]}
        ),
        encoding="utf-8",
    )
    label_path = tmp_path / "pose.txt"
    label_path.write_text(
        "0 0.5 0.5 0.2 0.2 "
        + " ".join(["0.5 0.5 2"] * 17)
        + "\n",
        encoding="utf-8",
    )

    report = PoseLabelValidator(data_yaml).validate(label_path)

    assert not report.valid
    assert any("kpt_shape" in error for error in report.errors)
