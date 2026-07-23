import sys
import types
import json

import cv2
import numpy as np

from visionsort.annotations.auto import (
    DetectionAutoAnnotator,
    LocalTrackingExporter,
    MultiCameraReIDExporter,
    PoseAutoAnnotator,
    SegmentationAutoAnnotator,
)
from visionsort.annotations.quality import QualityGate
from visionsort.annotations.review import (
    export_review_cases,
    import_review_annotations,
    render_review_overlay,
)
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import ArtifactRepository


def test_task_specific_annotators_emit_yolo_formats(tmp_path):
    db = VisionSortDB(tmp_path / "annotations.db")
    db.initialize()
    row = {
        "id": "demo",
        "backend": "demo",
        "weights_path": "",
        "task": "detection",
    }
    detection = DetectionAutoAnnotator(db, row)
    segmentation = SegmentationAutoAnnotator(db, {**row, "task": "segmentation"})
    pose = PoseAutoAnnotator(db, {**row, "task": "pose"})
    names = {"parcel": 0, "person": 1}

    assert detection._label_line(
        {"class_name": "parcel", "bbox": [0, 0, 20, 10]}, names, 100, 100
    ).split() == ["0", "0.100000", "0.050000", "0.200000", "0.100000"]
    assert len(
        segmentation._label_line(
            {
                "class_name": "parcel",
                "bbox": [0, 0, 20, 10],
                "mask": [[0, 0], [20, 0], [20, 10], [0, 10]],
            },
            names,
            100,
            100,
        ).split()
    ) == 9
    pose_line = pose._label_line(
        {
            "class_name": "person",
            "bbox": [0, 0, 20, 40],
            "keypoints": [[10, 20, 0.9], [15, 25, 0.1]],
        },
        names,
        100,
        100,
    )
    assert pose_line is not None
    assert pose_line.endswith("0.150000 0.250000 1")


def test_ultralytics_model_is_loaded_once_for_multiple_images(tmp_path, monkeypatch):
    calls = {"loads": 0, "predicts": 0}

    class FakeYOLO:
        def __init__(self, checkpoint):
            calls["loads"] += 1
            self.checkpoint = checkpoint

        def predict(self, image_path, verbose=False):
            calls["predicts"] += 1
            return []

    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    db = VisionSortDB(tmp_path / "load-once.db")
    db.initialize()
    annotator = DetectionAutoAnnotator(
        db,
        {
            "id": "model",
            "backend": "ultralytics",
            "weights_path": "checkpoint.pt",
            "task": "detection",
        },
    )
    image_path = tmp_path / "image.jpg"
    cv2.imwrite(str(image_path), np.zeros((20, 20, 3), dtype=np.uint8))

    annotator.predict(source_id="src", frame_index=1, image_path=image_path)
    annotator.predict(source_id="src", frame_index=2, image_path=image_path)

    assert calls == {"loads": 1, "predicts": 2}


def test_quality_gate_uses_temporal_merge_size_and_mask_signals():
    gate = QualityGate()
    accepted, stats = gate.assess(
        source_id="src",
        detections=[
            {
                "class_name": "parcel",
                "confidence": 0.95,
                "bbox": [20, 20, 60, 60],
                "attributes": {"tracker_consistent": True},
            }
        ],
        image_shape=(100, 100),
        task="detection",
    )
    assert accepted == "AUTO_ACCEPTED"
    assert stats["temporal_stability"] == 1.0

    needs_review, stats2 = gate.assess(
        source_id="src",
        detections=[
            {
                "class_name": "parcel",
                "confidence": 0.9,
                "bbox": [0, 0, 70, 70],
                "mask": [],
                "attributes": {"model_agreement": 0.4},
            },
            {
                "class_name": "parcel",
                "confidence": 0.9,
                "bbox": [1, 1, 69, 69],
                "mask": [],
                "attributes": {"model_agreement": 0.4},
            },
        ],
        image_shape=(100, 100),
        task="segmentation",
    )
    assert needs_review == "REJECTED"
    assert stats2["probable_merge_or_split"] is True
    assert stats2["truncated_ratio"] > 0


def test_segmentation_rejects_degenerate_polygon(tmp_path):
    db = VisionSortDB(tmp_path / "polygon.db")
    db.initialize()
    annotator = SegmentationAutoAnnotator(
        db,
        {
            "id": "demo",
            "backend": "demo",
            "weights_path": "",
            "task": "segmentation",
        },
    )
    line = annotator._label_line(
        {
            "class_name": "parcel",
            "bbox": [0, 0, 20, 20],
            "mask": [[0, 0], [10, 10], [20, 20]],
        },
        {"parcel": 0},
        100,
        100,
    )
    assert line is None


def test_tracking_and_reid_exporters_emit_trainable_artifacts(tmp_path):
    items = []
    for parcel_index, parcel_id in enumerate(("parcel-a", "parcel-b")):
        for camera_index, camera_id in enumerate(("camera-1", "camera-2")):
            image_path = tmp_path / f"{parcel_id}-{camera_id}.jpg"
            image = np.zeros((80, 100, 3), dtype=np.uint8)
            image[:, :, parcel_index] = 100 + camera_index * 50
            cv2.imwrite(str(image_path), image)
            items.append(
                {
                    "id": f"{parcel_id}-{camera_id}",
                    "session_id": "session-1",
                    "source_id": camera_id,
                    "frame_index": camera_index,
                    "timestamp_global": 10.0 + camera_index,
                    "split": "train",
                    "image_path": str(image_path),
                    "metadata_json": json.dumps(
                        {
                            "observations": [
                                {
                                    "class_name": "parcel",
                                    "local_track_id": parcel_index + 1,
                                    "global_parcel_id": parcel_id,
                                    "match_result": "MATCHED",
                                    "bbox": [10, 10, 60, 50],
                                }
                            ]
                        }
                    ),
                }
            )

    tracking_path = tmp_path / "tracking.jsonl"
    tracking_rows = LocalTrackingExporter().export(items, tracking_path)
    tracking = json.loads(tracking_path.read_text(encoding="utf-8").splitlines()[0])
    assert tracking_rows == 4
    assert tracking["timestamp_global"] == 10.0
    assert tracking["track_identity"] == ["camera-1", 1]

    result = MultiCameraReIDExporter().export(
        items,
        tmp_path / "reid_manifest.jsonl",
        crops_dir=tmp_path / "crops",
    )
    assert result["status"] == "READY"
    assert result["crops"] == 4
    assert result["positive_pairs"] == 2
    assert result["negative_pairs"] > 0
    assert len(list((tmp_path / "crops").glob("*.jpg"))) == 4


def test_visual_review_overlay_export_and_reimport(tmp_path):
    image_path = tmp_path / "review.jpg"
    cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
    metadata = {
        "instance_count": 1,
        "pseudo_label_count": 1,
        "annotation_task": "detection",
        "quality_stats": {"avg_conf": 0.6},
        "annotation_provenance": {"model_id": "model-a"},
        "pseudo_labels": [
            {
                "class_name": "parcel",
                "confidence": 0.6,
                "bbox": [20, 10, 120, 60],
                "mask": [[20, 10], [120, 10], [120, 60], [20, 60]],
                "keypoints": [[30, 20, 0.9]],
            }
        ],
    }
    item = {
        "id": "item-review",
        "dataset_id": "dataset-review",
        "session_id": "session-review",
        "source_id": "camera-1",
        "split": "train",
        "image_path": str(image_path),
        "reason": "low_confidence",
        "metadata_json": json.dumps(metadata),
    }

    overlay, details = render_review_overlay(item)
    assert overlay.shape == (100, 200, 3)
    assert details["expected_count"] == details["annotated_count"] == 1
    label_studio = export_review_cases([item], export_format="label_studio")
    exported = json.loads(label_studio)
    rectangle = exported[0]["predictions"][0]["result"][0]
    assert rectangle["value"]["x"] == 10.0
    assert export_review_cases([item], export_format="cvat").startswith(
        b"<?xml"
    )

    db = VisionSortDB(tmp_path / "review.db")
    db.initialize()
    now = utc_now()
    dataset_root = tmp_path / "dataset"
    db.execute(
        """
        INSERT INTO datasets
        (id, name, root_path, status, manifest_path, data_yaml_path,
         summary_json, created_at, updated_at)
        VALUES ('dataset-review', 'Review', ?, 'REVIEW_PENDING', ?, ?, '{}', ?, ?)
        """,
        (
            str(dataset_root),
            str(dataset_root / "manifest.csv"),
            str(dataset_root / "data.yaml"),
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
        VALUES ('item-review', 'dataset-review', 'session-review', 'group-1',
                'train', 'camera-1', 'C1', 1, 1.0, ?, NULL, 'NEEDS_REVIEW',
                'low_confidence', 0.6, ?, ?)
        """,
        (str(image_path), json.dumps(metadata), now),
    )
    result = import_review_annotations(
        db,
        ArtifactRepository(db),
        dataset_id="dataset-review",
        content=label_studio,
        filename="review.json",
    )
    updated = db.fetch_one(
        "SELECT * FROM dataset_items WHERE id = 'item-review'"
    )
    assert result["updated_items"] == 1
    assert updated is not None
    assert updated["annotation_status"] == "HUMAN_VALIDATED"
    assert updated["label_path"]
