import sys
import types

import cv2
import numpy as np

from visionsort.annotations.auto import (
    DetectionAutoAnnotator,
    PoseAutoAnnotator,
    SegmentationAutoAnnotator,
)
from visionsort.annotations.quality import QualityGate
from visionsort.database.db import VisionSortDB


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
