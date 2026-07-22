from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Any

import cv2
import yaml

from visionsort.core.config import relative_to_root
from visionsort.core.enums import AnnotationStatus
from visionsort.core.paths import DATASETS_DIR
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import ArtifactRepository
from visionsort.tracking.engine import bbox_iou


def _load_track_observations(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select_reason(observations: list[dict[str, Any]]) -> tuple[str, float]:
    confidences = [float(obs["confidence"]) for obs in observations]
    avg_conf = sum(confidences) / max(len(confidences), 1)
    overlaps = 0.0
    for idx in range(len(observations) - 1):
        overlaps = max(overlaps, bbox_iou(tuple(observations[idx]["bbox"]), tuple(observations[idx + 1]["bbox"])))
    short_track = len(observations) < 5
    if avg_conf < 0.65:
        return "low_confidence", 1.0 - avg_conf
    if overlaps > 0.65:
        return "adjacent_overlap", overlaps
    if short_track:
        return "lost_track", 0.75
    return "control_example", 0.20


def build_dataset(db: VisionSortDB, name: str = "autodataset") -> dict[str, Any]:
    artifact_repo = ArtifactRepository(db)
    dataset_id = f"dataset-{uuid.uuid4().hex[:8]}"
    root = DATASETS_DIR / dataset_id
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    source_by_role = {row["role"]: dict(row) for row in db.fetch_all("SELECT * FROM sources")}
    tracklets = [dict(row) for row in db.fetch_all("SELECT * FROM tracklets ORDER BY ended_at ASC")]
    manifest_rows: list[dict[str, Any]] = []

    for index, tracklet in enumerate(tracklets):
        obs_path = Path(tracklet["observation_path"])
        if not obs_path.exists():
            continue
        observations = _load_track_observations(obs_path)
        if not observations:
            continue
        reason, score = _select_reason(observations)
        middle = observations[len(observations) // 2]
        source = source_by_role.get(tracklet["camera_id"])
        if source is None:
            continue
        capture = cv2.VideoCapture(source["uri"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(middle["frame_index"]))
        ok, image = capture.read()
        capture.release()
        if not ok:
            continue
        split = "train" if index % 10 < 7 else "val" if index % 10 < 9 else "test"
        image_name = f"{tracklet['tracklet_id']}.jpg"
        label_name = f"{tracklet['tracklet_id']}.txt"
        image_path = root / "images" / split / image_name
        label_path = root / "labels" / split / label_name
        h, w = image.shape[:2]
        x1, y1, x2, y2 = middle["bbox"]
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h
        cv2.imwrite(str(image_path), image)
        label_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
        annotation_status = AnnotationStatus.NEEDS_REVIEW.value if score >= 0.5 else AnnotationStatus.AUTO_ACCEPTED.value
        artifact_repo.add_dataset_item(
            dataset_id=dataset_id,
            image_path=relative_to_root(image_path),
            label_path=relative_to_root(label_path),
            annotation_status=annotation_status,
            reason=reason,
            score=score,
            metadata={"tracklet_id": tracklet["tracklet_id"], "camera_id": tracklet["camera_id"], "validated_on_site": False},
        )
        manifest_rows.append(
            {
                "split": split,
                "image_path": relative_to_root(image_path),
                "label_path": relative_to_root(label_path),
                "annotation_status": annotation_status,
                "reason": reason,
                "score": score,
            }
        )

    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image_path", "label_path", "annotation_status", "reason", "score"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    data_yaml_path = root / "data.yaml"
    data_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": relative_to_root(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "parcel"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    summary = {"items": len(manifest_rows), "validated_on_site": False, "selection_strategy": "tracklet_jsonl_from_replay"}
    artifact_repo.upsert_dataset(
        dataset_id=dataset_id,
        name=name,
        root_path=relative_to_root(root),
        status="READY",
        manifest_path=relative_to_root(manifest_path),
        data_yaml_path=relative_to_root(data_yaml_path),
        summary=summary,
    )
    return {"dataset_id": dataset_id, "root": str(root), "manifest_rows": len(manifest_rows)}
