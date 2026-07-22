from __future__ import annotations

import csv
import hashlib
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


def _stable_split_key(session_id: str, sample_group_id: str) -> str:
    digest = hashlib.sha256(f"{session_id}:{sample_group_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    return "train" if bucket < 7 else "val" if bucket < 9 else "test"


def _ahash64(image) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = (small > mean).astype("uint8").flatten().tolist()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return int(value)


def _min_hamming(value: int, seen: list[int]) -> int:
    if not seen:
        return 64
    return min(int((value ^ other).bit_count()) for other in seen)


def build_dataset(db: VisionSortDB, *, session_id: str, name: str = "autodataset") -> dict[str, Any]:
    artifact_repo = ArtifactRepository(db)
    dataset_id = f"dataset-{uuid.uuid4().hex[:8]}"
    root = DATASETS_DIR / dataset_id
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    sources_by_id = {row["id"]: dict(row) for row in db.fetch_all("SELECT * FROM sources")}
    tracklets = [
        dict(row)
        for row in db.fetch_all(
            "SELECT * FROM tracklets WHERE session_id = ? ORDER BY ended_at_global ASC",
            (session_id,),
        )
    ]
    manifest_rows: list[dict[str, Any]] = []
    seen_hashes: list[int] = []
    dedup_skipped = 0

    for index, tracklet in enumerate(tracklets):
        obs_path = Path(tracklet["observation_path"])
        if not obs_path.is_absolute():
            from visionsort.core.paths import ROOT_DIR

            obs_path = ROOT_DIR / obs_path
        if not obs_path.exists():
            continue
        observations = _load_track_observations(obs_path)
        if not observations:
            continue
        reason, score = _select_reason(observations)
        middle = observations[len(observations) // 2]
        source = sources_by_id.get(tracklet.get("source_id") or tracklet.get("camera_id"))
        if source is None:
            continue
        capture = cv2.VideoCapture(source["uri"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(middle["frame_index"]))
        ok, image = capture.read()
        capture.release()
        if not ok:
            continue
        image_hash = _ahash64(image)
        if _min_hamming(image_hash, seen_hashes) <= 3:
            dedup_skipped += 1
            continue
        seen_hashes.append(image_hash)
        sample_group_id = f"{session_id}-{int(float(middle.get('timestamp_global', tracklet.get('ended_at_global', 0.0))) * 10):012d}"
        split = _stable_split_key(session_id, sample_group_id)
        image_name = f"{tracklet['tracklet_id']}.jpg"
        image_path = root / "images" / split / image_name
        cv2.imwrite(str(image_path), image)
        annotation_status = AnnotationStatus.NEEDS_REVIEW.value
        artifact_repo.add_dataset_item(
            dataset_id=dataset_id,
            session_id=session_id,
            sample_group_id=sample_group_id,
            split=split,
            source_id=tracklet.get("source_id"),
            camera_role=tracklet.get("camera_role"),
            frame_index=int(middle.get("frame_index", 0)),
            timestamp_global=float(middle.get("timestamp_global", tracklet.get("ended_at_global", 0.0))),
            image_path=relative_to_root(image_path),
            label_path=None,
            annotation_status=annotation_status,
            reason=reason,
            score=score,
            metadata={
                "tracklet_id": tracklet["tracklet_id"],
                "session_id": session_id,
                "source_id": tracklet.get("source_id"),
                "camera_role": tracklet.get("camera_role"),
                "frame_index": int(middle.get("frame_index", 0)),
                "timestamp_global": float(middle.get("timestamp_global", tracklet.get("ended_at_global", 0.0))),
                "model_id": tracklet.get("model_id"),
                "tracker_id": tracklet.get("tracker_id"),
                "validated_on_site": False,
                "image_ahash64": f"{image_hash:016x}",
            },
        )
        manifest_rows.append(
            {
                "split": split,
                "image_path": relative_to_root(image_path),
                "label_path": "",
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
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    data_yaml_path = root / "data.yaml"
    data_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": relative_to_root(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "parcel", 1: "person", 2: "left_wrist", 3: "right_wrist"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    summary = {
        "items": len(manifest_rows),
        "dedup_skipped": dedup_skipped,
        "validated_on_site": False,
        "selection_strategy": "tracklet_jsonl_from_replay",
        "session_id": session_id,
        "source_session_id": session_id,
        "dataset_version": dataset_id,
        "parent_dataset_id": None,
        "manifest_sha256": manifest_sha256,
        "sampling_code_version": "v2_stable_session_split",
    }
    artifact_repo.upsert_dataset(
        dataset_id=dataset_id,
        name=name,
        root_path=relative_to_root(root),
        status="SAMPLED",
        manifest_path=relative_to_root(manifest_path),
        data_yaml_path=relative_to_root(data_yaml_path),
        summary=summary,
    )
    return {"dataset_id": dataset_id, "root": str(root), "manifest_rows": len(manifest_rows)}
