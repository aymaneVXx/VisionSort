from __future__ import annotations

import csv
import hashlib
import json
import statistics
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml

from visionsort.core.config import relative_to_root
from visionsort.core.enums import AnnotationStatus
from visionsort.core.paths import DATASETS_DIR, ROOT_DIR
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import ArtifactRepository
from visionsort.tracking.engine import bbox_iou


def _load_track_observations(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_session_split(session_id: str) -> str:
    """Assign an entire capture session to one immutable split."""
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    return "train" if bucket < 7 else "val" if bucket < 9 else "test"


def _ahash64(image) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = (small > mean).astype("uint8").flatten().tolist()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return int(value)


def _min_hamming(value: int, seen: list[int]) -> int:
    if not seen:
        return 64
    return min(int((value ^ other).bit_count()) for other in seen)


def _frame_signals(
    observations: list[dict[str, Any]],
    *,
    previous_count: int | None,
    ambiguous: bool,
    track_endpoint: bool,
) -> tuple[str | None, float, dict[str, Any]]:
    confidences = [float(item.get("confidence", 0.0)) for item in observations]
    max_iou = 0.0
    for left in range(len(observations)):
        for right in range(left + 1, len(observations)):
            if (
                observations[left].get("camera_id"),
                observations[left].get("local_track_id"),
            ) == (
                observations[right].get("camera_id"),
                observations[right].get("local_track_id"),
            ):
                continue
            max_iou = max(
                max_iou,
                bbox_iou(
                    tuple(observations[left]["bbox"]),
                    tuple(observations[right]["bbox"]),
                ),
            )
    count_variation = (
        abs(len(observations) - previous_count) if previous_count is not None else 0
    )
    minimum_confidence = min(confidences, default=1.0)
    signals = {
        "instance_count": len(observations),
        "previous_instance_count": previous_count,
        "count_variation": count_variation,
        "minimum_confidence": minimum_confidence,
        "maximum_pairwise_iou": max_iou,
        "ambiguous_handoff": ambiguous,
        "track_endpoint": track_endpoint,
    }
    if ambiguous:
        return "ambiguous_handoff", 1.0, signals
    if minimum_confidence < 0.65:
        return "low_confidence", 1.0 - minimum_confidence, signals
    if max_iou > 0.55:
        return "adjacent_overlap", max_iou, signals
    if count_variation > 0:
        return "count_change", min(1.0, count_variation / 3.0 + 0.4), signals
    if track_endpoint:
        return "lost_or_new_track", 0.7, signals
    return None, 0.0, signals


def _collect_frames(
    db: VisionSortDB, session_id: str
) -> dict[tuple[str, int], dict[str, Any]]:
    tracklets = [
        dict(row)
        for row in db.fetch_all(
            "SELECT * FROM tracklets WHERE session_id = ? ORDER BY ended_at_global ASC",
            (session_id,),
        )
    ]
    frames: dict[tuple[str, int], dict[str, Any]] = {}
    for tracklet in tracklets:
        observation_path = Path(str(tracklet["observation_path"]))
        if not observation_path.is_absolute():
            observation_path = ROOT_DIR / observation_path
        if not observation_path.exists():
            continue
        observations = _load_track_observations(observation_path)
        for observation_index, observation in enumerate(observations):
            source_id = str(
                observation.get("source_id")
                or tracklet.get("source_id")
                or tracklet["camera_id"]
            )
            frame_index = int(observation["frame_index"])
            key = (source_id, frame_index)
            frame = frames.setdefault(
                key,
                {
                    "source_id": source_id,
                    "camera_role": observation.get("camera_role")
                    or tracklet.get("camera_role"),
                    "frame_index": frame_index,
                    "timestamp_global": float(
                        observation.get(
                            "timestamp_global", tracklet["ended_at_global"]
                        )
                    ),
                    "observations": {},
                    "ambiguous": False,
                    "track_endpoint": False,
                    "tracklet_ids": set(),
                },
            )
            identity = (
                str(observation.get("camera_id") or source_id),
                int(observation.get("local_track_id") or -1),
            )
            enriched = dict(observation)
            enriched["tracklet_id"] = tracklet["tracklet_id"]
            enriched["global_parcel_id"] = tracklet.get("parcel_id")
            enriched["match_result"] = tracklet.get("match_result")
            frame["observations"][identity] = enriched
            frame["ambiguous"] = frame["ambiguous"] or (
                tracklet.get("match_result") == "AMBIGUOUS"
            )
            frame["track_endpoint"] = frame["track_endpoint"] or observation_index in {
                0,
                len(observations) - 1,
            }
            frame["tracklet_ids"].add(tracklet["tracklet_id"])
    for frame in frames.values():
        frame["observations"] = list(frame["observations"].values())
        frame["tracklet_ids"] = sorted(frame["tracklet_ids"])
    return frames


def _select_and_synchronize(
    frames: dict[tuple[str, int], dict[str, Any]],
    *,
    sync_window_seconds: float = 0.25,
    sync_tolerance_seconds: float = 0.35,
    control_stride: int = 12,
) -> list[tuple[str, float, list[dict[str, Any]]]]:
    frames_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected: list[dict[str, Any]] = []
    for frame in frames.values():
        frames_by_source[frame["source_id"]].append(frame)
    for source_frames in frames_by_source.values():
        source_frames.sort(key=lambda item: item["frame_index"])
        previous_count: int | None = None
        for position, frame in enumerate(source_frames):
            reason, score, signals = _frame_signals(
                frame["observations"],
                previous_count=previous_count,
                ambiguous=bool(frame["ambiguous"]),
                track_endpoint=bool(frame["track_endpoint"]),
            )
            previous_count = len(frame["observations"])
            if reason is None and position % control_stride == 0:
                reason, score = "control_example", 0.2
            frame["selection_reason"] = reason
            frame["selection_score"] = score
            frame["selection_signals"] = signals
            if reason is not None:
                selected.append(frame)

    anchors_by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in selected:
        bucket = int(round(frame["timestamp_global"] / sync_window_seconds))
        anchors_by_bucket[bucket].append(frame)

    groups: list[tuple[str, float, list[dict[str, Any]]]] = []
    for bucket, anchors in sorted(anchors_by_bucket.items()):
        anchor_time = statistics.median(
            float(frame["timestamp_global"]) for frame in anchors
        )
        synchronized: list[dict[str, Any]] = []
        for source_id, source_frames in sorted(frames_by_source.items()):
            nearest = min(
                source_frames,
                key=lambda frame: abs(float(frame["timestamp_global"]) - anchor_time),
            )
            if abs(float(nearest["timestamp_global"]) - anchor_time) <= sync_tolerance_seconds:
                synchronized.append(nearest)
        if not synchronized:
            synchronized = anchors
        reasons = sorted(
            {
                frame.get("selection_reason")
                for frame in anchors
                if frame.get("selection_reason")
            }
        )
        sample_group_id = f"{bucket:016x}"
        for frame in synchronized:
            frame["group_reasons"] = reasons
            frame["selection_reason"] = frame.get("selection_reason") or "synchronized_context"
            frame["selection_score"] = max(
                float(frame.get("selection_score") or 0.0),
                max(float(anchor.get("selection_score") or 0.0) for anchor in anchors),
            )
        groups.append((sample_group_id, anchor_time, synchronized))
    return groups


def validate_dataset_splits(
    db: VisionSortDB, dataset_id: str
) -> dict[str, Any]:
    items = [
        dict(row)
        for row in db.fetch_all(
            "SELECT * FROM dataset_items WHERE dataset_id = ?", (dataset_id,)
        )
    ]
    leaks: list[dict[str, Any]] = []
    for field in ("session_id", "sample_group_id"):
        split_by_value: dict[str, set[str]] = defaultdict(set)
        for item in items:
            if item.get(field):
                split_by_value[str(item[field])].add(str(item.get("split")))
        for value, splits in split_by_value.items():
            if len(splits) > 1:
                leaks.append(
                    {"kind": f"{field}_cross_split", "value": value, "splits": sorted(splits)}
                )
    hashes: dict[str, set[str]] = defaultdict(set)
    for item in items:
        metadata = json.loads(item.get("metadata_json") or "{}")
        image_hash = metadata.get("image_ahash64")
        if image_hash:
            hashes[str(image_hash)].add(str(item.get("split")))
    for image_hash, splits in hashes.items():
        if len(splits) > 1:
            leaks.append(
                {"kind": "image_hash_cross_split", "value": image_hash, "splits": sorted(splits)}
            )
    test_rows = sorted(
        (
            str(item.get("session_id")),
            str(item.get("sample_group_id")),
            str(item.get("source_id")),
            int(item.get("frame_index") or 0),
        )
        for item in items
        if item.get("split") == "test"
    )
    frozen_test_sha256 = hashlib.sha256(
        json.dumps(test_rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "valid": not leaks,
        "leaks": leaks,
        "session_assignments": {
            session_id: sorted(
                {
                    str(item.get("split"))
                    for item in items
                    if item.get("session_id") == session_id
                }
            )
            for session_id in sorted(
                {str(item.get("session_id")) for item in items if item.get("session_id")}
            )
        },
        "test_frozen": True,
        "frozen_test_sha256": frozen_test_sha256,
        "test_items": len(test_rows),
    }


def rewrite_training_manifest(
    db: VisionSortDB, dataset_id: str, manifest_path: Path
) -> int:
    trainable_statuses = {
        AnnotationStatus.AUTO_ACCEPTED.value,
        AnnotationStatus.HUMAN_VALIDATED.value,
    }
    items = [
        dict(row)
        for row in db.fetch_all(
            "SELECT * FROM dataset_items WHERE dataset_id = ? ORDER BY sample_group_id, camera_role",
            (dataset_id,),
        )
    ]
    rows = [
        {
            "item_id": item["id"],
            "session_id": item.get("session_id") or "",
            "sample_group_id": item.get("sample_group_id") or "",
            "split": item.get("split") or "",
            "source_id": item.get("source_id") or "",
            "camera_role": item.get("camera_role") or "",
            "frame_index": item.get("frame_index") or 0,
            "timestamp_global": item.get("timestamp_global") or 0.0,
            "image_path": item["image_path"],
            "label_path": item.get("label_path") or "",
            "annotation_status": item["annotation_status"],
            "reason": item["reason"],
            "score": item["score"],
        }
        for item in items
        if item["annotation_status"] in trainable_statuses and item.get("label_path")
    ]
    fieldnames = [
        "item_id",
        "session_id",
        "sample_group_id",
        "split",
        "source_id",
        "camera_role",
        "frame_index",
        "timestamp_global",
        "image_path",
        "label_path",
        "annotation_status",
        "reason",
        "score",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_dataset(
    db: VisionSortDB, *, session_id: str, name: str = "autodataset"
) -> dict[str, Any]:
    artifact_repo = ArtifactRepository(db)
    dataset_id = f"dataset-{uuid.uuid4().hex[:8]}"
    root = DATASETS_DIR / dataset_id
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    frames = _collect_frames(db, session_id)
    groups = _select_and_synchronize(frames)
    sources_by_id = {
        str(row["id"]): dict(row)
        for row in db.fetch_all("SELECT * FROM sources")
    }
    split = stable_session_split(session_id)
    manifest_path = root / "manifest.csv"
    seen_hashes_by_source: dict[str, list[int]] = defaultdict(list)
    dedup_groups_skipped = 0
    manifest_items = 0

    for sample_group_id, anchor_time, synchronized_frames in groups:
        prepared: list[tuple[dict[str, Any], Any, int]] = []
        for frame in synchronized_frames:
            source = sources_by_id.get(str(frame["source_id"]))
            if source is None:
                continue
            capture = cv2.VideoCapture(str(source["uri"]))
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame["frame_index"]))
            ok, image = capture.read()
            capture.release()
            if not ok:
                continue
            prepared.append((frame, image, _ahash64(image)))
        if not prepared:
            continue
        if all(
            _min_hamming(image_hash, seen_hashes_by_source[frame["source_id"]]) <= 3
            for frame, _, image_hash in prepared
        ):
            dedup_groups_skipped += 1
            continue
        for frame, image, image_hash in prepared:
            seen_hashes_by_source[frame["source_id"]].append(image_hash)
            role = str(frame.get("camera_role") or "camera")
            image_name = (
                f"{sample_group_id}_{role}_{frame['source_id']}_{frame['frame_index']}.jpg"
            )
            image_path = root / "images" / split / image_name
            if not cv2.imwrite(str(image_path), image):
                raise RuntimeError(f"Échec d'écriture image: {image_path}")
            metadata = {
                "session_id": session_id,
                "source_id": frame["source_id"],
                "camera_role": role,
                "frame_index": frame["frame_index"],
                "timestamp_global": frame["timestamp_global"],
                "sample_group_anchor": anchor_time,
                "selection_signals": frame["selection_signals"],
                "group_reasons": frame["group_reasons"],
                "tracklet_ids": frame["tracklet_ids"],
                "observations": frame["observations"],
                "instance_count": len(frame["observations"]),
                "model_ids": sorted(
                    {
                        str(item.get("model_id"))
                        for item in frame["observations"]
                        if item.get("model_id")
                    }
                ),
                "tracker_ids": sorted(
                    {
                        str(item.get("tracker_id"))
                        for item in frame["observations"]
                        if item.get("tracker_id")
                    }
                ),
                "validated_on_site": False,
                "image_ahash64": f"{image_hash:016x}",
            }
            artifact_repo.add_dataset_item(
                dataset_id=dataset_id,
                session_id=session_id,
                sample_group_id=f"{session_id}:{sample_group_id}",
                split=split,
                source_id=frame["source_id"],
                camera_role=role,
                frame_index=int(frame["frame_index"]),
                timestamp_global=float(frame["timestamp_global"]),
                image_path=relative_to_root(image_path),
                label_path=None,
                annotation_status=AnnotationStatus.NEEDS_REVIEW.value,
                reason=str(frame["selection_reason"]),
                score=float(frame["selection_score"]),
                metadata=metadata,
            )
            manifest_items += 1

    rewrite_training_manifest(db, dataset_id, manifest_path)
    integrity = validate_dataset_splits(db, dataset_id)
    if not integrity["valid"]:
        raise RuntimeError(f"Fuite de split détectée: {integrity['leaks']}")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    data_yaml_path = root / "data.yaml"
    data_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": relative_to_root(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {
                    0: "parcel",
                    1: "person",
                    2: "left_wrist",
                    3: "right_wrist",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    summary = {
        "items": manifest_items,
        "sample_groups": len(
            {
                row["sample_group_id"]
                for row in db.fetch_all(
                    "SELECT sample_group_id FROM dataset_items WHERE dataset_id = ?",
                    (dataset_id,),
                )
            }
        ),
        "dedup_groups_skipped": dedup_groups_skipped,
        "validated_on_site": False,
        "selection_strategy": "frame_complete_synchronized_v3",
        "session_id": session_id,
        "source_session_id": session_id,
        "split_assignment": split,
        "dataset_version": dataset_id,
        "parent_dataset_id": None,
        "manifest_sha256": manifest_sha256,
        "sampling_code_version": "v3_frame_complete_session_split",
        "split_integrity": integrity,
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
    return {
        "dataset_id": dataset_id,
        "root": str(root),
        "manifest_rows": manifest_items,
        "sample_groups": summary["sample_groups"],
        "split": split,
        "split_integrity": integrity,
    }
