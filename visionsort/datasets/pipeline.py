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
from visionsort.datasets.integrity import DatasetIntegrityValidator
from visionsort.media.archive import (
    FrameArchiveKey,
    FrameArchiveResolver,
)
from visionsort.tracking.engine import bbox_iou


def _load_track_observations(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_session_split(session_id: str) -> str:
    """Backward-compatible split for legacy single-session datasets."""
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10
    return "train" if bucket < 7 else "val" if bucket < 9 else "test"


def resolve_split_assignments(
    session_ids: list[str],
    explicit: dict[str, str] | None = None,
) -> dict[str, str]:
    """Resolve and persist a deterministic project-level session assignment."""
    unique_ids = list(dict.fromkeys(str(value) for value in session_ids))
    if not unique_ids:
        raise ValueError("Au moins une CaptureSession est requise.")
    assignments = {
        str(session_id): str(split).lower()
        for session_id, split in (explicit or {}).items()
    }
    unknown = set(assignments) - set(unique_ids)
    if unknown:
        raise ValueError(f"Sessions inconnues dans les splits: {sorted(unknown)}")
    invalid = {
        session_id: split
        for session_id, split in assignments.items()
        if split not in {"train", "val", "test"}
    }
    if invalid:
        raise ValueError(f"Splits invalides: {invalid}")

    remaining = [session_id for session_id in unique_ids if session_id not in assignments]
    ordered = sorted(
        remaining,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if len(unique_ids) == 1 and ordered:
        assignments[ordered[0]] = stable_session_split(ordered[0])
    else:
        required = [
            split
            for split in ("train", "val", "test")
            if split not in assignments.values()
        ]
        for session_id, split in zip(ordered, required):
            assignments[session_id] = split
        for session_id in ordered[len(required) :]:
            counts = {
                split: sum(value == split for value in assignments.values())
                for split in ("train", "val", "test")
            }
            assignments[session_id] = min(
                ("train", "val", "test"),
                key=lambda split: (
                    counts[split] / {"train": 0.7, "val": 0.2, "test": 0.1}[split],
                    split,
                ),
            )
    return {session_id: assignments[session_id] for session_id in unique_ids}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
) -> dict[tuple[str, int, int], dict[str, Any]]:
    tracklets = [
        dict(row)
        for row in db.fetch_all(
            "SELECT * FROM tracklets WHERE session_id = ? ORDER BY ended_at_global ASC",
            (session_id,),
        )
    ]
    frames: dict[tuple[str, int, int], dict[str, Any]] = {}
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
            extra = observation.get("extra") or {}
            stream_epoch = int(extra.get("_stream_epoch") or 0)
            key = (source_id, stream_epoch, frame_index)
            frame = frames.setdefault(
                key,
                {
                    "source_id": source_id,
                    "camera_role": observation.get("camera_role")
                    or tracklet.get("camera_role"),
                    "frame_index": frame_index,
                    "stream_epoch": stream_epoch,
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
    frames: dict[tuple[str, int, int], dict[str, Any]],
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
    perceptual_hashes: list[tuple[str, int, str]] = []
    for item in items:
        metadata = json.loads(item.get("metadata_json") or "{}")
        image_sha256 = metadata.get("image_sha256")
        if image_sha256:
            hashes[str(image_sha256)].add(str(item.get("split")))
        image_hash = metadata.get("image_ahash64")
        if image_hash:
            try:
                perceptual_hashes.append(
                    (str(item["id"]), int(str(image_hash), 16), str(item.get("split")))
                )
            except ValueError:
                leaks.append(
                    {
                        "kind": "invalid_image_ahash64",
                        "value": str(image_hash),
                        "splits": [str(item.get("split"))],
                    }
                )
    for image_hash, splits in hashes.items():
        if len(splits) > 1:
            leaks.append(
                {
                    "kind": "image_sha256_cross_split",
                    "value": image_hash,
                    "splits": sorted(splits),
                }
            )
    for left in range(len(perceptual_hashes)):
        left_id, left_hash, left_split = perceptual_hashes[left]
        for right in range(left + 1, len(perceptual_hashes)):
            right_id, right_hash, right_split = perceptual_hashes[right]
            if left_split == right_split:
                continue
            distance = int((left_hash ^ right_hash).bit_count())
            if distance <= 3:
                leaks.append(
                    {
                        "kind": "near_duplicate_cross_split",
                        "value": f"{left_id}:{right_id}",
                        "splits": sorted({left_split, right_split}),
                        "hamming_distance": distance,
                    }
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
    split_counts = {
        split: sum(str(item.get("split")) == split for item in items)
        for split in ("train", "val", "test")
    }
    stored_assignments = {
        str(row["session_id"]): str(row["split"])
        for row in db.fetch_all(
            "SELECT session_id, split FROM dataset_sessions WHERE dataset_id = ?",
            (dataset_id,),
        )
    }
    item_assignments = {
        session_id: next(iter(splits))
        for session_id in {
            str(item.get("session_id")) for item in items if item.get("session_id")
        }
        if len(
            splits := {
                str(item.get("split"))
                for item in items
                if str(item.get("session_id")) == session_id
            }
        )
        == 1
    }
    for session_id, split in stored_assignments.items():
        if session_id in item_assignments and item_assignments[session_id] != split:
            leaks.append(
                {
                    "kind": "stored_session_split_mismatch",
                    "value": session_id,
                    "splits": sorted({split, item_assignments[session_id]}),
                }
            )
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
        "split_counts": split_counts,
        "all_splits_nonempty": all(split_counts.values()),
        "stored_session_assignments": stored_assignments,
    }


def compute_dataset_fingerprint(db: VisionSortDB, dataset_id: str) -> str:
    dataset = db.fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        raise RuntimeError("Dataset introuvable.")
    integrity = DatasetIntegrityValidator(db, dataset_id).validate()
    if not integrity["valid"]:
        raise RuntimeError(
            "Fingerprint refusé: intégrité du dataset invalide. "
            f"Rapport: {integrity['report_path']}. "
            + " ".join(integrity["errors"][:5])
        )
    files: list[dict[str, str]] = []
    for item in db.fetch_all(
        """
        SELECT id, session_id, split, source_id, frame_index,
               annotation_status, image_path, label_path
        FROM dataset_items WHERE dataset_id = ? ORDER BY id
        """,
        (dataset_id,),
    ):
        for kind, value in (
            ("image", item["image_path"]),
            ("label", item["label_path"]),
        ):
            if not value and kind == "label":
                continue
            if not value:
                raise RuntimeError(
                    f"Fingerprint refusé: {kind} absent pour {item['id']}."
                )
            path = Path(str(value))
            if not path.is_absolute():
                path = ROOT_DIR / path
            if not path.is_file():
                raise RuntimeError(
                    f"Fingerprint refusé: fichier absent {path}."
                )
            files.append(
                {
                    "kind": kind,
                    "item_id": str(item["id"]),
                    "path": str(value),
                    "sha256": _sha256_file(path),
                }
            )
    for kind, column in (
        ("manifest", "manifest_path"),
        ("data_yaml", "data_yaml_path"),
    ):
        value = dataset[column]
        if not value:
            raise RuntimeError(
                f"Fingerprint refusé: {kind} non référencé."
            )
        path = Path(str(value))
        if not path.is_absolute():
            path = ROOT_DIR / path
        if not path.is_file():
            raise RuntimeError(
                f"Fingerprint refusé: fichier absent {path}."
            )
        files.append(
            {
                "kind": kind,
                "item_id": "",
                "path": str(value),
                "sha256": _sha256_file(path),
            }
        )
    dataset_root = Path(str(dataset["root_path"]))
    if not dataset_root.is_absolute():
        dataset_root = ROOT_DIR / dataset_root
    for artifact_name in (
        "tracking_manifest.jsonl",
        "reid_manifest.jsonl",
        "reid_pairs.jsonl",
    ):
        artifact_path = dataset_root / artifact_name
        if artifact_path.is_file():
            files.append(
                {
                    "kind": "task_artifact",
                    "item_id": "",
                    "path": relative_to_root(artifact_path),
                    "sha256": _sha256_file(artifact_path),
                }
            )
    if str(dataset["task"]) == "reid_multicamera":
        reid_manifest = dataset_root / "reid_manifest.jsonl"
        for row in _load_track_observations(reid_manifest):
            crop_value = row.get("crop_path")
            if not crop_value:
                raise RuntimeError(
                    "Fingerprint refusé: crop ReID non référencé."
                )
            crop_path = Path(str(crop_value))
            if not crop_path.is_absolute():
                crop_path = ROOT_DIR / crop_path
            if not crop_path.is_file():
                raise RuntimeError(
                    f"Fingerprint refusé: crop ReID absent {crop_path}."
                )
            files.append(
                {
                    "kind": "reid_crop",
                    "item_id": str(row.get("dataset_item_id") or ""),
                    "path": str(crop_value),
                    "sha256": _sha256_file(crop_path),
                }
            )
    sessions = [
        {
            "session_id": str(row["session_id"]),
            "split": str(row["split"]),
        }
        for row in db.fetch_all(
            """
            SELECT session_id, split FROM dataset_sessions
            WHERE dataset_id = ? ORDER BY session_id
            """,
            (dataset_id,),
        )
    ]
    payload = {
        "dataset_id": dataset_id,
        "task": str(dataset["task"]),
        "generation_config": json.loads(dataset["generation_config_json"] or "{}"),
        "generation_strategy_version": "visionsort-dataset-v4",
        "sessions": sessions,
        "items": [
            {
                "id": str(row["id"]),
                "session_id": str(row["session_id"] or ""),
                "split": str(row["split"] or ""),
                "source_id": str(row["source_id"] or ""),
                "frame_index": int(row["frame_index"] or 0),
                "annotation_status": str(row["annotation_status"]),
            }
            for row in db.fetch_all(
                """
                SELECT id, session_id, split, source_id, frame_index,
                       annotation_status
                FROM dataset_items
                WHERE dataset_id = ? ORDER BY id
                """,
                (dataset_id,),
            )
        ],
        "files": sorted(
            files,
            key=lambda value: (
                value["kind"],
                value["item_id"],
                value["path"],
            ),
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_dataset_fingerprint(
    db: VisionSortDB, dataset_id: str
) -> dict[str, Any]:
    dataset = db.fetch_one(
        "SELECT dataset_fingerprint FROM datasets WHERE id = ?", (dataset_id,)
    )
    if dataset is None:
        raise RuntimeError("Dataset introuvable.")
    expected = dataset["dataset_fingerprint"]
    try:
        actual = compute_dataset_fingerprint(db, dataset_id)
    except RuntimeError as exc:
        return {
            "valid": False,
            "expected": expected,
            "actual": None,
            "error": str(exc),
        }
    return {
        "valid": bool(expected) and str(expected) == actual,
        "expected": expected,
        "actual": actual,
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
    db: VisionSortDB,
    *,
    session_id: str | None = None,
    session_ids: list[str] | None = None,
    split_assignments: dict[str, str] | None = None,
    name: str = "autodataset",
    task: str = "detection",
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_repo = ArtifactRepository(db)
    supported_tasks = {
        "detection",
        "segmentation",
        "pose",
        "local_tracking",
        "reid_multicamera",
    }
    if task not in supported_tasks:
        raise ValueError(f"Tâche dataset non supportée: {task}")
    selected_sessions = list(
        dict.fromkeys(
            str(value)
            for value in (session_ids or ([session_id] if session_id else []))
        )
    )
    assignments = resolve_split_assignments(
        selected_sessions, explicit=split_assignments
    )
    placeholders = ",".join("?" for _ in selected_sessions)
    session_rows = {
        str(row["id"]): dict(row)
        for row in db.fetch_all(
            f"SELECT * FROM capture_sessions WHERE id IN ({placeholders})",
            tuple(selected_sessions),
        )
    }
    missing_sessions = set(selected_sessions) - set(session_rows)
    if missing_sessions:
        raise RuntimeError(f"CaptureSessions introuvables: {sorted(missing_sessions)}")
    if len(selected_sessions) > 1:
        unfinished = [
            value
            for value in selected_sessions
            if session_rows[value].get("ended_at") is None
        ]
        if unfinished:
            raise RuntimeError(
                f"Les CaptureSessions doivent être terminées: {unfinished}"
            )
    frame_resolver = FrameArchiveResolver(db)
    media_reports = {
        current_session_id: frame_resolver.assert_session_ready(
            current_session_id
        )
        for current_session_id in selected_sessions
    }
    dataset_id = f"dataset-{uuid.uuid4().hex[:8]}"
    root = DATASETS_DIR / dataset_id
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest.csv"
    seen_hashes: list[tuple[int, str]] = []
    dedup_groups_skipped = 0
    manifest_items = 0

    for current_session_id in selected_sessions:
        split = assignments[current_session_id]
        frames = _collect_frames(db, current_session_id)
        groups = _select_and_synchronize(frames)
        for sample_group_id, anchor_time, synchronized_frames in groups:
            prepared: list[tuple[dict[str, Any], Any, int]] = []
            for frame in synchronized_frames:
                image, media_provenance = frame_resolver.resolve(
                    FrameArchiveKey(
                        session_id=current_session_id,
                        source_id=str(frame["source_id"]),
                        stream_epoch=int(frame.get("stream_epoch") or 0),
                        frame_index=int(frame["frame_index"]),
                        timestamp_global=float(frame["timestamp_global"]),
                    )
                )
                frame["media_provenance"] = media_provenance
                prepared.append((frame, image, _ahash64(image)))
            if not prepared:
                continue
            cross_split_duplicate = any(
                any(
                    prior_split != split
                    and int((image_hash ^ prior_hash).bit_count()) <= 3
                    for prior_hash, prior_split in seen_hashes
                )
                for _, _, image_hash in prepared
            )
            all_duplicates = all(
                any(
                    int((image_hash ^ prior_hash).bit_count()) <= 3
                    for prior_hash, _ in seen_hashes
                )
                for _, _, image_hash in prepared
            )
            if cross_split_duplicate or all_duplicates:
                dedup_groups_skipped += 1
                continue
            for frame, image, image_hash in prepared:
                seen_hashes.append((image_hash, split))
                role = str(frame.get("camera_role") or "camera")
                image_name = (
                    f"{current_session_id}_{sample_group_id}_{role}_"
                    f"{frame['source_id']}_{frame['frame_index']}.jpg"
                )
                image_path = root / "images" / split / image_name
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"Échec d'écriture image: {image_path}")
                metadata = {
                    "session_id": current_session_id,
                    "source_id": frame["source_id"],
                    "camera_role": role,
                    "frame_index": frame["frame_index"],
                    "stream_epoch": int(
                        frame.get("stream_epoch") or 0
                    ),
                    "timestamp_global": frame["timestamp_global"],
                    "media_provenance": frame["media_provenance"],
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
                    "image_sha256": _sha256_file(image_path),
                }
                artifact_repo.add_dataset_item(
                    dataset_id=dataset_id,
                    session_id=current_session_id,
                    sample_group_id=f"{current_session_id}:{sample_group_id}",
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
    data_yaml_path = root / "data.yaml"
    data_yaml: dict[str, Any] = {
        "path": relative_to_root(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "task": task,
        "names": (
            {0: "person"} if task == "pose" else {0: "parcel", 1: "person"}
        ),
    }
    if task == "pose":
        data_yaml["kpt_shape"] = [17, 3]
        data_yaml["flip_idx"] = [
            0,
            2,
            1,
            4,
            3,
            6,
            5,
            8,
            7,
            10,
            9,
            12,
            11,
            14,
            13,
            16,
            15,
        ]
    data_yaml_path.write_text(
        yaml.safe_dump(
            data_yaml,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    generation = {
        "sampling_code_version": "v4_multi_session_project",
        "selection_strategy": "frame_complete_synchronized_v4",
        "near_duplicate_hamming_threshold": 3,
        **(generation_config or {}),
    }
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
        "task": task,
        "selection_strategy": generation["selection_strategy"],
        "session_id": selected_sessions[0],
        "session_ids": selected_sessions,
        "source_session_id": selected_sessions[0],
        "split_assignment": (
            assignments[selected_sessions[0]] if len(selected_sessions) == 1 else None
        ),
        "split_assignments": assignments,
        "dataset_version": dataset_id,
        "parent_dataset_id": None,
        "manifest_sha256": _sha256_file(manifest_path),
        "sampling_code_version": generation["sampling_code_version"],
        "generation_config": generation,
        "media_coverage": media_reports,
    }
    artifact_repo.upsert_dataset(
        dataset_id=dataset_id,
        name=name,
        root_path=relative_to_root(root),
        status="SAMPLED",
        manifest_path=relative_to_root(manifest_path),
        data_yaml_path=relative_to_root(data_yaml_path),
        summary=summary,
        task=task,
        generation_config=generation,
    )
    artifact_repo.set_dataset_sessions(dataset_id, assignments)
    integrity = validate_dataset_splits(db, dataset_id)
    if not integrity["valid"]:
        raise RuntimeError(f"Fuite de split détectée: {integrity['leaks']}")
    summary["split_integrity"] = integrity
    artifact_repo.upsert_dataset(
        dataset_id=dataset_id,
        name=name,
        root_path=relative_to_root(root),
        status="SAMPLED",
        manifest_path=relative_to_root(manifest_path),
        data_yaml_path=relative_to_root(data_yaml_path),
        summary=summary,
        task=task,
        generation_config=generation,
    )
    return {
        "dataset_id": dataset_id,
        "root": str(root),
        "manifest_rows": manifest_items,
        "sample_groups": summary["sample_groups"],
        "split": (
            assignments[selected_sessions[0]] if len(selected_sessions) == 1 else None
        ),
        "split_assignments": assignments,
        "split_integrity": integrity,
    }
