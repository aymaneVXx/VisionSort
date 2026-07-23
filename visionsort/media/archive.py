from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from visionsort.core.paths import ROOT_DIR
from visionsort.database.db import VisionSortDB, utc_now


class FrameArchiveError(RuntimeError):
    """Raised when a historical frame cannot be resolved immutably."""


@dataclass(frozen=True, slots=True)
class FrameArchiveKey:
    session_id: str
    source_id: str
    stream_epoch: int
    frame_index: int
    timestamp_global: float


def _absolute_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_segment(row: dict[str, Any]) -> tuple[Path, str | None]:
    path = _absolute_path(str(row["segment_path"]))
    if not bool(row.get("immutable", 1)):
        return path, "segment non immuable"
    if bool(row.get("corrupted")):
        return path, "segment marqué corrompu"
    if not path.is_file() or path.stat().st_size <= 0:
        return path, "segment absent ou vide"
    expected = str(row.get("sha256") or "")
    if expected and _sha256(path) != expected:
        return path, "empreinte du segment invalide"
    return path, None


class FrameArchiveResolver:
    """Resolve historical frames without consulting the mutable source URI."""

    def __init__(self, db: VisionSortDB):
        self.db = db
        self._validated_segments: set[str] = set()

    def resolve(
        self, key: FrameArchiveKey
    ) -> tuple[Any, dict[str, Any]]:
        archived = self.db.fetch_one(
            """
            SELECT rf.*, r.segment_path, r.sha256, r.corrupted, r.immutable,
                   r.codec, r.fps
            FROM recording_frames rf
            JOIN recordings r ON r.id = rf.recording_id
            WHERE rf.session_id = ? AND rf.source_id = ?
              AND rf.stream_epoch = ? AND rf.frame_index = ?
            ORDER BY ABS(rf.timestamp_global - ?) ASC
            LIMIT 1
            """,
            (
                key.session_id,
                key.source_id,
                int(key.stream_epoch),
                int(key.frame_index),
                float(key.timestamp_global),
            ),
        )
        if archived is not None:
            row = dict(archived)
            path = _absolute_path(str(row["segment_path"]))
            if str(row["recording_id"]) not in self._validated_segments:
                path, error = _validate_segment(row)
                if error:
                    raise FrameArchiveError(
                        f"Archive inutilisable pour {key.source_id}: {error} ({path})"
                    )
                self._validated_segments.add(str(row["recording_id"]))
            capture = cv2.VideoCapture(str(path))
            capture.set(
                cv2.CAP_PROP_POS_FRAMES, int(row["segment_frame_index"])
            )
            ok, image = capture.read()
            capture.release()
            if not ok or image is None:
                raise FrameArchiveError(
                    "Frame archivée illisible: "
                    f"{key.session_id}/{key.source_id}/"
                    f"{key.stream_epoch}/{key.frame_index}"
                )
            return image, {
                "origin": "session_archive",
                "recording_id": row["recording_id"],
                "segment_path": str(row["segment_path"]),
                "segment_frame_index": int(row["segment_frame_index"]),
                "stream_epoch": int(row["stream_epoch"]),
                "archived_timestamp_global": float(row["timestamp_global"]),
            }

        snapshot = self.db.fetch_one(
            """
            SELECT css.source_type_snapshot, css.source_uri_snapshot,
                   css.source_sha256, css.archive_required,
                   s.source_type AS legacy_source_type,
                   s.uri AS legacy_uri
            FROM capture_session_sources css
            LEFT JOIN sources s ON s.id = css.source_id
            WHERE css.session_id = ? AND css.source_id = ?
            LIMIT 1
            """,
            (key.session_id, key.source_id),
        )
        if snapshot is None:
            legacy_source = self.db.fetch_one(
                "SELECT source_type, uri FROM sources WHERE id = ?",
                (key.source_id,),
            )
            if (
                legacy_source is None
                or str(legacy_source["source_type"]).upper() != "REPLAY"
            ):
                raise FrameArchiveError(
                    f"Source {key.source_id} absente de la session "
                    f"{key.session_id}; fallback interdit."
                )
            snapshot = {
                "source_type_snapshot": "REPLAY",
                "source_uri_snapshot": str(legacy_source["uri"]),
                "source_sha256": None,
                "archive_required": 0,
                "legacy_source_type": "REPLAY",
                "legacy_uri": str(legacy_source["uri"]),
            }
        source_type = str(
            snapshot["source_type_snapshot"]
            or snapshot["legacy_source_type"]
            or ""
        ).upper()
        archive_required = bool(snapshot["archive_required"]) or source_type in {
            "RTSP",
            "VIDEO_FILE",
        }
        if archive_required:
            raise FrameArchiveError(
                "Frame obligatoire absente de l'archive de session: "
                f"{key.session_id}/{key.source_id}/"
                f"{key.stream_epoch}/{key.frame_index}"
            )
        if source_type != "REPLAY":
            raise FrameArchiveError(
                f"Fallback historique interdit pour une source {source_type}."
            )
        uri = str(snapshot["source_uri_snapshot"] or snapshot["legacy_uri"] or "")
        path = Path(uri)
        if not path.is_file():
            raise FrameArchiveError(
                f"Replay figé introuvable pour la session: {path}"
            )
        expected_sha = str(snapshot["source_sha256"] or "")
        if expected_sha and _sha256(path) != expected_sha:
            raise FrameArchiveError(
                f"Replay figé modifié depuis la capture: {path}"
            )
        capture = cv2.VideoCapture(str(path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(key.frame_index))
        ok, image = capture.read()
        capture.release()
        if not ok or image is None:
            raise FrameArchiveError(
                f"Frame {key.frame_index} illisible dans le Replay figé {path}."
            )
        return image, {
            "origin": "frozen_replay",
            "source_uri_snapshot": uri,
            "source_sha256": expected_sha or None,
            "stream_epoch": int(key.stream_epoch),
        }

    def assert_session_ready(self, session_id: str) -> dict[str, Any]:
        report = build_session_media_report(self.db, session_id)
        if not report["valid"]:
            raise FrameArchiveError(
                "Couverture média insuffisante: "
                + "; ".join(report["errors"])
            )
        return report


def build_session_media_report(
    db: VisionSortDB,
    session_id: str,
    *,
    persist: bool = True,
) -> dict[str, Any]:
    source_rows = [
        dict(row)
        for row in db.fetch_all(
            """
            SELECT css.source_id, css.camera_role, css.archive_required,
                   css.source_type_snapshot, s.source_type AS legacy_source_type
            FROM capture_session_sources css
            LEFT JOIN sources s ON s.id = css.source_id
            WHERE css.session_id = ?
            ORDER BY css.camera_role, css.source_id
            """,
            (session_id,),
        )
    ]
    coverage_rows = {
        str(row["source_id"]): dict(row)
        for row in db.fetch_all(
            """
            SELECT * FROM session_media_coverage WHERE session_id = ?
            """,
            (session_id,),
        )
    }
    errors: list[str] = []
    warnings: list[str] = []
    sources: list[dict[str, Any]] = []
    totals = {
        "frames_acquired": 0,
        "frames_processed": 0,
        "frames_archived": 0,
        "frames_unarchived": 0,
        "segments_produced": 0,
        "segments_corrupted": 0,
        "bytes_used": 0,
    }
    for source in source_rows:
        source_id = str(source["source_id"])
        source_type = str(
            source.get("source_type_snapshot")
            or source.get("legacy_source_type")
            or ""
        ).upper()
        required = bool(source.get("archive_required")) or source_type in {
            "RTSP",
            "VIDEO_FILE",
        }
        coverage = coverage_rows.get(source_id)
        segment_rows = [
            dict(row)
            for row in db.fetch_all(
                """
                SELECT * FROM recordings
                WHERE session_id = ? AND source_id = ?
                ORDER BY segment_index, started_at
                """,
                (session_id, source_id),
            )
        ]
        segment_errors: list[str] = []
        for segment in segment_rows:
            _, error = _validate_segment(segment)
            if error:
                segment_errors.append(
                    f"{segment['id']}: {error}"
                )
        if coverage is None:
            frame_count = int(
                (
                    db.fetch_one(
                        """
                        SELECT COUNT(*) AS count FROM recording_frames
                        WHERE session_id = ? AND source_id = ?
                        """,
                        (session_id, source_id),
                    )
                    or {"count": 0}
                )["count"]
            )
            coverage = {
                "frames_acquired": 0,
                "frames_processed": 0,
                "frames_archived": frame_count,
                "frames_unarchived": 0,
                "segments_produced": len(segment_rows),
                "segments_corrupted": len(segment_errors),
                "bytes_used": sum(
                    int(row.get("size_bytes") or 0) for row in segment_rows
                ),
                "coverage_ratio": 1.0 if frame_count else 0.0,
                "status": (
                    "COMPLETE"
                    if frame_count and not segment_errors
                    else "INSUFFICIENT"
                    if required
                    else "NOT_REQUIRED"
                ),
            }
        item = {
            "source_id": source_id,
            "camera_role": source.get("camera_role"),
            "source_type": source_type,
            "archive_required": required,
            **{
                key: coverage.get(key, 0)
                for key in totals
            },
            "coverage_ratio": float(coverage.get("coverage_ratio") or 0.0),
            "status": str(coverage.get("status") or "INSUFFICIENT"),
            "segment_errors": segment_errors,
        }
        if required and (
            item["status"] != "COMPLETE"
            or item["coverage_ratio"] < 1.0
            or segment_errors
        ):
            errors.append(
                f"{source_id}: archive obligatoire incomplète "
                f"(status={item['status']}, couverture={item['coverage_ratio']:.3f})"
            )
        elif not required and not segment_rows:
            warnings.append(
                f"{source_id}: Replay figé utilisé sans copie d'archive."
            )
        for key in totals:
            totals[key] += int(item[key])
        sources.append(item)
    report = {
        "session_id": session_id,
        "valid": not errors,
        "status": "COMPLETE" if not errors else "INSUFFICIENT",
        "errors": errors,
        "warnings": warnings,
        "sources": sources,
        "totals": totals,
        "generated_at": utc_now(),
    }
    if persist:
        db.execute(
            """
            UPDATE capture_sessions
            SET media_report_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (json.dumps(report), utc_now(), session_id),
        )
    return report
