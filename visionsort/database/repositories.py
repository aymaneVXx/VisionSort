from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from visionsort.core.enums import CommandStatus, CommandType, MatchResult, SourceStatus
from visionsort.core.types import GlobalParcel, Tracklet
from visionsort.database.db import VisionSortDB, utc_now


def _source_file_sha256(uri: str) -> str | None:
    path = Path(uri)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pipeline_role_for_task(task: str) -> str:
    return {
        "detection": "parcel_detection",
        "segmentation": "parcel_segmentation",
        "pose": "operator_pose",
    }.get(str(task), str(task))


class ControlRepository:
    def __init__(self, db: VisionSortDB):
        self.db = db

    def enqueue_command(self, command_type: CommandType | str, payload: dict[str, Any], owner: str = "streamlit") -> str:
        command_id = str(uuid.uuid4())
        now = utc_now()
        command_value = command_type.value if isinstance(command_type, CommandType) else str(command_type)
        self.db.execute(
            """
            INSERT INTO commands (id, command_type, payload_json, status, error_text, created_at, updated_at, owner)
            VALUES (?, ?, ?, ?, NULL, ?, ?, ?)
            """,
            (command_id, command_value, json.dumps(payload), CommandStatus.PENDING.value, now, now, owner),
        )
        return command_id

    def list_pending_commands(self) -> list[dict[str, Any]]:
        rows = self.db.fetch_all(
            "SELECT * FROM commands WHERE status = ? ORDER BY created_at ASC",
            (CommandStatus.PENDING.value,),
        )
        return [dict(row) for row in rows]

    def mark_command(self, command_id: str, status: CommandStatus | str, error_text: str | None = None) -> None:
        status_value = status.value if isinstance(status, CommandStatus) else str(status)
        self.db.execute(
            "UPDATE commands SET status = ?, error_text = ?, updated_at = ? WHERE id = ?",
            (status_value, error_text, utc_now(), command_id),
        )

    def list_sources(self) -> list[dict[str, Any]]:
        rows = self.db.fetch_all(
            """
            SELECT s.*, ss.status, ss.fps, ss.last_error, ss.last_frame_ts, ss.preview_path, ss.recording_enabled, ss.metrics_json
            FROM sources s
            LEFT JOIN source_state ss ON ss.source_id = s.id
            ORDER BY s.role ASC, s.name ASC
            """
        )
        output = [dict(row) for row in rows]
        for source in output:
            source["model_assignments"] = self.list_source_model_assignments(
                str(source["id"])
            )
        return output

    def list_capture_sessions(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM capture_sessions ORDER BY created_at DESC")]

    def get_capture_session(self, session_id: str) -> dict[str, Any] | None:
        row = self.db.fetch_one("SELECT * FROM capture_sessions WHERE id = ?", (session_id,))
        return dict(row) if row else None

    def list_capture_session_sources(self, session_id: str) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM capture_session_sources WHERE session_id = ? ORDER BY camera_role ASC", (session_id,))]

    def create_capture_session(self, *, name: str, demo_mode: bool, sources: list[dict[str, Any]], config: dict[str, Any]) -> str:
        session_id = f"session-{uuid.uuid4().hex[:10]}"
        now = utc_now()
        self.db.execute(
            """
            INSERT INTO capture_sessions (id, name, pipeline_state, demo_mode, site_validated, config_json, report_path, started_at, ended_at, created_at, updated_at)
            VALUES (?, ?, 'CAPTURED', ?, 0, ?, NULL, NULL, NULL, ?, ?)
            """,
            (session_id, name, int(demo_mode), json.dumps(config), now, now),
        )
        rows: list[tuple[Any, ...]] = []
        for item in sources:
            source = self.db.fetch_one(
                "SELECT source_type, uri FROM sources WHERE id = ?",
                (item["source_id"],),
            )
            if source is None:
                raise RuntimeError(
                    f"Source introuvable pour la session: {item['source_id']}"
                )
            source_type = str(source["source_type"]).upper()
            source_uri = str(source["uri"])
            archive_required = bool(
                item.get(
                    "archive_required",
                    source_type in {"RTSP", "VIDEO_FILE"}
                    or (
                        not demo_mode
                        and bool(config.get("archive_media", True))
                    ),
                )
            )
            model_pipeline = self.list_source_model_assignments(
                str(item["source_id"])
            )
            rows.append(
                (
                    f"sesssrc-{uuid.uuid4().hex[:10]}",
                    session_id,
                    item["source_id"],
                    item["camera_role"],
                    float(item.get("time_offset_ms", 0.0)),
                    item.get("replay_fps"),
                    source_type,
                    source_uri,
                    _source_file_sha256(source_uri),
                    int(archive_required),
                    json.dumps(model_pipeline),
                    now,
                    now,
                )
            )
        if rows:
            self.db.execute_many(
                """
                INSERT INTO capture_session_sources
                (id, session_id, source_id, camera_role, time_offset_ms,
                 replay_fps, source_type_snapshot, source_uri_snapshot,
                 source_sha256, archive_required, model_pipeline_json,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return session_id

    def update_capture_session(
        self,
        session_id: str,
        *,
        pipeline_state: str | None = None,
        started_at: float | None = None,
        ended_at: float | None = None,
        report_path: str | None = None,
        media_report: dict[str, Any] | None = None,
    ) -> None:
        fields: list[str] = []
        params: list[Any] = []
        if pipeline_state is not None:
            fields.append("pipeline_state = ?")
            params.append(pipeline_state)
        if started_at is not None:
            fields.append("started_at = ?")
            params.append(started_at)
        if ended_at is not None:
            fields.append("ended_at = ?")
            params.append(ended_at)
        if report_path is not None:
            fields.append("report_path = ?")
            params.append(report_path)
        if media_report is not None:
            fields.append("media_report_json = ?")
            params.append(json.dumps(media_report))
        fields.append("updated_at = ?")
        params.append(utc_now())
        params.append(session_id)
        self.db.execute(f"UPDATE capture_sessions SET {', '.join(fields)} WHERE id = ?", tuple(params))

    def upsert_source(self, payload: dict[str, Any]) -> str:
        now = utc_now()
        source_id = payload.get("id") or str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO sources (id, name, role, source_type, uri, model_id, tracker_id, enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                role = excluded.role,
                source_type = excluded.source_type,
                uri = excluded.uri,
                model_id = excluded.model_id,
                tracker_id = excluded.tracker_id,
                enabled = excluded.enabled,
                updated_at = excluded.updated_at
            """,
            (
                source_id,
                payload["name"],
                payload["role"],
                payload["source_type"],
                payload["uri"],
                payload["model_id"],
                payload["tracker_id"],
                int(payload.get("enabled", True)),
                now,
                now,
            ),
        )
        self.db.execute(
            """
            INSERT INTO source_state (source_id, status, fps, last_error, last_frame_ts, preview_path, details_path, recording_enabled, metrics_json, updated_at)
            VALUES (?, ?, 0, NULL, NULL, NULL, NULL, 0, '{}', ?)
            ON CONFLICT(source_id) DO NOTHING
            """,
            (source_id, SourceStatus.OFFLINE.value, now),
        )
        assignments = payload.get("model_assignments")
        if assignments is not None:
            self.set_source_model_assignments(
                source_id, list(assignments)
            )
        else:
            existing = self.db.fetch_one(
                """
                SELECT COUNT(*) AS count FROM source_model_assignments
                WHERE source_id = ?
                """,
                (source_id,),
            )
            if int((existing["count"] if existing else 0) or 0) == 0:
                model = self.db.fetch_one(
                    "SELECT task FROM model_registry WHERE id = ?",
                    (payload["model_id"],),
                )
                task = str(model["task"] if model else "detection")
                self.set_source_model_assignments(
                    source_id,
                    [
                        {
                            "pipeline_role": _pipeline_role_for_task(
                                task
                            ),
                            "task": task,
                            "model_id": payload["model_id"],
                            "use_active": True,
                            "enabled": True,
                        }
                    ],
                )
        return source_id

    def list_source_model_assignments(
        self, source_id: str
    ) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.fetch_all(
                """
                SELECT * FROM source_model_assignments
                WHERE source_id = ? AND enabled = 1
                ORDER BY pipeline_role
                """,
                (source_id,),
            )
        ]

    def set_source_model_assignments(
        self, source_id: str, assignments: list[dict[str, Any]]
    ) -> None:
        if not assignments:
            raise RuntimeError(
                "Une source doit conserver au moins un pipeline d'inférence."
            )
        normalized: list[dict[str, Any]] = []
        seen_roles: set[str] = set()
        for assignment in assignments:
            role = str(assignment["pipeline_role"])
            if role in seen_roles:
                raise RuntimeError(
                    f"Pipeline dupliqué pour la source: {role}"
                )
            seen_roles.add(role)
            model_id = assignment.get("model_id")
            task = str(assignment["task"])
            if model_id:
                model = self.db.fetch_one(
                    "SELECT task FROM model_registry WHERE id = ?",
                    (str(model_id),),
                )
                if model is None:
                    raise RuntimeError(
                        f"Modèle de pipeline introuvable: {model_id}"
                    )
                if str(model["task"]) != task:
                    raise RuntimeError(
                        f"Le modèle {model_id} n'est pas compatible "
                        f"avec la tâche {task}."
                    )
            normalized.append(
                {
                    "pipeline_role": role,
                    "task": task,
                    "model_id": str(model_id) if model_id else None,
                    "use_active": bool(
                        assignment.get("use_active", True)
                    ),
                    "enabled": bool(assignment.get("enabled", True)),
                }
            )
        now = utc_now()
        with self.db.connect() as conn:
            conn.execute(
                "DELETE FROM source_model_assignments WHERE source_id = ?",
                (source_id,),
            )
            conn.executemany(
                """
                INSERT INTO source_model_assignments
                (id, source_id, pipeline_role, task, model_id, use_active,
                 enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f"pipeline-{uuid.uuid4().hex[:12]}",
                        source_id,
                        item["pipeline_role"],
                        item["task"],
                        item["model_id"],
                        int(item["use_active"]),
                        int(item["enabled"]),
                        now,
                        now,
                    )
                    for item in normalized
                ],
            )

    def update_source_state(
        self,
        source_id: str,
        *,
        status: SourceStatus | str,
        fps: float = 0.0,
        last_error: str | None = None,
        last_frame_ts: float | None = None,
        preview_path: str | None = None,
        details_path: str | None = None,
        recording_enabled: bool | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        current = self.db.fetch_one("SELECT * FROM source_state WHERE source_id = ?", (source_id,))
        metrics_json = json.dumps(metrics if metrics is not None else json.loads(current["metrics_json"]) if current else {})
        recording_value = (
            int(recording_enabled)
            if recording_enabled is not None
            else int(current["recording_enabled"])
            if current is not None
            else 0
        )
        status_value = status.value if isinstance(status, SourceStatus) else str(status)
        self.db.execute(
            """
            INSERT INTO source_state
            (source_id, status, fps, last_error, last_frame_ts, preview_path, details_path, recording_enabled, metrics_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                status = excluded.status,
                fps = excluded.fps,
                last_error = excluded.last_error,
                last_frame_ts = excluded.last_frame_ts,
                preview_path = COALESCE(excluded.preview_path, source_state.preview_path),
                details_path = COALESCE(excluded.details_path, source_state.details_path),
                recording_enabled = COALESCE(excluded.recording_enabled, source_state.recording_enabled),
                metrics_json = excluded.metrics_json,
                updated_at = excluded.updated_at
            """,
            (
                source_id,
                status_value,
                fps,
                last_error,
                last_frame_ts,
                preview_path,
                details_path,
                recording_value,
                metrics_json,
                utc_now(),
            ),
        )

    def list_models(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM model_registry ORDER BY created_at DESC")]

    def list_trackers(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM tracker_registry ORDER BY id ASC")]

    def activate_model(self, model_id: str) -> None:
        model = self.db.fetch_one(
            "SELECT task FROM model_registry WHERE id = ?", (model_id,)
        )
        if model is None:
            raise RuntimeError("Modèle introuvable.")
        now = utc_now()
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE model_registry SET is_active = 0, updated_at = ?
                WHERE task = ? AND is_active = 1
                """,
                (now, str(model["task"])),
            )
            conn.execute(
                """
                UPDATE model_registry SET is_active = 1, updated_at = ?
                WHERE id = ?
                """,
                (now, model_id),
            )

    def upsert_site_config(self, config_json: dict[str, Any]) -> None:
        self.db.execute(
            """
            INSERT INTO site_config (id, config_json, updated_at)
            VALUES ('default', ?, ?)
            ON CONFLICT(id) DO UPDATE SET config_json = excluded.config_json, updated_at = excluded.updated_at
            """,
            (json.dumps(config_json), utc_now()),
        )

    def get_site_config(self) -> dict[str, Any]:
        row = self.db.fetch_one("SELECT config_json FROM site_config WHERE id = 'default'")
        return json.loads(row["config_json"]) if row and row["config_json"] else {}

    def recent_events(self, limit: int = 200) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,))]

    def list_parcels(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM global_parcels ORDER BY last_seen_at DESC")]

    def list_tracklets(self, limit: int = 200) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM tracklets ORDER BY ended_at_global DESC LIMIT ?", (limit,))]

    def list_recordings(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM recordings ORDER BY started_at DESC")]

    def list_media_coverage(
        self, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        if session_id is None:
            rows = self.db.fetch_all(
                """
                SELECT * FROM session_media_coverage
                ORDER BY updated_at DESC
                """
            )
        else:
            rows = self.db.fetch_all(
                """
                SELECT * FROM session_media_coverage
                WHERE session_id = ? ORDER BY source_id
                """,
                (session_id,),
            )
        return [dict(row) for row in rows]

    def list_datasets(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM datasets ORDER BY created_at DESC")]

    def list_dataset_sessions(self, dataset_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.fetch_all(
                """
                SELECT ds.dataset_id, ds.session_id, ds.split, cs.name,
                       cs.pipeline_state, cs.started_at, cs.ended_at,
                       COUNT(css.id) AS camera_count,
                       GROUP_CONCAT(css.camera_role) AS camera_roles
                FROM dataset_sessions ds
                JOIN capture_sessions cs ON cs.id = ds.session_id
                LEFT JOIN capture_session_sources css ON css.session_id = ds.session_id
                WHERE ds.dataset_id = ?
                GROUP BY ds.dataset_id, ds.session_id, ds.split
                ORDER BY ds.split, cs.created_at
                """,
                (dataset_id,),
            )
        ]

    def list_dataset_items(self, dataset_id: str, limit: int = 500) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.fetch_all(
                "SELECT * FROM dataset_items WHERE dataset_id = ? ORDER BY created_at DESC LIMIT ?",
                (dataset_id, limit),
            )
        ]

    def list_pipeline_steps(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.fetch_all(
                "SELECT * FROM pipeline_step_runs WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            )
        ]

    def list_training_jobs(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM training_jobs ORDER BY created_at DESC")]

    def list_jobs(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM job_runs ORDER BY started_at DESC")]

    def list_handoff_hypotheses(
        self, status: str | None = None
    ) -> list[dict[str, Any]]:
        if status is None:
            rows = self.db.fetch_all(
                "SELECT * FROM handoff_hypotheses ORDER BY created_at DESC"
            )
        else:
            rows = self.db.fetch_all(
                """
                SELECT * FROM handoff_hypotheses
                WHERE status = ? ORDER BY created_at DESC
                """,
                (status,),
            )
        return [dict(row) for row in rows]


class EventRepository:
    def __init__(self, db: VisionSortDB):
        self.db = db

    def add_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        parcel_id: str | None = None,
        camera_id: str | None = None,
        severity: str = "info",
        *,
        session_id: str | None = None,
        source_id: str | None = None,
        frame_index: int | None = None,
        timestamp_global: float | None = None,
        model_id: str | None = None,
        tracker_id: str | None = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO events
            (id, event_type, parcel_id, camera_id, severity, payload_json, session_id, source_id, frame_index, timestamp_global, model_id, tracker_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                parcel_id,
                camera_id,
                severity,
                json.dumps(payload),
                session_id,
                source_id,
                frame_index,
                timestamp_global,
                model_id,
                tracker_id,
                utc_now(),
            ),
        )
        return event_id


class TrackingRepository:
    def __init__(self, db: VisionSortDB):
        self.db = db

    def upsert_tracklet(self, tracklet: Tracklet, parcel_id: str | None = None, match_result: MatchResult | str = MatchResult.UNMATCHED) -> None:
        self.db.execute(
            """
            INSERT INTO tracklets
            (tracklet_id, parcel_id, session_id, source_id, camera_id, camera_role, local_track_id, started_at_local, ended_at_local, started_at_global, ended_at_global,
             class_name, last_zone_id, frame_count, avg_speed, observation_path, summary_json, match_result, model_id, tracker_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tracklet_id) DO UPDATE SET
                parcel_id = excluded.parcel_id,
                ended_at_local = excluded.ended_at_local,
                ended_at_global = excluded.ended_at_global,
                last_zone_id = excluded.last_zone_id,
                frame_count = excluded.frame_count,
                avg_speed = excluded.avg_speed,
                summary_json = excluded.summary_json,
                match_result = excluded.match_result,
                model_id = excluded.model_id,
                tracker_id = excluded.tracker_id
            """,
            (
                tracklet.tracklet_id,
                parcel_id,
                tracklet.session_id,
                tracklet.source_id,
                tracklet.camera_id,
                tracklet.camera_role,
                tracklet.local_track_id,
                tracklet.started_at_local,
                tracklet.ended_at_local,
                tracklet.started_at_global,
                tracklet.ended_at_global,
                tracklet.class_name,
                tracklet.last_zone_id,
                tracklet.frame_count,
                tracklet.avg_speed,
                tracklet.observation_path,
                json.dumps(tracklet.summary_json),
                str(match_result),
                tracklet.model_id,
                tracklet.tracker_id,
            ),
        )

    def upsert_global_parcel(self, parcel: GlobalParcel) -> None:
        payload = asdict(parcel)
        self.db.execute(
            """
            INSERT INTO global_parcels
            (parcel_id, state, last_camera_id, first_seen_at, last_seen_at, current_tracklet_id, assigned_destination, operator_id, appearance_json, site_validated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(parcel_id) DO UPDATE SET
                state = excluded.state,
                last_camera_id = excluded.last_camera_id,
                last_seen_at = excluded.last_seen_at,
                current_tracklet_id = excluded.current_tracklet_id,
                assigned_destination = excluded.assigned_destination,
                operator_id = excluded.operator_id,
                appearance_json = excluded.appearance_json,
                site_validated = excluded.site_validated
            """,
            (
                payload["parcel_id"],
                str(payload["state"]),
                payload["last_camera_id"],
                payload["first_seen_at"],
                payload["last_seen_at"],
                payload["current_tracklet_id"],
                payload["assigned_destination"],
                payload["operator_id"],
                json.dumps(payload["appearance_signature"] or []),
                0,
            ),
        )


class HandoffHypothesisRepository:
    def __init__(self, db: VisionSortDB):
        self.db = db

    def create(
        self,
        *,
        session_id: str,
        incoming_tracklet_id: str,
        candidates: list[dict[str, Any]],
        expiry_seconds: float,
    ) -> str:
        existing = self.db.fetch_one(
            """
            SELECT id FROM handoff_hypotheses
            WHERE session_id = ? AND incoming_tracklet_id = ? AND status = 'PENDING'
            ORDER BY created_at DESC LIMIT 1
            """,
            (session_id, incoming_tracklet_id),
        )
        if existing is not None:
            return str(existing["id"])
        hypothesis_id = f"hypothesis-{uuid.uuid4().hex[:12]}"
        now = utc_now()
        self.db.execute(
            """
            INSERT INTO handoff_hypotheses
            (id, session_id, incoming_tracklet_id, candidates_json, status,
             resolution_json, expires_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'PENDING', NULL, ?, ?, ?)
            """,
            (
                hypothesis_id,
                session_id,
                incoming_tracklet_id,
                json.dumps(candidates),
                time.time() + max(1.0, float(expiry_seconds)),
                now,
                now,
            ),
        )
        return hypothesis_id

    def get(self, hypothesis_id: str) -> dict[str, Any] | None:
        row = self.db.fetch_one(
            "SELECT * FROM handoff_hypotheses WHERE id = ?", (hypothesis_id,)
        )
        return dict(row) if row else None

    def pending(self, session_id: str | None = None) -> list[dict[str, Any]]:
        self.expire()
        if session_id is None:
            rows = self.db.fetch_all(
                """
                SELECT * FROM handoff_hypotheses
                WHERE status = 'PENDING' ORDER BY created_at
                """
            )
        else:
            rows = self.db.fetch_all(
                """
                SELECT * FROM handoff_hypotheses
                WHERE status = 'PENDING' AND session_id = ? ORDER BY created_at
                """,
                (session_id,),
            )
        return [dict(row) for row in rows]

    def resolve(
        self,
        hypothesis_id: str,
        *,
        outgoing_tracklet_id: str,
        resolution: dict[str, Any],
    ) -> None:
        row = self.get(hypothesis_id)
        if row is None or row["status"] != "PENDING":
            raise RuntimeError("Hypothèse de handoff non résoluble.")
        candidates = json.loads(row["candidates_json"] or "[]")
        if outgoing_tracklet_id not in {
            str(candidate["from_tracklet_id"]) for candidate in candidates
        }:
            raise RuntimeError("Le tracklet choisi ne fait pas partie des candidats.")
        self.db.execute(
            """
            UPDATE handoff_hypotheses
            SET status = 'RESOLVED', resolution_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                json.dumps(
                    {
                        **resolution,
                        "outgoing_tracklet_id": outgoing_tracklet_id,
                    }
                ),
                utc_now(),
                hypothesis_id,
            ),
        )

    def reject(
        self, hypothesis_id: str, *, reason: str = "rejet humain"
    ) -> None:
        row = self.get(hypothesis_id)
        if row is None or row["status"] != "PENDING":
            raise RuntimeError("Hypothèse de handoff non rejetable.")
        self.db.execute(
            """
            UPDATE handoff_hypotheses
            SET status = 'REJECTED', resolution_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (json.dumps({"reason": reason}), utc_now(), hypothesis_id),
        )

    def expire(self, *, now: float | None = None) -> int:
        current = float(now if now is not None else time.time())
        count = self.db.fetch_one(
            """
            SELECT COUNT(*) AS count FROM handoff_hypotheses
            WHERE status = 'PENDING' AND expires_at <= ?
            """,
            (current,),
        )
        self.db.execute(
            """
            UPDATE handoff_hypotheses
            SET status = 'EXPIRED',
                resolution_json = '{"reason":"fenêtre de résolution expirée"}',
                updated_at = ?
            WHERE status = 'PENDING' AND expires_at <= ?
            """,
            (utc_now(), current),
        )
        return int((count["count"] if count else 0) or 0)

class ArtifactRepository:
    def __init__(self, db: VisionSortDB):
        self.db = db

    def add_recording(
        self,
        *,
        source_id: str,
        session_id: str | None,
        segment_path: str,
        started_at: float,
        ended_at: float,
        frame_count: int,
        size_bytes: int,
        recording_id: str | None = None,
        camera_role: str | None = None,
        stream_epoch: int | None = None,
        segment_index: int | None = None,
        fps: float | None = None,
        codec: str | None = None,
        sha256: str | None = None,
        corrupted: bool = False,
        immutable: bool = True,
        metadata: dict[str, Any] | None = None,
        frames: list[dict[str, Any]] | None = None,
    ) -> str:
        rec_id = recording_id or str(uuid.uuid4())
        now = utc_now()
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO recordings
                (id, source_id, session_id, camera_role, stream_epoch,
                 segment_index, segment_path, started_at, ended_at,
                 frame_count, size_bytes, fps, codec, sha256, corrupted,
                 immutable, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec_id,
                    source_id,
                    session_id,
                    camera_role,
                    stream_epoch,
                    segment_index,
                    segment_path,
                    started_at,
                    ended_at,
                    frame_count,
                    size_bytes,
                    fps,
                    codec,
                    sha256,
                    int(corrupted),
                    int(immutable),
                    json.dumps(metadata or {}),
                    now,
                ),
            )
            if session_id and frames:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO recording_frames
                    (id, recording_id, session_id, source_id, camera_role,
                     stream_epoch, frame_index, timestamp_local,
                     timestamp_global, segment_frame_index, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            str(
                                frame.get("id")
                                or f"{rec_id}:{int(frame['segment_frame_index'])}"
                            ),
                            rec_id,
                            session_id,
                            source_id,
                            str(frame.get("camera_role") or camera_role or ""),
                            int(frame.get("stream_epoch") or 0),
                            int(frame["frame_index"]),
                            float(frame["timestamp_local"]),
                            float(frame["timestamp_global"]),
                            int(frame["segment_frame_index"]),
                            now,
                        )
                        for frame in frames
                    ],
                )
        return rec_id

    def upsert_media_coverage(
        self,
        *,
        session_id: str,
        source_id: str,
        archive_required: bool,
        frames_acquired: int,
        frames_processed: int,
        frames_archived: int,
        segments_produced: int,
        segments_corrupted: int,
        bytes_used: int,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        required_frames = max(0, int(frames_processed))
        archived = max(0, int(frames_archived))
        ratio = (
            min(1.0, archived / required_frames)
            if required_frames
            else (1.0 if not archive_required else 0.0)
        )
        frames_unarchived = max(0, int(frames_acquired) - archived)
        status = (
            "NOT_REQUIRED"
            if not archive_required
            else "COMPLETE"
            if ratio >= 1.0 and segments_corrupted <= 0
            else "INSUFFICIENT"
        )
        self.db.execute(
            """
            INSERT INTO session_media_coverage
            (session_id, source_id, archive_required, frames_acquired,
             frames_processed, frames_archived, frames_unarchived,
             segments_produced, segments_corrupted, bytes_used,
             coverage_ratio, status, details_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, source_id) DO UPDATE SET
                archive_required = excluded.archive_required,
                frames_acquired = excluded.frames_acquired,
                frames_processed = excluded.frames_processed,
                frames_archived = excluded.frames_archived,
                frames_unarchived = excluded.frames_unarchived,
                segments_produced = excluded.segments_produced,
                segments_corrupted = excluded.segments_corrupted,
                bytes_used = excluded.bytes_used,
                coverage_ratio = excluded.coverage_ratio,
                status = excluded.status,
                details_json = excluded.details_json,
                updated_at = excluded.updated_at
            """,
            (
                session_id,
                source_id,
                int(archive_required),
                int(frames_acquired),
                required_frames,
                archived,
                frames_unarchived,
                int(segments_produced),
                int(segments_corrupted),
                int(bytes_used),
                ratio,
                status,
                json.dumps(details or {}),
                utc_now(),
            ),
        )
        return {
            "session_id": session_id,
            "source_id": source_id,
            "archive_required": bool(archive_required),
            "frames_acquired": int(frames_acquired),
            "frames_processed": required_frames,
            "frames_archived": archived,
            "frames_unarchived": frames_unarchived,
            "segments_produced": int(segments_produced),
            "segments_corrupted": int(segments_corrupted),
            "bytes_used": int(bytes_used),
            "coverage_ratio": ratio,
            "status": status,
        }

    def upsert_dataset(
        self,
        dataset_id: str,
        name: str,
        root_path: str,
        status: str,
        manifest_path: str,
        data_yaml_path: str,
        summary: dict[str, Any],
        *,
        task: str = "detection",
        dataset_fingerprint: str | None = None,
        finalized_at: str | None = None,
        generation_config: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now()
        existing = self.db.fetch_one(
            """
            SELECT status, task, dataset_fingerprint, finalized_at,
                   generation_config_json
            FROM datasets WHERE id = ?
            """,
            (dataset_id,),
        )
        if existing is not None and existing["status"] == "DATASET_READY":
            raise RuntimeError(
                "Dataset finalisé immuable: créez une nouvelle version pour toute correction."
            )
        if existing is not None:
            task = str(existing["task"])
            dataset_fingerprint = (
                dataset_fingerprint or existing["dataset_fingerprint"]
            )
            finalized_at = finalized_at or existing["finalized_at"]
            if generation_config is None:
                generation_config = json.loads(
                    existing["generation_config_json"] or "{}"
                )
        self.db.execute(
            """
            INSERT INTO datasets
            (id, name, root_path, task, status, manifest_path, data_yaml_path,
             dataset_fingerprint, finalized_at, generation_config_json,
             summary_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                manifest_path = excluded.manifest_path,
                data_yaml_path = excluded.data_yaml_path,
                dataset_fingerprint = excluded.dataset_fingerprint,
                finalized_at = excluded.finalized_at,
                generation_config_json = excluded.generation_config_json,
                summary_json = excluded.summary_json,
                updated_at = excluded.updated_at
            """,
            (
                dataset_id,
                name,
                root_path,
                task,
                status,
                manifest_path,
                data_yaml_path,
                dataset_fingerprint,
                finalized_at,
                json.dumps(generation_config or {}),
                json.dumps(summary),
                now,
                now,
            ),
        )

    def set_dataset_sessions(
        self, dataset_id: str, split_assignments: dict[str, str]
    ) -> None:
        dataset = self.db.fetch_one(
            "SELECT status FROM datasets WHERE id = ?", (dataset_id,)
        )
        if dataset is None:
            raise RuntimeError("Dataset introuvable.")
        if dataset["status"] == "DATASET_READY":
            raise RuntimeError("Les splits d'un dataset finalisé sont immuables.")
        self.db.execute(
            "DELETE FROM dataset_sessions WHERE dataset_id = ?", (dataset_id,)
        )
        now = utc_now()
        self.db.execute_many(
            """
            INSERT INTO dataset_sessions (dataset_id, session_id, split, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (dataset_id, session_id, split, now)
                for session_id, split in sorted(split_assignments.items())
            ],
        )

    def add_dataset_item(
        self,
        *,
        dataset_id: str,
        session_id: str | None,
        sample_group_id: str | None,
        split: str | None,
        source_id: str | None,
        camera_role: str | None,
        frame_index: int | None,
        timestamp_global: float | None,
        image_path: str,
        label_path: str | None,
        annotation_status: str,
        reason: str,
        score: float,
        metadata: dict[str, Any],
    ) -> str:
        item_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO dataset_items
            (id, dataset_id, session_id, sample_group_id, split, source_id, camera_role, frame_index, timestamp_global,
             image_path, label_path, annotation_status, reason, score, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                dataset_id,
                session_id,
                sample_group_id,
                split,
                source_id,
                camera_role,
                frame_index,
                timestamp_global,
                image_path,
                label_path,
                annotation_status,
                reason,
                score,
                json.dumps(metadata),
                utc_now(),
            ),
        )
        return item_id

    def update_dataset_item(self, item_id: str, *, annotation_status: str | None = None, label_path: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        dataset = self.db.fetch_one(
            """
            SELECT d.id AS dataset_id, d.status, d.task, d.data_yaml_path,
                   di.label_path, di.metadata_json
            FROM datasets d
            JOIN dataset_items di ON di.dataset_id = d.id
            WHERE di.id = ?
            """,
            (item_id,),
        )
        if dataset is not None and dataset["status"] == "DATASET_READY":
            raise RuntimeError(
                "Dataset finalisé immuable: créez une nouvelle version pour modifier cet item."
            )
        if dataset is not None and annotation_status == "HUMAN_VALIDATED":
            from visionsort.datasets.integrity import (
                DatasetIntegrityValidator,
            )

            validation_errors = DatasetIntegrityValidator(
                self.db, str(dataset["dataset_id"])
            ).validate_item_label(
                item_id,
                label_path=label_path or dataset["label_path"],
                metadata=metadata,
            )
            if validation_errors:
                task_name = str(dataset["task"]).capitalize()
                raise RuntimeError(
                    f"Validation {task_name} refusée: "
                    + " ".join(validation_errors)
                )
        fields: list[str] = []
        params: list[Any] = []
        if annotation_status is not None:
            fields.append("annotation_status = ?")
            params.append(annotation_status)
        if label_path is not None:
            fields.append("label_path = ?")
            params.append(label_path)
        if metadata is not None:
            fields.append("metadata_json = ?")
            params.append(json.dumps(metadata))
        if not fields:
            return
        params.append(item_id)
        self.db.execute(f"UPDATE dataset_items SET {', '.join(fields)} WHERE id = ?", tuple(params))

    def claim_pipeline_step(
        self,
        session_id: str,
        step: str,
        inputs: dict[str, Any],
        *,
        log_path: str | None = None,
        stale_after_seconds: float = 3600.0,
    ) -> tuple[str, bool, dict[str, Any]]:
        normalized_inputs = json.dumps(
            inputs, sort_keys=True, separators=(",", ":")
        )
        explicit_key = inputs.get("idempotency_key")
        force = bool(inputs.get("force", False))
        idempotency_key = str(explicit_key) if explicit_key else hashlib.sha256(
            f"{session_id}:{step}:{normalized_inputs}".encode("utf-8")
        ).hexdigest()
        if force and not explicit_key:
            idempotency_key = f"{idempotency_key}:{uuid.uuid4().hex}"
        now_iso = utc_now()
        now_ts = time.time()
        with self.db.connect() as conn:
            existing = conn.execute(
                "SELECT * FROM pipeline_step_runs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                outputs = json.loads(existing["outputs_json"] or "{}")
                if existing["status"] == "COMPLETED":
                    return str(existing["id"]), False, outputs
                started_at = float(existing["started_at"] or 0.0)
                if (
                    existing["status"] == "RUNNING"
                    and now_ts - started_at < stale_after_seconds
                ):
                    return str(existing["id"]), False, outputs
                conn.execute(
                    """
                    UPDATE pipeline_step_runs
                    SET status = 'RUNNING', attempt_count = attempt_count + 1,
                        outputs_json = '{}', error_text = NULL, log_path = ?,
                        started_at = ?, ended_at = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (log_path, now_ts, now_iso, existing["id"]),
                )
                return str(existing["id"]), True, {}
            step_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO pipeline_step_runs
                (id, session_id, step, idempotency_key, attempt_count, status,
                 inputs_json, outputs_json, error_text, log_path, started_at,
                 ended_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, 1, 'RUNNING', ?, '{}', NULL, ?, ?, NULL, ?, ?)
                """,
                (
                    step_id,
                    session_id,
                    step,
                    idempotency_key,
                    normalized_inputs,
                    log_path,
                    now_ts,
                    now_iso,
                    now_iso,
                ),
            )
            return step_id, True, {}

    def start_pipeline_step(
        self,
        session_id: str,
        step: str,
        inputs: dict[str, Any],
        log_path: str | None = None,
    ) -> str:
        step_id, _, _ = self.claim_pipeline_step(
            session_id, step, inputs, log_path=log_path
        )
        return step_id

    def finish_pipeline_step(self, step_id: str, *, status: str, outputs: dict[str, Any] | None = None, error_text: str | None = None) -> None:
        self.db.execute(
            "UPDATE pipeline_step_runs SET status = ?, outputs_json = ?, error_text = ?, ended_at = ?, updated_at = ? WHERE id = ?",
            (status, json.dumps(outputs or {}), error_text, time.time(), utc_now(), step_id),
        )

    def cancel_pipeline_step(self, session_id: str, step: str) -> None:
        self.db.execute(
            """
            UPDATE pipeline_step_runs
            SET status = 'CANCELLED', error_text = 'Annulé par utilisateur',
                ended_at = ?, updated_at = ?
            WHERE id = (
                SELECT id FROM pipeline_step_runs
                WHERE session_id = ? AND step = ? AND status = 'RUNNING'
                ORDER BY created_at DESC LIMIT 1
            )
            """,
            (time.time(), utc_now(), session_id, step),
        )

    def add_training_job(self, dataset_id: str, model_id: str, status: str, recipe: dict[str, Any], log_path: str) -> str:
        job_id = str(uuid.uuid4())
        now = utc_now()
        self.db.execute(
            """
            INSERT INTO training_jobs (id, dataset_id, model_id, status, recipe_json, log_path, metrics_json, error_text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, '{}', NULL, ?, ?)
            """,
            (job_id, dataset_id, model_id, status, json.dumps(recipe), log_path, now, now),
        )
        return job_id

    def update_training_job(self, job_id: str, status: str, metrics: dict[str, Any] | None = None, error_text: str | None = None) -> None:
        self.db.execute(
            "UPDATE training_jobs SET status = ?, metrics_json = ?, error_text = ?, updated_at = ? WHERE id = ?",
            (status, json.dumps(metrics or {}), error_text, utc_now(), job_id),
        )


class JobRepository:
    def __init__(self, db: VisionSortDB):
        self.db = db

    def upsert_job_run(self, job_type: str, job_key: str, pid: int, status: str, info: dict[str, Any] | None = None) -> str:
        job_id = f"{job_type}:{job_key}"
        now = utc_now()
        self.db.execute(
            """
            INSERT INTO job_runs (id, job_type, job_key, pid, status, info_json, started_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                pid = excluded.pid,
                status = excluded.status,
                info_json = excluded.info_json,
                updated_at = excluded.updated_at
            """,
            (job_id, job_type, job_key, pid, status, json.dumps(info or {}), now, now),
        )
        return job_id

    def mark_job_stopped(self, job_type: str, job_key: str, status: str = "STOPPED") -> None:
        self.db.execute(
            "UPDATE job_runs SET status = ?, updated_at = ? WHERE id = ?",
            (status, utc_now(), f"{job_type}:{job_key}"),
        )


def detail_path_exists(path_text: str | None) -> bool:
    return bool(path_text and Path(path_text).exists())
