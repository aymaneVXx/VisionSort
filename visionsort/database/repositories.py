from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from visionsort.core.enums import CommandStatus, CommandType, MatchResult, SourceStatus
from visionsort.core.types import GlobalParcel, Tracklet
from visionsort.database.db import VisionSortDB, utc_now


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
        return [dict(row) for row in rows]

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
            rows.append(
                (
                    f"sesssrc-{uuid.uuid4().hex[:10]}",
                    session_id,
                    item["source_id"],
                    item["camera_role"],
                    float(item.get("time_offset_ms", 0.0)),
                    item.get("replay_fps"),
                    now,
                    now,
                )
            )
        if rows:
            self.db.execute_many(
                """
                INSERT INTO capture_session_sources
                (id, session_id, source_id, camera_role, time_offset_ms, replay_fps, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return session_id

    def update_capture_session(self, session_id: str, *, pipeline_state: str | None = None, started_at: float | None = None, ended_at: float | None = None, report_path: str | None = None) -> None:
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
        return source_id

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
                None if recording_enabled is None else int(recording_enabled),
                metrics_json,
                utc_now(),
            ),
        )

    def list_models(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM model_registry ORDER BY created_at DESC")]

    def list_trackers(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM tracker_registry ORDER BY id ASC")]

    def activate_model(self, model_id: str) -> None:
        self.db.execute("UPDATE model_registry SET is_active = 0, updated_at = ?", (utc_now(),))
        self.db.execute("UPDATE model_registry SET is_active = 1, updated_at = ? WHERE id = ?", (utc_now(), model_id))

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

    def list_datasets(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.db.fetch_all("SELECT * FROM datasets ORDER BY created_at DESC")]

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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    ) -> str:
        rec_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO recordings (id, source_id, session_id, segment_path, started_at, ended_at, frame_count, size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (rec_id, source_id, session_id, segment_path, started_at, ended_at, frame_count, size_bytes, utc_now()),
        )
        return rec_id

    def upsert_dataset(self, dataset_id: str, name: str, root_path: str, status: str, manifest_path: str, data_yaml_path: str, summary: dict[str, Any]) -> None:
        now = utc_now()
        self.db.execute(
            """
            INSERT INTO datasets (id, name, root_path, status, manifest_path, data_yaml_path, summary_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                manifest_path = excluded.manifest_path,
                data_yaml_path = excluded.data_yaml_path,
                summary_json = excluded.summary_json,
                updated_at = excluded.updated_at
            """,
            (dataset_id, name, root_path, status, manifest_path, data_yaml_path, json.dumps(summary), now, now),
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

    def start_pipeline_step(self, session_id: str, step: str, inputs: dict[str, Any], log_path: str | None = None) -> str:
        step_id = str(uuid.uuid4())
        now = utc_now()
        self.db.execute(
            """
            INSERT INTO pipeline_step_runs
            (id, session_id, step, status, inputs_json, outputs_json, error_text, log_path, started_at, ended_at, created_at, updated_at)
            VALUES (?, ?, ?, 'RUNNING', ?, '{}', NULL, ?, ?, NULL, ?, ?)
            """,
            (step_id, session_id, step, json.dumps(inputs), log_path, time.time(), now, now),
        )
        return step_id

    def finish_pipeline_step(self, step_id: str, *, status: str, outputs: dict[str, Any] | None = None, error_text: str | None = None) -> None:
        self.db.execute(
            "UPDATE pipeline_step_runs SET status = ?, outputs_json = ?, error_text = ?, ended_at = ?, updated_at = ? WHERE id = ?",
            (status, json.dumps(outputs or {}), error_text, time.time(), utc_now(), step_id),
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
