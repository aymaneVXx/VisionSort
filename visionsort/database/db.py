from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from visionsort.core.enums import ModelStatus, ModelTask
from visionsort.core.paths import DB_PATH, ensure_project_dirs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


DEFAULT_MODELS = [
    {
        "id": "demo_synth_det",
        "name": "Demo Synthetic Detector",
        "task": ModelTask.DETECTION.value,
        "backend": "demo",
        "weights_path": "",
        "status": ModelStatus.CANDIDATE.value,
        "is_active": 1,
        "notes_json": json.dumps(
            {
                "demo_only": True,
                "validated_on_site": False,
                "description": "Détections issues des annotations de démonstration, utilisables uniquement avec DEMO_MODE.",
            }
        ),
    },
    {
        "id": "yolo11n_det",
        "name": "YOLO11n Detection",
        "task": ModelTask.DETECTION.value,
        "backend": "ultralytics",
        "weights_path": "yolo11n.pt",
        "status": ModelStatus.CANDIDATE.value,
        "is_active": 0,
        "notes_json": json.dumps(
            {
                "demo_only": False,
                "validated_on_site": False,
                "description": "Poids préentraînés génériques, à revalider sur données colis réelles.",
            }
        ),
    },
    {
        "id": "yolo11n_seg",
        "name": "YOLO11n Segmentation",
        "task": ModelTask.SEGMENTATION.value,
        "backend": "ultralytics",
        "weights_path": "yolo11n-seg.pt",
        "status": ModelStatus.CANDIDATE.value,
        "is_active": 0,
        "notes_json": json.dumps({"demo_only": False, "validated_on_site": False}),
    },
    {
        "id": "yolo11n_pose",
        "name": "YOLO11n Pose",
        "task": ModelTask.POSE.value,
        "backend": "ultralytics",
        "weights_path": "yolo11n-pose.pt",
        "status": ModelStatus.CANDIDATE.value,
        "is_active": 0,
        "notes_json": json.dumps({"demo_only": False, "validated_on_site": False}),
    },
]


DEFAULT_TRACKERS = [
    {
        "id": "greedy_iou",
        "name": "Greedy IoU Tracker",
        "implementation": "builtin",
        "notes_json": json.dumps({"validated_on_site": False}),
    },
    {
        "id": "bytetrack_cpu",
        "name": "ByteTrack CPU Wrapper",
        "implementation": "ultralytics_bytetrack",
        "notes_json": json.dumps({"validated_on_site": False}),
    },
    {
        "id": "botsort_cpu",
        "name": "BoT-SORT CPU Wrapper",
        "implementation": "ultralytics_botsort",
        "notes_json": json.dumps({"validated_on_site": False}),
    },
]


class VisionSortDB:
    def __init__(self, db_path: Path | None = None):
        ensure_project_dirs()
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    tracker_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS capture_sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    pipeline_state TEXT NOT NULL DEFAULT 'CAPTURED',
                    demo_mode INTEGER NOT NULL DEFAULT 0,
                    site_validated INTEGER NOT NULL DEFAULT 0,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    report_path TEXT,
                    last_dataset_id TEXT,
                    last_training_job_id TEXT,
                    last_candidate_model_id TEXT,
                    started_at REAL,
                    ended_at REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS capture_session_sources (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    camera_role TEXT NOT NULL,
                    time_offset_ms REAL NOT NULL DEFAULT 0,
                    replay_fps REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_state (
                    source_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    fps REAL NOT NULL DEFAULT 0,
                    last_error TEXT,
                    last_frame_ts REAL,
                    preview_path TEXT,
                    details_path TEXT,
                    recording_enabled INTEGER NOT NULL DEFAULT 0,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS commands (
                    id TEXT PRIMARY KEY,
                    command_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'streamlit'
                );
                CREATE TABLE IF NOT EXISTS model_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    task TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    weights_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 0,
                    notes_json TEXT NOT NULL DEFAULT '{}',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    parent_model_id TEXT,
                    created_from_job_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tracker_registry (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    implementation TEXT NOT NULL,
                    notes_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    parcel_id TEXT,
                    camera_id TEXT,
                    severity TEXT NOT NULL DEFAULT 'info',
                    payload_json TEXT NOT NULL,
                    session_id TEXT,
                    source_id TEXT,
                    frame_index INTEGER,
                    timestamp_global REAL,
                    model_id TEXT,
                    tracker_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tracklets (
                    tracklet_id TEXT PRIMARY KEY,
                    parcel_id TEXT,
                    session_id TEXT,
                    source_id TEXT,
                    camera_id TEXT NOT NULL,
                    camera_role TEXT,
                    local_track_id INTEGER NOT NULL,
                    started_at_local REAL,
                    ended_at_local REAL,
                    started_at_global REAL NOT NULL,
                    ended_at_global REAL NOT NULL,
                    class_name TEXT NOT NULL,
                    last_zone_id TEXT,
                    frame_count INTEGER NOT NULL,
                    avg_speed REAL NOT NULL,
                    observation_path TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    match_result TEXT NOT NULL DEFAULT 'UNMATCHED',
                    model_id TEXT,
                    tracker_id TEXT
                );
                CREATE TABLE IF NOT EXISTS global_parcels (
                    parcel_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    last_camera_id TEXT NOT NULL,
                    first_seen_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    current_tracklet_id TEXT NOT NULL,
                    assigned_destination TEXT,
                    operator_id TEXT,
                    appearance_json TEXT NOT NULL DEFAULT '[]',
                    site_validated INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS recordings (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    session_id TEXT,
                    segment_path TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL NOT NULL,
                    frame_count INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    manifest_path TEXT,
                    data_yaml_path TEXT,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_items (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    session_id TEXT,
                    sample_group_id TEXT,
                    split TEXT,
                    source_id TEXT,
                    camera_role TEXT,
                    frame_index INTEGER,
                    timestamp_global REAL,
                    image_path TEXT NOT NULL,
                    label_path TEXT,
                    annotation_status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    score REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pipeline_step_runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    step TEXT NOT NULL,
                    status TEXT NOT NULL,
                    inputs_json TEXT NOT NULL DEFAULT '{}',
                    outputs_json TEXT NOT NULL DEFAULT '{}',
                    error_text TEXT,
                    log_path TEXT,
                    started_at REAL,
                    ended_at REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    recipe_json TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    error_text TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_runs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    job_key TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    info_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS site_config (
                    id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
                CREATE INDEX IF NOT EXISTS idx_commands_status ON commands(status);
                CREATE INDEX IF NOT EXISTS idx_tracklets_camera ON tracklets(camera_id, local_track_id);
                CREATE INDEX IF NOT EXISTS idx_pipeline_steps_session ON pipeline_step_runs(session_id, step, created_at);
                """
            )
            self._migrate(conn)
            now = utc_now()
            for model in DEFAULT_MODELS:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO model_registry
                    (id, name, task, backend, weights_path, status, is_active, notes_json, metrics_json, parent_model_id, created_from_job_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', NULL, NULL, ?, ?)
                    """,
                    (
                        model["id"],
                        model["name"],
                        model["task"],
                        model["backend"],
                        model["weights_path"],
                        model["status"],
                        model["is_active"],
                        model["notes_json"],
                        now,
                        now,
                    ),
                )
            for tracker in DEFAULT_TRACKERS:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO tracker_registry
                    (id, name, implementation, notes_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tracker["id"],
                        tracker["name"],
                        tracker["implementation"],
                        tracker["notes_json"],
                        now,
                        now,
                    ),
                )
            conn.execute(
                "INSERT OR IGNORE INTO site_config (id, config_json, updated_at) VALUES ('default', '{}', ?)",
                (now,),
            )

    def _migrate(self, conn: sqlite3.Connection) -> None:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])

        def has_column(table: str, column: str) -> bool:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            return any(r[1] == column for r in rows)

        def add_column(table: str, definition: str) -> None:
            name = definition.strip().split()[0]
            if has_column(table, name):
                return
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

        if version < 1:
            add_column("events", "session_id TEXT")
            add_column("events", "source_id TEXT")
            add_column("events", "frame_index INTEGER")
            add_column("events", "timestamp_global REAL")
            add_column("events", "model_id TEXT")
            add_column("events", "tracker_id TEXT")
            add_column("recordings", "session_id TEXT")
            add_column("tracklets", "session_id TEXT")
            add_column("tracklets", "source_id TEXT")
            add_column("tracklets", "camera_role TEXT")
            add_column("tracklets", "started_at_local REAL")
            add_column("tracklets", "ended_at_local REAL")
            add_column("tracklets", "started_at_global REAL")
            add_column("tracklets", "ended_at_global REAL")
            add_column("tracklets", "model_id TEXT")
            add_column("tracklets", "tracker_id TEXT")
            conn.execute("PRAGMA user_version = 1")

        if version < 2:
            add_column("capture_sessions", "last_dataset_id TEXT")
            add_column("capture_sessions", "last_training_job_id TEXT")
            add_column("capture_sessions", "last_candidate_model_id TEXT")
            add_column("model_registry", "metrics_json TEXT NOT NULL DEFAULT '{}'")
            add_column("model_registry", "parent_model_id TEXT")
            add_column("model_registry", "created_from_job_id TEXT")
            add_column("dataset_items", "session_id TEXT")
            add_column("dataset_items", "sample_group_id TEXT")
            add_column("dataset_items", "split TEXT")
            add_column("dataset_items", "source_id TEXT")
            add_column("dataset_items", "camera_role TEXT")
            add_column("dataset_items", "frame_index INTEGER")
            add_column("dataset_items", "timestamp_global REAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_step_runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    step TEXT NOT NULL,
                    status TEXT NOT NULL,
                    inputs_json TEXT NOT NULL DEFAULT '{}',
                    outputs_json TEXT NOT NULL DEFAULT '{}',
                    error_text TEXT,
                    log_path TEXT,
                    started_at REAL,
                    ended_at REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_steps_session ON pipeline_step_runs(session_id, step, created_at)")
            conn.execute("PRAGMA user_version = 2")

    def fetch_all(self, query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(conn.execute(query, params).fetchall())

    def fetch_one(self, query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(query, params).fetchone()

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        with self.connect() as conn:
            conn.execute(query, params)

    def execute_many(self, query: str, rows: list[tuple[Any, ...]]) -> None:
        with self.connect() as conn:
            conn.executemany(query, rows)
