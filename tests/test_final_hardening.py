from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from visionsort.core.config import AppConfig, DEFAULT_CONFIG
from visionsort.core.enums import ModelStatus, ParcelState
from visionsort.core.types import Tracklet
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import (
    ControlRepository,
    EventRepository,
    HandoffHypothesisRepository,
    TrackingRepository,
)
from visionsort.deployment.registry import (
    import_baseline_model,
    set_model_status,
    validate_activation_candidate,
    validate_promotion_candidate,
)
from visionsort.runtime.supervisor import RuntimeSupervisor
from visionsort.tracking.engine import GlobalParcelTracker
from visionsort.training.pipeline import _promotion_eligible


def _insert_model(
    db: VisionSortDB,
    model_id: str,
    *,
    task: str = "detection",
    status: str = "ARCHIVED",
    active: int = 0,
    metrics: dict | None = None,
    job_id: str | None = None,
) -> None:
    now = utc_now()
    db.execute(
        """
        INSERT INTO model_registry
        (id, name, task, backend, weights_path, status, is_active,
         notes_json, metrics_json, parent_model_id,
         created_from_job_id, created_at, updated_at)
        VALUES (?, ?, ?, 'demo', '', ?, ?, '{}', ?, NULL, ?, ?, ?)
        """,
        (
            model_id,
            model_id,
            task,
            status,
            active,
            json.dumps(metrics or {}),
            job_id,
            now,
            now,
        ),
    )


def _switch_supervisor(db: VisionSortDB) -> RuntimeSupervisor:
    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.db = db
    supervisor.config = AppConfig(
        values={
            "runtime": {
                "model_switch_validation_timeout_seconds": 0.01,
                "model_unload_timeout_seconds": 0.01,
            }
        }
    )
    supervisor.active_model_id = None
    supervisor.active_model_ids_by_task = {}
    supervisor.active_runtime_models_by_task = {}
    supervisor.active_source_pipelines = {}
    supervisor.active_source_sessions = {}
    supervisor.rollback_model_holds = set()
    supervisor.ensure_model_loaded = lambda model_id: None
    supervisor.safe_unload_model = lambda model_id: {
        "model_id": model_id,
        "status": "UNLOADED",
        "unloaded": True,
    }
    return supervisor


def test_candidate_archived_is_not_activable_and_active_model_is_protected(
    tmp_path,
):
    db = VisionSortDB(tmp_path / "lifecycle.db")
    db.initialize()
    _insert_model(
        db,
        "candidate",
        status=ModelStatus.CANDIDATE.value,
    )
    set_model_status(db, "candidate", ModelStatus.ARCHIVED.value)
    with pytest.raises(RuntimeError, match="archive depuis CANDIDATE"):
        validate_activation_candidate(db, "candidate")

    db.execute("UPDATE model_registry SET is_active = 0 WHERE task = 'detection'")
    _insert_model(db, "active", status="CHAMPION", active=1)
    with pytest.raises(RuntimeError, match="modele actif"):
        set_model_status(db, "active", ModelStatus.ARCHIVED.value)
    with pytest.raises(RuntimeError, match="modele actif"):
        set_model_status(db, "active", ModelStatus.REJECTED.value)


def test_failed_first_switch_removes_route_and_invalid_promotion_loads_nothing(
    tmp_path,
):
    db = VisionSortDB(tmp_path / "switch.db")
    db.initialize()
    _insert_model(db, "first")
    supervisor = _switch_supervisor(db)
    supervisor.validate_model_routing = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("validation failed")
    )
    with pytest.raises(RuntimeError, match="validation failed"):
        supervisor.switch_runtime_model("first")
    assert supervisor.runtime_route("detection")["model_id"] is None
    assert "detection" not in supervisor.active_model_ids_by_task

    _insert_model(
        db,
        "invalid-candidate",
        status=ModelStatus.CANDIDATE.value,
        metrics={"precision": 0.9},
    )
    loaded: list[str] = []
    supervisor.ensure_model_loaded = loaded.append
    with pytest.raises(RuntimeError, match="Promotion refus"):
        supervisor.switch_runtime_model("invalid-candidate", promote=True)
    assert loaded == []


def test_routing_generation_uses_sqlite_history_after_restart(tmp_path):
    db = VisionSortDB(tmp_path / "generation.db")
    db.initialize()
    db.execute("UPDATE model_registry SET is_active = 0")
    _insert_model(db, "active-v7", status="CHAMPION", active=1)
    now = utc_now()
    db.execute(
        """
        INSERT INTO model_activation_history
        (id, task, previous_model_id, activated_model_id,
         routing_generation, status, runtime_applied, actor, reason,
         source_ids_json, activated_at, completed_at, metadata_json)
        VALUES ('generation-7', 'detection', NULL, 'active-v7', 7,
                'ACTIVE', 1, 'test', 'restart', '[]', ?, ?, '{}')
        """,
        (now, now),
    )
    supervisor = RuntimeSupervisor(db_path=db.db_path)
    try:
        assert supervisor.runtime_route("detection")["generation"] == 7
        assert supervisor.last_routing_generation("detection") == 7
    finally:
        supervisor.manager.shutdown()


def test_baseline_import_stages_immutable_ultralytics_artifact(
    tmp_path, monkeypatch
):
    import visionsort.deployment.registry as registry
    import visionsort.inference.engine as inference_engine

    root = tmp_path / "registry-root"
    versions = root / "models"
    versions.mkdir(parents=True)
    source = tmp_path / "parcel.pt"
    source.write_bytes(b"immutable-ultralytics-weights")
    monkeypatch.setattr(registry, "ROOT_DIR", root)
    monkeypatch.setattr(registry, "MODELS_DIR", versions)
    monkeypatch.setattr(inference_engine, "ROOT_DIR", root)
    db = VisionSortDB(tmp_path / "baseline.db")
    db.initialize()

    result = import_baseline_model(
        db,
        source_path=str(source),
        task="detection",
        name="Parcel baseline",
    )

    row = db.fetch_one(
        "SELECT * FROM model_registry WHERE id = ?", (result["model_id"],)
    )
    notes = json.loads(row["notes_json"])
    staged = root / row["weights_path"]
    assert row["status"] == ModelStatus.BASELINE.value
    assert row["backend"] == "ultralytics"
    assert staged.read_bytes() == source.read_bytes()
    assert notes["artifact_sha256"] == result["sha256"]
    assert notes["validated_on_site"] is False
    assert validate_activation_candidate(db, result["model_id"])["id"] == result[
        "model_id"
    ]


def _source(repo: ControlRepository, source_id: str, role: str) -> None:
    repo.upsert_source(
        {
            "id": source_id,
            "name": source_id,
            "role": role,
            "source_type": "REPLAY",
            "uri": "unused.mp4",
            "model_id": "demo_synth_det",
            "tracker_id": "greedy_iou",
            "enabled": True,
        }
    )


def _session_supervisor(db: VisionSortDB) -> RuntimeSupervisor:
    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.db = db
    supervisor.db_path = db.db_path
    supervisor.control_repo = ControlRepository(db)
    supervisor.base_config_values = DEFAULT_CONFIG
    supervisor.config = AppConfig(values=DEFAULT_CONFIG)
    supervisor.active_source_sessions = {}
    supervisor.session_site_configs = {}
    supervisor.camera_processes = {}
    return supervisor


def test_session_start_is_atomic_and_site_config_is_snapshotted(tmp_path):
    db = VisionSortDB(tmp_path / "sessions.db")
    db.initialize()
    repo = ControlRepository(db)
    _source(repo, "source-c1", "C1")
    _source(repo, "source-c2", "C2")
    config_a = {
        "tracking": {
            "zones": {
                "C1": [
                    {
                        "zone_id": "door-a",
                        "kind": "exit",
                        "x1": 0.8,
                        "y1": 0.0,
                        "x2": 1.0,
                        "y2": 1.0,
                    }
                ]
            }
        }
    }
    repo.upsert_site_config(config_a)
    session_id = repo.create_capture_session(
        name="atomic",
        demo_mode=True,
        sources=[
            {"source_id": "source-c1", "camera_role": "C1"},
            {"source_id": "source-c2", "camera_role": "C2"},
        ],
        config={},
    )
    supervisor = _session_supervisor(db)
    stopped: list[str] = []

    def fake_start(source_id: str, **kwargs) -> None:
        supervisor.active_source_sessions[source_id] = session_id
        if source_id == "source-c2":
            raise RuntimeError("second source failed")

    def fake_stop(source_id: str) -> None:
        stopped.append(source_id)
        supervisor.active_source_sessions.pop(source_id, None)

    supervisor.start_source = fake_start  # type: ignore[method-assign]
    supervisor.stop_source = fake_stop  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="Echec atomique"):
        supervisor.start_session(session_id)
    failed = repo.get_capture_session(session_id)
    assert failed["runtime_status"] == "FAILED"
    assert "second source failed" in failed["start_error"]
    assert supervisor.active_source_sessions == {}
    assert set(stopped) == {"source-c1", "source-c2"}
    assert json.loads(failed["site_config_snapshot_json"]) == config_a
    with pytest.raises(RuntimeError, match="terminee"):
        supervisor.start_session(session_id)


def test_duplicate_source_and_role_mismatch_are_rejected(tmp_path):
    db = VisionSortDB(tmp_path / "session-invariants.db")
    db.initialize()
    repo = ControlRepository(db)
    _source(repo, "source-c1", "C1")
    with pytest.raises(RuntimeError, match="meme source"):
        repo.create_capture_session(
            name="duplicate",
            demo_mode=True,
            sources=[
                {"source_id": "source-c1", "camera_role": "C1"},
                {"source_id": "source-c1", "camera_role": "C2"},
            ],
            config={},
        )
    with pytest.raises(RuntimeError, match="Role incoherent"):
        repo.create_capture_session(
            name="role",
            demo_mode=True,
            sources=[{"source_id": "source-c1", "camera_role": "C2"}],
            config={},
        )


def _tracklet(
    tracklet_id: str,
    role: str,
    source_id: str,
    *,
    started: float,
    ended: float,
    first_zone: str,
    last_zone: str,
) -> Tracklet:
    return Tracklet(
        tracklet_id=tracklet_id,
        session_id="session-a",
        source_id=source_id,
        camera_id=source_id,
        camera_role=role,
        local_track_id=1,
        started_at_local=started,
        ended_at_local=ended,
        started_at_global=started,
        ended_at_global=ended,
        class_name="parcel",
        first_bbox=(0.0, 0.0, 12.0, 10.0),
        last_bbox=(1.0, 0.0, 13.0, 10.0),
        avg_speed=4.0,
        last_zone_id=last_zone,
        frame_count=8,
        observation_path="details.jsonl",
        summary_json={
            "first_zone_id": first_zone,
            "last_zone_id": last_zone,
            "avg_dimensions": [12.0, 10.0],
            "avg_velocity": [4.0, 0.0],
            "appearance_embedding": [1.0, 0.0],
        },
        model_id="demo_synth_det",
        tracker_id="greedy_iou",
    )


def test_global_parcel_keeps_carry_state_and_drops_in_semantic_destination(
    tmp_path,
):
    db = VisionSortDB(tmp_path / "global-events.db")
    db.initialize()
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["tracking"]["site_topology"]["edges"] = [
        {
            "from_role": "C2",
            "to_role": "C3",
            "min_transit_s": 0.1,
            "max_transit_s": 5.0,
        }
    ]
    config["tracking"]["zones"] = {
        "C2": [
            {
                "zone_id": "handoff-door-renamed",
                "kind": "exit",
                "x1": 0.8,
                "y1": 0.0,
                "x2": 1.0,
                "y2": 1.0,
            }
        ],
        "C3": [
            {
                "zone_id": "arrival-renamed",
                "kind": "entry",
                "x1": 0.0,
                "y1": 0.0,
                "x2": 0.2,
                "y2": 1.0,
            },
            {
                "zone_id": "dock-renamed",
                "kind": "destination",
                "x1": 0.5,
                "y1": 0.0,
                "x2": 1.0,
                "y2": 1.0,
            },
        ],
    }
    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.db = db
    supervisor.config = AppConfig(values=config)
    supervisor.session_site_configs = {"session-a": config}
    supervisor.tracking_repo = TrackingRepository(db)
    supervisor.hypothesis_repo = HandoffHypothesisRepository(db)
    supervisor.event_repo = EventRepository(db)
    supervisor.global_tracker = GlobalParcelTracker(
        config["tracking"]["site_topology"]["edges"],
        {"source-c2": "C2", "source-c3": "C3"},
        zones_by_role=config["tracking"]["zones"],
    )

    for event_type, timestamp in (
        ("pickup_candidate", 1.2),
        ("parcel_picked", 1.4),
        ("parcel_carried", 1.8),
    ):
        supervisor.event_repo.add_event(
            event_type,
            {"state": event_type},
            session_id="session-a",
            source_id="source-c2",
            timestamp_global=timestamp,
            local_parcel_key="source-c2:1",
        )
    c2 = _tracklet(
        "track-c2",
        "C2",
        "source-c2",
        started=1.0,
        ended=2.0,
        first_zone="handoff-door-renamed",
        last_zone="handoff-door-renamed",
    )
    supervisor.handle_tracklet(asdict(c2))
    parcel_id = supervisor.global_tracker.tracklet_to_parcel["track-c2"]
    assert supervisor.global_tracker.parcels[parcel_id].state == ParcelState.CARRIED

    for event_type, timestamp in (
        ("destination_observed", 3.2),
        ("destination_confirmed", 3.7),
    ):
        supervisor.event_repo.add_event(
            event_type,
            {"destination_zone": "dock-renamed"},
            session_id="session-a",
            source_id="source-c3",
            timestamp_global=timestamp,
            local_parcel_key="source-c3:1",
        )
    c3 = _tracklet(
        "track-c3",
        "C3",
        "source-c3",
        started=3.0,
        ended=4.0,
        first_zone="arrival-renamed",
        last_zone="dock-renamed",
    )
    supervisor.handle_tracklet(asdict(c3))

    assert supervisor.global_tracker.tracklet_to_parcel["track-c3"] == parcel_id
    parcel = db.fetch_one(
        """
        SELECT state, expected_destination, observed_destination,
               destination_result
        FROM global_parcels WHERE parcel_id = ?
        """,
        (parcel_id,),
    )
    assert parcel["state"] == "DROPPED"
    assert parcel["expected_destination"] is None
    assert parcel["observed_destination"] == "dock-renamed"
    assert parcel["destination_result"] == "DESTINATION_UNVERIFIED"
    bound = db.fetch_all(
        """
        SELECT event_type, parcel_id FROM events
        WHERE local_parcel_key IN ('source-c2:1', 'source-c3:1')
        """
    )
    assert bound
    assert all(row["parcel_id"] == parcel_id for row in bound)
    assert any(row["event_type"] == "parcel_dropped" for row in bound)


def test_task_aware_promotion_does_not_require_parcel_count_for_pose_or_masks():
    criteria = {
        "precision_min": 0.5,
        "recall_min": 0.5,
        "map50_min": 0.5,
        "count_accuracy_min": 0.8,
        "merge_rate_max": 0.15,
        "fps_min": 5.0,
    }
    segmentation, segmentation_failures = _promotion_eligible(
        {
            "mask_precision": 0.8,
            "mask_recall": 0.8,
            "mask_mAP50": 0.8,
            "fps": 20.0,
        },
        criteria=criteria,
        frozen_test=True,
        task="segmentation",
    )
    pose, pose_failures = _promotion_eligible(
        {
            "pose_precision": 0.8,
            "pose_recall": 0.8,
            "pose_mAP50": 0.8,
            "fps": 20.0,
        },
        criteria=criteria,
        frozen_test=True,
        task="pose",
    )
    detection, detection_failures = _promotion_eligible(
        {"precision": 0.8, "recall": 0.8, "mAP50": 0.8, "fps": 20.0},
        criteria=criteria,
        frozen_test=True,
        task="detection",
    )
    assert segmentation and not segmentation_failures
    assert pose and not pose_failures
    assert not detection
    assert "count_accuracy_min:UNAVAILABLE" in detection_failures


def test_real_test_auto_labels_block_promotion(tmp_path, monkeypatch):
    db = VisionSortDB(tmp_path / "real-gate.db")
    db.initialize()
    now = utc_now()
    db.execute(
        """
        INSERT INTO capture_sessions
        (id, name, pipeline_state, demo_mode, site_validated, config_json,
         created_at, updated_at)
        VALUES ('real-session', 'Real', 'DATASET_READY', 0, 0, '{}', ?, ?)
        """,
        (now, now),
    )
    db.execute(
        """
        INSERT INTO datasets
        (id, name, root_path, status, manifest_path, data_yaml_path,
         summary_json, task, created_at, updated_at)
        VALUES ('real-dataset', 'Real', '.', 'DATASET_READY', 'm.csv',
                'data.yaml', '{}', 'detection', ?, ?)
        """,
        (now, now),
    )
    db.execute(
        """
        INSERT INTO dataset_items
        (id, dataset_id, session_id, split, source_id, camera_role,
         frame_index, timestamp_global, image_path, label_path,
         annotation_status, reason, score, metadata_json, created_at)
        VALUES ('real-test-item', 'real-dataset', 'real-session', 'test',
                'source', 'C1', 1, 1.0, 'image.jpg', 'label.txt',
                'AUTO_ACCEPTED', 'pseudo', 1.0, '{}', ?)
        """,
        (now,),
    )
    db.execute(
        """
        INSERT INTO training_jobs
        (id, dataset_id, model_id, status, recipe_json, log_path,
         metrics_json, created_at, updated_at)
        VALUES ('real-job', 'real-dataset', 'demo_synth_det', 'COMPLETED',
                '{}', 'train.log', '{}', ?, ?)
        """,
        (now, now),
    )
    metrics = {
        "precision": 0.9,
        "recall": 0.9,
        "mAP50": 0.9,
        "count_accuracy": 0.9,
        "merge_rate": 0.01,
        "fps": 20.0,
        "promotion_eligible": True,
        "promotion_criteria": {"precision_min": 0.5},
        "test": {"status": "COMPLETED", "frozen": True},
    }
    _insert_model(
        db,
        "real-candidate",
        status="CANDIDATE",
        metrics=metrics,
        job_id="real-job",
    )

    class ValidIntegrity:
        def __init__(self, *_args, **_kwargs):
            pass

        def validate(self):
            return {"valid": True, "errors": []}

    monkeypatch.setattr(
        "visionsort.datasets.integrity.DatasetIntegrityValidator",
        ValidIntegrity,
    )
    monkeypatch.setattr(
        "visionsort.datasets.pipeline.verify_dataset_fingerprint",
        lambda *_args, **_kwargs: {"valid": True},
    )
    with pytest.raises(RuntimeError, match="HUMAN_VALIDATED"):
        validate_promotion_candidate(db, "real-candidate")
