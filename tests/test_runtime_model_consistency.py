from visionsort.core.enums import JobType
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import (
    ControlRepository,
    JobRepository,
)
from visionsort.runtime.supervisor import RuntimeSupervisor


def _supervisor_state(db: VisionSortDB) -> RuntimeSupervisor:
    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.db = db
    supervisor.active_model_id = "demo_synth_det"
    supervisor.active_model_ids_by_task = {
        "detection": "demo_synth_det"
    }
    supervisor.active_runtime_models_by_task = {
        "detection": {
            "task": "detection",
            "model_id": "demo_synth_det",
            "generation": 3,
            "activated_at": 1.0,
        }
    }
    supervisor.active_source_pipelines = {}
    supervisor.loaded_model_ids = {"demo_synth_det"}
    supervisor.control_flags = {}
    supervisor.rollback_model_holds = set()
    supervisor.last_model_load_error = {}
    supervisor.last_model_unload_error = {}
    supervisor.inference_runtime_snapshot = {}
    return supervisor


def test_runtime_registry_consistency_reports_matching_state(tmp_path):
    db = VisionSortDB(tmp_path / "consistent.db")
    db.initialize()
    supervisor = _supervisor_state(db)

    result = supervisor.runtime_registry_consistency(
        worker_status={
            "loaded_model_ids": ["demo_synth_det"],
            "inflight_by_model": {"demo_synth_det": 2},
        }
    )

    assert result["consistent"] is True
    assert result["tasks"] == [
        {
            "task": "detection",
            "registry_model_id": "demo_synth_det",
            "registry_active_ids": ["demo_synth_det"],
            "runtime_model_id": "demo_synth_det",
            "routing_generation": 3,
            "loaded": True,
            "references": 1,
            "inflight": 2,
            "consistent": True,
            "errors": [],
        }
    ]


def test_runtime_registry_consistency_exposes_route_mismatch(tmp_path):
    db = VisionSortDB(tmp_path / "mismatch.db")
    db.initialize()
    supervisor = _supervisor_state(db)
    supervisor.active_runtime_models_by_task["detection"] = {
        "task": "detection",
        "model_id": "yolo11n_det",
        "generation": 4,
        "activated_at": 2.0,
    }
    supervisor.active_model_ids_by_task["detection"] = "yolo11n_det"
    supervisor.loaded_model_ids = set()

    result = supervisor.runtime_registry_consistency(
        worker_status={
            "loaded_model_ids": [],
            "inflight_by_model": {},
        }
    )

    assert result["consistent"] is False
    assert any(
        "registre et routage runtime différents" in error
        for error in result["errors"]
    )
    assert any(
        "modèle routé non chargé" in error
        for error in result["errors"]
    )


def test_runtime_model_state_is_readable_by_ui_repository(tmp_path):
    db = VisionSortDB(tmp_path / "published.db")
    db.initialize()
    JobRepository(db).upsert_job_run(
        JobType.GPU_INFERENCE.value,
        "shared",
        42,
        "RUNNING",
        {
            "published_at": "2026-01-01T00:00:00+00:00",
            "consistency": {"consistent": True, "tasks": []},
        },
    )

    state = ControlRepository(db).get_runtime_model_state()

    assert state["job_status"] == "RUNNING"
    assert state["consistency"]["consistent"] is True
