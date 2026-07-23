import json

from visionsort.core.config import AppConfig
from visionsort.core.enums import CommandType, ModelStatus
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import (
    ControlRepository,
    EventRepository,
)
from visionsort.runtime.supervisor import RuntimeSupervisor


class _AliveProcess:
    def is_alive(self) -> bool:
        return True


def test_activation_reloads_inference_worker_and_next_job_uses_active_model(tmp_path):
    db = VisionSortDB(tmp_path / "activation.db")
    db.initialize()
    now = utc_now()
    db.execute(
        """
        INSERT INTO model_registry
        (id, name, task, backend, weights_path, status, is_active, notes_json,
         metrics_json, parent_model_id, created_from_job_id, created_at, updated_at)
        VALUES ('deployable', 'Deployable', 'detection', 'demo', 'immutable/best.pt',
                ?, 0, '{}', '{}', NULL, NULL, ?, ?)
        """,
        (ModelStatus.ARCHIVED.value, now, now),
    )
    repo = ControlRepository(db)
    command_id = repo.enqueue_command(
        CommandType.ACTIVATE_MODEL, {"model_id": "deployable"}
    )
    command = db.fetch_one("SELECT * FROM commands WHERE id = ?", (command_id,))
    assert command is not None

    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.db = db
    supervisor.config = AppConfig(
        values={"runtime": {"model_selection": "active_registry"}}
    )
    supervisor.control_repo = repo
    supervisor.event_repo = EventRepository(db)
    supervisor.inference_process = _AliveProcess()
    supervisor.active_model_id = "old-model"
    loaded: list[str] = []

    def fake_ensure_model_loaded(model_id: str) -> None:
        loaded.append(model_id)
        supervisor.active_model_id = model_id

    supervisor.ensure_model_loaded = fake_ensure_model_loaded
    supervisor.handle_command(dict(command))

    active = db.fetch_one(
        "SELECT id FROM model_registry WHERE is_active = 1 ORDER BY updated_at DESC LIMIT 1"
    )
    completed = db.fetch_one("SELECT * FROM commands WHERE id = ?", (command_id,))
    assert active is not None
    assert active["id"] == "deployable"
    assert loaded == ["deployable"]
    assert completed is not None
    assert completed["status"] == "COMPLETED"
    assert supervisor.runtime_model_id("configured-old") == "deployable"
    event = db.fetch_one(
        "SELECT payload_json FROM events WHERE event_type = 'command_completed' ORDER BY created_at DESC LIMIT 1"
    )
    assert event is not None
    assert json.loads(event["payload_json"])["result"]["runtime_reloaded"] is True
