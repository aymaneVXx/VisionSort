from visionsort.core.enums import CommandType
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import ArtifactRepository, ControlRepository, EventRepository, JobRepository, TrackingRepository
from visionsort.runtime.supervisor import RuntimeSupervisor


def test_stop_session_triggers_process_step(tmp_path):
    db = VisionSortDB(tmp_path / "visionsort.db")
    db.initialize()
    repo = ControlRepository(db)
    session_id = repo.create_capture_session(name="pytest stop", demo_mode=True, sources=[], config={})
    cmd_id = repo.enqueue_command(CommandType.STOP_SESSION, {"session_id": session_id})
    command = db.fetch_one("SELECT * FROM commands WHERE id = ?", (cmd_id,))
    assert command is not None

    supervisor = RuntimeSupervisor()
    supervisor.db = db
    supervisor.control_repo = ControlRepository(db)
    supervisor.artifact_repo = ArtifactRepository(db)
    supervisor.event_repo = EventRepository(db)
    supervisor.job_repo = JobRepository(db)
    supervisor.tracking_repo = TrackingRepository(db)
    supervisor.pipeline_processes = {}
    supervisor.camera_processes = {}
    supervisor.control_flags = {}
    supervisor.training_processes = {}

    called = {}

    def fake_stop_session(sid: str) -> None:
        called["stop_session"] = sid
        supervisor.control_repo.update_capture_session(sid, ended_at=1.0)

    def fake_start_pipeline_step(*, session_id: str, step: str, params: dict):
        called["session_id"] = session_id
        called["step"] = step
        called["params"] = params
        return "job-key"

    supervisor.stop_session = fake_stop_session  # type: ignore[method-assign]
    supervisor.start_pipeline_step = fake_start_pipeline_step  # type: ignore[method-assign]

    supervisor.handle_command(dict(command))
    assert called["stop_session"] == session_id
    assert called["session_id"] == session_id
    assert str(called["step"]).upper() == "PROCESS_SESSION"
