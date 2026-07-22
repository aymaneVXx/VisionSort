import json

from visionsort.core.enums import CommandType
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import ArtifactRepository, ControlRepository
from visionsort.runtime.supervisor import RuntimeSupervisor


def test_supervisor_update_dataset_item_triggers_finalize(tmp_path):
    db = VisionSortDB(tmp_path / "visionsort.db")
    db.initialize()
    repo = ControlRepository(db)
    now = utc_now()

    db.execute(
        """
        INSERT INTO capture_sessions (id, name, pipeline_state, demo_mode, site_validated, config_json, report_path, last_dataset_id, started_at, ended_at, created_at, updated_at)
        VALUES (?, ?, 'REVIEW_PENDING', 1, 0, '{}', NULL, ?, NULL, NULL, ?, ?)
        """,
        ("session-cmd", "Session Cmd", "dataset-cmd", now, now),
    )
    db.execute(
        """
        INSERT INTO datasets (id, name, root_path, status, manifest_path, data_yaml_path, summary_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "dataset-cmd",
            "Dataset Cmd",
            "data/datasets/dataset-cmd",
            "REVIEW_PENDING",
            "data/datasets/dataset-cmd/manifest.csv",
            "data/datasets/dataset-cmd/data.yaml",
            "{}",
            now,
            now,
        ),
    )
    db.execute(
        """
        INSERT INTO dataset_items (id, dataset_id, session_id, sample_group_id, split, source_id, camera_role, frame_index, timestamp_global, image_path, label_path, annotation_status, reason, score, metadata_json, created_at)
        VALUES (?, ?, ?, NULL, 'train', 'src1', 'C1', 1, 1.0, 'img.jpg', 'lbl.txt', ?, 'manual', 0.5, '{}', ?)
        """,
        ("item-cmd", "dataset-cmd", "session-cmd", "NEEDS_REVIEW", now),
    )

    supervisor = RuntimeSupervisor()
    supervisor.db = db
    supervisor.control_repo = ControlRepository(db)
    supervisor.artifact_repo = ArtifactRepository(db)
    supervisor.pipeline_processes = {}

    started = {}

    def fake_start_pipeline_step(*, session_id: str, step: str, params: dict):
        started["session_id"] = session_id
        started["step"] = step
        started["params"] = params
        return "job-test"

    supervisor.start_pipeline_step = fake_start_pipeline_step  # type: ignore[method-assign]
    command_id = repo.enqueue_command(CommandType.UPDATE_DATASET_ITEM, {"item_id": "item-cmd", "annotation_status": "HUMAN_VALIDATED"})
    command = db.fetch_one("SELECT * FROM commands WHERE id = ?", (command_id,))
    assert command is not None
    supervisor.handle_command(dict(command))

    updated = db.fetch_one("SELECT annotation_status FROM dataset_items WHERE id = ?", ("item-cmd",))
    assert updated is not None
    assert updated["annotation_status"] == "HUMAN_VALIDATED"
    assert started["session_id"] == "session-cmd"
    assert started["step"] == "FINALIZE_DATASET"
    assert started["params"]["dataset_id"] == "dataset-cmd"
