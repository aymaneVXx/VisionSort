import json

import cv2
import numpy as np

from visionsort.core.enums import CommandType
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import ArtifactRepository, ControlRepository
from visionsort.runtime.supervisor import RuntimeSupervisor


def test_supervisor_update_dataset_item_triggers_finalize(tmp_path):
    db = VisionSortDB(tmp_path / "visionsort.db")
    db.initialize()
    repo = ControlRepository(db)
    now = utc_now()
    root = tmp_path / "dataset-cmd"
    root.mkdir()
    image = root / "img.jpg"
    label = root / "lbl.txt"
    data_yaml = root / "data.yaml"
    cv2.imwrite(
        str(image), np.zeros((32, 32, 3), dtype=np.uint8)
    )
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        "test: images/test\ntask: detection\nnames:\n  0: parcel\n",
        encoding="utf-8",
    )

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
            str(root),
            "REVIEW_PENDING",
            str(root / "manifest.csv"),
            str(data_yaml),
            "{}",
            now,
            now,
        ),
    )
    db.execute(
        """
        INSERT INTO dataset_items (id, dataset_id, session_id, sample_group_id, split, source_id, camera_role, frame_index, timestamp_global, image_path, label_path, annotation_status, reason, score, metadata_json, created_at)
        VALUES (?, ?, ?, NULL, 'train', 'src1', 'C1', 1, 1.0, ?, ?, ?,
                'manual', 0.5, '{"instance_count":1}', ?)
        """,
        (
            "item-cmd",
            "dataset-cmd",
            "session-cmd",
            str(image),
            str(label),
            "NEEDS_REVIEW",
            now,
        ),
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
