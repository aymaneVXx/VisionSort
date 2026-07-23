import json
import queue
import threading
import time

import numpy as np

from visionsort.core.config import AppConfig
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import ControlRepository
from visionsort.deployment.registry import activate_model
from visionsort.inference.engine import inference_worker_loop
from visionsort.runtime.supervisor import RuntimeSupervisor


def _insert_demo_pose(
    db: VisionSortDB,
    model_id: str = "demo-pose",
    *,
    active: bool = True,
    status: str = "CHAMPION",
) -> None:
    now = utc_now()
    db.execute(
        """
        INSERT INTO model_registry
        (id, name, task, backend, weights_path, status, is_active,
         notes_json, metrics_json, parent_model_id, created_from_job_id,
         created_at, updated_at)
        VALUES (?, ?, 'pose', 'demo', '', ?, ?, '{}', '{}', NULL,
                NULL, ?, ?)
        """,
        (model_id, model_id, status, int(active), now, now),
    )


def _next_result(result_queue: queue.Queue, kind: str) -> dict:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        result = result_queue.get(timeout=1.0)
        if result["kind"] == kind:
            return result
    raise AssertionError(f"Résultat {kind} non reçu")


def test_multi_model_worker_caches_shared_models_and_reloads_independently(
    tmp_path,
):
    db_path = tmp_path / "multi-model.db"
    db = VisionSortDB(db_path)
    db.initialize()
    _insert_demo_pose(db)
    source_uri = tmp_path / "camera.avi"
    source_uri.with_suffix(".jsonl").write_text(
        json.dumps(
            {
                "frame_index": 0,
                "class_name": "person",
                "confidence": 0.9,
                "bbox": [1, 1, 7, 7],
                "keypoints": [[2.0, 2.0, 2.0]] * 17,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    request_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=inference_worker_loop,
        args=(
            request_queue,
            result_queue,
            stop_event,
            str(db_path),
            {"app": {"demo_mode": True}},
        ),
        daemon=True,
    )
    worker.start()
    request_queue.put(
        {
            "kind": "SYNC_SOURCES",
            "source_map": {
                "camera-1": {"uri": str(source_uri)}
            },
        }
    )

    request_queue.put(
        {"kind": "LOAD_MODEL", "model_id": "demo_synth_det"}
    )
    detection_ready = _next_result(result_queue, "MODEL_READY")
    request_queue.put(
        {"kind": "LOAD_MODEL", "model_id": "demo-pose"}
    )
    pose_ready = _next_result(result_queue, "MODEL_READY")
    request_queue.put(
        {"kind": "LOAD_MODEL", "model_id": "demo_synth_det"}
    )
    shared_ready = _next_result(result_queue, "MODEL_READY")

    assert detection_ready["load_count"] == 1
    assert pose_ready["load_count"] == 1
    assert shared_ready["load_count"] == 1
    assert shared_ready["loaded_model_ids"] == [
        "demo-pose",
        "demo_synth_det",
    ]

    base = {
        "kind": "INFER",
        "session_id": "session-1",
        "source_id": "camera-1",
        "camera_id": "camera-1",
        "camera_role": "C2",
        "stream_epoch": 0,
        "frame_index": 0,
        "timestamp_local": 0.0,
        "timestamp_global": 1.0,
        "created_at": time.time(),
        "expires_at": time.time() + 5.0,
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    request_queue.put(
        {
            **base,
            "request_id": "request-detection",
            "model_id": "demo_synth_det",
            "task": "detection",
            "pipeline_role": "parcel_detection",
        }
    )
    request_queue.put(
        {
            **base,
            "request_id": "request-pose",
            "model_id": "demo-pose",
            "task": "pose",
            "pipeline_role": "operator_pose",
        }
    )
    results = {
        result["request_id"]: result
        for result in (
            _next_result(result_queue, "INFER_RESULT"),
            _next_result(result_queue, "INFER_RESULT"),
        )
    }
    assert results["request-detection"]["task"] == "detection"
    assert results["request-pose"]["task"] == "pose"
    assert (
        len(results["request-pose"]["observations"][0]["keypoints"])
        == 17
    )

    request_queue.put(
        {
            "kind": "LOAD_MODEL",
            "model_id": "demo-pose",
            "task": "pose",
            "reload": True,
        }
    )
    reloaded = _next_result(result_queue, "MODEL_READY")
    assert reloaded["load_count"] == 2
    assert reloaded["loaded_model_ids"] == [
        "demo-pose",
        "demo_synth_det",
    ]
    stop_event.set()
    worker.join(timeout=2.0)


def test_source_pipeline_resolves_active_model_per_task(tmp_path):
    db = VisionSortDB(tmp_path / "pipeline.db")
    db.initialize()
    _insert_demo_pose(db)
    repo = ControlRepository(db)
    source_id = repo.upsert_source(
        {
            "name": "C2 multi",
            "role": "C2",
            "source_type": "REPLAY",
            "uri": "camera.avi",
            "model_id": "demo_synth_det",
            "tracker_id": "greedy_iou",
            "model_assignments": [
                {
                    "pipeline_role": "parcel_detection",
                    "task": "detection",
                    "model_id": "demo_synth_det",
                    "use_active": True,
                },
                {
                    "pipeline_role": "operator_pose",
                    "task": "pose",
                    "model_id": "demo-pose",
                    "use_active": True,
                },
            ],
        }
    )
    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.db = db
    supervisor.control_repo = repo
    supervisor.config = AppConfig(
        values={"runtime": {"model_selection": "active_registry"}}
    )

    pipeline = supervisor.resolve_model_pipeline(
        source_id, configured_model_id="demo_synth_det"
    )

    assert pipeline == [
        {
            "pipeline_role": "operator_pose",
            "task": "pose",
            "model_id": "demo-pose",
        },
        {
            "pipeline_role": "parcel_detection",
            "task": "detection",
            "model_id": "demo_synth_det",
        },
    ]


def test_pose_activation_does_not_deactivate_parcel_model(tmp_path):
    db = VisionSortDB(tmp_path / "activation-per-task.db")
    db.initialize()
    _insert_demo_pose(db, active=False, status="ARCHIVED")

    activate_model(db, "demo-pose")

    active = {
        str(row["task"]): str(row["id"])
        for row in db.fetch_all(
            "SELECT id, task FROM model_registry WHERE is_active = 1"
        )
    }
    assert active == {
        "detection": "demo_synth_det",
        "pose": "demo-pose",
    }
