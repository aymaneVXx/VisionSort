from __future__ import annotations

import argparse
import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionsort.core.config import AppConfig, DEFAULT_CONFIG
from visionsort.core.enums import CommandType
from visionsort.core.paths import REPORTS_DIR, ROOT_DIR
from visionsort.database.db import utc_now
from visionsort.runtime.supervisor import RuntimeSupervisor
from visionsort.runtime.supervisor_e2e import (
    _execute_command,
    _wait_pipeline_step,
    _wait_until,
    _write_report,
)


def _insert_demo_pose(
    supervisor: RuntimeSupervisor,
    model_id: str,
    *,
    active: bool,
    status: str,
) -> None:
    now = utc_now()
    supervisor.db.execute(
        """
        INSERT INTO model_registry
        (id, name, task, backend, weights_path, status, is_active,
         notes_json, metrics_json, parent_model_id, created_from_job_id,
         created_at, updated_at)
        VALUES (?, ?, 'pose', 'demo', '', ?, ?, ?, '{}', NULL, NULL, ?, ?)
        """,
        (
            model_id,
            model_id,
            status,
            int(active),
            json.dumps(
                {
                    "simulated_backend": True,
                    "validated_on_site": False,
                }
            ),
            now,
            now,
        ),
    )


def _write_multimodel_asset(
    video_path: Path,
    *,
    max_frames: int,
    include_pose: bool,
) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 200, 200
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Impossible de créer {video_path}.")
    annotations: list[dict[str, Any]] = []
    for frame_index in range(max_frames):
        image = np.full(
            (height, width, 3),
            20 + frame_index % 20,
            dtype=np.uint8,
        )
        parcel_bbox = [90, 88, 130, 116]
        cv2.rectangle(
            image,
            tuple(parcel_bbox[:2]),
            tuple(parcel_bbox[2:]),
            (0, 220, 220),
            2,
        )
        annotations.append(
            {
                "frame_index": frame_index,
                "class_name": "parcel",
                "confidence": 0.98,
                "bbox": parcel_bbox,
                "attributes": {"parcel_hint": "multi-e2e"},
            }
        )
        if include_pose:
            person_bbox = [60, 40, 170, 190]
            cv2.rectangle(
                image,
                tuple(person_bbox[:2]),
                tuple(person_bbox[2:]),
                (220, 140, 20),
                2,
            )
            keypoints = [[80.0, 60.0, 2.0] for _ in range(17)]
            keypoints[9] = [105.0, 100.0, 2.0]
            keypoints[10] = [115.0, 100.0, 2.0]
            annotations.append(
                {
                    "frame_index": frame_index,
                    "class_name": "person",
                    "confidence": 0.97,
                    "bbox": person_bbox,
                    "keypoints": keypoints,
                    "attributes": {"operator_id": "OP-E2E"},
                }
            )
        writer.write(image)
    writer.release()
    video_path.with_suffix(".jsonl").write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=True) for item in annotations
        )
        + "\n",
        encoding="utf-8",
    )


def _observation_frames(
    supervisor: RuntimeSupervisor,
    *,
    session_id: str,
    source_id: str,
) -> list[dict[str, Any]]:
    row = supervisor.db.fetch_one(
        """
        SELECT ss.details_path
        FROM capture_session_sources css
        JOIN source_state ss ON ss.source_id = css.source_id
        WHERE css.session_id = ? AND css.source_id = ?
        """,
        (session_id, source_id),
    )
    if row is None or not row["details_path"]:
        return []
    path = Path(str(row["details_path"]))
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_multimodel_e2e(
    db_path: Path,
    *,
    report_path: Path | None = None,
    max_frames: int = 24,
    replay_fps: float = 60.0,
) -> dict[str, Any]:
    """Exercise task-aware models through the real supervisor processes."""
    started_at = time.time()
    run_token = uuid.uuid4().hex[:8]
    report_path = report_path or (
        REPORTS_DIR / f"multimodel-e2e-{run_token}.json"
    )
    asset_dir = report_path.parent / f"multimodel-e2e-assets-{run_token}"
    parcel_video = asset_dir / "parcel-only.mp4"
    pose_video = asset_dir / "parcel-pose.mp4"
    _write_multimodel_asset(
        parcel_video,
        max_frames=max_frames,
        include_pose=False,
    )
    _write_multimodel_asset(
        pose_video,
        max_frames=max_frames,
        include_pose=True,
    )

    config_values = copy.deepcopy(DEFAULT_CONFIG)
    config_values["app"]["demo_mode"] = True
    config_values["runtime"]["poll_interval_seconds"] = 0.05
    config_values["runtime"]["max_inference_queue"] = 32
    supervisor = RuntimeSupervisor(
        db_path=db_path,
        config=AppConfig(values=config_values),
    )
    pose_model_id = f"demo-pose-{run_token}"
    next_pose_model_id = f"demo-pose-v2-{run_token}"
    parcel_source_id = f"multi-{run_token}-parcel"
    pose_source_id = f"multi-{run_token}-pose"
    session_id = ""
    try:
        _insert_demo_pose(
            supervisor,
            pose_model_id,
            active=True,
            status="CHAMPION",
        )
        _insert_demo_pose(
            supervisor,
            next_pose_model_id,
            active=False,
            status="CHAMPION",
        )
        parcel_source_id = supervisor.control_repo.upsert_source(
            {
                "id": parcel_source_id,
                "name": "Multi-model parcel only",
                "role": "C1",
                "source_type": "REPLAY",
                "uri": str(parcel_video),
                "model_id": "demo_synth_det",
                "tracker_id": "greedy_iou",
                "enabled": True,
                "model_assignments": [
                    {
                        "pipeline_role": "parcel_detection",
                        "task": "detection",
                        "model_id": "demo_synth_det",
                        "use_active": True,
                    }
                ],
            }
        )
        pose_source_id = supervisor.control_repo.upsert_source(
            {
                "id": pose_source_id,
                "name": "Multi-model parcel and pose",
                "role": "C2",
                "source_type": "REPLAY",
                "uri": str(pose_video),
                "model_id": "demo_synth_det",
                "tracker_id": "greedy_iou",
                "enabled": True,
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
                        "model_id": pose_model_id,
                        "use_active": True,
                    },
                ],
            }
        )
        session_id = supervisor.control_repo.create_capture_session(
            name=f"Multi-model E2E {run_token}",
            demo_mode=True,
            sources=[
                {
                    "source_id": parcel_source_id,
                    "camera_role": "C1",
                    "replay_fps": replay_fps,
                },
                {
                    "source_id": pose_source_id,
                    "camera_role": "C2",
                    "replay_fps": replay_fps,
                },
            ],
            config={
                "replay_loop": False,
                "simulated_backend": True,
                "validated_on_site": False,
            },
        )
        supervisor.start()
        _execute_command(
            supervisor,
            CommandType.START_SESSION,
            {"session_id": session_id},
        )
        expected_sources = {parcel_source_id, pose_source_id}
        _wait_until(
            supervisor,
            lambda: not expected_sources.intersection(
                supervisor.camera_processes
            ),
            description="fin des sources multi-modèles",
            timeout=120.0,
        )
        _execute_command(
            supervisor,
            CommandType.STOP_SESSION,
            {"session_id": session_id},
        )
        _wait_pipeline_step(
            supervisor,
            session_id=session_id,
            step="PROCESS_SESSION",
        )

        parcel_frames = _observation_frames(
            supervisor,
            session_id=session_id,
            source_id=parcel_source_id,
        )
        pose_frames = _observation_frames(
            supervisor,
            session_id=session_id,
            source_id=pose_source_id,
        )
        parcel_pipeline_verified = bool(parcel_frames) and all(
            frame.get("tasks") == ["detection"]
            and frame.get("model_ids") == ["demo_synth_det"]
            for frame in parcel_frames
        )
        combined_frames = [
            frame
            for frame in pose_frames
            if set(frame.get("tasks") or []) == {"detection", "pose"}
            and set(frame.get("model_ids") or [])
            == {"demo_synth_det", pose_model_id}
        ]
        pose_observations = [
            observation
            for frame in combined_frames
            for observation in frame.get("observations") or []
            if observation.get("class_name") == "person"
            and len(observation.get("keypoints") or []) == 17
            and observation.get("model_id") == pose_model_id
        ]
        model_requests: dict[str, int] = {}
        for frame in [*parcel_frames, *pose_frames]:
            for pipeline in frame.get("pipeline_results") or []:
                model_id = str(pipeline["model_id"])
                requests = int(
                    (pipeline.get("model_metrics") or {}).get(
                        "requests", 0
                    )
                )
                model_requests[model_id] = max(
                    model_requests.get(model_id, 0),
                    requests,
                )
        event_types = [
            str(row["event_type"])
            for row in supervisor.db.fetch_all(
                """
                SELECT event_type FROM events
                WHERE session_id = ? ORDER BY created_at
                """,
                (session_id,),
            )
        ]
        keypoint_event_verified = any(
            event_type
            in {
                "pickup_candidate",
                "parcel_picked",
                "parcel_carried",
            }
            for event_type in event_types
        )
        if not parcel_pipeline_verified:
            raise RuntimeError(
                "La source parcelle seule n'a pas conservé son pipeline dédié."
            )
        if not combined_frames or not pose_observations:
            raise RuntimeError(
                "Le pipeline parcelle + pose n'a pas produit les deux tâches."
            )
        if (
            model_requests.get("demo_synth_det", 0) <= 0
            or model_requests.get(pose_model_id, 0) <= 0
        ):
            raise RuntimeError(
                f"Les deux modèles n'ont pas été appelés: {model_requests}"
            )
        if not keypoint_event_verified:
            raise RuntimeError(
                f"Les keypoints n'ont déclenché aucun événement: {event_types}"
            )

        load_counts_before = dict(supervisor.model_load_counts)
        _execute_command(
            supervisor,
            CommandType.ACTIVATE_MODEL,
            {"model_id": next_pose_model_id},
        )
        active_models = {
            str(row["task"]): str(row["id"])
            for row in supervisor.db.fetch_all(
                """
                SELECT id, task FROM model_registry
                WHERE is_active = 1 ORDER BY task
                """
            )
        }
        load_counts_after = dict(supervisor.model_load_counts)
        parcel_not_reloaded = (
            load_counts_before.get("demo_synth_det") == 1
            and load_counts_after.get("demo_synth_det") == 1
        )
        if active_models.get("detection") != "demo_synth_det":
            raise RuntimeError(
                f"Le modèle parcelle a été désactivé: {active_models}"
            )
        if active_models.get("pose") != next_pose_model_id:
            raise RuntimeError(
                f"La nouvelle pose n'est pas active: {active_models}"
            )
        if not parcel_not_reloaded:
            raise RuntimeError(
                "Le modèle parcelle a été rechargé pendant l'activation pose."
            )

        supervisor.shutdown()
        shutdown_clean = (
            not supervisor.inference_process.is_alive()
            and not supervisor.camera_processes
            and not supervisor.training_processes
            and not supervisor.pipeline_processes
        )
        report = {
            "status": "COMPLETED",
            "mode": "SUPERVISOR_MULTI_MODEL_E2E",
            "simulated_backend": True,
            "validated_on_site": False,
            "site_validation_status": "NON_VALIDÉ_SUR_SITE",
            "started_at": started_at,
            "ended_at": time.time(),
            "duration_seconds": time.time() - started_at,
            "db_path": str(db_path),
            "session_id": session_id,
            "parcel_source_frames": len(parcel_frames),
            "combined_source_frames": len(combined_frames),
            "parcel_pipeline_verified": parcel_pipeline_verified,
            "pose_keypoint_observations": len(pose_observations),
            "model_requests": model_requests,
            "event_types": event_types,
            "keypoint_event_verified": keypoint_event_verified,
            "load_counts_before_pose_activation": load_counts_before,
            "load_counts_after_pose_activation": load_counts_after,
            "active_models": active_models,
            "parcel_model_not_reloaded": parcel_not_reloaded,
            "loaded_model_ids": sorted(supervisor.loaded_model_ids),
            "shutdown_clean": shutdown_clean,
            "limits": [
                "Backend, modèles, vidéos et keypoints simulés.",
                "Aucune validation avec les caméras RTSP réelles.",
                "Latence GPU et robustesse des événements à valider sur site.",
            ],
        }
        _write_report(report_path, report)
        return report
    finally:
        if not supervisor._shutdown_complete:
            supervisor.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VisionSort E2E multi-modèle via le superviseur réel"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/runtime/multimodel-e2e.db"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/runtime/reports/multimodel-e2e.json"),
    )
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--replay-fps", type=float, default=60.0)
    args = parser.parse_args()
    report = run_multimodel_e2e(
        args.db,
        report_path=args.report,
        max_frames=args.max_frames,
        replay_fps=args.replay_fps,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
