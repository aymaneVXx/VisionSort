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


def _insert_test_model(
    supervisor: RuntimeSupervisor,
    model_id: str,
    *,
    task: str,
    active: bool,
    status: str,
    backend: str = "demo",
    weights_path: str = "",
) -> None:
    now = utc_now()
    supervisor.db.execute(
        """
        INSERT INTO model_registry
        (id, name, task, backend, weights_path, status, is_active,
         notes_json, metrics_json, parent_model_id, created_from_job_id,
         created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', NULL, NULL, ?, ?)
        """,
        (
            model_id,
            model_id,
            task,
            backend,
            weights_path,
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
    parcel_v2_model_id = f"demo-parcel-v2-{run_token}"
    broken_parcel_model_id = f"broken-parcel-v3-{run_token}"
    parcel_source_id = f"multi-{run_token}-parcel"
    pose_source_id = f"multi-{run_token}-pose"
    session_id = ""
    try:
        supervisor.db.execute(
            """
            UPDATE model_registry
            SET status = 'CHAMPION', updated_at = ?
            WHERE id = 'demo_synth_det'
            """,
            (utc_now(),),
        )
        _insert_test_model(
            supervisor,
            pose_model_id,
            task="pose",
            active=True,
            status="CHAMPION",
        )
        _insert_test_model(
            supervisor,
            next_pose_model_id,
            task="pose",
            active=False,
            status="CHAMPION",
        )
        _insert_test_model(
            supervisor,
            parcel_v2_model_id,
            task="detection",
            active=False,
            status="CHAMPION",
        )
        _insert_test_model(
            supervisor,
            broken_parcel_model_id,
            task="detection",
            active=False,
            status="ARCHIVED",
            backend="ultralytics",
            weights_path=str(asset_dir / "missing-parcel-v3.pt"),
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
                "replay_loop": True,
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

        def current_frames(source_id: str) -> list[dict[str, Any]]:
            return _observation_frames(
                supervisor,
                session_id=session_id,
                source_id=source_id,
            )

        def matching_frames(
            source_id: str,
            *,
            task: str,
            model_id: str,
            minimum_generation: int = 0,
        ) -> list[dict[str, Any]]:
            return [
                frame
                for frame in current_frames(source_id)
                if any(
                    str(result.get("task")) == task
                    and str(result.get("model_id")) == model_id
                    and int(
                        result.get("routing_generation") or 0
                    )
                    >= int(minimum_generation)
                    for result in frame.get("pipeline_results") or []
                )
            ]

        _wait_until(
            supervisor,
            lambda: (
                expected_sources.issubset(supervisor.camera_processes)
                and len(
                    matching_frames(
                        pose_source_id,
                        task="pose",
                        model_id=pose_model_id,
                    )
                )
                >= 3
                and len(
                    matching_frames(
                        parcel_source_id,
                        task="detection",
                        model_id="demo_synth_det",
                    )
                )
                >= 3
            ),
            description="frames initiales parcel v1 + pose v1",
            timeout=120.0,
        )
        loaded_initial = sorted(supervisor.loaded_model_ids)
        references_initial = supervisor.models_in_use()
        inflight_initial = supervisor.request_inference_runtime_status()
        load_counts_initial = dict(supervisor.model_load_counts)

        if not expected_sources.issubset(supervisor.camera_processes):
            raise RuntimeError(
                "Les sources doivent rester actives pendant la bascule."
            )
        _execute_command(
            supervisor,
            CommandType.ACTIVATE_MODEL,
            {
                "model_id": next_pose_model_id,
                "reason": "E2E hot switch pose v1 vers v2",
            },
        )
        pose_v2_generation = int(
            supervisor.runtime_route("pose").get("generation") or 0
        )
        _wait_until(
            supervisor,
            lambda: len(
                matching_frames(
                    pose_source_id,
                    task="pose",
                    model_id=next_pose_model_id,
                    minimum_generation=pose_v2_generation,
                )
            )
            >= 2,
            description="frames pose v2 après bascule à chaud",
            timeout=90.0,
        )
        loaded_after_pose = sorted(supervisor.loaded_model_ids)
        references_after_pose = supervisor.models_in_use()
        inflight_after_pose = supervisor.request_inference_runtime_status()
        load_counts_after_pose = dict(supervisor.model_load_counts)
        parcel_not_reloaded = (
            load_counts_initial.get("demo_synth_det") == 1
            and load_counts_after_pose.get("demo_synth_det") == 1
        )
        if pose_model_id in loaded_after_pose:
            raise RuntimeError(
                "Pose v1 est encore chargé après drainage de ses requêtes."
            )
        if not parcel_not_reloaded:
            raise RuntimeError(
                "Le modèle parcel a été rechargé pendant la bascule pose."
            )

        _execute_command(
            supervisor,
            CommandType.ACTIVATE_MODEL,
            {
                "model_id": parcel_v2_model_id,
                "reason": "E2E hot switch parcel v1 vers v2",
            },
        )
        parcel_v2_generation = int(
            supervisor.runtime_route("detection").get(
                "generation"
            )
            or 0
        )
        _wait_until(
            supervisor,
            lambda: (
                len(
                    matching_frames(
                        pose_source_id,
                        task="detection",
                        model_id=parcel_v2_model_id,
                        minimum_generation=parcel_v2_generation,
                    )
                )
                >= 2
                and len(
                    matching_frames(
                        pose_source_id,
                        task="pose",
                        model_id=next_pose_model_id,
                        minimum_generation=pose_v2_generation,
                    )
                )
                >= 2
            ),
            description="parcel v2 sans interruption de pose v2",
            timeout=90.0,
        )
        loaded_after_parcel = sorted(supervisor.loaded_model_ids)
        references_after_parcel = supervisor.models_in_use()
        inflight_after_parcel = (
            supervisor.request_inference_runtime_status()
        )
        pose_not_reloaded = (
            load_counts_after_pose.get(next_pose_model_id) == 1
            and supervisor.model_load_counts.get(next_pose_model_id) == 1
        )
        if not pose_not_reloaded:
            raise RuntimeError(
                "Pose v2 a été rechargé pendant la bascule parcel."
            )

        broken_activation_error = ""
        try:
            _execute_command(
                supervisor,
                CommandType.ACTIVATE_MODEL,
                {
                    "model_id": broken_parcel_model_id,
                    "reason": "E2E échec parcel v3 invalide",
                },
            )
        except RuntimeError as exc:
            broken_activation_error = str(exc)
        if not broken_activation_error:
            raise RuntimeError(
                "L'activation du modèle parcel v3 invalide devait échouer."
            )
        if (
            supervisor.runtime_route("detection").get("model_id")
            != parcel_v2_model_id
        ):
            raise RuntimeError(
                "Le routage parcel v2 n'a pas été conservé après l'échec."
            )
        registry_after_failure = supervisor.db.fetch_one(
            """
            SELECT id FROM model_registry
            WHERE task = 'detection' AND is_active = 1
            """
        )
        if (
            registry_after_failure is None
            or registry_after_failure["id"] != parcel_v2_model_id
        ):
            raise RuntimeError(
                "Le registre n'a pas conservé parcel v2 après l'échec."
            )
        failed_history = supervisor.db.fetch_one(
            """
            SELECT * FROM model_activation_history
            WHERE activated_model_id = ?
            ORDER BY activated_at DESC LIMIT 1
            """,
            (broken_parcel_model_id,),
        )
        if (
            failed_history is None
            or failed_history["status"] != "FAILED"
            or int(failed_history["runtime_applied"]) != 0
        ):
            raise RuntimeError(
                "L'échec parcel v3 n'est pas tracé comme FAILED."
            )

        _execute_command(
            supervisor,
            CommandType.ROLLBACK_MODEL,
            {
                "task": "detection",
                "reason": "rollback explicite E2E vers parcel v1",
            },
        )
        rollback_generation = int(
            supervisor.runtime_route("detection").get(
                "generation"
            )
            or 0
        )
        rollback_model_id = str(
            supervisor.runtime_route("detection").get("model_id")
            or ""
        )
        if rollback_model_id != "demo_synth_det":
            raise RuntimeError(
                "Le rollback n'a pas sélectionné parcel v1 réellement "
                f"déployé: {rollback_model_id}"
            )
        _wait_until(
            supervisor,
            lambda: len(
                matching_frames(
                    pose_source_id,
                    task="detection",
                    model_id="demo_synth_det",
                    minimum_generation=rollback_generation,
                )
            )
            >= 2,
            description="frames parcel v1 après rollback explicite",
            timeout=90.0,
        )
        final_runtime_state = supervisor.publish_runtime_model_state(
            force=True
        )
        final_consistency = final_runtime_state.get(
            "consistency", {}
        )
        if not final_consistency.get("consistent"):
            raise RuntimeError(
                "État final runtime/registre incohérent: "
                f"{final_consistency.get('errors')}"
            )
        loaded_after_rollback = sorted(supervisor.loaded_model_ids)
        references_after_rollback = supervisor.models_in_use()
        inflight_after_rollback = (
            supervisor.request_inference_runtime_status()
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

        parcel_frames = current_frames(parcel_source_id)
        pose_frames = current_frames(pose_source_id)
        all_frames = [*parcel_frames, *pose_frames]
        request_ids = [
            str(request_id)
            for frame in all_frames
            for request_id in frame.get("request_ids") or []
        ]
        unique_request_ids = len(request_ids) == len(set(request_ids))
        session_frame_context_valid = all(
            frame.get("session_id") == session_id
            and frame.get("source_id")
            in {parcel_source_id, pose_source_id}
            and int(frame.get("frame_index", -1)) >= 0
            and int(frame.get("stream_epoch", -1)) >= 0
            for frame in all_frames
        )
        frame_coordinates = [
            (
                str(frame.get("source_id")),
                int(frame.get("stream_epoch", -1)),
                int(frame.get("frame_index", -1)),
            )
            for frame in all_frames
        ]
        unique_frame_coordinates = len(frame_coordinates) == len(
            set(frame_coordinates)
        )
        if not (
            unique_request_ids
            and session_frame_context_valid
            and unique_frame_coordinates
        ):
            raise RuntimeError(
                "La provenance request/session/epoch/frame est invalide."
            )
        pose_observations = [
            observation
            for frame in pose_frames
            for observation in frame.get("observations") or []
            if observation.get("class_name") == "person"
            and len(observation.get("keypoints") or []) == 17
        ]
        if not pose_observations:
            raise RuntimeError(
                "Le pipeline pose n'a produit aucun jeu de 17 keypoints."
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
        if not keypoint_event_verified:
            raise RuntimeError(
                "Les keypoints n'ont déclenché aucun événement métier: "
                f"{event_types}"
            )
        parcel_pipeline_verified = bool(parcel_frames) and all(
            frame.get("tasks") == ["detection"]
            for frame in parcel_frames
        )
        combined_frames = [
            frame
            for frame in pose_frames
            if set(frame.get("tasks") or []) == {"detection", "pose"}
        ]
        if not parcel_pipeline_verified or not combined_frames:
            raise RuntimeError(
                "Les pipelines parcel seul et parcel + pose sont invalides."
            )
        model_requests: dict[str, int] = {}
        for frame in all_frames:
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
        history = [
            dict(row)
            for row in supervisor.db.fetch_all(
                """
                SELECT * FROM model_activation_history
                WHERE task IN ('detection', 'pose')
                ORDER BY activated_at
                """
            )
        ]
        activation_timeline = [
            {
                **row,
                "source_ids": json.loads(
                    row.get("source_ids_json") or "[]"
                ),
                "metadata": json.loads(
                    row.get("metadata_json") or "{}"
                ),
            }
            for row in history
        ]
        unload_confirmations = [
            {
                "activation_id": row["id"],
                "old_model_id": row.get("previous_model_id"),
                "unload": (
                    json.loads(row.get("metadata_json") or "{}").get(
                        "unload_previous"
                    )
                    or {}
                ),
            }
            for row in history
            if (
                json.loads(row.get("metadata_json") or "{}").get(
                    "unload_previous"
                )
            )
        ]
        models_by_frame = [
            {
                "source_id": frame.get("source_id"),
                "stream_epoch": frame.get("stream_epoch"),
                "frame_index": frame.get("frame_index"),
                "request_ids": frame.get("request_ids"),
                "model_ids": frame.get("model_ids"),
                "routing_generations": frame.get(
                    "routing_generations"
                ),
            }
            for frame in all_frames
        ]
        active_models = {
            str(row["task"]): str(row["id"])
            for row in supervisor.db.fetch_all(
                """
                SELECT id, task FROM model_registry
                WHERE is_active = 1 ORDER BY task
                """
            )
        }

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
            "sources_active_during_switches": True,
            "request_ids_unique": unique_request_ids,
            "session_frame_context_valid": session_frame_context_valid,
            "frame_coordinates_unique": unique_frame_coordinates,
            "models_by_frame": models_by_frame,
            "activation_timeline": activation_timeline,
            "loaded_models": {
                "initial": loaded_initial,
                "after_pose_v2": loaded_after_pose,
                "after_parcel_v2": loaded_after_parcel,
                "after_rollback": loaded_after_rollback,
            },
            "references": {
                "initial": references_initial,
                "after_pose_v2": references_after_pose,
                "after_parcel_v2": references_after_parcel,
                "after_rollback": references_after_rollback,
            },
            "inflight": {
                "initial": inflight_initial.get("inflight_by_model", {}),
                "after_pose_v2": inflight_after_pose.get(
                    "inflight_by_model", {}
                ),
                "after_parcel_v2": inflight_after_parcel.get(
                    "inflight_by_model", {}
                ),
                "after_rollback": inflight_after_rollback.get(
                    "inflight_by_model", {}
                ),
            },
            "unload_confirmations": unload_confirmations,
            "load_counts": {
                "initial": load_counts_initial,
                "after_pose_v2": load_counts_after_pose,
                "final": dict(supervisor.model_load_counts),
            },
            "pose_v2_generation": pose_v2_generation,
            "parcel_v2_generation": parcel_v2_generation,
            "rollback_generation": rollback_generation,
            "broken_activation_error": broken_activation_error,
            "failed_activation_history_id": str(
                failed_history["id"]
            ),
            "rollback": {
                "selected_model_id": rollback_model_id,
                "was_previously_deployed": any(
                    row.get("activated_model_id")
                    == "demo_synth_det"
                    and row.get("status")
                    in {"SUPERSEDED", "ROLLED_BACK", "ACTIVE"}
                    and int(row.get("runtime_applied") or 0) == 1
                    for row in history
                ),
                "candidate_selected": False,
            },
            "final_consistency": final_consistency,
            "active_models": active_models,
            "parcel_model_not_reloaded": parcel_not_reloaded,
            "pose_model_not_reloaded": pose_not_reloaded,
            "old_pose_unloaded": pose_model_id
            not in loaded_after_pose,
            "old_parcel_v2_unloaded_after_rollback": (
                parcel_v2_model_id not in loaded_after_rollback
            ),
            "loaded_model_ids": loaded_after_rollback,
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
