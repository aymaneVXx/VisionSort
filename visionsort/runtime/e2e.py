from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2

from visionsort.core.config import AppConfig, load_config
from visionsort.core.enums import MatchResult
from visionsort.core.paths import REPORTS_DIR, ROOT_DIR
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import (
    ArtifactRepository,
    ControlRepository,
    TrackingRepository,
)
from visionsort.deployment.registry import activate_model, promote_model
from visionsort.inference.engine import DemoDetectionBackend, SharedInferenceEngine
from visionsort.runtime.demo_assets import ensure_demo_assets
from visionsort.runtime.pipeline_worker import pipeline_worker_loop
from visionsort.tracking.engine import GlobalParcelTracker, build_tracker
from visionsort.training.pipeline import create_training_job, training_worker_loop


def _json_dict(text: str | None) -> dict[str, Any]:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def run_demo_e2e(
    db_path: Path,
    *,
    report_path: Path | None = None,
    max_frames: int = 72,
) -> dict[str, Any]:
    """Execute the complete CPU Replay lifecycle with explicit simulated backends."""
    started_at = time.time()
    db = VisionSortDB(db_path)
    db.initialize()
    control_repo = ControlRepository(db)
    tracking_repo = TrackingRepository(db)
    artifact_repo = ArtifactRepository(db)
    assets = ensure_demo_assets()
    source_ids: dict[str, str] = {}
    offsets_ms = {"C1": 0.0, "C2": 10_000.0, "C3": 20_000.0}
    for role in ("C1", "C2", "C3"):
        source_id = control_repo.upsert_source(
            {
                "id": f"e2e-{role.lower()}",
                "name": f"E2E Replay {role}",
                "role": role,
                "source_type": "REPLAY",
                "uri": assets[role],
                "model_id": "demo_synth_det",
                "tracker_id": "greedy_iou",
                "enabled": True,
            }
        )
        source_ids[role] = source_id
    session_id = control_repo.create_capture_session(
        name="VisionSort E2E CPU",
        demo_mode=True,
        sources=[
            {
                "source_id": source_ids[role],
                "camera_role": role,
                "time_offset_ms": offsets_ms[role],
                "replay_fps": 8.0,
            }
            for role in ("C1", "C2", "C3")
        ],
        config={
            "backend": "demo_synth_det",
            "simulated_backend": True,
            "validated_on_site": False,
        },
    )
    session_start = 1_000.0
    control_repo.update_capture_session(session_id, started_at=session_start)
    config = load_config()
    config.values.setdefault("app", {})["demo_mode"] = True
    backend = DemoDetectionBackend()
    all_tracklets = []
    observation_count = 0
    local_identities: set[tuple[str, int]] = set()
    frames_by_role: dict[str, int] = {}
    for role in ("C1", "C2", "C3"):
        source_id = source_ids[role]
        backend.register_sidecar(source_id, assets[role])
        tracker = build_tracker(
            tracker_id="greedy_iou",
            session_id=session_id,
            source_id=source_id,
            camera_id=source_id,
            camera_role=role,
            zones=config.get("tracking", "zones", default={}).get(role, []),
        )
        capture = cv2.VideoCapture(assets[role])
        frame_index = 0
        while frame_index < max_frames:
            ok, image = capture.read()
            if not ok:
                break
            timestamp_local = frame_index / 8.0
            timestamp_global = (
                session_start
                + offsets_ms[role] / 1000.0
                + timestamp_local
            )
            observations = backend.predict(source_id, frame_index, image)
            for observation in observations:
                observation.model_id = "demo_synth_det"
                observation.model_version = "demo-sidecar-v1"
            track_observations, finalized = tracker.update(
                frame_index=frame_index,
                timestamp_local=timestamp_local,
                timestamp_global=timestamp_global,
                image_size=(image.shape[1], image.shape[0]),
                image=image,
                observations=observations,
            )
            observation_count += len(observations)
            local_identities.update(
                (item.camera_id, item.local_track_id)
                for item in track_observations
            )
            all_tracklets.extend(finalized)
            frame_index += 1
        capture.release()
        frames_by_role[role] = frame_index
        all_tracklets.extend(tracker.flush())

    global_tracker = GlobalParcelTracker(
        topology_edges=config.get(
            "tracking", "site_topology", "edges", default=[]
        ),
        source_roles={
            source_ids[role]: role for role in ("C1", "C2", "C3")
        },
    )
    parcel_tracklets = [
        tracklet
        for tracklet in all_tracklets
        if tracklet.class_name == "parcel"
    ]
    outcome_by_tracklet = {}
    for role in ("C1", "C2", "C3"):
        role_tracklets = sorted(
            (
                tracklet
                for tracklet in parcel_tracklets
                if tracklet.camera_role == role
            ),
            key=lambda item: (item.started_at_global, item.ended_at_global),
        )
        role_outcomes = global_tracker.process_tracklets(role_tracklets)
        outcome_by_tracklet.update(
            {
                tracklet.tracklet_id: outcome
                for tracklet, outcome in zip(
                    role_tracklets, role_outcomes, strict=True
                )
            }
        )
    match_counts = {
        MatchResult.MATCHED.value: 0,
        MatchResult.AMBIGUOUS.value: 0,
        MatchResult.UNMATCHED.value: 0,
    }
    for tracklet in all_tracklets:
        if tracklet.class_name != "parcel":
            tracking_repo.upsert_tracklet(
                tracklet, match_result=MatchResult.UNMATCHED.value
            )
            continue
        parcel_id, result, _, _ = outcome_by_tracklet[tracklet.tracklet_id]
        match_counts[result.value] += 1
        tracking_repo.upsert_tracklet(
            tracklet,
            parcel_id=parcel_id or None,
            match_result=result.value,
        )
        if parcel_id:
            tracking_repo.upsert_global_parcel(
                global_tracker.parcels[parcel_id]
            )
    control_repo.update_capture_session(
        session_id,
        ended_at=session_start + 30.0,
    )

    pipeline_worker_loop(str(db_path), session_id, "PROCESS_SESSION", {})
    pipeline_worker_loop(
        str(db_path),
        session_id,
        "SAMPLE",
        {"name": "visionsort-e2e-dataset"},
    )
    session = control_repo.get_capture_session(session_id) or {}
    dataset_id = str(session["last_dataset_id"])
    pipeline_worker_loop(
        str(db_path),
        session_id,
        "AUTO_ANNOTATE",
        {
            "dataset_id": dataset_id,
            "model_id": "demo_synth_det",
            "force": False,
        },
    )
    review_rows = db.fetch_all(
        "SELECT id FROM dataset_items WHERE dataset_id = ? AND annotation_status = 'NEEDS_REVIEW'",
        (dataset_id,),
    )
    for row in review_rows:
        artifact_repo.update_dataset_item(
            row["id"], annotation_status="HUMAN_VALIDATED"
        )
    pipeline_worker_loop(
        str(db_path),
        session_id,
        "FINALIZE_DATASET",
        {"dataset_id": dataset_id},
    )
    dataset = db.fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None or dataset["status"] != "DATASET_READY":
        raise RuntimeError("Le scénario E2E n'a pas produit de DATASET_READY.")

    training_recipe = {
        "dataset_id": dataset_id,
        "model_id": "demo_synth_det",
        "task": "detection",
        "architecture": "demo-sidecar",
        "imgsz": 320,
        "epochs": 1,
        "batch": 1,
        "device": "cpu",
        "patience": 1,
        "mode": "demo",
    }
    training_job_id = create_training_job(
        db, dataset_id, "demo_synth_det", training_recipe
    )
    training_worker_loop(
        str(db_path), training_job_id, training_recipe, True
    )
    training_job = db.fetch_one(
        "SELECT * FROM training_jobs WHERE id = ?", (training_job_id,)
    )
    if training_job is None or training_job["status"] != "COMPLETED":
        raise RuntimeError("L'entraînement E2E ne s'est pas terminé.")
    training_metrics = _json_dict(training_job["metrics_json"])
    candidate_id = str(training_metrics["candidate_model_id"])
    promote_model(db, candidate_id)
    activate_model(db, candidate_id)

    second_session_id = control_repo.create_capture_session(
        name="VisionSort E2E Reload",
        demo_mode=True,
        sources=[
            {
                "source_id": source_ids["C1"],
                "camera_role": "C1",
                "time_offset_ms": 0.0,
            }
        ],
        config={
            "expected_model_id": candidate_id,
            "simulated_backend": True,
        },
    )
    source_map = {
        source_id: dict(
            db.fetch_one("SELECT * FROM sources WHERE id = ?", (source_id,))
        )
        for source_id in source_ids.values()
    }
    inference = SharedInferenceEngine(
        db,
        AppConfig(values={"app": {"demo_mode": True}, "gpu": {"device": "cpu"}}),
    )
    inference.load_model(candidate_id, source_map)
    capture = cv2.VideoCapture(assets["C1"])
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Impossible de relire la fixture C1.")
    reloaded_observations = inference.predict(source_ids["C1"], 0, image)
    if not reloaded_observations or any(
        item.model_id != candidate_id for item in reloaded_observations
    ):
        raise RuntimeError("Le modèle activé n'a pas été utilisé au second run.")

    dataset_summary = _json_dict(dataset["summary_json"])
    report = {
        "status": "COMPLETED",
        "mode": "CPU_DEMO_SIMULATED_BACKENDS",
        "started_at": started_at,
        "ended_at": time.time(),
        "duration_seconds": time.time() - started_at,
        "session_id": session_id,
        "second_session_id": second_session_id,
        "sources": source_ids,
        "frames_by_role": frames_by_role,
        "observations": observation_count,
        "local_track_identities": len(local_identities),
        "tracklets": len(all_tracklets),
        "global_match_results": match_counts,
        "dataset_id": dataset_id,
        "dataset_status": dataset["status"],
        "dataset_items": int(
            db.fetch_one(
                "SELECT COUNT(*) AS count FROM dataset_items WHERE dataset_id = ?",
                (dataset_id,),
            )["count"]
        ),
        "sample_groups": dataset_summary.get("sample_groups"),
        "split_integrity": dataset_summary.get("split_integrity"),
        "reviewed_items": len(review_rows),
        "training_job_id": training_job_id,
        "training_status": training_job["status"],
        "best_pt": training_metrics.get("weights_path"),
        "candidate_model_id": candidate_id,
        "candidate_metrics": {
            key: training_metrics.get(key)
            for key in (
                "precision",
                "recall",
                "mAP50",
                "mAP50_95",
                "count_accuracy",
                "merge_rate",
                "fps",
            )
        },
        "comparison": training_metrics.get("comparison"),
        "active_model_id": candidate_id,
        "reload_verified": True,
        "reload_backend_info": inference.backend_info,
        "site_validation_status": "NON_VALIDÉ_SUR_SITE",
        "real_camera_dependencies": [
            "RTSP reconnect tuning",
            "camera calibration and normalized zones",
            "pickup/drop thresholds",
            "real-world accuracy and throughput",
        ],
    }
    output_path = report_path or (
        REPORTS_DIR / "e2e" / f"{int(time.time())}_E2E.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    report["report_path"] = (
        str(output_path.relative_to(ROOT_DIR))
        if output_path.is_relative_to(ROOT_DIR)
        else str(output_path)
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the VisionSort CPU demo end-to-end validation."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=ROOT_DIR / "data" / "runtime" / f"e2e-{int(time.time())}.db",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=72)
    args = parser.parse_args()
    report = run_demo_e2e(
        args.db,
        report_path=args.report,
        max_frames=args.max_frames,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
