from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from visionsort.core.config import AppConfig, DEFAULT_CONFIG
from visionsort.core.enums import CommandStatus, CommandType
from visionsort.core.paths import REPORTS_DIR, ROOT_DIR
from visionsort.database.db import VisionSortDB
from visionsort.runtime.demo_assets import ensure_demo_assets
from visionsort.runtime.supervisor import RuntimeSupervisor


def _json_dict(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )


def _pump(supervisor: RuntimeSupervisor) -> None:
    supervisor.drain_inference_results()
    supervisor.drain_runtime_messages()
    supervisor.refresh_jobs()
    for command in supervisor.control_repo.list_pending_commands():
        supervisor.handle_command(command)


def _wait_until(
    supervisor: RuntimeSupervisor,
    predicate: Callable[[], bool],
    *,
    description: str,
    timeout: float = 90.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _pump(supervisor)
        if predicate():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Délai dépassé: {description}")


def _execute_command(
    supervisor: RuntimeSupervisor,
    command_type: CommandType,
    payload: dict[str, Any],
    *,
    timeout: float = 90.0,
) -> str:
    command_id = supervisor.control_repo.enqueue_command(
        command_type, payload, owner="supervisor-e2e"
    )

    def terminal() -> bool:
        row = supervisor.db.fetch_one(
            "SELECT status FROM commands WHERE id = ?", (command_id,)
        )
        return bool(
            row
            and row["status"]
            in {
                CommandStatus.COMPLETED.value,
                CommandStatus.FAILED.value,
                CommandStatus.CANCELLED.value,
            }
        )

    _wait_until(
        supervisor,
        terminal,
        description=f"commande {command_type.value}",
        timeout=timeout,
    )
    row = supervisor.db.fetch_one(
        "SELECT status, error_text FROM commands WHERE id = ?", (command_id,)
    )
    if row is None or row["status"] != CommandStatus.COMPLETED.value:
        error = row["error_text"] if row else "commande disparue"
        raise RuntimeError(f"{command_type.value} a échoué: {error}")
    return command_id


def _wait_pipeline_step(
    supervisor: RuntimeSupervisor,
    *,
    session_id: str,
    step: str,
    timeout: float = 90.0,
) -> dict[str, Any]:
    step_name = step.upper()

    def terminal() -> bool:
        row = supervisor.db.fetch_one(
            """
            SELECT status FROM pipeline_step_runs
            WHERE session_id = ? AND step = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (session_id, step_name),
        )
        return bool(row and row["status"] in {"COMPLETED", "FAILED", "CANCELLED"})

    _wait_until(
        supervisor,
        terminal,
        description=f"pipeline {step_name} ({session_id})",
        timeout=timeout,
    )
    row = supervisor.db.fetch_one(
        """
        SELECT * FROM pipeline_step_runs
        WHERE session_id = ? AND step = ?
        ORDER BY created_at DESC LIMIT 1
        """,
        (session_id, step_name),
    )
    result = dict(row) if row else {}
    if result.get("status") != "COMPLETED":
        raise RuntimeError(
            f"{step_name} a échoué: {result.get('error_text') or 'erreur inconnue'}"
        )
    return result


def _prepare_replay_variants(
    output_dir: Path, *, max_frames: int
) -> dict[str, dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_assets = ensure_demo_assets()
    variants: dict[str, dict[str, str]] = {
        "train": {},
        "val": {},
        "test": {},
    }
    for role, source_path_text in demo_assets.items():
        source_path = Path(source_path_text)
        for split in ("train", "val", "test"):
            destination = output_dir / f"{split}_{role}.mp4"
            capture = cv2.VideoCapture(str(source_path))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 8.0)
            writer = cv2.VideoWriter(
                str(destination),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not capture.isOpened() or not writer.isOpened():
                capture.release()
                writer.release()
                raise RuntimeError(
                    f"Impossible de créer le Replay E2E {destination}"
                )
            frame_index = 0
            while frame_index < max_frames:
                ok, image = capture.read()
                if not ok:
                    break
                if split == "val":
                    image = cv2.bitwise_not(image)
                elif split == "test":
                    block = 45
                    yy, xx = np.indices((height, width))
                    checker = (((xx // block) + (yy // block)) % 2) * 255
                    pattern = np.repeat(
                        checker.astype(np.uint8)[:, :, None], 3, axis=2
                    )
                    image = cv2.addWeighted(image, 0.20, pattern, 0.80, 0)
                writer.write(image)
                frame_index += 1
            capture.release()
            writer.release()
            shutil.copy2(
                source_path.with_suffix(".jsonl"),
                destination.with_suffix(".jsonl"),
            )
            variants[split][role] = str(destination)
    return variants


def _observation_summary(db: VisionSortDB, session_id: str) -> dict[str, Any]:
    rows = db.fetch_all(
        """
        SELECT ss.details_path
        FROM capture_session_sources css
        JOIN source_state ss ON ss.source_id = css.source_id
        WHERE css.session_id = ?
        """,
        (session_id,),
    )
    frames = 0
    observations = 0
    model_ids: set[str] = set()
    for row in rows:
        path = Path(str(row["details_path"] or ""))
        if not path.is_absolute():
            path = ROOT_DIR / path
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            frames += 1
            observations += len(payload.get("observations") or [])
            if payload.get("model_id"):
                model_ids.add(str(payload["model_id"]))
    return {
        "frames": frames,
        "observations": observations,
        "model_ids": sorted(model_ids),
    }


def _capture_session(
    supervisor: RuntimeSupervisor,
    *,
    name: str,
    source_ids: dict[str, str],
    replay_fps: float,
) -> str:
    before = {
        row["id"]
        for row in supervisor.db.fetch_all("SELECT id FROM capture_sessions")
    }
    _execute_command(
        supervisor,
        CommandType.CREATE_SESSION,
        {
            "name": name,
            "demo_mode": True,
            "sources": [
                {
                    "source_id": source_ids[role],
                    "camera_role": role,
                    "time_offset_ms": {
                        "C1": 0.0,
                        "C2": 10_000.0,
                        "C3": 20_000.0,
                    }[role],
                    "replay_fps": replay_fps,
                }
                for role in ("C1", "C2", "C3")
            ],
            "config": {
                "backend": "demo_synth_det",
                "simulated_backend": True,
                "replay_loop": False,
                "validated_on_site": False,
            },
        },
    )
    rows = supervisor.db.fetch_all(
        "SELECT id FROM capture_sessions ORDER BY created_at DESC"
    )
    session_id = next(str(row["id"]) for row in rows if row["id"] not in before)
    _execute_command(
        supervisor,
        CommandType.START_SESSION,
        {"session_id": session_id},
    )
    expected_sources = set(source_ids.values())
    _wait_until(
        supervisor,
        lambda: not expected_sources.intersection(supervisor.camera_processes),
        description=f"fin des Replay de {session_id}",
        timeout=120.0,
    )
    _execute_command(
        supervisor,
        CommandType.STOP_SESSION,
        {"session_id": session_id},
    )
    _wait_pipeline_step(
        supervisor, session_id=session_id, step="PROCESS_SESSION"
    )
    tracklets = supervisor.db.fetch_one(
        "SELECT COUNT(*) AS count FROM tracklets WHERE session_id = ?",
        (session_id,),
    )
    observations = _observation_summary(supervisor.db, session_id)
    if int((tracklets["count"] if tracklets else 0) or 0) <= 0:
        raise RuntimeError(f"Aucun tracklet produit pour {session_id}")
    if observations["observations"] <= 0:
        raise RuntimeError(f"Aucune observation produite pour {session_id}")
    return session_id


def run_supervisor_e2e(
    db_path: Path,
    *,
    report_path: Path | None = None,
    max_frames: int = 72,
    replay_fps: float = 60.0,
) -> dict[str, Any]:
    """Run the complete simulated lifecycle through the real supervisor."""
    started_at = time.time()
    run_token = uuid.uuid4().hex[:8]
    report_path = report_path or (
        REPORTS_DIR / f"supervisor-e2e-{run_token}.json"
    )
    asset_dir = report_path.parent / f"supervisor-e2e-assets-{run_token}"
    variants = _prepare_replay_variants(
        asset_dir, max_frames=max_frames
    )
    config_values = copy.deepcopy(DEFAULT_CONFIG)
    config_values["app"]["demo_mode"] = True
    config_values["runtime"]["poll_interval_seconds"] = 0.05
    config_values["runtime"]["max_inference_queue"] = 32
    config_values["tracking"]["handoff_window_seconds"] = 0.10
    config_values["tracking"]["handoff_expiry_seconds"] = 5.0
    supervisor = RuntimeSupervisor(
        db_path=db_path, config=AppConfig(values=config_values)
    )
    sessions: dict[str, str] = {}
    source_ids_by_split: dict[str, dict[str, str]] = {}
    verify_source_id = f"sup-{run_token}-verify-c1"
    try:
        for split in ("train", "val", "test"):
            source_ids_by_split[split] = {}
            for role in ("C1", "C2", "C3"):
                source_id = f"sup-{run_token}-{split}-{role.lower()}"
                source_ids_by_split[split][role] = source_id
                _execute_command(
                    supervisor,
                    CommandType.REGISTER_SOURCE,
                    {
                        "id": source_id,
                        "name": f"Supervisor E2E {split} {role}",
                        "role": role,
                        "source_type": "REPLAY",
                        "uri": variants[split][role],
                        "model_id": "demo_synth_det",
                        "tracker_id": "greedy_iou",
                        "enabled": True,
                    },
                )
        _execute_command(
            supervisor,
            CommandType.REGISTER_SOURCE,
            {
                "id": verify_source_id,
                "name": "Supervisor E2E activated model verification",
                "role": "C1",
                "source_type": "REPLAY",
                "uri": variants["test"]["C1"],
                "model_id": "demo_synth_det",
                "tracker_id": "greedy_iou",
                "enabled": True,
            },
        )
        supervisor.start()

        for split in ("train", "val", "test"):
            sessions[split] = _capture_session(
                supervisor,
                name=f"Supervisor E2E {run_token} {split}",
                source_ids=source_ids_by_split[split],
                replay_fps=replay_fps,
            )

        anchor_session = sessions["train"]
        _execute_command(
            supervisor,
            CommandType.RUN_PIPELINE_STEP,
            {
                "session_id": anchor_session,
                "step": "SAMPLE",
                "params": {
                    "name": f"supervisor-e2e-{run_token}",
                    "task": "detection",
                    "session_ids": list(sessions.values()),
                    "split_assignments": {
                        session_id: split
                        for split, session_id in sessions.items()
                    },
                    "force": True,
                },
            },
        )
        _wait_pipeline_step(
            supervisor, session_id=anchor_session, step="SAMPLE"
        )
        session_row = supervisor.control_repo.get_capture_session(
            anchor_session
        ) or {}
        dataset_id = str(session_row.get("last_dataset_id") or "")
        if not dataset_id:
            raise RuntimeError("SAMPLE n'a pas lié de dataset à la session.")

        _execute_command(
            supervisor,
            CommandType.RUN_PIPELINE_STEP,
            {
                "session_id": anchor_session,
                "step": "AUTO_ANNOTATE",
                "params": {
                    "dataset_id": dataset_id,
                    "model_id": "demo_synth_det",
                    "force": False,
                },
            },
        )
        _wait_pipeline_step(
            supervisor, session_id=anchor_session, step="AUTO_ANNOTATE"
        )
        review_rows = supervisor.db.fetch_all(
            """
            SELECT id FROM dataset_items
            WHERE dataset_id = ? AND annotation_status = 'NEEDS_REVIEW'
            """,
            (dataset_id,),
        )
        for row in review_rows:
            supervisor.artifact_repo.update_dataset_item(
                str(row["id"]), annotation_status="HUMAN_VALIDATED"
            )

        _execute_command(
            supervisor,
            CommandType.RUN_PIPELINE_STEP,
            {
                "session_id": anchor_session,
                "step": "FINALIZE_DATASET",
                "params": {"dataset_id": dataset_id},
            },
        )
        _wait_pipeline_step(
            supervisor, session_id=anchor_session, step="FINALIZE_DATASET"
        )
        dataset = supervisor.db.fetch_one(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
        )
        if dataset is None or dataset["status"] != "DATASET_READY":
            raise RuntimeError("Le dataset multi-session n'est pas DATASET_READY.")
        dataset_summary = _json_dict(dataset["summary_json"])
        split_integrity = dataset_summary.get("split_integrity") or {}
        if not split_integrity.get("all_splits_nonempty"):
            raise RuntimeError(
                f"Les splits réels sont incomplets: {split_integrity}"
            )

        _execute_command(
            supervisor,
            CommandType.START_TRAINING,
            {
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
            },
            timeout=120.0,
        )
        training_job = supervisor.db.fetch_one(
            "SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT 1"
        )
        if training_job is None:
            raise RuntimeError("Aucun job d'entraînement créé.")
        training_job_id = str(training_job["id"])

        def training_terminal() -> bool:
            row = supervisor.db.fetch_one(
                "SELECT status FROM training_jobs WHERE id = ?",
                (training_job_id,),
            )
            return bool(row and row["status"] in {"COMPLETED", "FAILED", "CANCELLED"})

        _wait_until(
            supervisor,
            training_terminal,
            description="entraînement supervisé",
            timeout=180.0,
        )
        training_job = supervisor.db.fetch_one(
            "SELECT * FROM training_jobs WHERE id = ?", (training_job_id,)
        )
        if training_job is None or training_job["status"] != "COMPLETED":
            raise RuntimeError(
                "Entraînement échoué: "
                + str(training_job["error_text"] if training_job else "introuvable")
            )
        _wait_until(
            supervisor,
            lambda: training_job_id not in supervisor.training_processes,
            description="arrêt du processus d'entraînement",
            timeout=30.0,
        )
        training_metrics = _json_dict(training_job["metrics_json"])
        candidate_id = str(training_metrics["candidate_model_id"])

        _execute_command(
            supervisor,
            CommandType.PROMOTE_MODEL,
            {"model_id": candidate_id},
        )
        _execute_command(
            supervisor,
            CommandType.ACTIVATE_MODEL,
            {"model_id": candidate_id},
        )
        active = supervisor.db.fetch_one(
            "SELECT id FROM model_registry WHERE is_active = 1"
        )
        if active is None or active["id"] != candidate_id:
            raise RuntimeError("Le candidat promu n'est pas le modèle actif.")

        _execute_command(
            supervisor,
            CommandType.CREATE_SESSION,
            {
                "name": f"Supervisor E2E {run_token} activated model",
                "demo_mode": True,
                "sources": [
                    {
                        "source_id": verify_source_id,
                        "camera_role": "C1",
                        "time_offset_ms": 0.0,
                        "replay_fps": replay_fps,
                    }
                ],
                "config": {
                    "replay_loop": False,
                    "validated_on_site": False,
                },
            },
        )
        verify_session_row = supervisor.db.fetch_one(
            """
            SELECT id FROM capture_sessions
            WHERE name = ? ORDER BY created_at DESC LIMIT 1
            """,
            (f"Supervisor E2E {run_token} activated model",),
        )
        verify_session_id = str(verify_session_row["id"])
        _execute_command(
            supervisor,
            CommandType.START_SESSION,
            {"session_id": verify_session_id},
        )
        _wait_until(
            supervisor,
            lambda: verify_source_id not in supervisor.camera_processes,
            description="Replay avec le modèle activé",
            timeout=120.0,
        )
        _execute_command(
            supervisor,
            CommandType.STOP_SESSION,
            {"session_id": verify_session_id},
        )
        _wait_pipeline_step(
            supervisor,
            session_id=verify_session_id,
            step="PROCESS_SESSION",
        )
        activation_observations = _observation_summary(
            supervisor.db, verify_session_id
        )
        if activation_observations["model_ids"] != [candidate_id]:
            raise RuntimeError(
                "La nouvelle session n'a pas utilisé le candidat actif: "
                f"{activation_observations['model_ids']}"
            )

        command_counts = {
            str(row["status"]): int(row["count"])
            for row in supervisor.db.fetch_all(
                "SELECT status, COUNT(*) AS count FROM commands GROUP BY status"
            )
        }
        capture_summaries = {
            split: _observation_summary(supervisor.db, session_id)
            for split, session_id in sessions.items()
        }
        total_tracklets = int(
            (
                supervisor.db.fetch_one(
                    """
                    SELECT COUNT(*) AS count FROM tracklets
                    WHERE session_id IN (?, ?, ?)
                    """,
                    tuple(sessions.values()),
                )
                or {"count": 0}
            )["count"]
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
            "mode": "SUPERVISOR_PROCESS_E2E",
            "simulated_backend": True,
            "validated_on_site": False,
            "site_validation_status": "NON_VALIDÉ_SUR_SITE",
            "started_at": started_at,
            "ended_at": time.time(),
            "duration_seconds": time.time() - started_at,
            "db_path": str(db_path),
            "session_ids": sessions,
            "capture_summaries": capture_summaries,
            "tracklets": total_tracklets,
            "dataset_id": dataset_id,
            "dataset_status": dataset["status"],
            "dataset_fingerprint": dataset["dataset_fingerprint"],
            "split_integrity": split_integrity,
            "training_job_id": training_job_id,
            "training_status": training_job["status"],
            "candidate_model_id": candidate_id,
            "candidate_metrics": training_metrics,
            "active_model_id": str(active["id"]),
            "activated_session_id": verify_session_id,
            "activated_session_observations": activation_observations,
            "runtime_reload_verified": supervisor.active_model_id == candidate_id,
            "command_counts": command_counts,
            "shutdown_clean": shutdown_clean,
            "limits": [
                "Backends, poids et vidéos simulés.",
                "Aucune validation avec les caméras RTSP réelles.",
                "Calibration, latence GPU et règles métier à valider sur site.",
            ],
        }
        _write_report(report_path, report)
        return report
    finally:
        if not supervisor._shutdown_complete:
            supervisor.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="VisionSort E2E via le RuntimeSupervisor réel"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/runtime/supervisor-e2e.db"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/runtime/reports/supervisor-e2e.json"),
    )
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--replay-fps", type=float, default=60.0)
    args = parser.parse_args()
    report = run_supervisor_e2e(
        args.db,
        report_path=args.report,
        max_frames=args.max_frames,
        replay_fps=args.replay_fps,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
