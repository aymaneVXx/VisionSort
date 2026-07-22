from __future__ import annotations

import json
import multiprocessing as mp
import queue
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ajoute le répertoire racine du projet au sys.path pour l'exécution en standalone
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visionsort.acquisition.worker import camera_worker_loop
from visionsort.core.config import load_config
from visionsort.core.enums import CommandStatus, CommandType, JobType, MatchResult, SourceStatus
from visionsort.core.paths import DB_PATH, ROOT_DIR, ensure_project_dirs
from visionsort.core.types import GlobalParcel, Tracklet
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import (
    ArtifactRepository,
    ControlRepository,
    EventRepository,
    JobRepository,
    TrackingRepository,
)
from visionsort.datasets.pipeline import build_dataset
from visionsort.deployment.registry import activate_model, rollback_to_previous_active
from visionsort.inference.engine import inference_worker_loop
from visionsort.runtime.demo_assets import ensure_demo_assets
from visionsort.sources.frame_sources import can_open_uri
from visionsort.tracking.engine import GlobalParcelTracker
from visionsort.training.pipeline import create_training_job, training_worker_loop


class GPUResourceArbiter:
    def __init__(self, allow_training_while_inference: bool, max_concurrent_live_sources: int):
        self.allow_training_while_inference = allow_training_while_inference
        self.max_concurrent_live_sources = max_concurrent_live_sources

    def can_start_source(self, active_sources: int, training_active: bool) -> tuple[bool, str]:
        if training_active and not self.allow_training_while_inference:
            return False, "Refusé: entraînement en cours, GPU réservé."
        if active_sources >= self.max_concurrent_live_sources:
            return False, "Refusé: limite de flux inférés simultanés atteinte."
        return True, ""

    def can_start_training(self, active_sources: int) -> tuple[bool, str]:
        if active_sources > 0 and not self.allow_training_while_inference:
            return False, "Refusé: arrêtez les sources actives avant un entraînement GPU."
        return True, ""


class RuntimeSupervisor:
    def __init__(self):
        ensure_project_dirs()
        self.config = load_config()
        self.db = VisionSortDB(DB_PATH)
        self.db.initialize()
        self.control_repo = ControlRepository(self.db)
        self.event_repo = EventRepository(self.db)
        self.tracking_repo = TrackingRepository(self.db)
        self.artifact_repo = ArtifactRepository(self.db)
        self.job_repo = JobRepository(self.db)
        self.ctx = mp.get_context("spawn")
        self.manager = self.ctx.Manager()
        self.inference_request_queue = self.ctx.Queue(maxsize=int(self.config.get("runtime", "max_inference_queue", default=8)))
        self.inference_result_queue = self.ctx.Queue()
        self.inference_result_store = self.manager.dict()
        self.runtime_queue = self.ctx.Queue()
        self.inference_stop_event = self.ctx.Event()
        self.control_flags = self.manager.dict()
        self.camera_processes: dict[str, tuple[mp.Process, Any]] = {}
        self.training_processes: dict[str, mp.Process] = {}
        self.active_model_id: str | None = None
        self.arbiter = GPUResourceArbiter(
            allow_training_while_inference=bool(self.config.get("gpu", "allow_training_while_inference", default=False)),
            max_concurrent_live_sources=int(self.config.get("gpu", "max_concurrent_live_sources", default=3)),
        )
        topology_edges = self.config.get("tracking", "site_topology", "edges", default=[])
        self.global_tracker = GlobalParcelTracker(topology_edges=topology_edges, source_roles=self._source_roles())
        self.inference_process = self.ctx.Process(
            target=inference_worker_loop,
            args=(
                self.inference_request_queue,
                self.inference_result_queue,
                self.inference_stop_event,
                str(DB_PATH),
                self.config.values,
            ),
            daemon=True,
            name="visionsort-gpu-inference",
        )

    def _source_roles(self) -> dict[str, str]:
        return {row["role"]: row["role"] for row in self.db.fetch_all("SELECT role FROM sources")}

    def bootstrap_demo_sources(self) -> None:
        if not self.config.demo_mode:
            return
        assets = ensure_demo_assets()
        if self.db.fetch_one("SELECT id FROM sources LIMIT 1"):
            return
        for role, uri in assets.items():
            self.control_repo.upsert_source(
                {
                    "name": f"Replay {role}",
                    "role": role,
                    "source_type": "REPLAY",
                    "uri": uri,
                    "model_id": "demo_synth_det",
                    "tracker_id": "greedy_iou",
                    "enabled": True,
                }
            )
        self.event_repo.add_event(
            "demo_bootstrapped",
            {"sources": assets, "validated_on_site": False, "requires_demo_mode": True},
            severity="info",
        )

    def start(self) -> None:
        self.bootstrap_demo_sources()
        if not self.inference_process.is_alive():
            self.inference_process.start()
            self.job_repo.upsert_job_run(JobType.GPU_INFERENCE.value, "shared", self.inference_process.pid or 0, "RUNNING", {"active_model_id": self.active_model_id})
        self.job_repo.upsert_job_run(JobType.SUPERVISOR.value, "main", mp.current_process().pid or 0, "RUNNING", {"demo_mode": self.config.demo_mode})

    def shutdown(self) -> None:
        for source_id in list(self.camera_processes):
            self.stop_source(source_id)
        for job_id, process in list(self.training_processes.items()):
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            self.job_repo.mark_job_stopped(JobType.TRAINING.value, job_id, status="STOPPED")
        self.inference_stop_event.set()
        if self.inference_process.is_alive():
            self.inference_process.terminate()
            self.inference_process.join(timeout=3)
        self.job_repo.mark_job_stopped(JobType.GPU_INFERENCE.value, "shared", status="STOPPED")
        self.job_repo.mark_job_stopped(JobType.SUPERVISOR.value, "main", status="STOPPED")

    def sync_inference_sources(self) -> None:
        sources = {row["role"]: dict(row) for row in self.db.fetch_all("SELECT * FROM sources")}
        self.inference_request_queue.put({"kind": "SYNC_SOURCES", "source_map": sources})

    def ensure_model_loaded(self, model_id: str) -> None:
        if model_id == self.active_model_id:
            return
        self.sync_inference_sources()
        self.inference_request_queue.put({"kind": "LOAD_MODEL", "model_id": model_id})
        timeout = time.time() + 15
        while time.time() < timeout:
            self.drain_inference_results()
            ready = self.inference_result_store.pop("__model_ready__", None)
            if ready and ready.get("model_id") == model_id:
                self.active_model_id = model_id
                self.job_repo.upsert_job_run(JobType.GPU_INFERENCE.value, "shared", self.inference_process.pid or 0, "RUNNING", ready)
                return
            time.sleep(0.1)
        raise RuntimeError(f"Chargement du modèle expiré: {model_id}")

    def start_source(self, source_id: str) -> None:
        row = self.db.fetch_one("SELECT * FROM sources WHERE id = ?", (source_id,))
        if row is None:
            raise RuntimeError("Source introuvable.")
        active_sources = len(self.camera_processes)
        training_active = any(process.is_alive() for process in self.training_processes.values())
        allowed, reason = self.arbiter.can_start_source(active_sources, training_active)
        if not allowed:
            raise RuntimeError(reason)
        if self.camera_processes and any(self.db.fetch_one("SELECT model_id FROM sources WHERE id = ?", (sid,))["model_id"] != row["model_id"] for sid in self.camera_processes):
            raise RuntimeError("Toutes les sources actives doivent partager le même model_id pour le worker GPU unique.")
        self.ensure_model_loaded(row["model_id"])
        stop_event = self.ctx.Event()
        cfg = dict(row)
        cfg["replay_fps"] = 8.0
        self.control_flags[source_id] = {"recording": False}
        process = self.ctx.Process(
            target=camera_worker_loop,
            args=(
                cfg,
                str(DB_PATH),
                self.config.values,
                self.inference_request_queue,
                self.inference_result_store,
                self.runtime_queue,
                stop_event,
                self.control_flags,
            ),
            daemon=True,
            name=f"visionsort-camera-{row['role']}",
        )
        process.start()
        self.camera_processes[source_id] = (process, stop_event)
        self.job_repo.upsert_job_run(JobType.CAMERA.value, source_id, process.pid or 0, "RUNNING", {"role": row["role"], "model_id": row["model_id"]})
        self.control_repo.update_source_state(source_id, status=SourceStatus.CONNECTING.value, fps=0.0)

    def stop_source(self, source_id: str) -> None:
        data = self.camera_processes.pop(source_id, None)
        if data is None:
            self.control_repo.update_source_state(source_id, status=SourceStatus.OFFLINE.value, fps=0.0)
            return
        process, stop_event = data
        stop_event.set()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        self.job_repo.mark_job_stopped(JobType.CAMERA.value, source_id)
        self.control_repo.update_source_state(source_id, status=SourceStatus.OFFLINE.value, fps=0.0)

    def set_recording(self, source_id: str, enabled: bool) -> None:
        flags = dict(self.control_flags.get(source_id, {}))
        flags["recording"] = enabled
        self.control_flags[source_id] = flags
        self.control_repo.update_source_state(source_id, status=self.db.fetch_one("SELECT status FROM source_state WHERE source_id = ?", (source_id,))["status"], fps=0.0, recording_enabled=enabled)

    def test_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        ok, message = can_open_uri(payload["uri"])
        self.event_repo.add_event("source_tested", {"payload": payload, "ok": ok, "message": message}, camera_id=payload.get("role"))
        return {"ok": ok, "message": message}

    def start_training(self, payload: dict[str, Any]) -> str:
        active_sources = len(self.camera_processes)
        allowed, reason = self.arbiter.can_start_training(active_sources)
        if not allowed:
            raise RuntimeError(reason)
        job_id = create_training_job(self.db, payload["dataset_id"], payload["model_id"], payload)
        process = self.ctx.Process(
            target=training_worker_loop,
            args=(str(DB_PATH), job_id, payload, self.config.demo_mode),
            daemon=True,
            name=f"visionsort-training-{job_id}",
        )
        process.start()
        self.training_processes[job_id] = process
        self.job_repo.upsert_job_run(JobType.TRAINING.value, job_id, process.pid or 0, "RUNNING", {"model_id": payload["model_id"]})
        self.artifact_repo.update_training_job(job_id, "RUNNING")
        return job_id

    def handle_tracklet(self, payload: dict[str, Any]) -> None:
        tracklet = Tracklet(
            tracklet_id=payload["tracklet_id"],
            camera_id=payload["camera_id"],
            local_track_id=int(payload["local_track_id"]),
            started_at=float(payload["started_at"]),
            ended_at=float(payload["ended_at"]),
            class_name=payload["class_name"],
            first_bbox=tuple(payload["first_bbox"]),
            last_bbox=tuple(payload["last_bbox"]),
            avg_speed=float(payload["avg_speed"]),
            last_zone_id=payload.get("last_zone_id"),
            frame_count=int(payload["frame_count"]),
            observation_path=payload["observation_path"],
            summary_json=payload["summary_json"],
        )
        self.global_tracker.source_roles = self._source_roles()
        parcel_id, result, reasons, candidate = self.global_tracker.process_tracklet(tracklet)
        self.tracking_repo.upsert_tracklet(tracklet, parcel_id=parcel_id or None, match_result=result.value)
        if parcel_id:
            parcel = self.global_tracker.parcels[parcel_id]
            self.tracking_repo.upsert_global_parcel(parcel)
        event_type = "handoff_ambiguous" if result == MatchResult.AMBIGUOUS else "handoff_matched" if result == MatchResult.MATCHED else "tracklet_unmatched"
        self.event_repo.add_event(
            event_type,
            {
                "tracklet_id": tracklet.tracklet_id,
                "reasons": reasons,
                "candidate": asdict(candidate) if candidate else None,
                "validated_on_site": False,
            },
            parcel_id=parcel_id or None,
            camera_id=tracklet.camera_id,
            severity="warning" if result == MatchResult.AMBIGUOUS else "info",
        )

    def handle_command(self, command: dict[str, Any]) -> None:
        payload = json.loads(command["payload_json"])
        command_id = command["id"]
        command_type = command["command_type"]
        self.control_repo.mark_command(command_id, CommandStatus.RUNNING)
        try:
            if command_type == CommandType.REGISTER_SOURCE.value:
                source_id = self.control_repo.upsert_source(payload)
                result_payload = {"source_id": source_id}
            elif command_type == CommandType.START_SOURCE.value:
                self.start_source(payload["source_id"])
                result_payload = {"source_id": payload["source_id"]}
            elif command_type == CommandType.STOP_SOURCE.value:
                self.stop_source(payload["source_id"])
                result_payload = {"source_id": payload["source_id"]}
            elif command_type == CommandType.START_RECORDING.value:
                self.set_recording(payload["source_id"], True)
                result_payload = {"source_id": payload["source_id"], "recording": True}
            elif command_type == CommandType.STOP_RECORDING.value:
                self.set_recording(payload["source_id"], False)
                result_payload = {"source_id": payload["source_id"], "recording": False}
            elif command_type == CommandType.TEST_SOURCE.value:
                result_payload = self.test_source(payload)
            elif command_type == CommandType.CREATE_DATASET.value:
                result_payload = build_dataset(self.db, name=payload.get("name", "autodataset"))
            elif command_type == CommandType.START_TRAINING.value:
                result_payload = {"job_id": self.start_training(payload)}
            elif command_type == CommandType.ACTIVATE_MODEL.value:
                activate_model(self.db, payload["model_id"])
                result_payload = {"model_id": payload["model_id"]}
            elif command_type == CommandType.ROLLBACK_MODEL.value:
                result_payload = {"model_id": rollback_to_previous_active(self.db)}
            elif command_type == CommandType.BOOTSTRAP_DEMO.value:
                self.bootstrap_demo_sources()
                result_payload = {"demo_mode": self.config.demo_mode}
            elif command_type == CommandType.UPSERT_SITE_CONFIG.value:
                self.control_repo.upsert_site_config(payload)
                result_payload = {"updated": True}
            else:
                raise RuntimeError(f"Commande non supportée: {command_type}")
            self.control_repo.mark_command(command_id, CommandStatus.COMPLETED)
            self.event_repo.add_event("command_completed", {"command_type": command_type, "result": result_payload}, severity="info")
        except Exception as exc:
            self.control_repo.mark_command(command_id, CommandStatus.FAILED, error_text=str(exc))
            self.event_repo.add_event("command_failed", {"command_type": command_type, "error": str(exc)}, severity="error")

    def drain_runtime_messages(self) -> None:
        while True:
            try:
                message = self.runtime_queue.get_nowait()
            except queue.Empty:
                return
            if message["kind"] == "EVENT":
                self.event_repo.add_event(
                    message["event_type"],
                    message["payload"],
                    parcel_id=message.get("parcel_id"),
                    camera_id=message.get("camera_id"),
                    severity="warning" if "ambiguous" in message["event_type"] else "info",
                )
            elif message["kind"] == "TRACKLET":
                self.handle_tracklet(message["tracklet"])
            elif message["kind"] == "RECORDING":
                self.artifact_repo.add_recording(
                    source_id=message["source_id"],
                    segment_path=message["segment_path"],
                    started_at=message["started_at"],
                    ended_at=message["ended_at"],
                    frame_count=message["frame_count"],
                    size_bytes=message["size_bytes"],
                )

    def drain_inference_results(self) -> None:
        while True:
            try:
                message = self.inference_result_queue.get_nowait()
            except queue.Empty:
                return
            if message["kind"] == "MODEL_READY":
                self.inference_result_store["__model_ready__"] = message
            elif message["kind"] == "INFER_RESULT":
                self.inference_result_store[f"{message['camera_id']}:{message['frame_index']}"] = message
            elif message["kind"] == "INFER_ERROR":
                self.inference_result_store[f"{message['camera_id']}:{message['frame_index']}"] = {"error": message["error"]}

    def refresh_jobs(self) -> None:
        for source_id, (process, _) in list(self.camera_processes.items()):
            if not process.is_alive():
                self.job_repo.mark_job_stopped(JobType.CAMERA.value, source_id, status="EXITED")
                self.camera_processes.pop(source_id, None)
        for job_id, process in list(self.training_processes.items()):
            if not process.is_alive():
                self.job_repo.mark_job_stopped(JobType.TRAINING.value, job_id, status="EXITED")
                self.training_processes.pop(job_id, None)

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                self.drain_inference_results()
                self.drain_runtime_messages()
                self.refresh_jobs()
                for command in self.control_repo.list_pending_commands():
                    self.handle_command(command)
                time.sleep(float(self.config.get("runtime", "poll_interval_seconds", default=1.0)))
        finally:
            self.shutdown()


def main() -> int:
    supervisor = RuntimeSupervisor()
    stop = {"value": False}

    def _signal_handler(*_args):
        stop["value"] = True
        supervisor.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)
    supervisor.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
