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
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import (
    ArtifactRepository,
    ControlRepository,
    EventRepository,
    JobRepository,
    TrackingRepository,
)
from visionsort.datasets.pipeline import build_dataset
from visionsort.deployment.registry import activate_model, promote_model, rollback_to_previous_active, set_model_status
from visionsort.inference.engine import inference_worker_loop
from visionsort.runtime.demo_assets import ensure_demo_assets
from visionsort.runtime.pipeline_worker import pipeline_worker_loop
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
        self.pipeline_processes: dict[str, mp.Process] = {}
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
        return {row["id"]: row["role"] for row in self.db.fetch_all("SELECT id, role FROM sources")}

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
        sources = {row["id"]: dict(row) for row in self.db.fetch_all("SELECT * FROM sources")}
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

    def runtime_model_id(self, configured_model_id: str) -> str:
        if (
            self.config.get(
                "runtime", "model_selection", default="active_registry"
            )
            != "active_registry"
        ):
            return configured_model_id
        active = self.db.fetch_one(
            "SELECT id FROM model_registry WHERE is_active = 1 ORDER BY updated_at DESC LIMIT 1"
        )
        return str(active["id"]) if active is not None else configured_model_id

    def reload_runtime_model(self, model_id: str) -> bool:
        inference_process = getattr(self, "inference_process", None)
        if inference_process is not None and inference_process.is_alive():
            self.active_model_id = None
            self.ensure_model_loaded(model_id)
            return True
        self.active_model_id = None
        return False

    def start_source(self, source_id: str, *, session_id: str, session_start_global: float, replay_offset_ms: float = 0.0) -> None:
        row = self.db.fetch_one("SELECT * FROM sources WHERE id = ?", (source_id,))
        if row is None:
            raise RuntimeError("Source introuvable.")
        active_sources = len(self.camera_processes)
        training_active = any(process.is_alive() for process in self.training_processes.values())
        allowed, reason = self.arbiter.can_start_source(active_sources, training_active)
        if not allowed:
            raise RuntimeError(reason)
        runtime_model_id = self.runtime_model_id(str(row["model_id"]))
        if (
            self.camera_processes
            and self.active_model_id is not None
            and self.active_model_id != runtime_model_id
        ):
            raise RuntimeError("Toutes les sources actives doivent partager le même model_id pour le worker GPU unique.")
        self.ensure_model_loaded(runtime_model_id)
        stop_event = self.ctx.Event()
        cfg = dict(row)
        cfg["model_id"] = runtime_model_id
        cfg["replay_fps"] = 8.0
        cfg["session_id"] = session_id
        cfg["session_start_global"] = float(session_start_global)
        cfg["replay_offset_ms"] = float(replay_offset_ms)
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
        self.job_repo.upsert_job_run(
            JobType.CAMERA.value,
            source_id,
            process.pid or 0,
            "RUNNING",
            {
                "role": row["role"],
                "configured_model_id": row["model_id"],
                "runtime_model_id": runtime_model_id,
            },
        )
        self.control_repo.update_source_state(source_id, status=SourceStatus.CONNECTING.value, fps=0.0)

    def start_session(self, session_id: str) -> None:
        session = self.control_repo.get_capture_session(session_id)
        if session is None:
            raise RuntimeError("Session introuvable.")
        start_global = float(time.time())
        self.control_repo.update_capture_session(session_id, started_at=start_global)
        sources = self.control_repo.list_capture_session_sources(session_id)
        if not sources:
            raise RuntimeError("Session sans caméras assignées.")
        for sess_src in sources:
            self.start_source(
                sess_src["source_id"],
                session_id=session_id,
                session_start_global=start_global,
                replay_offset_ms=float(sess_src.get("time_offset_ms") or 0.0),
            )

    def stop_session(self, session_id: str) -> None:
        sources = self.control_repo.list_capture_session_sources(session_id)
        for sess_src in sources:
            self.stop_source(sess_src["source_id"])
        self.control_repo.update_capture_session(session_id, ended_at=float(time.time()))

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
        job_id = create_training_job(
            self.db, payload["dataset_id"], payload["model_id"], payload
        )
        if not allowed:
            if (
                self.config.get("gpu", "training_policy", default="queue")
                != "queue"
            ):
                self.artifact_repo.update_training_job(
                    job_id, "FAILED", error_text=reason
                )
                raise RuntimeError(reason)
            self.job_repo.upsert_job_run(
                JobType.TRAINING.value,
                job_id,
                0,
                "QUEUED",
                {
                    "model_id": payload["model_id"],
                    "reason": reason,
                    "priority": int(payload.get("priority", 0)),
                },
            )
            return job_id
        self.launch_training_job(job_id, payload)
        return job_id

    def launch_training_job(
        self, job_id: str, payload: dict[str, Any]
    ) -> None:
        dataset = self.db.fetch_one("SELECT summary_json FROM datasets WHERE id = ?", (payload["dataset_id"],))
        if dataset and dataset["summary_json"]:
            session_id = json.loads(dataset["summary_json"]).get("session_id")
            if session_id:
                self.db.execute(
                    "UPDATE capture_sessions SET pipeline_state = ?, last_training_job_id = ?, updated_at = ? WHERE id = ?",
                    ("TRAINING", job_id, utc_now(), session_id),
                )
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

    def start_pipeline_step(self, *, session_id: str, step: str, params: dict[str, Any]) -> str:
        step_name = str(step).upper()
        job_key = f"{session_id}:{step_name}:{int(time.time() * 1000)}"
        process = self.ctx.Process(
            target=pipeline_worker_loop,
            args=(str(DB_PATH), session_id, step_name, params),
            daemon=True,
            name=f"visionsort-pipeline-{job_key}",
        )
        process.start()
        self.pipeline_processes[job_key] = process
        self.job_repo.upsert_job_run(JobType.DATASET.value, job_key, process.pid or 0, "RUNNING", {"session_id": session_id, "step": step_name})
        return job_key

    def handle_tracklet(self, payload: dict[str, Any]) -> None:
        self.handle_tracklets([payload])

    def handle_tracklets(self, payloads: list[dict[str, Any]]) -> None:
        tracklets = [Tracklet(**payload) for payload in payloads]
        self.global_tracker.source_roles = self._source_roles()
        outcomes = self.global_tracker.process_tracklets(tracklets)
        for tracklet, (parcel_id, result, reasons, candidate) in zip(
            tracklets, outcomes, strict=True
        ):
            self.tracking_repo.upsert_tracklet(
                tracklet,
                parcel_id=parcel_id or None,
                match_result=result.value,
            )
            if parcel_id:
                parcel = self.global_tracker.parcels[parcel_id]
                self.tracking_repo.upsert_global_parcel(parcel)
            event_type = (
                "handoff_ambiguous"
                if result == MatchResult.AMBIGUOUS
                else "handoff_matched"
                if result == MatchResult.MATCHED
                else "tracklet_unmatched"
            )
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
                session_id=tracklet.session_id,
                source_id=tracklet.source_id,
                timestamp_global=tracklet.ended_at_global,
                model_id=tracklet.model_id,
                tracker_id=tracklet.tracker_id,
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
            elif command_type == CommandType.CREATE_SESSION.value:
                session_id = self.control_repo.create_capture_session(
                    name=payload["name"],
                    demo_mode=bool(payload.get("demo_mode", False)),
                    sources=payload.get("sources", []),
                    config=payload.get("config", {}),
                )
                result_payload = {"session_id": session_id}
            elif command_type == CommandType.START_SESSION.value:
                self.start_session(payload["session_id"])
                result_payload = {"session_id": payload["session_id"]}
            elif command_type == CommandType.STOP_SESSION.value:
                self.stop_session(payload["session_id"])
                job_key = self.start_pipeline_step(session_id=payload["session_id"], step="PROCESS_SESSION", params={})
                result_payload = {"session_id": payload["session_id"], "job_key": job_key}
            elif command_type == CommandType.START_SOURCE.value:
                sid = payload.get("session_id", "session-adhoc")
                start_global = float(payload.get("session_start_global") or time.time())
                self.start_source(payload["source_id"], session_id=sid, session_start_global=start_global, replay_offset_ms=float(payload.get("replay_offset_ms") or 0.0))
                result_payload = {"source_id": payload["source_id"], "session_id": sid}
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
                if "session_id" not in payload:
                    raise RuntimeError("session_id requis pour CREATE_DATASET.")
                result_payload = build_dataset(self.db, session_id=payload["session_id"], name=payload.get("name", "autodataset"))
            elif command_type == CommandType.START_TRAINING.value:
                result_payload = {"job_id": self.start_training(payload)}
            elif command_type == CommandType.RUN_PIPELINE_STEP.value:
                job_key = self.start_pipeline_step(session_id=payload["session_id"], step=payload["step"], params=payload.get("params", {}))
                result_payload = {"job_key": job_key}
            elif command_type == CommandType.UPDATE_DATASET_ITEM.value:
                self.artifact_repo.update_dataset_item(
                    payload["item_id"],
                    annotation_status=payload.get("annotation_status"),
                )
                item = self.db.fetch_one("SELECT dataset_id, session_id FROM dataset_items WHERE id = ?", (payload["item_id"],))
                result_payload = {"item_id": payload["item_id"]}
                if item and item["dataset_id"] and item["session_id"]:
                    job_key = self.start_pipeline_step(
                        session_id=item["session_id"],
                        step="FINALIZE_DATASET",
                        params={"dataset_id": item["dataset_id"]},
                    )
                    result_payload["job_key"] = job_key
            elif command_type == CommandType.PROMOTE_MODEL.value:
                promote_model(self.db, payload["model_id"])
                reloaded = self.reload_runtime_model(payload["model_id"])
                result_payload = {
                    "model_id": payload["model_id"],
                    "status": "CHAMPION",
                    "runtime_reloaded": reloaded,
                }
            elif command_type == CommandType.REJECT_MODEL.value:
                set_model_status(self.db, payload["model_id"], "REJECTED")
                result_payload = {"model_id": payload["model_id"], "status": "REJECTED"}
            elif command_type == CommandType.ARCHIVE_MODEL.value:
                set_model_status(self.db, payload["model_id"], "ARCHIVED")
                result_payload = {"model_id": payload["model_id"], "status": "ARCHIVED"}
            elif command_type == CommandType.ACTIVATE_MODEL.value:
                activate_model(self.db, payload["model_id"])
                reloaded = self.reload_runtime_model(payload["model_id"])
                result_payload = {
                    "model_id": payload["model_id"],
                    "runtime_reloaded": reloaded,
                }
            elif command_type == CommandType.ROLLBACK_MODEL.value:
                rollback_model_id = rollback_to_previous_active(self.db)
                reloaded = (
                    self.reload_runtime_model(rollback_model_id)
                    if rollback_model_id
                    else False
                )
                result_payload = {
                    "model_id": rollback_model_id,
                    "runtime_reloaded": reloaded,
                }
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
        tracklet_payloads: list[dict[str, Any]] = []
        while True:
            try:
                message = self.runtime_queue.get_nowait()
            except queue.Empty:
                if tracklet_payloads:
                    self.handle_tracklets(tracklet_payloads)
                return
            if message["kind"] == "EVENT":
                self.event_repo.add_event(
                    message["event_type"],
                    message["payload"],
                    parcel_id=message.get("parcel_id"),
                    camera_id=message.get("camera_id"),
                    severity="warning" if "ambiguous" in message["event_type"] else "info",
                    session_id=message.get("session_id"),
                    source_id=message.get("source_id"),
                    frame_index=message.get("frame_index"),
                    timestamp_global=message.get("timestamp_global"),
                    model_id=message.get("model_id"),
                    tracker_id=message.get("tracker_id"),
                )
            elif message["kind"] == "TRACKLET":
                tracklet_payloads.append(message["tracklet"])
            elif message["kind"] == "RECORDING":
                self.artifact_repo.add_recording(
                    source_id=message["source_id"],
                    session_id=message.get("session_id"),
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
                training_job = self.db.fetch_one(
                    "SELECT status FROM training_jobs WHERE id = ?", (job_id,)
                )
                terminal_status = (
                    str(training_job["status"])
                    if training_job is not None
                    else "FAILED"
                )
                self.job_repo.mark_job_stopped(
                    JobType.TRAINING.value, job_id, status=terminal_status
                )
                self.training_processes.pop(job_id, None)
        for job_key, process in list(self.pipeline_processes.items()):
            if not process.is_alive():
                self.job_repo.mark_job_stopped(JobType.DATASET.value, job_key, status="EXITED")
                self.pipeline_processes.pop(job_key, None)
        if not self.camera_processes and not any(
            process.is_alive() for process in self.training_processes.values()
        ):
            queued = self.db.fetch_one(
                "SELECT * FROM training_jobs WHERE status = 'QUEUED' ORDER BY created_at ASC LIMIT 1"
            )
            if queued is not None:
                self.launch_training_job(
                    str(queued["id"]),
                    json.loads(queued["recipe_json"] or "{}"),
                )

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
