from __future__ import annotations

import json
import multiprocessing as mp
import queue
import signal
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ajoute le répertoire racine du projet au sys.path pour l'exécution en standalone
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from visionsort.acquisition.worker import camera_worker_loop
from visionsort.core.config import load_config
from visionsort.core.enums import (
    CommandStatus,
    CommandType,
    JobType,
    MatchResult,
    ParcelState,
    SourceStatus,
)
from visionsort.core.paths import DB_PATH, ROOT_DIR, ensure_project_dirs
from visionsort.core.types import GlobalParcel, Tracklet
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import (
    ArtifactRepository,
    ControlRepository,
    EventRepository,
    HandoffHypothesisRepository,
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
from visionsort.tracking.handoffs import PendingHandoffBuffer
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
        self.hypothesis_repo = HandoffHypothesisRepository(self.db)
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
        self.active_source_sessions: dict[str, str] = {}
        self.latest_stream_epoch_by_source: dict[str, int] = {}
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
        self.pending_handoff_buffer = PendingHandoffBuffer(
            self.db,
            topology_edges,
            window_seconds=float(
                self.config.get(
                    "tracking", "handoff_window_seconds", default=0.75
                )
            ),
            max_items=int(
                self.config.get(
                    "tracking", "handoff_buffer_max_items", default=1000
                )
            ),
            expiry_seconds=float(
                self.config.get(
                    "tracking", "handoff_expiry_seconds", default=30.0
                )
            ),
        )
        self._restore_global_tracker_state()
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

    @staticmethod
    def _tracklet_from_row(row: dict[str, Any]) -> Tracklet:
        summary = json.loads(row.get("summary_json") or "{}")
        first_bbox = tuple(
            float(value)
            for value in summary.get("first_bbox", summary.get("avg_bbox", [0, 0, 0, 0]))
        )
        last_bbox = tuple(
            float(value)
            for value in summary.get("last_bbox", summary.get("avg_bbox", [0, 0, 0, 0]))
        )
        return Tracklet(
            tracklet_id=str(row["tracklet_id"]),
            session_id=str(row.get("session_id") or ""),
            source_id=str(row.get("source_id") or row["camera_id"]),
            camera_id=str(row["camera_id"]),
            camera_role=str(row.get("camera_role") or row["camera_id"]),
            local_track_id=int(row["local_track_id"]),
            started_at_local=float(row.get("started_at_local") or 0.0),
            ended_at_local=float(row.get("ended_at_local") or 0.0),
            started_at_global=float(row["started_at_global"]),
            ended_at_global=float(row["ended_at_global"]),
            class_name=str(row["class_name"]),
            first_bbox=first_bbox,
            last_bbox=last_bbox,
            avg_speed=float(row["avg_speed"]),
            last_zone_id=row.get("last_zone_id"),
            frame_count=int(row["frame_count"]),
            observation_path=str(row["observation_path"]),
            summary_json=summary,
            model_id=row.get("model_id"),
            tracker_id=row.get("tracker_id"),
        )

    def _restore_global_tracker_state(self) -> None:
        for raw_row in self.db.fetch_all(
            "SELECT * FROM tracklets WHERE class_name = 'parcel'"
        ):
            row = dict(raw_row)
            tracklet = self._tracklet_from_row(row)
            self.global_tracker.tracklets[tracklet.tracklet_id] = tracklet
            if row.get("parcel_id"):
                self.global_tracker.tracklet_to_parcel[tracklet.tracklet_id] = str(
                    row["parcel_id"]
                )
        valid_states = {state.value: state for state in ParcelState}
        for raw_row in self.db.fetch_all("SELECT * FROM global_parcels"):
            row = dict(raw_row)
            raw_state = str(row["state"])
            state_value = raw_state.split(".")[-1]
            state = valid_states.get(state_value, ParcelState.ON_CONVEYOR)
            self.global_tracker.parcels[str(row["parcel_id"])] = GlobalParcel(
                parcel_id=str(row["parcel_id"]),
                state=state,
                last_camera_id=str(row["last_camera_id"]),
                first_seen_at=float(row["first_seen_at"]),
                last_seen_at=float(row["last_seen_at"]),
                current_tracklet_id=str(row["current_tracklet_id"]),
                assigned_destination=row.get("assigned_destination"),
                operator_id=row.get("operator_id"),
                appearance_signature=json.loads(row.get("appearance_json") or "[]"),
            )

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
        self.recover_interrupted_jobs()
        self.bootstrap_demo_sources()
        if not self.inference_process.is_alive():
            self.inference_process.start()
            self.job_repo.upsert_job_run(JobType.GPU_INFERENCE.value, "shared", self.inference_process.pid or 0, "RUNNING", {"active_model_id": self.active_model_id})
        self.job_repo.upsert_job_run(JobType.SUPERVISOR.value, "main", mp.current_process().pid or 0, "RUNNING", {"demo_mode": self.config.demo_mode})

    def recover_interrupted_jobs(self) -> None:
        self.db.execute(
            "UPDATE training_jobs SET status = 'QUEUED', error_text = 'Repris après redémarrage supervisor', updated_at = ? WHERE status = 'RUNNING'",
            (utc_now(),),
        )
        self.db.execute(
            "UPDATE pipeline_step_runs SET status = 'FAILED', error_text = 'Interrompu; étape reprenable', ended_at = ?, updated_at = ? WHERE status = 'RUNNING'",
            (time.time(), utc_now()),
        )
        self.db.execute(
            "UPDATE job_runs SET status = 'FAILED', updated_at = ? WHERE status = 'RUNNING'",
            (utc_now(),),
        )

    def shutdown(self) -> None:
        for source_id in list(self.camera_processes):
            self.stop_source(source_id)
        self.drain_runtime_messages()
        self.flush_pending_handoffs(force=True)
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
        flags = getattr(self, "control_flags", None)
        if flags is not None:
            flags["__inference_paused__"] = True
        try:
            drain_deadline = time.time() + float(
                self.config.get(
                    "runtime", "inference_result_ttl_seconds", default=5.0
                )
            )
            while flags is not None and any(
                str(key).startswith("__inflight__:") for key in list(flags.keys())
            ):
                self.drain_inference_results()
                if time.time() >= drain_deadline:
                    raise RuntimeError(
                        "Rechargement annulé: requêtes d'inférence en vol non terminées."
                    )
                time.sleep(0.02)
            self.sync_inference_sources()
            self.inference_result_store.pop("__model_ready__", None)
            self.inference_result_store.pop("__model_load_failed__", None)
            self.inference_request_queue.put(
                {"kind": "LOAD_MODEL", "model_id": model_id}
            )
            timeout = time.time() + 15
            while time.time() < timeout:
                self.drain_inference_results()
                failed = self.inference_result_store.pop(
                    "__model_load_failed__", None
                )
                if failed and failed.get("model_id") == model_id:
                    self.active_model_id = failed.get("active_model_id")
                    raise RuntimeError(str(failed.get("error")))
                ready = self.inference_result_store.pop("__model_ready__", None)
                if ready and ready.get("model_id") == model_id:
                    self.active_model_id = model_id
                    if hasattr(self, "job_repo"):
                        self.job_repo.upsert_job_run(
                            JobType.GPU_INFERENCE.value,
                            "shared",
                            self.inference_process.pid or 0,
                            "RUNNING",
                            ready,
                        )
                    return
                time.sleep(0.1)
            raise RuntimeError(f"Chargement du modèle expiré: {model_id}")
        finally:
            if flags is not None:
                flags["__inference_paused__"] = False

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
            self.ensure_model_loaded(model_id)
            return True
        return False

    def start_source(
        self,
        source_id: str,
        *,
        session_id: str,
        session_start_global: float,
        replay_offset_ms: float = 0.0,
        replay_loop: bool = False,
    ) -> None:
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
        cfg["replay_loop"] = bool(replay_loop)
        self.active_source_sessions[source_id] = session_id
        self.latest_stream_epoch_by_source[source_id] = -1
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
        session_config = json.loads(session.get("config_json") or "{}")
        replay_loop = bool(session_config.get("replay_loop", False))
        for sess_src in sources:
            self.start_source(
                sess_src["source_id"],
                session_id=session_id,
                session_start_global=start_global,
                replay_offset_ms=float(sess_src.get("time_offset_ms") or 0.0),
                replay_loop=replay_loop,
            )

    def stop_session(self, session_id: str) -> None:
        sources = self.control_repo.list_capture_session_sources(session_id)
        for sess_src in sources:
            self.stop_source(sess_src["source_id"])
        self.drain_runtime_messages()
        self.flush_pending_handoffs(force=True, session_id=session_id)
        self.control_repo.update_capture_session(session_id, ended_at=float(time.time()))

    def stop_source(self, source_id: str) -> None:
        if hasattr(self, "active_source_sessions"):
            self.active_source_sessions.pop(source_id, None)
        if hasattr(self, "latest_stream_epoch_by_source"):
            self.latest_stream_epoch_by_source.pop(source_id, None)
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
        prefix = f"{session_id}:{step_name}:"
        for existing_key, existing_process in self.pipeline_processes.items():
            if existing_key.startswith(prefix) and existing_process.is_alive():
                return existing_key
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

    def cancel_job(self, job_type: str, job_key: str) -> None:
        normalized = str(job_type).upper()
        if normalized == JobType.TRAINING.value:
            process = self.training_processes.pop(job_key, None)
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5)
            self.artifact_repo.update_training_job(
                job_key, "CANCELLED", error_text="Annulé par utilisateur"
            )
            self.job_repo.mark_job_stopped(
                JobType.TRAINING.value, job_key, status="CANCELLED"
            )
            return
        if normalized == JobType.DATASET.value:
            process = self.pipeline_processes.pop(job_key, None)
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5)
            parts = job_key.split(":", 2)
            if len(parts) >= 2:
                self.artifact_repo.cancel_pipeline_step(parts[0], parts[1])
            self.job_repo.mark_job_stopped(
                JobType.DATASET.value, job_key, status="CANCELLED"
            )
            return
        raise RuntimeError(f"Type de job non annulable: {job_type}")

    def handle_tracklet(self, payload: dict[str, Any]) -> None:
        self.handle_tracklets([payload])

    def _topology_rank(self, role: str) -> int:
        edges = self.config.get(
            "tracking", "site_topology", "edges", default=[]
        )
        ranks: dict[str, int] = {}
        for _ in range(len(edges) + 1):
            changed = False
            for edge in edges:
                left = str(edge["from_role"])
                right = str(edge["to_role"])
                proposed = ranks.get(left, 0) + 1
                if proposed > ranks.get(right, 0):
                    ranks[right] = proposed
                    changed = True
            if not changed:
                break
        return ranks.get(role, 0)

    def handle_tracklets(self, payloads: list[dict[str, Any]]) -> None:
        tracklets = [Tracklet(**payload) for payload in payloads]
        self.global_tracker.source_roles = self._source_roles()
        outcomes_by_id: dict[
            str,
            tuple[str, MatchResult, list[str], Any],
        ] = {}
        parcel_tracklets = sorted(
            (
                tracklet
                for tracklet in tracklets
                if tracklet.class_name == "parcel"
            ),
            key=lambda tracklet: (
                self._topology_rank(tracklet.camera_role),
                tracklet.ended_at_global,
                tracklet.tracklet_id,
            ),
        )
        by_rank: dict[int, list[Tracklet]] = {}
        for tracklet in parcel_tracklets:
            by_rank.setdefault(
                self._topology_rank(tracklet.camera_role), []
            ).append(tracklet)
        for rank in sorted(by_rank):
            wave = by_rank[rank]
            for tracklet in wave:
                self._try_resolve_hypotheses_with_later_evidence(tracklet)
            wave_outcomes = self.global_tracker.process_tracklets(wave)
            outcomes_by_id.update(
                {
                    tracklet.tracklet_id: outcome
                    for tracklet, outcome in zip(
                        wave, wave_outcomes, strict=True
                    )
                }
            )
        for tracklet in tracklets:
            if tracklet.class_name != "parcel":
                self.tracking_repo.upsert_tracklet(
                    tracklet,
                    parcel_id=None,
                    match_result=MatchResult.UNMATCHED.value,
                )
                continue
            parcel_id, result, reasons, candidate = outcomes_by_id[
                tracklet.tracklet_id
            ]
            hypothesis_id = None
            candidate_set = self.global_tracker.last_candidate_sets.get(
                tracklet.tracklet_id, []
            )
            if result == MatchResult.AMBIGUOUS:
                hypothesis_id = self.hypothesis_repo.create(
                    session_id=tracklet.session_id,
                    incoming_tracklet_id=tracklet.tracklet_id,
                    candidates=[asdict(item) for item in candidate_set],
                    expiry_seconds=float(
                        self.config.get(
                            "tracking",
                            "hypothesis_expiry_seconds",
                            default=120.0,
                        )
                    ),
                )
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
                    "candidates": [asdict(item) for item in candidate_set],
                    "hypothesis_id": hypothesis_id,
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

    def _try_resolve_hypotheses_with_later_evidence(
        self, later: Tracklet
    ) -> None:
        proposals: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        edges = self.config.get(
            "tracking", "site_topology", "edges", default=[]
        )
        for hypothesis in self.hypothesis_repo.pending(later.session_id):
            ambiguous = self.global_tracker.tracklets.get(
                str(hypothesis["incoming_tracklet_id"])
            )
            if ambiguous is None:
                continue
            edge = next(
                (
                    item
                    for item in edges
                    if str(item["from_role"]) == ambiguous.camera_role
                    and str(item["to_role"]) == later.camera_role
                ),
                None,
            )
            if edge is None:
                continue
            transit = later.started_at_global - ambiguous.ended_at_global
            if not (
                float(edge["min_transit_s"])
                <= transit
                <= float(edge["max_transit_s"])
            ):
                continue
            for candidate in json.loads(
                hypothesis.get("candidates_json") or "[]"
            ):
                evidence = self.global_tracker.continuation_evidence(
                    str(candidate["from_tracklet_id"]), later
                )
                if evidence is None:
                    continue
                combined = 0.6 * float(candidate["score"]) + 0.4 * evidence
                proposals.append((combined, hypothesis, candidate))
        proposals.sort(key=lambda item: item[0], reverse=True)
        if not proposals:
            return
        best_score, hypothesis, candidate = proposals[0]
        second_score = proposals[1][0] if len(proposals) > 1 else 0.0
        if (
            best_score < self.global_tracker.minimum_score
            or best_score - second_score < self.global_tracker.ambiguity_margin
        ):
            return
        outgoing_tracklet_id = str(candidate["from_tracklet_id"])
        incoming_tracklet_id = str(hypothesis["incoming_tracklet_id"])
        parcel_id = self.global_tracker.resolve_ambiguous(
            incoming_tracklet_id, outgoing_tracklet_id
        )
        self.hypothesis_repo.resolve(
            str(hypothesis["id"]),
            outgoing_tracklet_id=outgoing_tracklet_id,
            resolution={
                "mode": "automatic_later_evidence",
                "later_tracklet_id": later.tracklet_id,
                "combined_score": best_score,
            },
        )
        incoming = self.global_tracker.tracklets[incoming_tracklet_id]
        self.tracking_repo.upsert_tracklet(
            incoming,
            parcel_id=parcel_id,
            match_result=MatchResult.MATCHED.value,
        )
        self.tracking_repo.upsert_global_parcel(
            self.global_tracker.parcels[parcel_id]
        )
        self.event_repo.add_event(
            "handoff_ambiguity_resolved",
            {
                "hypothesis_id": hypothesis["id"],
                "incoming_tracklet_id": incoming_tracklet_id,
                "outgoing_tracklet_id": outgoing_tracklet_id,
                "later_tracklet_id": later.tracklet_id,
                "combined_score": best_score,
                "mode": "automatic_later_evidence",
            },
            parcel_id=parcel_id,
            session_id=later.session_id,
            timestamp_global=later.started_at_global,
        )

    def resolve_handoff_hypothesis(
        self,
        hypothesis_id: str,
        outgoing_tracklet_id: str,
        *,
        actor: str = "human",
    ) -> str:
        hypothesis = self.hypothesis_repo.get(hypothesis_id)
        if hypothesis is None:
            raise RuntimeError("Hypothèse introuvable.")
        incoming_tracklet_id = str(hypothesis["incoming_tracklet_id"])
        try:
            parcel_id = self.global_tracker.resolve_ambiguous(
                incoming_tracklet_id, outgoing_tracklet_id
            )
            incoming = self.global_tracker.tracklets[incoming_tracklet_id]
            self.tracking_repo.upsert_tracklet(
                incoming,
                parcel_id=parcel_id,
                match_result=MatchResult.MATCHED.value,
            )
            self.tracking_repo.upsert_global_parcel(
                self.global_tracker.parcels[parcel_id]
            )
        except RuntimeError:
            outgoing = self.db.fetch_one(
                "SELECT parcel_id FROM tracklets WHERE tracklet_id = ?",
                (outgoing_tracklet_id,),
            )
            if outgoing is None or not outgoing["parcel_id"]:
                raise
            parcel_id = str(outgoing["parcel_id"])
            incoming_row = self.db.fetch_one(
                """
                SELECT camera_id, ended_at_global FROM tracklets
                WHERE tracklet_id = ?
                """,
                (incoming_tracklet_id,),
            )
            self.db.execute(
                """
                UPDATE tracklets SET parcel_id = ?, match_result = 'MATCHED'
                WHERE tracklet_id = ?
                """,
                (parcel_id, incoming_tracklet_id),
            )
            if incoming_row is not None:
                self.db.execute(
                    """
                    UPDATE global_parcels
                    SET current_tracklet_id = ?, last_camera_id = ?,
                        last_seen_at = ?
                    WHERE parcel_id = ?
                    """,
                    (
                        incoming_tracklet_id,
                        incoming_row["camera_id"],
                        incoming_row["ended_at_global"],
                        parcel_id,
                    ),
                )
        self.hypothesis_repo.resolve(
            hypothesis_id,
            outgoing_tracklet_id=outgoing_tracklet_id,
            resolution={"mode": "human", "actor": actor},
        )
        return parcel_id

    def flush_pending_handoffs(
        self, *, force: bool = False, session_id: str | None = None
    ) -> None:
        if not hasattr(self, "pending_handoff_buffer"):
            return
        for _, payloads in self.pending_handoff_buffer.pop_ready_batches(
            force=force, session_id=session_id
        ):
            self.handle_tracklets(payloads)

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
            elif command_type == CommandType.CANCEL_JOB.value:
                self.cancel_job(payload["job_type"], payload["job_key"])
                result_payload = {
                    "job_type": payload["job_type"],
                    "job_key": payload["job_key"],
                    "status": "CANCELLED",
                }
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
            elif command_type == CommandType.RESOLVE_HANDOFF.value:
                parcel_id = self.resolve_handoff_hypothesis(
                    str(payload["hypothesis_id"]),
                    str(payload["outgoing_tracklet_id"]),
                    actor=str(payload.get("actor") or command.get("owner") or "human"),
                )
                result_payload = {
                    "hypothesis_id": payload["hypothesis_id"],
                    "parcel_id": parcel_id,
                    "status": "RESOLVED",
                }
            elif command_type == CommandType.REJECT_HANDOFF.value:
                self.hypothesis_repo.reject(
                    str(payload["hypothesis_id"]),
                    reason=str(payload.get("reason") or "rejet humain"),
                )
                result_payload = {
                    "hypothesis_id": payload["hypothesis_id"],
                    "status": "REJECTED",
                }
            elif command_type == CommandType.PROMOTE_MODEL.value:
                previous_active = self.db.fetch_one(
                    "SELECT id FROM model_registry WHERE is_active = 1 LIMIT 1"
                )
                try:
                    reloaded = self.reload_runtime_model(payload["model_id"])
                    promote_model(self.db, payload["model_id"])
                except Exception:
                    if previous_active is not None:
                        self.reload_runtime_model(str(previous_active["id"]))
                    raise
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
                previous_active = self.db.fetch_one(
                    "SELECT id FROM model_registry WHERE is_active = 1 LIMIT 1"
                )
                try:
                    reloaded = self.reload_runtime_model(payload["model_id"])
                    activate_model(self.db, payload["model_id"])
                except Exception:
                    if previous_active is not None:
                        self.reload_runtime_model(str(previous_active["id"]))
                    raise
                result_payload = {
                    "model_id": payload["model_id"],
                    "runtime_reloaded": reloaded,
                }
            elif command_type == CommandType.ROLLBACK_MODEL.value:
                previous_active = self.db.fetch_one(
                    "SELECT id FROM model_registry WHERE is_active = 1 LIMIT 1"
                )
                rollback_model_id = rollback_to_previous_active(self.db)
                try:
                    reloaded = (
                        self.reload_runtime_model(rollback_model_id)
                        if rollback_model_id
                        else False
                    )
                except Exception:
                    if previous_active is not None:
                        self.db.execute(
                            "UPDATE model_registry SET is_active = CASE WHEN id = ? THEN 1 ELSE 0 END, updated_at = ?",
                            (str(previous_active["id"]), utc_now()),
                        )
                        self.reload_runtime_model(str(previous_active["id"]))
                    raise
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
        immediate_tracklet_payloads: list[dict[str, Any]] = []
        while True:
            try:
                message = self.runtime_queue.get_nowait()
            except queue.Empty:
                if immediate_tracklet_payloads:
                    self.handle_tracklets(immediate_tracklet_payloads)
                self.flush_pending_handoffs()
                self.hypothesis_repo.expire()
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
                payload = message["tracklet"]
                if payload.get("class_name") == "parcel":
                    immediate_tracklet_payloads.extend(
                        self.pending_handoff_buffer.add(payload)
                    )
                else:
                    immediate_tracklet_payloads.append(payload)
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
        self._cleanup_expired_inference_results()
        while True:
            try:
                message = self.inference_result_queue.get_nowait()
            except queue.Empty:
                return
            if message["kind"] == "MODEL_READY":
                self.inference_result_store["__model_ready__"] = message
                continue
            if message["kind"] == "MODEL_LOAD_FAILED":
                self.inference_result_store["__model_load_failed__"] = message
                continue
            if message["kind"] not in {"INFER_RESULT", "INFER_ERROR"}:
                continue
            source_id = str(message.get("source_id") or "")
            request_id = str(message.get("request_id") or "")
            try:
                uuid.UUID(request_id)
            except (ValueError, TypeError):
                self._increment_inference_result_metric(source_id, "ignored")
                continue
            active_sessions = getattr(self, "active_source_sessions", None)
            if active_sessions is not None and active_sessions.get(source_id) != str(
                message.get("session_id")
            ):
                self._increment_inference_result_metric(source_id, "ignored")
                continue
            epoch = int(message.get("stream_epoch") or 0)
            epochs = getattr(self, "latest_stream_epoch_by_source", None)
            if epochs is not None:
                latest_epoch = int(epochs.get(source_id, -1))
                if epoch < latest_epoch:
                    self._increment_inference_result_metric(source_id, "ignored")
                    continue
                epochs[source_id] = max(latest_epoch, epoch)
            if time.time() > float(message.get("expires_at") or 0.0):
                self._increment_inference_result_metric(source_id, "late")
                continue
            stored = dict(message)
            stored["stored_at"] = time.time()
            if message["kind"] == "INFER_ERROR":
                stored["error"] = message["error"]
            self.inference_result_store[request_id] = stored

    def _increment_inference_result_metric(
        self, source_id: str, metric: str
    ) -> None:
        key = f"__inference_metrics__:{source_id}"
        current = dict(self.inference_result_store.get(key, {}))
        current[metric] = int(current.get(metric, 0)) + 1
        self.inference_result_store[key] = current

    def _cleanup_expired_inference_results(
        self, *, now: float | None = None
    ) -> None:
        current_time = float(now if now is not None else time.time())
        for request_id, result in list(self.inference_result_store.items()):
            if str(request_id).startswith("__"):
                continue
            if not isinstance(result, dict):
                continue
            expires_at = float(result.get("expires_at") or 0.0)
            if expires_at and expires_at <= current_time:
                self.inference_result_store.pop(request_id, None)
                self._increment_inference_result_metric(
                    str(result.get("source_id") or ""), "expired"
                )

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
                parts = job_key.split(":", 2)
                step_run = (
                    self.db.fetch_one(
                        """
                        SELECT status FROM pipeline_step_runs
                        WHERE session_id = ? AND step = ?
                        ORDER BY created_at DESC LIMIT 1
                        """,
                        (parts[0], parts[1]),
                    )
                    if len(parts) >= 2
                    else None
                )
                terminal_status = (
                    str(step_run["status"]) if step_run is not None else "FAILED"
                )
                self.job_repo.mark_job_stopped(
                    JobType.DATASET.value, job_key, status=terminal_status
                )
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
