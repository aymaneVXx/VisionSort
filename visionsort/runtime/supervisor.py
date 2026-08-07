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
from visionsort.core.config import AppConfig, load_config
from visionsort.core.enums import (
    CommandStatus,
    CommandType,
    JobType,
    MatchResult,
    ModelStatus,
    ParcelState,
    SourceStatus,
)
from visionsort.core.paths import DB_PATH, ROOT_DIR, ensure_project_dirs
from visionsort.core.site_config import apply_site_config, validate_site_config
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
from visionsort.deployment.registry import (
    activate_model,
    create_activation_history,
    finish_activation_history,
    mark_previous_activation,
    promote_model,
    rollback_to_previous_active,
    set_model_status,
    import_baseline_model,
    validate_activation_candidate,
    validate_promotion_candidate,
)
from visionsort.inference.engine import inference_worker_loop
from visionsort.media.archive import build_session_media_report
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
    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        config: AppConfig | None = None,
    ):
        ensure_project_dirs()
        configured = config or load_config()
        self.base_config_values = dict(configured.values)
        self.db_path = Path(db_path or DB_PATH)
        self.db = VisionSortDB(self.db_path)
        self.db.initialize()
        self.control_repo = ControlRepository(self.db)
        self.config = AppConfig(
            values=apply_site_config(
                self.base_config_values,
                self.control_repo.get_site_config(),
            )
        )
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
        self.active_runtime_models_by_task = self.manager.dict()
        self.active_source_sessions: dict[str, str] = {}
        self.session_site_configs: dict[str, dict[str, Any]] = {}
        self.active_source_pipelines: dict[str, list[dict[str, Any]]] = {}
        self.latest_stream_epoch_by_source: dict[str, int] = {}
        self.camera_processes: dict[str, tuple[mp.Process, Any]] = {}
        self.training_processes: dict[str, mp.Process] = {}
        self.pipeline_processes: dict[str, mp.Process] = {}
        self.active_model_id: str | None = None
        self.loaded_model_ids: set[str] = set()
        self.model_load_counts: dict[str, int] = {}
        self.rollback_model_holds: set[str] = set()
        self.last_model_load_error: dict[str, str] = {}
        self.last_model_unload_error: dict[str, str] = {}
        self.inference_runtime_snapshot: dict[str, Any] = {}
        self.last_inference_by_task: dict[str, dict[str, Any]] = {}
        self._last_runtime_model_state_publish = 0.0
        self.active_model_ids_by_task: dict[str, str] = {
            str(row["task"]): str(row["id"])
            for row in self.db.fetch_all(
                "SELECT id, task FROM model_registry WHERE is_active = 1"
            )
        }
        activated_at = time.time()
        for task, model_id in self.active_model_ids_by_task.items():
            generation_row = self.db.fetch_one(
                """
                SELECT MAX(routing_generation) AS generation
                FROM model_activation_history WHERE task = ?
                """,
                (task,),
            )
            self.active_runtime_models_by_task[task] = {
                "task": task,
                "model_id": model_id,
                "generation": max(
                    1,
                    int(
                        (generation_row["generation"] if generation_row else 0)
                        or 0
                    ),
                ),
                "activated_at": activated_at,
            }
        self.active_model_id = self.active_model_ids_by_task.get(
            "detection"
        ) or self.active_model_ids_by_task.get("segmentation")
        self._shutdown_complete = False
        self.arbiter = GPUResourceArbiter(
            allow_training_while_inference=bool(self.config.get("gpu", "allow_training_while_inference", default=False)),
            max_concurrent_live_sources=int(self.config.get("gpu", "max_concurrent_live_sources", default=3)),
        )
        topology_edges = self.config.get("tracking", "site_topology", "edges", default=[])
        self.global_tracker = GlobalParcelTracker(
            topology_edges=topology_edges,
            source_roles=self._source_roles(),
            zones_by_role=self.config.get("tracking", "zones", default={}),
        )
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
                str(self.db_path),
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
        self.refresh_runtime_routes_from_registry()
        if not self.inference_process.is_alive():
            self.inference_process.start()
        for task in sorted(self.active_model_ids_by_task):
            model_id = str(
                self.runtime_route(task).get("model_id") or ""
            )
            if not model_id:
                continue
            try:
                self.ensure_model_loaded(model_id, task=task)
                self.last_model_load_error.pop(model_id, None)
            except Exception as exc:
                self.last_model_load_error[model_id] = str(exc)
                self.event_repo.add_event(
                    "active_model_load_failed",
                    {
                        "task": task,
                        "model_id": model_id,
                        "error": str(exc),
                    },
                    severity="error",
                    model_id=model_id,
                )
        self.publish_runtime_model_state(force=True)
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
        if self._shutdown_complete:
            return
        for source_id in list(self.camera_processes):
            self.stop_source(source_id)
        self.drain_runtime_messages()
        self.flush_pending_handoffs(force=True)
        for job_id, process in list(self.training_processes.items()):
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            self.job_repo.mark_job_stopped(JobType.TRAINING.value, job_id, status="STOPPED")
            self.training_processes.pop(job_id, None)
        for job_key, process in list(self.pipeline_processes.items()):
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            self.job_repo.mark_job_stopped(
                JobType.DATASET.value, job_key, status="STOPPED"
            )
            self.pipeline_processes.pop(job_key, None)
        self.inference_stop_event.set()
        if self.inference_process.is_alive():
            self.inference_process.join(timeout=3)
            if self.inference_process.is_alive():
                self.inference_process.terminate()
                self.inference_process.join(timeout=3)
        self.job_repo.mark_job_stopped(JobType.GPU_INFERENCE.value, "shared", status="STOPPED")
        self.job_repo.mark_job_stopped(JobType.SUPERVISOR.value, "main", status="STOPPED")
        self._shutdown_complete = True
        self.manager.shutdown()

    def sync_inference_sources(self) -> None:
        sources = {row["id"]: dict(row) for row in self.db.fetch_all("SELECT * FROM sources")}
        self.inference_request_queue.put({"kind": "SYNC_SOURCES", "source_map": sources})

    def runtime_route(self, task: str) -> dict[str, Any]:
        routes = getattr(self, "active_runtime_models_by_task", None)
        if routes is not None:
            route = dict(routes.get(str(task), {}) or {})
            if route:
                return route
        model_id = getattr(self, "active_model_ids_by_task", {}).get(
            str(task)
        )
        return {
            "task": str(task),
            "model_id": model_id,
            "generation": 0,
            "activated_at": None,
        }

    def last_routing_generation(self, task: str) -> int:
        row = self.db.fetch_one(
            """
            SELECT MAX(routing_generation) AS generation
            FROM model_activation_history WHERE task = ?
            """,
            (str(task),),
        )
        historical = int((row["generation"] if row else 0) or 0)
        current = int(self.runtime_route(task).get("generation") or 0)
        return max(historical, current)

    def remove_runtime_route(self, task: str) -> None:
        task = str(task)
        routes = getattr(self, "active_runtime_models_by_task", None)
        if routes is not None:
            routes.pop(task, None)
        getattr(self, "active_model_ids_by_task", {}).pop(task, None)
        if task in {"detection", "segmentation"}:
            remaining = getattr(self, "active_model_ids_by_task", {})
            self.active_model_id = remaining.get("detection") or remaining.get(
                "segmentation"
            )

    def refresh_runtime_routes_from_registry(self) -> None:
        with self.db.connect() as conn:
            self.db._seed_model_activation_history(conn)
        rows = self.db.fetch_all(
            """
            SELECT id, task FROM model_registry
            WHERE is_active = 1 ORDER BY updated_at DESC
            """
        )
        for row in rows:
            task = str(row["task"])
            model_id = str(row["id"])
            current = self.runtime_route(task)
            if current.get("model_id") == model_id:
                continue
            self.apply_runtime_route(
                task,
                model_id,
                generation=max(1, self.last_routing_generation(task)),
            )

    def apply_runtime_route(
        self,
        task: str,
        model_id: str,
        *,
        generation: int | None = None,
        activated_at: float | None = None,
    ) -> dict[str, Any]:
        task = str(task)
        previous = self.runtime_route(task)
        next_generation = int(
            generation
            if generation is not None
            else int(previous.get("generation") or 0) + 1
        )
        route = {
            "task": task,
            "model_id": str(model_id),
            "generation": next_generation,
            "activated_at": float(
                activated_at if activated_at is not None else time.time()
            ),
        }
        routes = getattr(self, "active_runtime_models_by_task", None)
        if routes is not None:
            routes[task] = route
        if not hasattr(self, "active_model_ids_by_task"):
            self.active_model_ids_by_task = {}
        self.active_model_ids_by_task[task] = str(model_id)
        if task in {"detection", "segmentation"}:
            self.active_model_id = str(model_id)
        return route

    def model_references(self, model_id: str) -> dict[str, Any]:
        model_id = str(model_id)
        route_tasks = [
            str(task)
            for task in set(
                list(
                    getattr(
                        self, "active_model_ids_by_task", {}
                    ).keys()
                )
                + list(
                    getattr(
                        self, "active_runtime_models_by_task", {}
                    ).keys()
                )
            )
            if str(self.runtime_route(task).get("model_id") or "")
            == model_id
        ]
        fixed_sources: list[str] = []
        for source_id, pipelines in getattr(
            self, "active_source_pipelines", {}
        ).items():
            if any(
                not bool(pipeline.get("use_active", True))
                and str(
                    pipeline.get("configured_model_id")
                    or pipeline.get("model_id")
                    or ""
                )
                == model_id
                for pipeline in pipelines
            ):
                fixed_sources.append(str(source_id))
        inflight_request_ids: list[str] = []
        flags = getattr(self, "control_flags", {})
        for key, value in list(flags.items()):
            if not str(key).startswith("__inflight__:"):
                continue
            if isinstance(value, dict) and str(
                value.get("model_id") or ""
            ) == model_id:
                inflight_request_ids.append(
                    str(key).split(":", 1)[-1]
                )
        rollback_hold = model_id in getattr(
            self, "rollback_model_holds", set()
        )
        total = (
            len(route_tasks)
            + len(fixed_sources)
            + len(inflight_request_ids)
            + int(rollback_hold)
        )
        return {
            "model_id": model_id,
            "route_tasks": sorted(route_tasks),
            "fixed_sources": sorted(fixed_sources),
            "inflight_request_ids": sorted(inflight_request_ids),
            "rollback_hold": rollback_hold,
            "total": total,
        }

    def models_in_use(self) -> dict[str, dict[str, Any]]:
        candidates = set(getattr(self, "loaded_model_ids", set()))
        candidates.update(
            str(route.get("model_id"))
            for route in (
                self.runtime_route(task)
                for task in getattr(
                    self, "active_model_ids_by_task", {}
                )
            )
            if route.get("model_id")
        )
        for pipelines in getattr(
            self, "active_source_pipelines", {}
        ).values():
            for pipeline in pipelines:
                model_id = pipeline.get("configured_model_id") or pipeline.get(
                    "model_id"
                )
                if model_id:
                    candidates.add(str(model_id))
        return {
            model_id: self.model_references(model_id)
            for model_id in sorted(candidates)
        }

    def runtime_registry_consistency(
        self,
        *,
        worker_status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        active_rows = [
            dict(row)
            for row in self.db.fetch_all(
                """
                SELECT id, task, status, updated_at
                FROM model_registry
                WHERE is_active = 1
                ORDER BY task, updated_at DESC
                """
            )
        ]
        registry_by_task: dict[str, list[str]] = {}
        for row in active_rows:
            registry_by_task.setdefault(
                str(row["task"]), []
            ).append(str(row["id"]))
        routes_proxy = getattr(
            self, "active_runtime_models_by_task", {}
        )
        route_tasks = set(str(task) for task in routes_proxy.keys())
        tasks = sorted(set(registry_by_task) | route_tasks)
        status = dict(
            worker_status
            or getattr(self, "inference_runtime_snapshot", {})
            or {}
        )
        loaded = sorted(
            set(
                str(item)
                for item in status.get(
                    "loaded_model_ids",
                    getattr(self, "loaded_model_ids", set()),
                )
            )
        )
        inflight = {
            str(model_id): int(count)
            for model_id, count in dict(
                status.get("inflight_by_model", {})
            ).items()
        }
        references = self.models_in_use()
        errors: list[str] = []
        warnings: list[str] = []
        task_rows: list[dict[str, Any]] = []
        for task in tasks:
            registry_ids = registry_by_task.get(task, [])
            route = self.runtime_route(task)
            runtime_model_id = str(route.get("model_id") or "")
            registry_model_id = (
                registry_ids[0] if len(registry_ids) == 1 else ""
            )
            task_errors: list[str] = []
            if len(registry_ids) != 1:
                task_errors.append(
                    f"{len(registry_ids)} modèle(s) actif(s) en base"
                )
            if registry_model_id != runtime_model_id:
                task_errors.append(
                    "registre et routage runtime différents"
                )
            if runtime_model_id and runtime_model_id not in loaded:
                task_errors.append("modèle routé non chargé")
            task_errors.extend(
                str(error)
                for model_id, error in getattr(
                    self, "last_model_load_error", {}
                ).items()
                if model_id == runtime_model_id
            )
            errors.extend(f"{task}: {item}" for item in task_errors)
            task_rows.append(
                {
                    "task": task,
                    "registry_model_id": registry_model_id or None,
                    "registry_active_ids": registry_ids,
                    "runtime_model_id": runtime_model_id or None,
                    "routing_generation": int(
                        route.get("generation") or 0
                    ),
                    "loaded": runtime_model_id in loaded,
                    "references": int(
                        references.get(
                            runtime_model_id, {}
                        ).get("total", 0)
                    ),
                    "inflight": int(
                        inflight.get(runtime_model_id, 0)
                    ),
                    "consistent": not task_errors,
                    "errors": task_errors,
                }
            )
        required_by_sources: dict[str, list[str]] = {}
        for source_id, pipelines in getattr(
            self, "active_source_pipelines", {}
        ).items():
            for pipeline in pipelines:
                task = str(pipeline.get("task") or "detection")
                if bool(pipeline.get("use_active", True)):
                    target = str(
                        self.runtime_route(task).get("model_id") or ""
                    )
                else:
                    target = str(
                        pipeline.get("configured_model_id")
                        or pipeline.get("model_id")
                        or ""
                    )
                if target:
                    required_by_sources.setdefault(
                        target, []
                    ).append(str(source_id))
        for model_id, source_ids in sorted(required_by_sources.items()):
            if model_id not in loaded:
                errors.append(
                    f"{model_id}: requis par les sources actives "
                    f"{sorted(set(source_ids))}, mais non chargé"
                )
        unreferenced_loaded = [
            model_id
            for model_id in loaded
            if int(references.get(model_id, {}).get("total", 0)) == 0
        ]
        if unreferenced_loaded:
            warnings.append(
                "Modèles chargés sans référence: "
                + ", ".join(unreferenced_loaded)
            )
        return {
            "consistent": not errors,
            "checked_at": utc_now(),
            "tasks": task_rows,
            "loaded_model_ids": loaded,
            "inflight_by_model": inflight,
            "references_by_model": references,
            "required_by_active_sources": required_by_sources,
            "unreferenced_loaded_model_ids": unreferenced_loaded,
            "load_errors": dict(
                getattr(self, "last_model_load_error", {})
            ),
            "unload_errors": dict(
                getattr(self, "last_model_unload_error", {})
            ),
            "errors": errors,
            "warnings": warnings,
        }

    def publish_runtime_model_state(
        self, *, force: bool = False
    ) -> dict[str, Any]:
        now = time.monotonic()
        last_publish = float(
            getattr(self, "_last_runtime_model_state_publish", 0.0)
        )
        if not force and now - last_publish < 1.0:
            return {}
        self._last_runtime_model_state_publish = now
        worker_status = dict(
            getattr(self, "inference_runtime_snapshot", {}) or {}
        )
        worker_error = None
        inference_process = getattr(self, "inference_process", None)
        if (
            inference_process is not None
            and inference_process.is_alive()
            and hasattr(self, "inference_request_queue")
        ):
            try:
                worker_status = self.request_inference_runtime_status(
                    timeout=0.75
                )
            except Exception as exc:
                worker_error = str(exc)
        consistency = self.runtime_registry_consistency(
            worker_status=worker_status
        )
        if worker_error:
            consistency["warnings"].append(worker_error)
        history_rows = [
            dict(row)
            for row in self.db.fetch_all(
                """
                SELECT * FROM model_activation_history
                ORDER BY activated_at DESC LIMIT 50
                """
            )
        ]
        latest_by_task: dict[str, dict[str, Any]] = {}
        latest_rollback_by_task: dict[str, dict[str, Any]] = {}
        for row in history_rows:
            task = str(row["task"])
            latest_by_task.setdefault(task, row)
            if (
                row.get("rolled_back_from_activation_id")
                or str(row.get("reason") or "").lower().startswith(
                    "rollback"
                )
            ):
                latest_rollback_by_task.setdefault(task, row)
        state = {
            "active_model_id": self.active_model_id,
            "runtime_routes": {
                task: self.runtime_route(task)
                for task in sorted(
                    set(
                        getattr(
                            self,
                            "active_model_ids_by_task",
                            {},
                        )
                    )
                    | set(
                        getattr(
                            self,
                            "active_runtime_models_by_task",
                            {},
                        ).keys()
                    )
                )
            },
            "worker": worker_status,
            "consistency": consistency,
            "latest_activation_by_task": latest_by_task,
            "latest_rollback_by_task": latest_rollback_by_task,
            "published_at": utc_now(),
        }
        if hasattr(self, "job_repo"):
            self.job_repo.upsert_job_run(
                JobType.GPU_INFERENCE.value,
                "shared",
                (
                    int(inference_process.pid or 0)
                    if inference_process is not None
                    else 0
                ),
                (
                    "RUNNING"
                    if inference_process is not None
                    and inference_process.is_alive()
                    else "STOPPED"
                ),
                state,
            )
        return state

    def request_inference_runtime_status(
        self, *, timeout: float = 2.0
    ) -> dict[str, Any]:
        operation_id = str(uuid.uuid4())
        key = f"__inference_runtime_status__:{operation_id}"
        self.inference_result_store.pop(key, None)
        self.inference_request_queue.put(
            {
                "kind": "GET_RUNTIME_STATUS",
                "operation_id": operation_id,
            }
        )
        deadline = time.monotonic() + max(0.1, float(timeout))
        while time.monotonic() < deadline:
            self.drain_inference_results()
            status = self.inference_result_store.pop(key, None)
            if status:
                self.inference_runtime_snapshot = dict(status)
                return dict(status)
            time.sleep(0.02)
        raise RuntimeError("Timeout du statut du worker d'inférence.")

    def safe_unload_model(
        self,
        model_id: str,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        model_id = str(model_id)
        if model_id not in getattr(self, "loaded_model_ids", set()):
            return {
                "model_id": model_id,
                "status": "NOT_LOADED",
                "unloaded": False,
                "references": self.model_references(model_id),
            }
        references = self.model_references(model_id)
        blocking = (
            len(references["route_tasks"])
            + len(references["fixed_sources"])
            + int(references["rollback_hold"])
        )
        if blocking:
            return {
                "model_id": model_id,
                "status": "REFERENCED",
                "unloaded": False,
                "references": references,
            }
        deadline = time.monotonic() + float(
            timeout
            if timeout is not None
            else self.config.get(
                "runtime",
                "model_unload_timeout_seconds",
                default=5.0,
            )
        )
        stable_zero_checks = 0
        while time.monotonic() < deadline:
            self.drain_inference_results()
            references = self.model_references(model_id)
            if references["inflight_request_ids"]:
                stable_zero_checks = 0
                time.sleep(0.02)
                continue
            stable_zero_checks += 1
            if stable_zero_checks >= 2:
                break
            time.sleep(0.02)
        else:
            error = (
                "Timeout: requêtes en vol encore présentes; "
                "modèle conservé en mémoire."
            )
            if not hasattr(self, "last_model_unload_error"):
                self.last_model_unload_error = {}
            self.last_model_unload_error[model_id] = error
            if hasattr(self, "event_repo"):
                self.event_repo.add_event(
                    "model_unload_timeout",
                    {
                        "model_id": model_id,
                        "references": references,
                        "error": error,
                    },
                    severity="warning",
                )
            return {
                "model_id": model_id,
                "status": "TIMEOUT",
                "unloaded": False,
                "references": references,
                "error": error,
            }
        operation_id = str(uuid.uuid4())
        unloaded_key = f"__model_unloaded__:{operation_id}"
        deferred_key = f"__model_unload_deferred__:{operation_id}"
        self.inference_result_store.pop(unloaded_key, None)
        self.inference_result_store.pop(deferred_key, None)
        self.inference_request_queue.put(
            {
                "kind": "UNLOAD_MODEL",
                "model_id": model_id,
                "operation_id": operation_id,
            }
        )
        while time.monotonic() < deadline:
            self.drain_inference_results()
            deferred = self.inference_result_store.pop(
                deferred_key, None
            )
            if deferred:
                return {
                    "model_id": model_id,
                    "status": "DEFERRED",
                    "unloaded": False,
                    "references": self.model_references(model_id),
                    "worker": deferred,
                }
            confirmation = self.inference_result_store.pop(
                unloaded_key, None
            )
            if confirmation:
                self.loaded_model_ids.discard(model_id)
                getattr(self, "last_model_unload_error", {}).pop(
                    model_id, None
                )
                return {
                    "model_id": model_id,
                    "status": "UNLOADED",
                    "unloaded": True,
                    "references": self.model_references(model_id),
                    "worker": confirmation,
                }
            time.sleep(0.02)
        error = "Timeout en attente de MODEL_UNLOADED; modèle conservé."
        if not hasattr(self, "last_model_unload_error"):
            self.last_model_unload_error = {}
        self.last_model_unload_error[model_id] = error
        return {
            "model_id": model_id,
            "status": "TIMEOUT",
            "unloaded": False,
            "references": self.model_references(model_id),
            "error": error,
        }

    def active_dynamic_sources_for_task(self, task: str) -> list[str]:
        task = str(task)
        active_processes = set(
            getattr(self, "camera_processes", {}).keys()
        )
        return sorted(
            str(source_id)
            for source_id, pipelines in getattr(
                self, "active_source_pipelines", {}
            ).items()
            if source_id in active_processes
            and any(
                bool(pipeline.get("use_active", True))
                and str(pipeline.get("task") or "") == task
                for pipeline in pipelines
            )
        )

    def validate_model_routing(
        self,
        *,
        task: str,
        model_id: str,
        routing_generation: int,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        task = str(task)
        model_id = str(model_id)
        deadline = time.monotonic() + float(
            timeout
            if timeout is not None
            else self.config.get(
                "runtime",
                "model_switch_validation_timeout_seconds",
                default=8.0,
            )
        )
        dynamic_sources = self.active_dynamic_sources_for_task(task)
        if dynamic_sources:
            while time.monotonic() < deadline:
                self.drain_inference_results()
                latest = dict(
                    getattr(self, "last_inference_by_task", {}).get(
                        task, {}
                    )
                )
                if (
                    latest.get("model_id") == model_id
                    and int(latest.get("routing_generation") or 0)
                    == int(routing_generation)
                    and latest.get("source_id") in dynamic_sources
                    and latest.get("kind") == "INFER_RESULT"
                ):
                    return {
                        "mode": "ACTIVE_SOURCE_INFERENCE",
                        "dynamic_sources": dynamic_sources,
                        **latest,
                    }
                time.sleep(0.02)
            raise RuntimeError(
                "Aucune frame active n'a confirmé le nouveau routage "
                f"{task} -> {model_id} génération {routing_generation}."
            )
        operation_id = str(uuid.uuid4())
        key = f"__model_validation__:{operation_id}"
        self.inference_result_store.pop(key, None)
        self.inference_request_queue.put(
            {
                "kind": "VALIDATE_MODEL",
                "operation_id": operation_id,
                "task": task,
                "model_id": model_id,
                "routing_generation": int(routing_generation),
            }
        )
        while time.monotonic() < deadline:
            self.drain_inference_results()
            result = self.inference_result_store.pop(key, None)
            if result:
                if result["kind"] == "MODEL_VALIDATION_FAILED":
                    raise RuntimeError(str(result.get("error")))
                return {
                    "mode": "WORKER_READINESS_REQUEST",
                    **dict(result),
                }
            time.sleep(0.02)
        raise RuntimeError(
            f"Validation du modèle expirée: {task} -> {model_id}."
        )

    def _registry_task_snapshot(
        self, task: str
    ) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.fetch_all(
                """
                SELECT id, status, is_active, updated_at
                FROM model_registry WHERE task = ?
                """,
                (str(task),),
            )
        ]

    def _restore_registry_task_snapshot(
        self, task: str, snapshot: list[dict[str, Any]]
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE model_registry
                SET is_active = 0, updated_at = ?
                WHERE task = ?
                """,
                (utc_now(), str(task)),
            )
            for row in snapshot:
                conn.execute(
                    """
                    UPDATE model_registry
                    SET status = ?, is_active = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        str(row["status"]),
                        int(row["is_active"]),
                        str(row["updated_at"]),
                        str(row["id"]),
                    ),
                )

    def switch_runtime_model(
        self,
        model_id: str,
        *,
        actor: str = "runtime-supervisor",
        reason: str = "activation demandée",
        promote: bool = False,
        rollback: bool = False,
    ) -> dict[str, Any]:
        model = self.db.fetch_one(
            "SELECT * FROM model_registry WHERE id = ?",
            (str(model_id),),
        )
        if model is None:
            raise RuntimeError("Modèle introuvable.")
        model = dict(model)
        task = str(model["task"])
        if promote:
            validate_promotion_candidate(self.db, str(model_id))
        allowed_statuses = (
            {ModelStatus.CANDIDATE.value}
            if promote
            else {
                ModelStatus.BASELINE.value,
                ModelStatus.CHAMPION.value,
                ModelStatus.ARCHIVED.value,
            }
        )
        if str(model["status"]) not in allowed_statuses:
            raise RuntimeError(
                f"Activation refusée: statut {model['status']} invalide."
            )
        previous_route = self.runtime_route(task)
        previous_model_id = str(
            previous_route.get("model_id") or ""
        ) or None
        next_generation = self.last_routing_generation(task) + 1
        source_ids = self.active_dynamic_sources_for_task(task)
        session_ids = sorted(
            {
                str(
                    getattr(self, "active_source_sessions", {}).get(
                        source_id
                    )
                )
                for source_id in source_ids
                if getattr(self, "active_source_sessions", {}).get(
                    source_id
                )
            }
        )
        current_activation = self.db.fetch_one(
            """
            SELECT id, status, completed_at
            FROM model_activation_history
            WHERE task = ? AND activated_model_id = ?
              AND status = 'ACTIVE' AND runtime_applied = 1
            ORDER BY COALESCE(completed_at, activated_at) DESC LIMIT 1
            """,
            (task, previous_model_id),
        )
        activation_id = create_activation_history(
            self.db,
            task=task,
            previous_model_id=previous_model_id,
            activated_model_id=str(model_id),
            routing_generation=next_generation,
            actor=actor,
            reason=reason,
            session_id=session_ids[0] if len(session_ids) == 1 else None,
            source_ids=source_ids,
            rolled_back_from_activation_id=(
                str(current_activation["id"])
                if rollback and current_activation is not None
                else None
            ),
            metadata={
                "session_ids": session_ids,
                "switch_steps": {
                    "loading": "PENDING",
                    "routing": "PENDING",
                    "verification": "PENDING",
                    "registry": "PENDING",
                    "unload_previous": "PENDING",
                },
            },
        )
        registry_snapshot = self._registry_task_snapshot(task)
        route_applied = False
        registry_persisted = False
        previous_history_marked = False
        validation: dict[str, Any] | None = None
        unload_result: dict[str, Any] | None = None
        steps = {
            "loading": "PENDING",
            "routing": "PENDING",
            "verification": "PENDING",
            "registry": "PENDING",
            "unload_previous": "PENDING",
        }
        if not hasattr(self, "rollback_model_holds"):
            self.rollback_model_holds = set()
        if previous_model_id:
            self.rollback_model_holds.add(previous_model_id)
        self.rollback_model_holds.add(str(model_id))
        try:
            if not promote:
                validate_activation_candidate(self.db, str(model_id))
            self.ensure_model_loaded(str(model_id))
            steps["loading"] = "COMPLETED"
            route = self.apply_runtime_route(
                task,
                str(model_id),
                generation=next_generation,
            )
            route_applied = True
            steps["routing"] = "COMPLETED"
            validation = self.validate_model_routing(
                task=task,
                model_id=str(model_id),
                routing_generation=int(route["generation"]),
            )
            steps["verification"] = "COMPLETED"
            if promote:
                promote_model(self.db, str(model_id))
            else:
                activate_model(self.db, str(model_id))
            registry_persisted = True
            steps["registry"] = "COMPLETED"
            previous_history_marked = bool(
                mark_previous_activation(
                    self.db,
                    task=task,
                    model_id=previous_model_id,
                    status=(
                        "ROLLED_BACK"
                        if rollback
                        else "SUPERSEDED"
                    ),
                )
            )
            finish_activation_history(
                self.db,
                activation_id,
                status="ACTIVE",
                runtime_applied=True,
                metadata={
                    "switch_steps": steps,
                    "validation": validation,
                },
            )
            self.rollback_model_holds.discard(str(model_id))
            if previous_model_id:
                self.rollback_model_holds.discard(previous_model_id)
            if previous_model_id and previous_model_id != str(model_id):
                unload_result = self.safe_unload_model(
                    previous_model_id
                )
            else:
                unload_result = {
                    "status": "NOT_REQUIRED",
                    "unloaded": False,
                }
            steps["unload_previous"] = str(
                unload_result.get("status") or "UNKNOWN"
            )
            finish_activation_history(
                self.db,
                activation_id,
                status="ACTIVE",
                runtime_applied=True,
                metadata={
                    "switch_steps": steps,
                    "validation": validation,
                    "unload_previous": unload_result,
                },
            )
            if hasattr(self, "event_repo"):
                self.event_repo.add_event(
                    "runtime_model_switched",
                    {
                        "activation_id": activation_id,
                        "task": task,
                        "previous_model_id": previous_model_id,
                        "new_model_id": str(model_id),
                        "routing_generation": next_generation,
                        "session_ids": session_ids,
                        "source_ids": source_ids,
                        "validation": validation,
                        "unload_previous": unload_result,
                    },
                    severity="info",
                    session_id=(
                        session_ids[0]
                        if len(session_ids) == 1
                        else None
                    ),
                    model_id=str(model_id),
                )
            result = {
                "activation_id": activation_id,
                "task": task,
                "previous_model_id": previous_model_id,
                "model_id": str(model_id),
                "routing_generation": next_generation,
                "runtime_applied": True,
                "validation": validation,
                "unload_previous": unload_result,
                "steps": steps,
            }
            if hasattr(self, "job_repo"):
                self.publish_runtime_model_state(force=True)
            return result
        except Exception as exc:
            rollback_validation = None
            rollback_error = None
            try:
                if route_applied and previous_model_id:
                    restored = self.apply_runtime_route(
                        task,
                        previous_model_id,
                        generation=int(
                            previous_route.get("generation") or 0
                        ),
                        activated_at=previous_route.get("activated_at"),
                    )
                    rollback_validation = self.validate_model_routing(
                        task=task,
                        model_id=previous_model_id,
                        routing_generation=int(
                            restored["generation"]
                        ),
                    )
                elif route_applied:
                    self.remove_runtime_route(task)
                if registry_persisted:
                    self._restore_registry_task_snapshot(
                        task, registry_snapshot
                    )
                if (
                    previous_history_marked
                    and current_activation is not None
                ):
                    self.db.execute(
                        """
                        UPDATE model_activation_history
                        SET status = ?, completed_at = ?
                        WHERE id = ?
                        """,
                        (
                            str(current_activation["status"]),
                            current_activation["completed_at"],
                            str(current_activation["id"]),
                        ),
                    )
            except Exception as restore_exc:
                rollback_error = str(restore_exc)
            finally:
                self.rollback_model_holds.discard(str(model_id))
                if previous_model_id:
                    self.rollback_model_holds.discard(previous_model_id)
            failed_unload = None
            if str(model_id) != previous_model_id:
                failed_unload = self.safe_unload_model(str(model_id))
            finish_activation_history(
                self.db,
                activation_id,
                status="FAILED",
                runtime_applied=False,
                error_text=str(exc),
                metadata={
                    "switch_steps": steps,
                    "rollback_validation": rollback_validation,
                    "rollback_error": rollback_error,
                    "failed_model_unload": failed_unload,
                },
            )
            if hasattr(self, "event_repo"):
                self.event_repo.add_event(
                    "runtime_model_switch_failed",
                    {
                        "activation_id": activation_id,
                        "task": task,
                        "previous_model_id": previous_model_id,
                        "new_model_id": str(model_id),
                        "routing_generation": next_generation,
                        "error": str(exc),
                        "rollback_error": rollback_error,
                        "failed_model_unload": failed_unload,
                    },
                    severity="error",
                    model_id=str(model_id),
                )
            if hasattr(self, "job_repo"):
                self.publish_runtime_model_state(force=True)
            suffix = (
                f" Restauration incomplète: {rollback_error}"
                if rollback_error
                else ""
            )
            raise RuntimeError(
                f"Échec de la bascule {task} vers {model_id}: {exc}.{suffix}"
            ) from exc

    def ensure_model_loaded(
        self,
        model_id: str,
        *,
        task: str | None = None,
        force_reload: bool = False,
    ) -> None:
        loaded = getattr(self, "loaded_model_ids", set())
        if model_id in loaded and not force_reload:
            return
        flags = getattr(self, "control_flags", None)
        pause_key = f"__inference_paused__:{model_id}"
        if flags is not None:
            flags[pause_key] = True
        try:
            drain_deadline = time.time() + float(
                self.config.get(
                    "runtime", "inference_result_ttl_seconds", default=5.0
                )
            )

            def matching_inflight() -> bool:
                if flags is None:
                    return False
                for key, value in list(flags.items()):
                    if not str(key).startswith("__inflight__:"):
                        continue
                    if not isinstance(value, dict):
                        return True
                    if str(value.get("model_id") or "") == model_id:
                        return True
                return False

            while matching_inflight():
                self.drain_inference_results()
                if time.time() >= drain_deadline:
                    raise RuntimeError(
                        "Rechargement annulé: requêtes d'inférence en vol "
                        f"non terminées pour {model_id}."
                    )
                time.sleep(0.02)
            self.sync_inference_sources()
            ready_key = f"__model_ready__:{model_id}"
            failed_key = f"__model_load_failed__:{model_id}"
            self.inference_result_store.pop(ready_key, None)
            self.inference_result_store.pop(failed_key, None)
            self.inference_request_queue.put(
                {
                    "kind": "LOAD_MODEL",
                    "model_id": model_id,
                    "task": task,
                    "reload": bool(force_reload),
                }
            )
            timeout = time.time() + 15
            while time.time() < timeout:
                self.drain_inference_results()
                failed = self.inference_result_store.pop(failed_key, None)
                if failed:
                    raise RuntimeError(str(failed.get("error")))
                ready = self.inference_result_store.pop(ready_key, None)
                if ready:
                    if not hasattr(self, "loaded_model_ids"):
                        self.loaded_model_ids = set()
                    self.loaded_model_ids.update(
                        str(item)
                        for item in ready.get(
                            "loaded_model_ids", [model_id]
                        )
                    )
                    getattr(
                        self, "last_model_load_error", {}
                    ).pop(model_id, None)
                    resolved_task = str(
                        ready.get("task") or task or "detection"
                    )
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
                flags[pause_key] = False

    def runtime_model_id(self, configured_model_id: str) -> str:
        if (
            self.config.get(
                "runtime", "model_selection", default="active_registry"
            )
            != "active_registry"
        ):
            return configured_model_id
        configured = self.db.fetch_one(
            "SELECT task FROM model_registry WHERE id = ?",
            (configured_model_id,),
        )
        task = str(configured["task"] if configured else "detection")
        route = self.runtime_route(task)
        return str(route.get("model_id") or configured_model_id)

    def resolve_model_pipeline(
        self,
        source_id: str,
        *,
        configured_model_id: str,
        snapshot: str | list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(snapshot, str):
            try:
                raw_assignments = json.loads(snapshot or "[]")
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Snapshot de pipeline invalide pour {source_id}."
                ) from exc
        elif snapshot is not None:
            raw_assignments = list(snapshot)
        else:
            raw_assignments = self.control_repo.list_source_model_assignments(
                source_id
            )
        if not raw_assignments:
            model = self.db.fetch_one(
                "SELECT task FROM model_registry WHERE id = ?",
                (configured_model_id,),
            )
            task = str(model["task"] if model else "detection")
            raw_assignments = [
                {
                    "pipeline_role": {
                        "detection": "parcel_detection",
                        "segmentation": "parcel_segmentation",
                        "pose": "operator_pose",
                    }.get(task, task),
                    "task": task,
                    "model_id": configured_model_id,
                    "use_active": True,
                    "enabled": True,
                }
            ]
        pipeline: list[dict[str, Any]] = []
        seen_roles: set[str] = set()
        use_registry = (
            self.config.get(
                "runtime", "model_selection", default="active_registry"
            )
            == "active_registry"
        )
        for assignment in raw_assignments:
            if not bool(assignment.get("enabled", True)):
                continue
            role = str(assignment["pipeline_role"])
            task = str(assignment["task"])
            if role in seen_roles:
                raise RuntimeError(
                    f"Pipeline dupliqué pour {source_id}: {role}"
                )
            seen_roles.add(role)
            configured_assignment_model_id = assignment.get("model_id")
            use_active = bool(assignment.get("use_active", True))
            model_id = configured_assignment_model_id
            if use_registry and use_active:
                active = self.db.fetch_one(
                    """
                    SELECT id FROM model_registry
                    WHERE task = ? AND is_active = 1
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (task,),
                )
                if active is not None:
                    model_id = active["id"]
            if not model_id:
                raise RuntimeError(
                    f"Aucun modèle disponible pour le pipeline {role}."
                )
            model = self.db.fetch_one(
                "SELECT task FROM model_registry WHERE id = ?",
                (str(model_id),),
            )
            if model is None or str(model["task"]) != task:
                raise RuntimeError(
                    f"Modèle {model_id} incompatible avec la tâche {task}."
                )
            pipeline.append(
                {
                    "pipeline_role": role,
                    "task": task,
                    "model_id": str(model_id),
                    "configured_model_id": (
                        str(configured_assignment_model_id)
                        if configured_assignment_model_id
                        else str(model_id)
                    ),
                    "use_active": use_active,
                }
            )
        if not pipeline:
            raise RuntimeError(
                f"Aucun pipeline d'inférence actif pour {source_id}."
            )
        return pipeline

    def reload_runtime_model(self, model_id: str) -> bool:
        inference_process = getattr(self, "inference_process", None)
        if inference_process is not None and inference_process.is_alive():
            row = self.db.fetch_one(
                "SELECT task FROM model_registry WHERE id = ?",
                (model_id,),
            )
            task = str(row["task"] if row else "")
            previous_model_id = str(
                self.runtime_route(task).get("model_id") or ""
            )
            self.ensure_model_loaded(model_id)
            self.apply_runtime_route(task, model_id)
            if previous_model_id and previous_model_id != model_id:
                self.safe_unload_model(previous_model_id)
            return True
        return False

    def apply_site_configuration(
        self, site_config: dict[str, Any]
    ) -> dict[str, Any]:
        validated = validate_site_config(site_config)
        self.control_repo.upsert_site_config(validated)
        effective = apply_site_config(
            getattr(self, "base_config_values", self.config.values),
            validated,
        )
        self.config = AppConfig(values=effective)
        return validated

    def session_runtime_config(
        self, session_id: str
    ) -> dict[str, Any]:
        cached = getattr(self, "session_site_configs", {}).get(
            str(session_id)
        )
        if cached is not None:
            return dict(cached)
        if not hasattr(self, "control_repo"):
            return dict(self.config.values)
        session = self.control_repo.get_capture_session(str(session_id))
        snapshot: dict[str, Any] = {}
        if session is not None:
            snapshot = json.loads(
                session.get("site_config_snapshot_json") or "{}"
            )
        effective = apply_site_config(
            getattr(self, "base_config_values", self.config.values),
            snapshot,
        )
        if not hasattr(self, "session_site_configs"):
            self.session_site_configs = {}
        self.session_site_configs[str(session_id)] = dict(effective)
        return effective

    def start_source(
        self,
        source_id: str,
        *,
        session_id: str,
        session_start_global: float,
        replay_offset_ms: float = 0.0,
        replay_fps: float = 8.0,
        replay_loop: bool = False,
        archive_required: bool = False,
        source_type_snapshot: str | None = None,
        source_uri_snapshot: str | None = None,
        model_pipeline_snapshot: str | list[dict[str, Any]] | None = None,
        runtime_config_values: dict[str, Any] | None = None,
    ) -> None:
        existing_session = getattr(self, "active_source_sessions", {}).get(
            source_id
        )
        if source_id in getattr(self, "camera_processes", {}) or existing_session:
            raise RuntimeError(
                f"Source deja active dans la session {existing_session or 'inconnue'}: {source_id}"
            )
        row = self.db.fetch_one("SELECT * FROM sources WHERE id = ?", (source_id,))
        if row is None:
            raise RuntimeError("Source introuvable.")
        active_sources = len(self.camera_processes)
        training_active = any(process.is_alive() for process in self.training_processes.values())
        allowed, reason = self.arbiter.can_start_source(active_sources, training_active)
        if not allowed:
            raise RuntimeError(reason)
        model_pipeline = self.resolve_model_pipeline(
            source_id,
            configured_model_id=str(row["model_id"]),
            snapshot=model_pipeline_snapshot,
        )
        for pipeline in model_pipeline:
            self.ensure_model_loaded(pipeline["model_id"])
        stop_event = self.ctx.Event()
        cfg = dict(row)
        if source_type_snapshot:
            cfg["source_type"] = source_type_snapshot
        if source_uri_snapshot:
            cfg["uri"] = source_uri_snapshot
        cfg["model_id"] = model_pipeline[0]["model_id"]
        cfg["model_task"] = model_pipeline[0]["task"]
        cfg["model_pipeline"] = model_pipeline
        cfg["replay_fps"] = float(replay_fps)
        cfg["session_id"] = session_id
        cfg["session_start_global"] = float(session_start_global)
        cfg["replay_offset_ms"] = float(replay_offset_ms)
        cfg["replay_loop"] = bool(replay_loop)
        cfg["archive_required"] = bool(archive_required)
        self.active_source_sessions[source_id] = session_id
        self.active_source_pipelines[source_id] = [
            dict(item) for item in model_pipeline
        ]
        self.latest_stream_epoch_by_source[source_id] = -1
        self.control_flags[source_id] = {"recording": False}
        process = self.ctx.Process(
            target=camera_worker_loop,
            args=(
                cfg,
                str(self.db_path),
                runtime_config_values or self.config.values,
                self.inference_request_queue,
                self.inference_result_store,
                self.runtime_queue,
                stop_event,
                self.control_flags,
                self.active_runtime_models_by_task,
            ),
            daemon=True,
            name=f"visionsort-camera-{row['role']}",
        )
        try:
            process.start()
        except Exception:
            self.active_source_sessions.pop(source_id, None)
            self.active_source_pipelines.pop(source_id, None)
            self.latest_stream_epoch_by_source.pop(source_id, None)
            self.control_flags.pop(source_id, None)
            raise
        self.camera_processes[source_id] = (process, stop_event)
        self.job_repo.upsert_job_run(
            JobType.CAMERA.value,
            source_id,
            process.pid or 0,
            "RUNNING",
            {
                "role": row["role"],
                "configured_model_id": row["model_id"],
                "runtime_model_ids": [
                    item["model_id"] for item in model_pipeline
                ],
                "model_pipeline": model_pipeline,
            },
        )
        self.control_repo.update_source_state(source_id, status=SourceStatus.CONNECTING.value, fps=0.0)

    def _start_session_legacy(self, session_id: str) -> None:
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
                replay_fps=float(sess_src.get("replay_fps") or 8.0),
                replay_loop=replay_loop,
                archive_required=bool(
                    sess_src.get("archive_required")
                ),
                source_type_snapshot=sess_src.get(
                    "source_type_snapshot"
                ),
                source_uri_snapshot=sess_src.get(
                    "source_uri_snapshot"
                ),
                model_pipeline_snapshot=sess_src.get(
                    "model_pipeline_json"
                ),
            )

    def start_session(self, session_id: str) -> None:
        session = self.control_repo.get_capture_session(session_id)
        if session is None:
            raise RuntimeError("Session introuvable.")
        sources = self.control_repo.list_capture_session_sources(session_id)
        if not sources:
            raise RuntimeError("Session sans cameras assignees.")
        runtime_status = str(session.get("runtime_status") or "CREATED")
        if runtime_status in {"STARTING", "RUNNING", "STOPPING"} or session.get(
            "started_at"
        ) is not None:
            raise RuntimeError("Session deja demarree.")
        if runtime_status in {"STOPPED", "FAILED"} or session.get(
            "ended_at"
        ) is not None:
            raise RuntimeError("Une session terminee ne peut pas etre redemarree.")
        source_ids = [str(row["source_id"]) for row in sources]
        if len(source_ids) != len(set(source_ids)):
            raise RuntimeError("Session invalide: une source est assignee plusieurs fois.")
        for row in sources:
            source = self.db.fetch_one(
                "SELECT role FROM sources WHERE id = ?",
                (str(row["source_id"]),),
            )
            if source is None or str(source["role"]) != str(row["camera_role"]):
                raise RuntimeError(
                    f"Session invalide: role incoherent pour {row['source_id']}."
                )
            owner = getattr(self, "active_source_sessions", {}).get(
                str(row["source_id"])
            )
            if owner and owner != session_id:
                raise RuntimeError(
                    f"Source {row['source_id']} deja utilisee par la session {owner}."
                )
        site_snapshot = validate_site_config(
            self.control_repo.get_site_config()
        )
        effective_config = apply_site_config(
            getattr(self, "base_config_values", self.config.values),
            site_snapshot,
        )
        if not hasattr(self, "session_site_configs"):
            self.session_site_configs = {}
        self.session_site_configs[session_id] = dict(effective_config)
        self.control_repo.update_capture_session(
            session_id,
            runtime_status="STARTING",
            start_error="",
            site_config_snapshot=site_snapshot,
        )
        start_global = float(time.time())
        session_config = json.loads(session.get("config_json") or "{}")
        replay_loop = bool(session_config.get("replay_loop", False))
        started_sources: list[str] = []
        try:
            for sess_src in sources:
                self.start_source(
                    sess_src["source_id"],
                    session_id=session_id,
                    session_start_global=start_global,
                    replay_offset_ms=float(sess_src.get("time_offset_ms") or 0.0),
                    replay_fps=float(sess_src.get("replay_fps") or 8.0),
                    replay_loop=replay_loop,
                    archive_required=bool(sess_src.get("archive_required")),
                    source_type_snapshot=sess_src.get("source_type_snapshot"),
                    source_uri_snapshot=sess_src.get("source_uri_snapshot"),
                    model_pipeline_snapshot=sess_src.get("model_pipeline_json"),
                    runtime_config_values=effective_config,
                )
                started_sources.append(str(sess_src["source_id"]))
        except Exception as exc:
            active_for_session = [
                source_id
                for source_id, owner in getattr(
                    self, "active_source_sessions", {}
                ).items()
                if owner == session_id
            ]
            for started_source in reversed(
                list(dict.fromkeys(started_sources + active_for_session))
            ):
                try:
                    self.stop_source(started_source)
                except Exception:
                    pass
            self.control_repo.update_capture_session(
                session_id,
                ended_at=float(time.time()),
                runtime_status="FAILED",
                start_error=str(exc),
            )
            raise RuntimeError(
                f"Echec atomique du demarrage de la session: {exc}"
            ) from exc
        self.control_repo.update_capture_session(
            session_id,
            started_at=start_global,
            runtime_status="RUNNING",
            start_error="",
        )

    def stop_session(self, session_id: str) -> None:
        session = self.control_repo.get_capture_session(session_id)
        if session is None:
            raise RuntimeError("Session introuvable.")
        if str(session.get("runtime_status") or "") != "RUNNING":
            raise RuntimeError("Seule une session RUNNING peut etre arretee.")
        self.control_repo.update_capture_session(
            session_id, runtime_status="STOPPING"
        )
        sources = self.control_repo.list_capture_session_sources(session_id)
        for sess_src in sources:
            self.stop_source(sess_src["source_id"])
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            self.drain_runtime_messages()
            coverage_count = self.db.fetch_one(
                """
                SELECT COUNT(*) AS count
                FROM session_media_coverage
                WHERE session_id = ?
                """,
                (session_id,),
            )
            if int(
                (coverage_count["count"] if coverage_count else 0) or 0
            ) >= len(sources):
                break
            time.sleep(0.05)
        self.drain_runtime_messages()
        self.flush_pending_handoffs(force=True, session_id=session_id)
        report = build_session_media_report(self.db, session_id)
        self.control_repo.update_capture_session(
            session_id,
            ended_at=float(time.time()),
            media_report=report,
            runtime_status="STOPPED",
        )
        getattr(self, "session_site_configs", {}).pop(session_id, None)

    def stop_source(self, source_id: str) -> None:
        if hasattr(self, "active_source_sessions"):
            self.active_source_sessions.pop(source_id, None)
        if hasattr(self, "active_source_pipelines"):
            self.active_source_pipelines.pop(source_id, None)
        if hasattr(self, "latest_stream_epoch_by_source"):
            self.latest_stream_epoch_by_source.pop(source_id, None)
        data = self.camera_processes.pop(source_id, None)
        if data is None:
            self.control_repo.update_source_state(source_id, status=SourceStatus.OFFLINE.value, fps=0.0)
            return
        process, stop_event = data
        stop_event.set()
        process.join(
            timeout=max(
                8.0,
                float(
                    self.config.get(
                        "runtime",
                        "inference_result_ttl_seconds",
                        default=5.0,
                    )
                )
                + 3.0,
            )
        )
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
            args=(str(self.db_path), job_id, payload, self.config.demo_mode),
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
            args=(str(self.db_path), session_id, step_name, params),
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

    def _apply_global_parcel_events(
        self,
        parcel: GlobalParcel,
        events: list[dict[str, Any]],
        *,
        tracklet: Tracklet,
    ) -> None:
        state_rank = {
            ParcelState.ON_CONVEYOR: 0,
            ParcelState.PICK_CANDIDATE: 1,
            ParcelState.PICKED: 2,
            ParcelState.CARRIED: 3,
            ParcelState.DROP_CANDIDATE: 4,
            ParcelState.DROPPED: 5,
        }
        transitions = {
            "pickup_candidate": ParcelState.PICK_CANDIDATE,
            "parcel_picked": ParcelState.PICKED,
            "parcel_carried": ParcelState.CARRIED,
            "drop_candidate": ParcelState.DROP_CANDIDATE,
            "parcel_dropped": ParcelState.DROPPED,
        }
        for event in events:
            event_type = str(event.get("event_type") or "")
            target = transitions.get(event_type)
            if event_type == "destination_observed" and state_rank[parcel.state] >= state_rank[ParcelState.CARRIED]:
                target = ParcelState.DROP_CANDIDATE
            if event_type == "destination_confirmed" and state_rank[parcel.state] >= state_rank[ParcelState.CARRIED]:
                target = ParcelState.DROPPED
            if target is None or state_rank[target] < state_rank[parcel.state]:
                continue
            previous_state = parcel.state
            parcel.state = target
            try:
                payload = json.loads(event.get("payload_json") or "{}")
            except json.JSONDecodeError:
                payload = {}
            destination = payload.get("destination_zone")
            if target in {ParcelState.DROP_CANDIDATE, ParcelState.DROPPED} and destination:
                parcel.assigned_destination = str(destination)
            if (
                target == ParcelState.DROPPED
                and previous_state != ParcelState.DROPPED
                and event_type != "parcel_dropped"
            ):
                self.event_repo.add_event(
                    "parcel_dropped",
                    {
                        "global_parcel_id": parcel.parcel_id,
                        "local_parcel_key": event.get("local_parcel_key"),
                        "state": ParcelState.DROPPED.value,
                        "destination_zone": parcel.assigned_destination,
                        "derived_from": event_type,
                        "validated_on_site": False,
                    },
                    parcel_id=parcel.parcel_id,
                    camera_id=tracklet.camera_id,
                    session_id=tracklet.session_id,
                    source_id=tracklet.source_id,
                    timestamp_global=event.get("timestamp_global"),
                    model_id=tracklet.model_id,
                    tracker_id=tracklet.tracker_id,
                    local_parcel_key=event.get("local_parcel_key"),
                )

    def handle_tracklet(self, payload: dict[str, Any]) -> None:
        self.handle_tracklets([payload])

    def _topology_rank(
        self, role: str, session_id: str | None = None
    ) -> int:
        runtime_config = (
            AppConfig(values=self.session_runtime_config(session_id))
            if session_id
            else self.config
        )
        edges = runtime_config.get(
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
                tracklet.session_id,
                self._topology_rank(
                    tracklet.camera_role, tracklet.session_id
                ),
                tracklet.ended_at_global,
                tracklet.tracklet_id,
            ),
        )
        by_rank: dict[tuple[str, int], list[Tracklet]] = {}
        for tracklet in parcel_tracklets:
            by_rank.setdefault(
                (
                    tracklet.session_id,
                    self._topology_rank(
                        tracklet.camera_role, tracklet.session_id
                    ),
                ),
                [],
            ).append(tracklet)
        for session_rank in sorted(by_rank):
            wave = by_rank[session_rank]
            runtime_config = AppConfig(
                values=self.session_runtime_config(session_rank[0])
            )
            self.global_tracker.topology_edges = runtime_config.get(
                "tracking", "site_topology", "edges", default=[]
            )
            self.global_tracker.zones_by_role = runtime_config.get(
                "tracking", "zones", default={}
            )
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
                local_parcel_key = (
                    f"{tracklet.source_id}:{tracklet.local_track_id}"
                )
                local_events = self.event_repo.bind_local_parcel_events(
                    session_id=tracklet.session_id,
                    source_id=tracklet.source_id,
                    local_parcel_key=local_parcel_key,
                    parcel_id=parcel_id,
                )
                self._apply_global_parcel_events(
                    parcel, local_events, tracklet=tracklet
                )
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
        runtime_config = AppConfig(
            values=self.session_runtime_config(later.session_id)
        )
        edges = runtime_config.get(
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
        try:
            resolved = self.hypothesis_repo.resolve_transactional(
                str(hypothesis["id"]),
                outgoing_tracklet_id=outgoing_tracklet_id,
                actor="supervisor",
                topology_edges=edges,
                reason="preuve tardive automatique",
                resolution={
                    "mode": "automatic_later_evidence",
                    "later_tracklet_id": later.tracklet_id,
                    "combined_score": best_score,
                },
            )
        except RuntimeError:
            return
        parcel_id = str(resolved["parcel_id"])
        self._sync_resolved_handoff_in_memory(resolved)
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

    def _sync_resolved_handoff_in_memory(
        self, resolved: dict[str, Any]
    ) -> None:
        incoming_tracklet_id = str(
            resolved["incoming_tracklet_id"]
        )
        parcel_id = str(resolved["parcel_id"])
        incoming = self.global_tracker.tracklets.get(
            incoming_tracklet_id
        )
        if incoming is not None:
            self.global_tracker.tracklet_to_parcel[
                incoming_tracklet_id
            ] = parcel_id
        parcel = self.global_tracker.parcels.get(parcel_id)
        if parcel is not None and incoming is not None:
            parcel.last_camera_id = incoming.camera_id
            parcel.last_seen_at = incoming.ended_at_global
            parcel.current_tracklet_id = incoming_tracklet_id
            parcel.appearance_signature = (
                self.global_tracker._appearance(incoming)
                or parcel.appearance_signature
            )
            local_events = self.event_repo.bind_local_parcel_events(
                session_id=incoming.session_id,
                source_id=incoming.source_id,
                local_parcel_key=(
                    f"{incoming.source_id}:{incoming.local_track_id}"
                ),
                parcel_id=parcel_id,
            )
            self._apply_global_parcel_events(
                parcel, local_events, tracklet=incoming
            )
            self.tracking_repo.upsert_global_parcel(parcel)

    def resolve_handoff_hypothesis(
        self,
        hypothesis_id: str,
        outgoing_tracklet_id: str,
        *,
        actor: str = "human",
    ) -> str:
        hypothesis = self.hypothesis_repo.get(hypothesis_id)
        runtime_config = AppConfig(
            values=self.session_runtime_config(
                str(hypothesis.get("session_id") or "")
                if hypothesis
                else ""
            )
        )
        edges = runtime_config.get(
            "tracking", "site_topology", "edges", default=[]
        )
        resolved = self.hypothesis_repo.resolve_transactional(
            hypothesis_id,
            outgoing_tracklet_id=outgoing_tracklet_id,
            actor=actor,
            topology_edges=edges,
            reason="résolution humaine explicite",
            resolution={"mode": "human"},
        )
        self._sync_resolved_handoff_in_memory(resolved)
        return str(resolved["parcel_id"])

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
                    actor=str(
                        payload.get("actor")
                        or command.get("owner")
                        or "human"
                    ),
                )
                result_payload = {
                    "hypothesis_id": payload["hypothesis_id"],
                    "status": "REJECTED",
                }
            elif command_type == CommandType.PROMOTE_MODEL.value:
                switch = self.switch_runtime_model(
                    str(payload["model_id"]),
                    actor=str(
                        payload.get("actor")
                        or command.get("owner")
                        or "streamlit"
                    ),
                    reason=str(
                        payload.get("reason")
                        or "promotion du modèle candidat"
                    ),
                    promote=True,
                )
                result_payload = {
                    "model_id": payload["model_id"],
                    "status": "CHAMPION",
                    "task": switch["task"],
                    "runtime_reloaded": True,
                    **switch,
                }
            elif command_type == CommandType.REJECT_MODEL.value:
                set_model_status(self.db, payload["model_id"], "REJECTED")
                result_payload = {"model_id": payload["model_id"], "status": "REJECTED"}
            elif command_type == CommandType.ARCHIVE_MODEL.value:
                set_model_status(self.db, payload["model_id"], "ARCHIVED")
                result_payload = {"model_id": payload["model_id"], "status": "ARCHIVED"}
            elif command_type == CommandType.ACTIVATE_MODEL.value:
                switch = self.switch_runtime_model(
                    str(payload["model_id"]),
                    actor=str(
                        payload.get("actor")
                        or command.get("owner")
                        or "streamlit"
                    ),
                    reason=str(
                        payload.get("reason")
                        or "activation manuelle"
                    ),
                )
                result_payload = {
                    "model_id": payload["model_id"],
                    "task": switch["task"],
                    "runtime_reloaded": True,
                    **switch,
                }
            elif command_type == CommandType.ROLLBACK_MODEL.value:
                task = str(payload.get("task") or "detection")
                rollback_model_id = rollback_to_previous_active(
                    self.db, task, apply=False
                )
                switch = self.switch_runtime_model(
                    rollback_model_id,
                    actor=str(
                        payload.get("actor")
                        or command.get("owner")
                        or "streamlit"
                    ),
                    reason=str(
                        payload.get("reason")
                        or "rollback explicite"
                    ),
                    rollback=True,
                )
                result_payload = {
                    "model_id": rollback_model_id,
                    "task": task,
                    "runtime_reloaded": True,
                    **switch,
                }
            elif command_type == CommandType.BOOTSTRAP_DEMO.value:
                self.bootstrap_demo_sources()
                result_payload = {"demo_mode": self.config.demo_mode}
            elif command_type == CommandType.IMPORT_BASELINE_MODEL.value:
                result_payload = import_baseline_model(
                    self.db,
                    source_path=str(payload["source_path"]),
                    task=str(payload["task"]),
                    name=payload.get("name"),
                )
            elif command_type == CommandType.UPSERT_SITE_CONFIG.value:
                validated = self.apply_site_configuration(payload)
                result_payload = {
                    "updated": True,
                    "effective_for": "new_sessions",
                    "config": validated,
                }
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
                    local_parcel_key=message.get("local_parcel_key"),
                )
                local_parcel_key = message.get("local_parcel_key")
                if local_parcel_key:
                    known = self.db.fetch_one(
                        """
                        SELECT * FROM tracklets
                        WHERE session_id = ? AND source_id = ?
                          AND local_track_id = ? AND parcel_id IS NOT NULL
                        ORDER BY ended_at_global DESC LIMIT 1
                        """,
                        (
                            message.get("session_id"),
                            message.get("source_id"),
                            int(str(local_parcel_key).rsplit(":", 1)[-1]),
                        ),
                    )
                    if known is not None:
                        parcel_id = str(known["parcel_id"])
                        parcel = self.global_tracker.parcels.get(parcel_id)
                        if parcel is not None:
                            tracklet = self._tracklet_from_row(dict(known))
                            local_events = self.event_repo.bind_local_parcel_events(
                                session_id=tracklet.session_id,
                                source_id=tracklet.source_id,
                                local_parcel_key=str(local_parcel_key),
                                parcel_id=parcel_id,
                            )
                            self._apply_global_parcel_events(
                                parcel, local_events, tracklet=tracklet
                            )
                            self.tracking_repo.upsert_global_parcel(parcel)
            elif message["kind"] == "TRACKLET":
                payload = message["tracklet"]
                if payload.get("class_name") == "parcel":
                    runtime_config = AppConfig(
                        values=self.session_runtime_config(
                            str(payload.get("session_id") or "")
                        )
                    )
                    self.pending_handoff_buffer.configure(
                        runtime_config.get(
                            "tracking", "site_topology", "edges", default=[]
                        ),
                        window_seconds=float(
                            runtime_config.get(
                                "tracking",
                                "handoff_window_seconds",
                                default=0.75,
                            )
                        ),
                        expiry_seconds=float(
                            runtime_config.get(
                                "tracking",
                                "handoff_expiry_seconds",
                                default=30.0,
                            )
                        ),
                    )
                    immediate_tracklet_payloads.extend(
                        self.pending_handoff_buffer.add(payload)
                    )
                else:
                    immediate_tracklet_payloads.append(payload)
            elif message["kind"] == "RECORDING":
                self.artifact_repo.add_recording(
                    recording_id=message.get("recording_id"),
                    source_id=message["source_id"],
                    session_id=message.get("session_id"),
                    camera_role=message.get("camera_role"),
                    stream_epoch=message.get("stream_epoch"),
                    segment_index=message.get("segment_index"),
                    segment_path=message["segment_path"],
                    started_at=message["started_at"],
                    ended_at=message["ended_at"],
                    frame_count=message["frame_count"],
                    size_bytes=message["size_bytes"],
                    fps=message.get("fps"),
                    codec=message.get("codec"),
                    sha256=message.get("sha256"),
                    corrupted=bool(message.get("corrupted", False)),
                    immutable=bool(message.get("immutable", True)),
                    metadata=message.get("metadata"),
                    frames=message.get("frames"),
                )
            elif message["kind"] == "MEDIA_COVERAGE":
                self.artifact_repo.upsert_media_coverage(
                    session_id=message["session_id"],
                    source_id=message["source_id"],
                    archive_required=bool(
                        message.get("archive_required")
                    ),
                    frames_acquired=int(
                        message.get("frames_acquired") or 0
                    ),
                    frames_processed=int(
                        message.get("frames_processed") or 0
                    ),
                    frames_archived=int(
                        message.get("frames_archived") or 0
                    ),
                    segments_produced=int(
                        message.get("segments_produced") or 0
                    ),
                    segments_corrupted=int(
                        message.get("segments_corrupted") or 0
                    ),
                    bytes_used=int(message.get("bytes_used") or 0),
                    details=dict(message.get("details") or {}),
                )

    def drain_inference_results(self) -> None:
        self._cleanup_expired_inference_results()
        while True:
            try:
                message = self.inference_result_queue.get_nowait()
            except queue.Empty:
                return
            if message["kind"] == "MODEL_READY":
                model_id = str(message["model_id"])
                if hasattr(self, "model_load_counts"):
                    self.model_load_counts[model_id] = int(
                        message.get("load_count") or 0
                    )
                self.inference_result_store[
                    f"__model_ready__:{model_id}"
                ] = message
                self.inference_result_store["__model_ready__"] = message
                if hasattr(self, "loaded_model_ids"):
                    self.loaded_model_ids.update(
                        str(item)
                        for item in message.get(
                            "loaded_model_ids", [model_id]
                        )
                    )
                continue
            if message["kind"] == "MODEL_LOAD_FAILED":
                model_id = str(message["model_id"])
                self.inference_result_store[
                    f"__model_load_failed__:{model_id}"
                ] = message
                self.inference_result_store["__model_load_failed__"] = message
                continue
            if message["kind"] == "MODEL_UNLOADED":
                model_id = str(message["model_id"])
                if hasattr(self, "loaded_model_ids"):
                    self.loaded_model_ids.discard(model_id)
                operation_id = str(message.get("operation_id") or "")
                if operation_id:
                    self.inference_result_store[
                        f"__model_unloaded__:{operation_id}"
                    ] = message
                continue
            if message["kind"] == "MODEL_UNLOAD_DEFERRED":
                operation_id = str(message.get("operation_id") or "")
                if operation_id:
                    self.inference_result_store[
                        f"__model_unload_deferred__:{operation_id}"
                    ] = message
                continue
            if message["kind"] == "INFERENCE_RUNTIME_STATUS":
                operation_id = str(message.get("operation_id") or "")
                self.inference_runtime_snapshot = dict(message)
                if operation_id:
                    self.inference_result_store[
                        f"__inference_runtime_status__:{operation_id}"
                    ] = message
                continue
            if message["kind"] in {
                "MODEL_VALIDATED",
                "MODEL_VALIDATION_FAILED",
            }:
                operation_id = str(message.get("operation_id") or "")
                if operation_id:
                    self.inference_result_store[
                        f"__model_validation__:{operation_id}"
                    ] = message
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
            task = str(message.get("task") or "")
            if task:
                if not hasattr(self, "last_inference_by_task"):
                    self.last_inference_by_task = {}
                self.last_inference_by_task[task] = {
                    "request_id": request_id,
                    "model_id": str(message.get("model_id") or ""),
                    "task": task,
                    "pipeline_role": message.get("pipeline_role"),
                    "routing_generation": int(
                        message.get("routing_generation") or 0
                    ),
                    "session_id": message.get("session_id"),
                    "source_id": source_id,
                    "stream_epoch": epoch,
                    "frame_index": int(
                        message.get("frame_index") or 0
                    ),
                    "kind": message["kind"],
                    "stored_at": stored["stored_at"],
                }
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
                self.active_source_sessions.pop(source_id, None)
                self.active_source_pipelines.pop(source_id, None)
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
                self.publish_runtime_model_state()
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
