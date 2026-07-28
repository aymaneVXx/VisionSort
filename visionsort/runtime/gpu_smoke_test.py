from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from visionsort.core.config import AppConfig, DEFAULT_CONFIG
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.runtime.supervisor import RuntimeSupervisor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _register_real_model(
    db: VisionSortDB,
    *,
    model_id: str,
    task: str,
    weights_path: Path,
    active: bool,
) -> None:
    path = weights_path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Poids locaux introuvables: {path}")
    now = utc_now()
    db.execute(
        """
        INSERT INTO model_registry
        (id, name, task, backend, weights_path, status, is_active,
         notes_json, metrics_json, parent_model_id, created_from_job_id,
         created_at, updated_at)
        VALUES (?, ?, ?, 'ultralytics', ?, 'CHAMPION', ?, ?, '{}',
                NULL, NULL, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            task = excluded.task,
            backend = excluded.backend,
            weights_path = excluded.weights_path,
            status = excluded.status,
            is_active = excluded.is_active,
            notes_json = excluded.notes_json,
            updated_at = excluded.updated_at
        """,
        (
            model_id,
            model_id,
            task,
            str(path),
            int(active),
            json.dumps(
                {
                    "artifact_sha256": _sha256(path),
                    "gpu_smoke_test": True,
                    "validated_on_site": False,
                }
            ),
            now,
            now,
        ),
    )


def _infer_once(
    supervisor: RuntimeSupervisor,
    *,
    model_id: str,
    task: str,
    frame_index: int,
    image: np.ndarray,
    timeout: float = 30.0,
) -> dict[str, Any]:
    request_id = str(uuid.uuid4())
    source_id = "gpu-smoke-source"
    session_id = "gpu-smoke-session"
    supervisor.active_source_sessions[source_id] = session_id
    supervisor.latest_stream_epoch_by_source[source_id] = 0
    route = supervisor.runtime_route(task)
    created_at = time.time()
    supervisor.inference_request_queue.put(
        {
            "kind": "INFER",
            "request_id": request_id,
            "created_at": created_at,
            "expires_at": created_at + timeout,
            "session_id": session_id,
            "source_id": source_id,
            "camera_id": source_id,
            "camera_role": "GPU_SMOKE",
            "stream_epoch": 0,
            "frame_index": int(frame_index),
            "timestamp_local": float(frame_index),
            "timestamp_global": time.time(),
            "model_id": model_id,
            "task": task,
            "pipeline_role": (
                "operator_pose"
                if task == "pose"
                else "parcel_detection"
            ),
            "routing_generation": int(route.get("generation") or 0),
            "image": image,
        }
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        supervisor.drain_inference_results()
        result = supervisor.inference_result_store.pop(
            request_id, None
        )
        if result:
            if result.get("kind") == "INFER_ERROR":
                raise RuntimeError(str(result.get("error")))
            if str(result.get("model_id")) != model_id:
                raise RuntimeError(
                    "Provenance invalide: "
                    f"{result.get('model_id')} au lieu de {model_id}."
                )
            return dict(result)
        time.sleep(0.01)
    raise TimeoutError(f"Inférence expirée pour {model_id}.")


def _memory(supervisor: RuntimeSupervisor) -> dict[str, Any]:
    return dict(
        supervisor.request_inference_runtime_status(
            timeout=5.0
        ).get("cuda_memory")
        or {}
    )


def run_gpu_smoke_test(
    *,
    db_path: Path,
    parcel_v1: Path,
    parcel_v2: Path,
    pose: Path,
    parcel_task: str,
    iterations: int,
    image_size: int,
    memory_growth_limit_mb: float,
    report_path: Path | None,
) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch avec CUDA est requis pour ce smoke test."
        ) from exc
    if not torch.cuda.is_available():
        skipped = {
            "status": "SKIPPED",
            "reason": "CUDA indisponible",
            "validated_on_site": False,
        }
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(skipped, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return skipped
    if parcel_task not in {"detection", "segmentation"}:
        raise RuntimeError(
            "--parcel-task doit valoir detection ou segmentation."
        )
    db = VisionSortDB(db_path)
    db.initialize()
    parcel_v1_id = f"gpu-smoke-{parcel_task}-v1"
    parcel_v2_id = f"gpu-smoke-{parcel_task}-v2"
    pose_id = "gpu-smoke-pose-v1"
    db.execute(
        "UPDATE model_registry SET is_active = 0 WHERE task IN (?, 'pose')",
        (parcel_task,),
    )
    _register_real_model(
        db,
        model_id=parcel_v1_id,
        task=parcel_task,
        weights_path=parcel_v1,
        active=True,
    )
    _register_real_model(
        db,
        model_id=parcel_v2_id,
        task=parcel_task,
        weights_path=parcel_v2,
        active=False,
    )
    _register_real_model(
        db,
        model_id=pose_id,
        task="pose",
        weights_path=pose,
        active=True,
    )
    backends = db.fetch_all(
        """
        SELECT id, backend FROM model_registry
        WHERE id IN (?, ?, ?)
        """,
        (parcel_v1_id, parcel_v2_id, pose_id),
    )
    if len(backends) != 3 or any(
        str(row["backend"]) == "demo" for row in backends
    ):
        raise RuntimeError(
            "Smoke test refusé: aucun backend demo n'est autorisé."
        )

    config_values = copy.deepcopy(DEFAULT_CONFIG)
    config_values["app"]["demo_mode"] = False
    config_values["gpu"]["device"] = "0"
    config_values["runtime"]["max_inference_queue"] = 16
    config_values["runtime"]["model_unload_timeout_seconds"] = 20.0
    config_values["runtime"][
        "model_switch_validation_timeout_seconds"
    ] = 30.0
    supervisor = RuntimeSupervisor(
        db_path=db_path,
        config=AppConfig(values=config_values),
    )
    started_at = time.time()
    image = np.zeros(
        (int(image_size), int(image_size), 3), dtype=np.uint8
    )
    try:
        supervisor.start()
        warmup_v1 = _infer_once(
            supervisor,
            model_id=parcel_v1_id,
            task=parcel_task,
            frame_index=0,
            image=image,
        )
        warmup_pose = _infer_once(
            supervisor,
            model_id=pose_id,
            task="pose",
            frame_index=1,
            image=image,
        )
        memory_before_activation = _memory(supervisor)

        supervisor.active_source_pipelines["gpu-smoke-fixed"] = [
            {
                "pipeline_role": "parcel_detection",
                "task": parcel_task,
                "configured_model_id": parcel_v1_id,
                "use_active": False,
            }
        ]
        switch = supervisor.switch_runtime_model(
            parcel_v2_id,
            actor="gpu-smoke-test",
            reason="bascule locale CUDA v1 vers v2",
        )
        memory_after_activation = _memory(supervisor)
        if parcel_v1_id not in supervisor.loaded_model_ids:
            raise RuntimeError(
                "Le modèle v1 devait rester chargé pendant sa référence fixe."
            )
        if str(
            (switch.get("unload_previous") or {}).get("status")
        ) != "REFERENCED":
            raise RuntimeError(
                "La protection du modèle encore référencé n'a pas été vérifiée."
            )

        supervisor.active_source_pipelines.pop(
            "gpu-smoke-fixed", None
        )
        unload = supervisor.safe_unload_model(
            parcel_v1_id, timeout=20.0
        )
        if not unload.get("unloaded"):
            raise RuntimeError(
                f"Déchargement v1 non confirmé: {unload}"
            )
        memory_after_unload = _memory(supervisor)

        _infer_once(
            supervisor,
            model_id=parcel_v2_id,
            task=parcel_task,
            frame_index=2,
            image=image,
        )
        _infer_once(
            supervisor,
            model_id=pose_id,
            task="pose",
            frame_index=3,
            image=image,
        )
        benchmark_started = time.perf_counter()
        inference_seconds: list[float] = []
        memory_samples: list[dict[str, Any]] = []
        for index in range(max(1, int(iterations))):
            result = _infer_once(
                supervisor,
                model_id=parcel_v2_id,
                task=parcel_task,
                frame_index=10 + index,
                image=image,
            )
            inference_seconds.append(
                float(result.get("duration_seconds") or 0.0)
            )
            if index in {
                0,
                max(0, int(iterations) // 2),
                max(0, int(iterations) - 1),
            }:
                memory_samples.append(_memory(supervisor))
        elapsed = time.perf_counter() - benchmark_started
        fps = max(1, int(iterations)) / max(elapsed, 1e-9)
        allocated_samples = [
            int(sample.get("allocated_bytes") or 0)
            for sample in memory_samples
        ]
        growth_bytes = (
            max(allocated_samples[1:] or allocated_samples)
            - allocated_samples[0]
            if allocated_samples
            else 0
        )
        growth_limit_bytes = int(
            float(memory_growth_limit_mb) * 1024 * 1024
        )
        if growth_bytes > growth_limit_bytes:
            raise RuntimeError(
                "Croissance mémoire CUDA continue détectée: "
                f"{growth_bytes / 1024 / 1024:.1f} MiB."
            )
        report = {
            "status": "COMPLETED",
            "device": torch.cuda.get_device_name(0),
            "parcel_task": parcel_task,
            "models": {
                "parcel_v1": parcel_v1_id,
                "parcel_v2": parcel_v2_id,
                "pose": pose_id,
            },
            "real_inference": {
                "parcel_v1_model_id": warmup_v1.get("model_id"),
                "pose_model_id": warmup_pose.get("model_id"),
                "iterations": max(1, int(iterations)),
                "fps": fps,
                "mean_inference_seconds": (
                    sum(inference_seconds)
                    / max(len(inference_seconds), 1)
                ),
            },
            "memory": {
                "before_activation": memory_before_activation,
                "after_activation": memory_after_activation,
                "after_unload": memory_after_unload,
                "benchmark_samples": memory_samples,
                "growth_bytes": growth_bytes,
                "growth_limit_bytes": growth_limit_bytes,
            },
            "switch": switch,
            "unload": unload,
            "loaded_model_ids": sorted(
                supervisor.loaded_model_ids
            ),
            "duration_seconds": time.time() - started_at,
            "validated_on_site": False,
            "site_validation_status": "NON_VALIDÉ_SUR_SITE",
        }
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return report
    finally:
        supervisor.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test CUDA VisionSort avec poids Ultralytics locaux; "
            "le backend demo est interdit."
        )
    )
    parser.add_argument("--parcel-v1", type=Path, required=True)
    parser.add_argument("--parcel-v2", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument(
        "--parcel-task",
        choices=("detection", "segmentation"),
        default="detection",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/runtime/gpu-smoke.db"),
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--memory-growth-limit-mb", type=float, default=128.0
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/runtime/reports/gpu-smoke.json"),
    )
    args = parser.parse_args()
    report = run_gpu_smoke_test(
        db_path=args.db,
        parcel_v1=args.parcel_v1,
        parcel_v2=args.parcel_v2,
        pose=args.pose,
        parcel_task=args.parcel_task,
        iterations=args.iterations,
        image_size=args.image_size,
        memory_growth_limit_mb=args.memory_growth_limit_mb,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
