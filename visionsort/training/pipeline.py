from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2

from visionsort.core.enums import ModelStatus, PipelineState
from visionsort.core.paths import LOGS_DIR, MODELS_DIR, REPORTS_DIR, ROOT_DIR
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import ArtifactRepository
from visionsort.datasets.pipeline import (
    validate_dataset_splits,
    verify_dataset_fingerprint,
)
from visionsort.inference.engine import DemoDetectionBackend
from visionsort.runtime.demo_assets import ensure_demo_assets

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - depends on environment
    YOLO = None


DEFAULT_PROMOTION_CRITERIA = {
    "precision_min": 0.50,
    "recall_min": 0.50,
    "map50_min": 0.50,
    "count_accuracy_min": 0.80,
    "merge_rate_max": 0.15,
    "fps_min": 5.0,
}


def _json_dict(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _build_comparison(db: VisionSortDB, metrics: dict[str, Any]) -> dict[str, Any]:
    active = db.fetch_one(
        "SELECT id, metrics_json FROM model_registry WHERE is_active = 1 ORDER BY updated_at DESC LIMIT 1"
    )
    if active is None:
        return {
            "against_model_id": None,
            "deltas": {},
            "status": "no_active_baseline",
        }
    baseline = _json_dict(active["metrics_json"])
    keys = (
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
        "count_accuracy",
        "merge_rate",
        "fps",
    )
    deltas = {
        key: float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0))
        for key in keys
        if key in metrics or key in baseline
    }
    return {
        "against_model_id": active["id"],
        "deltas": deltas,
        "status": "compared",
    }


def _report_path_for(job_id: str, session_id: str | None) -> Path:
    report_dir = REPORTS_DIR / (session_id or "training")
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"{int(time.time())}_TRAINING_{job_id[:8]}.json"


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_immutable_artifact(
    *,
    candidate_id: str,
    source_path: Path | None,
    demo_payload: dict[str, Any] | None = None,
) -> tuple[str, str]:
    version_dir = MODELS_DIR / "versions" / candidate_id
    if version_dir.exists():
        raise RuntimeError(f"Répertoire de version déjà présent: {version_dir}")
    version_dir.mkdir(parents=True, exist_ok=False)
    destination = version_dir / "best.pt"
    if source_path is not None:
        if not source_path.exists():
            raise RuntimeError(f"best.pt introuvable: {source_path}")
        shutil.copy2(source_path, destination)
    else:
        destination.write_bytes(
            json.dumps(
                {
                    **(demo_payload or {}),
                    "simulated_checkpoint": True,
                    "format": "visionsort-demo-checkpoint",
                },
                sort_keys=True,
            ).encode("utf-8")
        )
    return str(destination.relative_to(ROOT_DIR)), _sha256(destination)


def _metrics_from_results(results: Any) -> dict[str, float]:
    values = getattr(results, "results_dict", {}) or {}
    return {
        "precision": float(values.get("metrics/precision(B)", 0.0)),
        "recall": float(values.get("metrics/recall(B)", 0.0)),
        "mAP50": float(values.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(values.get("metrics/mAP50-95(B)", 0.0)),
    }


def _cleanup_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _demo_replay_benchmark() -> dict[str, Any]:
    assets = ensure_demo_assets()
    backend = DemoDetectionBackend()
    sidecar_hash = hashlib.sha256()
    total_frames = 0
    exact_counts = 0
    started = time.perf_counter()
    for camera_id, video_path in sorted(assets.items()):
        backend.register_sidecar(camera_id, video_path)
        sidecar = Path(video_path).with_suffix(".jsonl")
        sidecar_hash.update(sidecar.read_bytes())
        capture = cv2.VideoCapture(video_path)
        frame_index = 0
        while frame_index < 24:
            ok, image = capture.read()
            if not ok:
                break
            expected_count = len(
                backend.sidecars.get(camera_id, {}).get(frame_index, [])
            )
            predicted_count = len(backend.predict(camera_id, frame_index, image))
            exact_counts += int(predicted_count == expected_count)
            total_frames += 1
            frame_index += 1
        capture.release()
    duration = max(time.perf_counter() - started, 1e-9)
    return {
        "status": "COMPLETED",
        "mode": "demo_replay_proxy",
        "frames": total_frames,
        "fps": total_frames / duration,
        "count_accuracy": exact_counts / max(total_frames, 1),
        "merge_rate": 0.0,
        "frozen": True,
        "fixture_sha256": sidecar_hash.hexdigest(),
        "validated_on_site": False,
        "simulated_backend": True,
    }


def _session_replay_paths(
    db: VisionSortDB, session_id: str | None
) -> list[str]:
    if not session_id:
        return []
    rows = db.fetch_all(
        """
        SELECT s.uri
        FROM capture_session_sources css
        JOIN sources s ON s.id = css.source_id
        WHERE css.session_id = ? AND s.source_type = 'REPLAY'
        ORDER BY css.camera_role
        """,
        (session_id,),
    )
    return [str(row["uri"]) for row in rows]


def _benchmark_replays(
    model: Any, replay_paths: list[str], *, max_frames_per_video: int = 60
) -> dict[str, Any]:
    total_frames = 0
    evaluated_frames = 0
    exact_counts = 0
    merge_frames = 0
    fingerprint = hashlib.sha256()
    started = time.perf_counter()
    for replay_path in replay_paths:
        path = Path(replay_path)
        if path.exists():
            fingerprint.update(path.name.encode("utf-8"))
            fingerprint.update(str(path.stat().st_size).encode("ascii"))
        ground_truth: dict[int, int] = {}
        sidecar = path.with_suffix(".jsonl")
        if sidecar.exists():
            fingerprint.update(sidecar.read_bytes())
            for line in sidecar.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    frame = json.loads(line)
                    index = int(frame["frame_index"])
                    ground_truth[index] = ground_truth.get(index, 0) + 1
        capture = cv2.VideoCapture(str(path))
        frame_index = 0
        while frame_index < max_frames_per_video:
            ok, image = capture.read()
            if not ok:
                break
            results = model.predict(image, verbose=False)
            predicted = (
                len(getattr(results[0], "boxes", [])) if results else 0
            )
            expected = ground_truth.get(frame_index)
            if expected is not None:
                evaluated_frames += 1
                exact_counts += int(predicted == expected)
                merge_frames += int(predicted < expected)
            total_frames += 1
            frame_index += 1
        capture.release()
    duration = max(time.perf_counter() - started, 1e-9)
    return {
        "status": "COMPLETED" if replay_paths else "SKIPPED",
        "reason": None if replay_paths else "Aucune vidéo Replay figée configurée.",
        "frames": total_frames,
        "fps": total_frames / duration if total_frames else 0.0,
        "count_accuracy": exact_counts / max(evaluated_frames, 1),
        "merge_rate": merge_frames / max(evaluated_frames, 1),
        "frozen": bool(replay_paths),
        "fixture_sha256": fingerprint.hexdigest() if replay_paths else None,
        "validated_on_site": False,
    }


def _promotion_eligible(
    metrics: dict[str, Any],
    *,
    criteria: dict[str, float],
    frozen_test: bool,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not frozen_test:
        failures.append("frozen_test_required")
    checks = {
        "precision_min": float(metrics.get("precision", 0.0)),
        "recall_min": float(metrics.get("recall", 0.0)),
        "map50_min": float(metrics.get("mAP50", 0.0)),
        "count_accuracy_min": float(metrics.get("count_accuracy", 0.0)),
        "fps_min": float(metrics.get("fps", 0.0)),
    }
    for criterion, actual in checks.items():
        if actual < float(criteria[criterion]):
            failures.append(f"{criterion}:{actual:.6f}<{criteria[criterion]:.6f}")
    merge_rate = float(metrics.get("merge_rate", 1.0))
    if merge_rate > float(criteria["merge_rate_max"]):
        failures.append(
            f"merge_rate_max:{merge_rate:.6f}>{criteria['merge_rate_max']:.6f}"
        )
    return not failures, failures


def _resolve_best_pt(results: Any, job_id: str) -> Path:
    candidates: list[Path] = []
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        candidates.append(Path(save_dir) / "weights" / "best.pt")
    candidates.append(MODELS_DIR / "runs" / job_id / "weights" / "best.pt")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise RuntimeError(
        "Ultralytics a terminé sans produire de fichier weights/best.pt."
    )


def _validate_training_dataset(db: VisionSortDB, dataset_id: str) -> None:
    fingerprint = verify_dataset_fingerprint(db, dataset_id)
    if not fingerprint["valid"]:
        raise RuntimeError(
            "Entraînement refusé: le dataset_fingerprint ne correspond plus "
            "aux fichiers finalisés."
        )
    integrity = validate_dataset_splits(db, dataset_id)
    if not integrity["valid"]:
        raise RuntimeError(
            f"Entraînement refusé: fuite entre splits ({integrity['leaks']})."
        )
    real_sessions = db.fetch_one(
        """
        SELECT COUNT(*) AS count
        FROM dataset_sessions ds
        JOIN capture_sessions cs ON cs.id = ds.session_id
        WHERE ds.dataset_id = ? AND cs.demo_mode = 0
        """,
        (dataset_id,),
    )
    if int((real_sessions["count"] if real_sessions else 0) or 0) > 0 and not integrity[
        "all_splits_nonempty"
    ]:
        raise RuntimeError(
            "Entraînement réel refusé: train, val et test doivent être non vides."
        )


def training_worker_loop(
    db_path: str, job_id: str, recipe: dict[str, Any], demo_mode: bool
) -> None:
    db = VisionSortDB(Path(db_path))
    repo = ArtifactRepository(db)
    row = db.fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if row is None:
        return
    if row["status"] == "COMPLETED":
        return
    model_row = db.fetch_one(
        "SELECT * FROM model_registry WHERE id = ?", (row["model_id"],)
    )
    dataset_row = db.fetch_one(
        "SELECT * FROM datasets WHERE id = ?", (row["dataset_id"],)
    )
    if model_row is None or dataset_row is None:
        repo.update_training_job(
            job_id, "FAILED", error_text="Dataset ou modèle introuvable."
        )
        return
    if dataset_row["status"] != "DATASET_READY":
        repo.update_training_job(
            job_id,
            "FAILED",
            error_text="Entraînement refusé: le dataset n'est pas DATASET_READY.",
        )
        return
    try:
        _validate_training_dataset(db, str(row["dataset_id"]))
    except RuntimeError as exc:
        repo.update_training_job(job_id, "FAILED", error_text=str(exc))
        return

    effective_recipe = dict(recipe)
    dataset_summary = _json_dict(dataset_row["summary_json"])
    session_id = dataset_summary.get("session_id")
    report_path = _report_path_for(job_id, session_id)
    report_rel = str(report_path.relative_to(ROOT_DIR))
    log_path = ROOT_DIR / row["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    candidate_id = f"candidate-{job_id[:8]}"

    def write_log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    write_log("Démarrage entraînement VisionSort")
    write_log(
        json.dumps(
            {"recipe": effective_recipe, "validated_on_site": False},
            ensure_ascii=True,
        )
    )
    try:
        criteria = {
            **DEFAULT_PROMOTION_CRITERIA,
            **dict(effective_recipe.get("promotion_criteria") or {}),
        }
        retries = 0
        retry_history: list[dict[str, Any]] = []
        simulated = effective_recipe.get("mode") == "demo"
        if simulated:
            if not demo_mode:
                raise RuntimeError(
                    "Mode demo explicitement demandé, mais DEMO_MODE est inactif."
                )
            for step in range(1, 4):
                time.sleep(0.05)
                write_log(f"[demo] epoch={step}/3 loss={1.0 / step:.4f}")
            benchmark = _demo_replay_benchmark()
            validation_metrics = {
                "precision": 0.88,
                "recall": 0.84,
                "mAP50": 0.89,
                "mAP50_95": 0.61,
            }
            test_metrics = {
                **validation_metrics,
                "status": "COMPLETED",
                "frozen": True,
                "type": "frozen_demo_replay_proxy",
                "fixture_sha256": benchmark["fixture_sha256"],
                "simulated_backend": True,
            }
            weights_path, artifact_sha256 = _stage_immutable_artifact(
                candidate_id=candidate_id,
                source_path=None,
                demo_payload={
                    "job_id": job_id,
                    "dataset_id": row["dataset_id"],
                    "base_model_id": model_row["id"],
                },
            )
        else:
            if YOLO is None:
                raise RuntimeError(
                    "Ultralytics indisponible pour un entraînement réel."
                )
            integrity = dataset_summary.get("split_integrity") or {}
            if not integrity.get("test_frozen") or int(
                integrity.get("test_items") or 0
            ) <= 0:
                raise RuntimeError(
                    "Entraînement réel refusé: un jeu de test figé non vide est requis."
                )
            yolo_dir = ROOT_DIR / "data" / "ultralytics"
            yolo_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_dir))
            data_yaml = ROOT_DIR / dataset_row["data_yaml_path"]
            model = YOLO(model_row["weights_path"])
            while True:
                try:
                    train_results = model.train(
                        data=str(data_yaml),
                        imgsz=int(effective_recipe.get("imgsz", 640)),
                        epochs=int(effective_recipe.get("epochs", 5)),
                        batch=int(effective_recipe.get("batch", 4)),
                        device=effective_recipe.get("device", "cpu"),
                        patience=int(effective_recipe.get("patience", 10)),
                        project=str(MODELS_DIR / "runs"),
                        name=job_id,
                        exist_ok=True,
                        verbose=False,
                    )
                    break
                except Exception as exc:  # pragma: no cover - GPU dependent
                    if "out of memory" not in str(exc).lower() or retries >= 2:
                        raise
                    retries += 1
                    if retries == 1:
                        effective_recipe["batch"] = max(
                            1, int(effective_recipe.get("batch", 4)) // 2
                        )
                    else:
                        effective_recipe["imgsz"] = max(
                            320, int(effective_recipe.get("imgsz", 640) * 0.8)
                        )
                    retry_history.append(
                        {
                            "attempt": retries,
                            "batch": effective_recipe.get("batch"),
                            "imgsz": effective_recipe.get("imgsz"),
                        }
                    )
                    _cleanup_cuda()
                    write_log(
                        f"Retry CUDA OOM {retries}/2 avec "
                        f"batch={effective_recipe.get('batch')} "
                        f"imgsz={effective_recipe.get('imgsz')}"
                    )
            best_pt = _resolve_best_pt(train_results, job_id)
            weights_path, artifact_sha256 = _stage_immutable_artifact(
                candidate_id=candidate_id,
                source_path=best_pt,
            )
            evaluation_model = YOLO(str(ROOT_DIR / weights_path))
            validation_results = evaluation_model.val(
                data=str(data_yaml), split="val", verbose=False
            )
            test_results = evaluation_model.val(
                data=str(data_yaml), split="test", verbose=False
            )
            validation_metrics = _metrics_from_results(validation_results)
            test_metrics = {
                **_metrics_from_results(test_results),
                "status": "COMPLETED",
                "frozen": True,
                "type": "dataset_test_split",
                "fixture_sha256": integrity.get("frozen_test_sha256"),
                "simulated_backend": False,
            }
            benchmark = _benchmark_replays(
                evaluation_model,
                list(
                    effective_recipe.get("benchmark_replays")
                    or _session_replay_paths(db, session_id)
                ),
            )

        metrics = {
            **test_metrics,
            "count_accuracy": float(benchmark.get("count_accuracy", 0.0)),
            "merge_rate": float(benchmark.get("merge_rate", 1.0)),
            "fps": float(benchmark.get("fps", 0.0)),
            "validated_on_site": False,
            "mode": "demo" if simulated else "ultralytics",
            "retries": retries,
            "retry_history": retry_history,
            "weights_path": weights_path,
            "artifact_sha256": artifact_sha256,
        }
        promotion_eligible, promotion_failures = _promotion_eligible(
            metrics,
            criteria=criteria,
            frozen_test=bool(test_metrics.get("frozen")),
        )
        evaluation_metrics = {
            **metrics,
            "evaluation_status": PipelineState.EVALUATED.value,
            "validation": {
                **validation_metrics,
                "status": "COMPLETED",
            },
            "test": test_metrics,
            "benchmark": benchmark,
            "comparison": _build_comparison(db, metrics),
            "promotion_criteria": criteria,
            "promotion_eligible": promotion_eligible,
            "promotion_failures": promotion_failures,
            "report_path": report_rel,
            "parent_model_id": model_row["id"],
        }
        repo.update_training_job(job_id, "EVALUATED", metrics=evaluation_metrics)
        if session_id:
            db.execute(
                "UPDATE capture_sessions SET pipeline_state = ?, last_training_job_id = ?, report_path = ?, updated_at = ? WHERE id = ?",
                (
                    PipelineState.EVALUATED.value,
                    job_id,
                    report_rel,
                    utc_now(),
                    session_id,
                ),
            )
        now = time.time()
        db.execute(
            """
            INSERT INTO model_registry
            (id, name, task, backend, weights_path, status, is_active, notes_json,
             metrics_json, parent_model_id, created_from_job_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                f"Candidate {candidate_id}",
                model_row["task"],
                "demo" if simulated else "ultralytics",
                weights_path,
                ModelStatus.CANDIDATE.value,
                json.dumps(
                    {
                        "validated_on_site": False,
                        "demo_only": simulated,
                        "simulated_checkpoint": simulated,
                        "source_training_job": job_id,
                        "report_path": report_rel,
                        "artifact_sha256": artifact_sha256,
                        "immutable_artifact": True,
                    }
                ),
                json.dumps(evaluation_metrics),
                model_row["id"],
                job_id,
                now,
                now,
            ),
        )
        final_metrics = {
            **evaluation_metrics,
            "candidate_model_id": candidate_id,
            "candidate_status": PipelineState.CANDIDATE.value,
        }
        report = {
            "job_id": job_id,
            "session_id": session_id,
            "dataset_id": row["dataset_id"],
            "base_model_id": model_row["id"],
            "candidate_model_id": candidate_id,
            "status": "COMPLETED",
            "started_at": started_at,
            "ended_at": time.time(),
            "recipe": effective_recipe,
            "evaluation": evaluation_metrics,
            "outputs": {
                "candidate_model_id": candidate_id,
                "weights_path": weights_path,
                "artifact_sha256": artifact_sha256,
                "mode": "demo" if simulated else "ultralytics",
            },
        }
        _write_report(report_path, report)
        repo.update_training_job(job_id, "COMPLETED", metrics=final_metrics)
        if session_id:
            db.execute(
                "UPDATE capture_sessions SET pipeline_state = ?, last_training_job_id = ?, last_candidate_model_id = ?, report_path = ?, updated_at = ? WHERE id = ?",
                (
                    PipelineState.CANDIDATE.value,
                    job_id,
                    candidate_id,
                    report_rel,
                    utc_now(),
                    session_id,
                ),
            )
        write_log(json.dumps(final_metrics, ensure_ascii=True))
    except Exception as exc:  # pragma: no cover - runtime failures are reported
        _cleanup_cuda()
        message = str(exc)
        if "out of memory" in message.lower():
            message = f"CUDA OOM: {message}"
        repo.update_training_job(job_id, "FAILED", error_text=message)
        if session_id:
            db.execute(
                "UPDATE capture_sessions SET pipeline_state = ?, last_training_job_id = ?, report_path = ?, updated_at = ? WHERE id = ?",
                (
                    PipelineState.REJECTED.value,
                    job_id,
                    report_rel,
                    utc_now(),
                    session_id,
                ),
            )
        _write_report(
            report_path,
            {
                "job_id": job_id,
                "session_id": session_id,
                "dataset_id": row["dataset_id"],
                "base_model_id": model_row["id"],
                "status": "FAILED",
                "started_at": started_at,
                "ended_at": time.time(),
                "recipe": effective_recipe,
                "error": message,
            },
        )
        write_log(message)


def create_training_job(
    db: VisionSortDB, dataset_id: str, model_id: str, recipe: dict[str, Any]
) -> str:
    dataset = db.fetch_one("SELECT status FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        raise RuntimeError("Dataset introuvable.")
    if dataset["status"] != "DATASET_READY":
        raise RuntimeError(
            "Entraînement refusé: le dataset doit être explicitement DATASET_READY."
        )
    _validate_training_dataset(db, dataset_id)
    model = db.fetch_one("SELECT id FROM model_registry WHERE id = ?", (model_id,))
    if model is None:
        raise RuntimeError("Modèle initial introuvable.")
    repo = ArtifactRepository(db)
    log_path = LOGS_DIR / f"training_{int(time.time())}.log"
    return repo.add_training_job(
        dataset_id=dataset_id,
        model_id=model_id,
        status="QUEUED",
        recipe=recipe,
        log_path=str(log_path.relative_to(ROOT_DIR)),
    )
