from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from visionsort.core.enums import ModelStatus, PipelineState
from visionsort.core.paths import LOGS_DIR, MODELS_DIR, REPORTS_DIR, ROOT_DIR
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import ArtifactRepository

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - dépend de l'environnement
    YOLO = None


def _json_dict(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _build_comparison(db: VisionSortDB, metrics: dict[str, Any]) -> dict[str, Any]:
    active = db.fetch_one("SELECT id, metrics_json FROM model_registry WHERE is_active = 1 ORDER BY updated_at DESC LIMIT 1")
    if active is None:
        return {"against_model_id": None, "deltas": {}, "status": "no_active_baseline"}
    baseline = _json_dict(active["metrics_json"])
    deltas = {
        key: float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0))
        for key in ("precision", "recall", "mAP50", "mAP50_95", "fps", "pickup_precision", "drop_recall")
        if key in metrics or key in baseline
    }
    return {"against_model_id": active["id"], "deltas": deltas, "status": "compared"}


def _report_path_for(job_id: str, session_id: str | None) -> Path:
    report_dir = REPORTS_DIR / (session_id or "training")
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir / f"{int(time.time())}_TRAINING_{job_id[:8]}.json"


def _write_report(report_path: Path, payload: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def training_worker_loop(db_path: str, job_id: str, recipe: dict[str, Any], demo_mode: bool) -> None:
    db = VisionSortDB(Path(db_path))
    repo = ArtifactRepository(db)
    row = db.fetch_one("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
    if row is None:
        return
    log_path = ROOT_DIR / row["log_path"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model_row = db.fetch_one("SELECT * FROM model_registry WHERE id = ?", (row["model_id"],))
    dataset_row = db.fetch_one("SELECT * FROM datasets WHERE id = ?", (row["dataset_id"],))
    if model_row is None or dataset_row is None:
        repo.update_training_job(job_id, "FAILED", error_text="Dataset ou modèle introuvable.")
        return
    dataset_summary = json.loads(dataset_row["summary_json"] or "{}")
    session_id = dataset_summary.get("session_id")
    report_path = _report_path_for(job_id, session_id)
    report_rel = str(report_path.relative_to(ROOT_DIR))
    started_at = time.time()

    def write_log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    write_log("Démarrage entraînement VisionSort")
    write_log(json.dumps({"recipe": recipe, "validated_on_site": False}, ensure_ascii=True))
    try:
        if recipe.get("mode") == "demo":
            if not demo_mode:
                raise RuntimeError("Mode demo explicitement demandé, mais DEMO_MODE est inactif.")
            for step in range(1, 6):
                time.sleep(0.3)
                write_log(f"[demo] epoch={step}/5 loss={1.0 / step:.4f}")
            metrics = {
                "precision": 0.88,
                "recall": 0.84,
                "mAP50": 0.89,
                "mAP50_95": 0.61,
                "fps": 23.5,
                "pickup_precision": 0.80,
                "drop_recall": 0.78,
                "validated_on_site": False,
                "mode": "demo",
            }
            evaluation_metrics = {
                **metrics,
                "evaluation_status": PipelineState.EVALUATED.value,
                "benchmark": {"status": "COMPLETED", "mode": "demo_replay_proxy", "fps": metrics["fps"], "validated_on_site": False},
                "comparison": _build_comparison(db, metrics),
                "report_path": report_rel,
                "parent_model_id": model_row["id"],
            }
            repo.update_training_job(job_id, "EVALUATED", metrics=evaluation_metrics)
            if session_id:
                db.execute(
                    "UPDATE capture_sessions SET pipeline_state = ?, last_training_job_id = ?, report_path = ?, updated_at = ? WHERE id = ?",
                    (PipelineState.EVALUATED.value, job_id, report_rel, utc_now(), session_id),
                )
            write_log(json.dumps({"evaluation": evaluation_metrics}, ensure_ascii=True))
            candidate_id = f"candidate-{job_id[:8]}"
            now = time.time()
            db.execute(
                """
                INSERT OR REPLACE INTO model_registry
                (id, name, task, backend, weights_path, status, is_active, notes_json, metrics_json, parent_model_id, created_from_job_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    f"Candidate {candidate_id}",
                    model_row["task"],
                    "demo",
                    "",
                    ModelStatus.CANDIDATE.value,
                    json.dumps({"validated_on_site": False, "demo_only": True, "source_training_job": job_id, "report_path": report_rel}),
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
                "recipe": recipe,
                "evaluation": evaluation_metrics,
                "outputs": {"candidate_model_id": candidate_id, "weights_path": "", "mode": "demo"},
            }
            _write_report(report_path, report)
            repo.update_training_job(job_id, "COMPLETED", metrics=final_metrics)
            if session_id:
                db.execute(
                    "UPDATE capture_sessions SET pipeline_state = ?, last_training_job_id = ?, last_candidate_model_id = ?, report_path = ?, updated_at = ? WHERE id = ?",
                    (PipelineState.CANDIDATE.value, job_id, candidate_id, report_rel, utc_now(), session_id),
                )
            write_log(json.dumps(final_metrics, ensure_ascii=True))
            return

        if YOLO is None:
            raise RuntimeError("Ultralytics indisponible pour un entraînement réel.")
        yolo_dir = ROOT_DIR / "data" / "ultralytics"
        yolo_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_dir))
        data_yaml = ROOT_DIR / dataset_row["data_yaml_path"]
        model = YOLO(model_row["weights_path"])
        retries = 0
        last_exc: Exception | None = None
        while True:
            try:
                results = model.train(
                    data=str(data_yaml),
                    imgsz=int(recipe.get("imgsz", 640)),
                    epochs=int(recipe.get("epochs", 5)),
                    batch=int(recipe.get("batch", 4)),
                    device=recipe.get("device", "cpu"),
                    patience=int(recipe.get("patience", 10)),
                    project=str(MODELS_DIR / "runs"),
                    name=job_id,
                    exist_ok=True,
                    verbose=False,
                )
                last_exc = None
                break
            except Exception as exc:  # pragma: no cover
                last_exc = exc
                msg = str(exc).lower()
                if "out of memory" in msg and retries < 2:
                    retries += 1
                    recipe["batch"] = 1
                    recipe["imgsz"] = min(int(recipe.get("imgsz", 640)), 512)
                    write_log(f"Retry CUDA OOM {retries}/2 avec batch={recipe['batch']} imgsz={recipe['imgsz']}")
                    continue
                raise

        weights = MODELS_DIR / "runs" / job_id / "weights" / "best.pt"
        weights_path = str(weights.relative_to(ROOT_DIR)) if weights.exists() else model_row["weights_path"]
        metrics = {
            "validated_on_site": False,
            "mode": "ultralytics",
            "retries": retries,
            "weights_path": weights_path,
        }
        if results is not None and hasattr(results, "results_dict"):
            metrics["precision"] = float(results.results_dict.get("metrics/precision(B)", 0.0))
            metrics["recall"] = float(results.results_dict.get("metrics/recall(B)", 0.0))
            metrics["mAP50"] = float(results.results_dict.get("metrics/mAP50(B)", 0.0))
            metrics["mAP50_95"] = float(results.results_dict.get("metrics/mAP50-95(B)", 0.0))
        evaluation_metrics = {
            **metrics,
            "evaluation_status": PipelineState.EVALUATED.value,
            "benchmark": {"status": "SKIPPED", "reason": "Benchmark vidéo non exécuté dans cet environnement."},
            "comparison": _build_comparison(db, metrics),
            "report_path": report_rel,
            "parent_model_id": model_row["id"],
        }
        repo.update_training_job(job_id, "EVALUATED", metrics=evaluation_metrics)
        if session_id:
            db.execute(
                "UPDATE capture_sessions SET pipeline_state = ?, last_training_job_id = ?, report_path = ?, updated_at = ? WHERE id = ?",
                (PipelineState.EVALUATED.value, job_id, report_rel, utc_now(), session_id),
            )
        write_log(json.dumps({"evaluation": evaluation_metrics}, ensure_ascii=True))

        candidate_id = f"candidate-{job_id[:8]}"
        now = time.time()
        db.execute(
            """
            INSERT OR REPLACE INTO model_registry
            (id, name, task, backend, weights_path, status, is_active, notes_json, metrics_json, parent_model_id, created_from_job_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                f"Candidate {candidate_id}",
                model_row["task"],
                "ultralytics",
                weights_path,
                ModelStatus.CANDIDATE.value,
                json.dumps({"validated_on_site": False, "source_training_job": job_id, "report_path": report_rel}),
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
            "recipe": recipe,
            "evaluation": evaluation_metrics,
            "outputs": {"candidate_model_id": candidate_id, "weights_path": weights_path, "mode": "ultralytics"},
        }
        _write_report(report_path, report)
        repo.update_training_job(job_id, "COMPLETED", metrics=final_metrics)
        if session_id:
            db.execute(
                "UPDATE capture_sessions SET pipeline_state = ?, last_training_job_id = ?, last_candidate_model_id = ?, report_path = ?, updated_at = ? WHERE id = ?",
                (PipelineState.CANDIDATE.value, job_id, candidate_id, report_rel, utc_now(), session_id),
            )
        write_log(json.dumps(final_metrics, ensure_ascii=True))
    except Exception as exc:  # pragma: no cover - runtime
        message = str(exc)
        if "out of memory" in message.lower():
            message = f"CUDA OOM: {message}"
        repo.update_training_job(job_id, "FAILED", error_text=message)
        if session_id:
            db.execute(
                "UPDATE capture_sessions SET pipeline_state = ?, last_training_job_id = ?, report_path = ?, updated_at = ? WHERE id = ?",
                (PipelineState.REJECTED.value, job_id, report_rel, utc_now(), session_id),
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
                "recipe": recipe,
                "error": message,
            },
        )
        write_log(message)


def create_training_job(db: VisionSortDB, dataset_id: str, model_id: str, recipe: dict[str, Any]) -> str:
    repo = ArtifactRepository(db)
    log_path = LOGS_DIR / f"training_{int(time.time())}.log"
    return repo.add_training_job(dataset_id=dataset_id, model_id=model_id, status="QUEUED", recipe=recipe, log_path=str(log_path.relative_to(ROOT_DIR)))
