from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from visionsort.core.paths import LOGS_DIR, ROOT_DIR
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import ArtifactRepository

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - dépend de l'environnement
    YOLO = None


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

    def write_log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    write_log("Démarrage entraînement VisionSort")
    write_log(json.dumps({"recipe": recipe, "validated_on_site": False}, ensure_ascii=True))
    try:
        if model_row["backend"] == "demo" or recipe.get("mode") == "demo":
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
            repo.update_training_job(job_id, "COMPLETED", metrics=metrics)
            write_log(json.dumps(metrics, ensure_ascii=True))
            return

        if YOLO is None:
            raise RuntimeError("Ultralytics indisponible pour un entraînement réel.")
        data_yaml = ROOT_DIR / dataset_row["data_yaml_path"]
        model = YOLO(model_row["weights_path"])
        results = model.train(
            data=str(data_yaml),
            imgsz=int(recipe.get("imgsz", 640)),
            epochs=int(recipe.get("epochs", 5)),
            batch=int(recipe.get("batch", 4)),
            device=recipe.get("device", "cpu"),
            patience=int(recipe.get("patience", 10)),
            project=str(ROOT_DIR / "data" / "models" / "runs"),
            name=job_id,
            exist_ok=True,
            verbose=False,
        )
        metrics = {
            "precision": float(getattr(results.results_dict, "get", lambda *_: 0.0)("metrics/precision(B)", 0.0))
            if hasattr(results, "results_dict")
            else 0.0,
            "recall": float(getattr(results.results_dict, "get", lambda *_: 0.0)("metrics/recall(B)", 0.0))
            if hasattr(results, "results_dict")
            else 0.0,
            "validated_on_site": False,
            "mode": "ultralytics",
        }
        repo.update_training_job(job_id, "COMPLETED", metrics=metrics)
        write_log(json.dumps(metrics, ensure_ascii=True))
    except Exception as exc:  # pragma: no cover - runtime
        message = str(exc)
        if "out of memory" in message.lower():
            message = f"CUDA OOM: {message}"
        repo.update_training_job(job_id, "FAILED", error_text=message)
        write_log(message)


def create_training_job(db: VisionSortDB, dataset_id: str, model_id: str, recipe: dict[str, Any]) -> str:
    repo = ArtifactRepository(db)
    log_path = LOGS_DIR / f"training_{int(time.time())}.log"
    return repo.add_training_job(dataset_id=dataset_id, model_id=model_id, status="QUEUED", recipe=recipe, log_path=str(log_path.relative_to(ROOT_DIR)))
