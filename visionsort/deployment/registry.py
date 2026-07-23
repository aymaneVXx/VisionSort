from __future__ import annotations

import json

from visionsort.core.enums import ModelStatus, PipelineState
from visionsort.database.db import VisionSortDB, utc_now


def activate_model(db: VisionSortDB, model_id: str) -> None:
    row = db.fetch_one("SELECT status FROM model_registry WHERE id = ?", (model_id,))
    if row is None:
        raise RuntimeError("Modèle introuvable.")
    if row["status"] not in {ModelStatus.CHAMPION.value, ModelStatus.ARCHIVED.value}:
        raise RuntimeError("Seuls les modèles CHAMPION ou ARCHIVED peuvent être activés.")
    with db.connect() as conn:
        conn.execute("UPDATE model_registry SET is_active = 0, updated_at = ?", (utc_now(),))
        conn.execute("UPDATE model_registry SET is_active = 1, updated_at = ? WHERE id = ?", (utc_now(), model_id))


def promote_model(db: VisionSortDB, model_id: str) -> None:
    row = db.fetch_one("SELECT * FROM model_registry WHERE id = ?", (model_id,))
    if row is None:
        raise RuntimeError("Modèle introuvable.")
    if row["status"] != ModelStatus.CANDIDATE.value:
        raise RuntimeError("Seul un modèle CANDIDATE peut être promu.")
    metrics = json.loads(row["metrics_json"] or "{}")
    test_metrics = metrics.get("test") or {}
    criteria = metrics.get("promotion_criteria") or {}
    required_metrics = (
        "precision",
        "recall",
        "mAP50",
        "count_accuracy",
        "merge_rate",
        "fps",
    )
    unavailable = [
        metric for metric in required_metrics if metrics.get(metric) is None
    ]
    if unavailable:
        raise RuntimeError(
            "Promotion refusée: métriques obligatoires indisponibles: "
            + ", ".join(unavailable)
        )
    if (
        not metrics.get("promotion_eligible")
        or test_metrics.get("status") != "COMPLETED"
        or not test_metrics.get("frozen")
        or not criteria
    ):
        raise RuntimeError(
            "Promotion refusée: test figé, critères configurés et seuils validés requis."
        )
    with db.connect() as conn:
        conn.execute(
            "UPDATE model_registry SET status = ?, updated_at = ? WHERE status = ?",
            (ModelStatus.ARCHIVED.value, utc_now(), ModelStatus.CHAMPION.value),
        )
        conn.execute(
            "UPDATE model_registry SET status = ?, is_active = 1, updated_at = ? WHERE id = ?",
            (ModelStatus.CHAMPION.value, utc_now(), model_id),
        )
        conn.execute("UPDATE model_registry SET is_active = 0, updated_at = ? WHERE id <> ?", (utc_now(), model_id))
        if row["created_from_job_id"]:
            job = conn.execute("SELECT dataset_id FROM training_jobs WHERE id = ?", (row["created_from_job_id"],)).fetchone()
            if job:
                ds = conn.execute("SELECT summary_json FROM datasets WHERE id = ?", (job["dataset_id"],)).fetchone()
                if ds and ds["summary_json"]:
                    session_id = json.loads(ds["summary_json"]).get("session_id")
                    if session_id:
                        conn.execute(
                            "UPDATE capture_sessions SET pipeline_state = ?, last_candidate_model_id = ?, updated_at = ? WHERE id = ?",
                            (PipelineState.DEPLOYED.value, model_id, utc_now(), session_id),
                        )


def set_model_status(db: VisionSortDB, model_id: str, status: ModelStatus | str) -> None:
    value = str(status)
    if value not in {item.value for item in ModelStatus}:
        raise RuntimeError(f"Statut de modèle non supporté: {value}")
    db.execute(
        "UPDATE model_registry SET status = ?, is_active = CASE WHEN ? = ? THEN 0 ELSE is_active END, updated_at = ? WHERE id = ?",
        (value, value, ModelStatus.REJECTED.value, utc_now(), model_id),
    )


def rollback_to_previous_active(db: VisionSortDB) -> str | None:
    row = db.fetch_one(
        """
        SELECT id
        FROM model_registry
        WHERE status IN (?, ?)
        ORDER BY
            CASE status WHEN ? THEN 0 WHEN ? THEN 1 ELSE 2 END,
            updated_at DESC
        LIMIT 1
        """,
        (
            ModelStatus.ARCHIVED.value,
            ModelStatus.CHAMPION.value,
            ModelStatus.ARCHIVED.value,
            ModelStatus.CHAMPION.value,
        ),
    )
    if row:
        activate_model(db, row["id"])
        return row["id"]
    return None
