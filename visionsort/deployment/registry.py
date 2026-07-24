from __future__ import annotations

import json

from visionsort.core.enums import ModelStatus, PipelineState
from visionsort.database.db import VisionSortDB, utc_now


def activate_model(db: VisionSortDB, model_id: str) -> None:
    row = db.fetch_one(
        "SELECT status, task FROM model_registry WHERE id = ?",
        (model_id,),
    )
    if row is None:
        raise RuntimeError("Modèle introuvable.")
    if row["status"] not in {ModelStatus.CHAMPION.value, ModelStatus.ARCHIVED.value}:
        raise RuntimeError("Seuls les modèles CHAMPION ou ARCHIVED peuvent être activés.")
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE model_registry SET is_active = 0, updated_at = ?
            WHERE task = ?
            """,
            (utc_now(), row["task"]),
        )
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
    if row["created_from_job_id"]:
        training_job = db.fetch_one(
            """
            SELECT dataset_id FROM training_jobs WHERE id = ?
            """,
            (row["created_from_job_id"],),
        )
        if training_job is None:
            raise RuntimeError(
                "Promotion refusée: job d'entraînement introuvable."
            )
        from visionsort.datasets.integrity import DatasetIntegrityValidator
        from visionsort.datasets.pipeline import verify_dataset_fingerprint

        integrity = DatasetIntegrityValidator(
            db, str(training_job["dataset_id"])
        ).validate()
        if not integrity["valid"]:
            raise RuntimeError(
                "Promotion refusée: intégrité du dataset invalide. "
                + " ".join(integrity["errors"][:5])
            )
        fingerprint = verify_dataset_fingerprint(
            db, str(training_job["dataset_id"])
        )
        if not fingerprint["valid"]:
            raise RuntimeError(
                "Promotion refusée: fingerprint du dataset invalide."
            )
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE model_registry
            SET is_active = 0
            WHERE task = ? AND is_active = 1 AND id <> ?
            """,
            (
                row["task"],
                model_id,
            ),
        )
        conn.execute(
            """
            UPDATE model_registry
            SET status = ?, is_active = 0, updated_at = ?
            WHERE status = ? AND task = ?
            """,
            (
                ModelStatus.ARCHIVED.value,
                utc_now(),
                ModelStatus.CHAMPION.value,
                row["task"],
            ),
        )
        conn.execute(
            """
            UPDATE model_registry SET is_active = 0, updated_at = ?
            WHERE task = ?
            """,
            (utc_now(), row["task"]),
        )
        conn.execute(
            "UPDATE model_registry SET status = ?, is_active = 1, updated_at = ? WHERE id = ?",
            (ModelStatus.CHAMPION.value, utc_now(), model_id),
        )
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


def rollback_to_previous_active(
    db: VisionSortDB, task: str | None = None
) -> str | None:
    selected_task = str(task or "detection")
    row = db.fetch_one(
        """
        SELECT id
        FROM model_registry
        WHERE task = ? AND status IN (?, ?, ?) AND is_active = 0
        ORDER BY
            CASE WHEN status = ? THEN 0 ELSE 1 END,
            updated_at DESC
        LIMIT 1
        """,
        (
            selected_task,
            ModelStatus.ARCHIVED.value,
            ModelStatus.CHAMPION.value,
            ModelStatus.CANDIDATE.value,
            ModelStatus.ARCHIVED.value,
        ),
    )
    if row:
        candidate = db.fetch_one(
            "SELECT status FROM model_registry WHERE id = ?", (row["id"],)
        )
        if (
            candidate is not None
            and candidate["status"]
            in {ModelStatus.CHAMPION.value, ModelStatus.ARCHIVED.value}
        ):
            activate_model(db, row["id"])
        else:
            with db.connect() as conn:
                conn.execute(
                    """
                    UPDATE model_registry SET is_active = 0, updated_at = ?
                    WHERE task = ?
                    """,
                    (utc_now(), selected_task),
                )
                conn.execute(
                    """
                    UPDATE model_registry SET is_active = 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (utc_now(), row["id"]),
                )
        return row["id"]
    return None
