from __future__ import annotations

import json
import uuid
from typing import Any

from visionsort.core.enums import ModelStatus, PipelineState
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.inference.engine import resolve_model_artifact


SUCCESSFUL_ACTIVATION_STATUSES = {
    "ACTIVE",
    "SUPERSEDED",
    "ROLLED_BACK",
}


def create_activation_history(
    db: VisionSortDB,
    *,
    task: str,
    previous_model_id: str | None,
    activated_model_id: str,
    routing_generation: int,
    actor: str,
    reason: str,
    session_id: str | None = None,
    source_ids: list[str] | None = None,
    rolled_back_from_activation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    activation_id = f"activation-{uuid.uuid4().hex}"
    db.execute(
        """
        INSERT INTO model_activation_history
        (id, task, previous_model_id, activated_model_id,
         routing_generation, status, runtime_applied, actor, reason,
         session_id, source_ids_json, activated_at, completed_at,
         rolled_back_from_activation_id, error_text, metadata_json)
        VALUES (?, ?, ?, ?, ?, 'PENDING', 0, ?, ?, ?, ?, ?, NULL, ?,
                NULL, ?)
        """,
        (
            activation_id,
            str(task),
            previous_model_id,
            str(activated_model_id),
            int(routing_generation),
            str(actor),
            str(reason),
            session_id,
            json.dumps(sorted(source_ids or [])),
            utc_now(),
            rolled_back_from_activation_id,
            json.dumps(metadata or {}),
        ),
    )
    return activation_id


def finish_activation_history(
    db: VisionSortDB,
    activation_id: str,
    *,
    status: str,
    runtime_applied: bool,
    error_text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    current = db.fetch_one(
        "SELECT metadata_json FROM model_activation_history WHERE id = ?",
        (activation_id,),
    )
    merged_metadata = json.loads(
        (current["metadata_json"] if current else None) or "{}"
    )
    merged_metadata.update(metadata or {})
    db.execute(
        """
        UPDATE model_activation_history
        SET status = ?, runtime_applied = ?, completed_at = ?,
            error_text = ?, metadata_json = ?
        WHERE id = ?
        """,
        (
            str(status),
            int(runtime_applied),
            utc_now(),
            error_text,
            json.dumps(merged_metadata),
            activation_id,
        ),
    )


def mark_previous_activation(
    db: VisionSortDB,
    *,
    task: str,
    model_id: str | None,
    status: str,
) -> str | None:
    if not model_id:
        return None
    row = db.fetch_one(
        """
        SELECT id FROM model_activation_history
        WHERE task = ? AND activated_model_id = ?
          AND status = 'ACTIVE' AND runtime_applied = 1
        ORDER BY COALESCE(completed_at, activated_at) DESC LIMIT 1
        """,
        (str(task), str(model_id)),
    )
    if row is None:
        return None
    db.execute(
        """
        UPDATE model_activation_history
        SET status = ?, completed_at = ?
        WHERE id = ?
        """,
        (str(status), utc_now(), str(row["id"])),
    )
    return str(row["id"])


def list_activation_history(
    db: VisionSortDB,
    *,
    task: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    if task is None:
        rows = db.fetch_all(
            """
            SELECT * FROM model_activation_history
            ORDER BY activated_at DESC LIMIT ?
            """,
            (int(limit),),
        )
    else:
        rows = db.fetch_all(
            """
            SELECT * FROM model_activation_history
            WHERE task = ? ORDER BY activated_at DESC LIMIT ?
            """,
            (str(task), int(limit)),
        )
    return [dict(row) for row in rows]


def _validate_rollback_artifact(model: dict[str, Any]) -> None:
    if str(model.get("backend") or "") == "demo":
        return
    resolve_model_artifact(model)


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
    db: VisionSortDB,
    task: str | None = None,
    *,
    apply: bool = True,
) -> str:
    selected_task = str(task or "detection")
    current = db.fetch_one(
        """
        SELECT id FROM model_registry
        WHERE task = ? AND is_active = 1
        LIMIT 1
        """,
        (selected_task,),
    )
    current_model_id = str(current["id"]) if current else None
    rows = db.fetch_all(
        """
        SELECT h.*, m.name, m.backend, m.weights_path, m.status AS model_status,
               m.notes_json, m.metrics_json, m.parent_model_id,
               m.created_from_job_id, m.created_at AS model_created_at,
               m.updated_at AS model_updated_at, m.is_active
        FROM model_activation_history h
        JOIN model_registry m ON m.id = h.activated_model_id
        WHERE h.task = ?
          AND h.runtime_applied = 1
          AND h.status IN ('ACTIVE', 'SUPERSEDED', 'ROLLED_BACK')
          AND h.activated_model_id <> COALESCE(?, '')
          AND m.task = ?
          AND m.status IN (?, ?)
        ORDER BY COALESCE(h.completed_at, h.activated_at) DESC
        """,
        (
            selected_task,
            current_model_id,
            selected_task,
            ModelStatus.CHAMPION.value,
            ModelStatus.ARCHIVED.value,
        ),
    )
    invalid_errors: list[str] = []
    for row in rows:
        candidate = dict(row)
        candidate["id"] = str(row["activated_model_id"])
        try:
            _validate_rollback_artifact(candidate)
        except Exception as exc:
            invalid_errors.append(
                f"{candidate['id']}: {exc}"
            )
            continue
        model_id = str(candidate["id"])
        if apply:
            activate_model(db, model_id)
        return model_id
    suffix = (
        " Artefacts invalides: " + " | ".join(invalid_errors)
        if invalid_errors
        else ""
    )
    raise RuntimeError(
        "Rollback refusé: aucun modèle précédemment déployé et valide "
        f"pour la tâche {selected_task}.{suffix}"
    )
