from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from visionsort.core.enums import ModelStatus, PipelineState
from visionsort.core.paths import MODELS_DIR, ROOT_DIR
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


def _json_dict(value: str | None) -> dict[str, Any]:
    try:
        payload = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def validate_activation_candidate(
    db: VisionSortDB, model_id: str
) -> dict[str, Any]:
    row = db.fetch_one(
        "SELECT * FROM model_registry WHERE id = ?", (model_id,)
    )
    if row is None:
        raise RuntimeError("Modele introuvable.")
    model = dict(row)
    status = str(model["status"])
    if status not in {
        ModelStatus.BASELINE.value,
        ModelStatus.CHAMPION.value,
        ModelStatus.ARCHIVED.value,
    }:
        raise RuntimeError(f"Activation refusee: statut {status} invalide.")
    notes = _json_dict(model.get("notes_json"))
    if (
        status == ModelStatus.ARCHIVED.value
        and notes.get("archived_from_status")
        == ModelStatus.CANDIDATE.value
    ):
        raise RuntimeError(
            "Activation refusee: modele archive depuis CANDIDATE sans deploiement."
        )
    if status == ModelStatus.BASELINE.value:
        active = db.fetch_one(
            """
            SELECT id, notes_json FROM model_registry
            WHERE task = ? AND is_active = 1
            """,
            (str(model["task"]),),
        )
        active_notes = _json_dict(active["notes_json"]) if active else {}
        if active is not None and not bool(active_notes.get("demo_only")):
            raise RuntimeError(
                "Activation baseline refusee: un modele actif existe deja pour cette tache."
            )
    _validate_rollback_artifact(model)
    return model


def _required_promotion_metrics(task: str) -> tuple[str, ...]:
    if task == "segmentation":
        return ("mask_precision", "mask_recall", "mask_mAP50", "fps")
    if task == "pose":
        return ("pose_precision", "pose_recall", "pose_mAP50", "fps")
    return (
        "precision",
        "recall",
        "mAP50",
        "count_accuracy",
        "merge_rate",
        "fps",
    )


def validate_promotion_candidate(
    db: VisionSortDB, model_id: str
) -> dict[str, Any]:
    row = db.fetch_one(
        "SELECT * FROM model_registry WHERE id = ?", (model_id,)
    )
    if row is None:
        raise RuntimeError("Modele introuvable.")
    model = dict(row)
    if model["status"] != ModelStatus.CANDIDATE.value:
        raise RuntimeError("Seul un modele CANDIDATE peut etre promu.")
    metrics = _json_dict(model.get("metrics_json"))
    test_metrics = metrics.get("test") or {}
    criteria = metrics.get("promotion_criteria") or {}
    unavailable = [
        metric
        for metric in _required_promotion_metrics(str(model["task"]))
        if metrics.get(metric) is None
    ]
    if unavailable:
        raise RuntimeError(
            "Promotion refusee: metriques obligatoires indisponibles: "
            + ", ".join(unavailable)
        )
    if (
        not metrics.get("promotion_eligible")
        or test_metrics.get("status") != "COMPLETED"
        or not test_metrics.get("frozen")
        or not criteria
    ):
        raise RuntimeError(
            "Promotion refusee: test fige, criteres configures et seuils valides requis."
        )
    if model.get("created_from_job_id"):
        training_job = db.fetch_one(
            "SELECT dataset_id FROM training_jobs WHERE id = ?",
            (model["created_from_job_id"],),
        )
        if training_job is None:
            raise RuntimeError(
                "Promotion refusee: job d'entrainement introuvable."
            )
        dataset_id = str(training_job["dataset_id"])
        from visionsort.datasets.integrity import DatasetIntegrityValidator
        from visionsort.datasets.pipeline import verify_dataset_fingerprint

        integrity = DatasetIntegrityValidator(db, dataset_id).validate()
        if not integrity["valid"]:
            raise RuntimeError(
                "Promotion refusee: integrite du dataset invalide. "
                + " ".join(integrity["errors"][:5])
            )
        if not verify_dataset_fingerprint(db, dataset_id)["valid"]:
            raise RuntimeError(
                "Promotion refusee: fingerprint du dataset invalide."
            )
        real_test = db.fetch_one(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN di.annotation_status = 'HUMAN_VALIDATED'
                            THEN 1 ELSE 0 END) AS human_count
            FROM dataset_items di
            JOIN capture_sessions cs ON cs.id = di.session_id
            WHERE di.dataset_id = ? AND di.split = 'test'
              AND cs.demo_mode = 0
            """,
            (dataset_id,),
        )
        total = int((real_test["total"] if real_test else 0) or 0)
        human = int((real_test["human_count"] if real_test else 0) or 0)
        if total and human != total:
            raise RuntimeError(
                "Promotion refusee: le test reel doit etre entierement HUMAN_VALIDATED."
            )
    return model


def activate_model(db: VisionSortDB, model_id: str) -> None:
    validate_activation_candidate(db, model_id)
    row = db.fetch_one(
        "SELECT status, task FROM model_registry WHERE id = ?",
        (model_id,),
    )
    if row is None:
        raise RuntimeError("Modèle introuvable.")
    if row["status"] not in {
        ModelStatus.BASELINE.value,
        ModelStatus.CHAMPION.value,
        ModelStatus.ARCHIVED.value,
    }:
        raise RuntimeError("Seuls les modèles CHAMPION ou ARCHIVED peuvent être activés.")
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE model_registry SET is_active = 0, updated_at = ?
            WHERE task = ?
            """,
            (utc_now(), row["task"]),
        )
        conn.execute(
            """
            UPDATE model_registry
            SET status = CASE WHEN status = ? THEN ? ELSE status END,
                is_active = 1, updated_at = ?
            WHERE id = ?
            """,
            (
                ModelStatus.BASELINE.value,
                ModelStatus.CHAMPION.value,
                utc_now(),
                model_id,
            ),
        )


def promote_model(db: VisionSortDB, model_id: str) -> None:
    row = db.fetch_one("SELECT * FROM model_registry WHERE id = ?", (model_id,))
    if row is None:
        raise RuntimeError("Modèle introuvable.")
    if row["status"] != ModelStatus.CANDIDATE.value:
        raise RuntimeError("Seul un modèle CANDIDATE peut être promu.")
    metrics = json.loads(row["metrics_json"] or "{}")
    test_metrics = metrics.get("test") or {}
    criteria = metrics.get("promotion_criteria") or {}
    required_metrics = _required_promotion_metrics(str(row["task"]))
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
    validate_promotion_candidate(db, model_id)
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
    row = db.fetch_one(
        "SELECT status, is_active, notes_json FROM model_registry WHERE id = ?",
        (model_id,),
    )
    if row is None:
        raise RuntimeError("Modele introuvable.")
    if int(row["is_active"] or 0) == 1 and value in {
        ModelStatus.REJECTED.value,
        ModelStatus.ARCHIVED.value,
    }:
        raise RuntimeError(
            "Operation refusee: basculez d'abord vers un autre modele avant de modifier le modele actif."
        )
    notes = _json_dict(row["notes_json"])
    if value == ModelStatus.ARCHIVED.value:
        notes["archived_from_status"] = str(row["status"])
    db.execute(
        """
        UPDATE model_registry
        SET status = ?,
            is_active = CASE WHEN ? = ? THEN 0 ELSE is_active END,
            notes_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            value,
            value,
            ModelStatus.REJECTED.value,
            json.dumps(notes, sort_keys=True),
            utc_now(),
            model_id,
        ),
    )


def import_baseline_model(
    db: VisionSortDB,
    *,
    source_path: str,
    task: str,
    name: str | None = None,
) -> dict[str, Any]:
    if task not in {"detection", "segmentation", "pose"}:
        raise RuntimeError("Import baseline refuse: tache Ultralytics non supportee.")
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"Fichier de poids introuvable: {source}")
    if source.suffix.lower() != ".pt":
        raise RuntimeError(
            "Import baseline refuse: un fichier Ultralytics .pt est requis."
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    model_id = f"baseline-{task}-{digest[:12]}"
    if db.fetch_one("SELECT id FROM model_registry WHERE id = ?", (model_id,)):
        raise RuntimeError(f"Ce baseline est deja enregistre: {model_id}")
    destination_dir = MODELS_DIR / "versions" / model_id
    destination_dir.mkdir(parents=True, exist_ok=False)
    destination = destination_dir / "weights.pt"
    try:
        shutil.copy2(source, destination)
        copied_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if copied_digest != digest:
            raise RuntimeError(
                "La copie immuable du baseline ne correspond pas au SHA256 source."
            )
        now = utc_now()
        db.execute(
            """
            INSERT INTO model_registry
            (id, name, task, backend, weights_path, status, is_active,
             notes_json, metrics_json, parent_model_id,
             created_from_job_id, created_at, updated_at)
            VALUES (?, ?, ?, 'ultralytics', ?, ?, 0, ?, '{}', NULL,
                    NULL, ?, ?)
            """,
            (
                model_id,
                str(name or source.stem),
                task,
                str(destination.relative_to(ROOT_DIR)),
                ModelStatus.BASELINE.value,
                json.dumps(
                    {
                        "artifact_sha256": digest,
                        "immutable_artifact": True,
                        "validated_on_site": False,
                        "import_source_name": source.name,
                    },
                    sort_keys=True,
                ),
                now,
                now,
            ),
        )
    except Exception:
        if destination.exists():
            destination.unlink()
        if destination_dir.exists():
            destination_dir.rmdir()
        raise
    return {
        "model_id": model_id,
        "task": task,
        "status": ModelStatus.BASELINE.value,
        "sha256": digest,
        "weights_path": str(destination.relative_to(ROOT_DIR)),
        "validated_on_site": False,
    }


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
