from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2

from visionsort.annotations import (
    LocalTrackingExporter,
    MultiCameraReIDExporter,
    QualityGate,
    build_auto_annotator,
)
from visionsort.core.config import relative_to_root
from visionsort.core.enums import AnnotationStatus, PipelineState
from visionsort.core.paths import OBSERVATIONS_DIR, REPORTS_DIR, ROOT_DIR
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import ArtifactRepository, ControlRepository
from visionsort.datasets.pipeline import (
    build_dataset,
    rewrite_training_manifest,
    validate_dataset_splits,
)
from visionsort.observations.export import jsonl_to_parquet

def _load_json_dict(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}

def _finalize_dataset_status(
    db: VisionSortDB,
    *,
    artifact_repo: ArtifactRepository,
    control_repo: ControlRepository,
    session_id: str,
    dataset_id: str,
    report_path: Path,
) -> dict[str, Any]:
    counts = db.fetch_one(
        """
        SELECT
            SUM(CASE WHEN annotation_status = ? THEN 1 ELSE 0 END) AS needs_review,
            SUM(CASE WHEN annotation_status = ? THEN 1 ELSE 0 END) AS auto_accepted,
            SUM(CASE WHEN annotation_status = ? THEN 1 ELSE 0 END) AS human_validated,
            SUM(CASE WHEN annotation_status = ? THEN 1 ELSE 0 END) AS rejected
        FROM dataset_items WHERE dataset_id = ?
        """,
        (
            AnnotationStatus.NEEDS_REVIEW.value,
            AnnotationStatus.AUTO_ACCEPTED.value,
            AnnotationStatus.HUMAN_VALIDATED.value,
            AnnotationStatus.REJECTED.value,
            dataset_id,
        ),
    )
    needs_review = int((counts["needs_review"] if counts else 0) or 0)
    auto_accepted = int((counts["auto_accepted"] if counts else 0) or 0)
    human_validated = int((counts["human_validated"] if counts else 0) or 0)
    rejected = int((counts["rejected"] if counts else 0) or 0)
    trainable = auto_accepted + human_validated
    dataset = db.fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        raise RuntimeError("Dataset introuvable.")
    summary = _load_json_dict(dataset["summary_json"])
    summary["review_counts"] = {
        "needs_review": needs_review,
        "auto_accepted": auto_accepted,
        "human_validated": human_validated,
        "rejected": rejected,
    }
    manifest_path = ROOT_DIR / str(dataset["manifest_path"])
    trainable_manifest_rows = rewrite_training_manifest(
        db, dataset_id, manifest_path
    )
    split_integrity = validate_dataset_splits(db, dataset_id)
    if not split_integrity["valid"]:
        raise RuntimeError(f"Fuite entre splits: {split_integrity['leaks']}")
    summary["split_integrity"] = split_integrity
    summary["trainable_manifest_rows"] = trainable_manifest_rows
    status = (
        "REVIEW_PENDING"
        if needs_review > 0 or trainable <= 0
        else "DATASET_READY"
    )
    pipeline_state = (
        PipelineState.REVIEW_PENDING.value
        if status == "REVIEW_PENDING"
        else PipelineState.DATASET_READY.value
    )
    artifact_repo.upsert_dataset(
        dataset_id=dataset_id,
        name=dataset["name"],
        root_path=dataset["root_path"],
        status=status,
        manifest_path=dataset["manifest_path"],
        data_yaml_path=dataset["data_yaml_path"],
        summary=summary,
    )
    control_repo.update_capture_session(
        session_id,
        pipeline_state=pipeline_state,
        report_path=str(report_path.relative_to(ROOT_DIR)),
    )
    return {
        "dataset_id": dataset_id,
        "dataset_status": status,
        "needs_review": needs_review,
        "auto_accepted": auto_accepted,
        "human_validated": human_validated,
        "rejected": rejected,
        "trainable": trainable,
        "split_integrity": split_integrity,
    }


def pipeline_worker_loop(db_path: str, session_id: str, step: str, params: dict[str, Any]) -> None:
    db = VisionSortDB(Path(db_path))
    artifact_repo = ArtifactRepository(db)
    control_repo = ControlRepository(db)
    step_name = str(step).upper()
    log_dir = REPORTS_DIR / session_id
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / f"{int(time.time())}_{step_name}.json"
    step_id = artifact_repo.start_pipeline_step(session_id, step_name, params, log_path=str(report_path.relative_to(ROOT_DIR)))
    started = time.time()
    try:
        outputs: dict[str, Any] = {}
        if step_name == "PROCESS_SESSION":
            session = control_repo.get_capture_session(session_id)
            if session is None:
                raise RuntimeError("Session introuvable.")
            ended_at = session.get("ended_at")
            if ended_at is None:
                ended_at = float(time.time())
                control_repo.update_capture_session(session_id, ended_at=ended_at)
                session = control_repo.get_capture_session(session_id) or session
            counts = db.fetch_one(
                """
                SELECT
                    (SELECT COUNT(*) FROM tracklets WHERE session_id = ?) AS tracklets,
                    (SELECT COUNT(*) FROM recordings WHERE session_id = ?) AS recordings,
                    (SELECT COUNT(*) FROM events WHERE session_id = ?) AS events,
                    (SELECT COUNT(*) FROM dataset_items WHERE session_id = ?) AS dataset_items
                """,
                (session_id, session_id, session_id, session_id),
            )
            outputs = {
                "session_id": session_id,
                "started_at": session.get("started_at"),
                "ended_at": ended_at,
                "tracklets": int((counts["tracklets"] if counts else 0) or 0),
                "recordings": int((counts["recordings"] if counts else 0) or 0),
                "events": int((counts["events"] if counts else 0) or 0),
                "dataset_items": int((counts["dataset_items"] if counts else 0) or 0),
            }
            current_state = str(session.get("pipeline_state") or PipelineState.CAPTURED.value)
            if current_state in {PipelineState.CAPTURED.value, PipelineState.PROCESSED.value}:
                control_repo.update_capture_session(
                    session_id,
                    pipeline_state=PipelineState.PROCESSED.value,
                    report_path=str(report_path.relative_to(ROOT_DIR)),
                )
        elif step_name == "EXPORT_OBSERVATIONS_PARQUET":
            session_dir = OBSERVATIONS_DIR / session_id
            if not session_dir.exists():
                raise RuntimeError("Aucune observation JSONL trouvée pour cette session.")
            exported: list[dict[str, Any]] = []
            for jsonl in sorted(session_dir.glob("*.jsonl")):
                parquet = jsonl.with_suffix(".parquet")
                exported.append(jsonl_to_parquet(jsonl_path=jsonl, parquet_path=parquet))
            outputs = {"session_id": session_id, "exported": exported}
        elif step_name == "SAMPLE":
            session = control_repo.get_capture_session(session_id)
            if session is None:
                raise RuntimeError("Session introuvable.")
            existing_dataset_id = str(session.get("last_dataset_id") or "")
            force = bool(params.get("force", False))
            existing_dataset = db.fetch_one("SELECT * FROM datasets WHERE id = ?", (existing_dataset_id,)) if existing_dataset_id else None
            if existing_dataset is not None and not force:
                dataset_id = str(existing_dataset["id"])
                result = {
                    "dataset_id": dataset_id,
                    "root": existing_dataset["root_path"],
                    "manifest_rows": int(
                        (db.fetch_one("SELECT COUNT(*) AS c FROM dataset_items WHERE dataset_id = ?", (dataset_id,)) or {"c": 0})["c"]
                    ),
                    "reused_existing": True,
                }
            else:
                result = build_dataset(db, session_id=session_id, name=str(params.get("name") or "dataset_from_session"))
                dataset_id = str(result["dataset_id"])
            db.execute(
                "UPDATE capture_sessions SET pipeline_state = ?, last_dataset_id = ?, report_path = ?, updated_at = ? WHERE id = ?",
                (PipelineState.SAMPLED.value, dataset_id, str(report_path.relative_to(ROOT_DIR)), utc_now(), session_id),
            )
            outputs = {"dataset_id": dataset_id, **result}
        elif step_name == "AUTO_ANNOTATE":
            session = control_repo.get_capture_session(session_id)
            if session is None:
                raise RuntimeError("Session introuvable.")
            dataset_id = str(params.get("dataset_id") or session.get("last_dataset_id") or "")
            if not dataset_id:
                raise RuntimeError("dataset_id introuvable pour AUTO_ANNOTATE.")
            dataset = db.fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            if dataset is None:
                raise RuntimeError("Dataset introuvable.")
            model_id = str(params.get("model_id") or params.get("pseudo_label_model_id") or "")
            if not model_id:
                model_id = str(params.get("fallback_model_id") or "demo_synth_det")
            model_row_raw = db.fetch_one(
                "SELECT * FROM model_registry WHERE id = ?", (model_id,)
            )
            if model_row_raw is None:
                raise RuntimeError("Modèle introuvable.")
            model_row = dict(model_row_raw)
            if str(model_row["status"]) == "CHAMPION" and not bool(params.get("allow_champion_pseudo_labels", False)):
                raise RuntimeError("Pseudo-annotation refusée avec le modèle CHAMPION sans autorisation explicite.")
            backend = str(model_row["backend"])
            names = {"parcel": 0, "person": 1, "left_wrist": 2, "right_wrist": 3}
            items = [
                dict(row)
                for row in db.fetch_all(
                    "SELECT * FROM dataset_items WHERE dataset_id = ? ORDER BY source_id, timestamp_global",
                    (dataset_id,),
                )
            ]
            annotator = build_auto_annotator(
                db,
                model_row,
                config={
                    "dataset_id": dataset_id,
                    "pipeline_step": step_name,
                    "params": params,
                },
            )
            quality_gate = QualityGate()
            updated = 0
            skipped = 0
            force = bool(params.get("force", False))
            for item in items:
                if (
                    not force
                    and item.get("label_path")
                    and item.get("annotation_status") in {AnnotationStatus.AUTO_ACCEPTED.value, AnnotationStatus.HUMAN_VALIDATED.value}
                ):
                    skipped += 1
                    continue
                image_path = ROOT_DIR / item["image_path"]
                split = str(item.get("split") or "train")
                label_path = ROOT_DIR / Path(dataset["root_path"]) / "labels" / split / f"{item['id']}.txt"
                frame_index = int(item.get("frame_index") or 0)
                source_id = str(item.get("source_id") or "")
                detections = annotator.predict(
                    source_id=source_id,
                    frame_index=frame_index,
                    image_path=image_path,
                )
                count = annotator.write_labels(
                    image_path=image_path,
                    label_path=label_path,
                    detections=detections,
                    names=names,
                )
                image = cv2.imread(str(image_path))
                if image is None:
                    raise RuntimeError(f"Image illisible: {image_path}")
                status, stats = quality_gate.assess(
                    source_id=source_id,
                    detections=detections,
                    image_shape=image.shape[:2],
                    task=annotator.task,
                )
                meta = json.loads(item.get("metadata_json") or "{}")
                expected_count = int(meta.get("instance_count") or 0)
                stats["expected_visible_instances"] = expected_count
                stats["frame_annotation_complete"] = count >= expected_count
                if (
                    count < expected_count
                    and status != AnnotationStatus.REJECTED.value
                ):
                    status = AnnotationStatus.NEEDS_REVIEW.value
                meta["pseudo_label_model_id"] = model_id
                meta["pseudo_label_backend"] = backend
                meta["annotation_task"] = annotator.task
                meta["pseudo_label_count"] = count
                meta["quality_stats"] = stats
                meta["annotation_provenance"] = annotator.provenance(
                    session_id=str(item.get("session_id") or session_id),
                    camera_id=source_id,
                    timestamp_global=float(item.get("timestamp_global") or 0.0),
                    quality_score=float(stats.get("avg_conf") or 0.0),
                )
                meta["validated_on_site"] = False
                artifact_repo.update_dataset_item(
                    item["id"],
                    annotation_status=status,
                    label_path=relative_to_root(label_path),
                    metadata=meta,
                )
                updated += 1
            refreshed_items = [
                dict(row)
                for row in db.fetch_all(
                    "SELECT * FROM dataset_items WHERE dataset_id = ?", (dataset_id,)
                )
            ]
            dataset_root = ROOT_DIR / str(dataset["root_path"])
            tracking_rows = LocalTrackingExporter().export(
                refreshed_items, dataset_root / "tracking_manifest.jsonl"
            )
            reid_rows = MultiCameraReIDExporter().export(
                refreshed_items, dataset_root / "reid_manifest.jsonl"
            )
            summary = json.loads(dataset["summary_json"] or "{}")
            summary["pseudo_label_model_id"] = model_id
            summary["pseudo_label_backend"] = backend
            summary["auto_annotate_items_updated"] = updated
            summary["auto_annotate_items_skipped"] = skipped
            summary["annotation_task"] = annotator.task
            summary["tracking_manifest_rows"] = tracking_rows
            summary["reid_manifest_rows"] = reid_rows
            summary["model_loaded_once_per_job"] = True
            artifact_repo.upsert_dataset(
                dataset_id=dataset_id,
                name=dataset["name"],
                root_path=dataset["root_path"],
                status="AUTO_ANNOTATED",
                manifest_path=dataset["manifest_path"],
                data_yaml_path=dataset["data_yaml_path"],
                summary=summary,
            )
            control_repo.update_capture_session(session_id, pipeline_state=PipelineState.AUTO_ANNOTATED.value, report_path=str(report_path.relative_to(ROOT_DIR)))
            outputs = {
                "dataset_id": dataset_id,
                "items_updated": updated,
                "items_skipped": skipped,
                "model_id": model_id,
                "backend": backend,
                "task": annotator.task,
                "tracking_manifest_rows": tracking_rows,
                "reid_manifest_rows": reid_rows,
            }
        elif step_name == "FINALIZE_DATASET":
            session = control_repo.get_capture_session(session_id)
            if session is None:
                raise RuntimeError("Session introuvable.")
            dataset_id = str(params.get("dataset_id") or session.get("last_dataset_id") or "")
            if not dataset_id:
                raise RuntimeError("dataset_id introuvable pour FINALIZE_DATASET.")
            outputs = _finalize_dataset_status(
                db,
                artifact_repo=artifact_repo,
                control_repo=control_repo,
                session_id=session_id,
                dataset_id=dataset_id,
                report_path=report_path,
            )
        else:
            raise RuntimeError(f"Step pipeline inconnue: {step_name}")

        report = {"session_id": session_id, "step": step_name, "status": "COMPLETED", "started_at": started, "ended_at": time.time(), "inputs": params, "outputs": outputs}
        report_path.write_text(json.dumps(report, ensure_ascii=True), encoding="utf-8")
        artifact_repo.finish_pipeline_step(step_id, status="COMPLETED", outputs={"report_path": str(report_path.relative_to(ROOT_DIR)), **outputs})
    except Exception as exc:  # pragma: no cover - runtime
        report = {"session_id": session_id, "step": step_name, "status": "FAILED", "started_at": started, "ended_at": time.time(), "inputs": params, "error": str(exc)}
        report_path.write_text(json.dumps(report, ensure_ascii=True), encoding="utf-8")
        artifact_repo.finish_pipeline_step(step_id, status="FAILED", outputs={"report_path": str(report_path.relative_to(ROOT_DIR))}, error_text=str(exc))
