from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import cv2

from visionsort.core.config import relative_to_root
from visionsort.core.enums import AnnotationStatus, PipelineState
from visionsort.core.paths import OBSERVATIONS_DIR, REPORTS_DIR, ROOT_DIR
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import ArtifactRepository, ControlRepository
from visionsort.datasets.pipeline import build_dataset
from visionsort.observations.export import jsonl_to_parquet

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


def _ensure_ultralytics_dirs() -> None:
    path = ROOT_DIR / "data" / "ultralytics"
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(path))


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


def _yolo_write_labels(
    *,
    image_path: Path,
    label_path: Path,
    detections: list[dict[str, Any]],
    names: dict[str, int],
) -> tuple[int, dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Image illisible: {image_path}")
    h, w = image.shape[:2]
    lines: list[str] = []
    stats = {"count": 0, "avg_conf": 0.0, "max_iou": 0.0}
    used_boxes: list[tuple[float, float, float, float]] = []
    used_conf: list[float] = []
    for det in detections:
        cls = str(det["class_name"])
        if cls not in names:
            continue
        x1, y1, x2, y2 = det["bbox"]
        box = (float(x1), float(y1), float(x2), float(y2))
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h
        lines.append(f"{names[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        used_boxes.append(box)
        used_conf.append(float(det.get("confidence", 0.0)))
    if used_conf:
        stats["avg_conf"] = float(sum(used_conf) / len(used_conf))
    max_iou = 0.0
    for i in range(len(used_boxes)):
        for j in range(i + 1, len(used_boxes)):
            max_iou = max(max_iou, _bbox_iou(used_boxes[i], used_boxes[j]))
    stats["max_iou"] = float(max_iou)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    stats["count"] = len(lines)
    return len(lines), stats


def _quality_gate(stats: dict[str, Any]) -> str:
    if stats.get("count", 0) <= 0:
        return AnnotationStatus.NEEDS_REVIEW.value
    if float(stats.get("avg_conf", 0.0)) < 0.60:
        return AnnotationStatus.NEEDS_REVIEW.value
    if float(stats.get("max_iou", 0.0)) > 0.80:
        return AnnotationStatus.NEEDS_REVIEW.value
    return AnnotationStatus.AUTO_ACCEPTED.value


def _load_json_dict(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _demo_predict(db: VisionSortDB, *, source_id: str, frame_index: int) -> list[dict[str, Any]]:
    row = db.fetch_one("SELECT * FROM sources WHERE id = ?", (source_id,))
    if row is None:
        return []
    uri = str(row["uri"])
    jsonl = Path(uri).with_suffix(".jsonl")
    if not jsonl.exists():
        return []
    output: list[dict[str, Any]] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("frame_index", -1)) == int(frame_index):
            output.append(record)
    return output


def _ultralytics_predict(model_id: str, weights_path: str, image_path: Path) -> list[dict[str, Any]]:
    if YOLO is None:
        raise RuntimeError("Ultralytics indisponible.")
    _ensure_ultralytics_dirs()
    model = YOLO(weights_path)
    results = model.predict(str(image_path), verbose=False)
    if not results:
        return []
    res0 = results[0]
    output: list[dict[str, Any]] = []
    names = getattr(res0, "names", {}) or {}
    boxes = getattr(res0, "boxes", None)
    if boxes is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy().tolist()
    conf = boxes.conf.cpu().numpy().tolist()
    cls = boxes.cls.cpu().numpy().tolist()
    for box, c, k in zip(xyxy, conf, cls, strict=False):
        output.append(
            {
                "class_name": str(names.get(int(k), int(k))),
                "confidence": float(c),
                "bbox": (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                "model_id": model_id,
            }
        )
    return output


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
    status = "REVIEW_PENDING" if needs_review > 0 else "DATASET_READY"
    pipeline_state = PipelineState.REVIEW_PENDING.value if needs_review > 0 else PipelineState.DATASET_READY.value
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
            model_row = db.fetch_one("SELECT * FROM model_registry WHERE id = ?", (model_id,))
            if model_row is None:
                raise RuntimeError("Modèle introuvable.")
            if str(model_row["status"]) == "CHAMPION" and not bool(params.get("allow_champion_pseudo_labels", False)):
                raise RuntimeError("Pseudo-annotation refusée avec le modèle CHAMPION sans autorisation explicite.")
            backend = str(model_row["backend"])
            weights = str(model_row["weights_path"])
            names = {"parcel": 0, "person": 1, "left_wrist": 2, "right_wrist": 3}
            items = [dict(r) for r in db.fetch_all("SELECT * FROM dataset_items WHERE dataset_id = ?", (dataset_id,))]
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
                if backend == "demo":
                    detections = _demo_predict(db, source_id=source_id, frame_index=frame_index)
                else:
                    detections = _ultralytics_predict(model_id, weights, image_path)
                count, stats = _yolo_write_labels(image_path=image_path, label_path=label_path, detections=detections, names=names)
                status = _quality_gate(stats)
                meta = json.loads(item.get("metadata_json") or "{}")
                meta["pseudo_label_model_id"] = model_id
                meta["pseudo_label_backend"] = backend
                meta["pseudo_label_count"] = count
                meta["quality_stats"] = stats
                meta["validated_on_site"] = False
                artifact_repo.update_dataset_item(
                    item["id"],
                    annotation_status=status,
                    label_path=relative_to_root(label_path),
                    metadata=meta,
                )
                updated += 1
            summary = json.loads(dataset["summary_json"] or "{}")
            summary["pseudo_label_model_id"] = model_id
            summary["pseudo_label_backend"] = backend
            summary["auto_annotate_items_updated"] = updated
            summary["auto_annotate_items_skipped"] = skipped
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
            outputs = {"dataset_id": dataset_id, "items_updated": updated, "items_skipped": skipped, "model_id": model_id, "backend": backend}
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
