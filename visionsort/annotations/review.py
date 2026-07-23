from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionsort.core.config import relative_to_root
from visionsort.core.enums import AnnotationStatus
from visionsort.core.paths import ROOT_DIR
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import ArtifactRepository


def render_review_overlay(item: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    image_path = Path(str(item["image_path"]))
    if not image_path.is_absolute():
        image_path = ROOT_DIR / image_path
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Image illisible: {image_path}")
    metadata = json.loads(item.get("metadata_json") or "{}")
    detections = metadata.get("pseudo_labels") or metadata.get("observations") or []
    overlay = image.copy()
    for detection in detections:
        bbox = detection.get("bbox") or []
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 220), 2)
            cv2.putText(
                overlay,
                f"{detection.get('class_name', '?')} "
                f"{float(detection.get('confidence', 0.0)):.2f}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 220),
                2,
            )
        mask = detection.get("mask") or []
        if len(mask) >= 3:
            polygon = np.asarray(mask, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [polygon], True, (255, 80, 80), 2)
            fill = overlay.copy()
            cv2.fillPoly(fill, [polygon], (255, 80, 80))
            overlay = cv2.addWeighted(fill, 0.2, overlay, 0.8, 0)
        for keypoint in detection.get("keypoints") or []:
            if len(keypoint) < 2:
                continue
            confidence = float(keypoint[2]) if len(keypoint) > 2 else 1.0
            if confidence <= 0:
                continue
            cv2.circle(
                overlay,
                (int(round(float(keypoint[0]))), int(round(float(keypoint[1])))),
                3,
                (80, 255, 80),
                -1,
            )
    details = {
        "detections": detections,
        "expected_count": int(metadata.get("instance_count") or 0),
        "annotated_count": int(
            metadata.get("pseudo_label_count")
            if metadata.get("pseudo_label_count") is not None
            else len(detections)
        ),
        "quality_stats": metadata.get("quality_stats") or {},
        "provenance": metadata.get("annotation_provenance") or {},
        "reason": item.get("reason"),
        "task": metadata.get("annotation_task"),
    }
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), details


def export_review_cases(
    items: list[dict[str, Any]], *, export_format: str
) -> bytes:
    normalized = export_format.lower().replace(" ", "_")
    if normalized == "label_studio":
        tasks: list[dict[str, Any]] = []
        for item in items:
            metadata = json.loads(item.get("metadata_json") or "{}")
            image_path = Path(str(item["image_path"]))
            if not image_path.is_absolute():
                image_path = ROOT_DIR / image_path
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2] if image is not None else (1, 1)
            tasks.append(
                {
                    "id": item["id"],
                    "data": {
                        "image": item["image_path"],
                        "session_id": item.get("session_id"),
                        "camera_id": item.get("source_id"),
                    },
                    "meta": {
                        "reason": item.get("reason"),
                        "task": metadata.get("annotation_task"),
                    },
                    "predictions": [
                        {
                            "model_version": (
                                metadata.get("annotation_provenance") or {}
                            ).get("model_version"),
                            "result": _label_studio_results(
                                metadata.get("pseudo_labels")
                                or metadata.get("observations")
                                or [],
                                width=width,
                                height=height,
                            ),
                        }
                    ],
                }
            )
        return json.dumps(tasks, ensure_ascii=False, indent=2).encode("utf-8")
    if normalized == "cvat":
        root = ET.Element("annotations")
        for index, item in enumerate(items):
            image_path = Path(str(item["image_path"]))
            if not image_path.is_absolute():
                image_path = ROOT_DIR / image_path
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2] if image is not None else (0, 0)
            image_node = ET.SubElement(
                root,
                "image",
                {
                    "id": str(index),
                    "name": str(item["id"]),
                    "width": str(width),
                    "height": str(height),
                },
            )
            metadata = json.loads(item.get("metadata_json") or "{}")
            for detection in metadata.get("pseudo_labels") or metadata.get(
                "observations"
            ) or []:
                bbox = detection.get("bbox") or []
                if len(bbox) == 4:
                    ET.SubElement(
                        image_node,
                        "box",
                        {
                            "label": str(detection.get("class_name") or "parcel"),
                            "xtl": str(bbox[0]),
                            "ytl": str(bbox[1]),
                            "xbr": str(bbox[2]),
                            "ybr": str(bbox[3]),
                        },
                    )
                mask = detection.get("mask") or []
                if len(mask) >= 3:
                    ET.SubElement(
                        image_node,
                        "polygon",
                        {
                            "label": str(detection.get("class_name") or "parcel"),
                            "points": ";".join(
                                f"{point[0]},{point[1]}" for point in mask
                            ),
                        },
                    )
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)
    raise ValueError(f"Format d'export inconnu: {export_format}")


def _label_studio_results(
    detections: list[dict[str, Any]], *, width: int, height: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        bbox = detection.get("bbox") or []
        if len(bbox) == 4:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            results.append(
                {
                    "id": f"box-{index}",
                    "type": "rectanglelabels",
                    "original_width": width,
                    "original_height": height,
                    "value": {
                        "x": 100.0 * x1 / width,
                        "y": 100.0 * y1 / height,
                        "width": 100.0 * (x2 - x1) / width,
                        "height": 100.0 * (y2 - y1) / height,
                        "rectanglelabels": [
                            str(detection.get("class_name") or "parcel")
                        ],
                    },
                }
            )
        mask = detection.get("mask") or []
        if len(mask) >= 3:
            results.append(
                {
                    "id": f"polygon-{index}",
                    "type": "polygonlabels",
                    "original_width": width,
                    "original_height": height,
                    "value": {
                        "points": [
                            [
                                100.0 * float(point[0]) / width,
                                100.0 * float(point[1]) / height,
                            ]
                            for point in mask
                        ],
                        "polygonlabels": [
                            str(detection.get("class_name") or "parcel")
                        ],
                    },
                }
            )
    return results


def import_review_annotations(
    db: VisionSortDB,
    repository: ArtifactRepository,
    *,
    dataset_id: str,
    content: bytes,
    filename: str,
) -> dict[str, Any]:
    dataset = db.fetch_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        raise RuntimeError("Dataset introuvable.")
    task = str(dataset["task"])
    items = {
        str(row["id"]): dict(row)
        for row in db.fetch_all(
            "SELECT * FROM dataset_items WHERE dataset_id = ?", (dataset_id,)
        )
    }
    parsed = (
        _parse_cvat(content)
        if filename.lower().endswith(".xml")
        else _parse_label_studio(content)
    )
    updated = 0
    for item_id, detections in parsed.items():
        item = items.get(item_id)
        if item is None:
            continue
        image_path = Path(str(item["image_path"]))
        if not image_path.is_absolute():
            image_path = ROOT_DIR / image_path
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        label_path = (
            ROOT_DIR
            / str(dataset["root_path"])
            / "labels"
            / str(item.get("split") or "train")
            / f"{item_id}.txt"
        )
        names = {"parcel": 0, "person": 1}
        if task == "pose":
            names = {"person": 0}
        lines = [
            line
            for detection in detections
            if (
                line := _format_yolo_label(
                    task, detection, names, image.shape[1], image.shape[0]
                )
            )
            is not None
        ]
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        metadata = json.loads(item.get("metadata_json") or "{}")
        metadata["human_import"] = {
            "filename": filename,
            "detections": detections,
        }
        metadata["pseudo_labels"] = detections
        repository.update_dataset_item(
            item_id,
            annotation_status=AnnotationStatus.HUMAN_VALIDATED.value,
            label_path=relative_to_root(label_path),
            metadata=metadata,
        )
        updated += 1
    return {"updated_items": updated, "format": "cvat" if filename.lower().endswith(".xml") else "label_studio"}


def _parse_label_studio(content: bytes) -> dict[str, list[dict[str, Any]]]:
    tasks = json.loads(content.decode("utf-8"))
    output: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        detections: list[dict[str, Any]] = []
        annotations = task.get("annotations") or task.get("predictions") or []
        results = annotations[0].get("result", []) if annotations else []
        for result in results:
            value = result.get("value") or {}
            if result.get("type") == "rectanglelabels":
                original_width = float(result.get("original_width") or 1.0)
                original_height = float(result.get("original_height") or 1.0)
                x = float(value.get("x", 0.0)) * original_width / 100.0
                y = float(value.get("y", 0.0)) * original_height / 100.0
                width = (
                    float(value.get("width", 0.0)) * original_width / 100.0
                )
                height = (
                    float(value.get("height", 0.0)) * original_height / 100.0
                )
                detections.append(
                    {
                        "class_name": (value.get("rectanglelabels") or ["parcel"])[0],
                        "confidence": 1.0,
                        "bbox": [x, y, x + width, y + height],
                    }
                )
            elif result.get("type") == "polygonlabels":
                original_width = float(result.get("original_width") or 1.0)
                original_height = float(result.get("original_height") or 1.0)
                points = [
                    [
                        float(point[0]) * original_width / 100.0,
                        float(point[1]) * original_height / 100.0,
                    ]
                    for point in value.get("points") or []
                ]
                detections.append(
                    {
                        "class_name": (value.get("polygonlabels") or ["parcel"])[0],
                        "confidence": 1.0,
                        "bbox": _bbox_from_points(points),
                        "mask": points,
                    }
                )
        output[str(task["id"])] = detections
    return output


def _parse_cvat(content: bytes) -> dict[str, list[dict[str, Any]]]:
    root = ET.fromstring(content)
    output: dict[str, list[dict[str, Any]]] = {}
    for image in root.findall("image"):
        detections: list[dict[str, Any]] = []
        for box in image.findall("box"):
            detections.append(
                {
                    "class_name": box.attrib.get("label", "parcel"),
                    "confidence": 1.0,
                    "bbox": [
                        float(box.attrib["xtl"]),
                        float(box.attrib["ytl"]),
                        float(box.attrib["xbr"]),
                        float(box.attrib["ybr"]),
                    ],
                }
            )
        for polygon in image.findall("polygon"):
            points = [
                [float(value) for value in point.split(",")]
                for point in polygon.attrib.get("points", "").split(";")
                if point
            ]
            detections.append(
                {
                    "class_name": polygon.attrib.get("label", "parcel"),
                    "confidence": 1.0,
                    "bbox": _bbox_from_points(points),
                    "mask": points,
                }
            )
        output[str(image.attrib["name"])] = detections
    return output


def _bbox_from_points(points: list[list[float]]) -> list[float]:
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(float(point[0]) for point in points),
        min(float(point[1]) for point in points),
        max(float(point[0]) for point in points),
        max(float(point[1]) for point in points),
    ]


def _format_yolo_label(
    task: str,
    detection: dict[str, Any],
    names: dict[str, int],
    width: int,
    height: int,
) -> str | None:
    class_name = str(detection.get("class_name") or "")
    if class_name not in names:
        return None
    if task == "segmentation":
        points = detection.get("mask") or []
        if len(points) < 3:
            return None
        return f"{names[class_name]} " + " ".join(
            f"{float(point[0]) / width:.6f} {float(point[1]) / height:.6f}"
            for point in points
        )
    bbox = detection.get("bbox") or []
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    base = (
        f"{names[class_name]} {((x1 + x2) / 2) / width:.6f} "
        f"{((y1 + y2) / 2) / height:.6f} "
        f"{abs(x2 - x1) / width:.6f} {abs(y2 - y1) / height:.6f}"
    )
    if task == "pose":
        keypoints = detection.get("keypoints") or []
        if not keypoints:
            return None
        return base + " " + " ".join(
            f"{float(point[0]) / width:.6f} "
            f"{float(point[1]) / height:.6f} "
            f"{2 if len(point) < 3 or float(point[2]) > 0.5 else 1}"
            for point in keypoints
        )
    return base
