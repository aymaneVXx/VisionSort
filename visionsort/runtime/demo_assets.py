from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from visionsort.core.paths import DEMO_DIR


def _rect(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    return (x, y, x + w, y + h)


def ensure_demo_assets() -> dict[str, str]:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    assets = {}
    for camera_id in ["C1", "C2", "C3"]:
        video_path = DEMO_DIR / f"{camera_id}.mp4"
        sidecar_path = DEMO_DIR / f"{camera_id}.jsonl"
        if not video_path.exists() or not sidecar_path.exists():
            _build_camera_asset(camera_id, video_path, sidecar_path)
        assets[camera_id] = str(video_path)
    return assets


def _build_camera_asset(camera_id: str, video_path: Path, sidecar_path: Path) -> None:
    width, height = 640, 360
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))
    annotations: list[dict] = []
    for frame_index in range(72):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (35, 35, 35)
        cv2.rectangle(frame, (40, 140), (600, 250), (80, 80, 80), -1)
        cv2.putText(frame, f"VisionSort DEMO {camera_id}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        objects = _frame_objects(camera_id, frame_index)
        for obj in objects:
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            if obj["class_name"] == "parcel":
                color = (40, 210, 210)
            elif obj["class_name"] == "person":
                color = (210, 140, 40)
            else:
                color = (180, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1 if obj["class_name"] != "parcel" else 2)
            cv2.putText(frame, obj["class_name"], (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            annotations.append({"frame_index": frame_index, **obj})

        writer.write(frame)
    writer.release()
    sidecar_path.write_text("\n".join(json.dumps(item) for item in annotations) + "\n", encoding="utf-8")


def _frame_objects(camera_id: str, frame_index: int) -> list[dict]:
    objects: list[dict] = []
    if camera_id == "C1":
        p1_x = 70 + frame_index * 6
        p2_x = 10 + frame_index * 4
        for parcel_hint, x in [("P1", p1_x), ("P2", p2_x)]:
            if -80 < x < 620:
                objects.append(
                    {
                        "class_name": "parcel",
                        "confidence": 0.98,
                        "bbox": _rect(x, 165 if parcel_hint == "P1" else 205, 70, 42),
                        "attributes": {"parcel_hint": parcel_hint, "validated_on_site": False},
                    }
                )
    elif camera_id == "C2":
        if 0 <= frame_index <= 40:
            p1_x = 20 + frame_index * 7
            objects.append(
                {
                    "class_name": "parcel",
                    "confidence": 0.97,
                    "bbox": _rect(p1_x, 182, 72, 42),
                    "attributes": {"parcel_hint": "P1", "validated_on_site": False},
                }
            )
        if 18 <= frame_index <= 48:
            p2_x = -70 + frame_index * 8
            objects.append(
                {
                    "class_name": "parcel",
                    "confidence": 0.74 if 28 <= frame_index <= 34 else 0.92,
                    "bbox": _rect(p2_x, 140, 75, 45),
                    "attributes": {"parcel_hint": "P2", "validated_on_site": False},
                }
            )
        if 24 <= frame_index <= 55:
            person_x = 420
            wrist_shift = max(0, frame_index - 28) * 2
            objects.extend(
                [
                    {"class_name": "person", "confidence": 0.95, "bbox": _rect(person_x, 95, 110, 215), "attributes": {"operator_id": "OP1"}},
                    {"class_name": "left_wrist", "confidence": 0.93, "bbox": _rect(430 - wrist_shift, 175, 22, 22), "attributes": {"operator_id": "OP1"}},
                    {"class_name": "right_wrist", "confidence": 0.93, "bbox": _rect(458 - wrist_shift, 185, 22, 22), "attributes": {"operator_id": "OP1"}},
                ]
            )
    elif camera_id == "C3":
        if 8 <= frame_index <= 40:
            person_x = 170 + frame_index * 3
            parcel_x = 195 + frame_index * 3
            parcel_y = 165 + max(0, 16 - abs(frame_index - 24))
            objects.extend(
                [
                    {"class_name": "person", "confidence": 0.95, "bbox": _rect(person_x, 80, 120, 230), "attributes": {"operator_id": "OP1"}},
                    {"class_name": "left_wrist", "confidence": 0.90, "bbox": _rect(person_x + 20, 175, 22, 22), "attributes": {"operator_id": "OP1"}},
                    {"class_name": "right_wrist", "confidence": 0.90, "bbox": _rect(person_x + 56, 185, 22, 22), "attributes": {"operator_id": "OP1"}},
                    {
                        "class_name": "parcel",
                        "confidence": 0.96,
                        "bbox": _rect(parcel_x, parcel_y, 68, 40),
                        "attributes": {"parcel_hint": "P1", "validated_on_site": False},
                    },
                ]
            )
        if 41 <= frame_index <= 60:
            objects.append(
                {
                    "class_name": "parcel",
                    "confidence": 0.96,
                    "bbox": _rect(410, 115 if frame_index < 51 else 230, 68, 40),
                    "attributes": {"parcel_hint": "P1", "validated_on_site": False},
                }
            )
            objects.append({"class_name": "person", "confidence": 0.92, "bbox": _rect(360, 75, 120, 230), "attributes": {"operator_id": "OP1"}})
    return objects
