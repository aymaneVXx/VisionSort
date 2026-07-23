from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from visionsort.core.enums import AnnotationStatus
from visionsort.tracking.engine import bbox_iou
from visionsort.annotations.validators import validate_pose_detections


class QualityGate:
    """Stateful, multi-signal quality gate evaluated in source timestamp order."""

    def __init__(self):
        self.previous_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def assess(
        self,
        *,
        source_id: str,
        detections: list[dict[str, Any]],
        image_shape: tuple[int, int],
        task: str,
    ) -> tuple[str, dict[str, Any]]:
        height, width = image_shape
        confidences = [float(item.get("confidence", 0.0)) for item in detections]
        boxes = [tuple(float(value) for value in item["bbox"]) for item in detections]
        max_iou = 0.0
        for left in range(len(boxes)):
            for right in range(left + 1, len(boxes)):
                max_iou = max(max_iou, bbox_iou(boxes[left], boxes[right]))

        plausible = 0
        truncated = 0
        invalid = 0
        for x1, y1, x2, y2 in boxes:
            normalized_area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(
                float(width * height), 1.0
            )
            if 0.0002 <= normalized_area <= 0.90:
                plausible += 1
            else:
                invalid += 1
            if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
                truncated += 1

        previous = self.previous_by_source.get(source_id, [])
        previous_count = len(previous)
        count_variation = abs(len(detections) - previous_count) if previous else 0
        temporal_matches = 0
        if previous and detections:
            for detection in detections:
                if max(
                    (
                        bbox_iou(
                            tuple(float(value) for value in detection["bbox"]),
                            tuple(float(value) for value in candidate["bbox"]),
                        )
                        for candidate in previous
                        if candidate.get("class_name") == detection.get("class_name")
                    ),
                    default=0.0,
                ) >= 0.2:
                    temporal_matches += 1
        temporal_stability = (
            temporal_matches / max(len(detections), previous_count, 1)
            if previous
            else 1.0
        )
        tracker_consistency = sum(
            1.0
            for item in detections
            if item.get("local_track_id") is not None
            or item.get("attributes", {}).get("tracker_consistent", True)
        ) / max(len(detections), 1)
        model_agreement = sum(
            float(item.get("attributes", {}).get("model_agreement", 1.0))
            for item in detections
        ) / max(len(detections), 1)
        mask_scores: list[float] = []
        if task == "segmentation":
            for item in detections:
                mask = item.get("mask") or []
                if len(mask) < 3:
                    mask_scores.append(0.0)
                    continue
                perimeter = sum(
                    math.dist(mask[index - 1], mask[index]) for index in range(len(mask))
                )
                area = abs(
                    sum(
                        float(mask[index - 1][0]) * float(mask[index][1])
                        - float(mask[index][0]) * float(mask[index - 1][1])
                        for index in range(len(mask))
                    )
                ) / 2.0
                mask_scores.append(
                    1.0 if perimeter > 4.0 and area > 1.0 else 0.0
                )
        mask_quality = (
            sum(mask_scores) / len(mask_scores) if mask_scores else 1.0
        )
        pose_errors = (
            validate_pose_detections(
                detections, width=width, height=height
            )
            if task == "pose" and detections
            else []
        )
        stats = {
            "count": len(detections),
            "previous_count": previous_count,
            "avg_conf": sum(confidences) / max(len(confidences), 1),
            "min_conf": min(confidences, default=0.0),
            "max_iou": max_iou,
            "temporal_stability": temporal_stability,
            "tracker_consistency": tracker_consistency,
            "count_variation": count_variation,
            "probable_merge_or_split": bool(count_variation > 1 or max_iou > 0.80),
            "plausible_size_ratio": plausible / max(len(boxes), 1),
            "truncated_ratio": truncated / max(len(boxes), 1),
            "model_agreement": model_agreement,
            "mask_quality": mask_quality,
            "invalid_instances": invalid,
            "pose_errors": pose_errors,
        }
        self.previous_by_source[source_id] = [dict(item) for item in detections]

        if (
            invalid > 0
            or (
                task == "segmentation"
                and detections
                and mask_quality <= 0.0
            )
            or bool(pose_errors)
        ):
            return AnnotationStatus.REJECTED.value, stats
        review_reasons = [
            not detections,
            stats["avg_conf"] < 0.65,
            stats["min_conf"] < 0.35,
            stats["temporal_stability"] < 0.55,
            stats["tracker_consistency"] < 0.75,
            stats["probable_merge_or_split"],
            stats["plausible_size_ratio"] < 0.85,
            stats["truncated_ratio"] > 0.5,
            stats["model_agreement"] < 0.75,
            stats["mask_quality"] < 0.75,
        ]
        if any(review_reasons):
            return AnnotationStatus.NEEDS_REVIEW.value, stats
        return AnnotationStatus.AUTO_ACCEPTED.value, stats
