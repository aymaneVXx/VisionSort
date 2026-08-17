from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from visionsort.core.types import (
    TrackObservation,
    Tracklet,
    TrackletAppearanceDescriptor,
)
from visionsort.reid.encoder import ParcelReIDEncoder, l2_normalize


@dataclass(slots=True)
class _View:
    frame_index: int
    quality: float
    crop: np.ndarray
    used_mask: bool


class HandoffKeyframeSelector:
    """Retain only a bounded set of high-quality crops while frames still exist."""

    def __init__(
        self,
        encoder: ParcelReIDEncoder,
        *,
        max_views: int = 5,
        min_views: int = 3,
        priority_zone_ids: set[str] | None = None,
    ) -> None:
        self.encoder = encoder
        self.max_views = max(1, int(max_views))
        self.min_views = max(1, min(int(min_views), self.max_views))
        self.priority_zone_ids = set(priority_zone_ids or set())
        self._views: dict[int, list[_View]] = {}

    def observe(self, tracks: list[TrackObservation], image: np.ndarray) -> None:
        height, width = image.shape[:2]
        for track in tracks:
            if track.class_name != "parcel" or track.identity_status == "AMBIGUOUS":
                continue
            current = self._views.setdefault(track.local_track_id, [])
            if current and track.frame_index - max(
                item.frame_index for item in current
            ) < 2:
                continue
            crop, used_mask, geometry_quality = self._crop(track, image)
            if crop is None:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness_quality = min(1.0, sharpness / 180.0)
            area_ratio = (
                max(0.0, track.bbox[2] - track.bbox[0])
                * max(0.0, track.bbox[3] - track.bbox[1])
                / max(1.0, float(width * height))
            )
            area_quality = min(1.0, area_ratio / 0.04)
            zone_bonus = (
                1.0
                if track.zone_id in self.priority_zone_ids
                else 0.65
                if track.zone_id
                else 0.40
            )
            integrity_quality = 1.0 if track.identity_status == "STABLE" else 0.55
            quality = (
                0.32 * max(0.0, min(1.0, track.confidence))
                + 0.18 * area_quality
                + 0.22 * sharpness_quality
                + 0.12 * geometry_quality
                + 0.10 * zone_bonus
                + 0.06 * integrity_quality
            )
            if quality < 0.30:
                continue
            current.append(
                _View(
                    frame_index=track.frame_index,
                    quality=float(quality),
                    crop=crop,
                    used_mask=used_mask,
                )
            )
            current.sort(key=lambda item: (-item.quality, item.frame_index))
            del current[self.max_views :]

    def attach(self, tracklet: Tracklet) -> Tracklet:
        if tracklet.class_name != "parcel":
            return tracklet
        views = self._views.pop(tracklet.local_track_id, [])
        if not views:
            return tracklet
        selected = sorted(views, key=lambda item: (-item.quality, item.frame_index))[
            : self.max_views
        ]
        embeddings = self.encoder.encode(item.crop for item in selected)
        if embeddings.size == 0:
            return tracklet
        aggregate = l2_normalize(np.median(embeddings, axis=0)).reshape(-1)
        descriptor = TrackletAppearanceDescriptor(
            embeddings=embeddings.astype(float).tolist(),
            aggregate_embedding=aggregate.astype(float).tolist(),
            view_count=len(selected),
            view_qualities=[round(item.quality, 6) for item in selected],
            model_version=self.encoder.model_version,
            used_mask=[item.used_mask for item in selected],
        )
        tracklet.summary_json["appearance_descriptor"] = descriptor.to_json()
        tracklet.summary_json["reid_keyframe_policy"] = {
            "requested_views": [self.min_views, self.max_views],
            "available_views": len(selected),
        }
        return tracklet

    @staticmethod
    def _crop(
        track: TrackObservation, image: np.ndarray
    ) -> tuple[np.ndarray | None, bool, float]:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in track.bbox)
        box_w, box_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        padding = 0.06 * max(box_w, box_h)
        left = max(0, int(np.floor(x1 - padding)))
        top = max(0, int(np.floor(y1 - padding)))
        right = min(width, int(np.ceil(x2 + padding)))
        bottom = min(height, int(np.ceil(y2 + padding)))
        if right - left < 12 or bottom - top < 12:
            return None, False, 0.0
        clipped_edges = sum(
            (
                left == 0,
                top == 0,
                right == width,
                bottom == height,
            )
        )
        geometry_quality = max(0.0, 1.0 - 0.22 * clipped_edges)
        crop = image[top:bottom, left:right].copy()
        mask_value = track.extra.get("mask")
        if not isinstance(mask_value, list) or len(mask_value) < 3:
            return crop, False, geometry_quality
        points = np.asarray(mask_value, dtype=np.float32).reshape(-1, 2)
        if points.size == 0:
            return crop, False, geometry_quality
        if float(np.max(points)) <= 1.5:
            points[:, 0] *= width
            points[:, 1] *= height
        points[:, 0] -= left
        points[:, 1] -= top
        binary = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(binary, [np.rint(points).astype(np.int32)], 255)
        if int(np.count_nonzero(binary)) < 25:
            return crop, False, geometry_quality
        background = np.full_like(crop, 114)
        masked = np.where(binary[..., None] > 0, crop, background)
        return masked, True, geometry_quality
