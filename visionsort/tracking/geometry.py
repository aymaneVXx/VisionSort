from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from visionsort.calibration.geometry import WorldGeometry


@dataclass(frozen=True, slots=True)
class GroundAnchor:
    pixel: tuple[float, float]
    normalized: tuple[float, float]
    world_m: tuple[float, float] | None
    world_frame_id: str | None
    method: str
    world_valid: bool


class GroundAnchorEstimator:
    """Estimate a parcel support point without inventing physical coordinates."""

    MASK_LOWER_BAND = "MASK_LOWER_BAND"
    BBOX_BOTTOM_CENTER = "BBOX_BOTTOM_CENTER"

    def estimate(
        self,
        *,
        bbox: Sequence[float],
        mask: Sequence[Sequence[float]] | None,
        image_size: tuple[int, int],
        world_geometry: WorldGeometry | None = None,
    ) -> GroundAnchor:
        width, height = int(image_size[0]), int(image_size[1])
        if width <= 0 or height <= 0:
            raise ValueError("image_size doit contenir une largeur et une hauteur positives.")

        pixel, method = self._pixel_anchor(bbox=bbox, mask=mask)
        normalized = (
            float(np.clip(pixel[0] / width, 0.0, 1.0)),
            float(np.clip(pixel[1] / height, 0.0, 1.0)),
        )
        world_m: tuple[float, float] | None = None
        world_frame_id: str | None = None
        if world_geometry is not None:
            try:
                candidate = world_geometry.image_to_world(pixel, image_size=(width, height))
                if np.isfinite(candidate).all():
                    world_m = (float(candidate[0]), float(candidate[1]))
                    world_frame_id = getattr(world_geometry, "world_frame_id", None)
            except (RuntimeError, ValueError, TypeError):
                world_m = None
        return GroundAnchor(
            pixel=pixel,
            normalized=normalized,
            world_m=world_m,
            world_frame_id=world_frame_id,
            method=method,
            world_valid=world_m is not None,
        )

    @classmethod
    def _pixel_anchor(
        cls,
        *,
        bbox: Sequence[float],
        mask: Sequence[Sequence[float]] | None,
    ) -> tuple[tuple[float, float], str]:
        if mask:
            points = np.asarray(mask, dtype=np.float64).reshape(-1, 2)
            points = points[np.isfinite(points).all(axis=1)]
            if len(points) >= 3:
                lower_threshold = float(np.percentile(points[:, 1], 60.0))
                lower_band = points[points[:, 1] >= lower_threshold]
                if len(lower_band):
                    return (
                        (
                            float(np.median(lower_band[:, 0])),
                            float(np.percentile(points[:, 1], 80.0)),
                        ),
                        cls.MASK_LOWER_BAND,
                    )

        x1, _y1, x2, y2 = (float(value) for value in bbox)
        return ((x1 + x2) / 2.0, y2), cls.BBOX_BOTTOM_CENTER
