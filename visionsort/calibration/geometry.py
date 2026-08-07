from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from visionsort.calibration.models import CalibrationProfile, CalibrationStatus
from visionsort.calibration.models import canonical_json
from visionsort.calibration.opencv_adapter import OpenCVCalibrationAdapter


class CalibrationResolutionError(RuntimeError):
    pass


def _check_profile(
    profile: CalibrationProfile, image_size: tuple[int, int]
) -> None:
    actual = (int(image_size[0]), int(image_size[1]))
    expected = (int(profile.image_width), int(profile.image_height))
    if actual != expected:
        raise CalibrationResolutionError(
            "Calibration incompatible avec la resolution runtime: "
            f"profil={expected[0]}x{expected[1]}, runtime={actual[0]}x{actual[1]}."
        )
    if profile.status is not CalibrationStatus.VALID:
        raise RuntimeError(
            f"Le profil {profile.profile_id} n'est pas applicable: status={profile.status.value}."
        )


def runtime_calibration_diagnostic(
    profile: CalibrationProfile | None,
    image_size: tuple[int, int],
    *,
    source_id: str | None = None,
    optical_configuration: dict[str, object] | None = None,
) -> dict[str, object]:
    if profile is None:
        return {
            "status": "NO_PROFILE",
            "applicable": False,
            "message": "Aucun profil de calibration n'est associe a cette source.",
        }
    if source_id is not None and str(source_id) != profile.source_id:
        return {
            "status": "WRONG_SOURCE",
            "applicable": False,
            "profile_id": profile.profile_id,
            "message": (
                f"Profil associe a {profile.source_id}, source runtime={source_id}."
            ),
        }
    expected_optics = profile.optical_configuration
    if expected_optics:
        if optical_configuration is None:
            return {
                "status": "OPTICAL_CONFIG_UNKNOWN",
                "applicable": False,
                "profile_id": profile.profile_id,
                "message": "Configuration optique runtime non fournie.",
            }
        if canonical_json(expected_optics) != canonical_json(optical_configuration):
            return {
                "status": "INCOMPATIBLE_OPTICAL_CONFIG",
                "applicable": False,
                "profile_id": profile.profile_id,
                "message": "La configuration optique/runtime differe du profil.",
            }
    try:
        _check_profile(profile, image_size)
    except Exception as exc:
        return {
            "status": (
                "INCOMPATIBLE_RESOLUTION"
                if isinstance(exc, CalibrationResolutionError)
                else "PROFILE_NOT_VALID"
            ),
            "applicable": False,
            "profile_id": profile.profile_id,
            "fingerprint_sha256": profile.fingerprint_sha256,
            "message": str(exc),
        }
    return {
        "status": "READY",
        "applicable": True,
        "profile_id": profile.profile_id,
        "fingerprint_sha256": profile.fingerprint_sha256,
        "resolution": [profile.image_width, profile.image_height],
        "message": "Profil valide pour la resolution runtime.",
    }


def _points(values: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("Une liste de points 2D finis est requise.")
    return points


def image_points_to_world(
    profile: CalibrationProfile,
    points: Sequence[Sequence[float]],
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    _check_profile(profile, image_size)
    raw = _points(points)
    undistorted = OpenCVCalibrationAdapter.undistort_points(
        raw,
        np.asarray(profile.camera_matrix),
        np.asarray(profile.distortion_coefficients),
    )
    return OpenCVCalibrationAdapter.perspective_transform(
        undistorted,
        np.asarray(profile.homography_image_undistorted_to_world),
    )


def world_points_to_image(
    profile: CalibrationProfile,
    points: Sequence[Sequence[float]],
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    _check_profile(profile, image_size)
    world = _points(points)
    undistorted_pixels = OpenCVCalibrationAdapter.perspective_transform(
        world,
        np.asarray(profile.homography_world_to_image_undistorted),
    )
    return OpenCVCalibrationAdapter.distort_points(
        undistorted_pixels,
        np.asarray(profile.camera_matrix),
        np.asarray(profile.distortion_coefficients),
    )


def undistort_point(
    profile: CalibrationProfile,
    point: Sequence[float],
    *,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    _check_profile(profile, image_size)
    result = OpenCVCalibrationAdapter.undistort_points(
        _points([point]),
        np.asarray(profile.camera_matrix),
        np.asarray(profile.distortion_coefficients),
    )[0]
    return float(result[0]), float(result[1])


def image_to_world(
    profile: CalibrationProfile,
    point: Sequence[float],
    *,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    result = image_points_to_world(profile, [point], image_size=image_size)[0]
    return float(result[0]), float(result[1])


def world_to_image(
    profile: CalibrationProfile,
    point: Sequence[float],
    *,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    result = world_points_to_image(profile, [point], image_size=image_size)[0]
    return float(result[0]), float(result[1])


@dataclass(frozen=True, slots=True)
class WorldGeometry:
    """Stable application-facing facade for physical camera geometry."""

    profile: CalibrationProfile

    def undistort_point(
        self, point: Sequence[float], *, image_size: tuple[int, int]
    ) -> tuple[float, float]:
        return undistort_point(self.profile, point, image_size=image_size)

    def image_to_world(
        self, point: Sequence[float], *, image_size: tuple[int, int]
    ) -> tuple[float, float]:
        return image_to_world(self.profile, point, image_size=image_size)

    def world_to_image(
        self, point: Sequence[float], *, image_size: tuple[int, int]
    ) -> tuple[float, float]:
        return world_to_image(self.profile, point, image_size=image_size)

    def image_points_to_world(
        self,
        points: Sequence[Sequence[float]],
        *,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        return image_points_to_world(self.profile, points, image_size=image_size)

    def world_points_to_image(
        self,
        points: Sequence[Sequence[float]],
        *,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        return world_points_to_image(self.profile, points, image_size=image_size)
