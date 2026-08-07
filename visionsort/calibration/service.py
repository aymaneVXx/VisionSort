from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import cv2
import numpy as np

from visionsort.calibration.models import (
    DEFAULT_WORLD_CONVENTION,
    CalibrationProfile,
    CalibrationStatus,
    CharucoBoardConfig,
)
from visionsort.calibration.opencv_adapter import (
    DetectedCharucoView,
    OpenCVCalibrationAdapter,
)


@dataclass(frozen=True, slots=True)
class CalibrationQualityThresholds:
    min_views: int = 6
    min_corners_per_view: int = 8
    min_total_corners: int = 60
    min_intrinsic_coverage: float = 0.08
    warning_intrinsic_rms_px: float = 1.0
    max_intrinsic_rms_px: float = 3.0
    outlier_sigma: float = 2.5
    max_view_error_px: float = 3.5
    min_homography_points: int = 4
    min_homography_image_coverage: float = 0.02
    min_homography_world_coverage_m2: float = 0.0001
    min_collinearity_ratio: float = 0.005
    min_inlier_ratio: float = 0.60
    ransac_world_threshold_m: float = 0.03
    warning_world_rmse_m: float = 0.015
    max_world_rmse_m: float = 0.06
    warning_reprojection_rmse_px: float = 2.0
    max_reprojection_rmse_px: float = 8.0
    max_world_error_m: float = 0.15
    max_homography_condition_number: float = 1.0e12
    ransac_confidence: float = 0.999
    ransac_max_iterations: int = 10000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Seuil de calibration invalide: {name}.")
        for name in (
            "min_intrinsic_coverage",
            "min_homography_image_coverage",
            "min_collinearity_ratio",
            "min_inlier_ratio",
            "ransac_confidence",
        ):
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"Seuil de calibration invalide: {name} > 1.")
        if self.min_homography_points < 4:
            raise ValueError("min_homography_points doit etre au moins 4.")
        if self.warning_intrinsic_rms_px > self.max_intrinsic_rms_px:
            raise ValueError("Seuils RMS intrinseques incoherents.")
        if self.warning_world_rmse_m > self.max_world_rmse_m:
            raise ValueError("Seuils RMSE monde incoherents.")
        if self.warning_reprojection_rmse_px > self.max_reprojection_rmse_px:
            raise ValueError("Seuils de reprojection incoherents.")

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any] | None
    ) -> "CalibrationQualityThresholds":
        values = dict(payload or {})
        known = cls.__dataclass_fields__
        defaults = cls()
        normalized: dict[str, Any] = {}
        for key in known:
            if key not in values:
                continue
            normalized[key] = (
                int(values[key])
                if isinstance(getattr(defaults, key), int)
                else float(values[key])
            )
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class IntrinsicCalibrationResult:
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    metrics: dict[str, Any]
    status: CalibrationStatus
    accepted_views: tuple[int, ...]
    rejected_views: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HomographyCalibrationResult:
    homography_image_undistorted_to_world: np.ndarray
    homography_world_to_image_undistorted: np.ndarray
    raw_image_points: np.ndarray
    undistorted_image_points: np.ndarray
    world_points: np.ndarray
    inlier_mask: np.ndarray
    metrics: dict[str, Any]
    status: CalibrationStatus


def _coverage(points: np.ndarray, image_size: tuple[int, int]) -> float:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(values) < 3:
        return 0.0
    hull = cv2.convexHull(values)
    area = float(abs(cv2.contourArea(hull)))
    return area / max(1.0, float(image_size[0] * image_size[1]))


def _world_area(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(values) < 3:
        return 0.0
    return float(abs(cv2.contourArea(cv2.convexHull(values))))


def _collinearity_ratio(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(values) < 3:
        return 0.0
    singular = np.linalg.svd(values - values.mean(axis=0), compute_uv=False)
    if len(singular) < 2 or singular[0] <= 1.0e-12:
        return 0.0
    return float(singular[1] / singular[0])


class CalibrationService:
    def __init__(
        self,
        thresholds: CalibrationQualityThresholds | dict[str, Any] | None = None,
        adapter: OpenCVCalibrationAdapter | None = None,
    ) -> None:
        self.thresholds = (
            thresholds
            if isinstance(thresholds, CalibrationQualityThresholds)
            else CalibrationQualityThresholds.from_dict(thresholds)
        )
        self.adapter = adapter or OpenCVCalibrationAdapter()

    @classmethod
    def from_site_config(cls, site_config: dict[str, Any]) -> "CalibrationService":
        thresholds = site_config.get("calibration_quality_thresholds") or {}
        return cls(thresholds=thresholds)

    def generate_charuco_board(
        self,
        config: CharucoBoardConfig,
        *,
        pixels_per_meter: int = 5000,
        margin_pixels: int = 40,
    ) -> np.ndarray:
        return self.adapter.generate_board_image(
            config,
            pixels_per_meter=pixels_per_meter,
            margin_pixels=margin_pixels,
        )

    def calibrate_intrinsics(
        self,
        images: Sequence[np.ndarray],
        board_config: CharucoBoardConfig,
    ) -> IntrinsicCalibrationResult:
        if not images:
            raise ValueError("Aucune image de calibration fournie.")
        image_size: tuple[int, int] | None = None
        detections: list[tuple[int, DetectedCharucoView]] = []
        for index, image in enumerate(images):
            if image is None or image.size == 0:
                continue
            current_size = (int(image.shape[1]), int(image.shape[0]))
            if image_size is None:
                image_size = current_size
            elif current_size != image_size:
                raise ValueError(
                    "Toutes les vues intrinseques doivent avoir la meme resolution."
                )
            detected = self.adapter.detect_charuco(image, board_config)
            if detected is not None:
                detections.append((index, detected))
        if image_size is None:
            raise ValueError("Aucune image de calibration lisible.")
        return self.calibrate_intrinsics_from_views(
            [item.object_points for _, item in detections],
            [item.image_points for _, item in detections],
            image_size=image_size,
            original_indices=[index for index, _ in detections],
            total_input_views=len(images),
        )

    def calibrate_intrinsics_from_views(
        self,
        object_points: Sequence[np.ndarray],
        image_points: Sequence[np.ndarray],
        *,
        image_size: tuple[int, int],
        original_indices: Sequence[int] | None = None,
        total_input_views: int | None = None,
    ) -> IntrinsicCalibrationResult:
        if len(object_points) != len(image_points):
            raise ValueError("Les listes de points objet et image sont incoherentes.")
        indices = list(original_indices or range(len(image_points)))
        accepted: list[int] = []
        objects: list[np.ndarray] = []
        images: list[np.ndarray] = []
        coverages: list[float] = []
        rejected = set(range(int(total_input_views or len(image_points)))) - set(indices)
        for original_index, objects_view, image_view in zip(
            indices, object_points, image_points
        ):
            objects_array = np.asarray(objects_view, dtype=np.float32).reshape(-1, 3)
            image_array = np.asarray(image_view, dtype=np.float32).reshape(-1, 2)
            coverage = _coverage(image_array, image_size)
            if (
                len(objects_array) != len(image_array)
                or len(image_array) < self.thresholds.min_corners_per_view
                or coverage < self.thresholds.min_intrinsic_coverage
            ):
                rejected.add(int(original_index))
                continue
            accepted.append(int(original_index))
            objects.append(objects_array)
            images.append(image_array)
            coverages.append(coverage)
        if len(images) < self.thresholds.min_views:
            raise ValueError(
                "Calibration intrinseque insuffisante: "
                f"{len(images)} vues valides, minimum {self.thresholds.min_views}."
            )
        total_corners = sum(len(item) for item in images)
        if total_corners < self.thresholds.min_total_corners:
            raise ValueError(
                "Calibration intrinseque insuffisante: nombre total de corners trop faible."
            )
        calibrated = self.adapter.calibrate_camera(objects, images, image_size)
        rms, camera_matrix, distortion, _rvecs, _tvecs, errors = calibrated
        if not np.isfinite(errors).all():
            raise ValueError("Calibration intrinseque degeneree: erreurs non finies.")
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        robust_cutoff = median + self.thresholds.outlier_sigma * max(
            1.4826 * mad, 0.05
        )
        cutoff = min(float(self.thresholds.max_view_error_px), robust_cutoff)
        keep_positions = [index for index, error in enumerate(errors) if error <= cutoff]
        if len(keep_positions) >= self.thresholds.min_views and len(keep_positions) < len(images):
            rejected.update(
                accepted[index]
                for index in range(len(accepted))
                if index not in keep_positions
            )
            accepted = [accepted[index] for index in keep_positions]
            objects = [objects[index] for index in keep_positions]
            images = [images[index] for index in keep_positions]
            coverages = [coverages[index] for index in keep_positions]
            rms, camera_matrix, distortion, _rvecs, _tvecs, errors = (
                self.adapter.calibrate_camera(objects, images, image_size)
            )
        determinant = float(np.linalg.det(camera_matrix))
        condition = float(np.linalg.cond(camera_matrix))
        if (
            not math.isfinite(rms)
            or rms > self.thresholds.max_intrinsic_rms_px
            or not np.isfinite(camera_matrix).all()
            or not np.isfinite(distortion).all()
            or determinant <= 0
            or condition > 1.0e8
            or float(camera_matrix[0, 0]) <= 0
            or float(camera_matrix[1, 1]) <= 0
        ):
            raise ValueError("Calibration intrinseque mathematiquement invalide.")
        aggregate_coverage = _coverage(np.vstack(images), image_size)
        status = (
            CalibrationStatus.WARNING
            if rms > self.thresholds.warning_intrinsic_rms_px
            else CalibrationStatus.VALID
        )
        metrics = {
            "rms_reprojection_error_px": float(rms),
            "per_view_errors_px": [float(value) for value in errors],
            "valid_view_count": len(images),
            "input_view_count": int(total_input_views or len(image_points)),
            "total_corner_count": sum(len(item) for item in images),
            "per_view_corner_count": [len(item) for item in images],
            "per_view_coverage": [float(value) for value in coverages],
            "image_coverage": float(aggregate_coverage),
            "resolution": [int(image_size[0]), int(image_size[1])],
            "rejected_view_indices": sorted(int(value) for value in rejected),
            "camera_matrix_condition_number": condition,
            "validated_on_site": False,
        }
        return IntrinsicCalibrationResult(
            image_width=int(image_size[0]),
            image_height=int(image_size[1]),
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion,
            metrics=metrics,
            status=status,
            accepted_views=tuple(accepted),
            rejected_views=tuple(sorted(int(value) for value in rejected)),
        )

    def estimate_homography(
        self,
        intrinsic: IntrinsicCalibrationResult,
        raw_image_points: Sequence[Sequence[float]],
        world_points_m: Sequence[Sequence[float]],
    ) -> HomographyCalibrationResult:
        raw = np.asarray(raw_image_points, dtype=np.float64).reshape(-1, 2)
        world = np.asarray(world_points_m, dtype=np.float64).reshape(-1, 2)
        minimum = max(4, int(self.thresholds.min_homography_points))
        if len(raw) != len(world) or len(raw) < minimum:
            raise ValueError(
                f"Au moins {minimum} correspondances pixel/monde sont requises."
            )
        if not np.isfinite(raw).all() or not np.isfinite(world).all():
            raise ValueError("Les correspondances contiennent une valeur non finie.")
        undistorted = self.adapter.undistort_points(
            raw, intrinsic.camera_matrix, intrinsic.distortion_coefficients
        )
        image_size = (intrinsic.image_width, intrinsic.image_height)
        image_coverage = _coverage(undistorted, image_size)
        world_coverage = _world_area(world)
        image_collinearity = _collinearity_ratio(undistorted)
        world_collinearity = _collinearity_ratio(world)
        if (
            image_collinearity < self.thresholds.min_collinearity_ratio
            or world_collinearity < self.thresholds.min_collinearity_ratio
        ):
            raise ValueError("Homographie refusee: points colineaires ou quasi-colineaires.")
        if image_coverage < self.thresholds.min_homography_image_coverage:
            raise ValueError("Homographie refusee: couverture spatiale image insuffisante.")
        if world_coverage < self.thresholds.min_homography_world_coverage_m2:
            raise ValueError("Homographie refusee: couverture spatiale monde insuffisante.")
        matrix, mask = self.adapter.find_homography(
            undistorted,
            world,
            ransac_threshold=self.thresholds.ransac_world_threshold_m,
            confidence=self.thresholds.ransac_confidence,
            max_iterations=self.thresholds.ransac_max_iterations,
        )
        if matrix is None or mask is None:
            raise ValueError("Estimation robuste de l'homographie impossible.")
        inliers = np.asarray(mask, dtype=np.uint8).reshape(-1).astype(bool)
        inlier_count = int(inliers.sum())
        if inlier_count < 4:
            raise ValueError("Homographie refusee: moins de quatre inliers.")
        inlier_ratio = inlier_count / len(raw)
        if inlier_ratio < self.thresholds.min_inlier_ratio:
            raise ValueError("Homographie refusee: ratio d'inliers insuffisant.")
        refined, _ = cv2.findHomography(undistorted[inliers], world[inliers], method=0)
        if refined is not None:
            matrix = refined
        determinant = float(np.linalg.det(matrix))
        condition = float(np.linalg.cond(matrix))
        if (
            not np.isfinite(matrix).all()
            or abs(determinant) <= 1.0e-14
            or not math.isfinite(condition)
            or condition > self.thresholds.max_homography_condition_number
        ):
            raise ValueError("Homographie refusee: matrice singuliere ou mal conditionnee.")
        inverse = np.linalg.inv(matrix)
        predicted_world = self.adapter.perspective_transform(undistorted, matrix)
        world_errors = np.linalg.norm(predicted_world - world, axis=1)
        predicted_undistorted = self.adapter.perspective_transform(world, inverse)
        predicted_raw = self.adapter.distort_points(
            predicted_undistorted,
            intrinsic.camera_matrix,
            intrinsic.distortion_coefficients,
        )
        pixel_errors = np.linalg.norm(predicted_raw - raw, axis=1)
        world_rmse = float(np.sqrt(np.mean(np.square(world_errors[inliers]))))
        pixel_rmse = float(np.sqrt(np.mean(np.square(pixel_errors[inliers]))))
        max_world_error = float(np.max(world_errors[inliers]))
        if (
            world_rmse > self.thresholds.max_world_rmse_m
            or pixel_rmse > self.thresholds.max_reprojection_rmse_px
            or max_world_error > self.thresholds.max_world_error_m
        ):
            raise ValueError("Homographie refusee: erreurs de reprojection excessives.")
        status = (
            CalibrationStatus.WARNING
            if (
                world_rmse > self.thresholds.warning_world_rmse_m
                or pixel_rmse > self.thresholds.warning_reprojection_rmse_px
            )
            else CalibrationStatus.VALID
        )
        metrics = {
            "inlier_ratio": float(inlier_ratio),
            "world_rmse_m": world_rmse,
            "reprojection_rmse_px": pixel_rmse,
            "max_world_error_m": max_world_error,
            "point_count": len(raw),
            "inlier_count": inlier_count,
            "image_coverage": float(image_coverage),
            "world_coverage_m2": float(world_coverage),
            "image_collinearity_ratio": image_collinearity,
            "world_collinearity_ratio": world_collinearity,
            "condition_number": condition,
            "determinant": determinant,
            "inlier_mask": [bool(value) for value in inliers],
            "world_bounds_m": {
                "x_min": float(world[:, 0].min()),
                "x_max": float(world[:, 0].max()),
                "y_min": float(world[:, 1].min()),
                "y_max": float(world[:, 1].max()),
            },
            "validated_on_site": False,
        }
        return HomographyCalibrationResult(
            homography_image_undistorted_to_world=np.asarray(matrix, dtype=np.float64),
            homography_world_to_image_undistorted=np.asarray(inverse, dtype=np.float64),
            raw_image_points=raw,
            undistorted_image_points=undistorted,
            world_points=world,
            inlier_mask=inliers,
            metrics=metrics,
            status=status,
        )

    def charuco_plane_correspondences(
        self,
        image: np.ndarray,
        board_config: CharucoBoardConfig,
        *,
        world_origin_m: tuple[float, float] = (0.0, 0.0),
        world_rotation_degrees: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        detected = self.adapter.detect_charuco(image, board_config)
        if detected is None:
            raise ValueError("Mire ChArUco non detectee sur le plan convoyeur.")
        local = detected.object_points[:, :2].astype(np.float64)
        angle = math.radians(float(world_rotation_degrees))
        rotation = np.asarray(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=np.float64,
        )
        world = local @ rotation.T + np.asarray(world_origin_m, dtype=np.float64)
        return detected.image_points.astype(np.float64), world

    def build_profile(
        self,
        *,
        source_id: str,
        version: int,
        intrinsic: IntrinsicCalibrationResult,
        homography: HomographyCalibrationResult,
        board_config: CharucoBoardConfig,
        optical_configuration: dict[str, Any] | None = None,
        world_coordinate_convention: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> CalibrationProfile:
        status = (
            CalibrationStatus.VALID
            if intrinsic.status is CalibrationStatus.VALID
            and homography.status is CalibrationStatus.VALID
            else CalibrationStatus.WARNING
        )
        profile_id = (
            f"cal-{source_id}-{int(version):04d}-{uuid.uuid4().hex[:8]}"
        )
        metrics = {
            "intrinsic": intrinsic.metrics,
            "homography": homography.metrics,
            "thresholds": self.thresholds.to_dict(),
            "validated_on_site": False,
        }
        return CalibrationProfile.create(
            profile_id=profile_id,
            source_id=source_id,
            version=int(version),
            image_width=intrinsic.image_width,
            image_height=intrinsic.image_height,
            camera_matrix=intrinsic.camera_matrix,
            distortion_coefficients=intrinsic.distortion_coefficients,
            homography_image_undistorted_to_world=(
                homography.homography_image_undistorted_to_world
            ),
            homography_world_to_image_undistorted=(
                homography.homography_world_to_image_undistorted
            ),
            board_config=board_config.to_dict(),
            optical_configuration=optical_configuration,
            quality_metrics=metrics,
            status=status,
            created_at=created_at
            or datetime.now(timezone.utc).isoformat(),
            world_coordinate_convention=(
                world_coordinate_convention or DEFAULT_WORLD_CONVENTION
            ),
        )
