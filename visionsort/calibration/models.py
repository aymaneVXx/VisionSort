from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class CalibrationStatus(str, Enum):
    VALID = "VALID"
    WARNING = "WARNING"
    INVALID = "INVALID"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _matrix3(value: Iterable[Iterable[float]], name: str) -> tuple[tuple[float, ...], ...]:
    matrix = tuple(tuple(float(item) for item in row) for row in value)
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError(f"{name} doit etre une matrice 3x3.")
    if not all(math.isfinite(item) for row in matrix for item in row):
        raise ValueError(f"{name} contient une valeur non finie.")
    return matrix


@dataclass(frozen=True, slots=True)
class CharucoBoardConfig:
    dictionary: str = "DICT_4X4_50"
    columns: int = 5
    rows: int = 7
    square_length_m: float = 0.04
    marker_length_m: float = 0.02
    legacy_pattern: bool = False

    def __post_init__(self) -> None:
        if self.columns < 3 or self.rows < 3:
            raise ValueError("La mire ChArUco requiert au moins 3 lignes et 3 colonnes.")
        if self.square_length_m <= 0 or self.marker_length_m <= 0:
            raise ValueError("Les dimensions physiques de la mire doivent etre positives.")
        if self.marker_length_m >= self.square_length_m:
            raise ValueError("marker_length_m doit etre inferieur a square_length_m.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "charuco",
            "dictionary": self.dictionary,
            "columns": int(self.columns),
            "rows": int(self.rows),
            "square_length_m": float(self.square_length_m),
            "marker_length_m": float(self.marker_length_m),
            "legacy_pattern": bool(self.legacy_pattern),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CharucoBoardConfig":
        return cls(
            dictionary=str(payload.get("dictionary") or "DICT_4X4_50"),
            columns=int(payload.get("columns") or 5),
            rows=int(payload.get("rows") or 7),
            square_length_m=float(payload.get("square_length_m") or 0.04),
            marker_length_m=float(payload.get("marker_length_m") or 0.02),
            legacy_pattern=bool(payload.get("legacy_pattern", False)),
        )


DEFAULT_WORLD_CONVENTION = {
    "unit": "m",
    "x_axis": "conveyor_longitudinal",
    "y_axis": "conveyor_transverse",
    "z_axis": "up",
    "conveyor_plane_z_m": 0.0,
}


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    profile_id: str
    source_id: str
    version: int
    image_width: int
    image_height: int
    camera_matrix: tuple[tuple[float, ...], ...]
    distortion_coefficients: tuple[float, ...]
    homography_image_undistorted_to_world: tuple[tuple[float, ...], ...]
    homography_world_to_image_undistorted: tuple[tuple[float, ...], ...]
    world_coordinate_convention_json: str
    board_config_json: str
    optical_configuration_json: str
    quality_metrics_json: str
    status: CalibrationStatus
    created_at: str
    fingerprint_sha256: str = ""
    validated_on_site: bool = False

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or not self.source_id.strip():
            raise ValueError("profile_id et source_id sont obligatoires.")
        if self.version < 1 or self.image_width < 1 or self.image_height < 1:
            raise ValueError("Version et resolution de calibration invalides.")
        object.__setattr__(
            self,
            "camera_matrix",
            _matrix3(self.camera_matrix, "camera_matrix"),
        )
        object.__setattr__(
            self,
            "homography_image_undistorted_to_world",
            _matrix3(
                self.homography_image_undistorted_to_world,
                "homography_image_undistorted_to_world",
            ),
        )
        object.__setattr__(
            self,
            "homography_world_to_image_undistorted",
            _matrix3(
                self.homography_world_to_image_undistorted,
                "homography_world_to_image_undistorted",
            ),
        )
        distortion = tuple(float(value) for value in self.distortion_coefficients)
        if not distortion or not all(math.isfinite(value) for value in distortion):
            raise ValueError("Les coefficients de distorsion sont invalides.")
        object.__setattr__(self, "distortion_coefficients", distortion)
        status = (
            self.status
            if isinstance(self.status, CalibrationStatus)
            else CalibrationStatus(str(self.status))
        )
        object.__setattr__(self, "status", status)
        for field_name in (
            "world_coordinate_convention_json",
            "board_config_json",
            "optical_configuration_json",
            "quality_metrics_json",
        ):
            parsed = json.loads(getattr(self, field_name) or "{}")
            object.__setattr__(self, field_name, canonical_json(parsed))
        expected = hashlib.sha256(
            canonical_json(self._fingerprint_dict()).encode("utf-8")
        ).hexdigest()
        if self.fingerprint_sha256 and self.fingerprint_sha256 != expected:
            raise ValueError("Fingerprint du profil de calibration incoherent.")
        object.__setattr__(self, "fingerprint_sha256", expected)
        if self.validated_on_site:
            raise ValueError(
                "Un profil cree par cette application ne peut pas etre marque VALIDATED_ON_SITE."
            )

    @classmethod
    def create(
        cls,
        *,
        profile_id: str,
        source_id: str,
        version: int,
        image_width: int,
        image_height: int,
        camera_matrix: Iterable[Iterable[float]],
        distortion_coefficients: Iterable[float],
        homography_image_undistorted_to_world: Iterable[Iterable[float]],
        homography_world_to_image_undistorted: Iterable[Iterable[float]],
        board_config: dict[str, Any],
        optical_configuration: dict[str, Any] | None,
        quality_metrics: dict[str, Any],
        status: CalibrationStatus | str,
        created_at: str,
        world_coordinate_convention: dict[str, Any] | None = None,
    ) -> "CalibrationProfile":
        status_value = (
            status
            if isinstance(status, CalibrationStatus)
            else CalibrationStatus(str(status))
        )
        return cls(
            profile_id=profile_id,
            source_id=source_id,
            version=version,
            image_width=image_width,
            image_height=image_height,
            camera_matrix=_matrix3(camera_matrix, "camera_matrix"),
            distortion_coefficients=tuple(float(value) for value in distortion_coefficients),
            homography_image_undistorted_to_world=_matrix3(
                homography_image_undistorted_to_world,
                "homography_image_undistorted_to_world",
            ),
            homography_world_to_image_undistorted=_matrix3(
                homography_world_to_image_undistorted,
                "homography_world_to_image_undistorted",
            ),
            world_coordinate_convention_json=canonical_json(
                world_coordinate_convention or DEFAULT_WORLD_CONVENTION
            ),
            board_config_json=canonical_json(board_config),
            optical_configuration_json=canonical_json(
                optical_configuration or {}
            ),
            quality_metrics_json=canonical_json(quality_metrics),
            status=status_value,
            created_at=str(created_at),
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "source_id": self.source_id,
            "version": int(self.version),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "camera_matrix": [list(row) for row in self.camera_matrix],
            "distortion_coefficients": list(self.distortion_coefficients),
            "homography_image_undistorted_to_world": [
                list(row) for row in self.homography_image_undistorted_to_world
            ],
            "homography_world_to_image_undistorted": [
                list(row) for row in self.homography_world_to_image_undistorted
            ],
            "world_coordinate_convention": json.loads(
                self.world_coordinate_convention_json
            ),
            "board_config": json.loads(self.board_config_json),
            "optical_configuration": json.loads(
                self.optical_configuration_json
            ),
            "quality_metrics": json.loads(self.quality_metrics_json),
            "status": self.status.value,
            "created_at": self.created_at,
            "validated_on_site": False,
        }

    def _fingerprint_dict(self) -> dict[str, Any]:
        content = self._content_dict()
        for key in ("profile_id", "version", "created_at"):
            content.pop(key, None)
        return content

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "fingerprint_sha256": self.fingerprint_sha256}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibrationProfile":
        return cls(
            profile_id=str(payload["profile_id"]),
            source_id=str(payload["source_id"]),
            version=int(payload["version"]),
            image_width=int(payload["image_width"]),
            image_height=int(payload["image_height"]),
            camera_matrix=_matrix3(payload["camera_matrix"], "camera_matrix"),
            distortion_coefficients=tuple(
                float(value) for value in payload["distortion_coefficients"]
            ),
            homography_image_undistorted_to_world=_matrix3(
                payload["homography_image_undistorted_to_world"],
                "homography_image_undistorted_to_world",
            ),
            homography_world_to_image_undistorted=_matrix3(
                payload["homography_world_to_image_undistorted"],
                "homography_world_to_image_undistorted",
            ),
            world_coordinate_convention_json=canonical_json(
                payload["world_coordinate_convention"]
            ),
            board_config_json=canonical_json(payload.get("board_config", {})),
            optical_configuration_json=canonical_json(
                payload.get("optical_configuration", {})
            ),
            quality_metrics_json=canonical_json(payload.get("quality_metrics", {})),
            status=CalibrationStatus(str(payload["status"])),
            created_at=str(payload["created_at"]),
            fingerprint_sha256=str(payload.get("fingerprint_sha256") or ""),
            validated_on_site=bool(payload.get("validated_on_site", False)),
        )

    @property
    def board_config(self) -> dict[str, Any]:
        return json.loads(self.board_config_json)

    @property
    def quality_metrics(self) -> dict[str, Any]:
        return json.loads(self.quality_metrics_json)

    @property
    def optical_configuration(self) -> dict[str, Any]:
        return json.loads(self.optical_configuration_json)

    @property
    def world_coordinate_convention(self) -> dict[str, Any]:
        return json.loads(self.world_coordinate_convention_json)
