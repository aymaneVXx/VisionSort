from __future__ import annotations

import json
import sqlite3
from dataclasses import FrozenInstanceError

import cv2
import numpy as np
import pytest

from visionsort.calibration.geometry import (
    CalibrationResolutionError,
    image_points_to_world,
    image_to_world,
    runtime_calibration_diagnostic,
    world_points_to_image,
)
from visionsort.calibration.models import (
    DEFAULT_WORLD_CONVENTION,
    CalibrationProfile,
    CalibrationStatus,
    CharucoBoardConfig,
)
from visionsort.calibration.opencv_adapter import OpenCVCalibrationAdapter
from visionsort.calibration.repository import CalibrationRepository
from visionsort.calibration.service import (
    CalibrationQualityThresholds,
    CalibrationService,
    IntrinsicCalibrationResult,
)
from visionsort.core.site_config import validate_site_config
from visionsort.core.config import AppConfig, DEFAULT_CONFIG
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import ControlRepository
from visionsort.runtime.supervisor import RuntimeSupervisor


IMAGE_SIZE = (1280, 720)
CAMERA_MATRIX = np.asarray(
    [[900.0, 0.0, 640.0], [0.0, 880.0, 360.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
KNOWN_H = np.asarray(
    [
        [0.0020, 0.00008, -1.25],
        [-0.00005, 0.00155, -0.48],
        [0.0000008, -0.0000003, 1.0],
    ],
    dtype=np.float64,
)


def _profile(
    *,
    source_id: str = "source-c1",
    version: int = 1,
    profile_id: str | None = None,
    distortion: np.ndarray | None = None,
    homography: np.ndarray | None = None,
    optical_configuration: dict[str, str] | None = None,
    status: CalibrationStatus = CalibrationStatus.VALID,
    created_at: str = "2026-08-07T10:00:00+00:00",
    world_frame_id: str | None = None,
) -> CalibrationProfile:
    matrix = np.asarray(homography if homography is not None else KNOWN_H)
    return CalibrationProfile.create(
        profile_id=profile_id or f"cal-{source_id}-{version}",
        source_id=source_id,
        version=version,
        image_width=IMAGE_SIZE[0],
        image_height=IMAGE_SIZE[1],
        camera_matrix=CAMERA_MATRIX,
        distortion_coefficients=(
            distortion
            if distortion is not None
            else np.zeros(5, dtype=np.float64)
        ),
        homography_image_undistorted_to_world=matrix,
        homography_world_to_image_undistorted=np.linalg.inv(matrix),
        board_config=CharucoBoardConfig().to_dict(),
        optical_configuration=optical_configuration or {},
        quality_metrics={
            "intrinsic": {"rms_reprojection_error_px": 0.1},
            "homography": {"world_rmse_m": 0.001},
            "validated_on_site": False,
        },
        status=status,
        created_at=created_at,
        world_coordinate_convention={
            **DEFAULT_WORLD_CONVENTION,
            **({"frame_id": world_frame_id} if world_frame_id else {}),
        },
    )


def _intrinsic(
    distortion: np.ndarray | None = None,
) -> IntrinsicCalibrationResult:
    return IntrinsicCalibrationResult(
        image_width=IMAGE_SIZE[0],
        image_height=IMAGE_SIZE[1],
        camera_matrix=CAMERA_MATRIX.copy(),
        distortion_coefficients=(
            np.asarray(distortion, dtype=np.float64)
            if distortion is not None
            else np.zeros(5, dtype=np.float64)
        ),
        metrics={},
        status=CalibrationStatus.VALID,
        accepted_views=(),
        rejected_views=(),
    )


def _register_source(db: VisionSortDB, source_id: str = "source-c1") -> None:
    ControlRepository(db).upsert_source(
        {
            "id": source_id,
            "name": "Camera C1",
            "role": "C1",
            "source_type": "REPLAY",
            "uri": "fixture.mp4",
            "model_id": "demo_synth_det",
            "tracker_id": "greedy_iou",
            "enabled": True,
        }
    )


def test_new_calibrations_use_distinct_camera_frames_by_default():
    camera_c1 = _profile(source_id="C1")
    camera_c2 = _profile(source_id="C2")

    assert camera_c1.world_coordinate_convention["frame_id"] == "camera:C1"
    assert camera_c2.world_coordinate_convention["frame_id"] == "camera:C2"


def test_common_world_frame_requires_an_explicit_identifier():
    camera_c1 = _profile(source_id="C1", world_frame_id="site_world")
    camera_c2 = _profile(source_id="C2", world_frame_id="site_world")

    assert camera_c1.world_coordinate_convention["frame_id"] == "site_world"
    assert camera_c2.world_coordinate_convention["frame_id"] == "site_world"


def test_existing_shared_frame_profile_remains_readable():
    original = _profile(source_id="legacy-C1", world_frame_id="site_world")
    restored = CalibrationProfile.from_dict(original.to_dict())

    assert restored.world_coordinate_convention["frame_id"] == "site_world"
    assert restored.fingerprint_sha256 == original.fingerprint_sha256


def test_known_homography_round_trip_has_low_error():
    profile = _profile()
    raw_pixels = np.asarray(
        [[120.0, 100.0], [640.0, 360.0], [1120.0, 650.0], [800.0, 180.0]]
    )
    world = image_points_to_world(
        profile, raw_pixels, image_size=IMAGE_SIZE
    )
    reconstructed = world_points_to_image(
        profile, world, image_size=IMAGE_SIZE
    )
    assert np.max(np.linalg.norm(reconstructed - raw_pixels, axis=1)) < 1.0e-6


def test_ransac_recovers_homography_with_multiple_outliers():
    rng = np.random.default_rng(42)
    raw = np.column_stack(
        [rng.uniform(80, 1200, 40), rng.uniform(60, 680, 40)]
    )
    expected_world = OpenCVCalibrationAdapter.perspective_transform(raw, KNOWN_H)
    measured_world = expected_world + rng.normal(0.0, 0.0005, expected_world.shape)
    outlier_indices = np.asarray([1, 5, 11, 18, 24, 33])
    measured_world[outlier_indices] += np.asarray([0.45, -0.35])
    service = CalibrationService()

    result = service.estimate_homography(_intrinsic(), raw, measured_world)
    predicted = OpenCVCalibrationAdapter.perspective_transform(
        raw, result.homography_image_undistorted_to_world
    )
    good = np.ones(len(raw), dtype=bool)
    good[outlier_indices] = False

    assert result.metrics["inlier_count"] >= 32
    assert result.metrics["inlier_ratio"] >= 0.80
    assert np.sqrt(np.mean(np.square(predicted[good] - expected_world[good]))) < 0.002


def test_collinear_homography_points_are_rejected():
    raw = np.asarray([[100.0 + 100 * index, 150.0 + 20 * index] for index in range(8)])
    world = np.asarray([[0.2 * index, 0.04 * index] for index in range(8)])
    with pytest.raises(ValueError, match="colineaires"):
        CalibrationService().estimate_homography(_intrinsic(), raw, world)


def test_runtime_resolution_mismatch_is_explicitly_rejected():
    profile = _profile()
    with pytest.raises(CalibrationResolutionError, match="1280x720"):
        image_to_world(profile, (100.0, 100.0), image_size=(1920, 1080))
    diagnostic = runtime_calibration_diagnostic(profile, (1920, 1080))
    assert diagnostic["applicable"] is False
    assert diagnostic["status"] == "INCOMPATIBLE_RESOLUTION"


def test_runtime_optical_configuration_mismatch_is_explicit():
    expected = {
        "source_id": "source-c1",
        "source_type": "RTSP",
        "source_uri": "rtsp://camera/stream",
        "camera_role": "C1",
        "optical_setup_id": "lens-35mm-focus-2m",
    }
    profile = _profile(optical_configuration=expected)
    ready = runtime_calibration_diagnostic(
        profile,
        IMAGE_SIZE,
        source_id="source-c1",
        optical_configuration=expected,
    )
    assert ready["status"] == "READY"
    assert ready["applicable"] is True
    diagnostic = runtime_calibration_diagnostic(
        profile,
        IMAGE_SIZE,
        source_id="source-c1",
        optical_configuration={**expected, "optical_setup_id": "lens-zoomed"},
    )
    assert diagnostic["applicable"] is False
    assert diagnostic["status"] == "INCOMPATIBLE_OPTICAL_CONFIG"


def test_distortion_undistortion_then_homography_pipeline_is_correct():
    distortion = np.asarray([0.08, -0.04, 0.001, -0.0005, 0.01])
    profile = _profile(distortion=distortion)
    undistorted_pixels = np.asarray(
        [[180.0, 150.0], [600.0, 300.0], [1040.0, 590.0], [760.0, 220.0]]
    )
    raw_pixels = OpenCVCalibrationAdapter.distort_points(
        undistorted_pixels, CAMERA_MATRIX, distortion
    )
    expected_world = OpenCVCalibrationAdapter.perspective_transform(
        undistorted_pixels, KNOWN_H
    )

    actual_world = image_points_to_world(
        profile, raw_pixels, image_size=IMAGE_SIZE
    )
    reconstructed_raw = world_points_to_image(
        profile, actual_world, image_size=IMAGE_SIZE
    )

    assert np.max(np.linalg.norm(actual_world - expected_world, axis=1)) < 1.0e-7
    assert np.max(np.linalg.norm(reconstructed_raw - raw_pixels, axis=1)) < 1.0e-5


def test_profile_is_frozen_versioned_and_sqlite_immutable(tmp_path):
    db = VisionSortDB(tmp_path / "calibration.db")
    db.initialize()
    _register_source(db)
    repository = CalibrationRepository(db)
    first = _profile(version=1, profile_id="profile-v1")
    repository.save_profile(first)
    second = _profile(
        version=2,
        profile_id="profile-v2",
        homography=KNOWN_H
        + np.asarray([[0.0, 0.0, 0.05], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        created_at="2026-08-07T11:00:00+00:00",
    )
    repository.save_profile(second)

    assert repository.next_version("source-c1") == 3
    with pytest.raises(FrozenInstanceError):
        first.version = 9  # type: ignore[misc]
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.execute(
            "UPDATE calibration_profiles SET status = 'WARNING' WHERE id = ?",
            (first.profile_id,),
        )


def test_only_valid_profile_can_be_activated(tmp_path):
    db = VisionSortDB(tmp_path / "activation.db")
    db.initialize()
    _register_source(db)
    repository = CalibrationRepository(db)
    warning = _profile(
        profile_id="warning-profile",
        status=CalibrationStatus.WARNING,
    )
    repository.save_profile(warning)
    with pytest.raises(RuntimeError, match="VALID"):
        repository.activate_profile("source-c1", warning.profile_id)


def test_session_snapshot_keeps_old_profile_after_new_activation(tmp_path):
    db = VisionSortDB(tmp_path / "sessions.db")
    db.initialize()
    _register_source(db)
    calibration_repo = CalibrationRepository(db)
    control_repo = ControlRepository(db)
    first = _profile(version=1, profile_id="profile-v1")
    second = _profile(
        version=2,
        profile_id="profile-v2",
        homography=KNOWN_H
        + np.asarray([[0.0, 0.0, 0.05], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        created_at="2026-08-07T11:00:00+00:00",
    )
    calibration_repo.save_profile(first)
    calibration_repo.save_profile(second)
    calibration_repo.activate_profile("source-c1", first.profile_id)
    first_session = control_repo.create_capture_session(
        name="before",
        demo_mode=True,
        sources=[{"source_id": "source-c1", "camera_role": "C1"}],
        config={},
    )
    calibration_repo.activate_profile("source-c1", second.profile_id)
    second_session = control_repo.create_capture_session(
        name="after",
        demo_mode=True,
        sources=[{"source_id": "source-c1", "camera_role": "C1"}],
        config={},
    )

    old_snapshot = control_repo.list_capture_session_sources(first_session)[0]
    new_snapshot = control_repo.list_capture_session_sources(second_session)[0]
    assert old_snapshot["calibration_profile_id"] == first.profile_id
    assert old_snapshot["calibration_profile_hash"] == first.fingerprint_sha256
    assert json.loads(old_snapshot["calibration_profile_json"]) == first.to_dict()
    assert new_snapshot["calibration_profile_id"] == second.profile_id
    assert calibration_repo.get_active_profile("source-c1") == second
    old_geometry = calibration_repo.session_geometry(
        first_session, "source-c1"
    )
    assert old_geometry is not None
    assert old_geometry.profile.profile_id == first.profile_id
    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.db = db
    supervisor.db_path = db.db_path
    supervisor.control_repo = control_repo
    supervisor.base_config_values = DEFAULT_CONFIG
    supervisor.config = AppConfig(values=DEFAULT_CONFIG)
    supervisor.active_source_sessions = {}
    supervisor.session_site_configs = {}
    supervisor.camera_processes = {}
    supervisor.start_source = lambda source_id, **kwargs: None  # type: ignore[method-assign]
    supervisor.start_session(first_session)
    session_row = control_repo.get_capture_session(first_session)
    site_snapshot = json.loads(session_row["site_config_snapshot_json"])
    assert site_snapshot["calibration_profiles"]["active_by_source"] == {
        "source-c1": first.profile_id
    }


def test_legacy_site_config_still_works_without_world_conversion():
    legacy = {
        "tracking": {
            "zones": {
                "C1": [
                    {
                        "zone_id": "legacy-exit",
                        "kind": "exit",
                        "x1": 0.8,
                        "y1": 0.0,
                        "x2": 1.0,
                        "y2": 1.0,
                    }
                ]
            }
        }
    }
    validated = validate_site_config(legacy)
    zone = validated["tracking"]["zones"]["C1"][0]
    assert validated == legacy
    assert "polygon_world" not in zone


def test_polygon_world_is_validated_in_meters():
    valid = {
        "schema_version": 2,
        "world_coordinate_convention": DEFAULT_WORLD_CONVENTION,
        "tracking": {
            "zones": {
                "C3": [
                    {
                        "zone_id": "destination-a",
                        "kind": "destination",
                        "polygon_world": [[0.0, 0.0], [1.2, 0.0], [1.1, 0.8], [0.0, 0.7]],
                    }
                ]
            }
        },
    }
    assert validate_site_config(valid) == valid
    invalid = json.loads(json.dumps(valid))
    invalid["tracking"]["zones"]["C3"][0]["polygon_world"] = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ]
    with pytest.raises(RuntimeError, match="aire nulle"):
        validate_site_config(invalid)


def test_database_reload_preserves_exact_matrices_and_fingerprint(tmp_path):
    db_path = tmp_path / "reload.db"
    db = VisionSortDB(db_path)
    db.initialize()
    _register_source(db)
    profile = _profile()
    CalibrationRepository(db).save_profile(profile)

    reloaded_db = VisionSortDB(db_path)
    reloaded_db.initialize()
    reloaded = CalibrationRepository(reloaded_db).get_profile(profile.profile_id)

    assert reloaded is not None
    assert reloaded.to_dict() == profile.to_dict()
    assert reloaded.camera_matrix == profile.camera_matrix
    assert reloaded.fingerprint_sha256 == profile.fingerprint_sha256


def test_v10_database_migrates_to_calibration_schema_without_rebuild(tmp_path):
    db_path = tmp_path / "legacy-v10.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE sources (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL,
            source_type TEXT NOT NULL, uri TEXT NOT NULL,
            model_id TEXT NOT NULL, tracker_id TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE capture_session_sources (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            source_id TEXT NOT NULL, camera_role TEXT NOT NULL,
            time_offset_ms REAL NOT NULL DEFAULT 0, replay_fps REAL,
            source_type_snapshot TEXT, source_uri_snapshot TEXT,
            source_sha256 TEXT, archive_required INTEGER NOT NULL DEFAULT 0,
            model_pipeline_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE source_state (
            source_id TEXT PRIMARY KEY, status TEXT NOT NULL,
            fps REAL NOT NULL DEFAULT 0, last_error TEXT,
            last_frame_ts REAL, preview_path TEXT, details_path TEXT,
            recording_enabled INTEGER NOT NULL DEFAULT 0,
            metrics_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL
        );
        PRAGMA user_version = 10;
        """
    )
    connection.commit()
    connection.close()

    db = VisionSortDB(db_path)
    db.initialize()

    assert db.fetch_one("PRAGMA user_version")[0] == 13
    source_columns = {
        row["name"] for row in db.fetch_all("PRAGMA table_info(sources)")
    }
    session_source_columns = {
        row["name"]
        for row in db.fetch_all("PRAGMA table_info(capture_session_sources)")
    }
    state_columns = {
        row["name"] for row in db.fetch_all("PRAGMA table_info(source_state)")
    }
    assert "optical_setup_id" in source_columns
    assert {
        "optical_setup_id_snapshot",
        "calibration_profile_id",
        "calibration_profile_hash",
        "calibration_profile_json",
    } <= session_source_columns
    assert "calibration_frame_path" in state_columns
    assert db.fetch_one(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'calibration_profiles'"
    )


def test_synthetic_intrinsic_calibration_recovers_camera_parameters():
    adapter = OpenCVCalibrationAdapter()
    board = adapter.create_board(CharucoBoardConfig())
    objects = np.asarray(board.getChessboardCorners(), dtype=np.float32).reshape(-1, 3)
    rng = np.random.default_rng(7)
    object_views: list[np.ndarray] = []
    image_views: list[np.ndarray] = []
    distortion = np.asarray([0.02, -0.01, 0.0005, -0.0002, 0.002])
    for index in range(9):
        rvec = np.asarray(
            [0.10 + index * 0.012, -0.13 + index * 0.018, -0.08 + index * 0.02]
        )
        tvec = np.asarray(
            [-0.10 + index * 0.022, -0.12 + (index % 3) * 0.04, 0.72 + index * 0.025]
        )
        projected, _ = cv2.projectPoints(
            objects, rvec, tvec, CAMERA_MATRIX, distortion
        )
        image_points = projected.reshape(-1, 2) + rng.normal(
            0.0, 0.03, (len(objects), 2)
        )
        object_views.append(objects.copy())
        image_views.append(image_points.astype(np.float32))
    thresholds = CalibrationQualityThresholds(
        min_views=6,
        min_corners_per_view=8,
        min_total_corners=100,
        min_intrinsic_coverage=0.01,
        warning_intrinsic_rms_px=0.5,
        max_intrinsic_rms_px=1.5,
    )

    result = CalibrationService(thresholds).calibrate_intrinsics_from_views(
        object_views, image_views, image_size=IMAGE_SIZE
    )

    assert result.status is CalibrationStatus.VALID
    assert result.metrics["valid_view_count"] >= 6
    assert result.metrics["rms_reprojection_error_px"] < 0.2
    assert np.allclose(
        np.diag(result.camera_matrix)[:2],
        np.diag(CAMERA_MATRIX)[:2],
        rtol=0.05,
    )


def test_charuco_generator_uses_explicit_physical_board_config():
    config = CharucoBoardConfig(
        dictionary="DICT_5X5_100",
        columns=6,
        rows=8,
        square_length_m=0.035,
        marker_length_m=0.021,
    )
    image = CalibrationService().generate_charuco_board(
        config, pixels_per_meter=2500
    )
    assert image.ndim == 2
    assert image.shape[0] > image.shape[1]
    assert int(image.min()) == 0
    assert int(image.max()) == 255
    detected = OpenCVCalibrationAdapter().detect_charuco(image, config)
    assert detected is not None
    assert len(detected.corner_ids) >= 20
