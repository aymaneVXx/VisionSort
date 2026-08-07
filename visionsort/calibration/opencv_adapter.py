from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from visionsort.calibration.models import CharucoBoardConfig


@dataclass(frozen=True, slots=True)
class DetectedCharucoView:
    object_points: np.ndarray
    image_points: np.ndarray
    corner_ids: np.ndarray
    marker_corners: tuple[np.ndarray, ...]
    marker_ids: np.ndarray


class OpenCVCalibrationAdapter:
    """Keep OpenCV 4/5 ChArUco compatibility out of domain services."""

    def __init__(self) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "Cette installation OpenCV ne fournit pas le module aruco/ChArUco."
            )

    @staticmethod
    def _dictionary(dictionary_name: str):
        aruco = cv2.aruco
        dictionary_id = getattr(aruco, dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f"Dictionnaire ChArUco inconnu: {dictionary_name}")
        return aruco.getPredefinedDictionary(dictionary_id)

    def create_board(self, config: CharucoBoardConfig):
        aruco = cv2.aruco
        dictionary = self._dictionary(config.dictionary)
        if hasattr(aruco, "CharucoBoard"):
            board = aruco.CharucoBoard(
                (int(config.columns), int(config.rows)),
                float(config.square_length_m),
                float(config.marker_length_m),
                dictionary,
            )
        elif hasattr(aruco, "CharucoBoard_create"):  # OpenCV legacy
            board = aruco.CharucoBoard_create(
                int(config.columns),
                int(config.rows),
                float(config.square_length_m),
                float(config.marker_length_m),
                dictionary,
            )
        else:  # pragma: no cover - depends on unsupported OpenCV builds
            raise RuntimeError("API CharucoBoard absente de cette version OpenCV.")
        if hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(bool(config.legacy_pattern))
        return board

    def generate_board_image(
        self,
        config: CharucoBoardConfig,
        *,
        pixels_per_meter: int = 5000,
        margin_pixels: int = 40,
    ) -> np.ndarray:
        board = self.create_board(config)
        # Keep every square an integer number of pixels. Some OpenCV 5 builds
        # reject otherwise valid boards when the requested dimensions round
        # each full side independently.
        square_pixels = max(
            40, int(round(config.square_length_m * pixels_per_meter))
        )
        width = int(config.columns) * square_pixels
        height = int(config.rows) * square_pixels
        if hasattr(board, "generateImage"):
            board_image = board.generateImage(
                (width, height), marginSize=0, borderBits=1
            )
            return cv2.copyMakeBorder(
                board_image,
                int(margin_pixels),
                int(margin_pixels),
                int(margin_pixels),
                int(margin_pixels),
                cv2.BORDER_CONSTANT,
                value=255,
            )
        size = (width + 2 * margin_pixels, height + 2 * margin_pixels)
        image = np.zeros((size[1], size[0]), dtype=np.uint8)
        board.draw(size, image, int(margin_pixels), 1)
        return image

    def detect_charuco(
        self, image: np.ndarray, config: CharucoBoardConfig
    ) -> DetectedCharucoView | None:
        if image is None or image.size == 0:
            return None
        board = self.create_board(config)
        aruco = cv2.aruco
        if hasattr(aruco, "CharucoDetector"):
            detector = aruco.CharucoDetector(board)
            corners, ids, marker_corners, marker_ids = detector.detectBoard(image)
        else:  # OpenCV 4 legacy adapter
            gray = (
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.ndim == 3
                else image
            )
            marker_corners, marker_ids, _ = aruco.detectMarkers(
                gray, self._dictionary(config.dictionary)
            )
            if marker_ids is None or not marker_corners:
                return None
            _, corners, ids = aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )
        if corners is None or ids is None or len(ids) < 4:
            return None
        if hasattr(board, "matchImagePoints"):
            object_points, image_points = board.matchImagePoints(corners, ids)
        else:  # pragma: no cover - only very old OpenCV builds
            chessboard = np.asarray(board.chessboardCorners, dtype=np.float32)
            flat_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
            object_points = chessboard[flat_ids]
            image_points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        return DetectedCharucoView(
            object_points=np.asarray(object_points, dtype=np.float32).reshape(-1, 3),
            image_points=np.asarray(image_points, dtype=np.float32).reshape(-1, 2),
            corner_ids=np.asarray(ids, dtype=np.int32).reshape(-1),
            marker_corners=tuple(
                np.asarray(item, dtype=np.float32).reshape(-1, 2)
                for item in (marker_corners or [])
            ),
            marker_ids=np.asarray(
                marker_ids if marker_ids is not None else [], dtype=np.int32
            ).reshape(-1),
        )

    @staticmethod
    def calibrate_camera(
        object_points: Sequence[np.ndarray],
        image_points: Sequence[np.ndarray],
        image_size: tuple[int, int],
    ) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
        objects = [np.asarray(item, dtype=np.float32).reshape(-1, 1, 3) for item in object_points]
        images = [np.asarray(item, dtype=np.float32).reshape(-1, 1, 2) for item in image_points]
        if hasattr(cv2, "calibrateCameraExtended"):
            (
                rms,
                camera_matrix,
                distortion,
                rvecs,
                tvecs,
                _intrinsic_std,
                _extrinsic_std,
                per_view,
            ) = cv2.calibrateCameraExtended(
                objects,
                images,
                tuple(int(value) for value in image_size),
                None,
                None,
            )
            errors = np.asarray(per_view, dtype=np.float64).reshape(-1)
        else:  # pragma: no cover - OpenCV 4.10+ exposes Extended
            rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
                objects,
                images,
                tuple(int(value) for value in image_size),
                None,
                None,
            )
            errors = OpenCVCalibrationAdapter.per_view_errors(
                objects, images, rvecs, tvecs, camera_matrix, distortion
            )
        return (
            float(rms),
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64).reshape(-1),
            list(rvecs),
            list(tvecs),
            errors,
        )

    @staticmethod
    def per_view_errors(
        object_points: Sequence[np.ndarray],
        image_points: Sequence[np.ndarray],
        rvecs: Sequence[np.ndarray],
        tvecs: Sequence[np.ndarray],
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> np.ndarray:
        values: list[float] = []
        for objects, images, rvec, tvec in zip(
            object_points, image_points, rvecs, tvecs
        ):
            projected, _ = cv2.projectPoints(
                np.asarray(objects, dtype=np.float32).reshape(-1, 3),
                rvec,
                tvec,
                camera_matrix,
                distortion,
            )
            delta = projected.reshape(-1, 2) - np.asarray(images).reshape(-1, 2)
            values.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
        return np.asarray(values, dtype=np.float64)

    @staticmethod
    def undistort_points(
        points: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray
    ) -> np.ndarray:
        return cv2.undistortPoints(
            np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
            P=np.asarray(camera_matrix, dtype=np.float64),
        ).reshape(-1, 2)

    @staticmethod
    def distort_points(
        undistorted_pixel_points: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> np.ndarray:
        pixels = np.asarray(undistorted_pixel_points, dtype=np.float64).reshape(-1, 2)
        inverse_k = np.linalg.inv(np.asarray(camera_matrix, dtype=np.float64))
        homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
        normalized = (inverse_k @ homogeneous.T).T
        normalized = normalized[:, :2] / normalized[:, 2:3]
        objects = np.column_stack([normalized, np.ones(len(normalized))])
        projected, _ = cv2.projectPoints(
            objects,
            np.zeros(3),
            np.zeros(3),
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(distortion, dtype=np.float64),
        )
        return projected.reshape(-1, 2)

    @staticmethod
    def find_homography(
        source_points: np.ndarray,
        destination_points: np.ndarray,
        *,
        ransac_threshold: float,
        confidence: float,
        max_iterations: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        # USAC_DEFAULT is stable when source pixels and destination metres have
        # very different scales. MAGSAC's scale estimator is less reliable for
        # this specific pixel -> metre formulation.
        method = getattr(cv2, "USAC_DEFAULT", cv2.RANSAC)
        cv2.setRNGSeed(0)
        return cv2.findHomography(
            np.asarray(source_points, dtype=np.float64).reshape(-1, 2),
            np.asarray(destination_points, dtype=np.float64).reshape(-1, 2),
            method=method,
            ransacReprojThreshold=float(ransac_threshold),
            maxIters=int(max_iterations),
            confidence=float(confidence),
        )

    @staticmethod
    def perspective_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(
            np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
            np.asarray(matrix, dtype=np.float64),
        ).reshape(-1, 2)
