from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

from visionsort.calibration.models import CalibrationProfile, CalibrationStatus, CharucoBoardConfig
from visionsort.calibration.opencv_adapter import OpenCVCalibrationAdapter
from visionsort.calibration.repository import CalibrationRepository
from visionsort.calibration.service import CalibrationService
from visionsort.core.enums import CommandType
from visionsort.core.paths import ROOT_DIR
from visionsort.core.site_config import apply_site_config
from visionsort.ui.components.common import demo_warning, page_header
from visionsort.ui.state import UIContext


def _absolute_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _decode_upload(uploaded) -> np.ndarray | None:
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _parse_correspondences(value: str) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(value)
    if not isinstance(payload, list):
        raise ValueError("Une liste JSON de correspondances est requise.")
    raw: list[list[float]] = []
    world: list[list[float]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Chaque correspondance doit etre un objet JSON.")
        raw.append([float(item["pixel_x"]), float(item["pixel_y"])])
        world.append([float(item["world_x_m"]), float(item["world_y_m"])])
    return np.asarray(raw, dtype=np.float64), np.asarray(world, dtype=np.float64)


def _draw_preview(
    image: np.ndarray,
    profile: CalibrationProfile,
    raw_points: np.ndarray,
    inlier_mask: np.ndarray,
) -> np.ndarray:
    output = image.copy()
    for point, inlier in zip(raw_points, inlier_mask):
        center = tuple(int(round(value)) for value in point)
        cv2.circle(output, center, 7, (40, 210, 40) if inlier else (20, 20, 230), 2)
    bounds = profile.quality_metrics.get("homography", {}).get("world_bounds_m", {})
    if bounds:
        xs = np.linspace(float(bounds["x_min"]), float(bounds["x_max"]), 6)
        ys = np.linspace(float(bounds["y_min"]), float(bounds["y_max"]), 6)
        adapter = OpenCVCalibrationAdapter()
        inverse = np.asarray(profile.homography_world_to_image_undistorted)
        camera_matrix = np.asarray(profile.camera_matrix)
        distortion = np.asarray(profile.distortion_coefficients)
        for x_value in xs:
            world_line = np.asarray(
                [[x_value, float(bounds["y_min"])], [x_value, float(bounds["y_max"])]],
                dtype=np.float64,
            )
            undistorted = adapter.perspective_transform(world_line, inverse)
            raw = adapter.distort_points(undistorted, camera_matrix, distortion)
            cv2.line(output, tuple(np.rint(raw[0]).astype(int)), tuple(np.rint(raw[1]).astype(int)), (255, 180, 30), 1)
        for y_value in ys:
            world_line = np.asarray(
                [[float(bounds["x_min"]), y_value], [float(bounds["x_max"]), y_value]],
                dtype=np.float64,
            )
            undistorted = adapter.perspective_transform(world_line, inverse)
            raw = adapter.distort_points(undistorted, camera_matrix, distortion)
            cv2.line(output, tuple(np.rint(raw[0]).astype(int)), tuple(np.rint(raw[1]).astype(int)), (255, 180, 30), 1)
    return output


def render(context: UIContext) -> None:
    page_header(
        "Calibration",
        "Intrinseques ChArUco et repere convoyeur commun en metres",
    )
    demo_warning(context)
    sources = context.repo.list_sources()
    if not sources:
        st.info("Enregistrez d'abord une source dans la page Cameras.")
        return
    source_by_label = {
        f"{row['role']} | {row['name']} ({row['id']})": row for row in sources
    }
    selected_label = st.selectbox("Source / camera", list(source_by_label))
    source = source_by_label[selected_label]
    source_id = str(source["id"])
    repository = CalibrationRepository(context.db)
    site_config = context.repo.get_site_config()
    effective_config = apply_site_config(context.config_values, site_config)
    service = CalibrationService(
        thresholds=effective_config.get("calibration_quality_thresholds")
    )
    active = repository.get_active_profile(source_id)
    if active:
        st.success(
            f"Profil actif: {active.profile_id} · v{active.version} · "
            f"{active.image_width}x{active.image_height} · {active.status.value}"
        )
        st.caption(f"SHA-256: {active.fingerprint_sha256}")
    else:
        st.warning("Aucun profil actif pour cette source.")

    frames_key = f"calibration_frames:{source_id}"
    st.session_state.setdefault(frames_key, [])
    frames: list[np.ndarray] = st.session_state[frames_key]

    st.subheader("1. Mire et vues intrinseques")
    board_columns = st.columns(5)
    dictionary = board_columns[0].selectbox(
        "Dictionnaire", ["DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250"]
    )
    columns = board_columns[1].number_input("Colonnes", 3, 20, 5)
    rows = board_columns[2].number_input("Lignes", 3, 20, 7)
    square_length = board_columns[3].number_input(
        "Carre (m)", min_value=0.001, value=0.04, format="%.4f"
    )
    marker_length = board_columns[4].number_input(
        "Marqueur (m)", min_value=0.001, value=0.02, format="%.4f"
    )
    try:
        board_config = CharucoBoardConfig(
            dictionary=dictionary,
            columns=int(columns),
            rows=int(rows),
            square_length_m=float(square_length),
            marker_length_m=float(marker_length),
        )
        board_image = service.generate_charuco_board(board_config)
        ok, encoded = cv2.imencode(".png", board_image)
        if ok:
            st.download_button(
                "Telecharger la mire ChArUco",
                data=encoded.tobytes(),
                file_name=(
                    f"charuco-{columns}x{rows}-{square_length:.4f}m.png"
                ),
                mime="image/png",
            )
    except Exception as exc:
        st.error(str(exc))
        return

    uploads = st.file_uploader(
        "Importer des vues de calibration",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )
    capture_columns = st.columns(3)
    if capture_columns[0].button("Ajouter les images importees"):
        decoded = [image for image in (_decode_upload(item) for item in uploads) if image is not None]
        frames.extend(decoded)
        st.success(f"{len(decoded)} vue(s) ajoutee(s).")
    if capture_columns[1].button("Capturer la frame runtime courante"):
        path = _absolute_path(source.get("calibration_frame_path"))
        image = cv2.imread(str(path)) if path and path.exists() else None
        if image is None:
            st.error("Aucune frame brute runtime disponible. Demarrez une session avec cette source.")
        else:
            frames.append(image)
            st.success("Frame brute runtime ajoutee.")
    if capture_columns[2].button("Vider les vues"):
        frames.clear()
        st.session_state.pop(f"intrinsic:{source_id}", None)
        st.session_state.pop(f"homography:{source_id}", None)
        st.rerun()
    st.caption(f"Vues chargees: {len(frames)}")
    if frames:
        st.image(frames[-1], caption="Derniere vue", channels="BGR", width=500)
    if st.button("Calculer la calibration intrinseque", disabled=not frames):
        try:
            intrinsic = service.calibrate_intrinsics(frames, board_config)
            st.session_state[f"intrinsic:{source_id}"] = intrinsic
            st.success(f"Intrinseques calculees: {intrinsic.status.value}")
        except Exception as exc:
            st.error(str(exc))
    intrinsic = st.session_state.get(f"intrinsic:{source_id}")
    if intrinsic is not None:
        st.json(intrinsic.metrics, expanded=False)

    st.subheader("2. Plan convoyeur et homographie")
    mode = st.radio(
        "Mode de correspondances",
        ["Points pixel ↔ monde", "ChArUco sur le convoyeur"],
        horizontal=True,
    )
    default_points = json.dumps(
        [
            {"pixel_x": 100, "pixel_y": 100, "world_x_m": 0.0, "world_y_m": 0.0},
            {"pixel_x": 900, "pixel_y": 100, "world_x_m": 2.0, "world_y_m": 0.0},
            {"pixel_x": 900, "pixel_y": 600, "world_x_m": 2.0, "world_y_m": 1.0},
            {"pixel_x": 100, "pixel_y": 600, "world_x_m": 0.0, "world_y_m": 1.0},
        ],
        indent=2,
    )
    raw_points: np.ndarray | None = None
    world_points: np.ndarray | None = None
    if mode == "Points pixel ↔ monde":
        correspondence_text = st.text_area(
            "Correspondances JSON",
            value=default_points,
            height=220,
        )
        try:
            raw_points, world_points = _parse_correspondences(correspondence_text)
        except Exception as exc:
            st.error(str(exc))
    else:
        origin_columns = st.columns(3)
        origin_x = origin_columns[0].number_input("Origine X (m)", value=0.0)
        origin_y = origin_columns[1].number_input("Origine Y (m)", value=0.0)
        rotation = origin_columns[2].number_input("Rotation (degres)", value=0.0)
        if frames:
            frame_index = st.selectbox("Vue ChArUco sur le plan", range(len(frames)))
            try:
                raw_points, world_points = service.charuco_plane_correspondences(
                    frames[int(frame_index)],
                    board_config,
                    world_origin_m=(float(origin_x), float(origin_y)),
                    world_rotation_degrees=float(rotation),
                )
                st.caption(f"{len(raw_points)} correspondances detectees.")
            except Exception as exc:
                st.warning(str(exc))
    if st.button(
        "Estimer l'homographie robuste",
        disabled=intrinsic is None or raw_points is None or world_points is None,
    ):
        try:
            homography = service.estimate_homography(
                intrinsic, raw_points, world_points
            )
            st.session_state[f"homography:{source_id}"] = homography
            st.success(f"Homographie calculee: {homography.status.value}")
        except Exception as exc:
            st.error(str(exc))
    homography = st.session_state.get(f"homography:{source_id}")
    if homography is not None:
        st.json(homography.metrics, expanded=False)

    st.subheader("3. Version, preview et activation")
    built_profile: CalibrationProfile | None = None
    if intrinsic is not None and homography is not None:
        built_profile = service.build_profile(
            source_id=source_id,
            version=repository.next_version(source_id),
            intrinsic=intrinsic,
            homography=homography,
            board_config=board_config,
            optical_configuration={
                "source_id": source_id,
                "source_type": str(source.get("source_type") or ""),
                "source_uri": str(source.get("uri") or ""),
                "camera_role": str(source.get("role") or ""),
                "optical_setup_id": str(
                    source.get("optical_setup_id") or "default"
                ),
            },
            world_coordinate_convention={
                "frame_id": "site_world",
                "unit": "m",
                "x_axis": "conveyor_longitudinal",
                "y_axis": "conveyor_transverse",
                "z_axis": "up",
                "conveyor_plane_z_m": 0.0,
            },
        )
        if frames:
            preview = _draw_preview(
                frames[-1],
                built_profile,
                homography.raw_image_points,
                homography.inlier_mask,
            )
            st.image(
                preview,
                caption="Inliers (vert), outliers (rouge), grille monde (bleu)",
                channels="BGR",
                use_container_width=True,
            )
        if st.button("Enregistrer cette version immuable"):
            context.repo.enqueue_command(
                CommandType.SAVE_CALIBRATION_PROFILE,
                {"profile": built_profile.to_dict()},
            )
            st.success("Commande d'enregistrement envoyee au supervisor.")
    profiles = repository.list_profiles(source_id)
    for profile in profiles:
        with st.container(border=True):
            st.write(
                f"**v{profile.version}** · `{profile.status.value}` · "
                f"{profile.image_width}x{profile.image_height} · {profile.profile_id}"
            )
            st.caption(profile.fingerprint_sha256)
            is_active = bool(active and active.profile_id == profile.profile_id)
            if st.button(
                "Activer pour les nouvelles sessions",
                key=f"activate-calibration-{profile.profile_id}",
                disabled=is_active or profile.status is not CalibrationStatus.VALID,
            ):
                context.repo.enqueue_command(
                    CommandType.ACTIVATE_CALIBRATION_PROFILE,
                    {"source_id": source_id, "profile_id": profile.profile_id},
                )
                st.success("Commande d'activation envoyee au supervisor.")
    st.info(
        "Aucune metrique n'est marquee VALIDATED_ON_SITE. Une activation change "
        "uniquement les nouvelles CaptureSessions; les snapshots existants restent immuables."
    )
