from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from visionsort.calibration.models import (
    DEFAULT_WORLD_CONVENTION,
    CalibrationProfile,
    CalibrationStatus,
    CharucoBoardConfig,
    world_convention_for_source,
)
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
        "Un assistant en trois étapes pour relier l’image de la caméra au convoyeur.",
    )
    demo_warning(context)
    sources = context.repo.list_sources()
    if not sources:
        st.info("Enregistrez d’abord une caméra depuis la page Caméras.")
        return
    source_by_label = {
        f"{row['name']} — {row['role']}": row for row in sources
    }
    selected_label = st.selectbox("Caméra à calibrer", list(source_by_label))
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
            f"Calibration active · version {active.version} · "
            f"{active.image_width} × {active.image_height}"
        )
    else:
        st.warning("Cette caméra n’a pas encore de calibration active.")

    frames_key = f"calibration_frames:{source_id}"
    st.session_state.setdefault(frames_key, [])
    frames: list[np.ndarray] = st.session_state[frames_key]

    st.subheader("Étape 1 — Caméra")
    st.write(
        "Prenez plusieurs images de la mire sous différents angles. "
        "La caméra et son objectif ne doivent plus bouger après cette étape."
    )
    with st.expander("Paramètres de la mire"):
        board_columns = st.columns(5)
        dictionary = board_columns[0].selectbox(
            "Type de mire", ["DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250"]
        )
        columns = board_columns[1].number_input("Colonnes", 3, 20, 5)
        rows = board_columns[2].number_input("Lignes", 3, 20, 7)
        square_length = board_columns[3].number_input(
            "Côté d’un carré (m)", min_value=0.001, value=0.04, format="%.4f"
        )
        marker_length = board_columns[4].number_input(
            "Taille d’un marqueur (m)", min_value=0.001, value=0.02, format="%.4f"
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
                "Télécharger la mire",
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
        "Importer des photos de la mire",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
    )
    capture_columns = st.columns(3)
    if capture_columns[0].button("Ajouter les images importées"):
        decoded = [image for image in (_decode_upload(item) for item in uploads) if image is not None]
        frames.extend(decoded)
        st.success(f"{len(decoded)} image(s) ajoutée(s).")
    if capture_columns[1].button("Capturer l’image actuelle"):
        path = _absolute_path(source.get("calibration_frame_path"))
        image = cv2.imread(str(path)) if path and path.exists() else None
        if image is None:
            st.error("Aucune image disponible. Démarrez une session avec cette caméra.")
        else:
            frames.append(image)
            st.success("Image actuelle ajoutée.")
    if capture_columns[2].button("Effacer les images"):
        frames.clear()
        st.session_state.pop(f"intrinsic:{source_id}", None)
        st.session_state.pop(f"homography:{source_id}", None)
        st.rerun()
    st.caption(f"Images disponibles : {len(frames)}")
    if frames:
        st.image(frames[-1], caption="Dernière vue", channels="BGR", width=500)
    if st.button("Vérifier la qualité caméra", disabled=not frames, type="primary"):
        try:
            intrinsic = service.calibrate_intrinsics(frames, board_config)
            st.session_state[f"intrinsic:{source_id}"] = intrinsic
            st.success("Qualité caméra validée.")
        except Exception as exc:
            st.error(str(exc))
    intrinsic = st.session_state.get(f"intrinsic:{source_id}")
    if intrinsic is not None:
        with st.expander("Détails techniques — qualité caméra"):
            st.json(intrinsic.metrics)

    st.subheader("Étape 2 — Repère au sol")
    st.write(
        "Indiquez quelques points connus sur le convoyeur pour relier les pixels "
        "aux distances réelles."
    )
    mode = st.radio(
        "Méthode",
        ["Points connus sur le convoyeur", "Mire posée sur le convoyeur"],
        horizontal=True,
    )
    default_points = [
        {"Pixel X": 100, "Pixel Y": 100, "Distance X (m)": 0.0, "Distance Y (m)": 0.0},
        {"Pixel X": 900, "Pixel Y": 100, "Distance X (m)": 2.0, "Distance Y (m)": 0.0},
        {"Pixel X": 900, "Pixel Y": 600, "Distance X (m)": 2.0, "Distance Y (m)": 1.0},
        {"Pixel X": 100, "Pixel Y": 600, "Distance X (m)": 0.0, "Distance Y (m)": 1.0},
    ]
    raw_points: np.ndarray | None = None
    world_points: np.ndarray | None = None
    if mode == "Points connus sur le convoyeur":
        point_table = st.data_editor(
            pd.DataFrame(default_points),
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
        )
        try:
            raw_points = point_table[["Pixel X", "Pixel Y"]].to_numpy(
                dtype=np.float64
            )
            world_points = point_table[
                ["Distance X (m)", "Distance Y (m)"]
            ].to_numpy(dtype=np.float64)
        except Exception as exc:
            st.error(str(exc))
    else:
        origin_columns = st.columns(3)
        origin_x = origin_columns[0].number_input("Origine X (m)", value=0.0)
        origin_y = origin_columns[1].number_input("Origine Y (m)", value=0.0)
        rotation = origin_columns[2].number_input("Rotation (degres)", value=0.0)
        if frames:
            frame_index = st.selectbox("Image utilisée", range(len(frames)))
            try:
                raw_points, world_points = service.charuco_plane_correspondences(
                    frames[int(frame_index)],
                    board_config,
                    world_origin_m=(float(origin_x), float(origin_y)),
                    world_rotation_degrees=float(rotation),
                )
                st.caption(f"{len(raw_points)} points détectés.")
            except Exception as exc:
                st.warning(str(exc))
    if st.button(
        "Calculer le repère au sol",
        disabled=intrinsic is None or raw_points is None or world_points is None,
    ):
        try:
            homography = service.estimate_homography(
                intrinsic, raw_points, world_points
            )
            st.session_state[f"homography:{source_id}"] = homography
            st.success("Repère au sol calculé.")
        except Exception as exc:
            st.error(str(exc))
    homography = st.session_state.get(f"homography:{source_id}")
    if homography is not None:
        with st.expander("Détails techniques — repère au sol"):
            st.json(homography.metrics)

    st.subheader("Repère de coordonnées")
    local_frame_id = f"camera:{source_id}"
    active_frame_id = (
        str(active.world_coordinate_convention.get("frame_id") or "")
        if active
        else ""
    )
    frame_mode = st.radio(
        "Comment les coordonnées ont-elles été mesurées ?",
        ["Repère propre à cette caméra (recommandé)", "Repère commun au site"],
        index=1 if active_frame_id and active_frame_id != local_frame_id else 0,
    )
    shared_frame_id = st.text_input(
        "Nom du repère commun",
        value=(
            active_frame_id
            if active_frame_id and active_frame_id != local_frame_id
            else "site_world"
        ),
        disabled=frame_mode.startswith("Repère propre"),
        help=(
            "Utilisez un repère commun uniquement si plusieurs caméras ont été "
            "mesurées dans le même système physique."
        ),
    )
    frame_id = (
        local_frame_id
        if frame_mode.startswith("Repère propre")
        else shared_frame_id.strip()
    )
    if frame_mode.startswith("Repère propre"):
        st.caption(f"Repère utilisé automatiquement : `{local_frame_id}`")
    elif not frame_id:
        st.error("Le nom du repère commun est obligatoire.")

    st.subheader("Étape 3 — Vérification")
    built_profile: CalibrationProfile | None = None
    if intrinsic is not None and homography is not None and frame_id:
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
            world_coordinate_convention=world_convention_for_source(
                source_id,
                {**DEFAULT_WORLD_CONVENTION, "frame_id": frame_id},
            ),
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
                caption="Vérification de la grille réelle sur l’image",
                channels="BGR",
                use_container_width=True,
            )
        if built_profile.status is CalibrationStatus.VALID:
            st.success("Calibration prête à être enregistrée.")
        save_columns = st.columns(2)
        if save_columns[0].button("Enregistrer cette version"):
            context.repo.enqueue_command(
                CommandType.SAVE_CALIBRATION_PROFILE,
                {"profile": built_profile.to_dict()},
            )
            st.success("Enregistrement demandé.")
        if save_columns[1].button(
            "Enregistrer et activer",
            type="primary",
            disabled=built_profile.status is not CalibrationStatus.VALID,
        ):
            context.repo.enqueue_command(
                CommandType.SAVE_CALIBRATION_PROFILE,
                {"profile": built_profile.to_dict()},
            )
            context.repo.enqueue_command(
                CommandType.ACTIVATE_CALIBRATION_PROFILE,
                {"source_id": source_id, "profile_id": built_profile.profile_id},
            )
            st.success("Enregistrement et activation demandés.")
    profiles = repository.list_profiles(source_id)
    with st.expander(f"Versions enregistrées ({len(profiles)})"):
        for profile in profiles:
            with st.container(border=True):
                st.write(
                    f"**Version {profile.version}** · {profile.status.value} · "
                    f"{profile.image_width} × {profile.image_height}"
                )
                st.caption(
                    f"Repère : {profile.world_coordinate_convention.get('frame_id') or 'non déclaré'}"
                )
                is_active = bool(active and active.profile_id == profile.profile_id)
                if st.button(
                    "Activer pour les nouvelles sessions",
                    key=f"activate-calibration-{profile.profile_id}",
                    disabled=is_active
                    or profile.status is not CalibrationStatus.VALID,
                ):
                    context.repo.enqueue_command(
                        CommandType.ACTIVATE_CALIBRATION_PROFILE,
                        {"source_id": source_id, "profile_id": profile.profile_id},
                    )
                    st.success("Activation demandée.")
    st.caption(
        "Une activation s’applique aux nouvelles sessions. Les sessions déjà "
        "démarrées conservent leur calibration."
    )
