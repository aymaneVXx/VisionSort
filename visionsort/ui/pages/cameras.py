from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from visionsort.calibration.repository import CalibrationRepository
from visionsort.core.enums import CommandType
from visionsort.ui.components.common import (
    demo_warning,
    page_header,
    render_status,
)
from visionsort.ui.state import UIContext


_ROLE_LABELS = {
    "C1": "Entrée",
    "C2": "Convoyeur",
    "C3": "Sortie",
}


def _source_label(source: dict) -> str:
    role = str(source.get("role") or "")
    return f"{source.get('name')} — {_ROLE_LABELS.get(role, role)}"


def render(context: UIContext) -> None:
    page_header(
        "Caméras",
        "Ajoutez les flux, vérifiez leur état et préparez une session d’exploitation.",
    )
    demo_warning(context)
    models = context.repo.list_models()
    parcel_models = [
        row for row in models if row.get("task") in {"detection", "segmentation"}
    ]
    pose_models = [row for row in models if row.get("task") == "pose"]
    trackers = context.repo.list_trackers()
    sources = context.repo.list_sources()
    calibration_repo = CalibrationRepository(context.db)

    st.subheader("Caméras enregistrées")
    if not sources:
        st.info("Aucune caméra configurée. Utilisez « Ajouter une caméra » ci-dessous.")
    else:
        columns = st.columns(min(3, len(sources)))
        for index, source in enumerate(sources):
            active_calibration = calibration_repo.get_active_profile(str(source["id"]))
            with columns[index % len(columns)].container(border=True):
                st.markdown(f"### {_source_label(source)}")
                render_status(source.get("status") or "OFFLINE")
                info = st.columns(2)
                info[0].metric("Fréquence", f"{float(source.get('fps') or 0):.1f} FPS")
                info[1].metric(
                    "Calibration",
                    "Prête" if active_calibration else "À faire",
                )
                controls = st.columns(2)
                if controls[0].button(
                    "Tester",
                    key=f"test-{source['id']}",
                    use_container_width=True,
                ):
                    context.repo.enqueue_command(
                        CommandType.TEST_SOURCE,
                        {"uri": source["uri"], "role": source["role"]},
                    )
                    st.info("Test de connexion demandé.")
                if controls[1].button(
                    "Arrêter",
                    key=f"stop-{source['id']}",
                    use_container_width=True,
                ):
                    context.repo.enqueue_command(
                        CommandType.STOP_SOURCE,
                        {"source_id": source["id"]},
                    )
                recording = st.columns(2)
                if recording[0].button(
                    "Enregistrer",
                    key=f"record-{source['id']}",
                    use_container_width=True,
                ):
                    context.repo.enqueue_command(
                        CommandType.START_RECORDING,
                        {"source_id": source["id"]},
                    )
                if recording[1].button(
                    "Arrêter l’enregistrement",
                    key=f"stop-record-{source['id']}",
                    use_container_width=True,
                ):
                    context.repo.enqueue_command(
                        CommandType.STOP_RECORDING,
                        {"source_id": source["id"]},
                    )
                with st.expander("Détails avancés"):
                    st.write(f"Identifiant : `{source['id']}`")
                    st.write(f"Type de flux : `{source['source_type']}`")
                    st.write(f"Adresse : `{source['uri']}`")
                    st.write(f"Suivi : `{source['tracker_id']}`")
                    st.write(
                        "Configuration optique : "
                        f"`{source.get('optical_setup_id') or 'default'}`"
                    )
                    if source.get("last_error"):
                        st.error(str(source["last_error"]))
                    st.json(source.get("model_assignments") or [])

    with st.expander("Ajouter une caméra", expanded=not sources):
        with st.form("register_source"):
            main = st.columns(3)
            name = main[0].text_input("Nom de la caméra", value="Caméra entrée")
            role = main[1].selectbox(
                "Emplacement",
                ["C1", "C2", "C3"],
                format_func=lambda value: f"{value} — {_ROLE_LABELS[value]}",
            )
            source_type = main[2].selectbox(
                "Type de flux",
                ["RTSP", "VIDEO_FILE", "REPLAY"],
                format_func=lambda value: {
                    "RTSP": "Caméra réseau",
                    "VIDEO_FILE": "Fichier vidéo",
                    "REPLAY": "Replay de test",
                }[value],
            )
            uri = st.text_input("Adresse du flux ou chemin de la vidéo")
            optical_setup_id = "default"
            model_id = parcel_models[0]["id"] if parcel_models else None
            enable_pose = False
            pose_model_id = pose_models[0]["id"] if pose_models else None
            tracker_id = trackers[0]["id"] if trackers else None
            if st.checkbox("Afficher les paramètres avancés"):
                optical_setup_id = st.text_input(
                    "Configuration optique",
                    value="default",
                    help="Modifiez cette valeur après un changement de lentille, zoom ou mise au point.",
                )
                model_id = st.selectbox(
                    "Modèle de détection des colis",
                    [row["id"] for row in parcel_models],
                    index=0 if parcel_models else None,
                )
                enable_pose = st.checkbox(
                    "Activer le suivi opérateur",
                    value=False,
                    disabled=not pose_models,
                )
                pose_model_id = st.selectbox(
                    "Modèle opérateur",
                    [row["id"] for row in pose_models],
                    index=0 if pose_models else None,
                    disabled=not enable_pose,
                )
                tracker_id = st.selectbox(
                    "Moteur de suivi",
                    [row["id"] for row in trackers],
                    index=0 if trackers else None,
                )
            submitted = st.form_submit_button(
                "Enregistrer la caméra",
                type="primary",
                disabled=not parcel_models or not trackers,
            )
            if submitted:
                selected_model = next(row for row in parcel_models if row["id"] == model_id)
                assignments = [
                    {
                        "pipeline_role": (
                            "parcel_segmentation"
                            if selected_model["task"] == "segmentation"
                            else "parcel_detection"
                        ),
                        "task": selected_model["task"],
                        "model_id": model_id,
                        "use_active": True,
                        "enabled": True,
                    }
                ]
                if enable_pose and pose_model_id:
                    assignments.append(
                        {
                            "pipeline_role": "operator_pose",
                            "task": "pose",
                            "model_id": pose_model_id,
                            "use_active": True,
                            "enabled": True,
                        }
                    )
                context.repo.enqueue_command(
                    CommandType.REGISTER_SOURCE,
                    {
                        "name": name,
                        "role": role,
                        "source_type": source_type,
                        "uri": uri,
                        "model_id": model_id,
                        "model_assignments": assignments,
                        "tracker_id": tracker_id,
                        "optical_setup_id": optical_setup_id,
                        "enabled": True,
                    },
                )
                st.success("Caméra envoyée pour enregistrement.")

    st.subheader("Sessions d’exploitation")
    sessions = context.repo.list_capture_sessions()
    for session in sessions[:8]:
        with st.container(border=True):
            title = st.columns([3, 1])
            title[0].markdown(f"**{session['name']}**")
            title[1].caption(str(session.get("pipeline_state") or "Prête"))
            media_report = json.loads(session.get("media_report_json") or "{}")
            if media_report and not media_report.get("valid", True):
                st.warning("L’archive vidéo de cette session est incomplète.")
            controls = st.columns(3)
            if controls[0].button(
                "Démarrer",
                key=f"start-session-{session['id']}",
                type="primary",
                use_container_width=True,
            ):
                context.repo.enqueue_command(
                    CommandType.START_SESSION, {"session_id": session["id"]}
                )
            if controls[1].button(
                "Arrêter",
                key=f"stop-session-{session['id']}",
                use_container_width=True,
            ):
                context.repo.enqueue_command(
                    CommandType.STOP_SESSION, {"session_id": session["id"]}
                )
            if controls[2].button(
                "Voir les caméras",
                key=f"show-session-{session['id']}",
                use_container_width=True,
            ):
                st.dataframe(
                    pd.DataFrame(
                        context.repo.list_capture_session_sources(session["id"])
                    ),
                    use_container_width=True,
                )
    if not sessions:
        st.caption("Aucune session créée.")

    with st.expander("Créer une session"):
        with st.form("create_session"):
            session_name = st.text_input("Nom", value="Exploitation principale")
            demo_mode = st.checkbox(
                "Session de démonstration / replay",
                value=bool(context.config_demo_mode),
            )
            source_options = {_source_label(row): row for row in sources}
            selected_sources = st.multiselect(
                "Caméras utilisées",
                list(source_options),
                default=list(source_options),
            )
            if st.form_submit_button(
                "Créer la session",
                disabled=not selected_sources,
            ):
                items = [
                    {
                        "source_id": source_options[label]["id"],
                        "camera_role": source_options[label]["role"],
                        "time_offset_ms": 0.0,
                    }
                    for label in selected_sources
                ]
                context.repo.enqueue_command(
                    CommandType.CREATE_SESSION,
                    {
                        "name": session_name,
                        "demo_mode": bool(demo_mode),
                        "sources": items,
                        "config": {"validated_on_site": False},
                    },
                )
                st.success("Création de session demandée.")
    if st.button("Préparer les données de démonstration", type="secondary"):
        context.repo.enqueue_command(CommandType.BOOTSTRAP_DEMO, {})
        st.info("Préparation de la démonstration demandée.")
