from __future__ import annotations

import pandas as pd
import streamlit as st

from visionsort.core.enums import CommandType
from visionsort.ui.components.common import demo_warning, page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Cameras", "Déclaration, test et pilotage des sources")
    demo_warning(context)
    models = context.repo.list_models()
    trackers = context.repo.list_trackers()
    sources = context.repo.list_sources()

    st.subheader("Capture Sessions")
    with st.form("create_session"):
        session_name = st.text_input("Nom de session", value="Session Replay")
        demo_mode = st.checkbox("DEMO/REPLAY (explicite)", value=bool(context.config_demo_mode))
        source_options = {f"{row['role']} | {row['name']} ({row['id']})": row for row in sources}
        c1 = st.selectbox("Source C1", list(source_options) if source_options else [], index=0 if source_options else None)
        c2 = st.selectbox("Source C2", list(source_options) if source_options else [], index=0 if source_options else None)
        c3_enabled = st.checkbox("Activer C3", value=False)
        c3 = st.selectbox("Source C3", list(source_options) if source_options else [], index=0 if source_options else None, disabled=not c3_enabled)
        offsets = st.columns(3)
        offset_c1 = offsets[0].number_input("Offset C1 (ms)", value=0.0, step=50.0)
        offset_c2 = offsets[1].number_input("Offset C2 (ms)", value=0.0, step=50.0)
        offset_c3 = offsets[2].number_input("Offset C3 (ms)", value=0.0, step=50.0, disabled=not c3_enabled)
        if st.form_submit_button("Créer la session"):
            items = []
            if c1:
                items.append({"source_id": source_options[c1]["id"], "camera_role": "C1", "time_offset_ms": float(offset_c1)})
            if c2:
                items.append({"source_id": source_options[c2]["id"], "camera_role": "C2", "time_offset_ms": float(offset_c2)})
            if c3_enabled and c3:
                items.append({"source_id": source_options[c3]["id"], "camera_role": "C3", "time_offset_ms": float(offset_c3)})
            context.repo.enqueue_command(
                CommandType.CREATE_SESSION,
                {"name": session_name, "demo_mode": bool(demo_mode), "sources": items, "config": {"validated_on_site": False}},
            )
            st.success("Commande de création de session envoyée.")

    sessions = context.repo.list_capture_sessions()
    if sessions:
        for sess in sessions[:10]:
            with st.container(border=True):
                st.write(f"**{sess['name']}** ({sess['id']}) - état `{sess.get('pipeline_state')}` - demo={bool(sess.get('demo_mode'))}")
                cols = st.columns(3)
                if cols[0].button("Démarrer session", key=f"start-session-{sess['id']}"):
                    context.repo.enqueue_command(CommandType.START_SESSION, {"session_id": sess["id"]})
                if cols[1].button("Arrêter session", key=f"stop-session-{sess['id']}"):
                    context.repo.enqueue_command(CommandType.STOP_SESSION, {"session_id": sess["id"]})
                if cols[2].button("Voir sources", key=f"show-session-{sess['id']}"):
                    st.dataframe(pd.DataFrame(context.repo.list_capture_session_sources(sess["id"])), use_container_width=True)
    else:
        st.info("Aucune session pour le moment.")
    with st.form("register_source"):
        name = st.text_input("Nom", value="Replay C1")
        role = st.selectbox("Rôle", ["C1", "C2", "C3"])
        source_type = st.selectbox("Type", ["REPLAY", "VIDEO_FILE", "RTSP"])
        uri = st.text_input("URI / chemin vidéo", value="")
        model_id = st.selectbox("Modèle", [row["id"] for row in models], index=0 if models else None)
        tracker_id = st.selectbox("Tracker", [row["id"] for row in trackers], index=0 if trackers else None)
        if st.form_submit_button("Enregistrer la source"):
            context.repo.enqueue_command(
                CommandType.REGISTER_SOURCE,
                {
                    "name": name,
                    "role": role,
                    "source_type": source_type,
                    "uri": uri,
                    "model_id": model_id,
                    "tracker_id": tracker_id,
                    "enabled": True,
                },
            )
            st.success("Commande d'enregistrement envoyée au supervisor.")
    if st.button("Bootstrap démo", type="secondary"):
        context.repo.enqueue_command(CommandType.BOOTSTRAP_DEMO, {})
        st.info("Commande de bootstrap démo envoyée.")
    st.subheader("Sources enregistrées")
    if not sources:
        st.info("Aucune source disponible.")
        return
    for row in sources:
        with st.container(border=True):
            st.write(f"**{row['name']}** - {row['role']} - {row['source_type']} - état `{row.get('status') or 'OFFLINE'}`")
            cols = st.columns(4)
            if cols[0].button("Tester", key=f"test-{row['id']}"):
                context.repo.enqueue_command(CommandType.TEST_SOURCE, {"uri": row["uri"], "role": row["role"]})
            cols[1].button("Démarrer via session", key=f"start-{row['id']}", disabled=True, help="Créez une CaptureSession puis utilisez 'Démarrer session'.")
            if cols[2].button("Arrêter", key=f"stop-{row['id']}"):
                context.repo.enqueue_command(CommandType.STOP_SOURCE, {"source_id": row["id"]})
            if cols[3].button("Enregistrer", key=f"rec-{row['id']}"):
                context.repo.enqueue_command(CommandType.START_RECORDING, {"source_id": row["id"]})
            if st.button("Arrêter enregistrement", key=f"stop-rec-{row['id']}"):
                context.repo.enqueue_command(CommandType.STOP_RECORDING, {"source_id": row["id"]})
    st.dataframe(pd.DataFrame(sources), use_container_width=True)
