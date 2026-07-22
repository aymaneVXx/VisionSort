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
    sources = context.repo.list_sources()
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
            if cols[1].button("Démarrer", key=f"start-{row['id']}"):
                context.repo.enqueue_command(CommandType.START_SOURCE, {"source_id": row["id"]})
            if cols[2].button("Arrêter", key=f"stop-{row['id']}"):
                context.repo.enqueue_command(CommandType.STOP_SOURCE, {"source_id": row["id"]})
            if cols[3].button("Enregistrer", key=f"rec-{row['id']}"):
                context.repo.enqueue_command(CommandType.START_RECORDING, {"source_id": row["id"]})
            if st.button("Arrêter enregistrement", key=f"stop-rec-{row['id']}"):
                context.repo.enqueue_command(CommandType.STOP_RECORDING, {"source_id": row["id"]})
    st.dataframe(pd.DataFrame(sources), use_container_width=True)
