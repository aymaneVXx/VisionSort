from __future__ import annotations

import pandas as pd
import streamlit as st

from visionsort.ui.components.common import (
    demo_warning,
    page_header,
    show_preview,
)
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header(
        "Live Tracking", "Previews annotees, tracks locaux et colis globaux"
    )
    demo_warning(context)
    sessions = context.repo.list_capture_sessions()
    session_map = {
        f"{row['name']} ({row['id']})": row["id"] for row in sessions
    }
    selected = st.selectbox(
        "CaptureSession",
        list(session_map),
        index=0 if session_map else None,
    )
    selected_session_id = session_map[selected] if selected else None
    sources = context.repo.list_sources()
    if not sources:
        st.info("Aucune source a afficher.")
        return
    if selected_session_id:
        session_sources = {
            row["source_id"]
            for row in context.repo.list_capture_session_sources(
                selected_session_id
            )
        }
        sources = [row for row in sources if row["id"] in session_sources]

    runtime_state = context.repo.get_runtime_model_state()
    runtime_tasks = (
        (runtime_state.get("consistency") or {}).get("tasks") or []
    )
    runtime_by_task = {
        str(row.get("task")): row for row in runtime_tasks
    }
    if runtime_tasks:
        st.subheader("Routage runtime effectif")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "tache": row.get("task"),
                        "modele_runtime": row.get("runtime_model_id"),
                        "generation": row.get("routing_generation"),
                        "charge": row.get("loaded"),
                        "coherent": row.get("consistent"),
                    }
                    for row in runtime_tasks
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    cols = st.columns(min(3, len(sources)))
    for index, source in enumerate(sources):
        with cols[index % len(cols)]:
            show_preview(
                source.get("preview_path"),
                f"{source['role']} - {source.get('status') or 'OFFLINE'}",
            )
            assignments = source.get("model_assignments") or []
            routes = []
            for assignment in assignments:
                if not assignment.get("enabled", 1):
                    continue
                task = str(assignment.get("task"))
                runtime = runtime_by_task.get(task) or {}
                model_id = runtime.get("runtime_model_id") or assignment.get(
                    "model_id"
                )
                generation = runtime.get("routing_generation") or 0
                routes.append(f"{task}={model_id}@g{generation}")
            st.caption(
                f"FPS: {source.get('fps') or 0:.2f} | "
                f"Runtime: {' | '.join(routes) or 'UNAVAILABLE'} | "
                f"Tracker: {source['tracker_id']}"
            )

    st.subheader("Tracklets")
    tracklets = context.repo.list_tracklets()
    if selected_session_id:
        tracklets = [
            row
            for row in tracklets
            if row.get("session_id") == selected_session_id
        ]
    st.dataframe(pd.DataFrame(tracklets), use_container_width=True)
    st.subheader("Colis globaux")
    parcels = context.repo.list_parcels()
    st.dataframe(
        pd.DataFrame(parcels)
        if parcels
        else pd.DataFrame(columns=["parcel_id", "state"]),
        use_container_width=True,
    )
