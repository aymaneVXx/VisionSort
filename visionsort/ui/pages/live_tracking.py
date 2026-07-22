from __future__ import annotations

import pandas as pd
import streamlit as st

from visionsort.ui.components.common import demo_warning, page_header, show_preview
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Live Tracking", "Previews annotées, tracks locaux et colis globaux")
    demo_warning(context)
    sessions = context.repo.list_capture_sessions()
    session_map = {f"{row['name']} ({row['id']})": row["id"] for row in sessions}
    selected = st.selectbox("CaptureSession", list(session_map), index=0 if session_map else None)
    selected_session_id = session_map[selected] if selected else None
    sources = context.repo.list_sources()
    if not sources:
        st.info("Aucune source à afficher.")
        return
    if selected_session_id:
        session_sources = {row["source_id"] for row in context.repo.list_capture_session_sources(selected_session_id)}
        sources = [row for row in sources if row["id"] in session_sources]
    cols = st.columns(min(3, len(sources)))
    for idx, source in enumerate(sources):
        with cols[idx % len(cols)]:
            show_preview(source.get("preview_path"), f"{source['role']} - {source.get('status') or 'OFFLINE'}")
            st.caption(f"FPS: {source.get('fps') or 0:.2f} | Modèle: {source['model_id']} | Tracker: {source['tracker_id']}")
    st.subheader("Tracklets")
    tracklets = context.repo.list_tracklets()
    if selected_session_id:
        tracklets = [row for row in tracklets if row.get("session_id") == selected_session_id]
    st.dataframe(pd.DataFrame(tracklets), use_container_width=True)
    st.subheader("Colis globaux")
    parcels = context.repo.list_parcels()
    st.dataframe(pd.DataFrame(parcels) if parcels else pd.DataFrame(columns=["parcel_id", "state"]), use_container_width=True)
