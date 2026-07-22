from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from visionsort.ui.components.common import demo_warning, page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Dashboard", "Vue de supervision générale VisionSort")
    demo_warning(context)
    sources = context.repo.list_sources()
    sessions = context.repo.list_capture_sessions()
    jobs = context.repo.list_jobs()
    events = context.repo.recent_events(limit=20)
    parcels = context.repo.list_parcels()
    source_states = [row.get("status") for row in sources if row.get("status")]
    st.columns(4)[0].metric("Sources", len(sources))
    st.columns(4)[1].metric("Jobs", len(jobs))
    st.columns(4)[2].metric("Événements", len(events))
    st.columns(4)[3].metric("Colis globaux", len(parcels))
    st.subheader("Capture Sessions")
    if sessions:
        st.dataframe(pd.DataFrame(sessions), use_container_width=True)
    else:
        st.info("Aucune session disponible.")
    st.subheader("État des sources")
    if sources:
        df = pd.DataFrame(sources)
        st.dataframe(df[["name", "role", "source_type", "status", "fps", "model_id", "tracker_id", "recording_enabled"]], use_container_width=True)
    else:
        st.info("Aucune source déclarée.")
    st.subheader("Jobs runtime")
    st.dataframe(pd.DataFrame(jobs) if jobs else pd.DataFrame(columns=["id", "job_type", "status"]), use_container_width=True)
    st.subheader("Derniers événements")
    if events:
        df = pd.DataFrame(events)
        if "payload_json" in df.columns:
            df["payload_json"] = df["payload_json"].apply(lambda value: json.dumps(json.loads(value), ensure_ascii=False)[:180] if value else "")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Aucun événement pour le moment.")
