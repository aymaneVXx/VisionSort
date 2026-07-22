from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from visionsort.ui.components.common import page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Events", "Journal des événements métier et techniques")
    events = context.repo.recent_events(limit=500)
    if not events:
        st.info("Aucun événement.")
        return
    df = pd.DataFrame(events)
    event_types = sorted(df["event_type"].dropna().unique().tolist()) if "event_type" in df else []
    selected_type = st.selectbox("Filtrer par type", ["Tous"] + event_types, index=0)
    if selected_type != "Tous":
        df = df[df["event_type"] == selected_type]
    df["payload_preview"] = df["payload_json"].apply(lambda value: json.dumps(json.loads(value), ensure_ascii=False)[:250])
    st.dataframe(df.drop(columns=["payload_json"]), use_container_width=True)
    if not df.empty:
        options = {f"{row['event_type']} | {row['created_at']}": row for _, row in df.iterrows()}
        selected = st.selectbox("Voir un payload complet", list(options), index=0)
        if selected:
            st.code(json.dumps(json.loads(options[selected]["payload_json"]), ensure_ascii=False, indent=2), language="json")
