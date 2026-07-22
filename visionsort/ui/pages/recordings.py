from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from visionsort.core.paths import ROOT_DIR
from visionsort.ui.components.common import page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Recordings", "Segments enregistrés et métadonnées")
    recordings = context.repo.list_recordings()
    if not recordings:
        st.info("Aucun segment enregistré.")
        return
    sessions = sorted({row.get("session_id") for row in recordings if row.get("session_id")})
    selected_session = st.selectbox("Filtrer par session", ["Toutes"] + sessions, index=0)
    if selected_session != "Toutes":
        recordings = [row for row in recordings if row.get("session_id") == selected_session]
    for row in recordings:
        row["exists"] = Path(row["segment_path"]).exists() or (ROOT_DIR / row["segment_path"]).exists()
    st.dataframe(pd.DataFrame(recordings), use_container_width=True)
    options = {f"{row['source_id']} | {row['segment_path']}": row for row in recordings if row.get("segment_path")}
    selected = st.selectbox("Prévisualiser un segment", list(options), index=0 if options else None)
    if selected:
        record = options[selected]
        segment_path = Path(record["segment_path"])
        if not segment_path.exists():
            segment_path = ROOT_DIR / record["segment_path"]
        if segment_path.exists():
            st.video(str(segment_path))
