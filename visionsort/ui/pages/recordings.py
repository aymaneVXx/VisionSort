from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from visionsort.core.paths import ROOT_DIR
from visionsort.ui.components.common import page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Recordings", "Segments enregistrés et métadonnées")
    recordings = context.repo.list_recordings()
    coverage = context.repo.list_media_coverage()
    if coverage:
        st.subheader("Couverture média par CaptureSession")
        st.dataframe(pd.DataFrame(coverage), use_container_width=True)
        if any(row.get("status") == "INSUFFICIENT" for row in coverage):
            st.error(
                "Certaines sessions ne disposent pas d'une archive "
                "suffisante pour construire un dataset."
            )
    if not recordings:
        st.info("Aucun segment enregistré.")
        return
    sessions = sorted({row.get("session_id") for row in recordings if row.get("session_id")})
    selected_session = st.selectbox("Filtrer par session", ["Toutes"] + sessions, index=0)
    if selected_session != "Toutes":
        recordings = [row for row in recordings if row.get("session_id") == selected_session]
    for row in recordings:
        row["exists"] = Path(row["segment_path"]).exists() or (ROOT_DIR / row["segment_path"]).exists()
        row["metadata"] = json.loads(row.get("metadata_json") or "{}")
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
