from __future__ import annotations

import pandas as pd
import streamlit as st

from visionsort.ui.components.common import demo_warning, page_header, show_preview
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Live Tracking", "Previews annotées, tracks locaux et colis globaux")
    demo_warning(context)
    sources = context.repo.list_sources()
    if not sources:
        st.info("Aucune source à afficher.")
        return
    cols = st.columns(min(3, len(sources)))
    for idx, source in enumerate(sources):
        with cols[idx % len(cols)]:
            show_preview(source.get("preview_path"), f"{source['role']} - {source.get('status') or 'OFFLINE'}")
            st.caption(f"FPS: {source.get('fps') or 0:.2f} | Modèle: {source['model_id']} | Tracker: {source['tracker_id']}")
    st.subheader("Tracklets")
    st.dataframe(pd.DataFrame(context.repo.list_tracklets()), use_container_width=True)
    st.subheader("Colis globaux")
    parcels = context.repo.list_parcels()
    st.dataframe(pd.DataFrame(parcels) if parcels else pd.DataFrame(columns=["parcel_id", "state"]), use_container_width=True)
