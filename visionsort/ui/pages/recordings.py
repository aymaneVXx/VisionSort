from __future__ import annotations

import pandas as pd
import streamlit as st

from visionsort.ui.components.common import page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Recordings", "Segments enregistrés et métadonnées")
    recordings = context.repo.list_recordings()
    if recordings:
        st.dataframe(pd.DataFrame(recordings), use_container_width=True)
    else:
        st.info("Aucun segment enregistré.")
