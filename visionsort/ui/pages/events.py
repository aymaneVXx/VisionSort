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
    df["payload_json"] = df["payload_json"].apply(lambda value: json.dumps(json.loads(value), ensure_ascii=False)[:250])
    st.dataframe(df, use_container_width=True)
