from __future__ import annotations

import pandas as pd
import streamlit as st

from visionsort.core.enums import CommandType
from visionsort.ui.components.common import demo_warning, page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Models", "Registre local, activation et rollback")
    demo_warning(context)
    models = context.repo.list_models()
    if not models:
        st.info("Aucun modèle en base.")
        return
    for row in models:
        with st.container(border=True):
            st.write(f"**{row['id']}** - {row['name']} - backend `{row['backend']}` - statut `{row['status']}` - actif `{row['is_active']}`")
            cols = st.columns(2)
            if cols[0].button("Activer", key=f"activate-{row['id']}"):
                context.repo.enqueue_command(CommandType.ACTIVATE_MODEL, {"model_id": row["id"]})
            if cols[1].button("Rollback champion", key=f"rollback-{row['id']}"):
                context.repo.enqueue_command(CommandType.ROLLBACK_MODEL, {})
    st.dataframe(pd.DataFrame(models), use_container_width=True)
