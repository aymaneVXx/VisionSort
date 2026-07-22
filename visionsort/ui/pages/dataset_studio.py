from __future__ import annotations

import pandas as pd
import streamlit as st

from visionsort.core.enums import CommandType
from visionsort.ui.components.common import demo_warning, page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Dataset Studio", "Sélection intelligente, pseudo-annotation et export YOLO")
    demo_warning(context)
    col1, col2 = st.columns(2)
    dataset_name = col1.text_input("Nom du dataset", value="autodataset_replay")
    if col2.button("Créer un dataset depuis les tracklets"):
        context.repo.enqueue_command(CommandType.CREATE_DATASET, {"name": dataset_name})
        st.success("Commande de création de dataset envoyée.")
    datasets = context.repo.list_datasets()
    if datasets:
        st.dataframe(pd.DataFrame(datasets), use_container_width=True)
    else:
        st.info("Aucun dataset généré pour l'instant.")
