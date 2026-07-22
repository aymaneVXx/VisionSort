from __future__ import annotations

import pandas as pd
import streamlit as st

from visionsort.core.enums import CommandType
from visionsort.ui.components.common import demo_warning, page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Training", "Lancement et suivi des recettes d'entraînement")
    demo_warning(context)
    datasets = context.repo.list_datasets()
    models = context.repo.list_models()
    if datasets and models:
        with st.form("training_form"):
            dataset_id = st.selectbox("Dataset", [row["id"] for row in datasets])
            model_id = st.selectbox("Modèle initial", [row["id"] for row in models])
            task = st.selectbox("Tâche", ["detection", "segmentation", "pose"])
            architecture = st.text_input("Architecture YOLO", value="yolo11n")
            imgsz = st.number_input("imgsz", 320, 1280, 640, 32)
            epochs = st.number_input("epochs", 1, 300, 5)
            batch = st.number_input("batch", 1, 64, 4)
            device = st.text_input("device", value="cpu")
            patience = st.number_input("patience", 1, 100, 10)
            mode = st.selectbox("Mode", ["demo", "ultralytics"])
            if st.form_submit_button("Lancer l'entraînement"):
                context.repo.enqueue_command(
                    CommandType.START_TRAINING,
                    {
                        "dataset_id": dataset_id,
                        "model_id": model_id,
                        "task": task,
                        "architecture": architecture,
                        "imgsz": int(imgsz),
                        "epochs": int(epochs),
                        "batch": int(batch),
                        "device": device,
                        "patience": int(patience),
                        "mode": mode,
                    },
                )
                st.success("Commande d'entraînement envoyée.")
    else:
        st.info("Il faut au moins un dataset et un modèle.")
    jobs = context.repo.list_training_jobs()
    st.dataframe(pd.DataFrame(jobs) if jobs else pd.DataFrame(columns=["id", "status"]), use_container_width=True)
