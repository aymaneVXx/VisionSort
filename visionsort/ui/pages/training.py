from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from visionsort.core.enums import CommandType
from visionsort.core.paths import ROOT_DIR
from visionsort.ui.components.common import demo_warning, page_header
from visionsort.ui.state import UIContext


def _load_json(text: str | None) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def render(context: UIContext) -> None:
    page_header("Training", "Lancement et suivi des recettes d'entraînement")
    demo_warning(context)
    datasets = [row for row in context.repo.list_datasets() if row.get("status") in {"DATASET_READY"}]
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
    if not jobs:
        st.dataframe(pd.DataFrame(columns=["id", "status"]), use_container_width=True)
        return

    for job in jobs[:15]:
        metrics = _load_json(job.get("metrics_json"))
        comparison = metrics.get("comparison") or {}
        benchmark = metrics.get("benchmark") or {}
        report_path = metrics.get("report_path")
        with st.container(border=True):
            st.write(f"**{job['id']}** - statut `{job['status']}` - dataset `{job['dataset_id']}` - modèle `{job['model_id']}`")
            info_cols = st.columns(4)
            info_cols[0].metric("Precision", f"{float(metrics.get('precision', 0.0)):.3f}")
            info_cols[1].metric("Recall", f"{float(metrics.get('recall', 0.0)):.3f}")
            info_cols[2].metric("mAP50", f"{float(metrics.get('mAP50', 0.0)):.3f}")
            info_cols[3].metric("FPS", f"{float(metrics.get('fps', 0.0)):.2f}")
            st.caption(
                f"evaluation={metrics.get('evaluation_status', '-')} | "
                f"candidate={metrics.get('candidate_status', '-')} | "
                f"compare_to={comparison.get('against_model_id') or '-'} | "
                f"benchmark={benchmark.get('status') or '-'}"
            )
            deltas = comparison.get("deltas") or {}
            if deltas:
                st.caption("Delta vs actif: " + ", ".join(f"{key}={float(value):+.3f}" for key, value in deltas.items()))
            if report_path:
                report_abs = ROOT_DIR / report_path
                if report_abs.exists():
                    st.caption(f"Rapport: `{report_path}`")
                    with st.expander("Voir le rapport JSON"):
                        st.code(report_abs.read_text(encoding='utf-8'), language="json")
            if job.get("error_text"):
                st.error(str(job["error_text"]))
            if job["status"] in {"QUEUED", "RUNNING"}:
                if st.button("Annuler le job", key=f"cancel-training-{job['id']}"):
                    context.repo.enqueue_command(
                        CommandType.CANCEL_JOB,
                        {"job_type": "TRAINING", "job_key": job["id"]},
                    )
                    st.info("Commande d'annulation envoyée.")

    st.dataframe(pd.DataFrame(jobs), use_container_width=True)
