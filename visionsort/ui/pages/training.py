from __future__ import annotations

import json

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


def _metric_text(value, digits: int = 3) -> str:
    return "UNAVAILABLE" if value is None else f"{float(value):.{digits}f}"


def render(context: UIContext) -> None:
    page_header(
        "Entraînement",
        "Lancez et suivez l’amélioration contrôlée des modèles.",
        eyebrow="Outils avancés",
    )
    demo_warning(context)
    datasets = [
        row
        for row in context.repo.list_datasets()
        if row.get("status") == "DATASET_READY"
        and row.get("task") in {"detection", "segmentation", "pose"}
    ]
    models = context.repo.list_models()
    if datasets and models:
        with st.form("training_form"):
            dataset_id = st.selectbox(
                "Dataset", [row["id"] for row in datasets]
            )
            selected_dataset = next(
                row for row in datasets if row["id"] == dataset_id
            )
            task = str(selected_dataset.get("task") or "detection")
            compatible_models = [
                row["id"]
                for row in models
                if str(row.get("task")) == task
            ]
            model_id = st.selectbox(
                "Modele initial compatible",
                compatible_models,
                index=0 if compatible_models else None,
            )
            st.caption(f"Tache imposee par le dataset: `{task}`")
            imgsz = st.number_input("imgsz", 320, 1280, 640, 32)
            epochs = st.number_input("epochs", 1, 300, 5)
            batch = st.number_input("batch", 1, 64, 4)
            device = st.text_input("device", value="cpu")
            patience = st.number_input("patience", 1, 100, 10)
            mode = st.selectbox("Mode", ["demo", "ultralytics"])
            if st.form_submit_button(
                "Lancer l'entrainement", disabled=not compatible_models
            ):
                context.repo.enqueue_command(
                    CommandType.START_TRAINING,
                    {
                        "dataset_id": dataset_id,
                        "model_id": model_id,
                        "task": task,
                        "imgsz": int(imgsz),
                        "epochs": int(epochs),
                        "batch": int(batch),
                        "device": device,
                        "patience": int(patience),
                        "mode": mode,
                    },
                )
                st.success("Commande d'entrainement envoyee.")
    else:
        st.info("Il faut un dataset DATASET_READY et un modele de meme tache.")

    jobs = context.repo.list_training_jobs()
    if not jobs:
        st.dataframe(
            pd.DataFrame(columns=["id", "status"]),
            use_container_width=True,
        )
        return

    for job in jobs[:15]:
        metrics = _load_json(job.get("metrics_json"))
        comparison = metrics.get("comparison") or {}
        benchmark = metrics.get("benchmark") or {}
        report_path = metrics.get("report_path")
        with st.container(border=True):
            st.write(
                f"**{job['id']}** - statut `{job['status']}` - "
                f"dataset `{job['dataset_id']}` - modele `{job['model_id']}`"
            )
            task = str(metrics.get("task") or "detection")
            if metrics.get("mask_precision") is not None:
                precision_key, recall_key, map_key = (
                    "mask_precision",
                    "mask_recall",
                    "mask_mAP50",
                )
                task = "segmentation"
            elif metrics.get("pose_precision") is not None:
                precision_key, recall_key, map_key = (
                    "pose_precision",
                    "pose_recall",
                    "pose_mAP50",
                )
                task = "pose"
            else:
                precision_key, recall_key, map_key = (
                    "precision",
                    "recall",
                    "mAP50",
                )
            info_cols = st.columns(4)
            info_cols[0].metric(
                f"Precision {task}", _metric_text(metrics.get(precision_key))
            )
            info_cols[1].metric(
                f"Recall {task}", _metric_text(metrics.get(recall_key))
            )
            info_cols[2].metric(
                f"mAP50 {task}", _metric_text(metrics.get(map_key))
            )
            info_cols[3].metric(
                "FPS", _metric_text(metrics.get("fps"), 2)
            )
            st.caption(
                f"evaluation={metrics.get('evaluation_status', '-')} | "
                f"candidate={metrics.get('candidate_status', '-')} | "
                f"compare_to={comparison.get('against_model_id') or '-'} | "
                f"test={benchmark.get('source') or benchmark.get('status') or '-'}"
            )
            deltas = comparison.get("deltas") or {}
            if deltas:
                st.caption(
                    "Delta vs actif: "
                    + ", ".join(
                        f"{key}={float(value):+.3f}"
                        for key, value in deltas.items()
                    )
                )
            if report_path:
                report_abs = ROOT_DIR / report_path
                if report_abs.exists():
                    st.caption(f"Rapport: `{report_path}`")
                    with st.expander("Voir le rapport JSON"):
                        st.code(
                            report_abs.read_text(encoding="utf-8"),
                            language="json",
                        )
            if job.get("error_text"):
                st.error(str(job["error_text"]))
            if job["status"] in {"QUEUED", "RUNNING"}:
                if st.button(
                    "Annuler le job", key=f"cancel-training-{job['id']}"
                ):
                    context.repo.enqueue_command(
                        CommandType.CANCEL_JOB,
                        {"job_type": "TRAINING", "job_key": job["id"]},
                    )
                    st.info("Commande d'annulation envoyee.")

    st.dataframe(pd.DataFrame(jobs), use_container_width=True)
