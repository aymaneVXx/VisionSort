from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from visionsort.core.enums import AnnotationStatus, CommandType
from visionsort.core.paths import ROOT_DIR
from visionsort.ui.components.common import demo_warning, page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Dataset Studio", "Sélection intelligente, pseudo-annotation et export YOLO")
    demo_warning(context)

    sessions = context.repo.list_capture_sessions()
    session_options = {f"{s['name']} ({s['id']})": s for s in sessions}
    selected = st.selectbox("CaptureSession", list(session_options) if session_options else [], index=0 if session_options else None)
    session = session_options[selected] if selected else None

    if session:
        st.write(f"État pipeline: `{session.get('pipeline_state')}` | demo={bool(session.get('demo_mode'))} | validé site={bool(session.get('site_validated'))}")
        steps = context.repo.list_pipeline_steps(session["id"])
        if steps:
            st.dataframe(pd.DataFrame(steps), use_container_width=True)
        model_ids = [row["id"] for row in context.repo.list_models()]

        col1, col2, col3, col4 = st.columns(4)
        dataset_name = col1.text_input("Nom dataset", value="dataset_session")
        pseudo_label_model_id = col4.selectbox("Modèle pseudo-label", model_ids, index=0 if model_ids else None)
        if col2.button("1) Sample"):
            context.repo.enqueue_command(
                CommandType.RUN_PIPELINE_STEP,
                {"session_id": session["id"], "step": "SAMPLE", "params": {"name": dataset_name}},
            )
            st.success("Step SAMPLE en file.")
        if col3.button("2) Auto-annotate"):
            context.repo.enqueue_command(
                CommandType.RUN_PIPELINE_STEP,
                {
                    "session_id": session["id"],
                    "step": "AUTO_ANNOTATE",
                    "params": {
                        "model_id": pseudo_label_model_id,
                        "fallback_model_id": "demo_synth_det",
                        "force": False,
                    },
                },
            )
            st.success("Step AUTO_ANNOTATE en file.")
        if st.button("3) Finalize dataset (DATASET_READY)"):
            context.repo.enqueue_command(
                CommandType.RUN_PIPELINE_STEP,
                {"session_id": session["id"], "step": "FINALIZE_DATASET", "params": {}},
            )
            st.success("Step FINALIZE_DATASET en file.")
        if st.button("Exporter observations (Parquet)"):
            context.repo.enqueue_command(
                CommandType.RUN_PIPELINE_STEP,
                {"session_id": session["id"], "step": "EXPORT_OBSERVATIONS_PARQUET", "params": {}},
            )
            st.success("Step EXPORT_OBSERVATIONS_PARQUET en file.")

    st.divider()
    datasets = context.repo.list_datasets()
    if not datasets:
        st.info("Aucun dataset généré pour l'instant.")
        return

    df = pd.DataFrame(datasets)
    st.dataframe(df, use_container_width=True)
    ds_options = {f"{row['name']} ({row['id']})": row for row in datasets}
    ds_sel = st.selectbox("Dataset", list(ds_options), index=0)
    ds = ds_options[ds_sel]
    items = context.repo.list_dataset_items(ds["id"])
    if not items:
        st.info("Aucun item dans ce dataset.")
        return

    summary = json.loads(ds.get("summary_json") or "{}")
    review_counts = summary.get("review_counts") or {}
    st.caption(
        f"Dataset status=`{ds.get('status')}` | "
        f"needs_review={int(review_counts.get('needs_review', 0))} | "
        f"auto_accepted={int(review_counts.get('auto_accepted', 0))} | "
        f"human_validated={int(review_counts.get('human_validated', 0))} | "
        f"rejected={int(review_counts.get('rejected', 0))}"
    )
    if ds.get("manifest_path"):
        st.caption(f"Manifest: `{ds['manifest_path']}`")

    needs_review = [it for it in items if it.get("annotation_status") == AnnotationStatus.NEEDS_REVIEW.value]
    st.subheader(f"Review ({len(needs_review)} NEEDS_REVIEW)")
    if needs_review:
        st.caption("Chaque action de review relance un recalcul de FINALIZE_DATASET pour mettre à jour automatiquement l'état REVIEW_PENDING/DATASET_READY.")
    for it in needs_review[:30]:
        with st.container(border=True):
            meta = json.loads(it.get("metadata_json") or "{}")
            st.write(f"Item `{it['id']}` | split={it.get('split')} | reason={it.get('reason')} | score={it.get('score')}")
            img_path = ROOT_DIR / it["image_path"]
            if img_path.exists():
                st.image(str(img_path), caption=str(Path(it["image_path"])))
            cols = st.columns(3)
            if cols[0].button("Accepter", key=f"accept-{it['id']}"):
                context.repo.enqueue_command(CommandType.UPDATE_DATASET_ITEM, {"item_id": it["id"], "annotation_status": AnnotationStatus.HUMAN_VALIDATED.value})
                st.info("Commande envoyée.")
            if cols[1].button("Rejeter", key=f"reject-{it['id']}"):
                context.repo.enqueue_command(CommandType.UPDATE_DATASET_ITEM, {"item_id": it["id"], "annotation_status": AnnotationStatus.REJECTED.value})
                st.info("Commande envoyée.")
            if cols[2].button("Auto-accept", key=f"autoaccept-{it['id']}"):
                context.repo.enqueue_command(CommandType.UPDATE_DATASET_ITEM, {"item_id": it["id"], "annotation_status": AnnotationStatus.AUTO_ACCEPTED.value})
                st.info("Commande envoyée.")
