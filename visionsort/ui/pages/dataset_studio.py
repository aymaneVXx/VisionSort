from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from visionsort.annotations import (
    export_review_cases,
    import_review_annotations,
    render_review_overlay,
)
from visionsort.annotations.validators import PoseLabelValidator
from visionsort.core.enums import AnnotationStatus, CommandType
from visionsort.core.paths import ROOT_DIR
from visionsort.database.repositories import ArtifactRepository
from visionsort.ui.components.common import demo_warning, page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Dataset Studio", "Sélection intelligente, pseudo-annotation et export YOLO")
    demo_warning(context)

    sessions = context.repo.list_capture_sessions()
    completed_sessions = [
        session for session in sessions if session.get("ended_at") is not None
    ]
    session_options = {
        f"{session['name']} ({session['id']})": session
        for session in completed_sessions
    }
    selected_labels = st.multiselect(
        "CaptureSessions terminées",
        list(session_options),
        help="Une session complète reste dans un seul split.",
    )
    selected_sessions = [session_options[label] for label in selected_labels]
    session = selected_sessions[0] if selected_sessions else None

    if selected_sessions:
        details: list[dict[str, object]] = []
        for selected_session in selected_sessions:
            sources = context.repo.list_capture_session_sources(
                selected_session["id"]
            )
            started_at = selected_session.get("started_at")
            ended_at = selected_session.get("ended_at")
            details.append(
                {
                    "session_id": selected_session["id"],
                    "name": selected_session["name"],
                    "status": selected_session.get("pipeline_state"),
                    "cameras": ", ".join(
                        str(source.get("camera_role")) for source in sources
                    ),
                    "duration_s": (
                        round(float(ended_at) - float(started_at), 2)
                        if started_at is not None and ended_at is not None
                        else None
                    ),
                }
            )
        st.dataframe(pd.DataFrame(details), use_container_width=True)
        steps = context.repo.list_pipeline_steps(session["id"])
        if steps:
            st.dataframe(pd.DataFrame(steps), use_container_width=True)
        models = context.repo.list_models()

        col1, col2, col3, col4 = st.columns(4)
        dataset_name = col1.text_input("Nom dataset", value="dataset_project")
        dataset_task = st.selectbox(
            "Tâche immuable du dataset",
            [
                "detection",
                "segmentation",
                "pose",
                "local_tracking",
                "reid_multicamera",
            ],
        )
        pseudo_label_task = (
            dataset_task
            if dataset_task in {"detection", "segmentation", "pose"}
            else "detection"
        )
        model_ids = [
            row["id"] for row in models if row.get("task") == pseudo_label_task
        ]
        pseudo_label_model_id = col4.selectbox("Modèle pseudo-label", model_ids, index=0 if model_ids else None)
        automatic_splits = st.checkbox(
            "Affectation automatique des splits",
            value=True,
            help="Avec au moins trois sessions, train, val et test sont tous attribués.",
        )
        split_assignments: dict[str, str] = {}
        if not automatic_splits:
            split_columns = st.columns(3)
            for index, selected_session in enumerate(selected_sessions):
                split_assignments[selected_session["id"]] = split_columns[
                    index % 3
                ].selectbox(
                    f"Split — {selected_session['name']}",
                    ["train", "val", "test"],
                    key=f"split-{selected_session['id']}",
                )
        if col2.button("1) Sample"):
            context.repo.enqueue_command(
                CommandType.RUN_PIPELINE_STEP,
                {
                    "session_id": session["id"],
                    "step": "SAMPLE",
                    "params": {
                        "name": dataset_name,
                        "session_ids": [
                            selected_session["id"]
                            for selected_session in selected_sessions
                        ],
                        "split_assignments": split_assignments,
                        "task": dataset_task,
                    },
                },
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
    if not needs_review:
        return

    filter_columns = st.columns(4)
    session_filter = filter_columns[0].selectbox(
        "Session",
        ["Toutes"]
        + sorted(
            {
                str(item.get("session_id"))
                for item in needs_review
                if item.get("session_id")
            }
        ),
    )
    camera_filter = filter_columns[1].selectbox(
        "Caméra",
        ["Toutes"]
        + sorted(
            {
                str(item.get("source_id"))
                for item in needs_review
                if item.get("source_id")
            }
        ),
    )
    task_filter = filter_columns[2].selectbox(
        "Tâche", ["Toutes", str(ds.get("task") or "detection")]
    )
    reason_filter = filter_columns[3].selectbox(
        "Raison",
        ["Toutes"]
        + sorted(
            {
                str(item.get("reason"))
                for item in needs_review
                if item.get("reason")
            }
        ),
    )
    filtered = [
        item
        for item in needs_review
        if (session_filter == "Toutes" or item.get("session_id") == session_filter)
        and (camera_filter == "Toutes" or item.get("source_id") == camera_filter)
        and (task_filter == "Toutes" or str(ds.get("task")) == task_filter)
        and (reason_filter == "Toutes" or item.get("reason") == reason_filter)
    ]
    if not filtered:
        st.info("Aucun cas ne correspond aux filtres.")
        return

    export_columns = st.columns(2)
    export_columns[0].download_button(
        "Exporter vers Label Studio",
        data=export_review_cases(filtered, export_format="label_studio"),
        file_name=f"{ds['id']}_needs_review_label_studio.json",
        mime="application/json",
    )
    export_columns[1].download_button(
        "Exporter vers CVAT",
        data=export_review_cases(filtered, export_format="cvat"),
        file_name=f"{ds['id']}_needs_review_cvat.xml",
        mime="application/xml",
    )
    uploaded = st.file_uploader(
        "Réimporter des annotations corrigées",
        type=["json", "xml"],
        key=f"review-import-{ds['id']}",
    )
    if uploaded is not None and st.button(
        "Importer et valider humainement", key=f"import-{ds['id']}"
    ):
        try:
            result = import_review_annotations(
                context.db,
                ArtifactRepository(context.db),
                dataset_id=ds["id"],
                content=uploaded.getvalue(),
                filename=uploaded.name,
            )
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.success(
                f"{result['updated_items']} item(s) réimporté(s)."
            )

    position = int(
        st.number_input(
            "Cas affiché",
            min_value=1,
            max_value=len(filtered),
            value=1,
            step=1,
        )
    )
    item = filtered[position - 1]
    with st.container(border=True):
        overlay, details = render_review_overlay(item)
        st.write(
            f"Item `{item['id']}` — {position}/{len(filtered)} | "
            f"split={item.get('split')} | raison={item.get('reason')}"
        )
        st.image(
            overlay,
            caption=str(Path(item["image_path"])),
            use_container_width=True,
        )
        st.write(
            f"Attendu: **{details['expected_count']}** | "
            f"annoté: **{details['annotated_count']}** | "
            f"tâche: `{details['task'] or ds.get('task')}`"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "classe": detection.get("class_name"),
                        "confiance": detection.get("confidence"),
                        "bbox": detection.get("bbox"),
                        "masque": bool(detection.get("mask")),
                        "keypoints": len(detection.get("keypoints") or []),
                    }
                    for detection in details["detections"]
                ]
            ),
            use_container_width=True,
        )
        st.json(
            {
                "quality_gate": details["quality_stats"],
                "provenance": details["provenance"],
            }
        )
        pose_report = None
        if str(ds.get("task")) == "pose":
            label_path = item.get("label_path")
            if label_path:
                pose_report = PoseLabelValidator(
                    str(ds.get("data_yaml_path") or "")
                ).validate(str(label_path))
            if pose_report is None or not pose_report.valid:
                st.error("Validation Pose impossible pour cet item.")
                for error in (
                    pose_report.errors
                    if pose_report is not None
                    else ["Aucun fichier label Pose."]
                ):
                    st.caption(error)
        actions = st.columns(2)
        if actions[0].button(
            "Valider humainement",
            key=f"accept-{item['id']}",
            disabled=bool(
                pose_report is not None and not pose_report.valid
            )
            or (
                str(ds.get("task")) == "pose"
                and pose_report is None
            ),
        ):
            context.repo.enqueue_command(
                CommandType.UPDATE_DATASET_ITEM,
                {
                    "item_id": item["id"],
                    "annotation_status": AnnotationStatus.HUMAN_VALIDATED.value,
                },
            )
            st.info("Commande envoyée.")
        if actions[1].button("Rejeter", key=f"reject-{item['id']}"):
            context.repo.enqueue_command(
                CommandType.UPDATE_DATASET_ITEM,
                {
                    "item_id": item["id"],
                    "annotation_status": AnnotationStatus.REJECTED.value,
                },
            )
            st.info("Commande envoyée.")
