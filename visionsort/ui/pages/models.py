from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from visionsort.core.paths import ROOT_DIR
from visionsort.core.enums import CommandType, ModelStatus
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
    page_header("Models", "Registre local, comparaison, promotion et rollback")
    demo_warning(context)
    models = context.repo.list_models()
    if not models:
        st.info("Aucun modèle en base.")
        return

    active_model = next((row for row in models if int(row.get("is_active") or 0) == 1), None)
    if active_model:
        st.caption(f"Modèle actif: `{active_model['id']}` | statut `{active_model['status']}`")

    for row in models:
        notes = _load_json(row.get("notes_json"))
        metrics = _load_json(row.get("metrics_json"))
        is_active = int(row.get("is_active") or 0) == 1
        is_candidate = row["status"] == ModelStatus.CANDIDATE.value
        can_promote = is_candidate and not is_active
        can_reject = row["status"] not in {ModelStatus.CHAMPION.value, ModelStatus.REJECTED.value}
        can_archive = row["status"] not in {ModelStatus.ARCHIVED.value}
        with st.container(border=True):
            st.write(
                f"**{row['id']}** - {row['name']} - backend `{row['backend']}` - "
                f"statut `{row['status']}` - actif `{is_active}`"
            )
            info_cols = st.columns(4)
            info_cols[0].metric("Precision", _metric_text(metrics.get("precision")))
            info_cols[1].metric("Recall", _metric_text(metrics.get("recall")))
            info_cols[2].metric("mAP50", _metric_text(metrics.get("mAP50")))
            info_cols[3].metric("FPS", _metric_text(metrics.get("fps"), 2))
            if notes:
                st.caption(
                    f"demo_only={bool(notes.get('demo_only'))} | "
                    f"validated_on_site={bool(notes.get('validated_on_site'))} | "
                    f"parent={metrics.get('parent_model_id') or row.get('parent_model_id') or '-'}"
                )
            comparison = metrics.get("comparison") or {}
            benchmark = metrics.get("benchmark") or {}
            if active_model and active_model["id"] != row["id"]:
                active_metrics = _load_json(active_model.get("metrics_json"))
                deltas = {
                    key: float(metrics[key]) - float(active_metrics[key])
                    for key in ("precision", "recall", "mAP50", "mAP50_95", "fps")
                    if metrics.get(key) is not None
                    and active_metrics.get(key) is not None
                }
                st.caption(
                    "Delta vs actif: "
                    + ", ".join(f"{key}={value:+.3f}" for key, value in deltas.items())
                )
            st.caption(
                f"evaluation={metrics.get('evaluation_status', '-')} | "
                f"benchmark={benchmark.get('status') or '-'} | "
                f"compare_to={comparison.get('against_model_id') or '-'}"
            )
            report_path = metrics.get("report_path") or notes.get("report_path")
            if report_path:
                report_abs = ROOT_DIR / str(report_path)
                if Path(report_abs).exists():
                    with st.expander(f"Rapport {row['id']}"):
                        st.code(report_abs.read_text(encoding="utf-8"), language="json")
            cols = st.columns(5)
            can_activate = row["status"] in {ModelStatus.CHAMPION.value, ModelStatus.ARCHIVED.value}
            if cols[0].button("Activer", key=f"activate-{row['id']}", disabled=is_active or not can_activate):
                context.repo.enqueue_command(CommandType.ACTIVATE_MODEL, {"model_id": row["id"]})
            if cols[1].button("Promouvoir", key=f"promote-{row['id']}", disabled=not can_promote):
                context.repo.enqueue_command(CommandType.PROMOTE_MODEL, {"model_id": row["id"]})
            if cols[2].button("Rejeter", key=f"reject-model-{row['id']}", disabled=not can_reject):
                context.repo.enqueue_command(CommandType.REJECT_MODEL, {"model_id": row["id"]})
            if cols[3].button("Archiver", key=f"archive-{row['id']}", disabled=not can_archive):
                context.repo.enqueue_command(CommandType.ARCHIVE_MODEL, {"model_id": row["id"]})
    st.divider()
    if st.button("Rollback vers précédent actif", type="secondary"):
        context.repo.enqueue_command(CommandType.ROLLBACK_MODEL, {})
    st.dataframe(pd.DataFrame(models), use_container_width=True)
