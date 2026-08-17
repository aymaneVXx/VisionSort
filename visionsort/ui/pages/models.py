from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from visionsort.core.enums import CommandType, ModelStatus
from visionsort.ui.components.common import (
    demo_warning,
    page_header,
    render_status,
    status_label,
)
from visionsort.ui.state import UIContext


_TASK_LABELS = {
    "detection": "Détection des colis",
    "segmentation": "Contour des colis",
    "pose": "Suivi des opérateurs",
    "local_tracking": "Suivi dans une caméra",
    "reid_multicamera": "Ré-identification entre caméras",
}


def _load_json(text: str | None) -> dict:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _metric(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def render(context: UIContext) -> None:
    page_header(
        "Modèles",
        "Vérifiez les modèles actifs et gérez leur cycle de vie sans interrompre l’exploitation.",
    )
    demo_warning(context)
    models = context.repo.list_models()
    if not models:
        st.info("Aucun modèle disponible.")
        return
    overview_tab, management_tab = st.tabs(["Vue générale", "Gestion avancée"])

    with overview_tab:
        active_models = [row for row in models if int(row.get("is_active") or 0)]
        runtime_state = context.repo.get_runtime_model_state()
        consistency = runtime_state.get("consistency") or {}
        if consistency.get("consistent"):
            st.success("Les modèles actifs sont chargés et opérationnels.")
        elif runtime_state:
            st.warning("Le chargement des modèles doit être vérifié dans Diagnostic.")
        else:
            st.info("L’état du moteur n’est pas encore disponible.")

        columns = st.columns(2)
        for index, model in enumerate(active_models):
            metrics = _load_json(model.get("metrics_json"))
            with columns[index % 2].container(border=True):
                st.markdown(
                    f"### {_TASK_LABELS.get(str(model.get('task')), model.get('task'))}"
                )
                render_status("READY")
                st.write(f"Modèle actif : **{model.get('name') or model.get('id')}**")
                values = st.columns(3)
                values[0].metric("Précision", _metric(metrics.get("precision")))
                values[1].metric("Rappel", _metric(metrics.get("recall")))
                values[2].metric("Images/s", _metric(metrics.get("fps"), 1))
        if not active_models:
            st.warning("Aucun modèle actif.")

        reid_model = next(
            (
                row
                for row in active_models
                if row.get("task") == "reid_multicamera"
            ),
            None,
        )
        adaptation = context.repo.latest_reid_adaptation_run()
        reid_config = (
            (context.config_values.get("tracking") or {}).get("reid") or {}
        )
        with st.container(border=True):
            st.markdown("### Adaptation au site")
            if not bool(reid_config.get("auto_adaptation", True)):
                render_status("OFFLINE")
                st.write("Adaptation automatique désactivée")
            elif adaptation:
                render_status(adaptation.get("state"))
                st.write(status_label(adaptation.get("state")))
            else:
                render_status("COLLECTING")
                st.write("Collecte d’exemples fiables en cours")
            st.caption(
                f"Base visuelle : {reid_model.get('name') if reid_model else 'MobileNetV3'}"
            )

    with management_tab:
        st.warning(
            "Ces actions modifient le modèle utilisé par le système. "
            "Elles sont destinées à un ingénieur habilité."
        )
        with st.form("import-baseline-model"):
            st.markdown("#### Importer un modèle de référence")
            import_columns = st.columns(3)
            baseline_path = import_columns[0].text_input("Fichier de poids local (.pt)")
            baseline_name = import_columns[1].text_input("Nom")
            baseline_task = import_columns[2].selectbox(
                "Usage", ["detection", "segmentation", "pose"]
            )
            if st.form_submit_button(
                "Importer",
                disabled=not baseline_path.strip(),
            ):
                context.repo.enqueue_command(
                    CommandType.IMPORT_BASELINE_MODEL,
                    {
                        "source_path": baseline_path.strip(),
                        "name": baseline_name.strip() or None,
                        "task": baseline_task,
                    },
                )
                st.info("Import envoyé.")

        for row in models:
            metrics = _load_json(row.get("metrics_json"))
            notes = _load_json(row.get("notes_json"))
            is_active = int(row.get("is_active") or 0) == 1
            is_candidate = row.get("status") == ModelStatus.CANDIDATE.value
            with st.container(border=True):
                title = st.columns([3, 1])
                title[0].markdown(
                    f"**{row.get('name') or row['id']}**  \n"
                    f"{_TASK_LABELS.get(str(row.get('task')), row.get('task'))}"
                )
                with title[1]:
                    render_status("ACTIVE" if is_active else row.get("status"))
                metric_columns = st.columns(4)
                metric_columns[0].metric("Précision", _metric(metrics.get("precision")))
                metric_columns[1].metric("Rappel", _metric(metrics.get("recall")))
                metric_columns[2].metric("mAP50", _metric(metrics.get("mAP50")))
                metric_columns[3].metric("Images/s", _metric(metrics.get("fps"), 1))
                controls = st.columns(4)
                can_activate = row.get("status") in {
                    ModelStatus.BASELINE.value,
                    ModelStatus.CHAMPION.value,
                    ModelStatus.ARCHIVED.value,
                }
                if controls[0].button(
                    "Activer",
                    key=f"activate-{row['id']}",
                    disabled=is_active or not can_activate,
                    use_container_width=True,
                ):
                    context.repo.enqueue_command(
                        CommandType.ACTIVATE_MODEL, {"model_id": row["id"]}
                    )
                if controls[1].button(
                    "Promouvoir",
                    key=f"promote-{row['id']}",
                    disabled=not is_candidate or is_active,
                    use_container_width=True,
                ):
                    context.repo.enqueue_command(
                        CommandType.PROMOTE_MODEL, {"model_id": row["id"]}
                    )
                if controls[2].button(
                    "Rejeter",
                    key=f"reject-model-{row['id']}",
                    disabled=is_active
                    or row.get("status")
                    in {ModelStatus.CHAMPION.value, ModelStatus.REJECTED.value},
                    use_container_width=True,
                ):
                    context.repo.enqueue_command(
                        CommandType.REJECT_MODEL, {"model_id": row["id"]}
                    )
                if controls[3].button(
                    "Archiver",
                    key=f"archive-{row['id']}",
                    disabled=is_active or row.get("status") == ModelStatus.ARCHIVED.value,
                    use_container_width=True,
                ):
                    context.repo.enqueue_command(
                        CommandType.ARCHIVE_MODEL, {"model_id": row["id"]}
                    )
                st.caption(
                    f"ID : {row['id']} · backend : {row.get('backend')} · "
                    f"parent : {row.get('parent_model_id') or '—'} · "
                    f"validation site : {bool(notes.get('validated_on_site'))}"
                )

        st.subheader("Restaurer une version précédente")
        rollback_task = st.selectbox(
            "Fonction concernée",
            sorted({str(row.get("task")) for row in models}),
            format_func=lambda value: _TASK_LABELS.get(value, value),
        )
        if st.button("Restaurer le précédent modèle actif", type="secondary"):
            context.repo.enqueue_command(
                CommandType.ROLLBACK_MODEL, {"task": rollback_task}
            )

        history = context.repo.list_model_activation_history(limit=100)
        st.subheader("Historique des changements")
        st.dataframe(
            pd.DataFrame(history)
            if history
            else pd.DataFrame(columns=["task", "status", "activated_at"]),
            hide_index=True,
            use_container_width=True,
        )
        st.subheader("État technique du moteur")
        st.json(context.repo.get_runtime_model_state())
        st.dataframe(pd.DataFrame(models), use_container_width=True)
