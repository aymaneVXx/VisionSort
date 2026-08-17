from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from visionsort.calibration.repository import CalibrationRepository
from visionsort.core.enums import CommandType
from visionsort.ui.components.common import (
    demo_warning,
    event_label,
    online_source,
    page_header,
    request_navigation,
    status_badge,
)
from visionsort.ui.state import UIContext


def _today_parcels(parcels: list[dict]) -> list[dict]:
    start = datetime.now().astimezone().replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()
    return [
        row
        for row in parcels
        if float(row.get("last_seen_at") or 0.0) >= start
    ]


def render(context: UIContext) -> None:
    page_header(
        "Accueil",
        "L’essentiel pour vérifier le système et lancer l’exploitation.",
    )
    demo_warning(context)
    sources = context.repo.list_sources()
    sessions = context.repo.list_capture_sessions()
    parcels = context.repo.list_parcels()
    today = _today_parcels(parcels)
    models = context.repo.list_models()
    online = [row for row in sources if online_source(row.get("status"))]
    active_models = [row for row in models if int(row.get("is_active") or 0)]
    calibration_repo = CalibrationRepository(context.db)
    calibrated = [
        row
        for row in sources
        if calibration_repo.get_active_profile(str(row["id"])) is not None
    ]
    running = next(
        (
            row
            for row in sessions
            if row.get("started_at") is not None and row.get("ended_at") is None
        ),
        None,
    )

    st.subheader("État du système")
    state_columns = st.columns(3)
    with state_columns[0].container(border=True):
        st.markdown(
            status_badge("LIVE" if online else "OFFLINE"),
            unsafe_allow_html=True,
        )
        st.metric("Caméras actives", f"{len(online)} / {len(sources)}")
    with state_columns[1].container(border=True):
        st.markdown(
            status_badge("READY" if active_models else "ERROR"),
            unsafe_allow_html=True,
        )
        st.metric("Modèles opérationnels", len(active_models))
    with state_columns[2].container(border=True):
        calibration_status = (
            "READY" if sources and len(calibrated) == len(sources) else "DEGRADED"
        )
        st.markdown(status_badge(calibration_status), unsafe_allow_html=True)
        st.metric("Caméras calibrées", f"{len(calibrated)} / {len(sources)}")

    st.subheader("Aujourd’hui")
    metrics = st.columns(4)
    metrics[0].metric("Colis suivis", len(today))
    metrics[1].metric(
        "Triés correctement",
        sum(row.get("destination_result") == "SORT_OK" for row in today),
    )
    metrics[2].metric(
        "Mauvaises destinations",
        sum(row.get("destination_result") == "WRONG_DESTINATION" for row in today),
    )
    metrics[3].metric(
        "Non vérifiés",
        sum(
            row.get("destination_result") == "DESTINATION_UNVERIFIED"
            for row in today
        ),
    )

    st.subheader("Actions rapides")
    actions = st.columns(4)
    start_target = sessions[0] if sessions else None
    if actions[0].button(
        "Démarrer le système",
        type="primary",
        use_container_width=True,
        disabled=running is not None or start_target is None,
    ):
        context.repo.enqueue_command(
            CommandType.START_SESSION,
            {"session_id": start_target["id"]},
        )
        st.success("Démarrage demandé.")
    if actions[1].button(
        "Arrêter",
        use_container_width=True,
        disabled=running is None,
    ):
        context.repo.enqueue_command(
            CommandType.STOP_SESSION,
            {"session_id": running["id"]},
        )
        st.info("Arrêt demandé.")
    if actions[2].button("Configurer le site", use_container_width=True):
        request_navigation("Configuration du site")
    if actions[3].button("Ajouter une caméra", use_container_width=True):
        request_navigation("Caméras")

    recent_events = context.repo.recent_events(limit=30)
    alerts = [
        row
        for row in recent_events
        if str(row.get("severity") or "").lower()
        in {"warning", "error", "critical"}
        or row.get("event_type")
        in {"WRONG_DESTINATION", "HANDOFF_AMBIGUOUS", "HANDOFF_UNRESOLVED"}
    ]
    st.subheader("Dernières alertes")
    if alerts:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Heure": row.get("created_at"),
                        "Information": event_label(row.get("event_type")),
                        "Caméra": row.get("camera_id")
                        or row.get("source_id")
                        or "—",
                        "Colis": row.get("parcel_id") or "—",
                    }
                    for row in alerts[:8]
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.success("Aucune alerte récente.")
