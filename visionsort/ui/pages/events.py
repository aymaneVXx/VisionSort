from __future__ import annotations

from datetime import datetime
import json

import pandas as pd
import streamlit as st

from visionsort.core.enums import CommandType
from visionsort.ui.components.common import (
    event_label,
    page_header,
    short_identifier,
    status_badge,
    status_label,
)
from visionsort.ui.state import UIContext


def _date(value: object):
    try:
        return datetime.fromtimestamp(float(value)).astimezone().date()
    except (TypeError, ValueError, OSError):
        return None


def _time(value: object) -> str:
    try:
        return datetime.fromtimestamp(float(value)).astimezone().strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "—"


def render(context: UIContext) -> None:
    page_header(
        "Historique & alertes",
        "Retrouvez les résultats du tri et la chronologie de chaque colis.",
    )
    hypotheses = context.repo.list_handoff_hypotheses("PENDING")
    if hypotheses:
        st.subheader(f"À confirmer ({len(hypotheses)})")
    for hypothesis in hypotheses:
        candidates = json.loads(hypothesis.get("candidates_json") or "[]")
        with st.container(border=True):
            st.markdown(status_badge("AMBIGUOUS"), unsafe_allow_html=True)
            st.write(
                "Le système hésite entre plusieurs passages pour un colis. "
                "Choisissez le suivi précédent uniquement si vous pouvez le confirmer."
            )
            candidate_ids = [
                str(candidate["from_tracklet_id"]) for candidate in candidates
            ]
            selected_candidate = st.selectbox(
                "Suivi précédent",
                candidate_ids,
                key=f"candidate-{hypothesis['id']}",
            )
            columns = st.columns(2)
            if columns[0].button(
                "Confirmer ce passage",
                key=f"resolve-{hypothesis['id']}",
                disabled=not selected_candidate,
                type="primary",
            ):
                context.repo.enqueue_command(
                    CommandType.RESOLVE_HANDOFF,
                    {
                        "hypothesis_id": hypothesis["id"],
                        "outgoing_tracklet_id": selected_candidate,
                        "actor": "streamlit",
                    },
                )
                st.success("Confirmation envoyée.")
            if columns[1].button(
                "Aucun candidat ne convient",
                key=f"reject-{hypothesis['id']}",
            ):
                context.repo.enqueue_command(
                    CommandType.REJECT_HANDOFF,
                    {
                        "hypothesis_id": hypothesis["id"],
                        "reason": "rejet depuis Streamlit",
                    },
                )
                st.info("Rejet envoyé.")
            with st.expander("Détails des candidats"):
                st.dataframe(pd.DataFrame(candidates), use_container_width=True)

    parcels = context.repo.list_parcels()
    st.subheader("Historique des colis")
    filter_columns = st.columns(4)
    selected_date = filter_columns[0].date_input("Date", value=None)
    camera_options = sorted(
        {str(row.get("last_camera_id")) for row in parcels if row.get("last_camera_id")}
    )
    selected_camera = filter_columns[1].selectbox(
        "Caméra", ["Toutes"] + camera_options
    )
    result_options = [
        "Tous",
        "SORT_OK",
        "WRONG_DESTINATION",
        "DESTINATION_UNVERIFIED",
    ]
    selected_result = filter_columns[2].selectbox(
        "Résultat",
        result_options,
        format_func=lambda value: status_label(value) if value != "Tous" else value,
    )
    destination_options = sorted(
        {
            str(value)
            for row in parcels
            for value in (
                row.get("expected_destination"),
                row.get("observed_destination"),
            )
            if value
        }
    )
    selected_destination = filter_columns[3].selectbox(
        "Destination", ["Toutes"] + destination_options
    )
    filtered = [
        row
        for row in parcels
        if (selected_date is None or _date(row.get("last_seen_at")) == selected_date)
        and (
            selected_camera == "Toutes"
            or str(row.get("last_camera_id")) == selected_camera
        )
        and (
            selected_result == "Tous"
            or row.get("destination_result") == selected_result
        )
        and (
            selected_destination == "Toutes"
            or selected_destination
            in {
                row.get("expected_destination"),
                row.get("observed_destination"),
            }
        )
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Heure": _time(row.get("last_seen_at")),
                    "Colis": short_identifier(row.get("parcel_id"), prefix="#"),
                    "Résultat": status_label(row.get("destination_result")),
                    "Destination attendue": row.get("expected_destination") or "—",
                    "Destination observée": row.get("observed_destination") or "—",
                    "Dernière caméra": row.get("last_camera_id") or "—",
                }
                for row in filtered
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Chronologie d’un colis")
    if filtered:
        parcel_options = {
            short_identifier(row.get("parcel_id"), prefix="Colis #"): row
            for row in filtered
        }
        selected_parcel = st.selectbox("Colis", list(parcel_options))
        parcel = parcel_options[selected_parcel]
        parcel_id = parcel.get("parcel_id")
        tracklets = [
            row
            for row in context.repo.list_tracklets(limit=500)
            if row.get("parcel_id") == parcel_id
        ]
        events = [
            row
            for row in context.repo.recent_events(limit=1000)
            if row.get("parcel_id") == parcel_id
        ]
        timeline = [
            (
                float(row.get("started_at_global") or 0.0),
                f"Détecté sur {row.get('camera_role') or row.get('camera_id')}",
            )
            for row in tracklets
        ]
        timeline.extend(
            (
                float(row.get("timestamp_global") or 0.0),
                event_label(row.get("event_type")),
            )
            for row in events
        )
        timeline.append(
            (
                float(parcel.get("last_seen_at") or 0.0),
                status_label(parcel.get("destination_result")),
            )
        )
        for _timestamp, label in sorted(timeline):
            st.markdown(f"**{label}**")
            st.caption("↓")
    else:
        st.caption("Aucun colis ne correspond aux filtres.")

    recent_events = context.repo.recent_events(limit=100)
    with st.expander("Journal des alertes récentes"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Date": row.get("created_at"),
                        "Information": event_label(row.get("event_type")),
                        "Caméra": row.get("camera_id")
                        or row.get("source_id")
                        or "—",
                        "Sévérité": row.get("severity"),
                    }
                    for row in recent_events
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
