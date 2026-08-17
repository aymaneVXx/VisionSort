from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from visionsort.ui.components.common import (
    demo_warning,
    page_header,
    short_identifier,
    show_preview,
    status_badge,
    status_label,
)
from visionsort.ui.state import UIContext


def _time(value: object) -> str:
    try:
        return datetime.fromtimestamp(float(value)).astimezone().strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "—"


def render(context: UIContext) -> None:
    page_header(
        "Exploitation",
        "Suivez les flux, les colis et le résultat du tri en temps réel.",
    )
    demo_warning(context)
    sessions = context.repo.list_capture_sessions()
    session_map = {f"{row['name']}": row["id"] for row in sessions}
    selected = st.selectbox(
        "Session affichée",
        list(session_map),
        index=0 if session_map else None,
    )
    selected_session_id = session_map[selected] if selected else None
    sources = context.repo.list_sources()
    if selected_session_id:
        session_sources = {
            row["source_id"]
            for row in context.repo.list_capture_session_sources(selected_session_id)
        }
        sources = [row for row in sources if row["id"] in session_sources]

    st.subheader("Flux vidéo")
    if not sources:
        st.info("Aucune caméra à afficher.")
    else:
        columns = st.columns(min(3, len(sources)))
        for index, source in enumerate(sources):
            with columns[index % len(columns)].container(border=True):
                st.markdown(f"**{source['name']} — {source['role']}**")
                st.markdown(
                    status_badge(source.get("status") or "OFFLINE"),
                    unsafe_allow_html=True,
                )
                show_preview(source.get("preview_path"), str(source["name"]))
                st.caption(f"{float(source.get('fps') or 0):.1f} images/s")

    tracklets = context.repo.list_tracklets(limit=500)
    if selected_session_id:
        tracklets = [
            row for row in tracklets if row.get("session_id") == selected_session_id
        ]
    parcel_ids = {row.get("parcel_id") for row in tracklets if row.get("parcel_id")}
    parcels = context.repo.list_parcels()
    if selected_session_id:
        parcels = [row for row in parcels if row.get("parcel_id") in parcel_ids]

    st.subheader("Colis suivis")
    if not parcels:
        st.info("Aucun colis suivi pour cette session.")
        return
    card_columns = st.columns(3)
    for index, parcel in enumerate(parcels[:9]):
        with card_columns[index % 3].container(border=True):
            st.markdown(
                f"### Colis {short_identifier(parcel.get('parcel_id'), prefix='#')}"
            )
            st.markdown(
                status_badge(parcel.get("destination_result")),
                unsafe_allow_html=True,
            )
            expected = parcel.get("expected_destination") or "Non renseignée"
            observed = parcel.get("observed_destination") or "Non observée"
            st.write(f"Destination attendue : **{expected}**")
            st.write(f"Destination observée : **{observed}**")
            if parcel.get("operator_id"):
                st.caption(f"Opérateur associé : {parcel['operator_id']}")

    table = pd.DataFrame(
        [
            {
                "Heure": _time(parcel.get("last_seen_at")),
                "Colis": short_identifier(parcel.get("parcel_id"), prefix="#"),
                "État": status_label(parcel.get("state")),
                "Destination attendue": parcel.get("expected_destination") or "—",
                "Destination observée": parcel.get("observed_destination") or "—",
                "Résultat": status_label(parcel.get("destination_result")),
                "Opérateur": parcel.get("operator_id") or "—",
            }
            for parcel in parcels
        ]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    with st.expander("Détails du suivi"):
        st.caption(
            "Informations utiles pour vérifier le passage d’un colis entre les caméras. "
            "Les scores complets sont disponibles dans Diagnostic."
        )
        public_columns = [
            "parcel_id",
            "camera_role",
            "started_at_global",
            "ended_at_global",
            "last_zone_id",
            "match_result",
        ]
        available = [name for name in public_columns if tracklets and name in tracklets[0]]
        st.dataframe(
            pd.DataFrame(tracklets)[available]
            if tracklets and available
            else pd.DataFrame(columns=public_columns),
            hide_index=True,
            use_container_width=True,
        )
