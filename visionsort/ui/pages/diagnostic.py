from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from visionsort.calibration.repository import CalibrationRepository
from visionsort.ui.components.common import page_header
from visionsort.ui.state import UIContext


def _json_value(value: str | None) -> object:
    try:
        return json.loads(value or "{}")
    except (json.JSONDecodeError, TypeError):
        return value or {}


def render(context: UIContext) -> None:
    page_header(
        "Diagnostic",
        "Détails techniques pour l’analyse, la maintenance et le support.",
        eyebrow="Outils avancés",
    )
    st.info(
        "Cet espace expose volontairement les identifiants, scores et états "
        "internes. Les écrans d’exploitation restent centrés sur le métier."
    )
    system_tab, tracking_tab, events_tab, models_tab = st.tabs(
        ["Système", "Suivi", "Événements", "Modèles & calibration"]
    )

    with system_tab:
        sources = context.repo.list_sources()
        jobs = context.repo.list_jobs()
        st.subheader("Workers et caméras")
        if sources:
            st.dataframe(pd.DataFrame(sources), use_container_width=True)
        else:
            st.caption("Aucune caméra enregistrée.")
        st.subheader("Processus runtime")
        st.dataframe(
            pd.DataFrame(jobs) if jobs else pd.DataFrame(columns=["id", "status"]),
            use_container_width=True,
        )
        with st.expander("État complet du routage des modèles"):
            st.json(context.repo.get_runtime_model_state())

    with tracking_tab:
        tracklets = context.repo.list_tracklets(limit=500)
        parcels = context.repo.list_parcels()
        hypotheses = context.repo.list_handoff_hypotheses()
        st.subheader("Suivis locaux")
        st.dataframe(
            pd.DataFrame(tracklets)
            if tracklets
            else pd.DataFrame(columns=["tracklet_id", "match_result"]),
            use_container_width=True,
        )
        st.subheader("Colis globaux")
        st.dataframe(
            pd.DataFrame(parcels)
            if parcels
            else pd.DataFrame(columns=["parcel_id", "state"]),
            use_container_width=True,
        )
        with st.expander("Hypothèses de passage entre caméras"):
            st.dataframe(
                pd.DataFrame(hypotheses)
                if hypotheses
                else pd.DataFrame(columns=["id", "status"]),
                use_container_width=True,
            )

    with events_tab:
        events = context.repo.recent_events(limit=1000)
        if not events:
            st.caption("Aucun événement technique disponible.")
        else:
            st.dataframe(pd.DataFrame(events), use_container_width=True)
            options = {
                f"{row.get('created_at')} · {row.get('event_type')} · {row.get('id')}": row
                for row in events
            }
            selected = st.selectbox("Inspecter un payload", list(options))
            if selected:
                st.json(_json_value(options[selected].get("payload_json")))

    with models_tab:
        models = context.repo.list_models()
        st.subheader("Registre des modèles")
        st.dataframe(
            pd.DataFrame(models)
            if models
            else pd.DataFrame(columns=["id", "task", "status"]),
            use_container_width=True,
        )
        repository = CalibrationRepository(context.db)
        profiles = repository.list_profiles()
        st.subheader("Profils de calibration")
        if profiles:
            rows = [
                {
                    "profile_id": profile.profile_id,
                    "source_id": profile.source_id,
                    "version": profile.version,
                    "status": profile.status.value,
                    "world_frame_id": profile.world_coordinate_convention.get(
                        "frame_id"
                    ),
                    "fingerprint_sha256": profile.fingerprint_sha256,
                }
                for profile in profiles
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            selected_profile = st.selectbox(
                "Inspecter un profil",
                [profile.profile_id for profile in profiles],
            )
            profile = next(
                item for item in profiles if item.profile_id == selected_profile
            )
            st.json(profile.to_dict())
        else:
            st.caption("Aucun profil de calibration enregistré.")
