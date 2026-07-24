from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from visionsort.core.enums import CommandType
from visionsort.ui.components.common import page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Events", "Journal des événements métier et techniques")
    hypotheses = context.repo.list_handoff_hypotheses("PENDING")
    st.subheader(f"Handoffs ambigus ({len(hypotheses)} en attente)")
    for hypothesis in hypotheses:
        candidates = json.loads(hypothesis.get("candidates_json") or "[]")
        with st.container(border=True):
            st.write(
                f"`{hypothesis['incoming_tracklet_id']}` — "
                f"session `{hypothesis['session_id']}`"
            )
            st.dataframe(pd.DataFrame(candidates), use_container_width=True)
            candidate_ids = [
                str(candidate["from_tracklet_id"]) for candidate in candidates
            ]
            selected_candidate = st.selectbox(
                "Identité sortante",
                candidate_ids,
                key=f"candidate-{hypothesis['id']}",
            )
            columns = st.columns(2)
            if columns[0].button(
                "Résoudre avec ce candidat",
                key=f"resolve-{hypothesis['id']}",
                disabled=not selected_candidate,
            ):
                context.repo.enqueue_command(
                    CommandType.RESOLVE_HANDOFF,
                    {
                        "hypothesis_id": hypothesis["id"],
                        "outgoing_tracklet_id": selected_candidate,
                        "actor": "streamlit",
                    },
                )
                st.success("Résolution envoyée au supervisor.")
            if columns[1].button(
                "Rejeter l'hypothèse", key=f"reject-{hypothesis['id']}"
            ):
                context.repo.enqueue_command(
                    CommandType.REJECT_HANDOFF,
                    {
                        "hypothesis_id": hypothesis["id"],
                        "reason": "rejet depuis Streamlit",
                    },
                )
                st.info("Rejet envoyé au supervisor.")
    audit_rows = context.repo.list_handoff_resolution_audit(limit=200)
    with st.expander(
        f"Historique des résolutions ({len(audit_rows)})"
    ):
        if audit_rows:
            st.dataframe(
                pd.DataFrame(audit_rows)[
                    [
                        "created_at",
                        "hypothesis_id",
                        "actor",
                        "result",
                        "reason",
                    ]
                ],
                use_container_width=True,
            )
        else:
            st.caption("Aucune résolution auditée.")
    st.divider()
    events = context.repo.recent_events(limit=500)
    if not events:
        st.info("Aucun événement.")
        return
    df = pd.DataFrame(events)
    event_types = sorted(df["event_type"].dropna().unique().tolist()) if "event_type" in df else []
    selected_type = st.selectbox("Filtrer par type", ["Tous"] + event_types, index=0)
    if selected_type != "Tous":
        df = df[df["event_type"] == selected_type]
    df["payload_preview"] = df["payload_json"].apply(lambda value: json.dumps(json.loads(value), ensure_ascii=False)[:250])
    st.dataframe(df.drop(columns=["payload_json"]), use_container_width=True)
    if not df.empty:
        options = {f"{row['event_type']} | {row['created_at']}": row for _, row in df.iterrows()}
        selected = st.selectbox("Voir un payload complet", list(options), index=0)
        if selected:
            st.code(json.dumps(json.loads(options[selected]["payload_json"]), ensure_ascii=False, indent=2), language="json")
