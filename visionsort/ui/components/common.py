from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from visionsort.core.paths import ROOT_DIR
from visionsort.ui.state import UIContext


_STATUS_LABELS = {
    "RUNNING": "En fonctionnement",
    "STOPPED": "Arrêté",
    "CREATED": "Prêt à démarrer",
    "LIVE": "En ligne",
    "REPLAY": "Replay actif",
    "CONNECTING": "Connexion",
    "RECONNECTING": "Reconnexion",
    "DEGRADED": "À surveiller",
    "ERROR": "Erreur",
    "OFFLINE": "Hors ligne",
    "SORT_OK": "Tri correct",
    "WRONG_DESTINATION": "Mauvaise destination",
    "DESTINATION_UNVERIFIED": "Non vérifié",
    "MATCHED": "Suivi confirmé",
    "AMBIGUOUS": "Identification incertaine",
    "UNRESOLVED": "Suivi non confirmé",
    "NEW_AT_INGRESS": "Nouveau colis",
    "STABLE": "Stable",
    "READY": "Prêt",
    "ACTIVE": "Actif",
    "COLLECTING": "Collecte en cours",
    "PROMOTED": "Modèle adapté",
    "FROZEN": "Adaptation terminée",
    "ON_CONVEYOR": "Sur le convoyeur",
    "PICK_CANDIDATE": "Prise possible",
    "PICKED": "Pris par un opérateur",
    "CARRIED": "Transporté",
    "DROP_CANDIDATE": "Dépôt possible",
    "DROPPED": "Déposé",
    "BASELINE": "Modèle de référence",
    "CANDIDATE": "Prêt à évaluer",
    "CHAMPION": "Validé",
    "REJECTED": "Rejeté",
    "ARCHIVED": "Archivé",
}

_EVENT_LABELS = {
    "HANDOFF_MATCHED": "Colis retrouvé sur une autre caméra",
    "HANDOFF_AMBIGUOUS": "Identification du colis à confirmer",
    "HANDOFF_UNRESOLVED": "Passage entre caméras non confirmé",
    "GLOBAL_PARCEL_CREATED": "Nouveau colis détecté",
    "PICKED": "Colis pris par un opérateur",
    "CARRIED": "Colis transporté par un opérateur",
    "DROPPED": "Colis déposé",
    "SORT_OK": "Colis déposé à la bonne destination",
    "WRONG_DESTINATION": "Mauvaise destination détectée",
}


def apply_app_style() -> None:
    """Small, stable visual layer shared by every Streamlit page."""
    st.markdown(
        """
        <style>
        :root { --vs-blue:#315f79; --vs-ink:#17232b; --vs-muted:#667680; }
        .stApp { background: #f6f8f9; color: var(--vs-ink); }
        [data-testid="stSidebar"] { background: #eef2f4; border-right: 1px solid #dce3e7; }
        [data-testid="stSidebar"] h1 { letter-spacing: -0.03em; }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
        h1, h2, h3 { color: var(--vs-ink); letter-spacing: -0.025em; }
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.35rem !important; margin-top: 1.6rem !important; }
        div[data-testid="stMetric"] { background: white; border: 1px solid #e1e7ea;
          border-radius: 12px; padding: .9rem 1rem; box-shadow: 0 2px 8px rgba(23,35,43,.035); }
        div[data-testid="stVerticalBlockBorderWrapper"] { background: white;
          border-color: #e1e7ea !important; border-radius: 12px; }
        .vs-eyebrow { color: var(--vs-blue); font-size: .76rem; font-weight: 700;
          letter-spacing: .09em; text-transform: uppercase; margin-bottom: .25rem; }
        .vs-subtitle { color: var(--vs-muted); max-width: 780px; margin-top: -.55rem;
          margin-bottom: 1.25rem; }
        .vs-badge { display:inline-flex; align-items:center; gap:.35rem; border-radius:999px;
          padding:.25rem .58rem; font-size:.78rem; font-weight:650; border:1px solid transparent; }
        .vs-good { color:#19633b; background:#eaf7ef; border-color:#cde9d7; }
        .vs-warn { color:#805a13; background:#fff7e2; border-color:#f1dfad; }
        .vs-bad { color:#9a3030; background:#fcecec; border-color:#f0cccc; }
        .vs-neutral { color:#4b5d68; background:#eef2f4; border-color:#dce3e7; }
        .vs-flow { background:white; border:1px solid #e1e7ea; border-radius:12px;
          padding:1rem 1.2rem; text-align:center; font-weight:650; line-height:1.9; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, *, eyebrow: str = "VisionSort") -> None:
    st.markdown(f'<div class="vs-eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="vs-subtitle">{escape(subtitle)}</div>', unsafe_allow_html=True)


def demo_warning(context: UIContext) -> None:
    if context.config_demo_mode:
        st.warning(
            "Mode démonstration actif : les résultats issus des replays ne sont pas "
            "considérés comme validés sur le site industriel."
        )


def show_preview(preview_path: str | None, label: str) -> None:
    if preview_path:
        path = Path(preview_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        if path.exists():
            st.image(str(path), caption=label, use_container_width=True)
            return
    st.caption(f"Aucun aperçu disponible pour {label}.")


def status_label(value: Any) -> str:
    normalized = str(value or "OFFLINE").upper()
    return _STATUS_LABELS.get(normalized, normalized.replace("_", " ").title())


def status_tone(value: Any) -> str:
    normalized = str(value or "").upper()
    if normalized in {
        "LIVE",
        "REPLAY",
        "RUNNING",
        "SORT_OK",
        "MATCHED",
        "STABLE",
        "READY",
        "ACTIVE",
        "PROMOTED",
        "FROZEN",
        "CHAMPION",
    }:
        return "good"
    if normalized in {
        "DEGRADED",
        "CONNECTING",
        "RECONNECTING",
        "AMBIGUOUS",
        "DESTINATION_UNVERIFIED",
        "COLLECTING",
        "CANDIDATE",
    }:
        return "warn"
    if normalized in {
        "ERROR",
        "OFFLINE",
        "WRONG_DESTINATION",
        "UNRESOLVED",
        "REJECTED",
    }:
        return "bad"
    return "neutral"


def status_badge(value: Any) -> str:
    tone = status_tone(value)
    dot = {"good": "●", "warn": "●", "bad": "●", "neutral": "○"}[tone]
    return (
        f'<span class="vs-badge vs-{tone}">{dot} '
        f'{escape(status_label(value))}</span>'
    )


def render_status(value: Any) -> None:
    st.markdown(status_badge(value), unsafe_allow_html=True)


def event_label(event_type: Any) -> str:
    normalized = str(event_type or "").upper()
    return _EVENT_LABELS.get(
        normalized,
        normalized.replace("_", " ").capitalize() or "Événement",
    )


def short_identifier(value: Any, *, prefix: str = "") -> str:
    text = str(value or "—")
    compact = text.split("-")[-1] if "-" in text else text
    return f"{prefix}{compact[:12]}"


def request_navigation(page: str) -> None:
    st.session_state["requested_page"] = page
    st.rerun()


def online_source(status: Any) -> bool:
    return str(status or "").upper() in {
        "LIVE",
        "REPLAY",
        "DEGRADED",
        "CONNECTING",
        "RECONNECTING",
    }
