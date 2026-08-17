from __future__ import annotations

import sys
from pathlib import Path

# Ajoute le répertoire racine du projet au sys.path pour l'exécution en standalone
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from visionsort.ui.components.common import apply_app_style, online_source
from visionsort.ui.pages import (
    calibration,
    cameras,
    dashboard,
    dataset_studio,
    diagnostic,
    events,
    live_tracking,
    models,
    recordings,
    settings,
    training,
)
from visionsort.ui.state import create_ui_context


st.set_page_config(page_title="VisionSort", page_icon="VS", layout="wide")


PAGES = {
    "Accueil": dashboard.render,
    "Caméras": cameras.render,
    "Configuration du site": settings.render,
    "Calibration": calibration.render,
    "Exploitation": live_tracking.render,
    "Historique & alertes": events.render,
    "Modèles": models.render,
    "Diagnostic": diagnostic.render,
    "Enregistrements": recordings.render,
    "Données & annotations": dataset_studio.render,
    "Entraînement": training.render,
}


def main() -> None:
    apply_app_style()
    context = create_ui_context()
    sources = context.repo.list_sources()
    online_count = sum(online_source(row.get("status")) for row in sources)
    st.sidebar.title("VisionSort")
    st.sidebar.caption("Supervision du tri par vision")
    requested = st.session_state.pop("requested_page", None)
    if requested in PAGES:
        st.session_state["navigation_page"] = requested
    selected = st.sidebar.radio(
        "Navigation",
        list(PAGES),
        key="navigation_page",
    )
    st.sidebar.divider()
    st.sidebar.caption(
        f"{online_count}/{len(sources)} caméra(s) active(s)"
        if sources
        else "Aucune caméra configurée"
    )
    if context.config_demo_mode:
        st.sidebar.caption("Mode démonstration")
    if st.sidebar.button("Actualiser", use_container_width=True):
        st.rerun()
    PAGES[selected](context)


if __name__ == "__main__":
    main()
