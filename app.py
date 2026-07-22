from __future__ import annotations

import sys
from pathlib import Path

# Ajoute le répertoire racine du projet au sys.path pour l'exécution en standalone
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from visionsort.ui.pages import cameras, dashboard, dataset_studio, events, live_tracking, models, recordings, settings, training
from visionsort.ui.state import create_ui_context


st.set_page_config(page_title="VisionSort", page_icon="VS", layout="wide")


def main() -> None:
    context = create_ui_context()
    st.sidebar.title("VisionSort")
    st.sidebar.caption("Pilotage par SQLite, traitements persistants via RuntimeSupervisor")
    pages = {
        "Dashboard": dashboard.render,
        "Cameras": cameras.render,
        "Live Tracking": live_tracking.render,
        "Recordings": recordings.render,
        "Dataset Studio": dataset_studio.render,
        "Training": training.render,
        "Models": models.render,
        "Events": events.render,
        "Settings": settings.render,
    }
    if st.sidebar.button("Rafraîchir"):
        st.rerun()
    selected = st.sidebar.radio("Navigation", list(pages))
    pages[selected](context)
    st.sidebar.markdown("---")
    st.sidebar.write(f"DEMO_MODE: `{context.config_demo_mode}`")


if __name__ == "__main__":
    main()
