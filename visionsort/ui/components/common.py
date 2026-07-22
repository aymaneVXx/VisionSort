from __future__ import annotations

from pathlib import Path

import streamlit as st

from visionsort.ui.state import UIContext


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def demo_warning(context: UIContext) -> None:
    if context.config_demo_mode:
        st.warning("DEMO_MODE actif: les handoffs, prises et dépôts sont testables en Replay mais restent non validés sur site.")
    else:
        st.info("DEMO_MODE inactif: aucun résultat simulé n'est autorisé. Configurez des flux réels ou activez explicitement DEMO_MODE.")


def show_preview(preview_path: str | None, label: str) -> None:
    if preview_path and Path(preview_path).exists():
        st.image(preview_path, caption=label, use_container_width=True)
    else:
        st.caption(f"Aucune preview disponible pour {label}.")
