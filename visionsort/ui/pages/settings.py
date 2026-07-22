from __future__ import annotations

import json

import streamlit as st

from visionsort.core.enums import CommandType
from visionsort.ui.components.common import demo_warning, page_header
from visionsort.ui.state import UIContext


def render(context: UIContext) -> None:
    page_header("Settings", "Topologie, zones et paramètres globaux")
    demo_warning(context)
    site_config = context.repo.get_site_config()
    current_text = json.dumps(site_config, indent=2, ensure_ascii=False)
    payload = st.text_area("Configuration site JSON", value=current_text or "{}", height=280)
    if st.button("Enregistrer la configuration"):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            st.error(f"JSON invalide: {exc}")
            return
        context.repo.enqueue_command(CommandType.UPSERT_SITE_CONFIG, parsed)
        st.success("Commande d'enregistrement de configuration envoyée.")
