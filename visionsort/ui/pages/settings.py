from __future__ import annotations

from copy import deepcopy
import json

import pandas as pd
import streamlit as st

from visionsort.calibration.models import DEFAULT_WORLD_CONVENTION
from visionsort.core.enums import CommandType
from visionsort.core.site_config import SITE_TRACKING_KEYS, validate_site_config
from visionsort.ui.components.common import demo_warning, page_header
from visionsort.ui.state import UIContext


_ZONE_LABELS = {
    "entry": "Entrée caméra",
    "exit": "Sortie caméra",
    "pick": "Prise opérateur",
    "destination": "Destination",
}


def _initial_site_config(context: UIContext) -> dict:
    stored = context.repo.get_site_config()
    if stored:
        return deepcopy(stored)
    base = context.config_values
    return {
        "schema_version": 2,
        "world_coordinate_convention": deepcopy(
            base.get("world_coordinate_convention") or DEFAULT_WORLD_CONVENTION
        ),
        "calibration_quality_thresholds": deepcopy(
            base.get("calibration_quality_thresholds") or {}
        ),
        "tracking": deepcopy(base.get("tracking") or {}),
    }


def _tracking(config: dict) -> dict:
    nested = config.get("tracking")
    if isinstance(nested, dict):
        return nested
    migrated = {key: config.pop(key) for key in list(config) if key in SITE_TRACKING_KEYS}
    config["tracking"] = migrated
    return migrated


def _send(context: UIContext, config: dict) -> bool:
    try:
        validated = validate_site_config(config)
    except RuntimeError as exc:
        st.error(str(exc))
        return False
    context.repo.enqueue_command(CommandType.UPSERT_SITE_CONFIG, validated)
    st.success("Configuration envoyée. Elle s’appliquera aux nouvelles sessions.")
    return True


def render(context: UIContext) -> None:
    page_header(
        "Configuration du site",
        "Décrivez simplement le chemin des colis, les zones utiles et les destinations.",
    )
    demo_warning(context)
    config = _initial_site_config(context)
    tracking = _tracking(config)
    sources = context.repo.list_sources()
    roles = [str(row.get("role")) for row in sources if row.get("role")]
    roles = list(dict.fromkeys(roles or ["C1", "C2", "C3"]))

    st.subheader("1. Caméras")
    if sources:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Caméra": row.get("name"),
                        "Étape": row.get("role"),
                        "Type de flux": row.get("source_type"),
                        "Activée": bool(row.get("enabled", 1)),
                    }
                    for row in sources
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Ajoutez d’abord les caméras depuis la page Caméras.")

    st.subheader("2. Chemin des colis")
    edges = list((tracking.get("site_topology") or {}).get("edges") or [])
    if edges:
        path = "<br>↓<br>".join(
            f"{edge.get('from_role')} → {edge.get('to_role')}"
            for edge in edges
        )
        st.markdown(f'<div class="vs-flow">{path}</div>', unsafe_allow_html=True)
    else:
        st.caption("Aucun passage entre caméras configuré.")
    edge_rows = pd.DataFrame(
        [
            {
                "Depuis": edge.get("from_role"),
                "Vers": edge.get("to_role"),
                "Minimum (s)": float(edge.get("min_transit_s") or 0.0),
                "Maximum (s)": float(edge.get("max_transit_s") or 1.0),
            }
            for edge in edges
        ]
        or [
            {
                "Depuis": roles[0],
                "Vers": roles[min(1, len(roles) - 1)],
                "Minimum (s)": 0.5,
                "Maximum (s)": 3.0,
            }
        ]
    )
    edited_edges = st.data_editor(
        edge_rows,
        hide_index=True,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Depuis": st.column_config.SelectboxColumn(options=roles, required=True),
            "Vers": st.column_config.SelectboxColumn(options=roles, required=True),
            "Minimum (s)": st.column_config.NumberColumn(min_value=0.0, step=0.1),
            "Maximum (s)": st.column_config.NumberColumn(min_value=0.0, step=0.1),
        },
        key="site-topology-editor",
    )
    if st.button("Enregistrer le chemin", type="primary"):
        updated = deepcopy(config)
        updated_tracking = _tracking(updated)
        updated_tracking["site_topology"] = {
            "edges": [
                {
                    "from_role": str(row["Depuis"]),
                    "to_role": str(row["Vers"]),
                    "min_transit_s": float(row["Minimum (s)"]),
                    "max_transit_s": float(row["Maximum (s)"]),
                }
                for row in edited_edges.to_dict("records")
            ]
        }
        _send(context, updated)

    st.subheader("3. Zones")
    zones_by_role = tracking.get("zones") or {}
    zone_rows = [
        {
            "Caméra": role,
            "Zone": zone.get("zone_id"),
            "Usage": _ZONE_LABELS.get(str(zone.get("kind")), zone.get("kind")),
        }
        for role, zones in zones_by_role.items()
        for zone in zones
    ]
    st.dataframe(
        pd.DataFrame(zone_rows)
        if zone_rows
        else pd.DataFrame(columns=["Caméra", "Zone", "Usage"]),
        hide_index=True,
        use_container_width=True,
    )
    with st.expander("Ajouter une zone rectangulaire"):
        with st.form("add-zone"):
            inputs = st.columns(3)
            zone_role = inputs[0].selectbox("Caméra", roles)
            zone_id = inputs[1].text_input("Nom de la zone")
            zone_kind = inputs[2].selectbox(
                "Usage",
                list(_ZONE_LABELS),
                format_func=lambda value: _ZONE_LABELS[value],
            )
            st.caption("Position dans l’image : 0 correspond au bord gauche/haut, 1 au bord droit/bas.")
            coordinates = st.columns(4)
            x1 = coordinates[0].number_input("Gauche", 0.0, 1.0, 0.0, 0.05)
            y1 = coordinates[1].number_input("Haut", 0.0, 1.0, 0.0, 0.05)
            x2 = coordinates[2].number_input("Droite", 0.0, 1.0, 1.0, 0.05)
            y2 = coordinates[3].number_input("Bas", 0.0, 1.0, 1.0, 0.05)
            if st.form_submit_button("Ajouter et enregistrer", disabled=not zone_id.strip()):
                updated = deepcopy(config)
                updated_zones = _tracking(updated).setdefault("zones", {})
                updated_zones.setdefault(zone_role, []).append(
                    {
                        "zone_id": zone_id.strip(),
                        "kind": zone_kind,
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )
                _send(context, updated)

    st.subheader("4. Destinations")
    destinations = [row for row in zone_rows if row["Usage"] == "Destination"]
    if destinations:
        st.dataframe(
            pd.DataFrame(destinations),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("Aucune destination définie. Ajoutez une zone de type Destination.")

    with st.expander("5. Paramètres avancés"):
        st.caption(
            "Cette vue conserve l’édition complète pour l’intégration et le diagnostic. "
            "Les sessions déjà démarrées gardent leur configuration immuable."
        )
        payload = st.text_area(
            "Configuration technique JSON",
            value=json.dumps(config, indent=2, ensure_ascii=False),
            height=360,
        )
        if st.button("Valider et enregistrer le JSON"):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError as exc:
                st.error(str(exc))
            else:
                _send(context, parsed)
