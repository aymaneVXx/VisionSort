from __future__ import annotations

from copy import deepcopy
from typing import Any


ZONE_KINDS = {"entry", "exit", "pick", "destination"}
SITE_TRACKING_KEYS = {
    "zones",
    "site_topology",
    "handoff_window_seconds",
    "handoff_buffer_max_items",
    "handoff_expiry_seconds",
    "hypothesis_expiry_seconds",
    "event_confirmation_seconds",
}


def _tracking_payload(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("tracking")
    if nested is not None:
        if not isinstance(nested, dict):
            raise RuntimeError("Configuration site invalide: tracking doit etre un objet.")
        return dict(nested)
    return {key: config[key] for key in SITE_TRACKING_KEYS if key in config}


def validate_site_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the SQLite site overlay without mutating the caller payload."""
    if not isinstance(config, dict):
        raise RuntimeError("Configuration site invalide: un objet JSON est requis.")
    tracking = _tracking_payload(config)
    zones_by_role = tracking.get("zones", {})
    if not isinstance(zones_by_role, dict):
        raise RuntimeError("Configuration site invalide: tracking.zones doit etre un objet.")
    for role, zones in zones_by_role.items():
        if not str(role).strip() or not isinstance(zones, list):
            raise RuntimeError("Configuration site invalide: chaque role doit contenir une liste de zones.")
        seen: set[str] = set()
        for zone in zones:
            if not isinstance(zone, dict):
                raise RuntimeError(f"Zone invalide pour {role}: un objet est requis.")
            zone_id = str(zone.get("zone_id") or "").strip()
            kind = str(zone.get("kind") or "").strip().lower()
            if not zone_id or zone_id in seen:
                raise RuntimeError(f"Zone invalide pour {role}: zone_id absent ou duplique.")
            if kind not in ZONE_KINDS:
                raise RuntimeError(
                    f"Zone {zone_id} invalide: kind doit etre entry, exit, pick ou destination."
                )
            seen.add(zone_id)
            try:
                x1, y1, x2, y2 = (float(zone[name]) for name in ("x1", "y1", "x2", "y2"))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Zone {zone_id} invalide: coordonnees numeriques requises.") from exc
            if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                raise RuntimeError(
                    f"Zone {zone_id} invalide: coordonnees normalisees attendues dans [0, 1]."
                )

    topology = tracking.get("site_topology", {})
    if topology and not isinstance(topology, dict):
        raise RuntimeError("Configuration site invalide: site_topology doit etre un objet.")
    edges = topology.get("edges", []) if isinstance(topology, dict) else []
    if not isinstance(edges, list):
        raise RuntimeError("Configuration site invalide: site_topology.edges doit etre une liste.")
    for edge in edges:
        if not isinstance(edge, dict):
            raise RuntimeError("Configuration site invalide: chaque transition doit etre un objet.")
        source = str(edge.get("from_role") or "").strip()
        target = str(edge.get("to_role") or "").strip()
        try:
            minimum = float(edge["min_transit_s"])
            maximum = float(edge["max_transit_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Transition invalide: fenetre de transit numerique requise.") from exc
        if not source or not target or source == target or minimum < 0 or maximum < minimum:
            raise RuntimeError(
                f"Transition invalide {source or '?'} -> {target or '?'}: roles et fenetre incoherents."
            )
    return deepcopy(config)


def apply_site_config(base: dict[str, Any], site_config: dict[str, Any]) -> dict[str, Any]:
    """Build the effective runtime config; SQLite only overrides site-owned settings."""
    validated = validate_site_config(site_config)
    effective = deepcopy(base)
    tracking = effective.setdefault("tracking", {})
    for key, value in _tracking_payload(validated).items():
        if key in SITE_TRACKING_KEYS:
            tracking[key] = deepcopy(value)
    site_parameters = validated.get("site_parameters", {})
    if site_parameters:
        if not isinstance(site_parameters, dict):
            raise RuntimeError("Configuration site invalide: site_parameters doit etre un objet.")
        effective["site_parameters"] = deepcopy(site_parameters)
    return effective


def zone_kind(zones: list[dict[str, Any]], zone_id: str | None) -> str | None:
    if not zone_id:
        return None
    for zone in zones:
        if str(zone.get("zone_id")) == str(zone_id):
            kind = str(zone.get("kind") or "").lower()
            return kind if kind in ZONE_KINDS else None
    return None
