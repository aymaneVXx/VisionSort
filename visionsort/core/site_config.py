from __future__ import annotations

from copy import deepcopy
import math
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
SITE_CALIBRATION_KEYS = {
    "schema_version",
    "calibration_profiles",
    "calibration_quality_thresholds",
    "world_coordinate_convention",
}
CALIBRATION_THRESHOLD_KEYS = {
    "min_views",
    "min_corners_per_view",
    "min_total_corners",
    "min_intrinsic_coverage",
    "warning_intrinsic_rms_px",
    "max_intrinsic_rms_px",
    "outlier_sigma",
    "max_view_error_px",
    "min_homography_points",
    "min_homography_image_coverage",
    "min_homography_world_coverage_m2",
    "min_collinearity_ratio",
    "min_inlier_ratio",
    "ransac_world_threshold_m",
    "warning_world_rmse_m",
    "max_world_rmse_m",
    "warning_reprojection_rmse_px",
    "max_reprojection_rmse_px",
    "max_world_error_m",
    "max_homography_condition_number",
    "ransac_confidence",
    "ransac_max_iterations",
}


def _validate_polygon_world(
    polygon: Any, *, zone_id: str
) -> list[list[float]]:
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise RuntimeError(
            f"Zone {zone_id} invalide: polygon_world requiert au moins 3 points."
        )
    points: list[list[float]] = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise RuntimeError(
                f"Zone {zone_id} invalide: chaque point world doit etre [x_m, y_m]."
            )
        try:
            x_value, y_value = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Zone {zone_id} invalide: coordonnees world numeriques requises."
            ) from exc
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise RuntimeError(
                f"Zone {zone_id} invalide: coordonnees world non finies."
            )
        points.append([x_value, y_value])
    if len({tuple(point) for point in points}) < 3:
        raise RuntimeError(
            f"Zone {zone_id} invalide: polygon_world degenere."
        )
    twice_area = sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    )
    if abs(twice_area) <= 1.0e-9:
        raise RuntimeError(
            f"Zone {zone_id} invalide: polygon_world d'aire nulle."
        )
    def orientation(a: list[float], b: list[float], c: list[float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (
            c[0] - a[0]
        )

    for first in range(len(points)):
        first_next = (first + 1) % len(points)
        for second in range(first + 1, len(points)):
            second_next = (second + 1) % len(points)
            if (
                first == second
                or first_next == second
                or second_next == first
            ):
                continue
            o1 = orientation(points[first], points[first_next], points[second])
            o2 = orientation(points[first], points[first_next], points[second_next])
            o3 = orientation(points[second], points[second_next], points[first])
            o4 = orientation(points[second], points[second_next], points[first_next])
            if o1 * o2 < -1.0e-12 and o3 * o4 < -1.0e-12:
                raise RuntimeError(
                    f"Zone {zone_id} invalide: polygon_world auto-intersecte."
                )
    return points


def _validate_world_convention(value: Any) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(
            "Configuration site invalide: world_coordinate_convention doit etre un objet."
        )
    if str(value.get("unit") or "") != "m":
        raise RuntimeError("Le repere monde VisionSort utilise obligatoirement les metres.")
    if str(value.get("x_axis") or "") != "conveyor_longitudinal":
        raise RuntimeError("L'axe X monde doit etre conveyor_longitudinal.")
    if str(value.get("y_axis") or "") != "conveyor_transverse":
        raise RuntimeError("L'axe Y monde doit etre conveyor_transverse.")
    try:
        plane_z = float(value.get("conveyor_plane_z_m", 0.0))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("conveyor_plane_z_m doit etre numerique.") from exc
    if not math.isfinite(plane_z) or abs(plane_z) > 1.0e-12:
        raise RuntimeError("Le plan convoyeur de reference doit etre Z = 0 m.")


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
    try:
        schema_version = int(config.get("schema_version") or 1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Configuration site invalide: schema_version entier requis.") from exc
    if schema_version not in {1, 2}:
        raise RuntimeError(
            f"Configuration site invalide: schema_version {schema_version} non supporte."
        )
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
            legacy_names = ("x1", "y1", "x2", "y2")
            has_any_legacy = any(name in zone for name in legacy_names)
            has_all_legacy = all(name in zone for name in legacy_names)
            has_world = "polygon_world" in zone
            if has_any_legacy and not has_all_legacy:
                raise RuntimeError(
                    f"Zone {zone_id} invalide: les 4 coordonnees legacy sont requises."
                )
            if has_all_legacy:
                try:
                    x1, y1, x2, y2 = (
                        float(zone[name]) for name in legacy_names
                    )
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Zone {zone_id} invalide: coordonnees numeriques requises."
                    ) from exc
                if not (
                    0.0 <= x1 < x2 <= 1.0
                    and 0.0 <= y1 < y2 <= 1.0
                ):
                    raise RuntimeError(
                        f"Zone {zone_id} invalide: coordonnees normalisees attendues dans [0, 1]."
                    )
            if has_world:
                _validate_polygon_world(zone["polygon_world"], zone_id=zone_id)
            if not has_all_legacy and not has_world:
                raise RuntimeError(
                    f"Zone {zone_id} invalide: coordonnees legacy ou polygon_world requis."
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
    calibration_profiles = config.get("calibration_profiles")
    if calibration_profiles is not None:
        if not isinstance(calibration_profiles, dict):
            raise RuntimeError(
                "Configuration site invalide: calibration_profiles doit etre un objet."
            )
        active = calibration_profiles.get("active_by_source", {})
        if not isinstance(active, dict) or any(
            not str(source_id).strip() or not str(profile_id).strip()
            for source_id, profile_id in active.items()
        ):
            raise RuntimeError(
                "Configuration site invalide: active_by_source doit associer source et profil."
            )
    thresholds = config.get("calibration_quality_thresholds")
    if thresholds is not None:
        if not isinstance(thresholds, dict):
            raise RuntimeError(
                "Configuration site invalide: calibration_quality_thresholds doit etre un objet."
            )
        unknown_thresholds = set(thresholds) - CALIBRATION_THRESHOLD_KEYS
        if unknown_thresholds:
            raise RuntimeError(
                "Seuils de calibration inconnus: "
                + ", ".join(sorted(str(value) for value in unknown_thresholds))
            )
        for key, value in thresholds.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Seuil de calibration invalide: {key} doit etre numerique."
                ) from exc
            if not math.isfinite(number) or number <= 0:
                raise RuntimeError(
                    f"Seuil de calibration invalide: {key} doit etre strictement positif."
                )
            if (
                str(key)
                in {
                    "min_intrinsic_coverage",
                    "min_homography_image_coverage",
                    "min_collinearity_ratio",
                    "min_inlier_ratio",
                    "ransac_confidence",
                }
                and number > 1
            ):
                raise RuntimeError(
                    f"Seuil de calibration invalide: {key} doit etre dans ]0, 1]."
                )
        for warning_key, maximum_key in (
            ("warning_intrinsic_rms_px", "max_intrinsic_rms_px"),
            ("warning_world_rmse_m", "max_world_rmse_m"),
            (
                "warning_reprojection_rmse_px",
                "max_reprojection_rmse_px",
            ),
        ):
            if (
                warning_key in thresholds
                and maximum_key in thresholds
                and float(thresholds[warning_key])
                > float(thresholds[maximum_key])
            ):
                raise RuntimeError(
                    f"Seuils de calibration incoherents: {warning_key} > {maximum_key}."
                )
        if int(float(thresholds.get("min_homography_points", 4))) < 4:
            raise RuntimeError("min_homography_points doit etre au moins 4.")
    convention = config.get("world_coordinate_convention")
    if convention is not None:
        _validate_world_convention(convention)
    if schema_version >= 2 and convention is None:
        # A v2 file may be introduced incrementally, but never with an ambiguous unit.
        calibration_sections = any(
            key in config
            for key in (
                "calibration_profiles",
                "calibration_quality_thresholds",
            )
        )
        if calibration_sections:
            raise RuntimeError(
                "site_config v2 avec calibration: world_coordinate_convention est requis."
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
    for key in SITE_CALIBRATION_KEYS:
        if key in validated:
            effective[key] = deepcopy(validated[key])
    return effective


def zone_kind(zones: list[dict[str, Any]], zone_id: str | None) -> str | None:
    if not zone_id:
        return None
    for zone in zones:
        if str(zone.get("zone_id")) == str(zone_id):
            kind = str(zone.get("kind") or "").lower()
            return kind if kind in ZONE_KINDS else None
    return None
