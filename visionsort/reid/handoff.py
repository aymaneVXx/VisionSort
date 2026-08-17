from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from visionsort.core.enums import MatchResult
from visionsort.core.site_config import zone_kind
from visionsort.core.types import HandoffCandidate, Tracklet
from visionsort.reid.encoder import (
    ProjectionHead,
    descriptor_is_reliable,
    descriptor_similarity,
)


@dataclass(slots=True)
class GateDecision:
    accepted: bool
    reasons: list[str]
    evidence: dict[str, Any]
    edge: dict[str, Any] | None = None


class HandoffCandidateGenerator:
    """Apply physical and integrity gates before any visual comparison."""

    def __init__(
        self,
        topology_edges: list[dict[str, Any]],
        source_roles: dict[str, str],
        zones_by_role: dict[str, list[dict[str, Any]]],
        *,
        max_speed_m_s: float = 3.0,
    ) -> None:
        self.topology_edges = topology_edges
        self.source_roles = source_roles
        self.zones_by_role = zones_by_role
        self.max_speed_m_s = max(0.1, float(max_speed_m_s))

    def evaluate(self, outgoing: Tracklet, incoming: Tracklet) -> GateDecision:
        previous_role = outgoing.camera_role or self.source_roles.get(
            outgoing.camera_id, outgoing.camera_id
        )
        incoming_role = incoming.camera_role or self.source_roles.get(
            incoming.camera_id, incoming.camera_id
        )
        evidence: dict[str, Any] = {
            "same_session": outgoing.session_id == incoming.session_id,
            "topology": f"{previous_role}->{incoming_role}",
        }
        if outgoing.session_id != incoming.session_id:
            return GateDecision(False, ["session incompatible"], evidence)
        edge = next(
            (
                item
                for item in self.topology_edges
                if item.get("from_role") == previous_role
                and item.get("to_role") == incoming_role
            ),
            None,
        )
        if edge is None:
            return GateDecision(False, ["edge topologique interdite"], evidence)
        evidence["edge_authorized"] = True
        transit = incoming.started_at_global - outgoing.ended_at_global
        minimum = float(edge["min_transit_s"])
        maximum = float(edge["max_transit_s"])
        evidence.update(
            {
                "transit_s": transit,
                "min_transit_s": minimum,
                "max_transit_s": maximum,
            }
        )
        if transit < minimum or transit > maximum:
            return GateDecision(False, ["transit hors fenêtre"], evidence, edge)
        outgoing_summary = _summary(outgoing)
        incoming_summary = _summary(incoming)
        identities = {
            str(outgoing.integrity_status),
            str(incoming.integrity_status),
            str(outgoing_summary.get("integrity_status") or ""),
            str(incoming_summary.get("integrity_status") or ""),
        }
        usable_identity = (
            outgoing.local_track_id >= 0
            and incoming.local_track_id >= 0
            and "AMBIGUOUS" not in identities
        )
        evidence["identity_usable"] = usable_identity
        if not usable_identity:
            return GateDecision(False, ["identité locale inexploitable"], evidence, edge)
        outgoing_zone = outgoing_summary.get("last_zone_id") or outgoing.last_zone_id
        incoming_zone = incoming_summary.get("first_zone_id")
        outgoing_kind = zone_kind(self.zones_by_role.get(previous_role, []), outgoing_zone)
        incoming_kind = zone_kind(self.zones_by_role.get(incoming_role, []), incoming_zone)
        evidence.update(
            {
                "from_zone": outgoing_zone,
                "to_zone": incoming_zone,
                "from_zone_kind": outgoing_kind,
                "to_zone_kind": incoming_kind,
            }
        )
        if outgoing_kind is not None and outgoing_kind != "exit":
            return GateDecision(False, ["zone sortante non exit"], evidence, edge)
        if incoming_kind is not None and incoming_kind != "entry":
            return GateDecision(False, ["zone entrante non entry"], evidence, edge)
        left_world = outgoing_summary.get("last_anchor_world_m")
        right_world = incoming_summary.get("first_anchor_world_m")
        left_frame = outgoing_summary.get("world_frame_id")
        right_frame = incoming_summary.get("world_frame_id")
        same_world_frame = bool(left_frame) and left_frame == right_frame
        evidence.update(
            {
                "outgoing_world_frame_id": left_frame,
                "incoming_world_frame_id": right_frame,
                "same_world_frame": same_world_frame,
            }
        )
        valid_world = False
        left_point: tuple[float, float] | None = None
        right_point: tuple[float, float] | None = None
        if (
            same_world_frame
            and left_world
            and right_world
            and len(left_world) >= 2
            and len(right_world) >= 2
        ):
            try:
                left_point = (float(left_world[0]), float(left_world[1]))
                right_point = (float(right_world[0]), float(right_world[1]))
                valid_world = all(math.isfinite(value) for value in (*left_point, *right_point))
            except (TypeError, ValueError):
                valid_world = False
        if valid_world and left_point is not None and right_point is not None:
            distance = math.dist(left_point, right_point)
            physical_limit = float(edge.get("max_speed_m_s", self.max_speed_m_s)) * transit + 0.35
            evidence.update(
                {
                    "world_available": True,
                    "world_distance_m": distance,
                    "world_distance_limit_m": physical_limit,
                }
            )
            if distance > physical_limit:
                return GateDecision(False, ["mouvement monde impossible"], evidence, edge)
        else:
            evidence["world_available"] = False
            evidence["world_unavailable_reason"] = (
                "different_or_unknown_world_frames"
                if not same_world_frame
                else "missing_or_invalid_world_coordinates"
            )
        return GateDecision(
            True,
            ["hard gates validés"],
            evidence,
            edge,
        )


class HandoffScorer:
    """Deterministic, explainable score over already feasible candidates."""

    def __init__(
        self,
        *,
        reid_enabled: bool = True,
        projection: ProjectionHead | None = None,
    ) -> None:
        self.reid_enabled = bool(reid_enabled)
        self.projection = projection or ProjectionHead()

    def score(
        self,
        outgoing: Tracklet,
        incoming: Tracklet,
        gates: GateDecision,
    ) -> HandoffCandidate:
        if not gates.accepted or gates.edge is None:
            raise ValueError("Un candidat rejeté par les hard gates ne peut pas être scoré.")
        edge = gates.edge
        transit = float(gates.evidence["transit_s"])
        minimum = float(edge["min_transit_s"])
        maximum = float(edge["max_transit_s"])
        midpoint = (minimum + maximum) / 2.0
        half_window = max((maximum - minimum) / 2.0, 1.0e-6)
        temporal = max(0.0, 1.0 - abs(transit - midpoint) / half_window)
        left_dims = _dimensions(outgoing)
        right_dims = _dimensions(incoming)
        dimensions = sum(
            min(left, right) / max(left, right, 1.0e-6)
            for left, right in zip(left_dims, right_dims)
        ) / 2.0
        left_velocity = _velocity(outgoing)
        right_velocity = _velocity(incoming)
        left_speed, right_speed = math.hypot(*left_velocity), math.hypot(*right_velocity)
        speed = min(left_speed, right_speed) / max(left_speed, right_speed, 1.0e-6)
        zone = 1.0 if (
            gates.evidence.get("from_zone_kind") == "exit"
            and gates.evidence.get("to_zone_kind") == "entry"
        ) else 0.60
        integrity = _integrity_score(outgoing, incoming)
        world: float | None = None
        if gates.evidence.get("world_available"):
            distance = float(gates.evidence["world_distance_m"])
            limit = max(float(gates.evidence["world_distance_limit_m"]), 1.0e-6)
            world = max(0.0, 1.0 - distance / limit)
        outgoing_summary = _summary(outgoing)
        incoming_summary = _summary(incoming)
        outgoing_reliable = descriptor_is_reliable(outgoing_summary)
        incoming_reliable = descriptor_is_reliable(incoming_summary)
        reid: float | None = None
        if self.reid_enabled and outgoing_reliable and incoming_reliable:
            reid = descriptor_similarity(
                outgoing_summary, incoming_summary, self.projection
            )
        outgoing_descriptor = outgoing_summary.get("appearance_descriptor") or {}
        incoming_descriptor = incoming_summary.get("appearance_descriptor") or {}
        backbone_versions = sorted(
            {
                str(value)
                for value in (
                    outgoing_descriptor.get("model_version"),
                    incoming_descriptor.get("model_version"),
                )
                if value
            }
        )
        components: dict[str, float] = {
            "temporal": temporal,
            "dimensions": dimensions,
            "zone": zone,
            "speed": speed,
            "integrity": integrity,
        }
        physical_weights = {
            "temporal": 0.24,
            "dimensions": 0.20,
            "zone": 0.16,
            "speed": 0.10,
            "integrity": 0.10,
        }
        if world is not None:
            components["world"] = world
            physical_weights["world"] = 0.08
        physical_score = sum(
            components[name] * physical_weights[name] for name in physical_weights
        ) / sum(physical_weights.values())
        weights = dict(physical_weights)
        if reid is not None:
            components["reid"] = reid
            weights["reid"] = 0.32
        denominator = sum(weights[name] for name in components)
        total = sum(components[name] * weights[name] for name in components) / denominator
        features: dict[str, float | str | bool | None] = {
            **components,
            "transit_s": transit,
            "topology": str(gates.evidence["topology"]),
            "physical_score": float(physical_score),
            "non_visual_score": float(physical_score),
            "reid_available": reid is not None,
            "reid_used": reid is not None,
            "outgoing_descriptor_reliable": outgoing_reliable,
            "incoming_descriptor_reliable": incoming_reliable,
            "reid_backbone_versions": ",".join(backbone_versions) or None,
        }
        reasons = [
            f"topology={gates.evidence['topology']}",
            f"transit={transit:.3f}s",
            *(f"{name}={value:.3f}" for name, value in components.items()),
        ]
        return HandoffCandidate(
            from_tracklet_id=outgoing.tracklet_id,
            to_tracklet_id=incoming.tracklet_id,
            score=float(total),
            result=MatchResult.UNRESOLVED,
            reasons=reasons,
            features=features,
            gate_evidence=dict(gates.evidence),
            model_version=(
                (
                    f"{'+'.join(backbone_versions)}|{self.projection.version}"
                    if backbone_versions
                    else self.projection.version
                )
                if self.reid_enabled
                else "reid-disabled"
            ),
        )


def _summary(tracklet: Tracklet) -> dict[str, Any]:
    return tracklet.summary_json if isinstance(tracklet.summary_json, dict) else {}


def _dimensions(tracklet: Tracklet) -> tuple[float, float]:
    dimensions = _summary(tracklet).get("avg_dimensions")
    if dimensions and len(dimensions) >= 2:
        return abs(float(dimensions[0])), abs(float(dimensions[1]))
    box = tracklet.last_bbox or tracklet.first_bbox
    return abs(float(box[2]) - float(box[0])), abs(float(box[3]) - float(box[1]))


def _velocity(tracklet: Tracklet) -> tuple[float, float]:
    value = _summary(tracklet).get("avg_velocity")
    if value and len(value) >= 2:
        return float(value[0]), float(value[1])
    return float(tracklet.avg_speed), 0.0


def _integrity_score(outgoing: Tracklet, incoming: Tracklet) -> float:
    scores = {
        "STABLE": 1.0,
        "RECOVERED": 0.82,
        "OCCLUDED": 0.65,
        "AMBIGUOUS": 0.0,
    }
    return min(
        scores.get(str(outgoing.integrity_status), 0.75),
        scores.get(str(incoming.integrity_status), 0.75),
    )
