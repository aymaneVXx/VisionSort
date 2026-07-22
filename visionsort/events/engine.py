from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from visionsort.core.enums import ParcelState
from visionsort.core.types import TrackObservation
from visionsort.tracking.engine import bbox_center, bbox_iou, box_area, zone_for_bbox


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


@dataclass
class ParcelEvidence:
    parcel_key: str
    state: ParcelState = ParcelState.ON_CONVEYOR
    pickup_score: float = 0.0
    carry_score: float = 0.0
    drop_score: float = 0.0
    destination_zone: str | None = None
    last_timestamp: float = 0.0
    last_bbox: tuple[float, float, float, float] | None = None
    validated_on_site: bool = False
    emitted: set[str] = field(default_factory=set)


class ParcelEventEngine:
    def __init__(self, zones_by_role: dict[str, list[dict[str, Any]]], source_roles: dict[str, str]):
        self.zones_by_role = zones_by_role
        self.source_roles = source_roles
        self.parcels: dict[str, ParcelEvidence] = {}

    def update(self, camera_id: str, parcel_tracks: list[TrackObservation], context_tracks: list[TrackObservation]) -> list[dict[str, Any]]:
        role = self.source_roles.get(camera_id, camera_id)
        zones = self.zones_by_role.get(role, [])
        wrists = [track for track in context_tracks if "wrist" in track.class_name]
        persons = [track for track in context_tracks if track.class_name == "person"]
        events: list[dict[str, Any]] = []
        for parcel_track in parcel_tracks:
            parcel_key = str(parcel_track.extra.get("parcel_hint") or f"{camera_id}:{parcel_track.local_track_id}")
            evidence = self.parcels.setdefault(parcel_key, ParcelEvidence(parcel_key=parcel_key))
            evidence.last_timestamp = parcel_track.timestamp
            evidence.last_bbox = parcel_track.bbox

            parcel_center = bbox_center(parcel_track.bbox)
            closest_wrist = min((euclidean(parcel_center, bbox_center(w.bbox)) for w in wrists), default=10.0)
            hand_overlap = max((bbox_iou(parcel_track.bbox, w.bbox) for w in wrists), default=0.0)
            person_overlap = max((bbox_iou(parcel_track.bbox, p.bbox) for p in persons), default=0.0)
            speed = math.sqrt(parcel_track.velocity[0] ** 2 + parcel_track.velocity[1] ** 2)
            current_zone = zone_for_bbox(parcel_track.bbox, zones)
            exit_signal = 1.0 if current_zone and "exit" in current_zone else 0.0
            destination_signal = 1.0 if current_zone and current_zone.startswith("zone_") else 0.0
            stillness = max(0.0, 1.0 - min(speed / 0.25, 1.0))
            proximity_signal = max(0.0, 1.0 - min(closest_wrist / 0.18, 1.0))

            evidence.pickup_score = max(
                0.0,
                evidence.pickup_score * 0.80
                + 0.35 * proximity_signal
                + 0.30 * min(hand_overlap * 4.0, 1.0)
                + 0.20 * min(person_overlap * 3.0, 1.0)
                + 0.15 * exit_signal,
            )
            evidence.carry_score = max(
                0.0,
                evidence.carry_score * 0.82
                + 0.40 * min(person_overlap * 3.0, 1.0)
                + 0.30 * min(speed / 0.4, 1.0)
                + 0.30 * proximity_signal,
            )
            evidence.drop_score = max(
                0.0,
                evidence.drop_score * 0.82
                + 0.35 * destination_signal
                + 0.25 * stillness
                + 0.20 * max(0.0, 1.0 - proximity_signal)
                + 0.20 * max(0.0, 1.0 - min(person_overlap * 3.0, 1.0)),
            )

            if evidence.state == ParcelState.ON_CONVEYOR and evidence.pickup_score > 0.85:
                evidence.state = ParcelState.PICK_CANDIDATE
                events.append(self._event("pickup_candidate", parcel_key, camera_id, parcel_track, evidence))
            if evidence.state in {ParcelState.ON_CONVEYOR, ParcelState.PICK_CANDIDATE} and evidence.pickup_score > 1.18:
                evidence.state = ParcelState.PICKED
                events.extend(self._once("parcel_picked", evidence, parcel_key, camera_id, parcel_track))
            if evidence.state in {ParcelState.PICKED, ParcelState.PICK_CANDIDATE} and evidence.carry_score > 0.95:
                evidence.state = ParcelState.CARRIED
                events.extend(self._once("parcel_carried", evidence, parcel_key, camera_id, parcel_track))
            if evidence.state == ParcelState.CARRIED and evidence.drop_score > 0.82:
                evidence.state = ParcelState.DROP_CANDIDATE
                evidence.destination_zone = current_zone
                events.append(self._event("drop_candidate", parcel_key, camera_id, parcel_track, evidence))
            if evidence.state in {ParcelState.DROP_CANDIDATE, ParcelState.CARRIED} and evidence.drop_score > 1.10 and destination_signal > 0:
                evidence.state = ParcelState.DROPPED
                evidence.destination_zone = current_zone
                events.extend(self._once("parcel_dropped", evidence, parcel_key, camera_id, parcel_track))

            if evidence.state == ParcelState.PICK_CANDIDATE and 0.72 < evidence.pickup_score < 1.0:
                events.extend(self._once("pickup_ambiguous", evidence, parcel_key, camera_id, parcel_track))
            if evidence.state == ParcelState.DROP_CANDIDATE and destination_signal == 0:
                events.extend(self._once("drop_ambiguous", evidence, parcel_key, camera_id, parcel_track))

        return events

    def _event(self, event_type: str, parcel_key: str, camera_id: str, track: TrackObservation, evidence: ParcelEvidence) -> dict[str, Any]:
        return {
            "event_type": event_type,
            "parcel_id": parcel_key,
            "camera_id": camera_id,
            "payload": {
                "bbox": track.bbox,
                "state": evidence.state.value,
                "pickup_score": evidence.pickup_score,
                "carry_score": evidence.carry_score,
                "drop_score": evidence.drop_score,
                "destination_zone": evidence.destination_zone,
                "validated_on_site": False,
            },
        }

    def _once(self, event_type: str, evidence: ParcelEvidence, parcel_key: str, camera_id: str, track: TrackObservation) -> list[dict[str, Any]]:
        if event_type in evidence.emitted:
            return []
        evidence.emitted.add(event_type)
        return [self._event(event_type, parcel_key, camera_id, track, evidence)]
