from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from visionsort.core.enums import ParcelState
from visionsort.core.site_config import zone_kind
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
    pickup_since: float | None = None
    carry_since: float | None = None
    drop_since: float | None = None
    validated_on_site: bool = False
    emitted: set[str] = field(default_factory=set)


class ParcelEventEngine:
    def __init__(
        self,
        zones_by_role: dict[str, list[dict[str, Any]]],
        source_roles: dict[str, str],
        confirmation_seconds: dict[str, float] | None = None,
    ):
        self.zones_by_role = zones_by_role
        self.source_roles = source_roles
        self.parcels: dict[str, ParcelEvidence] = {}
        self.confirmation_seconds = {
            "pickup_candidate": 0.05,
            "picked": 0.15,
            "carried": 0.25,
            "drop_candidate": 0.05,
            "dropped": 0.20,
            **(confirmation_seconds or {}),
        }

    @staticmethod
    def _sustained_since(
        current: float | None, active: bool, timestamp: float
    ) -> float | None:
        if not active:
            return None
        return timestamp if current is None else current

    def update(self, camera_id: str, parcel_tracks: list[TrackObservation], context_tracks: list[TrackObservation]) -> list[dict[str, Any]]:
        role = self.source_roles.get(camera_id, camera_id)
        zones = self.zones_by_role.get(role, [])
        wrists = [track for track in context_tracks if "wrist" in track.class_name]
        persons = [track for track in context_tracks if track.class_name == "person"]
        events: list[dict[str, Any]] = []
        for parcel_track in parcel_tracks:
            parcel_key = f"{camera_id}:{parcel_track.local_track_id}"
            evidence = self.parcels.setdefault(parcel_key, ParcelEvidence(parcel_key=parcel_key))
            timestamp = float(parcel_track.timestamp_global)
            evidence.last_timestamp = timestamp
            evidence.last_bbox = parcel_track.bbox

            iw = float(parcel_track.extra.get("_image_w") or 1.0)
            ih = float(parcel_track.extra.get("_image_h") or 1.0)
            parcel_center_px = bbox_center(parcel_track.bbox)
            parcel_center = (parcel_center_px[0] / iw, parcel_center_px[1] / ih)
            wrist_points: list[tuple[float, float]] = []
            for wrist in wrists:
                wx, wy = bbox_center(wrist.bbox)
                wrist_points.append((wx / iw, wy / ih))
            for person in persons:
                keypoints = person.extra.get("keypoints") or []
                for keypoint_index in (9, 10):  # COCO left/right wrist
                    if keypoint_index >= len(keypoints):
                        continue
                    x, y, *confidence = keypoints[keypoint_index]
                    if confidence and float(confidence[0]) <= 0.0:
                        continue
                    x_value, y_value = float(x), float(y)
                    if x_value > 1.0 or y_value > 1.0:
                        x_value, y_value = x_value / iw, y_value / ih
                    wrist_points.append((x_value, y_value))
            closest_wrist = min(
                (euclidean(parcel_center, point) for point in wrist_points),
                default=10.0,
            )
            hand_overlap = max((bbox_iou(parcel_track.bbox, w.bbox) for w in wrists), default=0.0)
            person_overlap = max((bbox_iou(parcel_track.bbox, p.bbox) for p in persons), default=0.0)
            speed = math.sqrt((parcel_track.velocity[0] / iw) ** 2 + (parcel_track.velocity[1] / ih) ** 2)
            current_zone = zone_for_bbox(
                parcel_track.bbox,
                zones,
                image_size=(int(iw), int(ih)),
            )
            current_kind = zone_kind(zones, current_zone)
            exit_signal = 1.0 if current_kind == "exit" else 0.0
            pick_zone_signal = 1.0 if current_kind == "pick" else 0.0
            destination_signal = 1.0 if current_kind == "destination" else 0.0
            stillness = max(0.0, 1.0 - min(speed / 0.25, 1.0))
            proximity_signal = max(0.0, 1.0 - min(closest_wrist / 0.18, 1.0))

            evidence.pickup_score = min(
                1.5,
                0.35 * proximity_signal
                + 0.30 * min(hand_overlap * 4.0, 1.0)
                + 0.20 * min(person_overlap * 3.0, 1.0)
                + 0.10 * pick_zone_signal
                + 0.05 * exit_signal,
            )
            evidence.carry_score = min(
                1.5,
                0.40 * min(person_overlap * 3.0, 1.0)
                + 0.30 * min(speed / 0.4, 1.0)
                + 0.30 * proximity_signal,
            )
            evidence.drop_score = min(
                1.5,
                0.35 * destination_signal
                + 0.25 * stillness
                + 0.20 * max(0.0, 1.0 - proximity_signal)
                + 0.20 * max(0.0, 1.0 - min(person_overlap * 3.0, 1.0)),
            )
            pickup_active = evidence.pickup_score >= 0.30
            carry_active = evidence.carry_score >= 0.30
            drop_active = destination_signal > 0 and evidence.drop_score >= 0.55
            evidence.pickup_since = self._sustained_since(
                evidence.pickup_since, pickup_active, timestamp
            )
            evidence.carry_since = self._sustained_since(
                evidence.carry_since, carry_active, timestamp
            )
            evidence.drop_since = self._sustained_since(
                evidence.drop_since, drop_active, timestamp
            )
            pickup_duration = (
                timestamp - evidence.pickup_since
                if evidence.pickup_since is not None
                else 0.0
            )
            carry_duration = (
                timestamp - evidence.carry_since
                if evidence.carry_since is not None
                else 0.0
            )
            drop_duration = (
                timestamp - evidence.drop_since
                if evidence.drop_since is not None
                else 0.0
            )

            if (
                evidence.state == ParcelState.ON_CONVEYOR
                and pickup_duration >= self.confirmation_seconds["pickup_candidate"]
            ):
                evidence.state = ParcelState.PICK_CANDIDATE
                events.extend(self._once("pickup_candidate", evidence, parcel_key, camera_id, parcel_track))
            if (
                evidence.state in {ParcelState.ON_CONVEYOR, ParcelState.PICK_CANDIDATE}
                and pickup_duration >= self.confirmation_seconds["picked"]
            ):
                evidence.state = ParcelState.PICKED
                events.extend(self._once("parcel_picked", evidence, parcel_key, camera_id, parcel_track))
            if (
                evidence.state in {ParcelState.PICKED, ParcelState.PICK_CANDIDATE}
                and carry_duration >= self.confirmation_seconds["carried"]
            ):
                evidence.state = ParcelState.CARRIED
                events.extend(self._once("parcel_carried", evidence, parcel_key, camera_id, parcel_track))
            if drop_active:
                evidence.destination_zone = current_zone
                events.extend(
                    self._once(
                        "destination_observed",
                        evidence,
                        parcel_key,
                        camera_id,
                        parcel_track,
                    )
                )
            if (
                evidence.state == ParcelState.CARRIED
                and drop_duration >= self.confirmation_seconds["drop_candidate"]
            ):
                evidence.state = ParcelState.DROP_CANDIDATE
                evidence.destination_zone = current_zone
                events.extend(self._once("drop_candidate", evidence, parcel_key, camera_id, parcel_track))
            if drop_duration >= self.confirmation_seconds["dropped"]:
                events.extend(
                    self._once(
                        "destination_confirmed",
                        evidence,
                        parcel_key,
                        camera_id,
                        parcel_track,
                    )
                )
            if (
                evidence.state in {ParcelState.DROP_CANDIDATE, ParcelState.CARRIED}
                and drop_duration >= self.confirmation_seconds["dropped"]
            ):
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
            "frame_index": track.frame_index,
            "timestamp_global": track.timestamp_global,
            "model_id": track.model_id,
            "tracker_id": track.tracker_id,
            "payload": {
                "bbox": track.bbox,
                "bbox_normalized": (
                    track.bbox[0] / float(track.extra.get("_image_w") or 1.0),
                    track.bbox[1] / float(track.extra.get("_image_h") or 1.0),
                    track.bbox[2] / float(track.extra.get("_image_w") or 1.0),
                    track.bbox[3] / float(track.extra.get("_image_h") or 1.0),
                ),
                "state": evidence.state.value,
                "pickup_score": evidence.pickup_score,
                "carry_score": evidence.carry_score,
                "drop_score": evidence.drop_score,
                "destination_zone": evidence.destination_zone,
                "zone_kind": zone_kind(
                    self.zones_by_role.get(
                        self.source_roles.get(camera_id, camera_id), []
                    ),
                    evidence.destination_zone,
                ),
                "confirmation_seconds": dict(self.confirmation_seconds),
                "validated_on_site": False,
                "site_validation_status": "NON_VALIDÉ_SUR_SITE",
                "coordinates_normalized": True,
            },
        }

    def _once(self, event_type: str, evidence: ParcelEvidence, parcel_key: str, camera_id: str, track: TrackObservation) -> list[dict[str, Any]]:
        if event_type in evidence.emitted:
            return []
        evidence.emitted.add(event_type)
        return [self._event(event_type, parcel_key, camera_id, track, evidence)]
