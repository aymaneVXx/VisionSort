from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from visionsort.core.enums import ParcelState
from visionsort.core.site_config import zone_kind
from visionsort.core.types import CarriedShadow, InteractionMatch, TrackObservation
from visionsort.events.interactions import InteractionMatcher
from visionsort.tracking.engine import zone_for_bbox
from visionsort.tracking.person import PersonTrackBuilder


@dataclass(slots=True)
class ParcelEvidence:
    parcel_key: str
    state: ParcelState = ParcelState.ON_CONVEYOR
    pickup_score: float = 0.0
    carry_score: float = 0.0
    drop_score: float = 0.0
    destination_zone: str | None = None
    operator_id: str | None = None
    candidate_operator_id: str | None = None
    last_timestamp: float = 0.0
    last_bbox: tuple[float, float, float, float] | None = None
    pickup_since: float | None = None
    carry_since: float | None = None
    drop_since: float | None = None
    shadow: CarriedShadow | None = None
    global_parcel_id: str | None = None
    validated_on_site: bool = False
    emitted: set[str] = field(default_factory=set)


class ParcelEventEngine:
    """Conservative parcel/operator state machine fed by stable local IDs."""

    def __init__(
        self,
        zones_by_role: dict[str, list[dict[str, Any]]],
        source_roles: dict[str, str],
        confirmation_seconds: dict[str, float] | None = None,
    ) -> None:
        self.zones_by_role = zones_by_role
        self.source_roles = source_roles
        self.parcels: dict[str, ParcelEvidence] = {}
        self.person_tracks = PersonTrackBuilder()
        self.interactions = InteractionMatcher()
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

    @staticmethod
    def _duration(since: float | None, timestamp: float) -> float:
        return timestamp - since if since is not None else 0.0

    @staticmethod
    def _operator_id(camera_id: str, match: InteractionMatch) -> str:
        return f"{camera_id}:{match.person_track_id}"

    def update(
        self,
        camera_id: str,
        parcel_tracks: list[TrackObservation],
        context_tracks: list[TrackObservation],
        *,
        timestamp_global: float | None = None,
    ) -> list[dict[str, Any]]:
        role = self.source_roles.get(camera_id, camera_id)
        zones = self.zones_by_role.get(role, [])
        timestamp = self._frame_timestamp(
            parcel_tracks, context_tracks, explicit=timestamp_global
        )
        persons = self.person_tracks.update(context_tracks)
        pick_zone_by_parcel: dict[str, bool] = {}
        zones_by_parcel: dict[str, tuple[str | None, str | None]] = {}
        for track in parcel_tracks:
            parcel_key = f"{camera_id}:{track.local_track_id}"
            image_size = self._image_size(track)
            zone_id = zone_for_bbox(track.bbox, zones, image_size=image_size)
            kind = zone_kind(zones, zone_id)
            zones_by_parcel[parcel_key] = (zone_id, kind)
            pick_zone_by_parcel[parcel_key] = kind == "pick"
        matches = self.interactions.match(
            parcel_tracks,
            persons,
            pick_zone_by_parcel=pick_zone_by_parcel,
        )

        events: list[dict[str, Any]] = []
        visible_keys = {
            f"{camera_id}:{track.local_track_id}" for track in parcel_tracks
        }
        if timestamp is not None:
            events.extend(
                self._update_carried_shadows(
                    camera_id, visible_keys, timestamp
                )
            )

        for track in parcel_tracks:
            parcel_key = f"{camera_id}:{track.local_track_id}"
            evidence = self.parcels.setdefault(
                parcel_key, ParcelEvidence(parcel_key=parcel_key)
            )
            current_time = float(track.timestamp_global)
            evidence.last_timestamp = current_time
            evidence.last_bbox = track.bbox
            if track.extra.get("global_parcel_id") is not None:
                evidence.global_parcel_id = str(track.extra["global_parcel_id"])
            current_zone, current_kind = zones_by_parcel[parcel_key]
            match = matches.get(parcel_key)
            integrity_ok = str(track.identity_status).upper() != "AMBIGUOUS"

            if not integrity_ok or (match is not None and match.ambiguous):
                evidence.pickup_since = None
                evidence.drop_since = None
                events.extend(
                    self._once(
                        "interaction_ambiguous",
                        evidence,
                        camera_id,
                        track,
                        match,
                    )
                )
                continue

            speed = self._normalized_speed(track)
            dynamic = speed >= 0.005 and (
                match is not None and match.relative_motion >= 0.35
            )
            operator_id = (
                self._operator_id(camera_id, match) if match is not None else None
            )
            pickup_active = bool(
                match
                and match.reliable
                and match.contact
                and current_kind == "pick"
                and dynamic
            )
            if pickup_active and evidence.candidate_operator_id in {None, operator_id}:
                evidence.candidate_operator_id = operator_id
            else:
                pickup_active = False
                if evidence.state in {
                    ParcelState.ON_CONVEYOR,
                    ParcelState.PICK_CANDIDATE,
                }:
                    evidence.candidate_operator_id = None
            evidence.pickup_since = self._sustained_since(
                evidence.pickup_since, pickup_active, current_time
            )
            evidence.pickup_score = match.score if pickup_active and match else 0.0
            pickup_duration = self._duration(evidence.pickup_since, current_time)

            if evidence.state == ParcelState.PICK_CANDIDATE and not pickup_active:
                evidence.state = ParcelState.ON_CONVEYOR
                evidence.emitted.discard("pickup_candidate")
                events.extend(
                    self._once(
                        "pickup_cancelled", evidence, camera_id, track, match
                    )
                )

            if (
                evidence.state == ParcelState.ON_CONVEYOR
                and pickup_duration
                >= self.confirmation_seconds["pickup_candidate"]
            ):
                evidence.state = ParcelState.PICK_CANDIDATE
                events.extend(
                    self._once(
                        "pickup_candidate", evidence, camera_id, track, match
                    )
                )
            if (
                evidence.state == ParcelState.PICK_CANDIDATE
                and pickup_duration >= self.confirmation_seconds["picked"]
                and operator_id == evidence.candidate_operator_id
            ):
                evidence.state = ParcelState.PICKED
                evidence.operator_id = operator_id
                events.extend(
                    self._once("parcel_picked", evidence, camera_id, track, match)
                )

            same_operator = bool(
                match
                and operator_id == evidence.operator_id
                and match.reliable
                and not match.ambiguous
            )
            carry_active = bool(
                evidence.state in {ParcelState.PICKED, ParcelState.CARRIED}
                and same_operator
                and dynamic
                and (match.contact or match.hand_distance <= 0.14)
            )
            evidence.carry_since = self._sustained_since(
                evidence.carry_since, carry_active, current_time
            )
            evidence.carry_score = match.score if carry_active and match else 0.0
            if (
                evidence.state == ParcelState.PICKED
                and self._duration(evidence.carry_since, current_time)
                >= self.confirmation_seconds["carried"]
            ):
                evidence.state = ParcelState.CARRIED
                events.extend(
                    self._once("parcel_carried", evidence, camera_id, track, match)
                )

            if evidence.shadow is not None:
                coherent_reappearance = bool(
                    same_operator
                    and evidence.shadow.operator_id == evidence.operator_id
                )
                if coherent_reappearance:
                    evidence.shadow = None
                    events.extend(
                        self._once(
                            "parcel_reappeared", evidence, camera_id, track, match
                        )
                    )
                else:
                    evidence.drop_since = None
                    continue

            association_coherent = bool(
                match
                and operator_id == evidence.operator_id
                and not match.ambiguous
                and match.score >= 0.20
            )
            drop_active = bool(
                evidence.state in {ParcelState.CARRIED, ParcelState.DROP_CANDIDATE}
                and current_kind == "destination"
                and association_coherent
                and not match.contact
                and match.separation_duration > 0.0
                and speed <= 0.03
            )
            evidence.drop_since = self._sustained_since(
                evidence.drop_since, drop_active, current_time
            )
            evidence.drop_score = (
                match.score + min(match.separation_duration, 0.25)
                if drop_active and match
                else 0.0
            )
            drop_duration = self._duration(evidence.drop_since, current_time)
            if evidence.state == ParcelState.DROP_CANDIDATE and not drop_active:
                evidence.state = ParcelState.CARRIED
                evidence.destination_zone = None
                evidence.emitted.discard("drop_candidate")
                events.extend(
                    self._once(
                        "drop_cancelled", evidence, camera_id, track, match
                    )
                )
            if (
                evidence.state == ParcelState.CARRIED
                and drop_duration
                >= self.confirmation_seconds["drop_candidate"]
            ):
                evidence.state = ParcelState.DROP_CANDIDATE
                evidence.destination_zone = current_zone
                events.extend(
                    self._once("destination_observed", evidence, camera_id, track, match)
                )
                events.extend(
                    self._once("drop_candidate", evidence, camera_id, track, match)
                )
            if (
                evidence.state == ParcelState.DROP_CANDIDATE
                and drop_duration >= self.confirmation_seconds["dropped"]
            ):
                evidence.state = ParcelState.DROPPED
                evidence.destination_zone = current_zone
                events.extend(
                    self._once(
                        "destination_confirmed", evidence, camera_id, track, match
                    )
                )
                events.extend(
                    self._once("parcel_dropped", evidence, camera_id, track, match)
                )
        return events

    @staticmethod
    def _frame_timestamp(
        parcel_tracks: list[TrackObservation],
        context_tracks: list[TrackObservation],
        *,
        explicit: float | None,
    ) -> float | None:
        if explicit is not None:
            return float(explicit)
        timestamps = [
            item.timestamp_global for item in [*parcel_tracks, *context_tracks]
        ]
        return max(timestamps, default=None)

    @staticmethod
    def _image_size(track: TrackObservation) -> tuple[int, int]:
        return (
            max(1, int(track.extra.get("_image_w") or 1)),
            max(1, int(track.extra.get("_image_h") or 1)),
        )

    @classmethod
    def _normalized_speed(cls, track: TrackObservation) -> float:
        width, height = cls._image_size(track)
        return math.hypot(track.velocity[0] / width, track.velocity[1] / height)

    def _update_carried_shadows(
        self,
        camera_id: str,
        visible_keys: set[str],
        timestamp: float,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        prefix = f"{camera_id}:"
        for parcel_key, evidence in self.parcels.items():
            if (
                not parcel_key.startswith(prefix)
                or parcel_key in visible_keys
                or evidence.state != ParcelState.CARRIED
                or evidence.operator_id is None
                or evidence.shadow is not None
            ):
                continue
            local_track_id = int(parcel_key.rsplit(":", 1)[1])
            evidence.shadow = CarriedShadow(
                parcel_key=parcel_key,
                local_track_id=local_track_id,
                global_parcel_id=evidence.global_parcel_id,
                operator_id=evidence.operator_id,
                last_reliable_timestamp=evidence.last_timestamp,
            )
            if "parcel_carried_shadow" not in evidence.emitted:
                evidence.emitted.add("parcel_carried_shadow")
                events.append(
                    {
                        "event_type": "parcel_carried_shadow",
                        "parcel_id": parcel_key,
                        "camera_id": camera_id,
                        "frame_index": None,
                        "timestamp_global": timestamp,
                        "model_id": None,
                        "tracker_id": None,
                        "payload": {
                            "state": ParcelState.CARRIED.value,
                            "operator_id": evidence.operator_id,
                            "local_track_id": local_track_id,
                            "global_parcel_id": evidence.global_parcel_id,
                            "last_reliable_timestamp": evidence.last_timestamp,
                            "identity_only": True,
                            "validated_on_site": False,
                            "site_validation_status": "NON_VALIDE_SUR_SITE",
                        },
                    }
                )
        return events

    def _event(
        self,
        event_type: str,
        camera_id: str,
        track: TrackObservation,
        evidence: ParcelEvidence,
        match: InteractionMatch | None,
    ) -> dict[str, Any]:
        image_w, image_h = self._image_size(track)
        return {
            "event_type": event_type,
            "parcel_id": evidence.parcel_key,
            "camera_id": camera_id,
            "frame_index": track.frame_index,
            "timestamp_global": track.timestamp_global,
            "model_id": track.model_id,
            "tracker_id": track.tracker_id,
            "payload": {
                "bbox": track.bbox,
                "bbox_normalized": (
                    track.bbox[0] / image_w,
                    track.bbox[1] / image_h,
                    track.bbox[2] / image_w,
                    track.bbox[3] / image_h,
                ),
                "state": evidence.state.value,
                "operator_id": evidence.operator_id
                or evidence.candidate_operator_id,
                "person_track_id": match.person_track_id if match else None,
                "interaction_score": match.score if match else 0.0,
                "contact_duration": match.contact_duration if match else 0.0,
                "separation_duration": match.separation_duration if match else 0.0,
                "interaction_reasons": list(match.reasons) if match else [],
                "pickup_score": evidence.pickup_score,
                "carry_score": evidence.carry_score,
                "drop_score": evidence.drop_score,
                "observed_destination": evidence.destination_zone,
                "destination_zone": evidence.destination_zone,
                "zone_kind": zone_kind(
                    self.zones_by_role.get(
                        self.source_roles.get(camera_id, camera_id), []
                    ),
                    evidence.destination_zone,
                ),
                "confirmation_seconds": dict(self.confirmation_seconds),
                "validated_on_site": False,
                "site_validation_status": "NON_VALIDE_SUR_SITE",
                "coordinates_normalized": True,
            },
        }

    def _once(
        self,
        event_type: str,
        evidence: ParcelEvidence,
        camera_id: str,
        track: TrackObservation,
        match: InteractionMatch | None,
    ) -> list[dict[str, Any]]:
        if event_type in evidence.emitted:
            return []
        evidence.emitted.add(event_type)
        return [self._event(event_type, camera_id, track, evidence, match)]
