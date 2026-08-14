from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import lap
import numpy as np

from visionsort.core.types import InteractionMatch, PersonTrack, TrackObservation, WristPoint


@dataclass(slots=True)
class _PairState:
    contact_since: float | None = None
    separation_since: float | None = None


@dataclass(frozen=True, slots=True)
class _PairSignals:
    parcel_key: str
    person_track_id: int
    score: float
    hand_distance: float
    hand_overlap: float
    contact: bool
    contact_duration: float
    separation_duration: float
    relative_motion: float
    integrity_ok: bool
    previous_operator: bool


def _point_in_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _point_segment_distance(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = first
    bx, by = second
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator <= 1.0e-12:
        return math.hypot(px - ax, py - ay)
    ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))


def _point_in_polygon(
    point: tuple[float, float], polygon: list[tuple[float, float]]
) -> bool:
    inside = False
    px, py = point
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if ((y1 > py) != (y2 > py)) and (
            px < (x2 - x1) * (py - y1) / (y2 - y1) + x1
        ):
            inside = not inside
        previous = current
    return inside


def _point_geometry_distance(
    point: tuple[float, float], parcel: TrackObservation
) -> tuple[float, bool]:
    mask_payload = parcel.extra.get("mask") or []
    polygon = [
        (float(item[0]), float(item[1]))
        for item in mask_payload
        if isinstance(item, (list, tuple)) and len(item) >= 2
    ]
    if len(polygon) >= 3:
        if _point_in_polygon(point, polygon):
            return 0.0, True
        return min(
            _point_segment_distance(point, polygon[index - 1], polygon[index])
            for index in range(len(polygon))
        ), False
    if _point_in_box(point, parcel.bbox):
        return 0.0, True
    px, py = point
    closest_x = min(max(px, parcel.bbox[0]), parcel.bbox[2])
    closest_y = min(max(py, parcel.bbox[1]), parcel.bbox[3])
    return math.hypot(px - closest_x, py - closest_y), False


class InteractionMatcher:
    """Associate current people and parcels on a small, explainable LAPJV subset."""

    def __init__(
        self,
        *,
        contact_distance: float = 0.08,
        candidate_distance: float = 0.28,
        ambiguity_margin: float = 0.08,
        switch_margin: float = 0.18,
    ) -> None:
        self.contact_distance = float(contact_distance)
        self.candidate_distance = float(candidate_distance)
        self.ambiguity_margin = float(ambiguity_margin)
        self.switch_margin = float(switch_margin)
        self._pairs: dict[tuple[str, int], _PairState] = {}
        self._previous_operator: dict[str, int] = {}

    def match(
        self,
        parcels: Iterable[TrackObservation],
        persons: Iterable[PersonTrack],
        *,
        pick_zone_by_parcel: dict[str, bool] | None = None,
    ) -> dict[str, InteractionMatch]:
        parcel_items = list(parcels)
        person_items = list(persons)
        pick_signals = pick_zone_by_parcel or {}
        signals: dict[tuple[str, int], _PairSignals] = {}
        seen_pairs: set[tuple[str, int]] = set()

        for parcel in parcel_items:
            parcel_key = f"{parcel.camera_id}:{parcel.local_track_id}"
            for person in person_items:
                pair = (parcel_key, person.person_track_id)
                seen_pairs.add(pair)
                item = self._signals(
                    parcel,
                    person,
                    parcel_key=parcel_key,
                    pick_zone=bool(pick_signals.get(parcel_key)),
                )
                if (
                    item.hand_distance <= self.candidate_distance
                    or item.previous_operator
                    or item.hand_overlap > 0.0
                ):
                    signals[pair] = item

        self._age_unseen_pairs(seen_pairs, parcel_items, person_items)
        if not signals:
            return {}

        parcel_keys = sorted({item.parcel_key for item in signals.values()})
        person_ids = sorted({item.person_track_id for item in signals.values()})
        parcel_index = {value: index for index, value in enumerate(parcel_keys)}
        person_index = {value: index for index, value in enumerate(person_ids)}
        scores = np.full((len(parcel_keys), len(person_ids)), -np.inf, dtype=np.float64)
        for item in signals.values():
            scores[parcel_index[item.parcel_key], person_index[item.person_track_id]] = item.score
        costs = np.full(scores.shape, 1_000_000.0, dtype=np.float64)
        compatible = np.isfinite(scores)
        costs[compatible] = 1.0 - scores[compatible]
        _, row_to_column, _ = lap.lapjv(costs, extend_cost=True, cost_limit=0.75)

        output: dict[str, InteractionMatch] = {}
        for row, column in enumerate(row_to_column.tolist()):
            if column < 0 or not compatible[row, column]:
                continue
            parcel_key = parcel_keys[row]
            person_id = person_ids[column]
            item = signals[(parcel_key, person_id)]
            alternatives = sorted(
                (
                    candidate.score
                    for candidate in signals.values()
                    if candidate.parcel_key == parcel_key
                    and candidate.person_track_id != person_id
                ),
                reverse=True,
            )
            ambiguous = bool(
                alternatives and item.score - alternatives[0] < self.ambiguity_margin
            ) or not item.integrity_ok
            reasons: list[str] = []
            if not item.integrity_ok:
                reasons.append("PARCEL_IDENTITY_AMBIGUOUS")
            if alternatives and item.score - alternatives[0] < self.ambiguity_margin:
                reasons.append("ASSOCIATION_MARGIN_TOO_SMALL")

            previous = self._previous_operator.get(parcel_key)
            hysteresis_blocked = False
            if previous is not None and previous != person_id:
                previous_signals = signals.get((parcel_key, previous))
                if (
                    previous_signals is not None
                    and item.score < previous_signals.score + self.switch_margin
                ):
                    hysteresis_blocked = True
                    reasons.append("OPERATOR_SWITCH_BLOCKED_BY_HYSTERESIS")

            minimum_score = 0.30 if item.previous_operator else 0.42
            reliable = (
                item.score >= minimum_score
                and not ambiguous
                and not hysteresis_blocked
            )
            if reliable and (item.contact or previous is None or previous == person_id):
                self._previous_operator[parcel_key] = person_id
            output[parcel_key] = InteractionMatch(
                parcel_key=parcel_key,
                person_track_id=person_id,
                score=item.score,
                reliable=reliable,
                ambiguous=ambiguous,
                hand_distance=item.hand_distance,
                hand_overlap=item.hand_overlap,
                contact=item.contact,
                contact_duration=item.contact_duration,
                separation_duration=item.separation_duration,
                relative_motion=item.relative_motion,
                reasons=tuple(reasons),
            )
        return output

    def _signals(
        self,
        parcel: TrackObservation,
        person: PersonTrack,
        *,
        parcel_key: str,
        pick_zone: bool,
    ) -> _PairSignals:
        image_w = float(parcel.extra.get("_image_w") or 1.0)
        image_h = float(parcel.extra.get("_image_h") or 1.0)
        diagonal = max(math.hypot(image_w, image_h), 1.0)
        wrists = [
            wrist
            for wrist in (person.left_wrist, person.right_wrist)
            if wrist is not None and wrist.confidence > 0.0
        ]
        distances_and_overlap = [
            _point_geometry_distance((wrist.x, wrist.y), parcel) for wrist in wrists
        ]
        if distances_and_overlap:
            raw_distance, overlap = min(distances_and_overlap, key=lambda item: item[0])
            hand_distance = raw_distance / diagonal
            hand_overlap = 1.0 if overlap else 0.0
        else:
            hand_distance = 10.0
            hand_overlap = 0.0

        timestamp = float(parcel.timestamp_global)
        pair = (parcel_key, person.person_track_id)
        state = self._pairs.setdefault(pair, _PairState())
        contact = bool(wrists) and hand_distance <= self.contact_distance
        if contact:
            state.contact_since = timestamp if state.contact_since is None else state.contact_since
            state.separation_since = None
        else:
            state.contact_since = None
            state.separation_since = (
                timestamp if state.separation_since is None else state.separation_since
            )
        contact_duration = (
            timestamp - state.contact_since if state.contact_since is not None else 0.0
        )
        separation_duration = (
            timestamp - state.separation_since if state.separation_since is not None else 0.0
        )
        relative_motion = self._relative_motion(parcel, person, diagonal)
        proximity = max(0.0, 1.0 - hand_distance / self.candidate_distance)
        previous = self._previous_operator.get(parcel_key) == person.person_track_id
        integrity_ok = str(parcel.identity_status).upper() != "AMBIGUOUS"
        score = (
            0.35 * proximity
            + 0.20 * hand_overlap
            + 0.15 * min(contact_duration / 0.15, 1.0)
            + 0.15 * relative_motion
            + 0.05 * float(pick_zone)
            + 0.10 * float(previous)
        )
        return _PairSignals(
            parcel_key=parcel_key,
            person_track_id=person.person_track_id,
            score=min(1.0, score),
            hand_distance=hand_distance,
            hand_overlap=hand_overlap,
            contact=contact,
            contact_duration=contact_duration,
            separation_duration=separation_duration,
            relative_motion=relative_motion,
            integrity_ok=integrity_ok,
            previous_operator=previous,
        )

    @staticmethod
    def _relative_motion(
        parcel: TrackObservation, person: PersonTrack, diagonal: float
    ) -> float:
        if len(person.history) < 2:
            return 0.5
        previous, current = person.history[-2], person.history[-1]
        dt = current.timestamp_global - previous.timestamp_global
        if dt <= 1.0e-6:
            return 0.5
        previous_center = (
            (previous.bbox[0] + previous.bbox[2]) / 2.0,
            (previous.bbox[1] + previous.bbox[3]) / 2.0,
        )
        current_center = (
            (current.bbox[0] + current.bbox[2]) / 2.0,
            (current.bbox[1] + current.bbox[3]) / 2.0,
        )
        person_velocity = (
            (current_center[0] - previous_center[0]) / dt,
            (current_center[1] - previous_center[1]) / dt,
        )
        difference = math.hypot(
            parcel.velocity[0] - person_velocity[0],
            parcel.velocity[1] - person_velocity[1],
        ) / diagonal
        return max(0.0, 1.0 - difference / 0.35)

    def _age_unseen_pairs(
        self,
        seen_pairs: set[tuple[str, int]],
        parcels: list[TrackObservation],
        persons: list[PersonTrack],
    ) -> None:
        timestamps = [item.timestamp_global for item in parcels] + [
            item.timestamp_global for item in persons
        ]
        if not timestamps:
            return
        now = max(timestamps)
        for pair in list(self._pairs):
            if pair in seen_pairs:
                continue
            state = self._pairs[pair]
            state.contact_since = None
            state.separation_since = now if state.separation_since is None else state.separation_since
            if now - state.separation_since > 3.0:
                self._pairs.pop(pair, None)
