from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Iterable

from visionsort.core.types import (
    PersonTrack,
    PersonTrackSample,
    TrackObservation,
    WristPoint,
)
from visionsort.tracking.engine import bbox_center


class PersonTrackBuilder:
    """Build operator views from the existing ByteTrack person identities.

    This class is deliberately not a tracker. ``person_track_id`` is the ID
    already produced by ByteTrack, and wrists are accepted only from the
    person's own pose or from an explicitly linked wrist observation.
    """

    def __init__(self, *, history_size: int = 12) -> None:
        self._history_size = max(2, int(history_size))
        self._history: dict[int, deque[PersonTrackSample]] = {}

    def update(self, tracks: Iterable[TrackObservation]) -> list[PersonTrack]:
        context = list(tracks)
        persons = [item for item in context if item.class_name == "person"]
        wrists = [item for item in context if "wrist" in item.class_name.lower()]
        built: list[PersonTrack] = []
        by_link_token: dict[str, list[int]] = {}

        for person in persons:
            person_id = self._native_person_id(person)
            for token in self._person_link_tokens(person):
                by_link_token.setdefault(token, []).append(person_id)
            left = self._pose_wrist(person, index=9, side="left")
            right = self._pose_wrist(person, index=10, side="right")
            built.append(
                PersonTrack(
                    person_track_id=person_id,
                    bbox=person.bbox,
                    left_wrist=left,
                    right_wrist=right,
                    confidence=person.confidence,
                    timestamp_global=person.timestamp_global,
                    source_operator_id=self._source_operator_id(person),
                )
            )

        built_by_id = {item.person_track_id: item for item in built}
        for wrist in wrists:
            origin = self._explicit_wrist_origin(wrist, by_link_token)
            if origin is None or origin not in built_by_id:
                continue
            person = built_by_id[origin]
            side = "right" if "right" in wrist.class_name.lower() else "left"
            x_value, y_value = bbox_center(wrist.bbox)
            point = WristPoint(
                x=x_value,
                y=y_value,
                confidence=wrist.confidence,
                side=side,
                source="EXPLICIT_WRIST_TRACK",
            )
            if side == "left" and person.left_wrist is None:
                built_by_id[origin] = replace(person, left_wrist=point)
            elif side == "right" and person.right_wrist is None:
                built_by_id[origin] = replace(person, right_wrist=point)

        output: list[PersonTrack] = []
        current_ids = set(built_by_id)
        for person_id in sorted(current_ids):
            person = built_by_id[person_id]
            history = self._history.setdefault(
                person_id, deque(maxlen=self._history_size)
            )
            history.append(
                PersonTrackSample(
                    timestamp_global=person.timestamp_global,
                    bbox=person.bbox,
                    left_wrist=person.left_wrist,
                    right_wrist=person.right_wrist,
                )
            )
            output.append(replace(person, history=tuple(history)))

        if persons:
            latest = max(item.timestamp_global for item in persons)
            for person_id in list(self._history):
                history = self._history[person_id]
                if (
                    person_id not in current_ids
                    and history
                    and latest - history[-1].timestamp_global > 2.0
                ):
                    self._history.pop(person_id, None)
        return output

    @staticmethod
    def _source_operator_id(person: TrackObservation) -> str | None:
        value = person.extra.get("operator_id")
        return str(value) if value is not None and str(value) else None

    @classmethod
    def _person_link_tokens(cls, person: TrackObservation) -> set[str]:
        tokens = {
            str(person.local_track_id),
            str(cls._native_person_id(person)),
        }
        for key in ("person_track_id", "operator_id"):
            value = person.extra.get(key)
            if value is not None and str(value):
                tokens.add(str(value))
        return tokens

    @staticmethod
    def _native_person_id(person: TrackObservation) -> int:
        """Keep the operator identity aligned with native ByteTrack tracklets."""
        return int(
            person.backend_track_id
            if person.backend_track_id is not None
            else person.local_track_id
        )

    @staticmethod
    def _explicit_wrist_origin(
        wrist: TrackObservation, by_link_token: dict[str, list[int]]
    ) -> int | None:
        for key in ("person_track_id", "operator_id"):
            value = wrist.extra.get(key)
            if value is None:
                continue
            matches = by_link_token.get(str(value), [])
            if len(matches) == 1:
                return matches[0]
        return None

    @staticmethod
    def _pose_wrist(
        person: TrackObservation, *, index: int, side: str
    ) -> WristPoint | None:
        keypoints = person.extra.get("keypoints") or []
        if index >= len(keypoints):
            return None
        point = keypoints[index]
        if len(point) < 2:
            return None
        confidence = float(point[2]) if len(point) >= 3 else 1.0
        if confidence <= 0.0:
            return None
        x_value, y_value = float(point[0]), float(point[1])
        image_w = float(person.extra.get("_image_w") or 1.0)
        image_h = float(person.extra.get("_image_h") or 1.0)
        if x_value <= 1.5 and y_value <= 1.5 and image_w > 1.0 and image_h > 1.0:
            x_value *= image_w
            y_value *= image_h
        return WristPoint(
            x=x_value,
            y=y_value,
            confidence=confidence,
            side=side,
            source="PERSON_POSE",
        )
