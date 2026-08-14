from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from visionsort.core.enums import DestinationResult, MatchResult, ParcelState


class FrameSource(Protocol):
    def open(self) -> None:
        ...

    def read(self) -> "Frame | None":
        ...

    def close(self) -> None:
        ...


@dataclass(slots=True)
class Frame:
    session_id: str
    camera_id: str
    camera_role: str
    frame_index: int
    timestamp_local: float
    timestamp_global: float
    image: np.ndarray
    source_fps: float = 0.0
    stream_epoch: int = 0


@dataclass(slots=True)
class Observation:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    model_id: str | None = None
    model_version: str | None = None
    mask: list[list[float]] | None = None
    keypoints: list[tuple[float, float, float]] | None = None
    embedding: list[float] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrackObservation:
    session_id: str
    source_id: str
    camera_id: str
    camera_role: str
    local_track_id: int
    frame_index: int
    timestamp_local: float
    timestamp_global: float
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    velocity: tuple[float, float]
    zone_id: str | None = None
    appearance_hint: list[float] | None = None
    model_id: str | None = None
    tracker_id: str | None = None
    backend_track_id: int | None = None
    anchor_px: tuple[float, float] | None = None
    anchor_world_m: tuple[float, float] | None = None
    anchor_method: str | None = None
    world_valid: bool = False
    identity_status: str = "STABLE"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Tracklet:
    tracklet_id: str
    session_id: str
    source_id: str
    camera_id: str
    camera_role: str
    local_track_id: int
    started_at_local: float
    ended_at_local: float
    started_at_global: float
    ended_at_global: float
    class_name: str
    first_bbox: tuple[float, float, float, float]
    last_bbox: tuple[float, float, float, float]
    avg_speed: float
    last_zone_id: str | None
    frame_count: int
    observation_path: str
    summary_json: dict[str, Any]
    model_id: str | None = None
    tracker_id: str | None = None
    backend_track_ids: list[int] = field(default_factory=list)
    integrity_status: str = "STABLE"


@dataclass(frozen=True, slots=True)
class WristPoint:
    x: float
    y: float
    confidence: float
    side: str
    source: str


@dataclass(frozen=True, slots=True)
class PersonTrackSample:
    timestamp_global: float
    bbox: tuple[float, float, float, float]
    left_wrist: WristPoint | None
    right_wrist: WristPoint | None


@dataclass(frozen=True, slots=True)
class PersonTrack:
    person_track_id: int
    bbox: tuple[float, float, float, float]
    left_wrist: WristPoint | None
    right_wrist: WristPoint | None
    confidence: float
    timestamp_global: float
    history: tuple[PersonTrackSample, ...] = ()
    source_operator_id: str | None = None


@dataclass(frozen=True, slots=True)
class InteractionMatch:
    parcel_key: str
    person_track_id: int
    score: float
    reliable: bool
    ambiguous: bool
    hand_distance: float
    hand_overlap: float
    contact: bool
    contact_duration: float
    separation_duration: float
    relative_motion: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CarriedShadow:
    parcel_key: str
    local_track_id: int
    global_parcel_id: str | None
    operator_id: str
    last_reliable_timestamp: float
    state: ParcelState = ParcelState.CARRIED


@dataclass(slots=True)
class GlobalParcel:
    parcel_id: str
    state: ParcelState
    last_camera_id: str
    first_seen_at: float
    last_seen_at: float
    current_tracklet_id: str
    expected_destination: str | None = None
    observed_destination: str | None = None
    destination_result: DestinationResult = DestinationResult.DESTINATION_UNVERIFIED
    operator_id: str | None = None
    appearance_signature: list[float] | None = None


def evaluate_destination(
    expected_destination: str | None,
    observed_destination: str | None,
) -> DestinationResult:
    """Apply the explicit routing contract without inventing an expectation."""
    if expected_destination is None or observed_destination is None:
        return DestinationResult.DESTINATION_UNVERIFIED
    if observed_destination == expected_destination:
        return DestinationResult.SORT_OK
    return DestinationResult.WRONG_DESTINATION


@dataclass(slots=True)
class HandoffCandidate:
    from_tracklet_id: str
    to_tracklet_id: str
    score: float
    result: MatchResult
    reasons: list[str]
