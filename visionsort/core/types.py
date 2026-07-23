from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from visionsort.core.enums import MatchResult, ParcelState


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


@dataclass(slots=True)
class GlobalParcel:
    parcel_id: str
    state: ParcelState
    last_camera_id: str
    first_seen_at: float
    last_seen_at: float
    current_tracklet_id: str
    assigned_destination: str | None = None
    operator_id: str | None = None
    appearance_signature: list[float] | None = None


@dataclass(slots=True)
class HandoffCandidate:
    from_tracklet_id: str
    to_tracklet_id: str
    score: float
    result: MatchResult
    reasons: list[str]
