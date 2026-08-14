from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import time
import uuid
from typing import Any, Iterable

import lap
import numpy as np

from visionsort.calibration.geometry import WorldGeometry
from visionsort.core.config import relative_to_root
from visionsort.core.enums import (
    LocalTrackLifecycle,
    TrackIntegrityEventType,
    TrackIntegrityStatus,
)
from visionsort.core.paths import DETAILS_DIR
from visionsort.core.types import TrackObservation, Tracklet
from visionsort.tracking.geometry import GroundAnchor, GroundAnchorEstimator


@dataclass(frozen=True, slots=True)
class TrackIntegrityConfig:
    max_occlusion_seconds: float = 0.75
    max_speed_m_s: float = 3.0
    min_relink_score: float = 0.55
    ambiguity_margin: float = 0.10

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "TrackIntegrityConfig":
        payload = values or {}
        defaults = cls()
        config = cls(
            max_occlusion_seconds=float(
                payload.get("max_occlusion_seconds", defaults.max_occlusion_seconds)
            ),
            max_speed_m_s=float(payload.get("max_speed_m_s", defaults.max_speed_m_s)),
            min_relink_score=float(
                payload.get("min_relink_score", defaults.min_relink_score)
            ),
            ambiguity_margin=float(
                payload.get("ambiguity_margin", defaults.ambiguity_margin)
            ),
        )
        if config.max_occlusion_seconds <= 0 or config.max_speed_m_s <= 0:
            raise ValueError("Les limites temporelle et physique doivent etre positives.")
        if not 0 < config.min_relink_score <= 1:
            raise ValueError("min_relink_score doit etre dans ]0, 1].")
        if not 0 < config.ambiguity_margin <= 1:
            raise ValueError("ambiguity_margin doit etre dans ]0, 1].")
        return config


@dataclass(frozen=True, slots=True)
class AnchorSample:
    timestamp: float
    pixel: tuple[float, float]
    normalized: tuple[float, float]
    world_m: tuple[float, float] | None
    bbox: tuple[float, float, float, float]


@dataclass(slots=True)
class CanonicalTrackState:
    local_track_id: int
    class_name: str
    stream_epoch: int
    lifecycle: LocalTrackLifecycle
    integrity_status: TrackIntegrityStatus
    active_backend_track_id: int | None
    backend_track_ids: set[int] = field(default_factory=set)
    history: list[TrackObservation] = field(default_factory=list)
    anchors: list[AnchorSample] = field(default_factory=list)
    last_bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_seen_at: float = 0.0
    missing_since: float | None = None
    relink_count: int = 0
    occlusion_periods: list[dict[str, float | None]] = field(default_factory=list)
    merge_group_ids: list[str] = field(default_factory=list)
    split_count: int = 0
    decision_reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OcclusionGroup:
    group_id: str
    member_local_track_ids: tuple[int, ...]
    started_at: float
    last_reliable_state: dict[int, AnchorSample]
    resolution: str = "PENDING"
    merged_backend_track_ids: set[int] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _BackendCandidate:
    observation: TrackObservation
    backend_track_id: int
    anchor: GroundAnchor


@dataclass(frozen=True, slots=True)
class _ContinuityEvidence:
    score: float
    compatible: bool
    reasons: tuple[str, ...]
    gates: dict[str, float | bool | str]


def _bbox_area(box: tuple[float, float, float, float]) -> float:
    return max(1.0, abs((box[2] - box[0]) * (box[3] - box[1])))


def _bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _bbox_intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _covers_anchor(
    box: tuple[float, float, float, float], anchor: tuple[float, float]
) -> bool:
    margin_x = max(2.0, (box[2] - box[0]) * 0.08)
    margin_y = max(2.0, (box[3] - box[1]) * 0.08)
    return (
        box[0] - margin_x <= anchor[0] <= box[2] + margin_x
        and box[1] - margin_y <= anchor[1] <= box[3] + margin_y
    )


def solve_integrity_assignment(
    score_matrix: np.ndarray, *, minimum_score: float
) -> dict[int, int]:
    """Solve only a small relink/split sub-problem, never the nominal tracking."""
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("La matrice d'association doit etre bidimensionnelle.")
    if not scores.size:
        return {}
    compatible = np.isfinite(scores) & (scores >= float(minimum_score))
    if scores.shape == (1, 1):
        return {0: 0} if compatible[0, 0] else {}
    costs = np.full(scores.shape, 1_000_000.0, dtype=np.float64)
    costs[compatible] = 1.0 - scores[compatible]
    _, row_to_column, _ = lap.lapjv(
        costs,
        extend_cost=True,
        cost_limit=max(0.0, 1.0 - float(minimum_score)) + 1.0e-12,
    )
    return {
        row: int(column)
        for row, column in enumerate(row_to_column.tolist())
        if column >= 0 and compatible[row, int(column)]
    }


def _score_matrix_payload(scores: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in np.asarray(scores, dtype=np.float64)
    ]


class TrackIntegrityManager:
    """Conservative identity layer placed after the nominal ByteTrack backend."""

    _NORMALIZED_SPEED_LIMIT_PER_S = 1.5
    _MAX_SIZE_RATIO = 2.2
    _ANCHOR_HISTORY_SIZE = 6

    def __init__(
        self,
        *,
        session_id: str,
        source_id: str,
        camera_id: str,
        camera_role: str,
        tracker_id: str,
        config: TrackIntegrityConfig | dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.source_id = source_id
        self.camera_id = camera_id
        self.camera_role = camera_role
        self.tracker_id = tracker_id
        self.config = (
            config
            if isinstance(config, TrackIntegrityConfig)
            else TrackIntegrityConfig.from_mapping(config)
        )
        self.anchor_estimator = GroundAnchorEstimator()
        self.next_local_track_id = 1
        self.current_stream_epoch: int | None = None
        self.states: dict[int, CanonicalTrackState] = {}
        self.backend_to_local: dict[int, int] = {}
        self.context_backend_to_local: dict[tuple[int, int], int] = {}
        self.occlusion_groups: dict[str, OcclusionGroup] = {}
        self._events: list[dict[str, Any]] = []
        self._frames = 0
        self._runtime_total_s = 0.0
        self._runtime_max_s = 0.0
        self._assignment_calls = 0
        self._relinks = 0
        self._ambiguous_refusals = 0
        self._created = 0

    def update(
        self,
        backend_tracks: list[TrackObservation],
        *,
        frame_index: int,
        timestamp_global: float,
        image_size: tuple[int, int],
        stream_epoch: int,
        world_geometry: WorldGeometry | None = None,
    ) -> tuple[list[TrackObservation], list[Tracklet]]:
        started = time.perf_counter()
        finalized: list[Tracklet] = []
        timestamp = float(timestamp_global)
        epoch = int(stream_epoch)
        if self.current_stream_epoch is None:
            self.current_stream_epoch = epoch
        elif epoch != self.current_stream_epoch:
            finalized.extend(
                self._finalize_all(
                    timestamp=timestamp,
                    frame_index=frame_index,
                    reason="STREAM_EPOCH_CHANGED",
                )
            )
            self.backend_to_local.clear()
            self.context_backend_to_local.clear()
            self.occlusion_groups.clear()
            self.current_stream_epoch = epoch

        parcel_candidates: list[_BackendCandidate] = []
        context_output: list[TrackObservation] = []
        for track in backend_tracks:
            backend_id = self._backend_id(track)
            if track.class_name != "parcel":
                context_output.append(self._context_observation(track, backend_id, epoch))
                continue
            anchor = self.anchor_estimator.estimate(
                bbox=track.bbox,
                mask=track.extra.get("mask"),
                image_size=image_size,
                world_geometry=world_geometry,
            )
            parcel_candidates.append(
                _BackendCandidate(
                    observation=track,
                    backend_track_id=backend_id,
                    anchor=anchor,
                )
            )

        suppressed: set[int] = set()
        canonical_output: list[TrackObservation] = []
        self._mark_missing_backends(parcel_candidates, timestamp, frame_index)
        group_output, group_suppressed = self._process_existing_groups(
            parcel_candidates,
            timestamp=timestamp,
            frame_index=frame_index,
        )
        canonical_output.extend(group_output)
        suppressed.update(group_suppressed)
        suppressed.update(
            self._detect_new_merges(
                parcel_candidates,
                already_suppressed=suppressed,
                timestamp=timestamp,
                frame_index=frame_index,
            )
        )

        unknown: list[_BackendCandidate] = []
        for candidate in parcel_candidates:
            if candidate.backend_track_id in suppressed:
                continue
            local_id = self.backend_to_local.get(candidate.backend_track_id)
            state = self.states.get(local_id) if local_id is not None else None
            if state is None or state.stream_epoch != epoch:
                unknown.append(candidate)
                continue
            recovered = state.lifecycle is LocalTrackLifecycle.TEMPORARILY_OCCLUDED
            if recovered:
                self._close_occlusion(state, timestamp)
                self._emit(
                    TrackIntegrityEventType.OCCLUSION_RECOVERED,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    local_track_ids=[state.local_track_id],
                    backend_track_ids=[candidate.backend_track_id],
                    reason="BACKEND_ID_RETURNED",
                )
            canonical_output.append(
                self._append_candidate(
                    state,
                    candidate,
                    status=(
                        TrackIntegrityStatus.AMBIGUOUS
                        if state.integrity_status is TrackIntegrityStatus.AMBIGUOUS
                        else (
                            TrackIntegrityStatus.RECOVERED
                            if recovered
                            else TrackIntegrityStatus.STABLE
                        )
                    ),
                    reasons=["BACKEND_MAPPING_KNOWN"],
                )
            )

        relinked_indices = self._relink_unknown(
            unknown,
            timestamp=timestamp,
            frame_index=frame_index,
            output=canonical_output,
        )
        for index, candidate in enumerate(unknown):
            if index in relinked_indices:
                continue
            evidence = self._best_candidate_evidence(candidate, timestamp)
            ambiguous = evidence is not None and evidence.score >= self.config.min_relink_score
            status = (
                TrackIntegrityStatus.AMBIGUOUS
                if ambiguous
                else TrackIntegrityStatus.STABLE
            )
            state = self._create_state(
                candidate,
                epoch=epoch,
                status=status,
                reason=("RELINK_AMBIGUOUS_NEW_ID" if ambiguous else "NEW_BACKEND_TRACK"),
            )
            canonical_output.append(
                self._append_candidate(
                    state,
                    candidate,
                    status=status,
                    reasons=[
                        "IDENTITY_NOT_FORCED" if ambiguous else "NEW_CANONICAL_ID"
                    ],
                )
            )
            if ambiguous:
                competing_local_ids = self._compatible_relink_local_ids(
                    candidate, timestamp
                )
                self._ambiguous_refusals += 1
                self._emit(
                    TrackIntegrityEventType.IDENTITY_AMBIGUOUS,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    local_track_ids=[state.local_track_id, *competing_local_ids],
                    backend_track_ids=[candidate.backend_track_id],
                    reason="RELINK_ALTERNATIVES_TOO_CLOSE",
                    details={
                        "association_score": evidence.score if evidence else None,
                        "gates": evidence.gates if evidence else {},
                    },
                )

        finalized.extend(self._expire_states(timestamp, frame_index))
        elapsed = time.perf_counter() - started
        self._frames += 1
        self._runtime_total_s += elapsed
        self._runtime_max_s = max(self._runtime_max_s, elapsed)
        canonical_output.sort(key=lambda item: (item.class_name, item.local_track_id))
        return context_output + canonical_output, finalized

    def flush(self, *, reason: str = "TRACKER_FLUSH") -> list[Tracklet]:
        timestamp = max(
            (state.last_seen_at for state in self.states.values()), default=time.time()
        )
        return self._finalize_all(timestamp=timestamp, frame_index=None, reason=reason)

    def pop_events(self) -> list[dict[str, Any]]:
        events, self._events = self._events, []
        return events

    def metrics(self) -> dict[str, int | float]:
        return {
            "frames": self._frames,
            "canonical_tracks_created": self._created,
            "relinks": self._relinks,
            "ambiguous_refusals": self._ambiguous_refusals,
            "lapjv_subproblems": self._assignment_calls,
            "runtime_avg_ms": (
                1000.0 * self._runtime_total_s / self._frames if self._frames else 0.0
            ),
            "runtime_max_ms": 1000.0 * self._runtime_max_s,
        }

    @staticmethod
    def _backend_id(track: TrackObservation) -> int:
        backend_id = (
            track.backend_track_id
            if track.backend_track_id is not None
            else track.local_track_id
        )
        return int(backend_id)

    def _allocate_local_id(self) -> int:
        local_id = self.next_local_track_id
        self.next_local_track_id += 1
        return local_id

    def _context_observation(
        self, track: TrackObservation, backend_id: int, epoch: int
    ) -> TrackObservation:
        key = (epoch, backend_id)
        local_id = self.context_backend_to_local.get(key)
        if local_id is None:
            local_id = self._allocate_local_id()
            self.context_backend_to_local[key] = local_id
        extra = dict(track.extra)
        extra.update(
            {
                "backend_track_id": backend_id,
                "stream_epoch": epoch,
                "track_identity": [self.camera_id, local_id],
            }
        )
        return replace(
            track,
            local_track_id=local_id,
            backend_track_id=backend_id,
            identity_status=TrackIntegrityStatus.STABLE.value,
            extra=extra,
        )

    def _create_state(
        self,
        candidate: _BackendCandidate,
        *,
        epoch: int,
        status: TrackIntegrityStatus,
        reason: str,
    ) -> CanonicalTrackState:
        local_id = self._allocate_local_id()
        state = CanonicalTrackState(
            local_track_id=local_id,
            class_name="parcel",
            stream_epoch=epoch,
            lifecycle=(
                LocalTrackLifecycle.AMBIGUOUS
                if status is TrackIntegrityStatus.AMBIGUOUS
                else LocalTrackLifecycle.ACTIVE
            ),
            integrity_status=status,
            active_backend_track_id=candidate.backend_track_id,
            backend_track_ids={candidate.backend_track_id},
            last_bbox=candidate.observation.bbox,
            last_seen_at=candidate.observation.timestamp_global,
            decision_reasons=[reason],
        )
        self.states[local_id] = state
        self.backend_to_local[candidate.backend_track_id] = local_id
        self._created += 1
        self._emit(
            TrackIntegrityEventType.LOCAL_TRACK_CREATED,
            timestamp=candidate.observation.timestamp_global,
            frame_index=candidate.observation.frame_index,
            local_track_ids=[local_id],
            backend_track_ids=[candidate.backend_track_id],
            reason=reason,
        )
        return state

    def _append_candidate(
        self,
        state: CanonicalTrackState,
        candidate: _BackendCandidate,
        *,
        status: TrackIntegrityStatus,
        reasons: list[str],
    ) -> TrackObservation:
        previous = state.history[-1] if state.history else None
        velocity = candidate.observation.velocity
        if previous is not None:
            dt = max(candidate.observation.timestamp_global - previous.timestamp_global, 1.0e-6)
            previous_center = _bbox_center(previous.bbox)
            current_center = _bbox_center(candidate.observation.bbox)
            velocity = (
                (current_center[0] - previous_center[0]) / dt,
                (current_center[1] - previous_center[1]) / dt,
            )
        extra = dict(candidate.observation.extra)
        extra.update(
            {
                "backend_track_id": candidate.backend_track_id,
                "stream_epoch": state.stream_epoch,
                "track_identity": [self.camera_id, state.local_track_id],
                "identity_reasons": list(reasons),
                "anchor_normalized": list(candidate.anchor.normalized),
            }
        )
        if candidate.anchor.world_m is not None:
            extra["anchor_world_m"] = list(candidate.anchor.world_m)
        canonical = replace(
            candidate.observation,
            local_track_id=state.local_track_id,
            backend_track_id=candidate.backend_track_id,
            velocity=velocity,
            anchor_px=candidate.anchor.pixel,
            anchor_world_m=candidate.anchor.world_m,
            anchor_method=candidate.anchor.method,
            world_valid=candidate.anchor.world_valid,
            identity_status=status.value,
            extra=extra,
        )
        state.history.append(canonical)
        state.anchors.append(
            AnchorSample(
                timestamp=canonical.timestamp_global,
                pixel=candidate.anchor.pixel,
                normalized=candidate.anchor.normalized,
                world_m=candidate.anchor.world_m,
                bbox=canonical.bbox,
            )
        )
        state.anchors = state.anchors[-self._ANCHOR_HISTORY_SIZE :]
        state.last_bbox = canonical.bbox
        state.last_seen_at = canonical.timestamp_global
        state.active_backend_track_id = candidate.backend_track_id
        state.backend_track_ids.add(candidate.backend_track_id)
        state.lifecycle = (
            LocalTrackLifecycle.AMBIGUOUS
            if status is TrackIntegrityStatus.AMBIGUOUS
            else LocalTrackLifecycle.ACTIVE
        )
        state.integrity_status = status
        state.missing_since = None
        state.decision_reasons.extend(reasons)
        self.backend_to_local[candidate.backend_track_id] = state.local_track_id
        return canonical

    def _mark_missing_backends(
        self,
        candidates: list[_BackendCandidate],
        timestamp: float,
        frame_index: int,
    ) -> None:
        active_backend_ids = {item.backend_track_id for item in candidates}
        grouped_members = {
            member
            for group in self.occlusion_groups.values()
            for member in group.member_local_track_ids
        }
        for state in self.states.values():
            backend_id = state.active_backend_track_id
            if (
                state.local_track_id in grouped_members
                or backend_id is None
                or backend_id in active_backend_ids
                or state.lifecycle in {LocalTrackLifecycle.LOST, LocalTrackLifecycle.FINALIZED}
            ):
                continue
            if state.lifecycle not in {
                LocalTrackLifecycle.TEMPORARILY_OCCLUDED,
                LocalTrackLifecycle.AMBIGUOUS,
            }:
                state.lifecycle = (
                    LocalTrackLifecycle.AMBIGUOUS
                    if state.integrity_status is TrackIntegrityStatus.AMBIGUOUS
                    else LocalTrackLifecycle.TEMPORARILY_OCCLUDED
                )
                if state.integrity_status is not TrackIntegrityStatus.AMBIGUOUS:
                    state.integrity_status = TrackIntegrityStatus.OCCLUDED
                state.missing_since = timestamp
                state.occlusion_periods.append({"started_at": timestamp, "ended_at": None})
                self._emit(
                    TrackIntegrityEventType.OCCLUSION_STARTED,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    local_track_ids=[state.local_track_id],
                    backend_track_ids=[backend_id],
                    reason="BACKEND_TRACK_MISSING",
                )

    def _relink_unknown(
        self,
        unknown: list[_BackendCandidate],
        *,
        timestamp: float,
        frame_index: int,
        output: list[TrackObservation],
    ) -> set[int]:
        if not unknown:
            return set()
        grouped_members = {
            member
            for group in self.occlusion_groups.values()
            for member in group.member_local_track_ids
        }
        states = [
            state
            for state in self.states.values()
            if state.class_name == "parcel"
            and state.stream_epoch == self.current_stream_epoch
            and state.local_track_id not in grouped_members
            and state.lifecycle is LocalTrackLifecycle.TEMPORARILY_OCCLUDED
            and 0.0 <= timestamp - state.last_seen_at <= self.config.max_occlusion_seconds
        ]
        if not states:
            return set()
        evidences = [
            [self._continuity_evidence(state, candidate, timestamp) for candidate in unknown]
            for state in states
        ]
        scores = np.asarray(
            [
                [evidence.score if evidence.compatible else -np.inf for evidence in row]
                for row in evidences
            ],
            dtype=np.float64,
        )
        if len(states) > 1 or len(unknown) > 1:
            self._assignment_calls += 1
        assignments = solve_integrity_assignment(
            scores, minimum_score=self.config.min_relink_score
        )
        accepted_unknown: set[int] = set()
        for state_index, candidate_index in assignments.items():
            score = float(scores[state_index, candidate_index])
            if not self._assignment_has_margin(scores, state_index, candidate_index):
                continue
            state = states[state_index]
            candidate = unknown[candidate_index]
            previous_backend = state.active_backend_track_id
            if previous_backend is not None:
                self.backend_to_local.pop(previous_backend, None)
            state.relink_count += 1
            self._relinks += 1
            self._close_occlusion(state, timestamp)
            output.append(
                self._append_candidate(
                    state,
                    candidate,
                    status=TrackIntegrityStatus.RECOVERED,
                    reasons=list(evidences[state_index][candidate_index].reasons),
                )
            )
            accepted_unknown.add(candidate_index)
            details = {
                "old_backend_track_id": previous_backend,
                "new_backend_track_id": candidate.backend_track_id,
                "association_score": score,
                "gates": evidences[state_index][candidate_index].gates,
            }
            self._emit(
                TrackIntegrityEventType.BACKEND_ID_RELINKED,
                timestamp=timestamp,
                frame_index=frame_index,
                local_track_ids=[state.local_track_id],
                backend_track_ids=[
                    value
                    for value in (previous_backend, candidate.backend_track_id)
                    if value is not None
                ],
                reason="CONTINUITY_GATES_AND_MARGIN_PASSED",
                details=details,
            )
            self._emit(
                TrackIntegrityEventType.OCCLUSION_RECOVERED,
                timestamp=timestamp,
                frame_index=frame_index,
                local_track_ids=[state.local_track_id],
                backend_track_ids=[candidate.backend_track_id],
                reason="RELINK_ACCEPTED",
                details={"association_score": score},
            )
        return accepted_unknown

    def _best_candidate_evidence(
        self, candidate: _BackendCandidate, timestamp: float
    ) -> _ContinuityEvidence | None:
        evidences = [
            self._continuity_evidence(state, candidate, timestamp)
            for state in self.states.values()
            if state.lifecycle is LocalTrackLifecycle.TEMPORARILY_OCCLUDED
            and state.stream_epoch == self.current_stream_epoch
        ]
        compatible = [item for item in evidences if item.compatible]
        return max(compatible, key=lambda item: item.score, default=None)

    def _compatible_relink_local_ids(
        self, candidate: _BackendCandidate, timestamp: float
    ) -> list[int]:
        return sorted(
            state.local_track_id
            for state in self.states.values()
            if state.lifecycle is LocalTrackLifecycle.TEMPORARILY_OCCLUDED
            and state.stream_epoch == self.current_stream_epoch
            and (
                evidence := self._continuity_evidence(state, candidate, timestamp)
            ).compatible
            and evidence.score >= self.config.min_relink_score
        )

    def _continuity_evidence(
        self,
        state: CanonicalTrackState,
        candidate: _BackendCandidate,
        timestamp: float,
    ) -> _ContinuityEvidence:
        gap = timestamp - state.last_seen_at
        gates: dict[str, float | bool | str] = {
            "same_class": candidate.observation.class_name == state.class_name,
            "gap_s": gap,
            "gap_allowed_s": self.config.max_occlusion_seconds,
        }
        if candidate.observation.class_name != "parcel" or state.class_name != "parcel":
            return _ContinuityEvidence(0.0, False, ("CLASS_GATE_FAILED",), gates)
        if gap < -1.0e-6 or gap > self.config.max_occlusion_seconds:
            return _ContinuityEvidence(0.0, False, ("TIME_GATE_FAILED",), gates)
        if not state.anchors:
            return _ContinuityEvidence(0.0, False, ("NO_RELIABLE_ANCHOR",), gates)

        previous_area = _bbox_area(state.last_bbox)
        current_area = _bbox_area(candidate.observation.bbox)
        size_ratio = max(previous_area, current_area) / min(previous_area, current_area)
        gates["size_ratio"] = size_ratio
        gates["max_size_ratio"] = self._MAX_SIZE_RATIO
        if size_ratio > self._MAX_SIZE_RATIO:
            return _ContinuityEvidence(0.0, False, ("SIZE_GATE_FAILED",), gates)

        use_world = candidate.anchor.world_valid and state.anchors[-1].world_m is not None
        coordinate_name = "world_m" if use_world else "normalized_image"
        predicted, residual, velocity = self._predict(state, timestamp, use_world=use_world)
        current = np.asarray(
            candidate.anchor.world_m if use_world else candidate.anchor.normalized,
            dtype=np.float64,
        )
        previous = np.asarray(
            state.anchors[-1].world_m if use_world else state.anchors[-1].normalized,
            dtype=np.float64,
        )
        speed_limit = (
            self.config.max_speed_m_s if use_world else self._NORMALIZED_SPEED_LIMIT_PER_S
        )
        base_tolerance = 0.04 if use_world else 0.025
        physical_gate = speed_limit * max(gap, 0.0) + base_tolerance
        displacement = float(np.linalg.norm(current - previous))
        prediction_gate = physical_gate + 2.0 * residual * max(gap, 1.0e-3)
        prediction_error = float(np.linalg.norm(current - predicted))
        gates.update(
            {
                "coordinate_space": coordinate_name,
                "displacement": displacement,
                "physical_gate": physical_gate,
                "prediction_error": prediction_error,
                "prediction_gate": prediction_gate,
            }
        )
        if displacement > physical_gate or prediction_error > prediction_gate:
            return _ContinuityEvidence(0.0, False, ("MOTION_GATE_FAILED",), gates)

        motion_score = 0.5
        movement = current - previous
        if np.linalg.norm(velocity) > 1.0e-9 and np.linalg.norm(movement) > 1.0e-9:
            cosine = float(
                np.dot(velocity, movement)
                / (np.linalg.norm(velocity) * np.linalg.norm(movement))
            )
            gates["direction_cosine"] = cosine
            if cosine < -0.5 and displacement > base_tolerance:
                return _ContinuityEvidence(0.0, False, ("DIRECTION_GATE_FAILED",), gates)
            motion_score = (cosine + 1.0) / 2.0
        position_score = max(0.0, 1.0 - prediction_error / max(prediction_gate, 1.0e-9))
        size_score = math.exp(-abs(math.log(size_ratio)))
        score = 0.65 * position_score + 0.25 * size_score + 0.10 * motion_score
        return _ContinuityEvidence(
            float(score),
            True,
            (
                "TIME_GATE_PASSED",
                "MOTION_GATE_PASSED",
                "SIZE_GATE_PASSED",
                f"COORDINATES_{coordinate_name.upper()}",
            ),
            gates,
        )

    def _predict(
        self,
        state: CanonicalTrackState,
        timestamp: float,
        *,
        use_world: bool,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        samples = [
            sample
            for sample in state.anchors
            if (sample.world_m is not None if use_world else True)
        ]
        coordinates = [
            np.asarray(sample.world_m if use_world else sample.normalized, dtype=np.float64)
            for sample in samples
        ]
        last = coordinates[-1]
        if len(samples) < 2:
            return last, 0.0, np.zeros(2, dtype=np.float64)
        velocities: list[np.ndarray] = []
        for index in range(1, len(samples)):
            dt = samples[index].timestamp - samples[index - 1].timestamp
            if dt > 1.0e-6:
                velocities.append((coordinates[index] - coordinates[index - 1]) / dt)
        if not velocities:
            return last, 0.0, np.zeros(2, dtype=np.float64)
        velocity_array = np.asarray(velocities, dtype=np.float64)
        velocity = np.median(velocity_array, axis=0)
        dispersion = float(
            np.median(np.linalg.norm(velocity_array - velocity, axis=1))
        )
        dt_future = max(0.0, timestamp - samples[-1].timestamp)
        return last + velocity * dt_future, dispersion, velocity

    def _assignment_has_margin(
        self, scores: np.ndarray, row: int, column: int
    ) -> bool:
        chosen = float(scores[row, column])
        row_alternatives = [
            float(scores[row, index])
            for index in range(scores.shape[1])
            if index != column and np.isfinite(scores[row, index])
        ]
        column_alternatives = [
            float(scores[index, column])
            for index in range(scores.shape[0])
            if index != row and np.isfinite(scores[index, column])
        ]
        next_best = max(row_alternatives + column_alternatives, default=-np.inf)
        return not np.isfinite(next_best) or chosen - next_best >= self.config.ambiguity_margin

    def _close_occlusion(self, state: CanonicalTrackState, timestamp: float) -> None:
        if state.occlusion_periods and state.occlusion_periods[-1]["ended_at"] is None:
            state.occlusion_periods[-1]["ended_at"] = timestamp
        state.missing_since = None

    def _detect_new_merges(
        self,
        candidates: list[_BackendCandidate],
        *,
        already_suppressed: set[int],
        timestamp: float,
        frame_index: int,
    ) -> set[int]:
        suppressed: set[int] = set()
        grouped_members = {
            member
            for group in self.occlusion_groups.values()
            for member in group.member_local_track_ids
        }
        available_states = [
            state
            for state in self.states.values()
            if state.local_track_id not in grouped_members
            and state.stream_epoch == self.current_stream_epoch
            and timestamp - state.last_seen_at <= self.config.max_occlusion_seconds
            and state.history
        ]
        for candidate in candidates:
            if candidate.backend_track_id in already_suppressed or candidate.backend_track_id in suppressed:
                continue
            members = [
                state
                for state in available_states
                if _bbox_intersection_area(candidate.observation.bbox, state.last_bbox) > 0
                or _covers_anchor(candidate.observation.bbox, state.anchors[-1].pixel)
            ]
            if len(members) < 2:
                continue
            member_area = sum(_bbox_area(state.last_bbox) for state in members)
            if _bbox_area(candidate.observation.bbox) < 0.45 * member_area:
                continue
            related_candidates = [
                item
                for item in candidates
                if item.backend_track_id not in already_suppressed
                and any(
                    _bbox_intersection_area(item.observation.bbox, state.last_bbox) > 0
                    for state in members
                )
            ]
            if len(related_candidates) >= len(members):
                continue
            group = self._start_group(members, related_candidates, timestamp, frame_index)
            suppressed.update(item.backend_track_id for item in related_candidates)
            available_states = [
                state
                for state in available_states
                if state.local_track_id not in group.member_local_track_ids
            ]
        return suppressed

    def _start_group(
        self,
        members: list[CanonicalTrackState],
        merged_candidates: list[_BackendCandidate],
        timestamp: float,
        frame_index: int,
    ) -> OcclusionGroup:
        group_id = f"merge-{uuid.uuid4()}"
        group = OcclusionGroup(
            group_id=group_id,
            member_local_track_ids=tuple(
                sorted(state.local_track_id for state in members)
            ),
            started_at=timestamp,
            last_reliable_state={state.local_track_id: state.anchors[-1] for state in members},
            merged_backend_track_ids={
                item.backend_track_id for item in merged_candidates
            },
        )
        self.occlusion_groups[group_id] = group
        old_backend_ids: list[int] = []
        for state in members:
            was_occluded = state.lifecycle is LocalTrackLifecycle.TEMPORARILY_OCCLUDED
            if state.active_backend_track_id is not None:
                old_backend_ids.append(state.active_backend_track_id)
                self.backend_to_local.pop(state.active_backend_track_id, None)
            state.active_backend_track_id = None
            state.lifecycle = LocalTrackLifecycle.TEMPORARILY_OCCLUDED
            state.integrity_status = TrackIntegrityStatus.OCCLUDED
            state.missing_since = timestamp
            state.merge_group_ids.append(group_id)
            if not state.occlusion_periods or state.occlusion_periods[-1]["ended_at"] is not None:
                state.occlusion_periods.append(
                    {"started_at": timestamp, "ended_at": None}
                )
            if not was_occluded:
                self._emit(
                    TrackIntegrityEventType.OCCLUSION_STARTED,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    local_track_ids=[state.local_track_id],
                    backend_track_ids=sorted(state.backend_track_ids),
                    reason="MERGE_GROUP_STARTED",
                )
        self._emit(
            TrackIntegrityEventType.MERGE_STARTED,
            timestamp=timestamp,
            frame_index=frame_index,
            local_track_ids=list(group.member_local_track_ids),
            backend_track_ids=old_backend_ids
            + [item.backend_track_id for item in merged_candidates],
            reason="MULTIPLE_CANONICAL_TRACKS_VISUALLY_MERGED",
            details={"group_id": group_id},
        )
        return group

    def _process_existing_groups(
        self,
        candidates: list[_BackendCandidate],
        *,
        timestamp: float,
        frame_index: int,
    ) -> tuple[list[TrackObservation], set[int]]:
        output: list[TrackObservation] = []
        suppressed: set[int] = set()
        for group_id, group in list(self.occlusion_groups.items()):
            members = [
                self.states[local_id]
                for local_id in group.member_local_track_ids
                if local_id in self.states
            ]
            if not members:
                self.occlusion_groups.pop(group_id, None)
                continue
            relevant = [
                candidate
                for candidate in candidates
                if any(
                    _bbox_intersection_area(candidate.observation.bbox, state.last_bbox) > 0
                    or self._candidate_near_state(state, candidate, timestamp)
                    for state in members
                )
            ]
            if len(relevant) < len(members):
                suppressed.update(item.backend_track_id for item in relevant)
                group.merged_backend_track_ids.update(
                    item.backend_track_id for item in relevant
                )
                if timestamp - group.started_at > self.config.max_occlusion_seconds:
                    group.resolution = "AMBIGUOUS_TIMEOUT"
                    for state in members:
                        state.lifecycle = LocalTrackLifecycle.AMBIGUOUS
                        state.integrity_status = TrackIntegrityStatus.AMBIGUOUS
                        state.decision_reasons.append("MERGE_TIMEOUT_IDENTITY_NOT_FORCED")
                    self._ambiguous_refusals += 1
                    self._emit(
                        TrackIntegrityEventType.IDENTITY_AMBIGUOUS,
                        timestamp=timestamp,
                        frame_index=frame_index,
                        local_track_ids=list(group.member_local_track_ids),
                        backend_track_ids=[item.backend_track_id for item in relevant],
                        reason="MERGE_EXCEEDED_OCCLUSION_WINDOW",
                        details={"group_id": group_id},
                    )
                    self.occlusion_groups.pop(group_id, None)
                continue
            scores = np.asarray(
                [
                    [
                        evidence.score if evidence.compatible else -np.inf
                        for evidence in (
                            self._continuity_evidence(state, candidate, timestamp)
                            for candidate in relevant
                        )
                    ]
                    for state in members
                ],
                dtype=np.float64,
            )
            self._assignment_calls += 1
            assignments = solve_integrity_assignment(
                scores, minimum_score=self.config.min_relink_score
            )
            unambiguous = len(assignments) == len(members) and all(
                self._assignment_has_margin(scores, row, column)
                for row, column in assignments.items()
            )
            if unambiguous:
                resolved_backends: list[int] = []
                for row, column in assignments.items():
                    state = members[row]
                    candidate = relevant[column]
                    resolved_backends.append(candidate.backend_track_id)
                    state.split_count += 1
                    backend_changed = candidate.backend_track_id not in state.backend_track_ids
                    state.relink_count += int(backend_changed)
                    self._relinks += int(backend_changed)
                    self._close_occlusion(state, timestamp)
                    output.append(
                        self._append_candidate(
                            state,
                            candidate,
                            status=TrackIntegrityStatus.RECOVERED,
                            reasons=["SPLIT_GLOBAL_ASSIGNMENT_ACCEPTED"],
                        )
                    )
                    self._emit(
                        TrackIntegrityEventType.OCCLUSION_RECOVERED,
                        timestamp=timestamp,
                        frame_index=frame_index,
                        local_track_ids=[state.local_track_id],
                        backend_track_ids=[candidate.backend_track_id],
                        reason="SPLIT_RESOLVED",
                    )
                group.resolution = "RESOLVED"
                self._emit(
                    TrackIntegrityEventType.SPLIT_RESOLVED,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    local_track_ids=list(group.member_local_track_ids),
                    backend_track_ids=resolved_backends,
                    reason="GLOBAL_ASSIGNMENT_WITH_CLEAR_MARGIN",
                    details={
                        "group_id": group_id,
                        "association_scores": _score_matrix_payload(scores),
                    },
                )
                self.occlusion_groups.pop(group_id, None)
                suppressed.update(candidate.backend_track_id for candidate in relevant)
                continue

            group.resolution = "AMBIGUOUS"
            for state in members:
                state.lifecycle = LocalTrackLifecycle.AMBIGUOUS
                state.integrity_status = TrackIntegrityStatus.AMBIGUOUS
                state.decision_reasons.append("SPLIT_AMBIGUOUS_IDENTITY_NOT_FORCED")
            self._ambiguous_refusals += 1
            self._emit(
                TrackIntegrityEventType.IDENTITY_AMBIGUOUS,
                timestamp=timestamp,
                frame_index=frame_index,
                local_track_ids=list(group.member_local_track_ids),
                backend_track_ids=[item.backend_track_id for item in relevant],
                reason="SPLIT_ALTERNATIVES_TOO_CLOSE",
                details={
                    "group_id": group_id,
                    "association_scores": _score_matrix_payload(scores),
                },
            )
            for candidate in relevant:
                new_state = self._create_state(
                    candidate,
                    epoch=int(self.current_stream_epoch or 0),
                    status=TrackIntegrityStatus.AMBIGUOUS,
                    reason="AMBIGUOUS_SPLIT_NEW_ID",
                )
                output.append(
                    self._append_candidate(
                        new_state,
                        candidate,
                        status=TrackIntegrityStatus.AMBIGUOUS,
                        reasons=["IDENTITY_NOT_FORCED_AFTER_SPLIT"],
                    )
                )
                suppressed.add(candidate.backend_track_id)
            self.occlusion_groups.pop(group_id, None)
        return output, suppressed

    def _candidate_near_state(
        self,
        state: CanonicalTrackState,
        candidate: _BackendCandidate,
        timestamp: float,
    ) -> bool:
        evidence = self._continuity_evidence(state, candidate, timestamp)
        return evidence.compatible

    def _expire_states(self, timestamp: float, frame_index: int) -> list[Tracklet]:
        grouped_members = {
            member
            for group in self.occlusion_groups.values()
            for member in group.member_local_track_ids
        }
        expired = [
            state.local_track_id
            for state in self.states.values()
            if state.local_track_id not in grouped_members
            and state.lifecycle
            in {LocalTrackLifecycle.TEMPORARILY_OCCLUDED, LocalTrackLifecycle.AMBIGUOUS}
            and timestamp - state.last_seen_at > self.config.max_occlusion_seconds
        ]
        return [
            self._finalize_state(
                local_id,
                timestamp=timestamp,
                frame_index=frame_index,
                reason="OCCLUSION_TIMEOUT",
            )
            for local_id in expired
        ]

    def _finalize_all(
        self,
        *,
        timestamp: float,
        frame_index: int | None,
        reason: str,
    ) -> list[Tracklet]:
        return [
            self._finalize_state(
                local_id,
                timestamp=timestamp,
                frame_index=frame_index,
                reason=reason,
            )
            for local_id in list(self.states)
        ]

    def _finalize_state(
        self,
        local_id: int,
        *,
        timestamp: float,
        frame_index: int | None,
        reason: str,
    ) -> Tracklet:
        state = self.states.pop(local_id)
        if state.active_backend_track_id is not None:
            self.backend_to_local.pop(state.active_backend_track_id, None)
        for backend_id in state.backend_track_ids:
            if self.backend_to_local.get(backend_id) == local_id:
                self.backend_to_local.pop(backend_id, None)
        self._close_occlusion(state, timestamp)
        state.lifecycle = LocalTrackLifecycle.FINALIZED
        state.decision_reasons.append(reason)
        tracklet = self._build_tracklet(state)
        self._emit(
            TrackIntegrityEventType.LOCAL_TRACK_FINALIZED,
            timestamp=timestamp,
            frame_index=frame_index,
            local_track_ids=[local_id],
            backend_track_ids=sorted(state.backend_track_ids),
            reason=reason,
            details={
                "integrity_status": state.integrity_status.value,
                "relink_count": state.relink_count,
            },
        )
        return tracklet

    def _build_tracklet(self, state: CanonicalTrackState) -> Tracklet:
        history = state.history
        if not history:
            raise RuntimeError("Un track canonique sans observation ne peut pas etre finalise.")
        DETAILS_DIR.mkdir(parents=True, exist_ok=True)
        tracklet_id = (
            f"{self.session_id}-{self.camera_id}-{state.local_track_id}-"
            f"{int(history[0].timestamp_global * 1000)}"
        )
        details_path = DETAILS_DIR / f"{tracklet_id}.jsonl"
        details_path.write_text(
            "\n".join(json.dumps(item.to_json()) for item in history) + "\n",
            encoding="utf-8",
        )
        boxes = [item.bbox for item in history]
        speeds = [math.hypot(*item.velocity) for item in history]
        avg_bbox = tuple(
            sum(box[index] for box in boxes) / len(boxes) for index in range(4)
        )
        avg_dimensions = (
            sum(abs(box[2] - box[0]) for box in boxes) / len(boxes),
            sum(abs(box[3] - box[1]) for box in boxes) / len(boxes),
        )
        avg_velocity = (
            sum(item.velocity[0] for item in history) / len(history),
            sum(item.velocity[1] for item in history) / len(history),
        )
        embeddings = [item.appearance_hint for item in history if item.appearance_hint]
        avg_embedding = None
        if embeddings:
            width = min(len(item) for item in embeddings)
            avg_embedding = [
                sum(float(item[index]) for item in embeddings) / len(embeddings)
                for index in range(width)
            ]
        ground_truth_hint = history[-1].extra.get("parcel_hint") or history[0].extra.get(
            "parcel_hint"
        )
        summary = {
            "start_frame": history[0].frame_index,
            "end_frame": history[-1].frame_index,
            "duration_s": history[-1].timestamp_global - history[0].timestamp_global,
            "first_bbox": history[0].bbox,
            "last_bbox": history[-1].bbox,
            "avg_bbox": avg_bbox,
            "avg_dimensions": avg_dimensions,
            "avg_velocity": avg_velocity,
            "first_zone_id": history[0].zone_id,
            "last_zone_id": history[-1].zone_id,
            "visited_zones": list(
                dict.fromkeys(item.zone_id for item in history if item.zone_id)
            ),
            "appearance_embedding": avg_embedding,
            "ground_truth": {"parcel_hint": ground_truth_hint} if ground_truth_hint else {},
            "model_id": history[-1].model_id,
            "tracker_id": self.tracker_id,
            "backend_track_ids": sorted(state.backend_track_ids),
            "relink_count": state.relink_count,
            "occlusion_periods": state.occlusion_periods,
            "merge_group_ids": state.merge_group_ids,
            "split_count": state.split_count,
            "integrity_status": state.integrity_status.value,
            "integrity_decision_reasons": list(dict.fromkeys(state.decision_reasons)),
            "stream_epoch": state.stream_epoch,
            "validated_on_site": False,
        }
        return Tracklet(
            tracklet_id=tracklet_id,
            session_id=self.session_id,
            source_id=self.source_id,
            camera_id=self.camera_id,
            camera_role=self.camera_role,
            local_track_id=state.local_track_id,
            started_at_local=history[0].timestamp_local,
            ended_at_local=history[-1].timestamp_local,
            started_at_global=history[0].timestamp_global,
            ended_at_global=history[-1].timestamp_global,
            class_name=state.class_name,
            first_bbox=history[0].bbox,
            last_bbox=history[-1].bbox,
            avg_speed=sum(speeds) / len(speeds),
            last_zone_id=history[-1].zone_id,
            frame_count=len(history),
            observation_path=relative_to_root(details_path),
            summary_json=summary,
            model_id=history[-1].model_id,
            tracker_id=self.tracker_id,
            backend_track_ids=sorted(state.backend_track_ids),
            integrity_status=state.integrity_status.value,
        )

    def _emit(
        self,
        event_type: TrackIntegrityEventType,
        *,
        timestamp: float,
        frame_index: int | None,
        local_track_ids: Iterable[int],
        backend_track_ids: Iterable[int],
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        local_ids = sorted({int(value) for value in local_track_ids})
        backend_ids = sorted({int(value) for value in backend_track_ids})
        payload = {
            "event_type": event_type.value,
            "camera_id": self.camera_id,
            "local_track_ids": local_ids,
            "backend_track_ids": backend_ids,
            "timestamp": float(timestamp),
            "reason": reason,
            **(details or {}),
        }
        self._events.append(
            {
                "event_type": event_type.value,
                "payload": payload,
                "camera_id": self.camera_id,
                "frame_index": frame_index,
                "timestamp_global": float(timestamp),
                "local_parcel_key": (
                    f"{self.camera_id}:{local_ids[0]}" if len(local_ids) == 1 else None
                ),
            }
        )
