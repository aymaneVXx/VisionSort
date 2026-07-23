from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from visionsort.core.config import relative_to_root
from visionsort.core.enums import MatchResult, ParcelState
from visionsort.core.paths import DETAILS_DIR, ROOT_DIR
from visionsort.core.types import GlobalParcel, HandoffCandidate, Observation, TrackObservation, Tracklet


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-6)


def bbox_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return abs((x2 - x1) * (y2 - y1))


def zone_for_bbox(
    box: tuple[float, float, float, float],
    zones: list[dict[str, Any]] | None,
    *,
    image_size: tuple[int, int] | None = None,
) -> str | None:
    if not zones:
        return None
    cx, cy = bbox_center(box)
    iw = ih = None
    if image_size is not None:
        iw, ih = int(image_size[0]), int(image_size[1])
    for zone in zones:
        x1, y1, x2, y2 = float(zone["x1"]), float(zone["y1"]), float(zone["x2"]), float(zone["y2"])
        if iw and ih and max(x1, y1, x2, y2) <= 1.5:
            x1, x2 = x1 * iw, x2 * iw
            y1, y2 = y1 * ih, y2 * ih
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return zone["zone_id"]
    return None


def _observation_extra(obs: Observation, image_size: tuple[int, int] | None) -> dict[str, Any]:
    extra = dict(obs.attributes)
    if image_size:
        extra["_image_w"], extra["_image_h"] = int(image_size[0]), int(image_size[1])
    if obs.keypoints is not None:
        extra["keypoints"] = obs.keypoints
    if obs.mask is not None:
        extra["mask"] = obs.mask
    return extra


@dataclass
class _LiveTrack:
    local_track_id: int
    class_name: str
    last_bbox: tuple[float, float, float, float]
    last_timestamp: float
    last_frame_index: int
    history: list[TrackObservation] = field(default_factory=list)
    misses: int = 0


class GreedyIOUTracker:
    def __init__(
        self,
        *,
        session_id: str,
        source_id: str,
        camera_id: str,
        camera_role: str,
        tracker_id: str,
        zones: list[dict[str, Any]] | None = None,
        max_misses: int = 6,
    ):
        self.session_id = session_id
        self.source_id = source_id
        self.camera_id = camera_id
        self.camera_role = camera_role
        self.tracker_id = tracker_id
        self.zones = zones or []
        self.max_misses = max_misses
        self.next_track_id = 1
        self.live_tracks: dict[int, _LiveTrack] = {}

    def update(
        self,
        *,
        frame_index: int,
        timestamp_local: float,
        timestamp_global: float,
        image_size: tuple[int, int] | None = None,
        observations: list[Observation],
        image: np.ndarray | None = None,
    ) -> tuple[list[TrackObservation], list[Tracklet]]:
        _ = image
        produced: list[TrackObservation] = []
        finalized: list[Tracklet] = []
        unmatched_track_ids = set(self.live_tracks)
        matched_obs: set[int] = set()
        candidates: list[tuple[float, int, int]] = []

        for track_id, track in self.live_tracks.items():
            for obs_index, obs in enumerate(observations):
                if obs.class_name != track.class_name:
                    continue
                candidates.append((bbox_iou(track.last_bbox, obs.bbox), track_id, obs_index))

        for score, track_id, obs_index in sorted(candidates, reverse=True):
            if score < 0.10 or obs_index in matched_obs or track_id not in unmatched_track_ids:
                continue
            obs = observations[obs_index]
            live = self.live_tracks[track_id]
            prev_cx, prev_cy = bbox_center(live.last_bbox)
            cx, cy = bbox_center(obs.bbox)
            dt = max(timestamp_global - live.last_timestamp, 1e-3)
            track_obs = TrackObservation(
                session_id=self.session_id,
                source_id=self.source_id,
                camera_id=self.camera_id,
                camera_role=self.camera_role,
                local_track_id=track_id,
                frame_index=frame_index,
                timestamp_local=timestamp_local,
                timestamp_global=timestamp_global,
                class_name=obs.class_name,
                confidence=obs.confidence,
                bbox=obs.bbox,
                velocity=((cx - prev_cx) / dt, (cy - prev_cy) / dt),
                zone_id=zone_for_bbox(obs.bbox, self.zones, image_size=image_size),
                appearance_hint=obs.embedding,
                model_id=obs.model_id,
                tracker_id=self.tracker_id,
                extra=_observation_extra(obs, image_size),
            )
            live.last_bbox = obs.bbox
            live.last_timestamp = timestamp_global
            live.last_frame_index = frame_index
            live.history.append(track_obs)
            live.misses = 0
            matched_obs.add(obs_index)
            unmatched_track_ids.remove(track_id)
            produced.append(track_obs)

        for obs_index, obs in enumerate(observations):
            if obs_index in matched_obs:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            track_obs = TrackObservation(
                session_id=self.session_id,
                source_id=self.source_id,
                camera_id=self.camera_id,
                camera_role=self.camera_role,
                local_track_id=track_id,
                frame_index=frame_index,
                timestamp_local=timestamp_local,
                timestamp_global=timestamp_global,
                class_name=obs.class_name,
                confidence=obs.confidence,
                bbox=obs.bbox,
                velocity=(0.0, 0.0),
                zone_id=zone_for_bbox(obs.bbox, self.zones, image_size=image_size),
                appearance_hint=obs.embedding,
                model_id=obs.model_id,
                tracker_id=self.tracker_id,
                extra=_observation_extra(obs, image_size),
            )
            self.live_tracks[track_id] = _LiveTrack(
                local_track_id=track_id,
                class_name=obs.class_name,
                last_bbox=obs.bbox,
                last_timestamp=timestamp_global,
                last_frame_index=frame_index,
                history=[track_obs],
            )
            produced.append(track_obs)

        expired: list[int] = []
        for track_id in list(unmatched_track_ids):
            live = self.live_tracks[track_id]
            live.misses += 1
            if live.misses >= self.max_misses:
                expired.append(track_id)
        for track_id in expired:
            finalized.append(self._finalize(track_id))
        return produced, finalized

    def flush(self) -> list[Tracklet]:
        finalized: list[Tracklet] = []
        for track_id in list(self.live_tracks):
            finalized.append(self._finalize(track_id))
        return finalized

    def _finalize(self, track_id: int) -> Tracklet:
        live = self.live_tracks.pop(track_id)
        DETAILS_DIR.mkdir(parents=True, exist_ok=True)
        tracklet_id = f"{self.session_id}-{self.camera_id}-{track_id}-{int(live.history[0].timestamp_global * 1000)}"
        details_path = DETAILS_DIR / f"{tracklet_id}.jsonl"
        lines = [json.dumps(item.to_json()) for item in live.history]
        details_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        speeds = [math.sqrt(item.velocity[0] ** 2 + item.velocity[1] ** 2) for item in live.history]
        boxes = [item.bbox for item in live.history]
        avg_bbox = tuple(sum(box[index] for box in boxes) / len(boxes) for index in range(4))
        avg_dimensions = (
            sum(abs(box[2] - box[0]) for box in boxes) / len(boxes),
            sum(abs(box[3] - box[1]) for box in boxes) / len(boxes),
        )
        avg_velocity = (
            sum(item.velocity[0] for item in live.history) / len(live.history),
            sum(item.velocity[1] for item in live.history) / len(live.history),
        )
        embeddings = [item.appearance_hint for item in live.history if item.appearance_hint]
        avg_embedding = None
        if embeddings:
            width = min(len(item) for item in embeddings)
            avg_embedding = [sum(float(item[index]) for item in embeddings) / len(embeddings) for index in range(width)]
        visited_zones = list(dict.fromkeys(item.zone_id for item in live.history if item.zone_id))
        ground_truth_hint = live.history[-1].extra.get("parcel_hint") or live.history[0].extra.get("parcel_hint")
        return Tracklet(
            tracklet_id=tracklet_id,
            session_id=self.session_id,
            source_id=self.source_id,
            camera_id=self.camera_id,
            camera_role=self.camera_role,
            local_track_id=track_id,
            started_at_local=live.history[0].timestamp_local,
            ended_at_local=live.history[-1].timestamp_local,
            started_at_global=live.history[0].timestamp_global,
            ended_at_global=live.history[-1].timestamp_global,
            class_name=live.class_name,
            first_bbox=live.history[0].bbox,
            last_bbox=live.history[-1].bbox,
            avg_speed=sum(speeds) / max(len(speeds), 1),
            last_zone_id=live.history[-1].zone_id,
            frame_count=len(live.history),
            observation_path=relative_to_root(details_path),
            summary_json={
                "start_frame": live.history[0].frame_index,
                "end_frame": live.history[-1].frame_index,
                "duration_s": live.history[-1].timestamp_global - live.history[0].timestamp_global,
                "first_bbox": live.history[0].bbox,
                "last_bbox": live.history[-1].bbox,
                "avg_bbox": avg_bbox,
                "avg_dimensions": avg_dimensions,
                "avg_velocity": avg_velocity,
                "first_zone_id": live.history[0].zone_id,
                "last_zone_id": live.history[-1].zone_id,
                "visited_zones": visited_zones,
                "appearance_embedding": avg_embedding,
                "ground_truth": {"parcel_hint": ground_truth_hint} if ground_truth_hint else {},
                "model_id": live.history[-1].model_id,
                "tracker_id": self.tracker_id,
                "validated_on_site": False,
            },
            model_id=live.history[-1].model_id,
            tracker_id=self.tracker_id,
        )


class UltralyticsLocalTracker(GreedyIOUTracker):
    """Adapter around the maintained Ultralytics tracker implementations.

    Track state is native Ultralytics state. VisionSort only retains observation
    history so it can persist finalized tracklets with runtime provenance.
    """

    tracker_config_name = ""
    tracker_class_name = ""

    def __init__(
        self,
        *,
        session_id: str,
        source_id: str,
        camera_id: str,
        camera_role: str,
        tracker_id: str,
        zones: list[dict[str, Any]] | None = None,
        max_misses: int = 30,
    ):
        super().__init__(
            session_id=session_id,
            source_id=source_id,
            camera_id=camera_id,
            camera_role=camera_role,
            tracker_id=tracker_id,
            zones=zones,
            max_misses=max_misses,
        )
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT_DIR / "data" / "ultralytics"))
        try:
            import ultralytics
            from ultralytics.engine.results import Boxes
            from ultralytics.utils import IterableSimpleNamespace, YAML

            if self.tracker_class_name == "BYTETracker":
                from ultralytics.trackers.byte_tracker import BYTETracker as NativeTracker
            else:
                from ultralytics.trackers.bot_sort import BOTSORT as NativeTracker
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError(
                f"{self.tracker_class_name} exige Ultralytics et sa dépendance `lap`."
            ) from exc
        config_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / self.tracker_config_name
        if not config_path.exists():
            raise RuntimeError(f"Configuration Ultralytics introuvable: {config_path}")
        args = IterableSimpleNamespace(**YAML.load(config_path))
        args.device = "cpu"
        self._boxes_class = Boxes
        self.native_tracker = NativeTracker(args=args)
        self.class_ids: dict[str, int] = {}

    def update(
        self,
        *,
        frame_index: int,
        timestamp_local: float,
        timestamp_global: float,
        image_size: tuple[int, int] | None = None,
        observations: list[Observation],
        image: np.ndarray | None = None,
    ) -> tuple[list[TrackObservation], list[Tracklet]]:
        produced: list[TrackObservation] = []
        finalized: list[Tracklet] = []
        width, height = image_size or (
            (int(image.shape[1]), int(image.shape[0])) if image is not None else (1, 1)
        )
        rows = np.asarray(
            [
                [
                    *map(float, obs.bbox),
                    float(obs.confidence),
                    float(
                        self.class_ids.setdefault(
                            obs.class_name, len(self.class_ids)
                        )
                    ),
                ]
                for obs in observations
            ],
            dtype=np.float32,
        )
        if not len(rows):
            rows = np.empty((0, 6), dtype=np.float32)
        boxes = self._boxes_class(rows, orig_shape=(height, width))
        if image is None:
            image = np.zeros((height, width, 3), dtype=np.uint8)
        tracked = self.native_tracker.update(boxes, img=image)
        active_track_ids: set[int] = set()
        for result in tracked:
            track_id = int(result[4])
            obs_index = int(result[7])
            if obs_index < 0 or obs_index >= len(observations):
                continue
            obs = observations[obs_index]
            active_track_ids.add(track_id)
            live = self.live_tracks.get(track_id)
            previous_bbox = live.last_bbox if live else obs.bbox
            previous_timestamp = live.last_timestamp if live else timestamp_global
            prev_cx, prev_cy = bbox_center(previous_bbox)
            current_bbox = tuple(float(value) for value in result[:4])
            cx, cy = bbox_center(current_bbox)
            dt = max(timestamp_global - previous_timestamp, 1e-3)
            track_obs = TrackObservation(
                session_id=self.session_id,
                source_id=self.source_id,
                camera_id=self.camera_id,
                camera_role=self.camera_role,
                local_track_id=track_id,
                frame_index=frame_index,
                timestamp_local=timestamp_local,
                timestamp_global=timestamp_global,
                class_name=obs.class_name,
                confidence=obs.confidence,
                bbox=current_bbox,
                velocity=((cx - prev_cx) / dt, (cy - prev_cy) / dt),
                zone_id=zone_for_bbox(current_bbox, self.zones, image_size=image_size),
                appearance_hint=obs.embedding,
                model_id=obs.model_id,
                tracker_id=self.tracker_id,
                extra={
                    **_observation_extra(obs, image_size),
                    "track_identity": [self.camera_id, track_id],
                    "tracker_backend": type(self.native_tracker).__name__,
                },
            )
            if live is None:
                self.live_tracks[track_id] = _LiveTrack(
                    local_track_id=track_id,
                    class_name=obs.class_name,
                    last_bbox=current_bbox,
                    last_timestamp=timestamp_global,
                    last_frame_index=frame_index,
                    history=[track_obs],
                )
            else:
                live.last_bbox = current_bbox
                live.last_timestamp = timestamp_global
                live.last_frame_index = frame_index
                live.history.append(track_obs)
                live.misses = 0
            produced.append(track_obs)

        for track_id in list(self.live_tracks):
            if track_id in active_track_ids:
                continue
            self.live_tracks[track_id].misses += 1
            if self.live_tracks[track_id].misses >= self.max_misses:
                finalized.append(self._finalize(track_id))
        return produced, finalized


class ByteTrackTracker(UltralyticsLocalTracker):
    tracker_config_name = "bytetrack.yaml"
    tracker_class_name = "BYTETracker"


class BoTSORTTracker(UltralyticsLocalTracker):
    tracker_config_name = "botsort.yaml"
    tracker_class_name = "BOTSORT"


def build_tracker(
    *,
    tracker_id: str,
    session_id: str,
    source_id: str,
    camera_id: str,
    camera_role: str,
    zones: list[dict[str, Any]] | None,
) -> GreedyIOUTracker | UltralyticsLocalTracker:
    if tracker_id == "greedy_iou":
        return GreedyIOUTracker(
            session_id=session_id,
            source_id=source_id,
            camera_id=camera_id,
            camera_role=camera_role,
            tracker_id=tracker_id,
            zones=zones,
        )
    if tracker_id == "bytetrack_cpu":
        return ByteTrackTracker(
            session_id=session_id,
            source_id=source_id,
            camera_id=camera_id,
            camera_role=camera_role,
            tracker_id=tracker_id,
            zones=zones,
        )
    if tracker_id == "botsort_cpu":
        return BoTSORTTracker(
            session_id=session_id,
            source_id=source_id,
            camera_id=camera_id,
            camera_role=camera_role,
            tracker_id=tracker_id,
            zones=zones,
        )
    raise RuntimeError(f"Tracker non supporté: {tracker_id}")


class TrackletBuilder:
    def __init__(self):
        self.pending_by_camera: dict[str, list[Tracklet]] = {}

    def append(self, tracklet: Tracklet) -> None:
        self.pending_by_camera.setdefault(tracklet.camera_id, []).append(tracklet)

    def pop_all(self) -> list[Tracklet]:
        output: list[Tracklet] = []
        for camera_id in list(self.pending_by_camera):
            output.extend(self.pending_by_camera.pop(camera_id))
        return output


class GlobalParcelTracker:
    def __init__(
        self,
        topology_edges: list[dict[str, Any]],
        source_roles: dict[str, str],
        *,
        minimum_score: float = 0.48,
        ambiguity_margin: float = 0.08,
    ):
        self.topology_edges = topology_edges
        self.source_roles = source_roles
        self.minimum_score = minimum_score
        self.ambiguity_margin = ambiguity_margin
        self.parcels: dict[str, GlobalParcel] = {}
        self.tracklet_to_parcel: dict[str, str] = {}
        self.tracklets: dict[str, Tracklet] = {}
        self.last_candidate_sets: dict[str, list[HandoffCandidate]] = {}

    def process_tracklet(self, tracklet: Tracklet) -> tuple[str, MatchResult, list[str], HandoffCandidate | None]:
        return self.process_tracklets([tracklet])[0]

    def process_tracklets(
        self, incoming_tracklets: list[Tracklet]
    ) -> list[tuple[str, MatchResult, list[str], HandoffCandidate | None]]:
        """Associate a batch with a conservative global one-to-one assignment."""
        results: list[tuple[str, MatchResult, list[str], HandoffCandidate | None] | None] = [
            None
        ] * len(incoming_tracklets)
        candidates_by_incoming: dict[int, list[HandoffCandidate]] = {}
        candidates_by_outgoing: dict[str, list[tuple[int, HandoffCandidate]]] = {}

        for incoming_index, incoming in enumerate(incoming_tracklets):
            candidates: list[HandoffCandidate] = []
            for parcel in self.parcels.values():
                outgoing = self.tracklets.get(parcel.current_tracklet_id)
                if outgoing is None or outgoing.session_id != incoming.session_id:
                    continue
                candidate = self._score_candidate(outgoing, incoming)
                if candidate is None:
                    continue
                candidates.append(candidate)
                candidates_by_outgoing.setdefault(outgoing.tracklet_id, []).append(
                    (incoming_index, candidate)
                )
            candidates.sort(key=lambda item: item.score, reverse=True)
            candidates_by_incoming[incoming_index] = candidates
            self.last_candidate_sets[incoming.tracklet_id] = list(candidates)

        ambiguous: set[int] = set()
        for incoming_index, candidates in candidates_by_incoming.items():
            viable = [item for item in candidates if item.score >= self.minimum_score]
            if len(viable) > 1 and viable[0].score - viable[1].score < self.ambiguity_margin:
                ambiguous.add(incoming_index)
        for outgoing_candidates in candidates_by_outgoing.values():
            viable = sorted(
                (
                    (incoming_index, candidate)
                    for incoming_index, candidate in outgoing_candidates
                    if candidate.score >= self.minimum_score
                ),
                key=lambda item: item[1].score,
                reverse=True,
            )
            if len(viable) > 1 and viable[0][1].score - viable[1][1].score < self.ambiguity_margin:
                ambiguous.update((viable[0][0], viable[1][0]))

        proposals = sorted(
            (
                (candidate.score, incoming_index, candidate)
                for incoming_index, candidates in candidates_by_incoming.items()
                for candidate in candidates
                if candidate.score >= self.minimum_score and incoming_index not in ambiguous
            ),
            reverse=True,
            key=lambda item: item[0],
        )
        assigned_incoming: set[int] = set()
        assigned_outgoing: set[str] = set()
        for _, incoming_index, candidate in proposals:
            if incoming_index in assigned_incoming or candidate.from_tracklet_id in assigned_outgoing:
                continue
            incoming = incoming_tracklets[incoming_index]
            parcel_id = self.tracklet_to_parcel[candidate.from_tracklet_id]
            parcel = self.parcels[parcel_id]
            candidate.result = MatchResult.MATCHED
            parcel.last_camera_id = incoming.camera_id
            parcel.last_seen_at = incoming.ended_at_global
            parcel.current_tracklet_id = incoming.tracklet_id
            parcel.appearance_signature = self._appearance(incoming) or parcel.appearance_signature
            self.tracklet_to_parcel[incoming.tracklet_id] = parcel_id
            self.tracklets[incoming.tracklet_id] = incoming
            results[incoming_index] = (
                parcel_id,
                MatchResult.MATCHED,
                candidate.reasons,
                candidate,
            )
            assigned_incoming.add(incoming_index)
            assigned_outgoing.add(candidate.from_tracklet_id)

        for incoming_index, incoming in enumerate(incoming_tracklets):
            if results[incoming_index] is not None:
                continue
            candidates = candidates_by_incoming.get(incoming_index, [])
            viable = [item for item in candidates if item.score >= self.minimum_score]
            if incoming_index in ambiguous or viable and any(
                item.from_tracklet_id in assigned_outgoing for item in viable
            ):
                best = viable[0] if viable else candidates[0]
                best.result = MatchResult.AMBIGUOUS
                reasons = best.reasons + ["association concurrente ou scores trop proches"]
                results[incoming_index] = ("", MatchResult.AMBIGUOUS, reasons, best)
                self.tracklets[incoming.tracklet_id] = incoming
                continue
            parcel_id = self._register_new_parcel(incoming)
            reason = "aucun candidat compatible" if candidates else "nouveau colis en entrée"
            results[incoming_index] = (
                parcel_id,
                MatchResult.UNMATCHED,
                [reason],
                candidates[0] if candidates else None,
            )

        return [item for item in results if item is not None]

    def resolve_ambiguous(
        self, incoming_tracklet_id: str, outgoing_tracklet_id: str
    ) -> str:
        incoming = self.tracklets.get(incoming_tracklet_id)
        outgoing = self.tracklets.get(outgoing_tracklet_id)
        parcel_id = self.tracklet_to_parcel.get(outgoing_tracklet_id)
        if incoming is None or outgoing is None or parcel_id is None:
            raise RuntimeError("Tracklets de l'hypothèse indisponibles.")
        parcel = self.parcels.get(parcel_id)
        if parcel is None or parcel.current_tracklet_id != outgoing_tracklet_id:
            raise RuntimeError(
                "Le candidat sortant a déjà été consommé par un autre handoff."
            )
        parcel.last_camera_id = incoming.camera_id
        parcel.last_seen_at = incoming.ended_at_global
        parcel.current_tracklet_id = incoming.tracklet_id
        parcel.appearance_signature = (
            self._appearance(incoming) or parcel.appearance_signature
        )
        self.tracklet_to_parcel[incoming.tracklet_id] = parcel_id
        return parcel_id

    def continuation_evidence(
        self, outgoing_tracklet_id: str, later: Tracklet
    ) -> float | None:
        outgoing = self.tracklets.get(outgoing_tracklet_id)
        if outgoing is None:
            return None
        outgoing_dimensions = self._dimensions(outgoing)
        later_dimensions = self._dimensions(later)
        dimension_score = sum(
            min(left, right) / max(left, right, 1e-6)
            for left, right in zip(outgoing_dimensions, later_dimensions)
        ) / 2.0
        outgoing_appearance = self._appearance(outgoing)
        later_appearance = self._appearance(later)
        if not outgoing_appearance or not later_appearance:
            return dimension_score
        length = min(len(outgoing_appearance), len(later_appearance))
        left = np.asarray(outgoing_appearance[:length], dtype=np.float32)
        right = np.asarray(later_appearance[:length], dtype=np.float32)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1e-9:
            return dimension_score
        appearance_score = (float(np.dot(left, right) / denominator) + 1.0) / 2.0
        return 0.4 * dimension_score + 0.6 * appearance_score

    def _register_new_parcel(self, tracklet: Tracklet) -> str:
        parcel_id = f"parcel-{uuid.uuid4().hex[:10]}"
        appearance = self._appearance(tracklet)
        self.parcels[parcel_id] = GlobalParcel(
            parcel_id=parcel_id,
            state=ParcelState.ON_CONVEYOR,
            last_camera_id=tracklet.camera_id,
            first_seen_at=tracklet.started_at_global,
            last_seen_at=tracklet.ended_at_global,
            current_tracklet_id=tracklet.tracklet_id,
            appearance_signature=appearance,
        )
        self.tracklet_to_parcel[tracklet.tracklet_id] = parcel_id
        self.tracklets[tracklet.tracklet_id] = tracklet
        return parcel_id

    def _score_candidate(
        self, outgoing: Tracklet, incoming: Tracklet
    ) -> HandoffCandidate | None:
        previous_role = outgoing.camera_role or self.source_roles.get(
            outgoing.camera_id, outgoing.camera_id
        )
        incoming_role = incoming.camera_role or self.source_roles.get(
            incoming.camera_id, incoming.camera_id
        )
        edge = self._find_edge(previous_role, incoming_role)
        if edge is None:
            return None
        dt = incoming.started_at_global - outgoing.ended_at_global
        minimum = float(edge["min_transit_s"])
        maximum = float(edge["max_transit_s"])
        if dt < minimum or dt > maximum:
            return None
        midpoint = (minimum + maximum) / 2.0
        half_window = max((maximum - minimum) / 2.0, 1e-6)
        temporal_score = max(0.0, 1.0 - abs(dt - midpoint) / half_window)

        outgoing_dimensions = self._dimensions(outgoing)
        incoming_dimensions = self._dimensions(incoming)
        width_ratio = min(outgoing_dimensions[0], incoming_dimensions[0]) / max(
            outgoing_dimensions[0], incoming_dimensions[0], 1e-6
        )
        height_ratio = min(outgoing_dimensions[1], incoming_dimensions[1]) / max(
            outgoing_dimensions[1], incoming_dimensions[1], 1e-6
        )
        dimension_score = (width_ratio + height_ratio) / 2.0

        outgoing_zone = self._summary(outgoing).get("last_zone_id") or outgoing.last_zone_id
        incoming_zone = self._summary(incoming).get("first_zone_id")
        zone_score = 0.5
        if outgoing_zone:
            zone_score += 0.25 if "exit" in str(outgoing_zone).lower() else -0.15
        if incoming_zone:
            zone_score += 0.25 if "entry" in str(incoming_zone).lower() else -0.15
        zone_score = min(1.0, max(0.0, zone_score))

        outgoing_velocity = self._velocity(outgoing)
        incoming_velocity = self._velocity(incoming)
        outgoing_speed = math.hypot(*outgoing_velocity)
        incoming_speed = math.hypot(*incoming_velocity)
        speed_score = min(outgoing_speed, incoming_speed) / max(
            outgoing_speed, incoming_speed, 1e-6
        )
        trajectory_score = 0.5
        if outgoing_speed > 1e-6 and incoming_speed > 1e-6:
            cosine = (
                outgoing_velocity[0] * incoming_velocity[0]
                + outgoing_velocity[1] * incoming_velocity[1]
            ) / (outgoing_speed * incoming_speed)
            trajectory_score = (max(-1.0, min(1.0, cosine)) + 1.0) / 2.0

        outgoing_appearance = self._appearance(outgoing)
        incoming_appearance = self._appearance(incoming)
        appearance_score = None
        if outgoing_appearance and incoming_appearance:
            length = min(len(outgoing_appearance), len(incoming_appearance))
            a = np.asarray(outgoing_appearance[:length], dtype=np.float32)
            b = np.asarray(incoming_appearance[:length], dtype=np.float32)
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom > 1e-9:
                appearance_score = (float(np.dot(a, b) / denom) + 1.0) / 2.0

        weights = {
            "temporal": 0.25,
            "dimensions": 0.25,
            "zone": 0.20,
            "speed": 0.15,
            "trajectory": 0.05,
        }
        components = {
            "temporal": temporal_score,
            "dimensions": dimension_score,
            "zone": zone_score,
            "speed": speed_score,
            "trajectory": trajectory_score,
        }
        if appearance_score is not None:
            weights["appearance"] = 0.20
            components["appearance"] = appearance_score
        total_weight = sum(weights.values())
        score = sum(components[name] * weight for name, weight in weights.items()) / total_weight
        reasons = [
            f"edge={previous_role}->{incoming_role}",
            f"transit={dt:.3f}s",
            f"dimensions={dimension_score:.3f}",
            f"zones={zone_score:.3f}",
            f"speed={speed_score:.3f}",
            f"trajectory={trajectory_score:.3f}",
        ]
        if appearance_score is not None:
            reasons.append(f"appearance={appearance_score:.3f}")
        return HandoffCandidate(
            from_tracklet_id=outgoing.tracklet_id,
            to_tracklet_id=incoming.tracklet_id,
            score=float(score),
            result=MatchResult.UNMATCHED,
            reasons=reasons,
        )

    @staticmethod
    def _summary(tracklet: Tracklet) -> dict[str, Any]:
        value = getattr(tracklet, "summary_json", {}) or {}
        return value if isinstance(value, dict) else {}

    def _dimensions(self, tracklet: Tracklet) -> tuple[float, float]:
        dimensions = self._summary(tracklet).get("avg_dimensions")
        if dimensions and len(dimensions) >= 2:
            return float(dimensions[0]), float(dimensions[1])
        box = getattr(tracklet, "last_bbox", None) or getattr(tracklet, "first_bbox")
        return abs(float(box[2]) - float(box[0])), abs(float(box[3]) - float(box[1]))

    def _velocity(self, tracklet: Tracklet) -> tuple[float, float]:
        velocity = self._summary(tracklet).get("avg_velocity")
        if velocity and len(velocity) >= 2:
            return float(velocity[0]), float(velocity[1])
        speed = float(getattr(tracklet, "avg_speed", 0.0) or 0.0)
        return speed, 0.0

    def _appearance(self, tracklet: Tracklet) -> list[float] | None:
        appearance = self._summary(tracklet).get("appearance_embedding")
        return [float(value) for value in appearance] if appearance else None

    def _find_edge(self, from_role: str, to_role: str) -> dict[str, Any] | None:
        for edge in self.topology_edges:
            if edge["from_role"] == from_role and edge["to_role"] == to_role:
                return edge
        return None
