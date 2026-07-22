from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visionsort.core.enums import MatchResult, ParcelState
from visionsort.core.paths import DETAILS_DIR
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


def zone_for_bbox(box: tuple[float, float, float, float], zones: list[dict[str, Any]] | None) -> str | None:
    if not zones:
        return None
    cx, cy = bbox_center(box)
    for zone in zones:
        if zone["x1"] <= cx <= zone["x2"] and zone["y1"] <= cy <= zone["y2"]:
            return zone["zone_id"]
    return None


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
    def __init__(self, camera_id: str, zones: list[dict[str, Any]] | None = None, max_misses: int = 6):
        self.camera_id = camera_id
        self.zones = zones or []
        self.max_misses = max_misses
        self.next_track_id = 1
        self.live_tracks: dict[int, _LiveTrack] = {}

    def update(self, frame_index: int, timestamp: float, observations: list[Observation]) -> tuple[list[TrackObservation], list[Tracklet]]:
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
            dt = max(timestamp - live.last_timestamp, 1e-3)
            track_obs = TrackObservation(
                camera_id=self.camera_id,
                local_track_id=track_id,
                frame_index=frame_index,
                timestamp=timestamp,
                class_name=obs.class_name,
                confidence=obs.confidence,
                bbox=obs.bbox,
                velocity=((cx - prev_cx) / dt, (cy - prev_cy) / dt),
                zone_id=zone_for_bbox(obs.bbox, self.zones),
                appearance_hint=obs.embedding,
                extra=obs.attributes,
            )
            live.last_bbox = obs.bbox
            live.last_timestamp = timestamp
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
                camera_id=self.camera_id,
                local_track_id=track_id,
                frame_index=frame_index,
                timestamp=timestamp,
                class_name=obs.class_name,
                confidence=obs.confidence,
                bbox=obs.bbox,
                velocity=(0.0, 0.0),
                zone_id=zone_for_bbox(obs.bbox, self.zones),
                appearance_hint=obs.embedding,
                extra=obs.attributes,
            )
            self.live_tracks[track_id] = _LiveTrack(
                local_track_id=track_id,
                class_name=obs.class_name,
                last_bbox=obs.bbox,
                last_timestamp=timestamp,
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
        tracklet_id = f"{self.camera_id}-{track_id}-{int(live.history[0].timestamp * 1000)}"
        details_path = DETAILS_DIR / f"{tracklet_id}.jsonl"
        lines = [json.dumps(item.to_json()) for item in live.history]
        details_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        speeds = [math.sqrt(item.velocity[0] ** 2 + item.velocity[1] ** 2) for item in live.history]
        return Tracklet(
            tracklet_id=tracklet_id,
            camera_id=self.camera_id,
            local_track_id=track_id,
            started_at=live.history[0].timestamp,
            ended_at=live.history[-1].timestamp,
            class_name=live.class_name,
            first_bbox=live.history[0].bbox,
            last_bbox=live.history[-1].bbox,
            avg_speed=sum(speeds) / max(len(speeds), 1),
            last_zone_id=live.history[-1].zone_id,
            frame_count=len(live.history),
            observation_path=str(details_path),
            summary_json={
                "start_frame": live.history[0].frame_index,
                "end_frame": live.history[-1].frame_index,
                "duration_s": live.history[-1].timestamp - live.history[0].timestamp,
                "first_bbox": live.history[0].bbox,
                "last_bbox": live.history[-1].bbox,
                "parcel_hint": live.history[-1].extra.get("parcel_hint") or live.history[0].extra.get("parcel_hint"),
                "validated_on_site": False,
            },
        )


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
    def __init__(self, topology_edges: list[dict[str, Any]], source_roles: dict[str, str]):
        self.topology_edges = topology_edges
        self.source_roles = source_roles
        self.parcels: dict[str, GlobalParcel] = {}
        self.tracklet_to_parcel: dict[str, str] = {}

    def process_tracklet(self, tracklet: Tracklet) -> tuple[str, MatchResult, list[str], HandoffCandidate | None]:
        role = self.source_roles.get(tracklet.camera_id, tracklet.camera_id)
        candidates: list[HandoffCandidate] = []
        for parcel in self.parcels.values():
            prev_role = self.source_roles.get(parcel.last_camera_id, parcel.last_camera_id)
            edge = self._find_edge(prev_role, role)
            if edge is None:
                continue
            dt = tracklet.started_at - parcel.last_seen_at
            if dt < edge["min_transit_s"] or dt > edge["max_transit_s"]:
                continue
            size_prev = box_area(tracklet.first_bbox)
            size_curr = box_area(tracklet.last_bbox)
            size_ratio = min(size_prev, size_curr) / max(size_prev, size_curr, 1e-3)
            score = max(0.0, 1.0 - abs(dt - edge["min_transit_s"]) / max(edge["max_transit_s"], 1.0)) + size_ratio
            reasons = [f"transit={dt:.2f}s", f"size_ratio={size_ratio:.2f}", f"edge={prev_role}->{role}"]
            candidates.append(
                HandoffCandidate(
                    from_tracklet_id=parcel.current_tracklet_id,
                    to_tracklet_id=tracklet.tracklet_id,
                    score=score,
                    result=MatchResult.UNMATCHED,
                    reasons=reasons,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)

        if not candidates or candidates[0].score < 1.15:
            parcel_id = str(tracklet.summary_json.get("parcel_hint") or f"parcel-{uuid.uuid4().hex[:10]}")
            self.parcels[parcel_id] = GlobalParcel(
                parcel_id=parcel_id,
                state=ParcelState.ON_CONVEYOR,
                last_camera_id=tracklet.camera_id,
                first_seen_at=tracklet.started_at,
                last_seen_at=tracklet.ended_at,
                current_tracklet_id=tracklet.tracklet_id,
                appearance_signature=None,
            )
            self.tracklet_to_parcel[tracklet.tracklet_id] = parcel_id
            return parcel_id, MatchResult.UNMATCHED, ["nouveau colis"], None

        if len(candidates) > 1 and abs(candidates[0].score - candidates[1].score) < 0.12:
            candidate = candidates[0]
            candidate.result = MatchResult.AMBIGUOUS
            return "", MatchResult.AMBIGUOUS, candidate.reasons + ["scores trop proches"], candidate

        best = candidates[0]
        best.result = MatchResult.MATCHED
        parcel_id = self.tracklet_to_parcel[best.from_tracklet_id]
        parcel = self.parcels[parcel_id]
        parcel.last_camera_id = tracklet.camera_id
        parcel.last_seen_at = tracklet.ended_at
        parcel.current_tracklet_id = tracklet.tracklet_id
        self.tracklet_to_parcel[tracklet.tracklet_id] = parcel_id
        return parcel_id, MatchResult.MATCHED, best.reasons, best

    def _find_edge(self, from_role: str, to_role: str) -> dict[str, Any] | None:
        for edge in self.topology_edges:
            if edge["from_role"] == from_role and edge["to_role"] == to_role:
                return edge
        return None
