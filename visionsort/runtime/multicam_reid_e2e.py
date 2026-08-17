from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from visionsort.core.enums import MatchResult
from visionsort.core.types import Observation, Tracklet
from visionsort.reid.encoder import ParcelReIDEncoder
from visionsort.reid.keyframes import HandoffKeyframeSelector
from visionsort.tracking.engine import GlobalParcelTracker, build_tracker
import visionsort.tracking.integrity as integrity_module


def run_multicam_reid_replay(
    *,
    weights_path: str = "data/models/reid/mobilenet_v3_small-047dcff4.pth",
) -> dict[str, Any]:
    """Deterministic detections-to-GlobalParcel smoke replay using real ByteTrack/ReID."""
    topology = [
        {
            "from_role": "C1",
            "to_role": "C2",
            "min_transit_s": 0.5,
            "max_transit_s": 2.0,
        }
    ]
    zones = {
        "C1": [
            {
                "zone_id": "c1_exit",
                "kind": "exit",
                "x1": 0.55,
                "y1": 0.0,
                "x2": 1.0,
                "y2": 1.0,
            }
        ],
        "C2": [
            {
                "zone_id": "c2_entry",
                "kind": "entry",
                "x1": 0.0,
                "y1": 0.0,
                "x2": 0.45,
                "y2": 1.0,
            }
        ],
    }
    encoder = ParcelReIDEncoder(weights_path)
    original_details = integrity_module.DETAILS_DIR
    with tempfile.TemporaryDirectory(prefix="visionsort-pr3-e2e-") as temporary:
        integrity_module.DETAILS_DIR = Path(temporary)
        try:
            tracklets = [
                _camera_tracklet(
                    role=role,
                    start=start,
                    zones=zones[role],
                    encoder=encoder,
                )
                for role, start in (("C1", 0.0), ("C2", 1.2))
            ]
            hard_negative = _run_hard_negative_replay(
                encoder=encoder,
                topology=topology,
                zones=zones,
            )
        finally:
            integrity_module.DETAILS_DIR = original_details
    global_tracker = GlobalParcelTracker(
        topology,
        {},
        zones_by_role=zones,
        minimum_score=0.48,
        ambiguity_margin=0.08,
        reid_enabled=True,
    )
    ingress = global_tracker.process_tracklet(tracklets[0])
    handoff = global_tracker.process_tracklet(tracklets[1])
    candidate = handoff[3]
    if ingress[1] != MatchResult.NEW_AT_INGRESS:
        raise RuntimeError(f"Décision ingress inattendue: {ingress[1].value}")
    if handoff[1] != MatchResult.MATCHED or ingress[0] != handoff[0]:
        raise RuntimeError(
            f"Handoff E2E invalide: {ingress[0]} / {handoff[0]} / {handoff[1].value}"
        )
    return {
        "status": "PASS",
        "pipeline": [
            "detections",
            "ByteTrack",
            "TrackIntegrityManager",
            "canonical_tracklets",
            "multi_view_ReID",
            "LAPJV",
            "GlobalParcel",
        ],
        "global_parcel_id": handoff[0],
        "decisions": [ingress[1].value, handoff[1].value],
        "local_track_ids": [item.local_track_id for item in tracklets],
        "backend_track_ids": [item.backend_track_ids for item in tracklets],
        "views_per_tracklet": [
            int(item.summary_json["appearance_descriptor"]["view_count"])
            for item in tracklets
        ],
        "reid_similarity": (
            float(candidate.features["reid"])
            if candidate is not None and candidate.features.get("reid") is not None
            else None
        ),
        "handoff_score": float(candidate.score) if candidate is not None else None,
        "model_version": candidate.model_version if candidate is not None else None,
        "same_parcel_variations": {
            "lighting": True,
            "scale": True,
            "perspective": True,
            "different_padding": True,
            "mask_then_bbox": True,
            "decision": handoff[1].value,
        },
        "hard_negative_2x2": hard_negative,
    }


def _camera_tracklet(
    *,
    role: str,
    start: float,
    zones: list[dict[str, Any]],
    encoder: ParcelReIDEncoder,
) -> Tracklet:
    camera_id = f"camera-{role.lower()}"
    tracker = build_tracker(
        tracker_id="bytetrack_cpu",
        session_id="pr3-e2e-session",
        source_id=camera_id,
        camera_id=camera_id,
        camera_role=role,
        zones=zones,
        integrity_config={
            "max_occlusion_seconds": 0.75,
            "max_speed_m_s": 3.0,
            "min_relink_score": 0.55,
            "ambiguity_margin": 0.10,
        },
    )
    selector = HandoffKeyframeSelector(encoder)
    for index in range(6):
        image = np.full((180, 320, 3), 25 if role == "C1" else 42, dtype=np.uint8)
        x1 = (190 + index * 5) if role == "C1" else (20 + index * 5)
        box = (
            (float(x1), 55.0, float(x1 + 78), 132.0)
            if role == "C1"
            else (float(x1), 61.0, float(x1 + 92), 125.0)
        )
        polygon = _draw_textured_parcel(image, box, style="A", camera_role=role)
        observations = [
            Observation(
                class_name="parcel",
                confidence=0.97,
                bbox=box,
                model_id="pr3-e2e-detector",
                mask=polygon if role == "C1" else None,
                attributes={"parcel_hint": "A"},
            )
        ]
        timestamp = start + index * 0.08
        canonical, finalized = tracker.update(
            frame_index=index,
            timestamp_local=timestamp,
            timestamp_global=timestamp,
            image_size=(320, 180),
            observations=observations,
            image=image,
            stream_epoch=0,
        )
        if finalized:
            raise RuntimeError("Tracklet finalisée prématurément pendant le replay.")
        selector.observe(canonical, image)
    finalized = [selector.attach(item) for item in tracker.flush()]
    parcels = [item for item in finalized if item.class_name == "parcel"]
    if len(parcels) != 1:
        raise RuntimeError(f"Une tracklet colis attendue pour {role}, obtenu {len(parcels)}.")
    descriptor = parcels[0].summary_json.get("appearance_descriptor") or {}
    if int(descriptor.get("view_count") or 0) < 3:
        raise RuntimeError(f"Descripteur multi-vues incomplet pour {role}.")
    return parcels[0]


def _draw_textured_parcel(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    *,
    style: str,
    camera_role: str,
) -> list[list[float]]:
    x1, y1, x2, y2 = (int(value) for value in box)
    skew = 7 if camera_role == "C2" else 0
    polygon = np.asarray(
        [[x1 + skew, y1], [x2, y1 + 3], [x2 - skew, y2], [x1, y2 - 2]],
        dtype=np.int32,
    )
    base = (35, 138, 210) if style == "A" else (42, 132, 202)
    if camera_role == "C2":
        base = tuple(int(value * 0.72) for value in base)
    cv2.fillPoly(image, [polygon], base)
    accent = (45, 72, 165) if style == "A" else (52, 78, 158)
    cv2.polylines(image, [polygon], True, accent, 3)
    for offset in range(14, max(15, x2 - x1 - 8), 14):
        shift = 4 if camera_role == "C2" else 0
        cv2.line(
            image,
            (x1 + offset + shift, y1 + 5),
            (x1 + offset - shift, y2 - 5),
            (190, 190, 62) if camera_role == "C2" else (220, 220, 65),
            2,
        )
    label_x = x1 + (10 if style == "A" else max(10, (x2 - x1) // 2))
    cv2.circle(image, (label_x, y1 + 14), 5, (225, 225, 225), thickness=-1)
    return polygon.astype(float).tolist()


def _run_hard_negative_replay(
    *,
    encoder: ParcelReIDEncoder,
    topology: list[dict[str, Any]],
    zones: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    outgoing = _camera_pair_tracklets(
        role="C1",
        start=10.0,
        zones=zones["C1"],
        encoder=encoder,
    )
    incoming = _camera_pair_tracklets(
        role="C2",
        start=11.2,
        zones=zones["C2"],
        encoder=encoder,
    )
    tracker = GlobalParcelTracker(
        topology,
        {},
        zones_by_role=zones,
        minimum_score=0.48,
        ambiguity_margin=0.08,
        reid_enabled=True,
    )
    outgoing_hint_by_id: dict[str, str] = {}
    for tracklet in outgoing:
        tracker.process_tracklet(tracklet)
        outgoing_hint_by_id[tracklet.tracklet_id] = _parcel_hint(tracklet)
    decisions = tracker.process_tracklets(incoming)
    outcomes: list[dict[str, Any]] = []
    for tracklet, decision in zip(incoming, decisions):
        _parcel_id, result, _reasons, candidate = decision
        expected = _parcel_hint(tracklet)
        selected = (
            outgoing_hint_by_id.get(candidate.from_tracklet_id)
            if candidate is not None
            else None
        )
        if result == MatchResult.MATCHED and selected != expected:
            raise RuntimeError(
                f"Faux échange ReID 2x2: attendu={expected}, sélectionné={selected}."
            )
        if result not in {MatchResult.MATCHED, MatchResult.AMBIGUOUS}:
            raise RuntimeError(f"Décision hard-negative inattendue: {result.value}")
        outcomes.append(
            {
                "parcel": expected,
                "decision": result.value,
                "selected": selected,
            }
        )
    return {
        "status": "PASS",
        "parcel_count": len(outcomes),
        "silent_swaps": 0,
        "outcomes": outcomes,
    }


def _camera_pair_tracklets(
    *,
    role: str,
    start: float,
    zones: list[dict[str, Any]],
    encoder: ParcelReIDEncoder,
) -> list[Tracklet]:
    camera_id = f"camera-{role.lower()}"
    tracker = build_tracker(
        tracker_id="bytetrack_cpu",
        session_id="pr3-e2e-session",
        source_id=camera_id,
        camera_id=camera_id,
        camera_role=role,
        zones=zones,
        integrity_config={
            "max_occlusion_seconds": 0.75,
            "max_speed_m_s": 3.0,
            "min_relink_score": 0.55,
            "ambiguity_margin": 0.10,
        },
    )
    selector = HandoffKeyframeSelector(encoder)
    for index in range(6):
        image = np.full((200, 480, 3), 28 if role == "C1" else 45, dtype=np.uint8)
        observations: list[Observation] = []
        for style, base_x in (
            (("A", 295) if role == "C1" else ("A", 20)),
            (("B", 385) if role == "C1" else ("B", 125)),
        ):
            x1 = base_x + index * 4
            width, height = ((76, 68) if role == "C1" else (90, 61))
            box = (float(x1), 64.0, float(x1 + width), float(64 + height))
            polygon = _draw_textured_parcel(
                image,
                box,
                style=style,
                camera_role=role,
            )
            observations.append(
                Observation(
                    class_name="parcel",
                    confidence=0.97,
                    bbox=box,
                    model_id="pr3-e2e-hard-negative",
                    mask=polygon if role == "C1" else None,
                    attributes={"parcel_hint": style},
                )
            )
        timestamp = start + index * 0.08
        canonical, finalized = tracker.update(
            frame_index=index,
            timestamp_local=timestamp,
            timestamp_global=timestamp,
            image_size=(480, 200),
            observations=observations,
            image=image,
            stream_epoch=0,
        )
        if finalized:
            raise RuntimeError("Tracklet pair finalisée prématurément pendant le replay.")
        selector.observe(canonical, image)
    parcels = [
        selector.attach(item)
        for item in tracker.flush()
        if item.class_name == "parcel"
    ]
    if len(parcels) != 2:
        raise RuntimeError(f"Deux tracklets colis attendues pour {role}, obtenu {len(parcels)}.")
    return sorted(parcels, key=_parcel_hint)


def _parcel_hint(tracklet: Tracklet) -> str:
    return str((tracklet.summary_json.get("ground_truth") or {}).get("parcel_hint") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionSort PR3 multicamera ReID E2E")
    parser.add_argument("--weights", default="data/models/reid/mobilenet_v3_small-047dcff4.pth")
    args = parser.parse_args()
    print(json.dumps(run_multicam_reid_replay(weights_path=args.weights), indent=2))


if __name__ == "__main__":
    main()
