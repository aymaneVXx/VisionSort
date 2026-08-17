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
        image = np.full((180, 320, 3), 25, dtype=np.uint8)
        x1 = (190 + index * 5) if role == "C1" else (25 + index * 5)
        box = (float(x1), 55.0, float(x1 + 78), 132.0)
        _draw_textured_parcel(image, box)
        observations = [
            Observation(
                class_name="parcel",
                confidence=0.97,
                bbox=box,
                model_id="pr3-e2e-detector",
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
    image: np.ndarray, box: tuple[float, float, float, float]
) -> None:
    x1, y1, x2, y2 = (int(value) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (30, 145, 220), thickness=-1)
    cv2.rectangle(image, (x1 + 8, y1 + 8), (x2 - 8, y2 - 8), (40, 70, 170), 3)
    for offset in range(14, max(15, x2 - x1 - 8), 14):
        cv2.line(image, (x1 + offset, y1 + 4), (x1 + offset, y2 - 4), (220, 220, 65), 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionSort PR3 multicamera ReID E2E")
    parser.add_argument("--weights", default="data/models/reid/mobilenet_v3_small-047dcff4.pth")
    args = parser.parse_args()
    print(json.dumps(run_multicam_reid_replay(weights_path=args.weights), indent=2))


if __name__ == "__main__":
    main()
