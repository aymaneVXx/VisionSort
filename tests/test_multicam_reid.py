from __future__ import annotations

import numpy as np

from visionsort.core.enums import MatchResult
from visionsort.core.types import TrackObservation, Tracklet
from visionsort.reid.encoder import ParcelReIDEncoder, ProjectionHead, l2_normalize
from visionsort.reid.handoff import HandoffCandidateGenerator
from visionsort.reid.keyframes import HandoffKeyframeSelector
from visionsort.runtime.multicam_reid_e2e import run_multicam_reid_replay
from visionsort.tracking.engine import GlobalParcelTracker


TOPOLOGY = [
    {
        "from_role": "C1",
        "to_role": "C2",
        "min_transit_s": 0.5,
        "max_transit_s": 3.0,
    },
    {
        "from_role": "C2",
        "to_role": "C3",
        "min_transit_s": 0.4,
        "max_transit_s": 3.0,
    },
]
ZONES = {
    "C1": [{"zone_id": "c1_exit", "kind": "exit"}],
    "C2": [
        {"zone_id": "c2_entry", "kind": "entry"},
        {"zone_id": "c2_exit", "kind": "exit"},
    ],
    "C3": [{"zone_id": "c3_entry", "kind": "entry"}],
}


class _FakeEncoder:
    model_version = "fake-generic-object-encoder-v1"

    def encode(self, crops):
        values = []
        for crop in crops:
            mean = np.asarray(crop, dtype=np.float32).mean(axis=(0, 1)) / 255.0
            values.append([mean[0], mean[1], mean[2], float(np.std(crop) / 255.0)])
        return l2_normalize(np.asarray(values, dtype=np.float32))


def _descriptor(
    vector: list[float], *, view_count: int = 3, min_views: int = 3
) -> dict:
    normalized = l2_normalize(np.asarray(vector, dtype=np.float32)).tolist()
    return {
        "embeddings": [normalized for _ in range(view_count)],
        "aggregate_embedding": normalized,
        "view_count": view_count,
        "view_qualities": [0.9 for _ in range(view_count)],
        "model_version": "test-backbone-v1",
        "used_mask": [False for _ in range(view_count)],
        "min_views": min_views,
        "average_view_quality": 0.9,
        "descriptor_quality": "RELIABLE" if view_count >= min_views else "LOW",
    }


def _tracklet(
    name: str,
    role: str,
    *,
    started: float,
    ended: float,
    vector: list[float] | None = None,
    width: float = 30.0,
    first_zone: str | None = None,
    last_zone: str | None = None,
    session: str = "session-pr3",
    integrity: str = "STABLE",
) -> Tracklet:
    summary = {
        "avg_dimensions": [width, 20.0],
        "avg_velocity": [2.0, 0.0],
        "first_zone_id": first_zone or f"{role.lower()}_entry",
        "last_zone_id": last_zone or f"{role.lower()}_exit",
        "merge_group_ids": [],
        "split_count": 0,
        "integrity_status": integrity,
    }
    if vector is not None:
        summary["appearance_descriptor"] = _descriptor(vector)
    return Tracklet(
        tracklet_id=name,
        session_id=session,
        source_id=f"source-{role}",
        camera_id=f"camera-{role}",
        camera_role=role,
        local_track_id=1,
        started_at_local=started,
        ended_at_local=ended,
        started_at_global=started,
        ended_at_global=ended,
        class_name="parcel",
        first_bbox=(0.0, 0.0, width, 20.0),
        last_bbox=(1.0, 0.0, width + 1.0, 20.0),
        avg_speed=2.0,
        last_zone_id=last_zone or f"{role.lower()}_exit",
        frame_count=8,
        observation_path="details.jsonl",
        summary_json=summary,
        integrity_status=integrity,
    )


def test_keyframe_selector_builds_multi_view_descriptor_without_mask():
    selector = HandoffKeyframeSelector(_FakeEncoder(), max_views=5, min_views=3)
    for frame_index in range(7):
        image = np.full((160, 240, 3), 25, dtype=np.uint8)
        image[45:125, 70:170] = (40 + frame_index, 120, 210)
        track = TrackObservation(
            session_id="s",
            source_id="c1",
            camera_id="c1",
            camera_role="C1",
            local_track_id=1,
            frame_index=frame_index,
            timestamp_local=frame_index * 0.1,
            timestamp_global=frame_index * 0.1,
            class_name="parcel",
            confidence=0.96,
            bbox=(70.0, 45.0, 170.0, 125.0),
            velocity=(1.0, 0.0),
            zone_id="c1_exit",
            identity_status="STABLE",
            extra={"_image_w": 240, "_image_h": 160},
        )
        selector.observe([track], image)
    result = selector.attach(
        _tracklet("views", "C1", started=0.0, ended=0.6)
    )
    descriptor = result.summary_json["appearance_descriptor"]

    assert 3 <= descriptor["view_count"] <= 5
    assert len(descriptor["embeddings"]) == descriptor["view_count"]
    assert all(value is False for value in descriptor["used_mask"])
    assert descriptor["descriptor_quality"] == "RELIABLE"
    assert descriptor["average_view_quality"] > 0.0
    np.testing.assert_allclose(
        np.linalg.norm(descriptor["aggregate_embedding"]), 1.0, atol=1e-5
    )


def test_keyframe_selector_uses_mask_when_available():
    selector = HandoffKeyframeSelector(_FakeEncoder(), max_views=3, min_views=1)
    image = np.full((120, 160, 3), 30, dtype=np.uint8)
    image[30:100, 40:130] = (50, 150, 220)
    track = TrackObservation(
        session_id="s",
        source_id="c1",
        camera_id="c1",
        camera_role="C1",
        local_track_id=1,
        frame_index=0,
        timestamp_local=0.0,
        timestamp_global=0.0,
        class_name="parcel",
        confidence=0.98,
        bbox=(40.0, 30.0, 130.0, 100.0),
        velocity=(1.0, 0.0),
        extra={"mask": [[40, 30], [130, 30], [130, 100], [40, 100]]},
    )
    selector.observe([track], image)
    result = selector.attach(_tracklet("masked", "C1", started=0.0, ended=0.1))
    assert result.summary_json["appearance_descriptor"]["used_mask"] == [True]


def test_reid_letterbox_preserves_rectangular_parcel_ratio():
    crop = np.zeros((40, 160, 3), dtype=np.uint8)
    letterboxed = ParcelReIDEncoder._letterbox(crop, size=224)
    content = np.any(letterboxed != 114, axis=2)
    rows, columns = np.where(content)

    content_height = int(rows.max() - rows.min() + 1)
    content_width = int(columns.max() - columns.min() + 1)
    assert letterboxed.shape == (224, 224, 3)
    assert content_width / content_height == 4.0


def test_hard_gates_reject_topology_transit_and_wrong_zones_before_reid():
    generator = HandoffCandidateGenerator(TOPOLOGY, {}, ZONES)
    outgoing = _tracklet(
        "out", "C1", started=0.0, ended=1.0, last_zone="c1_exit"
    )

    assert not generator.evaluate(
        outgoing,
        _tracklet("wrong-edge", "C3", started=2.0, ended=2.5),
    ).accepted
    assert not generator.evaluate(
        outgoing,
        _tracklet("too-short", "C2", started=1.1, ended=1.5, first_zone="c2_entry"),
    ).accepted
    assert not generator.evaluate(
        outgoing,
        _tracklet("too-long", "C2", started=8.0, ended=8.5, first_zone="c2_entry"),
    ).accepted
    wrong_zone = _tracklet(
        "wrong-zone", "C2", started=2.0, ended=2.5, first_zone="c2_exit"
    )
    decision = generator.evaluate(outgoing, wrong_zone)
    assert decision.accepted is False
    assert "zone entrante non entry" in decision.reasons

    outgoing.summary_json["last_anchor_world_m"] = [0.0, 0.0]
    outgoing.summary_json["world_frame_id"] = "site-world"
    impossible_motion = _tracklet(
        "world-impossible",
        "C2",
        started=2.0,
        ended=2.5,
        first_zone="c2_entry",
    )
    impossible_motion.summary_json["first_anchor_world_m"] = [20.0, 0.0]
    impossible_motion.summary_json["world_frame_id"] = "site-world"
    world_decision = generator.evaluate(outgoing, impossible_motion)
    assert world_decision.accepted is False
    assert "mouvement monde impossible" in world_decision.reasons

    impossible_motion.summary_json["world_frame_id"] = "another-world"
    different_frames = generator.evaluate(outgoing, impossible_motion)
    assert different_frames.accepted is True
    assert different_frames.evidence["world_available"] is False
    assert different_frames.evidence["same_world_frame"] is False


def test_ingress_creation_match_and_intermediate_unresolved_without_world_calibration():
    tracker = GlobalParcelTracker(TOPOLOGY, {}, zones_by_role=ZONES)
    first = tracker.process_tracklet(
        _tracklet(
            "out", "C1", started=0.0, ended=1.0, vector=[1, 0, 0], last_zone="c1_exit"
        )
    )
    matched = tracker.process_tracklet(
        _tracklet(
            "in", "C2", started=2.0, ended=2.5, vector=[1, 0, 0], first_zone="c2_entry"
        )
    )
    unresolved = tracker.process_tracklet(
        _tracklet(
            "orphan", "C3", started=20.0, ended=20.5, vector=[0, 1, 0], first_zone="c3_entry"
        )
    )

    assert first[1] == MatchResult.NEW_AT_INGRESS
    assert matched[1] == MatchResult.MATCHED
    assert matched[0] == first[0]
    assert matched[3] is not None and matched[3].features["reid"] > 0.99
    assert matched[3].gate_evidence["world_available"] is False
    assert unresolved[1] == MatchResult.UNRESOLVED
    assert unresolved[0] == ""
    assert "orphan" not in tracker.tracklet_to_parcel


def test_score_below_threshold_is_unresolved_and_reid_disabled_still_matches():
    strict = GlobalParcelTracker(
        TOPOLOGY,
        {},
        zones_by_role=ZONES,
        minimum_score=0.99,
    )
    strict.process_tracklet(
        _tracklet("strict-out", "C1", started=0.0, ended=1.0, vector=[1, 0])
    )
    decision = strict.process_tracklet(
        _tracklet("strict-in", "C2", started=2.0, ended=2.5, vector=[0, 1])
    )
    assert decision[1] == MatchResult.UNRESOLVED
    assert decision[0] == ""

    no_reid = GlobalParcelTracker(
        TOPOLOGY,
        {},
        zones_by_role=ZONES,
        reid_enabled=False,
    )
    parcel_id = no_reid.process_tracklet(
        _tracklet("plain-out", "C1", started=0.0, ended=1.0, vector=None)
    )[0]
    matched = no_reid.process_tracklet(
        _tracklet("plain-in", "C2", started=2.0, ended=2.5, vector=None)
    )
    assert matched[1] == MatchResult.MATCHED
    assert matched[0] == parcel_id
    assert matched[3] is not None and "reid" not in matched[3].features

    legacy = GlobalParcelTracker(
        TOPOLOGY,
        {},
        zones_by_role=ZONES,
        projection=ProjectionHead(
            np.eye(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            version="projection-4d",
        ),
    )
    legacy_out = _tracklet("legacy-out", "C1", started=0.0, ended=1.0)
    legacy_in = _tracklet("legacy-in", "C2", started=2.0, ended=2.5)
    legacy_out.summary_json["appearance_embedding"] = [1.0, 0.0]
    legacy_in.summary_json["appearance_embedding"] = [1.0, 0.0]
    legacy_id = legacy.process_tracklet(legacy_out)[0]
    legacy_match = legacy.process_tracklet(legacy_in)
    assert legacy_match[1] == MatchResult.MATCHED
    assert legacy_match[0] == legacy_id
    assert "reid" not in legacy_match[3].features


def test_min_views_controls_whether_reid_is_used():
    low_views = GlobalParcelTracker(TOPOLOGY, {}, zones_by_role=ZONES)
    outgoing = _tracklet(
        "low-out", "C1", started=0.0, ended=1.0, vector=[1, 0, 0]
    )
    outgoing.summary_json["appearance_descriptor"] = _descriptor(
        [1, 0, 0], view_count=2, min_views=3
    )
    low_views.process_tracklet(outgoing)
    low_decision = low_views.process_tracklet(
        _tracklet(
            "low-in", "C2", started=2.0, ended=2.5, vector=[1, 0, 0]
        )
    )

    assert low_decision[3] is not None
    assert low_decision[3].features["reid_used"] is False
    assert "reid" not in low_decision[3].features

    enough_views = GlobalParcelTracker(TOPOLOGY, {}, zones_by_role=ZONES)
    enough_views.process_tracklet(
        _tracklet(
            "enough-out", "C1", started=0.0, ended=1.0, vector=[1, 0, 0]
        )
    )
    enough_decision = enough_views.process_tracklet(
        _tracklet(
            "enough-in", "C2", started=2.0, ended=2.5, vector=[1, 0, 0]
        )
    )
    assert enough_decision[3] is not None
    assert enough_decision[3].features["reid_used"] is True
    assert enough_decision[3].features["reid"] > 0.99


def test_lapjv_2x2_matches_both_parcels_without_silent_swap():
    tracker = GlobalParcelTracker(TOPOLOGY, {}, zones_by_role=ZONES)
    outgoing_a = _tracklet(
        "out-a", "C1", started=0.0, ended=1.0, vector=[1, 0, 0]
    )
    outgoing_b = _tracklet(
        "out-b", "C1", started=0.0, ended=1.0, vector=[0, 1, 0]
    )
    parcel_a = tracker.process_tracklet(outgoing_a)[0]
    parcel_b = tracker.process_tracklet(outgoing_b)[0]
    decisions = tracker.process_tracklets(
        [
            _tracklet(
                "in-b", "C2", started=2.0, ended=2.5, vector=[0.02, 1, 0]
            ),
            _tracklet(
                "in-a", "C2", started=2.0, ended=2.5, vector=[1, 0.02, 0]
            ),
        ]
    )

    assert [item[1] for item in decisions] == [MatchResult.MATCHED, MatchResult.MATCHED]
    assert [item[0] for item in decisions] == [parcel_b, parcel_a]
    assert [item[3].from_tracklet_id for item in decisions] == ["out-b", "out-a"]


def test_visually_indistinguishable_hard_negatives_become_ambiguous():
    tracker = GlobalParcelTracker(TOPOLOGY, {}, zones_by_role=ZONES)
    tracker.process_tracklet(
        _tracklet("close-a", "C1", started=0.0, ended=1.0, vector=[1, 0, 0])
    )
    tracker.process_tracklet(
        _tracklet(
            "close-b", "C1", started=0.0, ended=1.0, vector=[1, 0.01, 0]
        )
    )
    decisions = tracker.process_tracklets(
        [
            _tracklet(
                "close-in-a", "C2", started=2.0, ended=2.5, vector=[1, 0.002, 0]
            ),
            _tracklet(
                "close-in-b", "C2", started=2.0, ended=2.5, vector=[1, 0.008, 0]
            ),
        ]
    )

    assert all(item[1] == MatchResult.AMBIGUOUS for item in decisions)
    assert all(item[0] == "" for item in decisions)


def test_real_multicam_reid_replay_e2e():
    report = run_multicam_reid_replay()
    assert report["status"] == "PASS"
    assert report["decisions"] == ["NEW_AT_INGRESS", "MATCHED"]
    assert min(report["views_per_tracklet"]) >= 3
    assert report["reid_similarity"] >= 0.80
    assert report["handoff_score"] >= 0.48
    assert report["same_parcel_variations"]["decision"] == "MATCHED"
    assert report["hard_negative_2x2"]["status"] == "PASS"
    assert report["hard_negative_2x2"]["parcel_count"] == 2
    assert report["hard_negative_2x2"]["silent_swaps"] == 0
    assert all(
        item["decision"] in {"MATCHED", "AMBIGUOUS"}
        for item in report["hard_negative_2x2"]["outcomes"]
    )
