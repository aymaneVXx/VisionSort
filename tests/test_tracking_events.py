from visionsort.core.types import Observation
from visionsort.events.engine import ParcelEventEngine
from visionsort.tracking.engine import GlobalParcelTracker, GreedyIOUTracker


def test_tracker_and_events_generate_pick_signal():
    tracker = GreedyIOUTracker(
        session_id="session-test",
        source_id="src2",
        camera_id="src2",
        camera_role="C2",
        tracker_id="greedy_iou",
        zones=[{"zone_id": "c2_pick", "x1": 0, "y1": 0, "x2": 1000, "y2": 1000}],
    )
    engine = ParcelEventEngine(
        zones_by_role={"C2": [{"zone_id": "zone_A", "x1": 0, "y1": 0, "x2": 1000, "y2": 1000}]},
        source_roles={"src2": "C2"},
    )
    event_types = []
    for frame_index in range(8):
        observations = [
            Observation("parcel", 0.95, (100 + frame_index * 10, 100, 160 + frame_index * 10, 150), attributes={"parcel_hint": "P1"}),
            Observation("person", 0.95, (120 + frame_index * 10, 70, 220 + frame_index * 10, 250), attributes={"operator_id": "OP1"}),
            Observation("left_wrist", 0.95, (125 + frame_index * 10, 110, 145 + frame_index * 10, 130), attributes={"operator_id": "OP1"}),
        ]
        track_obs, _ = tracker.update(
            frame_index=frame_index,
            timestamp_local=frame_index * 0.2,
            timestamp_global=frame_index * 0.2,
            observations=observations,
        )
        events = engine.update("src2", [obs for obs in track_obs if obs.class_name == "parcel"], [obs for obs in track_obs if obs.class_name != "parcel"])
        event_types.extend(event["event_type"] for event in events)
    assert any(item in event_types for item in ["pickup_candidate", "parcel_picked", "parcel_carried"])


def test_global_tracker_matches_topology():
    tracker = GlobalParcelTracker(
        topology_edges=[{"from_role": "C1", "to_role": "C2", "min_transit_s": 0.1, "max_transit_s": 5.0}],
        source_roles={"C1": "C1", "C2": "C2"},
    )
    first = tracker.process_tracklet(
        type(
            "T",
            (),
            {
                "tracklet_id": "t1",
                "camera_id": "C1",
                "camera_role": "C1",
                "started_at_global": 0.0,
                "ended_at_global": 1.0,
                "first_bbox": (0, 0, 10, 10),
                "last_bbox": (0, 0, 12, 12),
                "summary_json": {"parcel_hint": "P1"},
            },
        )()
    )
    second = tracker.process_tracklet(
        type(
            "T",
            (),
            {
                "tracklet_id": "t2",
                "camera_id": "C2",
                "camera_role": "C2",
                "started_at_global": 2.0,
                "ended_at_global": 3.0,
                "first_bbox": (0, 0, 11, 11),
                "last_bbox": (0, 0, 12, 12),
                "summary_json": {"parcel_hint": "P1"},
            },
        )()
    )
    assert first[0] == "P1"
    assert second[0] == "P1"
    assert second[1].value == "MATCHED"


def test_global_tracker_marks_ambiguous_when_two_candidates_are_close():
    tracker = GlobalParcelTracker(
        topology_edges=[{"from_role": "C1", "to_role": "C2", "min_transit_s": 0.5, "max_transit_s": 5.0}],
        source_roles={"cam1a": "C1", "cam1b": "C1", "cam2": "C2"},
    )

    tracker.process_tracklet(
        type(
            "T",
            (),
            {
                "tracklet_id": "t1",
                "camera_id": "cam1a",
                "camera_role": "C1",
                "started_at_global": 0.0,
                "ended_at_global": 1.0,
                "first_bbox": (0, 0, 10, 10),
                "last_bbox": (0, 0, 12, 12),
                "summary_json": {"parcel_hint": "P1"},
            },
        )()
    )
    tracker.process_tracklet(
        type(
            "T",
            (),
            {
                "tracklet_id": "t2",
                "camera_id": "cam1b",
                "camera_role": "C1",
                "started_at_global": 0.2,
                "ended_at_global": 1.1,
                "first_bbox": (0, 0, 10.2, 10.2),
                "last_bbox": (0, 0, 12.1, 12.1),
                "summary_json": {"parcel_hint": "P2"},
            },
        )()
    )

    result = tracker.process_tracklet(
        type(
            "T",
            (),
            {
                "tracklet_id": "t3",
                "camera_id": "cam2",
                "camera_role": "C2",
                "started_at_global": 2.0,
                "ended_at_global": 3.0,
                "first_bbox": (0, 0, 12.0, 12.0),
                "last_bbox": (0, 0, 12.0, 12.0),
                "summary_json": {},
            },
        )()
    )

    assert result[0] == ""
    assert result[1].value == "AMBIGUOUS"
    assert result[3] is not None
