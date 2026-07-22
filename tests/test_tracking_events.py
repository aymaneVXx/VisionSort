from visionsort.core.types import Observation
from visionsort.events.engine import ParcelEventEngine
from visionsort.tracking.engine import GlobalParcelTracker, GreedyIOUTracker


def test_tracker_and_events_generate_pick_signal():
    tracker = GreedyIOUTracker("C2", zones=[{"zone_id": "c2_pick", "x1": 0.5, "y1": 0.0, "x2": 1.0, "y2": 1.0}])
    engine = ParcelEventEngine(zones_by_role={"C2": [{"zone_id": "zone_A", "x1": 0.6, "y1": 0.0, "x2": 1.0, "y2": 1.0}]}, source_roles={"C2": "C2"})
    event_types = []
    for frame_index in range(8):
        observations = [
            Observation("parcel", 0.95, (100 + frame_index * 10, 100, 160 + frame_index * 10, 150), attributes={"parcel_hint": "P1"}),
            Observation("person", 0.95, (120 + frame_index * 10, 70, 220 + frame_index * 10, 250), attributes={"operator_id": "OP1"}),
            Observation("left_wrist", 0.95, (125 + frame_index * 10, 110, 145 + frame_index * 10, 130), attributes={"operator_id": "OP1"}),
        ]
        track_obs, _ = tracker.update(frame_index, frame_index * 0.2, observations)
        events = engine.update("C2", [obs for obs in track_obs if obs.class_name == "parcel"], [obs for obs in track_obs if obs.class_name != "parcel"])
        event_types.extend(event["event_type"] for event in events)
    assert any(item in event_types for item in ["pickup_candidate", "parcel_picked", "parcel_carried"])


def test_global_tracker_matches_topology():
    tracker = GlobalParcelTracker(
        topology_edges=[{"from_role": "C1", "to_role": "C2", "min_transit_s": 0.1, "max_transit_s": 5.0}],
        source_roles={"C1": "C1", "C2": "C2"},
    )
    first = tracker.process_tracklet(
        type("T", (), {"tracklet_id": "t1", "camera_id": "C1", "started_at": 0.0, "ended_at": 1.0, "first_bbox": (0, 0, 10, 10), "last_bbox": (0, 0, 12, 12), "summary_json": {"parcel_hint": "P1"}})()
    )
    second = tracker.process_tracklet(
        type("T", (), {"tracklet_id": "t2", "camera_id": "C2", "started_at": 2.0, "ended_at": 3.0, "first_bbox": (0, 0, 11, 11), "last_bbox": (0, 0, 12, 12), "summary_json": {"parcel_hint": "P1"}})()
    )
    assert first[0] == "P1"
    assert second[0] == "P1"
    assert second[1].value == "MATCHED"
