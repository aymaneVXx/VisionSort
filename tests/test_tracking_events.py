from visionsort.core.types import Observation, Tracklet
from visionsort.events.engine import ParcelEventEngine
from visionsort.tracking.engine import (
    BoTSORTTracker,
    ByteTrackTracker,
    GlobalParcelTracker,
    GreedyIOUTracker,
    build_tracker,
)


def _tracklet(
    tracklet_id: str,
    role: str,
    *,
    started: float,
    ended: float,
    width: float = 12.0,
    speed: float = 5.0,
    parcel_hint: str | None = None,
) -> Tracklet:
    camera_id = f"cam-{role.lower()}-{tracklet_id}"
    summary = {
        "avg_dimensions": [width, 10.0],
        "avg_velocity": [speed, 0.0],
        "first_zone_id": f"{role.lower()}_entry",
        "last_zone_id": f"{role.lower()}_exit",
        "appearance_embedding": [1.0, 0.0] if width < 15.0 else [0.0, 1.0],
        "ground_truth": {"parcel_hint": parcel_hint} if parcel_hint else {},
    }
    return Tracklet(
        tracklet_id=tracklet_id,
        session_id="session-test",
        source_id=camera_id,
        camera_id=camera_id,
        camera_role=role,
        local_track_id=1,
        started_at_local=started,
        ended_at_local=ended,
        started_at_global=started,
        ended_at_global=ended,
        class_name="parcel",
        first_bbox=(0.0, 0.0, width, 10.0),
        last_bbox=(1.0, 0.0, width + 1.0, 10.0),
        avg_speed=speed,
        last_zone_id=f"{role.lower()}_exit",
        frame_count=8,
        observation_path="details.jsonl",
        summary_json=summary,
        model_id="demo_synth_det",
        tracker_id="greedy_iou",
    )


def _global_tracker() -> GlobalParcelTracker:
    return GlobalParcelTracker(
        topology_edges=[
            {
                "from_role": "C1",
                "to_role": "C2",
                "min_transit_s": 0.1,
                "max_transit_s": 5.0,
            }
        ],
        source_roles={},
    )


def test_selected_tracker_instantiates_real_ultralytics_backend():
    common = {
        "session_id": "s",
        "source_id": "src",
        "camera_id": "cam",
        "camera_role": "C1",
        "zones": [],
    }
    greedy = build_tracker(tracker_id="greedy_iou", **common)
    byte = build_tracker(tracker_id="bytetrack_cpu", **common)
    bot = build_tracker(tracker_id="botsort_cpu", **common)

    assert type(greedy) is GreedyIOUTracker
    assert type(byte) is ByteTrackTracker
    assert type(bot) is BoTSORTTracker
    assert type(byte.native_tracker).__name__ == "BYTETracker"
    assert type(bot.native_tracker).__name__ == "BOTSORT"


def test_tracker_and_events_generate_pick_signal_without_runtime_parcel_hint():
    tracker = GreedyIOUTracker(
        session_id="session-test",
        source_id="src2",
        camera_id="src2",
        camera_role="C2",
        tracker_id="greedy_iou",
        zones=[{"zone_id": "c2_pick", "x1": 0, "y1": 0, "x2": 1000, "y2": 1000}],
    )
    engine = ParcelEventEngine(
        zones_by_role={
            "C2": [{"zone_id": "zone_A", "x1": 0, "y1": 0, "x2": 1000, "y2": 1000}]
        },
        source_roles={"src2": "C2"},
    )
    event_types = []
    for frame_index in range(8):
        observations = [
            Observation(
                "parcel",
                0.95,
                (100 + frame_index * 10, 100, 160 + frame_index * 10, 150),
                attributes={"parcel_hint": "GROUND_TRUTH_ONLY"},
            ),
            Observation(
                "person",
                0.95,
                (120 + frame_index * 10, 70, 220 + frame_index * 10, 250),
            ),
            Observation(
                "left_wrist",
                0.95,
                (125 + frame_index * 10, 110, 145 + frame_index * 10, 130),
            ),
        ]
        track_obs, _ = tracker.update(
            frame_index=frame_index,
            timestamp_local=frame_index * 0.2,
            timestamp_global=frame_index * 0.2,
            image_size=(640, 360),
            observations=observations,
        )
        events = engine.update(
            "src2",
            [obs for obs in track_obs if obs.class_name == "parcel"],
            [obs for obs in track_obs if obs.class_name != "parcel"],
        )
        assert all(event["parcel_id"].startswith("src2:") for event in events)
        event_types.extend(event["event_type"] for event in events)
    assert any(
        item in event_types
        for item in ["pickup_candidate", "parcel_picked", "parcel_carried"]
    )


def test_global_tracker_matches_previous_and_incoming_tracklets_without_hint():
    tracker = _global_tracker()
    first = tracker.process_tracklet(
        _tracklet("t1", "C1", started=0.0, ended=1.0, parcel_hint="WRONG-A")
    )
    second = tracker.process_tracklet(
        _tracklet("t2", "C2", started=2.5, ended=3.0, parcel_hint="WRONG-B")
    )

    assert first[0].startswith("parcel-")
    assert first[0] not in {"WRONG-A", "WRONG-B"}
    assert second[0] == first[0]
    assert second[1].value == "MATCHED"
    assert second[3] is not None
    assert second[3].from_tracklet_id == "t1"
    assert second[3].to_tracklet_id == "t2"


def test_global_tracker_batch_is_one_to_one_when_incoming_order_is_reversed():
    tracker = _global_tracker()
    first_small = tracker.process_tracklet(
        _tracklet("out-small", "C1", started=0.0, ended=1.0, width=8.0)
    )[0]
    first_large = tracker.process_tracklet(
        _tracklet("out-large", "C1", started=0.1, ended=1.1, width=22.0)
    )[0]

    outcomes = tracker.process_tracklets(
        [
            _tracklet("in-large", "C2", started=2.4, ended=3.0, width=22.0),
            _tracklet("in-small", "C2", started=2.5, ended=3.1, width=8.0),
        ]
    )

    assert outcomes[0][0] == first_large
    assert outcomes[1][0] == first_small
    assert {item[3].from_tracklet_id for item in outcomes if item[3]} == {
        "out-small",
        "out-large",
    }


def test_global_tracker_does_not_force_ambiguous_competitors():
    tracker = _global_tracker()
    tracker.process_tracklet(_tracklet("out-a", "C1", started=0.0, ended=1.0, width=12.0))
    tracker.process_tracklet(_tracklet("out-b", "C1", started=0.0, ended=1.0, width=12.1))

    result = tracker.process_tracklet(
        _tracklet("incoming", "C2", started=2.5, ended=3.0, width=12.05)
    )

    assert result[0] == ""
    assert result[1].value == "AMBIGUOUS"
    assert result[3] is not None


def test_global_tracker_returns_unmatched_outside_transit_window():
    tracker = _global_tracker()
    first_id = tracker.process_tracklet(
        _tracklet("out", "C1", started=0.0, ended=1.0)
    )[0]
    result = tracker.process_tracklet(
        _tracklet("late", "C2", started=20.0, ended=21.0)
    )

    assert result[1].value == "UNMATCHED"
    assert result[0] != first_id
