from visionsort.core.enums import DestinationResult, ParcelState
from visionsort.core.types import (
    Observation,
    TrackObservation,
    Tracklet,
    evaluate_destination,
)
from visionsort.events.engine import ParcelEventEngine
from visionsort.events.interactions import InteractionMatcher
from visionsort.tracking.engine import (
    BoTSORTTracker,
    ByteTrackTracker,
    GlobalParcelTracker,
    GreedyIOUTracker,
    build_tracker,
)
from visionsort.tracking.person import PersonTrackBuilder


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

    canonical, _ = byte.update(
        frame_index=0,
        timestamp_local=0.0,
        timestamp_global=0.0,
        image_size=(640, 360),
        observations=[Observation("parcel", 0.95, (100, 100, 150, 150))],
        stream_epoch=0,
    )
    assert len(canonical) == 1
    assert canonical[0].backend_track_id is not None
    assert canonical[0].anchor_px == (125.0, 150.0)
    assert canonical[0].extra["track_identity"] == ["cam", canonical[0].local_track_id]


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
            "C2": [
                {
                    "zone_id": "c2_pick",
                    "kind": "pick",
                    "x1": 0,
                    "y1": 0,
                    "x2": 1000,
                    "y2": 1000,
                }
            ]
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
                attributes={"operator_id": "OP1"},
            ),
            Observation(
                "left_wrist",
                0.95,
                (125 + frame_index * 10, 110, 145 + frame_index * 10, 130),
                attributes={"operator_id": "OP1"},
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


def _context_track(
    person_id: int,
    *,
    timestamp: float,
    person_x: float,
    wrist: tuple[float, float] | None,
    operator_id: str | None = None,
) -> TrackObservation:
    keypoints = [(0.0, 0.0, 0.0) for _ in range(17)]
    if wrist is not None:
        keypoints[9] = (wrist[0], wrist[1], 0.95)
        keypoints[10] = (wrist[0] + 5.0, wrist[1], 0.95)
    return TrackObservation(
        session_id="session-events",
        source_id="cam-events",
        camera_id="cam-events",
        camera_role="C2",
        local_track_id=person_id,
        frame_index=round(timestamp * 100),
        timestamp_local=timestamp,
        timestamp_global=timestamp,
        class_name="person",
        confidence=0.95,
        bbox=(person_x, 50.0, person_x + 140.0, 330.0),
        velocity=(0.0, 0.0),
        tracker_id="bytetrack_cpu",
        extra={
            "_image_w": 1000,
            "_image_h": 500,
            "keypoints": keypoints,
            **({"operator_id": operator_id} if operator_id else {}),
        },
    )


def _parcel_track(
    parcel_id: int,
    *,
    timestamp: float,
    x: float,
    velocity_x: float = 20.0,
    identity_status: str = "STABLE",
) -> TrackObservation:
    return TrackObservation(
        session_id="session-events",
        source_id="cam-events",
        camera_id="cam-events",
        camera_role="C2",
        local_track_id=parcel_id,
        backend_track_id=parcel_id + 100,
        frame_index=round(timestamp * 100),
        timestamp_local=timestamp,
        timestamp_global=timestamp,
        class_name="parcel",
        confidence=0.95,
        bbox=(x, 150.0, x + 40.0, 200.0),
        velocity=(velocity_x, 0.0),
        tracker_id="bytetrack_cpu",
        identity_status=identity_status,
        extra={"_image_w": 1000, "_image_h": 500},
    )


def _event_engine() -> ParcelEventEngine:
    return ParcelEventEngine(
        zones_by_role={
            "C2": [
                {
                    "zone_id": "pick-zone",
                    "kind": "pick",
                    "x1": 0.0,
                    "y1": 0.0,
                    "x2": 0.5,
                    "y2": 1.0,
                },
                {
                    "zone_id": "destination-A",
                    "kind": "destination",
                    "x1": 0.5,
                    "y1": 0.0,
                    "x2": 1.0,
                    "y2": 1.0,
                },
            ]
        },
        source_roles={"cam-events": "C2"},
    )


def test_person_builder_never_attaches_an_unowned_nearby_wrist():
    person = _context_track(
        7, timestamp=0.0, person_x=100.0, wrist=None
    )
    stray_wrist = TrackObservation(
        **{
            **person.to_json(),
            "local_track_id": 99,
            "class_name": "left_wrist",
            "bbox": (110.0, 150.0, 120.0, 160.0),
            "extra": {"_image_w": 1000, "_image_h": 500},
        }
    )

    built = PersonTrackBuilder().update([person, stray_wrist])

    assert len(built) == 1
    assert built[0].person_track_id == 7
    assert built[0].left_wrist is None
    assert built[0].right_wrist is None


def test_interaction_matcher_one_operator_one_parcel():
    builder = PersonTrackBuilder()
    person = builder.update(
        [_context_track(7, timestamp=0.0, person_x=80.0, wrist=(120.0, 170.0))]
    )
    parcel = _parcel_track(1, timestamp=0.0, x=100.0)

    match = InteractionMatcher().match(
        [parcel], person, pick_zone_by_parcel={"cam-events:1": True}
    )["cam-events:1"]

    assert match.person_track_id == 7
    assert match.reliable is True
    assert match.contact is True


def test_interaction_matcher_two_operators_two_parcels_is_one_to_one():
    builder = PersonTrackBuilder()
    persons = builder.update(
        [
            _context_track(7, timestamp=0.0, person_x=70.0, wrist=(120.0, 170.0)),
            _context_track(8, timestamp=0.0, person_x=650.0, wrist=(700.0, 170.0)),
        ]
    )
    parcels = [
        _parcel_track(1, timestamp=0.0, x=100.0),
        _parcel_track(2, timestamp=0.0, x=680.0),
    ]

    matches = InteractionMatcher().match(parcels, persons)

    assert matches["cam-events:1"].person_track_id == 7
    assert matches["cam-events:2"].person_track_id == 8
    assert {item.person_track_id for item in matches.values()} == {7, 8}


def test_matcher_uses_owned_wrist_not_wrong_nearby_person_bbox():
    builder = PersonTrackBuilder()
    persons = builder.update(
        [
            _context_track(7, timestamp=0.0, person_x=20.0, wrist=(120.0, 170.0)),
            _context_track(8, timestamp=0.0, person_x=95.0, wrist=(400.0, 170.0)),
        ]
    )
    parcel = _parcel_track(1, timestamp=0.0, x=100.0)

    match = InteractionMatcher().match([parcel], persons)["cam-events:1"]

    assert match.person_track_id == 7


def test_transient_contact_never_confirms_pickup():
    engine = _event_engine()
    events = engine.update(
        "cam-events",
        [_parcel_track(1, timestamp=0.0, x=100.0)],
        [_context_track(7, timestamp=0.0, person_x=80.0, wrist=(120.0, 170.0))],
    )
    events += engine.update(
        "cam-events",
        [_parcel_track(1, timestamp=0.1, x=102.0)],
        [_context_track(7, timestamp=0.1, person_x=82.0, wrist=(122.0, 170.0))],
    )
    events += engine.update(
        "cam-events",
        [_parcel_track(1, timestamp=0.15, x=103.0)],
        [_context_track(7, timestamp=0.15, person_x=83.0, wrist=(400.0, 170.0))],
    )

    assert "pickup_candidate" in {event["event_type"] for event in events}
    assert "pickup_cancelled" in {event["event_type"] for event in events}
    assert "parcel_picked" not in {event["event_type"] for event in events}
    assert engine.parcels["cam-events:1"].state == ParcelState.ON_CONVEYOR


def _drive_persistent_pickup(engine: ParcelEventEngine) -> list[dict]:
    events: list[dict] = []
    for index, timestamp in enumerate((0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
        x_value = 100.0 + index * 2.0
        events.extend(
            engine.update(
                "cam-events",
                [_parcel_track(1, timestamp=timestamp, x=x_value)],
                [
                    _context_track(
                        7,
                        timestamp=timestamp,
                        person_x=80.0 + index * 2.0,
                        wrist=(x_value + 20.0, 170.0),
                    )
                ],
            )
        )
    return events


def test_persistent_contact_confirms_pickup_and_carried_state():
    engine = _event_engine()

    events = _drive_persistent_pickup(engine)
    event_types = {event["event_type"] for event in events}

    assert {"pickup_candidate", "parcel_picked", "parcel_carried"} <= event_types
    assert engine.parcels["cam-events:1"].state == ParcelState.CARRIED
    assert engine.parcels["cam-events:1"].operator_id == "cam-events:7"


def test_ambiguous_parcel_identity_cannot_be_picked():
    engine = _event_engine()
    events: list[dict] = []
    for index, timestamp in enumerate((0.0, 0.1, 0.2, 0.3)):
        x_value = 100.0 + index * 2.0
        events.extend(
            engine.update(
                "cam-events",
                [
                    _parcel_track(
                        1,
                        timestamp=timestamp,
                        x=x_value,
                        identity_status="AMBIGUOUS",
                    )
                ],
                [
                    _context_track(
                        7,
                        timestamp=timestamp,
                        person_x=80.0 + index * 2.0,
                        wrist=(x_value + 20.0, 170.0),
                    )
                ],
            )
        )

    assert "parcel_picked" not in {event["event_type"] for event in events}
    assert engine.parcels["cam-events:1"].state == ParcelState.ON_CONVEYOR


def test_carried_shadow_reappearance_false_drop_and_true_drop():
    engine = _event_engine()
    _drive_persistent_pickup(engine)

    missing_events = engine.update(
        "cam-events",
        [],
        [_context_track(7, timestamp=0.6, person_x=90.0, wrist=(130.0, 170.0))],
        timestamp_global=0.6,
    )
    evidence = engine.parcels["cam-events:1"]
    assert evidence.shadow is not None
    assert missing_events[0]["payload"].get("bbox") is None
    assert missing_events[0]["event_type"] == "parcel_carried_shadow"

    reappearance = engine.update(
        "cam-events",
        [_parcel_track(1, timestamp=0.7, x=600.0)],
        [_context_track(7, timestamp=0.7, person_x=570.0, wrist=(620.0, 170.0))],
    )
    assert "parcel_reappeared" in {event["event_type"] for event in reappearance}
    assert evidence.shadow is None

    contact_in_destination = engine.update(
        "cam-events",
        [_parcel_track(1, timestamp=0.75, x=600.0, velocity_x=0.0)],
        [_context_track(7, timestamp=0.75, person_x=570.0, wrist=(620.0, 170.0))],
    )
    assert "parcel_dropped" not in {
        event["event_type"] for event in contact_in_destination
    }

    drop_events: list[dict] = []
    for timestamp in (0.8, 0.9, 1.1):
        drop_events.extend(
            engine.update(
                "cam-events",
                [_parcel_track(1, timestamp=timestamp, x=600.0, velocity_x=0.0)],
                [
                    _context_track(
                        7,
                        timestamp=timestamp,
                        person_x=570.0,
                        wrist=(760.0, 170.0),
                    )
                ],
            )
        )
    event_types = {event["event_type"] for event in drop_events}
    assert {"drop_candidate", "parcel_dropped"} <= event_types
    assert evidence.state == ParcelState.DROPPED


def test_destination_result_contract():
    assert (
        evaluate_destination("destination-A", "destination-A")
        == DestinationResult.SORT_OK
    )
    assert (
        evaluate_destination("destination-A", "destination-B")
        == DestinationResult.WRONG_DESTINATION
    )
    assert (
        evaluate_destination(None, "destination-A")
        == DestinationResult.DESTINATION_UNVERIFIED
    )
    assert (
        evaluate_destination("destination-A", None)
        == DestinationResult.DESTINATION_UNVERIFIED
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
