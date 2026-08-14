from __future__ import annotations

from dataclasses import dataclass

import pytest

from visionsort.core.types import TrackObservation
from visionsort.core.site_config import validate_site_config
from visionsort.tracking.geometry import GroundAnchorEstimator
from visionsort.tracking.integrity import TrackIntegrityManager
from visionsort.tracking.replay_benchmark import ReplayFrame, benchmark_replay


def _backend_track(
    backend_id: int,
    *,
    x: float,
    timestamp: float,
    frame_index: int,
    y: float = 100.0,
    width: float = 24.0,
    height: float = 20.0,
    mask: list[list[float]] | None = None,
) -> TrackObservation:
    return TrackObservation(
        session_id="session-integrity",
        source_id="cam-1",
        camera_id="cam-1",
        camera_role="C1",
        local_track_id=backend_id,
        backend_track_id=backend_id,
        frame_index=frame_index,
        timestamp_local=timestamp,
        timestamp_global=timestamp,
        class_name="parcel",
        confidence=0.95,
        bbox=(x, y, x + width, y + height),
        velocity=(0.0, 0.0),
        model_id="model-test",
        tracker_id="bytetrack_cpu",
        extra={"mask": mask} if mask else {},
    )


@pytest.fixture
def manager(monkeypatch, tmp_path) -> TrackIntegrityManager:
    import visionsort.tracking.integrity as integrity_module

    monkeypatch.setattr(integrity_module, "DETAILS_DIR", tmp_path / "details")
    return TrackIntegrityManager(
        session_id="session-integrity",
        source_id="cam-1",
        camera_id="cam-1",
        camera_role="C1",
        tracker_id="bytetrack_cpu",
        config={
            "max_occlusion_seconds": 0.75,
            "max_speed_m_s": 3.0,
            "min_relink_score": 0.50,
            "ambiguity_margin": 0.10,
        },
    )


def _update(
    manager: TrackIntegrityManager,
    tracks: list[TrackObservation],
    timestamp: float,
    frame_index: int,
    *,
    epoch: int = 0,
    geometry=None,
):
    return manager.update(
        tracks,
        frame_index=frame_index,
        timestamp_global=timestamp,
        image_size=(1000, 500),
        stream_epoch=epoch,
        world_geometry=geometry,
    )


def test_nominal_backend_id_keeps_canonical_id(manager):
    first, _ = _update(manager, [_backend_track(17, x=100, timestamp=0.0, frame_index=0)], 0.0, 0)
    second, _ = _update(manager, [_backend_track(17, x=110, timestamp=0.1, frame_index=1)], 0.1, 1)

    assert first[0].local_track_id == second[0].local_track_id
    assert second[0].backend_track_id == 17
    assert second[0].local_track_id != 17


def test_irregular_fps_uses_real_timestamps(manager):
    ids = []
    for frame, timestamp in enumerate([0.0, 0.07, 0.31, 0.46]):
        output, _ = _update(
            manager,
            [_backend_track(7, x=100 + timestamp * 100, timestamp=timestamp, frame_index=frame)],
            timestamp,
            frame,
        )
        ids.append(output[0].local_track_id)
    assert len(set(ids)) == 1
    assert manager.states[ids[0]].anchors[-1].timestamp == pytest.approx(0.46)


def test_simple_fragmentation_relinks_backend_17_to_42(manager):
    first, _ = _update(manager, [_backend_track(17, x=100, timestamp=0.0, frame_index=0)], 0.0, 0)
    _update(manager, [_backend_track(17, x=110, timestamp=0.1, frame_index=1)], 0.1, 1)
    _update(manager, [], 0.2, 2)
    recovered, _ = _update(manager, [_backend_track(42, x=130, timestamp=0.3, frame_index=3)], 0.3, 3)

    assert recovered[0].local_track_id == first[0].local_track_id
    assert recovered[0].backend_track_id == 42
    assert recovered[0].identity_status == "RECOVERED"
    assert any(event["event_type"] == "BACKEND_ID_RELINKED" for event in manager.pop_events())


def test_long_gap_creates_new_identity(manager):
    first, _ = _update(manager, [_backend_track(17, x=100, timestamp=0.0, frame_index=0)], 0.0, 0)
    _, finalized = _update(manager, [], 1.0, 1)
    second, _ = _update(manager, [_backend_track(42, x=105, timestamp=1.1, frame_index=2)], 1.1, 2)

    assert finalized and finalized[0].local_track_id == first[0].local_track_id
    assert second[0].local_track_id != first[0].local_track_id


def test_physically_impossible_candidate_is_not_relinked(manager):
    first, _ = _update(manager, [_backend_track(17, x=10, timestamp=0.0, frame_index=0)], 0.0, 0)
    _update(manager, [], 0.1, 1)
    second, _ = _update(manager, [_backend_track(42, x=850, timestamp=0.2, frame_index=2)], 0.2, 2)

    assert second[0].local_track_id != first[0].local_track_id


def test_two_old_and_two_new_tracks_use_global_assignment(manager):
    first, _ = _update(
        manager,
        [
            _backend_track(10, x=100, timestamp=0.0, frame_index=0, width=20),
            _backend_track(20, x=500, timestamp=0.0, frame_index=0, width=42),
        ],
        0.0,
        0,
    )
    ids_by_width = {round(item.bbox[2] - item.bbox[0]): item.local_track_id for item in first}
    _update(manager, [], 0.1, 1)
    output, _ = _update(
        manager,
        [
            _backend_track(99, x=505, timestamp=0.2, frame_index=2, width=42),
            _backend_track(88, x=105, timestamp=0.2, frame_index=2, width=20),
        ],
        0.2,
        2,
    )
    mapped = {item.backend_track_id: item.local_track_id for item in output}

    assert mapped[88] == ids_by_width[20]
    assert mapped[99] == ids_by_width[42]
    assert manager.metrics()["lapjv_subproblems"] == 1


def test_ambiguous_relink_refuses_to_force_old_identity(manager):
    original, _ = _update(
        manager,
        [
            _backend_track(1, x=390, timestamp=0.0, frame_index=0),
            _backend_track(2, x=590, timestamp=0.0, frame_index=0),
        ],
        0.0,
        0,
    )
    old_ids = {item.local_track_id for item in original}
    _update(manager, [], 0.1, 1)
    output, _ = _update(manager, [_backend_track(3, x=490, timestamp=0.2, frame_index=2)], 0.2, 2)

    assert output[0].local_track_id not in old_ids
    assert output[0].identity_status == "AMBIGUOUS"
    assert any(event["event_type"] == "IDENTITY_AMBIGUOUS" for event in manager.pop_events())


def test_short_occlusion_with_same_backend_id_is_recovered(manager):
    first, _ = _update(manager, [_backend_track(5, x=100, timestamp=0.0, frame_index=0)], 0.0, 0)
    _update(manager, [], 0.2, 1)
    output, _ = _update(manager, [_backend_track(5, x=120, timestamp=0.3, frame_index=2)], 0.3, 2)

    assert output[0].local_track_id == first[0].local_track_id
    assert output[0].identity_status == "RECOVERED"


def test_clear_merge_two_to_one_to_two_recovers_both(manager):
    initial, _ = _update(
        manager,
        [
            _backend_track(1, x=100, timestamp=0.0, frame_index=0, width=24),
            _backend_track(2, x=260, timestamp=0.0, frame_index=0, width=46),
        ],
        0.0,
        0,
    )
    initial_ids = {round(item.bbox[2] - item.bbox[0]): item.local_track_id for item in initial}
    merged, _ = _update(
        manager,
        [_backend_track(1, x=95, timestamp=0.1, frame_index=1, width=215)],
        0.1,
        1,
    )
    split, _ = _update(
        manager,
        [
            _backend_track(31, x=110, timestamp=0.2, frame_index=2, width=24),
            _backend_track(32, x=250, timestamp=0.2, frame_index=2, width=46),
        ],
        0.2,
        2,
    )

    assert merged == []
    assert {item.local_track_id for item in split} == set(initial_ids.values())
    assert any(event["event_type"] == "SPLIT_RESOLVED" for event in manager.pop_events())


def test_ambiguous_merge_split_emits_event_without_silent_swap(manager):
    initial, _ = _update(
        manager,
        [
            _backend_track(1, x=200, timestamp=0.0, frame_index=0, width=30),
            _backend_track(2, x=270, timestamp=0.0, frame_index=0, width=30),
        ],
        0.0,
        0,
    )
    old_ids = {item.local_track_id for item in initial}
    _update(
        manager,
        [_backend_track(9, x=195, timestamp=0.1, frame_index=1, width=110)],
        0.1,
        1,
    )
    split, _ = _update(
        manager,
        [
            _backend_track(10, x=230, timestamp=0.2, frame_index=2, width=30),
            _backend_track(11, x=240, timestamp=0.2, frame_index=2, width=30),
        ],
        0.2,
        2,
    )

    assert old_ids.isdisjoint({item.local_track_id for item in split})
    assert all(item.identity_status == "AMBIGUOUS" for item in split)
    assert any(event["event_type"] == "IDENTITY_AMBIGUOUS" for event in manager.pop_events())


def test_three_parcel_merge_does_not_assign_arbitrarily(manager):
    _update(
        manager,
        [
            _backend_track(1, x=100, timestamp=0.0, frame_index=0),
            _backend_track(2, x=150, timestamp=0.0, frame_index=0),
            _backend_track(3, x=200, timestamp=0.0, frame_index=0),
        ],
        0.0,
        0,
    )
    merged, _ = _update(
        manager,
        [_backend_track(7, x=95, timestamp=0.1, frame_index=1, width=155)],
        0.1,
        1,
    )

    assert merged == []
    assert len(manager.occlusion_groups) == 1
    assert len(next(iter(manager.occlusion_groups.values())).member_local_track_ids) == 3


def test_segmentation_absent_falls_back_to_bbox_bottom_center():
    anchor = GroundAnchorEstimator().estimate(
        bbox=(10.0, 20.0, 50.0, 80.0),
        mask=None,
        image_size=(100, 100),
    )
    assert anchor.pixel == (30.0, 80.0)
    assert anchor.method == "BBOX_BOTTOM_CENTER"


def test_calibration_absent_keeps_world_invalid(manager):
    output, _ = _update(manager, [_backend_track(1, x=100, timestamp=0.0, frame_index=0)], 0.0, 0)
    assert output[0].world_valid is False
    assert output[0].anchor_world_m is None


@dataclass
class _LinearWorldGeometry:
    def image_to_world(self, point, *, image_size):
        return point[0] / image_size[0] * 10.0, point[1] / image_size[1] * 5.0


def test_valid_calibration_world_coordinates_are_used(manager):
    first, _ = _update(
        manager,
        [_backend_track(1, x=100, timestamp=0.0, frame_index=0)],
        0.0,
        0,
        geometry=_LinearWorldGeometry(),
    )
    _update(manager, [], 0.1, 1, geometry=_LinearWorldGeometry())
    second, _ = _update(
        manager,
        [_backend_track(2, x=200, timestamp=0.2, frame_index=2)],
        0.2,
        2,
        geometry=_LinearWorldGeometry(),
    )
    assert first[0].world_valid is True
    assert first[0].anchor_world_m == pytest.approx((1.12, 1.2))
    # One metre in 0.2 s exceeds the configured 3 m/s physical gate even
    # though the normalized-image fallback would still accept this move.
    assert second[0].local_track_id != first[0].local_track_id


def test_backend_id_change_is_persisted_in_tracklet_provenance(manager):
    _update(manager, [_backend_track(17, x=100, timestamp=0.0, frame_index=0)], 0.0, 0)
    _update(manager, [], 0.1, 1)
    _update(manager, [_backend_track(42, x=105, timestamp=0.2, frame_index=2)], 0.2, 2)
    tracklet = manager.flush()[0]

    assert tracklet.backend_track_ids == [17, 42]
    assert tracklet.summary_json["backend_track_ids"] == [17, 42]
    assert tracklet.summary_json["relink_count"] == 1


def test_stream_epoch_change_never_relinks_across_restart(manager):
    first, _ = _update(
        manager,
        [_backend_track(17, x=100, timestamp=0.0, frame_index=0)],
        0.0,
        0,
        epoch=0,
    )
    second, finalized = _update(
        manager,
        [_backend_track(42, x=102, timestamp=0.1, frame_index=1)],
        0.1,
        1,
        epoch=1,
    )

    assert finalized[0].local_track_id == first[0].local_track_id
    assert second[0].local_track_id != first[0].local_track_id
    assert finalized[0].summary_json["integrity_decision_reasons"][-1] == "STREAM_EPOCH_CHANGED"


def test_mask_lower_band_anchor_is_robust_to_single_low_outlier():
    anchor = GroundAnchorEstimator().estimate(
        bbox=(0.0, 0.0, 100.0, 100.0),
        mask=[[10, 10], [90, 10], [90, 80], [60, 82], [40, 81], [50, 200]],
        image_size=(200, 200),
    )
    assert anchor.method == "MASK_LOWER_BAND"
    assert anchor.pixel[0] == pytest.approx(50.0)
    assert anchor.pixel[1] < 200.0


def test_replay_benchmark_compares_raw_and_canonical_fragmentation(monkeypatch, tmp_path):
    import visionsort.tracking.integrity as integrity_module

    monkeypatch.setattr(integrity_module, "DETAILS_DIR", tmp_path / "details")
    frames = [
        ReplayFrame(
            frame_index=0,
            timestamp_global=0.0,
            stream_epoch=0,
            tracks=({"backend_track_id": 17, "bbox": [100, 100, 124, 120], "ground_truth_id": "P1"},),
        ),
        ReplayFrame(frame_index=1, timestamp_global=0.1, stream_epoch=0, tracks=()),
        ReplayFrame(
            frame_index=2,
            timestamp_global=0.2,
            stream_epoch=0,
            tracks=({"backend_track_id": 42, "bbox": [105, 100, 129, 120], "ground_truth_id": "P1"},),
        ),
    ]
    report = benchmark_replay(frames, image_size=(1000, 500))

    assert report["raw_bytetrack"]["fragmentations"] == 1
    assert report["bytetrack_plus_integrity"]["fragmentations"] == 0
    assert report["bytetrack_plus_integrity"]["relinks"] == 1
    assert report["integrity_runtime"]["average_ms_per_frame"] >= 0.0


def test_site_tracking_integrity_configuration_is_small_and_validated():
    validated = validate_site_config(
        {
            "tracking": {
                "integrity": {
                    "max_occlusion_seconds": 0.8,
                    "max_speed_m_s": 2.5,
                    "min_relink_score": 0.6,
                    "ambiguity_margin": 0.1,
                }
            }
        }
    )
    assert len(validated["tracking"]["integrity"]) == 4
    with pytest.raises(RuntimeError, match="Parametres d'integrite inconnus"):
        validate_site_config({"tracking": {"integrity": {"pixel_gate": 80}}})
