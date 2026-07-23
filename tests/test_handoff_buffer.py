from dataclasses import asdict

from visionsort.core.config import AppConfig, DEFAULT_CONFIG
from visionsort.core.types import Tracklet
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import (
    EventRepository,
    HandoffHypothesisRepository,
    TrackingRepository,
)
from visionsort.runtime.supervisor import RuntimeSupervisor
from visionsort.tracking.engine import GlobalParcelTracker
from visionsort.tracking.handoffs import PendingHandoffBuffer


TOPOLOGY = [
    {
        "from_role": "C1",
        "to_role": "C2",
        "min_transit_s": 0.5,
        "max_transit_s": 10.0,
    },
    {
        "from_role": "C2",
        "to_role": "C3",
        "min_transit_s": 0.2,
        "max_transit_s": 12.0,
    },
]


def _tracklet(
    tracklet_id: str,
    role: str,
    *,
    session_id: str = "session-a",
    started: float = 0.0,
    ended: float = 1.0,
    width: float = 12.0,
    appearance: list[float] | None = None,
) -> Tracklet:
    return Tracklet(
        tracklet_id=tracklet_id,
        session_id=session_id,
        source_id=f"source-{role}",
        camera_id=f"camera-{role}",
        camera_role=role,
        local_track_id=1,
        started_at_local=started,
        ended_at_local=ended,
        started_at_global=started,
        ended_at_global=ended,
        class_name="parcel",
        first_bbox=(0.0, 0.0, width, 10.0),
        last_bbox=(0.0, 0.0, width, 10.0),
        avg_speed=4.0,
        last_zone_id=f"{role.lower()}_exit",
        frame_count=3,
        observation_path="tracklet.jsonl",
        summary_json={
            "first_bbox": [0.0, 0.0, width, 10.0],
            "last_bbox": [0.0, 0.0, width, 10.0],
            "avg_dimensions": [width, 10.0],
            "avg_velocity": [4.0, 0.0],
            "first_zone_id": f"{role.lower()}_entry",
            "last_zone_id": f"{role.lower()}_exit",
            "appearance_embedding": appearance,
        },
        model_id="demo",
        tracker_id="greedy_iou",
    )


def _supervisor(db: VisionSortDB) -> RuntimeSupervisor:
    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.db = db
    supervisor.config = AppConfig(values=DEFAULT_CONFIG)
    supervisor.tracking_repo = TrackingRepository(db)
    supervisor.hypothesis_repo = HandoffHypothesisRepository(db)
    supervisor.event_repo = EventRepository(db)
    supervisor.global_tracker = GlobalParcelTracker(TOPOLOGY, {})
    return supervisor


def test_pending_buffer_batches_reordered_arrivals_after_window(tmp_path):
    db = VisionSortDB(tmp_path / "buffer.db")
    db.initialize()
    buffer = PendingHandoffBuffer(db, TOPOLOGY, window_seconds=0.75)
    later = asdict(_tracklet("incoming-later", "C2", started=2.6, ended=3.1))
    earlier = asdict(_tracklet("incoming-earlier", "C2", started=2.5, ended=3.0))

    buffer.add(later, received_at=100.0)
    buffer.add(earlier, received_at=100.3)

    assert buffer.pop_ready_batches(now=100.4) == []
    batches = buffer.pop_ready_batches(now=100.8)
    assert len(batches) == 1
    assert [item["tracklet_id"] for item in batches[0][1]] == [
        "incoming-earlier",
        "incoming-later",
    ]


def test_pending_buffer_is_persistent_bounded_and_session_isolated(tmp_path):
    db = VisionSortDB(tmp_path / "persistent.db")
    db.initialize()
    buffer = PendingHandoffBuffer(
        db, TOPOLOGY, window_seconds=1.0, max_items=2
    )
    buffer.add(asdict(_tracklet("a", "C2", session_id="session-a")), received_at=10.0)
    buffer.add(asdict(_tracklet("b", "C2", session_id="session-b")), received_at=10.1)
    evicted = buffer.add(
        asdict(_tracklet("c", "C2", session_id="session-a")), received_at=10.2
    )

    assert len(evicted) == 1
    assert buffer.pending_count() == 2
    restored = PendingHandoffBuffer(
        db, TOPOLOGY, window_seconds=1.0, max_items=2
    )
    batches = restored.pop_ready_batches(force=True, session_id="session-a")
    assert all(session_id == "session-a" for session_id, _ in batches)
    assert restored.pending_count("session-b") == 1


def test_ambiguous_handoff_is_persisted_and_human_resolvable(tmp_path):
    db = VisionSortDB(tmp_path / "hypothesis.db")
    db.initialize()
    supervisor = _supervisor(db)
    supervisor.handle_tracklets(
        [
            asdict(_tracklet("out-a", "C1", width=12.0)),
            asdict(_tracklet("out-b", "C1", width=12.1)),
            asdict(
                _tracklet(
                    "incoming",
                    "C2",
                    started=2.5,
                    ended=3.0,
                    width=12.05,
                )
            ),
        ]
    )

    hypotheses = supervisor.hypothesis_repo.pending("session-a")
    assert len(hypotheses) == 1
    assert hypotheses[0]["status"] == "PENDING"
    incoming = db.fetch_one(
        "SELECT parcel_id, match_result FROM tracklets WHERE tracklet_id = 'incoming'"
    )
    assert incoming is not None
    assert incoming["parcel_id"] is None
    assert incoming["match_result"] == "AMBIGUOUS"

    parcel_id = supervisor.resolve_handoff_hypothesis(
        hypotheses[0]["id"], "out-a", actor="pytest"
    )
    resolved = supervisor.hypothesis_repo.get(hypotheses[0]["id"])
    incoming = db.fetch_one(
        "SELECT parcel_id, match_result FROM tracklets WHERE tracklet_id = 'incoming'"
    )
    assert resolved is not None and resolved["status"] == "RESOLVED"
    assert incoming is not None and incoming["parcel_id"] == parcel_id
    assert incoming["match_result"] == "MATCHED"


def test_later_camera_evidence_can_resolve_pending_ambiguity(tmp_path):
    db = VisionSortDB(tmp_path / "later-evidence.db")
    db.initialize()
    supervisor = _supervisor(db)
    supervisor.handle_tracklets(
        [
            asdict(
                _tracklet(
                    "out-a", "C1", width=12.0, appearance=[1.0, 0.0]
                )
            ),
            asdict(
                _tracklet(
                    "out-b", "C1", width=12.1, appearance=[0.0, 1.0]
                )
            ),
            asdict(
                _tracklet(
                    "incoming",
                    "C2",
                    started=2.5,
                    ended=3.0,
                    width=12.05,
                )
            ),
        ]
    )
    hypothesis = supervisor.hypothesis_repo.pending("session-a")[0]

    supervisor.handle_tracklets(
        [
            asdict(
                _tracklet(
                    "later",
                    "C3",
                    started=4.0,
                    ended=4.5,
                    width=12.0,
                    appearance=[1.0, 0.0],
                )
            )
        ]
    )

    resolved = supervisor.hypothesis_repo.get(hypothesis["id"])
    assert resolved is not None
    assert resolved["status"] == "RESOLVED"
