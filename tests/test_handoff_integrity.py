from dataclasses import asdict

import numpy as np
import pytest

from visionsort.core.config import AppConfig, DEFAULT_CONFIG
from visionsort.core.enums import MatchResult
from visionsort.core.types import HandoffCandidate, Tracklet
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import (
    EventRepository,
    HandoffHypothesisRepository,
    TrackingRepository,
)
from visionsort.runtime.supervisor import RuntimeSupervisor
from visionsort.tracking.engine import (
    GlobalParcelTracker,
    solve_handoff_assignment,
)


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
    started: float = 0.0,
    ended: float = 1.0,
    width: float = 12.0,
) -> Tracklet:
    return Tracklet(
        tracklet_id=tracklet_id,
        session_id="session-a",
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
        frame_count=2,
        observation_path="tracklet.jsonl",
        summary_json={
            "avg_dimensions": [width, 10.0],
            "avg_velocity": [4.0, 0.0],
            "first_zone_id": f"{role.lower()}_entry",
            "last_zone_id": f"{role.lower()}_exit",
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


def _ambiguous_supervisor(tmp_path):
    db = VisionSortDB(tmp_path / "handoff.db")
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
    hypothesis = supervisor.hypothesis_repo.pending("session-a")[0]
    return db, supervisor, hypothesis


def test_transaction_refuses_consumed_candidate_without_partial_update(
    tmp_path,
):
    db, supervisor, hypothesis = _ambiguous_supervisor(tmp_path)
    outgoing = db.fetch_one(
        "SELECT parcel_id FROM tracklets WHERE tracklet_id = 'out-a'"
    )
    parcel_id = str(outgoing["parcel_id"])
    consumed = _tracklet(
        "already-consumed", "C2", started=2.4, ended=2.9
    )
    supervisor.tracking_repo.upsert_tracklet(
        consumed,
        parcel_id=parcel_id,
        match_result=MatchResult.MATCHED,
    )
    db.execute(
        """
        UPDATE global_parcels
        SET current_tracklet_id = ?, last_camera_id = ?,
            last_seen_at = ?
        WHERE parcel_id = ?
        """,
        (
            consumed.tracklet_id,
            consumed.camera_id,
            consumed.ended_at_global,
            parcel_id,
        ),
    )

    with pytest.raises(RuntimeError, match="déjà progressé"):
        supervisor.resolve_handoff_hypothesis(
            hypothesis["id"], "out-a", actor="pytest"
        )

    incoming = db.fetch_one(
        """
        SELECT parcel_id, match_result FROM tracklets
        WHERE tracklet_id = 'incoming'
        """
    )
    parcel = db.fetch_one(
        """
        SELECT current_tracklet_id FROM global_parcels
        WHERE parcel_id = ?
        """,
        (parcel_id,),
    )
    unresolved = supervisor.hypothesis_repo.get(hypothesis["id"])
    audit = supervisor.hypothesis_repo.list_audit(
        hypothesis_id=hypothesis["id"]
    )
    assert incoming["parcel_id"] is None
    assert incoming["match_result"] == "AMBIGUOUS"
    assert parcel["current_tracklet_id"] == "already-consumed"
    assert unresolved["status"] == "PENDING"
    assert audit[0]["result"] == "REFUSED"
    assert "déjà progressé" in audit[0]["reason"]


def test_transaction_rejects_candidate_outside_hypothesis(tmp_path):
    db, supervisor, hypothesis = _ambiguous_supervisor(tmp_path)

    with pytest.raises(RuntimeError, match="n'appartient pas"):
        supervisor.resolve_handoff_hypothesis(
            hypothesis["id"], "not-a-candidate", actor="pytest"
        )

    incoming = db.fetch_one(
        "SELECT parcel_id FROM tracklets WHERE tracklet_id = 'incoming'"
    )
    assert incoming["parcel_id"] is None
    assert (
        supervisor.hypothesis_repo.get(hypothesis["id"])["status"]
        == "PENDING"
    )
    assert (
        supervisor.hypothesis_repo.list_audit(
            hypothesis_id=hypothesis["id"]
        )[0]["result"]
        == "REFUSED"
    )


def test_lap_assignment_beats_greedy_and_respects_threshold():
    scores = np.asarray([[0.95, 0.80], [0.80, 0.10]])

    assignments, total = solve_handoff_assignment(
        scores, minimum_score=0.48
    )

    greedy_total = 0.95
    assert assignments == {0: 1, 1: 0}
    assert total == pytest.approx(1.60)
    assert total > greedy_total
    unmatched, unmatched_total = solve_handoff_assignment(
        np.asarray([[0.47]]), minimum_score=0.48
    )
    assert unmatched == {}
    assert unmatched_total == 0.0


def test_global_tracker_uses_optimal_matrix_and_keeps_close_swap_ambiguous():
    tracker = GlobalParcelTracker(TOPOLOGY, {})
    parcel_a = tracker.process_tracklet(_tracklet("out-a", "C1"))[0]
    parcel_b = tracker.process_tracklet(_tracklet("out-b", "C1"))[0]
    score_map = {
        ("in-a", "out-a"): 0.95,
        ("in-a", "out-b"): 0.80,
        ("in-b", "out-a"): 0.80,
        ("in-b", "out-b"): 0.10,
    }

    def score(outgoing: Tracklet, incoming: Tracklet):
        value = score_map[(incoming.tracklet_id, outgoing.tracklet_id)]
        return HandoffCandidate(
            from_tracklet_id=outgoing.tracklet_id,
            to_tracklet_id=incoming.tracklet_id,
            score=value,
            result=MatchResult.UNMATCHED,
            reasons=["test-score"],
        )

    tracker._score_candidate = score  # type: ignore[method-assign]
    outcomes = tracker.process_tracklets(
        [
            _tracklet("in-a", "C2", started=2.5, ended=3.0),
            _tracklet("in-b", "C2", started=2.6, ended=3.1),
        ]
    )
    assert outcomes[0][0] == parcel_b
    assert outcomes[1][0] == parcel_a
    assert all(outcome[1] == MatchResult.MATCHED for outcome in outcomes)

    close_tracker = GlobalParcelTracker(TOPOLOGY, {})
    close_tracker.process_tracklet(_tracklet("close-out-a", "C1"))
    close_tracker.process_tracklet(_tracklet("close-out-b", "C1"))

    def close_score(outgoing: Tracklet, incoming: Tracklet):
        same_suffix = incoming.tracklet_id.endswith(
            outgoing.tracklet_id[-1]
        )
        return HandoffCandidate(
            from_tracklet_id=outgoing.tracklet_id,
            to_tracklet_id=incoming.tracklet_id,
            score=0.85 if same_suffix else 0.84,
            result=MatchResult.UNMATCHED,
            reasons=["close-score"],
        )

    close_tracker._score_candidate = close_score  # type: ignore[method-assign]
    close_outcomes = close_tracker.process_tracklets(
        [
            _tracklet("close-in-a", "C2", started=2.5, ended=3.0),
            _tracklet("close-in-b", "C2", started=2.6, ended=3.1),
        ]
    )
    assert all(
        outcome[1] == MatchResult.AMBIGUOUS
        for outcome in close_outcomes
    )
