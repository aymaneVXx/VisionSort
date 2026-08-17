from __future__ import annotations

import types

import numpy as np

from visionsort.core.enums import MatchResult, ReIDAdaptationState
from visionsort.core.types import HandoffCandidate, Tracklet
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import ReIDRepository
from visionsort.reid.adaptation import AutoReIDAdapter
from visionsort.reid.encoder import l2_normalize
import visionsort.reid.adaptation as adaptation_module


def _descriptor(vector: list[float]) -> dict:
    value = l2_normalize(np.asarray(vector, dtype=np.float32)).tolist()
    return {
        "embeddings": [value, value, value],
        "aggregate_embedding": value,
        "view_count": 3,
        "view_qualities": [0.95, 0.9, 0.85],
        "model_version": "frozen-test-backbone",
        "used_mask": [False, False, False],
    }


def _tracklet(name: str, role: str, vector: list[float]) -> Tracklet:
    return Tracklet(
        tracklet_id=name,
        session_id="adapt-session",
        source_id=f"source-{role}",
        camera_id=f"camera-{role}",
        camera_role=role,
        local_track_id=1,
        started_at_local=0.0 if role == "C1" else 2.0,
        ended_at_local=1.0 if role == "C1" else 2.5,
        started_at_global=0.0 if role == "C1" else 2.0,
        ended_at_global=1.0 if role == "C1" else 2.5,
        class_name="parcel",
        first_bbox=(0.0, 0.0, 30.0, 20.0),
        last_bbox=(1.0, 0.0, 31.0, 20.0),
        avg_speed=2.0,
        last_zone_id="c1_exit" if role == "C1" else "c2_entry",
        frame_count=8,
        observation_path="details.jsonl",
        summary_json={
            "appearance_descriptor": _descriptor(vector),
            "avg_dimensions": [30.0, 20.0],
            "avg_velocity": [2.0, 0.0],
            "integrity_status": "STABLE",
            "merge_group_ids": [],
            "split_count": 0,
        },
        integrity_status="STABLE",
    )


def _candidate(
    outgoing: Tracklet,
    incoming: Tracklet,
    *,
    score: float,
) -> HandoffCandidate:
    return HandoffCandidate(
        from_tracklet_id=outgoing.tracklet_id,
        to_tracklet_id=incoming.tracklet_id,
        score=score,
        result=MatchResult.MATCHED,
        reasons=["test"],
        features={
            "topology": "C1->C2",
            "temporal": 1.0,
            "dimensions": 0.95,
            "zone": 1.0,
            "speed": 0.95,
            "integrity": 1.0,
            "reid": 0.9,
        },
        gate_evidence={
            "edge_authorized": True,
            "identity_usable": True,
        },
        model_version="active-test-model",
    )


def _add_training_pairs(repo: ReIDRepository, *, count: int = 4) -> None:
    for index in range(count):
        positive_left = _descriptor([1.0, 0.1 * index, 0.0, 0.0])
        positive_right = _descriptor([1.0, 0.1 * index + 0.02, 0.0, 0.0])
        negative_left = _descriptor([0.0, 1.0, 0.1 * index, 0.0])
        repo.add_pair(
            session_id="adapt-session",
            edge_key="C1->C2",
            label="POSITIVE",
            left_tracklet_id=f"positive-left-{index}",
            right_tracklet_id=f"incoming-{index}",
            left_descriptor=positive_left,
            right_descriptor=positive_right,
            metadata={},
            dataset_version="reid-pairs-v1",
        )
        repo.add_pair(
            session_id="adapt-session",
            edge_key="C1->C2",
            label="HARD_NEGATIVE",
            left_tracklet_id=f"negative-left-{index}",
            right_tracklet_id=f"incoming-{index}",
            left_descriptor=negative_left,
            right_descriptor=positive_right,
            metadata={},
            dataset_version="reid-pairs-v1",
        )


def test_reliable_match_generates_positive_and_hard_negative_only(tmp_path):
    db = VisionSortDB(tmp_path / "pairs.db")
    db.initialize()
    adapter = AutoReIDAdapter(
        db,
        min_positive_pairs=10,
        min_hard_negatives=10,
    )
    outgoing = _tracklet("outgoing", "C1", [1, 0, 0, 0])
    competitor = _tracklet("competitor", "C1", [0.9, 0.1, 0, 0])
    incoming = _tracklet("incoming", "C2", [1, 0.02, 0, 0])
    selected = _candidate(outgoing, incoming, score=0.96)
    negative = _candidate(competitor, incoming, score=0.60)

    assert adapter.observe_handoff(
        outgoing=outgoing,
        incoming=incoming,
        selected=selected,
        candidates=[selected, negative],
        tracklets_by_id={
            outgoing.tracklet_id: outgoing,
            competitor.tracklet_id: competitor,
        },
        result=MatchResult.MATCHED,
        ambiguity_margin=0.08,
    )
    assert adapter.repo.pair_counts() == {"POSITIVE": 1, "HARD_NEGATIVE": 1}

    ambiguous_incoming = _tracklet("ambiguous", "C2", [1, 0.03, 0, 0])
    assert not adapter.observe_handoff(
        outgoing=outgoing,
        incoming=ambiguous_incoming,
        selected=_candidate(outgoing, ambiguous_incoming, score=0.90),
        candidates=[],
        tracklets_by_id={outgoing.tracklet_id: outgoing},
        result=MatchResult.AMBIGUOUS,
        ambiguity_margin=0.08,
    )
    assert adapter.repo.pair_counts() == {"POSITIVE": 1, "HARD_NEGATIVE": 1}


def test_projection_head_training_changes_only_the_small_projection(tmp_path):
    db = VisionSortDB(tmp_path / "train.db")
    db.initialize()
    repo = ReIDRepository(db)
    _add_training_pairs(repo)
    adapter = AutoReIDAdapter(
        db,
        min_positive_pairs=2,
        min_hard_negatives=2,
        epochs=3,
    )
    pairs = repo.list_pairs(dataset_version="reid-pairs-v1")
    matrix, bias = adapter._train_projection(pairs)

    assert matrix.shape == (4, 4)
    assert bias.shape == (4,)
    assert not np.allclose(matrix, np.eye(4, dtype=np.float32))
    assert adapter.state == ReIDAdaptationState.READY


def _metrics(*, accuracy: float, separation: float, false_match: float, switches: int):
    return {
        "positive_pair_retrieval_accuracy": accuracy,
        "hard_negative_separation": separation,
        "handoff_accuracy": accuracy,
        "false_match_rate": false_match,
        "ambiguous_rate": 0.0,
        "global_id_switches": switches,
    }


def _controlled_adapter(
    db: VisionSortDB,
    tmp_path,
    monkeypatch,
    *,
    active_metrics,
    candidate_metrics,
) -> AutoReIDAdapter:
    repo = ReIDRepository(db)
    _add_training_pairs(repo)
    monkeypatch.setattr(adaptation_module, "MODELS_DIR", tmp_path / "models")
    adapter = AutoReIDAdapter(
        db,
        min_positive_pairs=2,
        min_hard_negatives=2,
        epochs=1,
    )
    monkeypatch.setattr(
        adapter,
        "_train_projection",
        lambda pairs: (np.eye(4, dtype=np.float32), np.zeros(4, dtype=np.float32)),
    )
    metrics = iter((active_metrics, candidate_metrics))

    def controlled_evaluate(self, pairs, projection):
        return next(metrics)

    monkeypatch.setattr(
        adapter,
        "evaluate",
        types.MethodType(controlled_evaluate, adapter),
    )
    return adapter


def test_inferior_candidate_is_rejected_and_frozen(tmp_path, monkeypatch):
    db = VisionSortDB(tmp_path / "reject.db")
    db.initialize()
    adapter = _controlled_adapter(
        db,
        tmp_path,
        monkeypatch,
        active_metrics=_metrics(accuracy=0.90, separation=0.50, false_match=0.05, switches=0),
        candidate_metrics=_metrics(accuracy=0.80, separation=0.40, false_match=0.10, switches=1),
    )
    result = adapter.run_once()
    candidate = db.fetch_one(
        "SELECT status, is_active FROM model_registry WHERE id = ?",
        (result["candidate_model_id"],),
    )

    assert result["outcome"] == "REJECTED"
    assert candidate["status"] == "REJECTED"
    assert candidate["is_active"] == 0
    assert adapter.state == ReIDAdaptationState.FROZEN


def test_superior_candidate_is_promoted_and_can_rollback(tmp_path, monkeypatch):
    db = VisionSortDB(tmp_path / "promote.db")
    db.initialize()
    adapter = _controlled_adapter(
        db,
        tmp_path,
        monkeypatch,
        active_metrics=_metrics(accuracy=0.70, separation=0.20, false_match=0.10, switches=1),
        candidate_metrics=_metrics(accuracy=0.95, separation=0.60, false_match=0.02, switches=0),
    )
    result = adapter.run_once()
    active = db.fetch_one(
        "SELECT id, status FROM model_registry WHERE task = 'reid_multicamera' AND is_active = 1"
    )

    assert result["outcome"] == "PROMOTED"
    assert active["id"] == result["candidate_model_id"]
    assert active["status"] == "CHAMPION"
    assert adapter.state == ReIDAdaptationState.FROZEN

    rolled_back = adapter.rollback()
    restored = db.fetch_one(
        "SELECT id FROM model_registry WHERE task = 'reid_multicamera' AND is_active = 1"
    )
    assert rolled_back == "parcel_reid_mobilenet_v3_small_v1"
    assert restored["id"] == rolled_back
