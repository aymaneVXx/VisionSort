from __future__ import annotations

import hashlib
import json
import threading
import uuid
from typing import Any, Callable

import numpy as np

from visionsort.core.enums import MatchResult, ModelStatus, ReIDAdaptationState
from visionsort.core.paths import MODELS_DIR, ROOT_DIR
from visionsort.core.types import HandoffCandidate, Tracklet
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import EventRepository, ReIDRepository
from visionsort.deployment.registry import (
    create_activation_history,
    finish_activation_history,
    mark_previous_activation,
    promote_model,
    rollback_to_previous_active,
    set_model_status,
)
from visionsort.inference.engine import resolve_model_artifact
from visionsort.reid.encoder import ProjectionHead, descriptor_embeddings


REID_TASK = "reid_multicamera"


class AutoReIDAdapter:
    """One-shot projection-head adaptation from conservative pseudo-pairs."""

    def __init__(
        self,
        db: VisionSortDB,
        *,
        event_repo: EventRepository | None = None,
        enabled: bool = True,
        min_positive_pairs: int = 32,
        min_hard_negatives: int = 16,
        dataset_version: str = "reid-pairs-v1",
        epochs: int = 24,
        on_promoted: Callable[[ProjectionHead], None] | None = None,
    ) -> None:
        self.db = db
        self.repo = ReIDRepository(db)
        self.event_repo = event_repo
        self.enabled = bool(enabled)
        self.min_positive_pairs = max(2, int(min_positive_pairs))
        self.min_hard_negatives = max(1, int(min_hard_negatives))
        self.dataset_version = str(dataset_version)
        self.epochs = max(1, int(epochs))
        self.on_promoted = on_promoted
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.state = (
            ReIDAdaptationState.BOOTSTRAP
            if self.enabled
            else ReIDAdaptationState.FROZEN
        )
        if self.enabled:
            self._refresh_collection_state(emit=False)

    def active_projection(self) -> ProjectionHead:
        row = self.db.fetch_one(
            """
            SELECT * FROM model_registry
            WHERE task = ? AND is_active = 1 LIMIT 1
            """,
            (REID_TASK,),
        )
        if row is None or str(row["backend"]) != "visionsort_reid_projection":
            version = str(row["id"]) if row is not None else "reid-projection-identity-v1"
            return ProjectionHead(version=version)
        model = dict(row)
        artifact, _ = resolve_model_artifact(model)
        return ProjectionHead.load(artifact, version=str(model["id"]))

    def observe_handoff(
        self,
        *,
        outgoing: Tracklet,
        incoming: Tracklet,
        selected: HandoffCandidate,
        candidates: list[HandoffCandidate],
        tracklets_by_id: dict[str, Tracklet],
        result: MatchResult,
        ambiguity_margin: float,
    ) -> bool:
        if not self.enabled or self.state == ReIDAdaptationState.FROZEN:
            return False
        if result != MatchResult.MATCHED or not self._extremely_reliable(
            outgoing,
            incoming,
            selected,
            candidates,
            ambiguity_margin=ambiguity_margin,
        ):
            return False
        left_descriptor = _descriptor(outgoing)
        right_descriptor = _descriptor(incoming)
        if left_descriptor is None or right_descriptor is None:
            return False
        topology = str(selected.features.get("topology") or "unknown")
        competitor_physical_scores = [
            self._physical_score(item)
            for item in candidates
            if item.from_tracklet_id != selected.from_tracklet_id
        ]
        physical_score = self._physical_score(selected)
        physical_margin = physical_score - max(
            competitor_physical_scores,
            default=0.0,
        )
        metadata = {
            "production_handoff_score": selected.score,
            "physical_score": physical_score,
            "physical_margin": physical_margin,
            "reid_similarity": selected.features.get("reid"),
            "model_version": selected.model_version,
            "pseudo_label_policy": "PR3_PHYSICAL_ONLY_V2",
            "replay_evidence": _replay_evidence(outgoing, incoming, selected),
        }
        inserted = self.repo.add_pair(
            session_id=incoming.session_id,
            edge_key=topology,
            label="POSITIVE",
            left_tracklet_id=outgoing.tracklet_id,
            right_tracklet_id=incoming.tracklet_id,
            left_descriptor=left_descriptor,
            right_descriptor=right_descriptor,
            metadata=metadata,
            dataset_version=self.dataset_version,
        )
        for competitor in candidates:
            if competitor.from_tracklet_id == selected.from_tracklet_id:
                continue
            negative = tracklets_by_id.get(competitor.from_tracklet_id)
            negative_descriptor = _descriptor(negative) if negative is not None else None
            if negative_descriptor is None:
                continue
            # Only plausible same-edge competitors become hard negatives.
            if (
                competitor.features.get("topology") != selected.features.get("topology")
                or float(competitor.features.get("dimensions") or 0.0) < 0.60
            ):
                continue
            self.repo.add_pair(
                session_id=incoming.session_id,
                edge_key=topology,
                label="HARD_NEGATIVE",
                left_tracklet_id=negative.tracklet_id,
                right_tracklet_id=incoming.tracklet_id,
                left_descriptor=negative_descriptor,
                right_descriptor=right_descriptor,
                metadata={
                    **metadata,
                    "competitor_score": competitor.score,
                    "competitor_physical_score": self._physical_score(competitor),
                    "selected_from_tracklet_id": selected.from_tracklet_id,
                    "replay_evidence": _replay_evidence(
                        negative,
                        incoming,
                        competitor,
                    ),
                },
                dataset_version=self.dataset_version,
            )
        self._refresh_collection_state(emit=True)
        return inserted

    def maybe_start_async(self) -> bool:
        with self._lock:
            if (
                not self.enabled
                or self.state != ReIDAdaptationState.READY
                or (self._thread is not None and self._thread.is_alive())
            ):
                return False
            self._thread = threading.Thread(
                target=self.run_once,
                name="visionsort-reid-adapter",
                daemon=True,
            )
            self._thread.start()
            return True

    def run_once(self) -> dict[str, Any]:
        if not self.enabled:
            return {"outcome": "DISABLED"}
        pairs = self.repo.list_pairs(dataset_version=self.dataset_version)
        counts = self.repo.pair_counts(dataset_version=self.dataset_version)
        if (
            counts["POSITIVE"] < self.min_positive_pairs
            or counts["HARD_NEGATIVE"] < self.min_hard_negatives
        ):
            self.state = ReIDAdaptationState.COLLECTING
            return {"outcome": "INSUFFICIENT_DATA", "counts": counts}
        active_row = self.db.fetch_one(
            "SELECT id FROM model_registry WHERE task = ? AND is_active = 1",
            (REID_TASK,),
        )
        active_model_id = str(active_row["id"]) if active_row else None
        run_id = self.repo.create_run(
            state=ReIDAdaptationState.TRAINING.value,
            dataset_version=self.dataset_version,
            active_model_id=active_model_id,
        )
        self.state = ReIDAdaptationState.TRAINING
        self._event("REID_TRAINING_STARTED", {"run_id": run_id, "counts": counts})
        try:
            training, heldout = _deterministic_split(pairs)
            adapter_down, adapter_up, bias = self._train_projection(training)
            candidate_version = f"parcel-reid-projection-{uuid.uuid4().hex[:12]}"
            candidate = ProjectionHead(
                bias=bias,
                adapter_down=adapter_down,
                adapter_up=adapter_up,
                version=candidate_version,
            )
            self.state = ReIDAdaptationState.VALIDATING
            self.repo.update_run(run_id, state=self.state.value)
            active_metrics = self.evaluate(heldout, self.active_projection())
            candidate_metrics = self.evaluate(heldout, candidate)
            better = self._clearly_better(active_metrics, candidate_metrics)
            metrics: dict[str, Any] = {
                **candidate_metrics,
                "active_baseline": active_metrics,
                "promotion_eligible": better,
                "promotion_criteria": {
                    "minimum_replay_episodes": 2,
                    "minimum_handoff_gain": 0.02,
                    "no_false_match_regression": True,
                    "no_id_switch_regression": True,
                    "minimum_separation_gain": 0.02,
                },
                "test": {
                    "status": "COMPLETED",
                    "frozen": True,
                    "dataset_version": self.dataset_version,
                    "pair_count": len(heldout),
                },
            }
            candidate_model_id = self._register_candidate(
                candidate, metrics=metrics, parent_model_id=active_model_id
            )
            self._event(
                "REID_CANDIDATE_VALIDATED",
                {
                    "run_id": run_id,
                    "candidate_model_id": candidate_model_id,
                    "metrics": metrics,
                    "better": better,
                },
            )
            if better:
                promote_model(self.db, candidate_model_id)
                mark_previous_activation(
                    self.db,
                    task=REID_TASK,
                    model_id=active_model_id,
                    status="SUPERSEDED",
                )
                generation_row = self.db.fetch_one(
                    """
                    SELECT COALESCE(MAX(routing_generation), 0) AS generation
                    FROM model_activation_history WHERE task = ?
                    """,
                    (REID_TASK,),
                )
                activation_id = create_activation_history(
                    self.db,
                    task=REID_TASK,
                    previous_model_id=active_model_id,
                    activated_model_id=candidate_model_id,
                    routing_generation=int(generation_row["generation"] or 0) + 1,
                    actor="auto_reid_adapter",
                    reason="deterministic held-out validation passed",
                    metadata={"run_id": run_id, "dataset_version": self.dataset_version},
                )
                finish_activation_history(
                    self.db,
                    activation_id,
                    status="ACTIVE",
                    runtime_applied=True,
                    metadata={"projection_loaded_by": "GlobalParcelTracker"},
                )
                self.state = ReIDAdaptationState.PROMOTED
                outcome = "PROMOTED"
                if self.on_promoted is not None:
                    self.on_promoted(candidate)
                self._event(
                    "REID_MODEL_PROMOTED",
                    {"run_id": run_id, "model_id": candidate_model_id, "metrics": metrics},
                )
            else:
                set_model_status(
                    self.db, candidate_model_id, ModelStatus.REJECTED.value
                )
                self.state = ReIDAdaptationState.REJECTED
                outcome = "REJECTED"
                self._event(
                    "REID_MODEL_REJECTED",
                    {"run_id": run_id, "model_id": candidate_model_id, "metrics": metrics},
                )
            result = {
                "outcome": outcome,
                "run_id": run_id,
                "candidate_model_id": candidate_model_id,
                "metrics": metrics,
            }
            self.repo.update_run(
                run_id,
                state=self.state.value,
                candidate_model_id=candidate_model_id,
                metrics=result,
            )
            self.state = ReIDAdaptationState.FROZEN
            self.repo.update_run(
                run_id,
                state=self.state.value,
                candidate_model_id=candidate_model_id,
                metrics=result,
            )
            return result
        except Exception as exc:
            self.state = ReIDAdaptationState.REJECTED
            self.repo.update_run(
                run_id,
                state=self.state.value,
                error_text=str(exc),
            )
            self._event(
                "REID_MODEL_REJECTED",
                {"run_id": run_id, "error": str(exc)},
                severity="error",
            )
            self.state = ReIDAdaptationState.FROZEN
            self.repo.update_run(
                run_id,
                state=self.state.value,
                error_text=str(exc),
            )
            return {"outcome": "REJECTED", "run_id": run_id, "error": str(exc)}

    def rollback(self) -> str:
        model_id = rollback_to_previous_active(
            self.db, task=REID_TASK, apply=True
        )
        projection = self.active_projection()
        if self.on_promoted is not None:
            self.on_promoted(projection)
        return model_id

    def shutdown(self, timeout: float = 3.0) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))

    def evaluate(
        self, pairs: list[dict[str, Any]], projection: ProjectionHead
    ) -> dict[str, float | int]:
        """Replay stored physical episodes through production handoff matching."""
        from visionsort.tracking.engine import GlobalParcelTracker

        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for pair in pairs:
            replay = (pair.get("metadata") or {}).get("replay_evidence")
            if not isinstance(replay, dict):
                continue
            key = (str(pair.get("session_id") or ""), str(pair.get("edge_key") or ""))
            groups.setdefault(key, []).append(pair)

        episode_count = 0
        correct = 0
        false_matches = 0
        ambiguous = 0
        retrieval_results: list[bool] = []
        separations: list[float] = []
        for rows in groups.values():
            first_replay = rows[0]["metadata"]["replay_evidence"]
            edge = dict(first_replay.get("edge") or {})
            if not edge:
                continue
            outgoing_by_id: dict[str, Tracklet] = {}
            incoming_by_id: dict[str, Tracklet] = {}
            positive_by_incoming: dict[str, set[str]] = {}
            labels_by_pair: dict[tuple[str, str], str] = {}
            zones_by_role: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                replay = row["metadata"]["replay_evidence"]
                outgoing = _tracklet_from_replay(
                    replay.get("outgoing"), row["left_descriptor"]
                )
                incoming = _tracklet_from_replay(
                    replay.get("incoming"), row["right_descriptor"]
                )
                if outgoing is None or incoming is None:
                    continue
                outgoing_by_id[outgoing.tracklet_id] = outgoing
                incoming_by_id[incoming.tracklet_id] = incoming
                label = str(row["label"])
                labels_by_pair[(outgoing.tracklet_id, incoming.tracklet_id)] = label
                if label == "POSITIVE":
                    positive_by_incoming.setdefault(incoming.tracklet_id, set()).add(
                        outgoing.tracklet_id
                    )
                _append_replay_zones(zones_by_role, outgoing, incoming)
            incoming_values = sorted(
                (
                    item
                    for item in incoming_by_id.values()
                    if item.tracklet_id in positive_by_incoming
                ),
                key=lambda item: item.tracklet_id,
            )
            if not outgoing_by_id or not incoming_values:
                continue
            tracker = GlobalParcelTracker(
                [edge],
                {},
                zones_by_role=zones_by_role,
                minimum_score=0.48,
                ambiguity_margin=0.08,
                reid_enabled=True,
                projection=projection,
            )
            for outgoing in sorted(
                outgoing_by_id.values(), key=lambda item: item.tracklet_id
            ):
                tracker.process_tracklet(outgoing)
            decisions = tracker.process_tracklets(incoming_values)
            for incoming, decision in zip(incoming_values, decisions):
                episode_count += 1
                _parcel_id, result, _reasons, selected = decision
                positives = positive_by_incoming[incoming.tracklet_id]
                if result == MatchResult.AMBIGUOUS:
                    ambiguous += 1
                elif (
                    result == MatchResult.MATCHED
                    and selected is not None
                    and selected.from_tracklet_id in positives
                ):
                    correct += 1
                elif result == MatchResult.MATCHED:
                    false_matches += 1
                candidates = tracker.last_candidate_sets.get(incoming.tracklet_id, [])
                positive_reid = [
                    float(item.features["reid"])
                    for item in candidates
                    if item.from_tracklet_id in positives
                    and item.features.get("reid") is not None
                ]
                negative_reid = [
                    float(item.features["reid"])
                    for item in candidates
                    if labels_by_pair.get(
                        (item.from_tracklet_id, incoming.tracklet_id)
                    )
                    == "HARD_NEGATIVE"
                    and item.features.get("reid") is not None
                ]
                if positive_reid:
                    best_positive = max(positive_reid)
                    best_negative = max(negative_reid, default=0.0)
                    retrieval_results.append(best_positive > best_negative)
                    if negative_reid:
                        separations.append(best_positive - best_negative)
        denominator = max(episode_count, 1)
        return {
            "positive_pair_retrieval_accuracy": (
                float(np.mean(retrieval_results)) if retrieval_results else 0.0
            ),
            "hard_negative_separation": (
                float(np.mean(separations)) if separations else 0.0
            ),
            "handoff_accuracy": correct / denominator,
            "false_match_rate": false_matches / denominator,
            "ambiguous_rate": ambiguous / denominator,
            "global_id_switches": int(false_matches),
            "replay_episode_count": int(episode_count),
        }

    def _train_projection(
        self, pairs: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        examples: list[tuple[np.ndarray, np.ndarray, float]] = []
        for pair in pairs:
            left = _aggregate(pair["left_descriptor"])
            right = _aggregate(pair["right_descriptor"])
            if left is None or right is None:
                continue
            examples.append(
                (left, right, 1.0 if pair["label"] == "POSITIVE" else -1.0)
            )
        if not examples:
            raise RuntimeError("Aucune paire ReID exploitable pour l'adaptation.")
        dimension = int(examples[0][0].shape[0])
        torch.manual_seed(17)
        rank = min(16, max(1, dimension // 4))
        down = torch.nn.Linear(dimension, rank, bias=False)
        up = torch.nn.Linear(rank, dimension, bias=True)
        with torch.no_grad():
            torch.nn.init.normal_(down.weight, mean=0.0, std=0.02)
            up.weight.zero_()
            up.bias.zero_()
        parameters = [*down.parameters(), *up.parameters()]
        optimizer = torch.optim.AdamW(parameters, lr=2.0e-3, weight_decay=1.0e-4)
        loss_function = torch.nn.CosineEmbeddingLoss(margin=0.25)
        left = torch.tensor(np.stack([item[0] for item in examples]), dtype=torch.float32)
        right = torch.tensor(np.stack([item[1] for item in examples]), dtype=torch.float32)
        labels = torch.tensor([item[2] for item in examples], dtype=torch.float32)
        down.train()
        up.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            left_projected = torch.nn.functional.normalize(
                left + up(torch.tanh(down(left))), dim=1
            )
            right_projected = torch.nn.functional.normalize(
                right + up(torch.tanh(down(right))), dim=1
            )
            loss = loss_function(left_projected, right_projected, labels)
            loss = loss + 1.0e-5 * (
                torch.mean(down.weight**2) + torch.mean(up.weight**2)
            )
            loss.backward()
            optimizer.step()
        return (
            down.weight.detach().cpu().numpy().astype(np.float32),
            up.weight.detach().cpu().numpy().astype(np.float32),
            up.bias.detach().cpu().numpy().astype(np.float32),
        )

    @staticmethod
    def _clearly_better(
        active: dict[str, float | int], candidate: dict[str, float | int]
    ) -> bool:
        return bool(
            int(candidate.get("replay_episode_count") or 0) >= 2
            and int(candidate.get("replay_episode_count") or 0)
            == int(active.get("replay_episode_count") or 0)
            and float(candidate["handoff_accuracy"])
            >= float(active["handoff_accuracy"]) + 0.02
            and float(candidate["hard_negative_separation"])
            >= float(active["hard_negative_separation"]) + 0.02
            and float(candidate["false_match_rate"])
            <= float(active["false_match_rate"])
            and int(candidate["global_id_switches"])
            <= int(active["global_id_switches"])
        )

    def _register_candidate(
        self,
        projection: ProjectionHead,
        *,
        metrics: dict[str, Any],
        parent_model_id: str | None,
    ) -> str:
        model_id = projection.version
        destination = MODELS_DIR / "versions" / model_id / "projection.npz"
        destination.parent.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            destination,
            adapter_down=projection.adapter_down,
            adapter_up=projection.adapter_up,
            bias=projection.bias,
            version=np.asarray(projection.version),
        )
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        try:
            relative_path = str(
                destination.resolve().relative_to(ROOT_DIR.resolve())
            )
        except ValueError:
            relative_path = str(destination.resolve())
        now = utc_now()
        self.db.execute(
            """
            INSERT INTO model_registry
            (id, name, task, backend, weights_path, status, is_active,
             notes_json, metrics_json, parent_model_id,
             created_from_job_id, created_at, updated_at)
            VALUES (?, ?, ?, 'visionsort_reid_projection', ?, ?, 0, ?, ?, ?, NULL, ?, ?)
            """,
            (
                model_id,
                f"Parcel ReID projection {model_id[-12:]}",
                REID_TASK,
                relative_path,
                ModelStatus.CANDIDATE.value,
                json.dumps(
                    {
                        "artifact_sha256": digest,
                        "backbone_frozen": True,
                        "adapter_type": "residual_low_rank",
                        "adapter_rank": projection.adapter_rank,
                        "trainable_parameter_count": projection.trainable_parameter_count,
                        "dataset_version": self.dataset_version,
                    }
                ),
                json.dumps({**metrics, "artifact_sha256": digest}),
                parent_model_id,
                now,
                now,
            ),
        )
        return model_id

    def _refresh_collection_state(self, *, emit: bool) -> None:
        counts = self.repo.pair_counts(dataset_version=self.dataset_version)
        ready = (
            counts["POSITIVE"] >= self.min_positive_pairs
            and counts["HARD_NEGATIVE"] >= self.min_hard_negatives
        )
        previous = self.state
        self.state = (
            ReIDAdaptationState.READY
            if ready
            else ReIDAdaptationState.COLLECTING
        )
        if emit and ready and previous != ReIDAdaptationState.READY:
            self._event(
                "REID_DATASET_READY",
                {"dataset_version": self.dataset_version, "counts": counts},
            )

    @classmethod
    def _extremely_reliable(
        cls,
        outgoing: Tracklet,
        incoming: Tracklet,
        selected: HandoffCandidate,
        candidates: list[HandoffCandidate],
        *,
        ambiguity_margin: float,
    ) -> bool:
        if outgoing.integrity_status != "STABLE" or incoming.integrity_status != "STABLE":
            return False
        for tracklet in (outgoing, incoming):
            summary = tracklet.summary_json
            if (
                summary.get("merge_group_ids")
                or int(summary.get("split_count") or 0) > 0
                or str(summary.get("integrity_status") or "STABLE") == "AMBIGUOUS"
            ):
                return False
        selected_physical_score = cls._physical_score(selected)
        competitors = [
            cls._physical_score(item)
            for item in candidates
            if item.from_tracklet_id != selected.from_tracklet_id
        ]
        physical_margin = selected_physical_score - max(competitors, default=0.0)
        return bool(
            selected.gate_evidence.get("edge_authorized")
            and selected.gate_evidence.get("identity_usable")
            and float(selected.features.get("temporal") or 0.0) >= 0.65
            and float(selected.features.get("zone") or 0.0) >= 0.95
            and float(selected.features.get("dimensions") or 0.0) >= 0.70
            and float(selected.features.get("integrity") or 0.0) >= 0.95
            and selected_physical_score >= 0.80
            and physical_margin >= max(0.16, 2.0 * float(ambiguity_margin))
        )

    @staticmethod
    def _physical_score(candidate: HandoffCandidate) -> float:
        explicit = candidate.features.get("physical_score")
        if explicit is not None:
            return float(explicit)
        weights = {
            "temporal": 0.24,
            "dimensions": 0.20,
            "zone": 0.16,
            "speed": 0.10,
            "integrity": 0.10,
        }
        if candidate.features.get("world") is not None:
            weights["world"] = 0.08
        return sum(
            float(candidate.features.get(name) or 0.0) * weight
            for name, weight in weights.items()
        ) / sum(weights.values())

    def _event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        severity: str = "info",
    ) -> None:
        if self.event_repo is not None:
            self.event_repo.add_event(event_type, payload, severity=severity)


def _descriptor(tracklet: Tracklet | None) -> dict[str, Any] | None:
    if tracklet is None:
        return None
    value = tracklet.summary_json.get("appearance_descriptor")
    return dict(value) if isinstance(value, dict) else None


def _aggregate(descriptor: dict[str, Any]) -> np.ndarray | None:
    summary = {"appearance_descriptor": descriptor}
    values = descriptor_embeddings(summary)
    if values is None or values.size == 0:
        return None
    aggregate = descriptor.get("aggregate_embedding")
    if aggregate:
        return np.asarray(aggregate, dtype=np.float32).reshape(-1)
    return np.median(values, axis=0).astype(np.float32)


def _replay_evidence(
    outgoing: Tracklet,
    incoming: Tracklet,
    candidate: HandoffCandidate,
) -> dict[str, Any]:
    evidence = candidate.gate_evidence
    return {
        "edge": {
            "from_role": outgoing.camera_role,
            "to_role": incoming.camera_role,
            "min_transit_s": float(evidence.get("min_transit_s") or 0.0),
            "max_transit_s": float(
                evidence.get("max_transit_s")
                or max(0.1, incoming.started_at_global - outgoing.ended_at_global + 1.0)
            ),
        },
        "outgoing": _physical_tracklet_snapshot(outgoing),
        "incoming": _physical_tracklet_snapshot(incoming),
        "physical_score": AutoReIDAdapter._physical_score(candidate),
        "gate_evidence": dict(evidence),
    }


def _physical_tracklet_snapshot(tracklet: Tracklet) -> dict[str, Any]:
    summary = tracklet.summary_json if isinstance(tracklet.summary_json, dict) else {}
    physical_keys = (
        "avg_dimensions",
        "avg_velocity",
        "first_zone_id",
        "last_zone_id",
        "integrity_status",
        "merge_group_ids",
        "split_count",
        "first_anchor_world_m",
        "last_anchor_world_m",
        "world_observation_count",
        "world_frame_id",
    )
    return {
        "tracklet_id": tracklet.tracklet_id,
        "session_id": tracklet.session_id,
        "source_id": tracklet.source_id,
        "camera_id": tracklet.camera_id,
        "camera_role": tracklet.camera_role,
        "local_track_id": int(tracklet.local_track_id),
        "started_at_local": float(tracklet.started_at_local),
        "ended_at_local": float(tracklet.ended_at_local),
        "started_at_global": float(tracklet.started_at_global),
        "ended_at_global": float(tracklet.ended_at_global),
        "class_name": tracklet.class_name,
        "first_bbox": list(tracklet.first_bbox),
        "last_bbox": list(tracklet.last_bbox),
        "avg_speed": float(tracklet.avg_speed),
        "last_zone_id": tracklet.last_zone_id,
        "frame_count": int(tracklet.frame_count),
        "integrity_status": tracklet.integrity_status,
        "summary": {key: summary.get(key) for key in physical_keys},
    }


def _tracklet_from_replay(
    snapshot: Any,
    descriptor: dict[str, Any],
) -> Tracklet | None:
    if not isinstance(snapshot, dict):
        return None
    try:
        summary = dict(snapshot.get("summary") or {})
        summary["appearance_descriptor"] = dict(descriptor)
        return Tracklet(
            tracklet_id=str(snapshot["tracklet_id"]),
            session_id=str(snapshot["session_id"]),
            source_id=str(snapshot["source_id"]),
            camera_id=str(snapshot["camera_id"]),
            camera_role=str(snapshot["camera_role"]),
            local_track_id=int(snapshot["local_track_id"]),
            started_at_local=float(snapshot["started_at_local"]),
            ended_at_local=float(snapshot["ended_at_local"]),
            started_at_global=float(snapshot["started_at_global"]),
            ended_at_global=float(snapshot["ended_at_global"]),
            class_name=str(snapshot.get("class_name") or "parcel"),
            first_bbox=tuple(float(value) for value in snapshot["first_bbox"]),
            last_bbox=tuple(float(value) for value in snapshot["last_bbox"]),
            avg_speed=float(snapshot.get("avg_speed") or 0.0),
            last_zone_id=snapshot.get("last_zone_id"),
            frame_count=int(snapshot.get("frame_count") or 1),
            observation_path="reid-heldout-replay",
            summary_json=summary,
            integrity_status=str(snapshot.get("integrity_status") or "STABLE"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _append_replay_zones(
    zones_by_role: dict[str, list[dict[str, Any]]],
    outgoing: Tracklet,
    incoming: Tracklet,
) -> None:
    pairs = (
        (
            outgoing.camera_role,
            outgoing.summary_json.get("last_zone_id") or outgoing.last_zone_id,
            "exit",
        ),
        (
            incoming.camera_role,
            incoming.summary_json.get("first_zone_id"),
            "entry",
        ),
    )
    for role, zone_id, kind in pairs:
        if not role or not zone_id:
            continue
        zones = zones_by_role.setdefault(str(role), [])
        if not any(item.get("zone_id") == zone_id for item in zones):
            zones.append({"zone_id": str(zone_id), "kind": kind})


def _deterministic_split(
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    episodes: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for pair in pairs:
        key = (
            str(pair.get("session_id") or ""),
            str(pair.get("edge_key") or ""),
            str(pair.get("right_tracklet_id") or ""),
        )
        episodes.setdefault(key, []).append(pair)
    ordered = sorted(
        episodes.items(),
        key=lambda item: hashlib.sha256("|".join(item[0]).encode()).hexdigest(),
    )
    holdout_count = max(1, len(ordered) // 5)
    heldout_keys = {key for key, _values in ordered[:holdout_count]}
    for key, values in ordered:
        (heldout if key in heldout_keys else training).extend(values)
    if not training:
        training = list(heldout)
    return training, heldout
