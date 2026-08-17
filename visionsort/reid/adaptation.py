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
        min_positive_pairs: int = 24,
        min_hard_negatives: int = 12,
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
        competitor_scores = [
            item.score
            for item in candidates
            if item.from_tracklet_id != selected.from_tracklet_id
        ]
        decision_margin = selected.score - max(competitor_scores, default=0.0)
        metadata = {
            "decision_score": selected.score,
            "decision_margin": decision_margin,
            "non_visual_confidence": self._non_visual_confidence(selected),
            "model_version": selected.model_version,
            "pseudo_label_policy": "PR3_EXTREMELY_RELIABLE_V1",
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
                    "selected_from_tracklet_id": selected.from_tracklet_id,
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
            matrix, bias = self._train_projection(training)
            candidate_version = f"parcel-reid-projection-{uuid.uuid4().hex[:12]}"
            candidate = ProjectionHead(matrix, bias, version=candidate_version)
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
        scored: list[tuple[str, str, float]] = []
        for pair in pairs:
            left = _aggregate(pair["left_descriptor"])
            right = _aggregate(pair["right_descriptor"])
            if left is None or right is None:
                continue
            left_value = projection.transform(left)[0]
            right_value = projection.transform(right)[0]
            similarity = float(np.dot(left_value, right_value))
            scored.append(
                (str(pair["right_tracklet_id"]), str(pair["label"]), similarity)
            )
        positives = [value for _, label, value in scored if label == "POSITIVE"]
        negatives = [value for _, label, value in scored if label == "HARD_NEGATIVE"]
        by_anchor: dict[str, dict[str, list[float]]] = {}
        for anchor, label, value in scored:
            by_anchor.setdefault(anchor, {"POSITIVE": [], "HARD_NEGATIVE": []})[
                label
            ].append(value)
        retrieval_results: list[bool] = []
        ambiguous_results: list[bool] = []
        switches = 0
        for values in by_anchor.values():
            if not values["POSITIVE"]:
                continue
            best_positive = max(values["POSITIVE"])
            best_negative = max(values["HARD_NEGATIVE"], default=-1.0)
            retrieval_results.append(best_positive > best_negative + 0.02)
            ambiguous_results.append(abs(best_positive - best_negative) < 0.08)
            switches += int(best_negative >= best_positive)
        retrieval = float(np.mean(retrieval_results)) if retrieval_results else 0.0
        separation = (
            float(np.mean(positives) - np.mean(negatives))
            if positives and negatives
            else 0.0
        )
        return {
            "positive_pair_retrieval_accuracy": retrieval,
            "hard_negative_separation": separation,
            "handoff_accuracy": retrieval,
            "false_match_rate": (
                float(np.mean(np.asarray(negatives) >= 0.75)) if negatives else 0.0
            ),
            "ambiguous_rate": (
                float(np.mean(ambiguous_results)) if ambiguous_results else 0.0
            ),
            "global_id_switches": int(switches),
        }

    def _train_projection(
        self, pairs: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
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
        head = torch.nn.Linear(dimension, dimension, bias=True)
        with torch.no_grad():
            head.weight.copy_(torch.eye(dimension))
            head.bias.zero_()
        initial = head.weight.detach().clone()
        optimizer = torch.optim.AdamW(head.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
        loss_function = torch.nn.CosineEmbeddingLoss(margin=0.25)
        left = torch.tensor(np.stack([item[0] for item in examples]), dtype=torch.float32)
        right = torch.tensor(np.stack([item[1] for item in examples]), dtype=torch.float32)
        labels = torch.tensor([item[2] for item in examples], dtype=torch.float32)
        head.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(head(left), head(right), labels)
            loss = loss + 1.0e-4 * torch.mean((head.weight - initial) ** 2)
            loss.backward()
            optimizer.step()
        return (
            head.weight.detach().cpu().numpy().astype(np.float32),
            head.bias.detach().cpu().numpy().astype(np.float32),
        )

    @staticmethod
    def _clearly_better(
        active: dict[str, float | int], candidate: dict[str, float | int]
    ) -> bool:
        return bool(
            float(candidate["handoff_accuracy"])
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
            matrix=projection.matrix,
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
        competitors = [
            item.score
            for item in candidates
            if item.from_tracklet_id != selected.from_tracklet_id
        ]
        decision_margin = selected.score - max(competitors, default=0.0)
        return bool(
            selected.gate_evidence.get("edge_authorized")
            and selected.gate_evidence.get("identity_usable")
            and float(selected.features.get("temporal") or 0.0) >= 0.65
            and float(selected.features.get("zone") or 0.0) >= 0.95
            and float(selected.features.get("dimensions") or 0.0) >= 0.70
            and float(selected.features.get("integrity") or 0.0) >= 0.95
            and cls._non_visual_confidence(selected) >= 0.80
            and decision_margin >= max(0.16, 2.0 * float(ambiguity_margin))
        )

    @staticmethod
    def _non_visual_confidence(candidate: HandoffCandidate) -> float:
        values = [
            float(candidate.features.get(name) or 0.0)
            for name in ("temporal", "dimensions", "zone", "speed", "integrity")
        ]
        if candidate.features.get("world") is not None:
            values.append(float(candidate.features["world"]))
        return float(np.mean(values)) if values else 0.0

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


def _deterministic_split(
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    by_label: dict[str, list[dict[str, Any]]] = {
        "POSITIVE": [],
        "HARD_NEGATIVE": [],
    }
    for pair in pairs:
        by_label.setdefault(str(pair["label"]), []).append(pair)
    for values in by_label.values():
        ordered = sorted(
            values,
            key=lambda item: hashlib.sha256(str(item["id"]).encode()).hexdigest(),
        )
        holdout_count = max(1, len(ordered) // 5)
        heldout.extend(ordered[:holdout_count])
        training.extend(ordered[holdout_count:])
    if not training:
        training = list(heldout)
    return training, heldout
