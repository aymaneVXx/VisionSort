import json
import time

from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import (
    ArtifactRepository,
    JobRepository,
)
from visionsort.runtime.supervisor import RuntimeSupervisor


class _RunningProcess:
    def __init__(self):
        self.terminated = False

    def is_alive(self) -> bool:
        return not self.terminated

    def terminate(self) -> None:
        self.terminated = True

    def join(self, timeout=None) -> None:
        return None


def test_pipeline_step_claim_is_idempotent_and_resumable(tmp_path):
    db = VisionSortDB(tmp_path / "jobs.db")
    db.initialize()
    repo = ArtifactRepository(db)

    step_id, should_run, _ = repo.claim_pipeline_step(
        "session", "SAMPLE", {"name": "dataset"}
    )
    assert should_run is True
    duplicate_id, duplicate_should_run, _ = repo.claim_pipeline_step(
        "session", "SAMPLE", {"name": "dataset"}
    )
    assert duplicate_id == step_id
    assert duplicate_should_run is False

    repo.finish_pipeline_step(
        step_id, status="FAILED", error_text="simulated crash"
    )
    resumed_id, resumed, _ = repo.claim_pipeline_step(
        "session", "SAMPLE", {"name": "dataset"}
    )
    assert resumed_id == step_id
    assert resumed is True
    repo.finish_pipeline_step(
        step_id, status="COMPLETED", outputs={"dataset_id": "dataset-1"}
    )
    completed_id, rerun, outputs = repo.claim_pipeline_step(
        "session", "SAMPLE", {"name": "dataset"}
    )
    row = db.fetch_one(
        "SELECT attempt_count FROM pipeline_step_runs WHERE id = ?", (step_id,)
    )
    assert completed_id == step_id
    assert rerun is False
    assert outputs["dataset_id"] == "dataset-1"
    assert row is not None
    assert row["attempt_count"] == 2


def test_supervisor_recovers_interrupted_jobs_after_restart(tmp_path):
    db = VisionSortDB(tmp_path / "recovery.db")
    db.initialize()
    now = utc_now()
    db.execute(
        """
        INSERT INTO training_jobs
        (id, dataset_id, model_id, status, recipe_json, log_path, metrics_json,
         error_text, created_at, updated_at)
        VALUES ('train-recover', 'dataset', 'model', 'RUNNING', '{}', 'log.txt',
                '{}', NULL, ?, ?)
        """,
        (now, now),
    )
    repo = ArtifactRepository(db)
    step_id, _, _ = repo.claim_pipeline_step("session", "SAMPLE", {})
    JobRepository(db).upsert_job_run(
        "TRAINING", "train-recover", 1234, "RUNNING", {}
    )
    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.db = db

    supervisor.recover_interrupted_jobs()

    training = db.fetch_one(
        "SELECT status, error_text FROM training_jobs WHERE id = 'train-recover'"
    )
    step = db.fetch_one(
        "SELECT status, error_text FROM pipeline_step_runs WHERE id = ?",
        (step_id,),
    )
    job_run = db.fetch_one(
        "SELECT status FROM job_runs WHERE id = 'TRAINING:train-recover'"
    )
    assert training is not None and training["status"] == "QUEUED"
    assert "Repris" in training["error_text"]
    assert step is not None and step["status"] == "FAILED"
    assert job_run is not None and job_run["status"] == "FAILED"


def test_training_job_cancellation_is_persisted(tmp_path):
    db = VisionSortDB(tmp_path / "cancel.db")
    db.initialize()
    now = utc_now()
    db.execute(
        """
        INSERT INTO training_jobs
        (id, dataset_id, model_id, status, recipe_json, log_path, metrics_json,
         error_text, created_at, updated_at)
        VALUES ('cancel-me', 'dataset', 'model', 'RUNNING', '{}', 'log.txt',
                '{}', NULL, ?, ?)
        """,
        (now, now),
    )
    process = _RunningProcess()
    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.db = db
    supervisor.training_processes = {"cancel-me": process}
    supervisor.pipeline_processes = {}
    supervisor.artifact_repo = ArtifactRepository(db)
    supervisor.job_repo = JobRepository(db)
    supervisor.job_repo.upsert_job_run(
        "TRAINING", "cancel-me", 42, "RUNNING", {}
    )

    supervisor.cancel_job("TRAINING", "cancel-me")

    training = db.fetch_one(
        "SELECT status, error_text FROM training_jobs WHERE id = 'cancel-me'"
    )
    run = db.fetch_one(
        "SELECT status FROM job_runs WHERE id = 'TRAINING:cancel-me'"
    )
    assert process.terminated is True
    assert training is not None and training["status"] == "CANCELLED"
    assert run is not None and run["status"] == "CANCELLED"
    assert "Annulé" in training["error_text"]
