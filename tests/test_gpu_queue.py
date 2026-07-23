from visionsort.core.config import AppConfig
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.database.repositories import ArtifactRepository, JobRepository
from visionsort.runtime.supervisor import GPUResourceArbiter, RuntimeSupervisor


def test_training_is_queued_while_inference_sources_are_active(tmp_path):
    db = VisionSortDB(tmp_path / "queue.db")
    db.initialize()
    now = utc_now()
    db.execute(
        """
        INSERT INTO datasets
        (id, name, root_path, status, manifest_path, data_yaml_path, summary_json,
         created_at, updated_at)
        VALUES ('ready', 'Ready', 'data/datasets/ready', 'DATASET_READY',
                'manifest.csv', 'data.yaml', '{}', ?, ?)
        """,
        (now, now),
    )
    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.db = db
    supervisor.config = AppConfig(
        values={"gpu": {"training_policy": "queue"}}
    )
    supervisor.arbiter = GPUResourceArbiter(
        allow_training_while_inference=False,
        max_concurrent_live_sources=3,
    )
    supervisor.camera_processes = {"src": (object(), object())}
    supervisor.training_processes = {}
    supervisor.artifact_repo = ArtifactRepository(db)
    supervisor.job_repo = JobRepository(db)

    job_id = supervisor.start_training(
        {
            "dataset_id": "ready",
            "model_id": "demo_synth_det",
            "mode": "demo",
            "priority": 10,
        }
    )

    training_job = db.fetch_one(
        "SELECT * FROM training_jobs WHERE id = ?", (job_id,)
    )
    job_run = db.fetch_one(
        "SELECT * FROM job_runs WHERE id = ?", (f"TRAINING:{job_id}",)
    )
    assert training_job is not None
    assert training_job["status"] == "QUEUED"
    assert job_run is not None
    assert job_run["status"] == "QUEUED"
