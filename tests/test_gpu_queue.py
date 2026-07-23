from visionsort.core.config import AppConfig
from visionsort.database.db import VisionSortDB, utc_now
from visionsort.datasets.pipeline import (
    compute_dataset_fingerprint,
    rewrite_training_manifest,
)
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
    root = tmp_path / "ready"
    root.mkdir()
    image = root / "image.jpg"
    label = root / "label.txt"
    manifest = root / "manifest.csv"
    data_yaml = root / "data.yaml"
    cv2.imwrite(
        str(image), np.zeros((32, 32, 3), dtype=np.uint8)
    )
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        "test: images/test\ntask: detection\nnames:\n  0: parcel\n",
        encoding="utf-8",
    )
    db.execute(
        """
        UPDATE datasets SET root_path = ?, manifest_path = ?,
                            data_yaml_path = ?
        WHERE id = 'ready'
        """,
        (str(root), str(manifest), str(data_yaml)),
    )
    db.execute(
        """
        INSERT INTO dataset_items
        (id, dataset_id, split, source_id, camera_role, frame_index,
         timestamp_global, image_path, label_path, annotation_status,
         reason, score, metadata_json, created_at)
        VALUES ('item-ready', 'ready', 'train', 'source', 'C1', 1, 1.0,
                ?, ?, 'HUMAN_VALIDATED', 'ready', 1.0,
                '{"instance_count":1}', ?)
        """,
        (str(image), str(label), now),
    )
    rewrite_training_manifest(db, "ready", manifest)
    db.execute(
        "UPDATE datasets SET dataset_fingerprint = ? WHERE id = ?",
        (compute_dataset_fingerprint(db, "ready"), "ready"),
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
import cv2
import numpy as np
