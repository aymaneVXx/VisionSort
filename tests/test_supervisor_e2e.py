import os

import pytest

from visionsort.runtime.supervisor_e2e import run_supervisor_e2e


@pytest.mark.skipif(
    os.getenv("RUN_SUPERVISOR_E2E") != "1",
    reason="Scénario multiprocessus explicite; exécuté dans une étape CI dédiée.",
)
def test_full_lifecycle_through_runtime_supervisor(tmp_path):
    report_path = tmp_path / "supervisor-e2e.json"
    report = run_supervisor_e2e(
        tmp_path / "supervisor-e2e.db",
        report_path=report_path,
        max_frames=72,
        replay_fps=60.0,
    )

    assert report["status"] == "COMPLETED"
    assert report["mode"] == "SUPERVISOR_PROCESS_E2E"
    assert report["tracklets"] > 0
    assert report["dataset_status"] == "DATASET_READY"
    assert report["archive_segments"] > 0
    assert report["archive_frames"] > 0
    assert report["source_uri_changed_after_capture"] is True
    assert report["archive_provenance_verified"] is True
    assert report["dataset_integrity"]["valid"] is True
    assert report["dataset_fingerprint_verified"] is True
    assert report["split_integrity"]["all_splits_nonempty"] is True
    assert report["training_status"] == "COMPLETED"
    assert report["active_model_id"] == report["candidate_model_id"]
    assert report["runtime_reload_verified"] is True
    assert report["activated_session_observations"]["model_ids"] == [
        report["candidate_model_id"]
    ]
    assert report["command_counts"] == {"COMPLETED": 28}
    assert report["shutdown_clean"] is True
    assert report["validated_on_site"] is False
    assert report_path.exists()
