import os

import pytest

from visionsort.runtime.multimodel_e2e import run_multimodel_e2e


@pytest.mark.skipif(
    os.getenv("RUN_SUPERVISOR_E2E") != "1",
    reason="Scénario multiprocessus explicite; exécuté dans une étape CI dédiée.",
)
def test_task_aware_models_through_runtime_supervisor(tmp_path):
    report_path = tmp_path / "multimodel-e2e.json"
    report = run_multimodel_e2e(
        tmp_path / "multimodel-e2e.db",
        report_path=report_path,
        max_frames=24,
        replay_fps=60.0,
    )

    assert report["status"] == "COMPLETED"
    assert report["mode"] == "SUPERVISOR_MULTI_MODEL_E2E"
    assert report["parcel_pipeline_verified"] is True
    assert report["combined_source_frames"] > 0
    assert report["pose_keypoint_observations"] > 0
    assert report["model_requests"]["demo_synth_det"] > 0
    assert report["keypoint_event_verified"] is True
    assert report["active_models"]["detection"] == "demo_synth_det"
    assert report["active_models"]["pose"].startswith("demo-pose-v2-")
    assert report["parcel_model_not_reloaded"] is True
    assert report["shutdown_clean"] is True
    assert report["validated_on_site"] is False
    assert report_path.exists()
