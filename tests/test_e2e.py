from pathlib import Path

from visionsort.core.paths import ROOT_DIR
from visionsort.runtime.e2e import run_demo_e2e


def test_cpu_demo_end_to_end_cycle(tmp_path):
    report_path = tmp_path / "e2e-report.json"
    report = run_demo_e2e(
        tmp_path / "e2e.db",
        report_path=report_path,
        max_frames=72,
    )

    assert report["status"] == "COMPLETED"
    assert report["mode"] == "CPU_DEMO_SIMULATED_BACKENDS"
    assert report["frames_by_role"] == {"C1": 72, "C2": 72, "C3": 72}
    assert report["observations"] > 0
    assert report["local_track_identities"] > 0
    assert report["tracklets"] > 0
    assert report["global_match_results"]["MATCHED"] > 0
    assert report["global_match_results"]["AMBIGUOUS"] > 0
    assert report["dataset_status"] == "DATASET_READY"
    assert report["dataset_items"] > 0
    assert report["split_integrity"]["valid"] is True
    assert report["training_status"] == "COMPLETED"
    assert report["candidate_metrics"]["count_accuracy"] == 1.0
    assert report["active_model_id"] == report["candidate_model_id"]
    assert report["reload_verified"] is True
    assert report["site_validation_status"] == "NON_VALIDÉ_SUR_SITE"
    assert (ROOT_DIR / Path(report["best_pt"])).exists()
    assert report_path.exists()
