from visionsort.database.db import VisionSortDB


def test_database_initializes_defaults(tmp_path):
    db = VisionSortDB(tmp_path / "visionsort.db")
    db.initialize()
    models = db.fetch_all("SELECT id FROM model_registry ORDER BY id")
    trackers = db.fetch_all("SELECT id FROM tracker_registry ORDER BY id")
    assert any(row["id"] == "demo_synth_det" for row in models)
    assert any(row["id"] == "greedy_iou" for row in trackers)
