from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import ControlRepository


def test_database_initializes_defaults(tmp_path):
    db = VisionSortDB(tmp_path / "visionsort.db")
    db.initialize()
    models = db.fetch_all("SELECT id FROM model_registry ORDER BY id")
    trackers = db.fetch_all("SELECT id FROM tracker_registry ORDER BY id")
    assert any(row["id"] == "demo_synth_det" for row in models)
    assert any(row["id"] == "greedy_iou" for row in trackers)


def test_source_state_partial_update_preserves_recording_flag(tmp_path):
    db = VisionSortDB(tmp_path / "visionsort.db")
    db.initialize()
    repository = ControlRepository(db)
    source_id = repository.upsert_source(
        {
            "id": "source-c1",
            "name": "Replay C1",
            "role": "C1",
            "source_type": "REPLAY",
            "uri": "fixture.mp4",
            "model_id": "demo_synth_det",
            "tracker_id": "greedy_iou",
            "enabled": True,
        }
    )

    repository.update_source_state(
        source_id, status="REPLAY", recording_enabled=True
    )
    repository.update_source_state(source_id, status="OFFLINE")

    state = db.fetch_one(
        "SELECT recording_enabled FROM source_state WHERE source_id = ?",
        (source_id,),
    )
    assert state["recording_enabled"] == 1
