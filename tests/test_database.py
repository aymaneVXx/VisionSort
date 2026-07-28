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


def test_v9_migration_seeds_existing_active_models(tmp_path):
    db = VisionSortDB(tmp_path / "legacy-v8.db")
    db.initialize()
    with db.connect() as conn:
        conn.execute("DROP TABLE model_activation_history")
        conn.execute(
            """
            UPDATE model_registry
            SET status = 'CANDIDATE', is_active = 1
            WHERE id = 'demo_synth_det'
            """
        )
        conn.execute("PRAGMA user_version = 8")

    db.initialize()

    version = db.fetch_one("PRAGMA user_version")
    model = db.fetch_one(
        """
        SELECT status, is_active FROM model_registry
        WHERE id = 'demo_synth_det'
        """
    )
    history = db.fetch_one(
        """
        SELECT status, runtime_applied
        FROM model_activation_history
        WHERE activated_model_id = 'demo_synth_det'
        """
    )
    assert version is not None and version[0] == 9
    assert model is not None and model["status"] == "CANDIDATE"
    assert history is not None
    assert history["status"] == "ACTIVE"
    assert history["runtime_applied"] == 1
