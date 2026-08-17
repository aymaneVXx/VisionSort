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
    assert version is not None and version[0] == 13
    assert model is not None and model["status"] == "CANDIDATE"
    assert history is not None
    assert history["status"] == "ACTIVE"
    assert history["runtime_applied"] == 1


def test_v12_migration_preserves_legacy_destination_as_unverified_observation(
    tmp_path,
):
    db = VisionSortDB(tmp_path / "legacy-v11.db")
    db.initialize()
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO global_parcels
            (parcel_id, state, last_camera_id, first_seen_at, last_seen_at,
             current_tracklet_id, assigned_destination, appearance_json)
            VALUES ('parcel-legacy', 'DROPPED', 'cam-1', 0, 1,
                    'tracklet-1', 'destination-A', '[]')
            """
        )
        conn.execute(
            "UPDATE global_parcels SET observed_destination = NULL WHERE parcel_id = 'parcel-legacy'"
        )
        conn.execute("PRAGMA user_version = 11")

    db.initialize()

    row = db.fetch_one(
        """
        SELECT expected_destination, observed_destination, destination_result
        FROM global_parcels WHERE parcel_id = 'parcel-legacy'
        """
    )
    assert db.fetch_one("PRAGMA user_version")[0] == 13
    assert row["expected_destination"] is None
    assert row["observed_destination"] == "destination-A"
    assert row["destination_result"] == "DESTINATION_UNVERIFIED"


def test_v13_migration_adds_reid_adaptation_tables(tmp_path):
    db = VisionSortDB(tmp_path / "legacy-v12.db")
    db.initialize()
    with db.connect() as conn:
        conn.execute("DROP TABLE reid_training_pairs")
        conn.execute("DROP TABLE reid_adaptation_runs")
        conn.execute("PRAGMA user_version = 12")

    db.initialize()

    assert db.fetch_one("PRAGMA user_version")[0] == 13
    tables = {
        str(row["name"])
        for row in db.fetch_all(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {"reid_training_pairs", "reid_adaptation_runs"} <= tables
