from pathlib import Path
import json

import cv2
import numpy as np
import yaml

from visionsort.database.db import VisionSortDB, utc_now
from visionsort.datasets.pipeline import (
    build_dataset,
    compute_dataset_fingerprint,
    resolve_split_assignments,
    rewrite_training_manifest,
    stable_session_split,
    validate_dataset_splits,
    verify_dataset_fingerprint,
)
from visionsort.database.repositories import ArtifactRepository
from visionsort.runtime.demo_assets import ensure_demo_assets


def test_build_dataset_from_tracklet_jsonl(tmp_path):
    assets = ensure_demo_assets()
    db = VisionSortDB(tmp_path / "visionsort.db")
    db.initialize()
    now = utc_now()
    db.execute(
        """
        INSERT INTO sources (id, name, role, source_type, uri, model_id, tracker_id, enabled, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        """,
        ("src1", "Replay C1", "C1", "REPLAY", assets["C1"], "demo_synth_det", "greedy_iou", now, now),
    )
    db.execute(
        """
        INSERT INTO capture_sessions (id, name, pipeline_state, demo_mode, site_validated, config_json, report_path, started_at, ended_at, created_at, updated_at)
        VALUES (?, ?, 'CAPTURED', 1, 0, '{}', NULL, NULL, NULL, ?, ?)
        """,
        ("session-test", "pytest session", now, now),
    )
    db.execute(
        """
        INSERT INTO capture_session_sources (id, session_id, source_id, camera_role, time_offset_ms, replay_fps, created_at, updated_at)
        VALUES (?, ?, ?, ?, 0, 8.0, ?, ?)
        """,
        ("sesssrc1", "session-test", "src1", "C1", now, now),
    )
    details = tmp_path / "tracklet.jsonl"
    details.write_text(
        '{"session_id":"session-test","source_id":"src1","camera_id":"src1","camera_role":"C1","local_track_id":1,"frame_index":5,"timestamp_local":0.5,"timestamp_global":0.5,"class_name":"parcel","confidence":0.95,"bbox":[100,160,160,200],"velocity":[0,0],"zone_id":"c1_exit","appearance_hint":null,"model_id":"demo_synth_det","tracker_id":"greedy_iou","extra":{"parcel_hint":"P1"}}\n',
        encoding="utf-8",
    )
    db.execute(
        """
        INSERT INTO tracklets
        (tracklet_id, parcel_id, session_id, source_id, camera_id, camera_role, local_track_id,
         started_at_local, ended_at_local, started_at_global, ended_at_global,
         class_name, last_zone_id, frame_count, avg_speed, observation_path, summary_json, match_result, model_id, tracker_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "t1",
            "P1",
            "session-test",
            "src1",
            "src1",
            "C1",
            1,
            0.0,
            1.0,
            0.0,
            1.0,
            "parcel",
            "c1_exit",
            1,
            0.0,
            str(details),
            '{"parcel_hint":"P1"}',
            "UNMATCHED",
            "demo_synth_det",
            "greedy_iou",
        ),
    )
    result = build_dataset(db, session_id="session-test", name="pytest_dataset")
    assert result["manifest_rows"] >= 1
    assert Path(result["root"]).exists()
    dataset = db.fetch_one("SELECT * FROM datasets WHERE id = ?", (result["dataset_id"],))
    assert dataset is not None
    summary = json.loads(dataset["summary_json"] or "{}")
    assert summary["dataset_version"] == result["dataset_id"]
    assert summary["manifest_sha256"]
    assert summary["split_integrity"]["valid"] is True
    assert summary["split_assignment"] == stable_session_split("session-test")

    pose_result = build_dataset(
        db, session_id="session-test", name="pytest_pose", task="pose"
    )
    pose_dataset = db.fetch_one(
        "SELECT * FROM datasets WHERE id = ?", (pose_result["dataset_id"],)
    )
    assert pose_dataset is not None
    pose_yaml = yaml.safe_load(
        Path(pose_dataset["data_yaml_path"]).read_text(encoding="utf-8")
        if Path(pose_dataset["data_yaml_path"]).is_absolute()
        else (Path.cwd() / pose_dataset["data_yaml_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert pose_yaml["task"] == "pose"
    assert pose_yaml["kpt_shape"] == [17, 3]
    assert len(pose_yaml["flip_idx"]) == 17


def test_dataset_groups_every_visible_instance_in_the_same_frame(tmp_path):
    assets = ensure_demo_assets()
    db = VisionSortDB(tmp_path / "complete.db")
    db.initialize()
    now = utc_now()
    db.execute(
        """
        INSERT INTO sources (id, name, role, source_type, uri, model_id, tracker_id, enabled, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
        """,
        (
            "src-complete",
            "Replay C1",
            "C1",
            "REPLAY",
            assets["C1"],
            "demo_synth_det",
            "greedy_iou",
            now,
            now,
        ),
    )
    db.execute(
        """
        INSERT INTO capture_sessions
        (id, name, pipeline_state, demo_mode, site_validated, config_json, report_path, started_at, ended_at, created_at, updated_at)
        VALUES (?, ?, 'CAPTURED', 1, 0, '{}', NULL, NULL, NULL, ?, ?)
        """,
        ("session-complete", "Complete", now, now),
    )
    for local_id, class_name, bbox in (
        (1, "parcel", [100, 160, 160, 200]),
        (2, "person", [200, 80, 300, 300]),
    ):
        details = tmp_path / f"track-{local_id}.jsonl"
        details.write_text(
            json.dumps(
                {
                    "session_id": "session-complete",
                    "source_id": "src-complete",
                    "camera_id": "src-complete",
                    "camera_role": "C1",
                    "local_track_id": local_id,
                    "frame_index": 5,
                    "timestamp_local": 0.5,
                    "timestamp_global": 100.5,
                    "class_name": class_name,
                    "confidence": 0.95,
                    "bbox": bbox,
                    "velocity": [0, 0],
                    "zone_id": None,
                    "appearance_hint": None,
                    "model_id": "demo_synth_det",
                    "tracker_id": "greedy_iou",
                    "extra": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        db.execute(
            """
            INSERT INTO tracklets
            (tracklet_id, parcel_id, session_id, source_id, camera_id, camera_role, local_track_id,
             started_at_local, ended_at_local, started_at_global, ended_at_global,
             class_name, last_zone_id, frame_count, avg_speed, observation_path, summary_json,
             match_result, model_id, tracker_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"tracklet-{local_id}",
                "global-1" if class_name == "parcel" else None,
                "session-complete",
                "src-complete",
                "src-complete",
                "C1",
                local_id,
                0.5,
                0.5,
                100.5,
                100.5,
                class_name,
                None,
                1,
                0.0,
                str(details),
                "{}",
                "UNMATCHED",
                "demo_synth_det",
                "greedy_iou",
            ),
        )

    result = build_dataset(
        db, session_id="session-complete", name="frame-complete"
    )
    items = db.fetch_all(
        "SELECT * FROM dataset_items WHERE dataset_id = ?", (result["dataset_id"],)
    )

    assert len(items) == 1
    metadata = json.loads(items[0]["metadata_json"])
    assert metadata["instance_count"] == 2
    assert {
        (observation["camera_id"], observation["local_track_id"])
        for observation in metadata["observations"]
    } == {("src-complete", 1), ("src-complete", 2)}
    assert validate_dataset_splits(db, result["dataset_id"])["valid"] is True


def test_split_validator_detects_session_leakage(tmp_path):
    db = VisionSortDB(tmp_path / "leak.db")
    db.initialize()
    now = utc_now()
    db.execute(
        """
        INSERT INTO datasets
        (id, name, root_path, status, manifest_path, data_yaml_path, summary_json, created_at, updated_at)
        VALUES ('dataset-leak', 'Leak', 'data/datasets/leak', 'SAMPLED', 'manifest.csv', 'data.yaml', '{}', ?, ?)
        """,
        (now, now),
    )
    for item_id, split in (("a", "train"), ("b", "test")):
        db.execute(
            """
            INSERT INTO dataset_items
            (id, dataset_id, session_id, sample_group_id, split, source_id, camera_role,
             frame_index, timestamp_global, image_path, label_path, annotation_status,
             reason, score, metadata_json, created_at)
            VALUES (?, 'dataset-leak', 'same-session', ?, ?, 'src', 'C1', 1, 1.0,
                    ?, NULL, 'NEEDS_REVIEW', 'test', 1.0, ?, ?)
            """,
            (
                item_id,
                f"group-{item_id}",
                split,
                f"{item_id}.jpg",
                json.dumps({"image_ahash64": f"hash-{item_id}"}),
                now,
            ),
        )

    integrity = validate_dataset_splits(db, "dataset-leak")
    assert integrity["valid"] is False
    assert any(leak["kind"] == "session_id_cross_split" for leak in integrity["leaks"])


def test_multi_session_dataset_persists_explicit_splits(tmp_path):
    db = VisionSortDB(tmp_path / "multi.db")
    db.initialize()
    now = utc_now()
    assignments = {
        "session-train": "train",
        "session-val": "val",
        "session-test": "test",
    }
    for index, (session_id, split) in enumerate(assignments.items()):
        source_id = f"source-{split}"
        frame_index = 0
        video_path = tmp_path / f"{split}.avi"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            8.0,
            (64, 64),
        )
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        if index == 0:
            image[:, :32] = 255
        elif index == 1:
            image[:32, :] = 255
        else:
            image[np.indices((64, 64)).sum(axis=0) % 2 == 0] = 255
        writer.write(image)
        writer.release()
        db.execute(
            """
            INSERT INTO sources
            (id, name, role, source_type, uri, model_id, tracker_id, enabled,
             created_at, updated_at)
            VALUES (?, ?, 'C1', 'REPLAY', ?, 'demo_synth_det', 'greedy_iou',
                    1, ?, ?)
            """,
            (source_id, source_id, str(video_path), now, now),
        )
        db.execute(
            """
            INSERT INTO capture_sessions
            (id, name, pipeline_state, demo_mode, site_validated, config_json,
             started_at, ended_at, created_at, updated_at)
            VALUES (?, ?, 'PROCESSED', 1, 0, '{}', 1.0, 2.0, ?, ?)
            """,
            (session_id, session_id, now, now),
        )
        db.execute(
            """
            INSERT INTO capture_session_sources
            (id, session_id, source_id, camera_role, time_offset_ms, replay_fps,
             created_at, updated_at)
            VALUES (?, ?, ?, 'C1', 0, 8.0, ?, ?)
            """,
            (f"link-{split}", session_id, source_id, now, now),
        )
        observation_path = tmp_path / f"{session_id}.jsonl"
        observation_path.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "source_id": source_id,
                    "camera_id": source_id,
                    "camera_role": "C1",
                    "local_track_id": 1,
                    "frame_index": frame_index,
                    "timestamp_global": float(index + 1),
                    "class_name": "parcel",
                    "confidence": 0.9,
                    "bbox": [50, 50, 120, 100],
                    "velocity": [1, 0],
                    "model_id": "demo_synth_det",
                    "tracker_id": "greedy_iou",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        db.execute(
            """
            INSERT INTO tracklets
            (tracklet_id, parcel_id, session_id, source_id, camera_id, camera_role,
             local_track_id, started_at_global, ended_at_global, class_name,
             last_zone_id, frame_count, avg_speed, observation_path, summary_json,
             match_result, model_id, tracker_id)
            VALUES (?, ?, ?, ?, ?, 'C1', 1, ?, ?, 'parcel', 'c1_exit', 1,
                    1.0, ?, '{}', 'UNMATCHED', 'demo_synth_det', 'greedy_iou')
            """,
            (
                f"track-{split}",
                f"parcel-{split}",
                session_id,
                source_id,
                source_id,
                float(index + 1),
                float(index + 1),
                str(observation_path),
            ),
        )

    result = build_dataset(
        db,
        session_ids=list(assignments),
        split_assignments=assignments,
        name="multi-session",
    )

    stored = {
        row["session_id"]: row["split"]
        for row in db.fetch_all(
            "SELECT session_id, split FROM dataset_sessions WHERE dataset_id = ?",
            (result["dataset_id"],),
        )
    }
    assert stored == assignments
    assert result["split_integrity"]["all_splits_nonempty"] is True
    assert resolve_split_assignments(list(assignments)).keys() == assignments.keys()


def test_finalized_dataset_fingerprint_detects_tampering_and_locks_items(tmp_path):
    db = VisionSortDB(tmp_path / "fingerprint.db")
    db.initialize()
    now = utc_now()
    root = tmp_path / "dataset"
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
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "task: detection\n"
        "names:\n"
        "  0: parcel\n",
        encoding="utf-8",
    )
    db.execute(
        """
        INSERT INTO capture_sessions
        (id, name, pipeline_state, demo_mode, site_validated, config_json,
         started_at, ended_at, created_at, updated_at)
        VALUES ('session-fp', 'Fingerprint', 'PROCESSED', 1, 0, '{}',
                1.0, 2.0, ?, ?)
        """,
        (now, now),
    )
    db.execute(
        """
        INSERT INTO datasets
        (id, name, root_path, status, manifest_path, data_yaml_path,
         generation_config_json, summary_json, created_at, updated_at)
        VALUES ('dataset-fp', 'Fingerprint', ?, 'SAMPLED', ?, ?, '{"seed":7}',
                '{}', ?, ?)
        """,
        (str(root), str(manifest), str(data_yaml), now, now),
    )
    db.execute(
        """
        INSERT INTO dataset_sessions (dataset_id, session_id, split, created_at)
        VALUES ('dataset-fp', 'session-fp', 'test', ?)
        """,
        (now,),
    )
    db.execute(
        """
        INSERT INTO dataset_items
        (id, dataset_id, session_id, sample_group_id, split, source_id,
         camera_role, frame_index, timestamp_global, image_path, label_path,
         annotation_status, reason, score, metadata_json, created_at)
        VALUES ('item-1', 'dataset-fp', 'session-fp', 'group-1', 'test',
                'source-1', 'C1', 1, 1.0, ?, ?, 'HUMAN_VALIDATED',
                    'test', 1.0, '{"instance_count":1}', ?)
        """,
        (str(image), str(label), now),
    )
    rewrite_training_manifest(db, "dataset-fp", manifest)
    fingerprint = compute_dataset_fingerprint(db, "dataset-fp")
    db.execute(
        """
        UPDATE datasets
        SET status = 'DATASET_READY', dataset_fingerprint = ?, finalized_at = ?
        WHERE id = 'dataset-fp'
        """,
        (fingerprint, now),
    )

    assert verify_dataset_fingerprint(db, "dataset-fp")["valid"] is True
    try:
        ArtifactRepository(db).update_dataset_item(
            "item-1", annotation_status="REJECTED"
        )
    except RuntimeError as exc:
        assert "immuable" in str(exc)
    else:
        raise AssertionError("Une version finalisée ne doit pas être modifiable.")

    label.write_text("0 0.4 0.4 0.1 0.1\n", encoding="utf-8")
    assert verify_dataset_fingerprint(db, "dataset-fp")["valid"] is False
