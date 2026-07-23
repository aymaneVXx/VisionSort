from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml

from visionsort.annotations.validators import PoseLabelValidator
from visionsort.core.config import relative_to_root
from visionsort.core.enums import AnnotationStatus
from visionsort.core.paths import ROOT_DIR
from visionsort.database.db import VisionSortDB


TRAINABLE_STATUSES = {
    AnnotationStatus.AUTO_ACCEPTED.value,
    AnnotationStatus.HUMAN_VALIDATED.value,
}
REQUIRED_MANIFEST_COLUMNS = {
    "item_id",
    "session_id",
    "sample_group_id",
    "split",
    "source_id",
    "camera_role",
    "frame_index",
    "timestamp_global",
    "image_path",
    "label_path",
    "annotation_status",
}


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT_DIR / value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_lines(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(
                f"{path.name}, ligne {line_number}: JSON invalide ({exc})."
            )
            continue
        if not isinstance(value, dict):
            errors.append(
                f"{path.name}, ligne {line_number}: objet JSON attendu."
            )
            continue
        rows.append(value)
    return rows, errors


class DatasetIntegrityValidator:
    """Central validation gate for every persisted VisionSort dataset."""

    def __init__(self, db: VisionSortDB, dataset_id: str):
        self.db = db
        self.dataset_id = str(dataset_id)

    def validate_item_label(
        self,
        item_id: str,
        *,
        label_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        dataset_row = self.db.fetch_one(
            "SELECT * FROM datasets WHERE id = ?", (self.dataset_id,)
        )
        item_row = self.db.fetch_one(
            """
            SELECT * FROM dataset_items
            WHERE id = ? AND dataset_id = ?
            """,
            (item_id, self.dataset_id),
        )
        if dataset_row is None or item_row is None:
            return ["Dataset ou item introuvable."]
        dataset = dict(dataset_row)
        item = dict(item_row)
        if metadata is not None:
            item["metadata_json"] = json.dumps(metadata)
        task = str(dataset.get("task") or "detection")
        config_errors: list[str] = []
        _, class_ids = self._validate_data_yaml(
            dataset, task, config_errors, []
        )
        effective_label = label_path or item.get("label_path")
        if not effective_label:
            return [*config_errors, "Aucun fichier label n'est référencé."]
        path = _absolute(effective_label)
        if not path.is_file():
            return [*config_errors, f"Fichier label absent: {path}."]
        pose_validator = (
            PoseLabelValidator(str(dataset.get("data_yaml_path") or ""))
            if task == "pose"
            else None
        )
        return [
            *config_errors,
            *self._validate_label(
                task=task,
                item=item,
                label_path=path,
                class_ids=class_ids,
                pose_validator=pose_validator,
            ),
        ]

    def validate(self, *, write_report: bool = True) -> dict[str, Any]:
        dataset_row = self.db.fetch_one(
            "SELECT * FROM datasets WHERE id = ?", (self.dataset_id,)
        )
        if dataset_row is None:
            raise RuntimeError("Dataset introuvable.")
        dataset = dict(dataset_row)
        task = str(dataset.get("task") or "detection")
        root = _absolute(str(dataset["root_path"]))
        report_path = root / "integrity_report.json"
        errors: list[str] = []
        warnings: list[str] = []
        missing_files: list[str] = []
        invalid_labels: list[dict[str, Any]] = []
        checked_items = 0

        items = [
            dict(row)
            for row in self.db.fetch_all(
                """
                SELECT * FROM dataset_items
                WHERE dataset_id = ? ORDER BY id
                """,
                (self.dataset_id,),
            )
        ]
        if not items:
            errors.append("Le dataset ne contient aucun item.")
        trainable_items = [
            item
            for item in items
            if item["annotation_status"] in TRAINABLE_STATUSES
        ]
        pending = [
            str(item["id"])
            for item in items
            if item["annotation_status"]
            == AnnotationStatus.NEEDS_REVIEW.value
        ]
        if pending:
            errors.append(
                f"{len(pending)} item(s) attendent encore une revue humaine."
            )

        data_yaml, class_ids = self._validate_data_yaml(
            dataset, task, errors, missing_files
        )
        pose_validator = (
            PoseLabelValidator(str(dataset.get("data_yaml_path") or ""))
            if task == "pose"
            else None
        )
        actual_image_hashes: dict[str, set[str]] = defaultdict(set)
        item_by_id = {str(item["id"]): item for item in items}
        for item in items:
            checked_items += 1
            item_id = str(item["id"])
            split = str(item.get("split") or "")
            if split not in {"train", "val", "test"}:
                errors.append(
                    f"Item {item_id}: split invalide ou absent ({split!r})."
                )
            image_path = _absolute(str(item["image_path"]))
            if not image_path.is_file():
                missing_files.append(str(image_path))
                errors.append(f"Item {item_id}: image absente.")
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                errors.append(f"Item {item_id}: image illisible.")
                continue
            actual_image_hashes[_file_sha256(image_path)].add(split)

            label_value = item.get("label_path")
            label_expected = item["annotation_status"] in TRAINABLE_STATUSES
            if not label_value:
                if label_expected:
                    missing_files.append(f"label:{item_id}")
                    errors.append(f"Item {item_id}: label attendu absent.")
                continue
            label_path = _absolute(str(label_value))
            if not label_path.is_file():
                missing_files.append(str(label_path))
                errors.append(f"Item {item_id}: fichier label absent.")
                continue
            label_errors = self._validate_label(
                task=task,
                item=item,
                label_path=label_path,
                class_ids=class_ids,
                pose_validator=pose_validator,
            )
            if label_errors:
                invalid_labels.append(
                    {"item_id": item_id, "errors": label_errors}
                )
                errors.extend(
                    f"Item {item_id}: {message}"
                    for message in label_errors
                )

        for image_hash, splits in actual_image_hashes.items():
            if len(splits) > 1:
                errors.append(
                    "Image dupliquée entre splits "
                    f"({image_hash[:12]}: {sorted(splits)})."
                )

        manifest_rows = self._validate_manifest(
            dataset,
            trainable_items,
            item_by_id,
            errors,
            missing_files,
        )
        split_report = self._validate_splits(items, trainable_items, errors)
        self._validate_sessions(items, errors)
        self._validate_task_artifacts(
            task, root, errors, warnings, missing_files
        )

        report = {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "checked_items": checked_items,
            "missing_files": sorted(set(missing_files)),
            "invalid_labels": invalid_labels,
            "split_counts": split_report["split_counts"],
            "task": task,
            "report_path": relative_to_root(report_path),
            "manifest_rows": manifest_rows,
            "trainable_items": len(trainable_items),
            "required_splits": split_report["required_splits"],
            "data_yaml": data_yaml,
        }
        if write_report:
            root.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return report

    def _validate_data_yaml(
        self,
        dataset: dict[str, Any],
        task: str,
        errors: list[str],
        missing_files: list[str],
    ) -> tuple[dict[str, Any], set[int]]:
        value = dataset.get("data_yaml_path")
        if not value:
            errors.append("data.yaml n'est pas référencé.")
            missing_files.append("data.yaml")
            return {}, set()
        path = _absolute(str(value))
        if not path.is_file():
            errors.append("data.yaml est absent.")
            missing_files.append(str(path))
            return {}, set()
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"data.yaml est illisible: {exc}")
            return {}, set()
        if not isinstance(loaded, dict):
            errors.append("data.yaml doit contenir un objet YAML.")
            return {}, set()
        if str(loaded.get("task") or task) != task:
            errors.append(
                "La tâche de data.yaml ne correspond pas au dataset."
            )
        for split in ("train", "val", "test"):
            if not isinstance(loaded.get(split), str):
                errors.append(
                    f"data.yaml ne définit pas correctement {split}."
                )
        names = loaded.get("names")
        if isinstance(names, dict):
            try:
                class_ids = {int(key) for key in names}
            except (TypeError, ValueError):
                class_ids = set()
        elif isinstance(names, list):
            class_ids = set(range(len(names)))
        else:
            class_ids = set()
        if not class_ids:
            errors.append("data.yaml ne définit aucune classe valide.")
        if task == "pose" and loaded.get("kpt_shape") != [17, 3]:
            errors.append(
                "data.yaml doit définir kpt_shape: [17, 3]."
            )
        return loaded, class_ids

    def _validate_label(
        self,
        *,
        task: str,
        item: dict[str, Any],
        label_path: Path,
        class_ids: set[int],
        pose_validator: PoseLabelValidator | None,
    ) -> list[str]:
        metadata = json.loads(item.get("metadata_json") or "{}")
        expected = metadata.get("expected_label_count")
        if expected is None:
            expected = metadata.get("instance_count")
        if expected is None:
            expected = metadata.get("pseudo_label_count")
        if task == "pose":
            if pose_validator is None:
                return ["Validateur Pose indisponible."]
            return pose_validator.validate(
                label_path,
                expected_instances=(
                    int(expected) if expected is not None else None
                ),
            ).errors
        lines = [
            line.strip()
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        errors: list[str] = []
        if expected is not None and int(expected) > 0 and not lines:
            errors.append("Le label est vide malgré une instance attendue.")
        if expected is not None and len(lines) != int(expected):
            errors.append(
                f"{len(lines)} instance(s), {int(expected)} attendue(s)."
            )
        for line_number, line in enumerate(lines, start=1):
            tokens = line.split()
            try:
                values = [float(token) for token in tokens]
            except ValueError:
                errors.append(
                    f"Ligne {line_number}: valeur non numérique."
                )
                continue
            if not values or not all(math.isfinite(value) for value in values):
                errors.append(
                    f"Ligne {line_number}: valeur non finie ou absente."
                )
                continue
            if (
                not values[0].is_integer()
                or int(values[0]) not in class_ids
            ):
                errors.append(f"Ligne {line_number}: classe invalide.")
                continue
            if task in {
                "detection",
                "local_tracking",
                "reid_multicamera",
            }:
                if len(values) != 5:
                    errors.append(
                        f"Ligne {line_number}: bbox YOLO attendue."
                    )
                    continue
                if not self._valid_bbox(values[1:5]):
                    errors.append(
                        f"Ligne {line_number}: bbox normalisée invalide."
                    )
            elif task == "segmentation":
                coordinates = values[1:]
                if len(coordinates) < 6 or len(coordinates) % 2:
                    errors.append(
                        f"Ligne {line_number}: polygone incomplet."
                    )
                    continue
                if any(value < 0 or value > 1 for value in coordinates):
                    errors.append(
                        f"Ligne {line_number}: polygone hors limites."
                    )
                    continue
                points = list(zip(coordinates[::2], coordinates[1::2]))
                area = abs(
                    sum(
                        points[index - 1][0] * points[index][1]
                        - points[index][0] * points[index - 1][1]
                        for index in range(len(points))
                    )
                ) / 2.0
                if area <= 0:
                    errors.append(
                        f"Ligne {line_number}: aire du polygone nulle."
                    )
            else:
                errors.append(f"Tâche de label non supportée: {task}.")
        observations = metadata.get("observations") or metadata.get(
            "pseudo_labels"
        ) or []
        if task == "local_tracking":
            identities = [
                (
                    observation.get("camera_id")
                    or item.get("source_id"),
                    observation.get("local_track_id"),
                )
                for observation in observations
                if observation.get("class_name") == "parcel"
            ]
            if not identities or any(
                camera_id in {None, ""}
                or local_track_id is None
                for camera_id, local_track_id in identities
            ):
                errors.append(
                    "Identité locale de tracking absente ou invalide."
                )
        elif task == "reid_multicamera":
            parcel_observations = [
                observation
                for observation in observations
                if observation.get("class_name") == "parcel"
            ]
            if not parcel_observations or any(
                not observation.get("global_parcel_id")
                for observation in parcel_observations
            ):
                errors.append(
                    "global_parcel_id absent pour une instance ReID."
                )
        return errors

    @staticmethod
    def _valid_bbox(values: list[float]) -> bool:
        if len(values) != 4:
            return False
        x, y, width, height = values
        return (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < width <= 1
            and 0 < height <= 1
            and x - width / 2 >= 0
            and x + width / 2 <= 1
            and y - height / 2 >= 0
            and y + height / 2 <= 1
        )

    def _validate_manifest(
        self,
        dataset: dict[str, Any],
        trainable_items: list[dict[str, Any]],
        item_by_id: dict[str, dict[str, Any]],
        errors: list[str],
        missing_files: list[str],
    ) -> int:
        value = dataset.get("manifest_path")
        if not value:
            errors.append("manifest.csv n'est pas référencé.")
            missing_files.append("manifest.csv")
            return 0
        path = _absolute(str(value))
        if not path.is_file():
            errors.append("manifest.csv est absent.")
            missing_files.append(str(path))
            return 0
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = set(reader.fieldnames or [])
                rows = list(reader)
        except Exception as exc:
            errors.append(f"manifest.csv est illisible: {exc}")
            return 0
        missing_columns = REQUIRED_MANIFEST_COLUMNS - columns
        if missing_columns:
            errors.append(
                "manifest.csv: colonnes absentes "
                f"{sorted(missing_columns)}."
            )
        row_ids = [str(row.get("item_id") or "") for row in rows]
        if len(row_ids) != len(set(row_ids)):
            errors.append("manifest.csv contient des item_id dupliqués.")
        expected_ids = {str(item["id"]) for item in trainable_items}
        if set(row_ids) != expected_ids:
            errors.append(
                "manifest.csv ne correspond pas exactement aux items "
                "entraînables."
            )
        for row in rows:
            item = item_by_id.get(str(row.get("item_id") or ""))
            if item is None:
                errors.append(
                    f"manifest.csv référence un item inconnu: "
                    f"{row.get('item_id')}."
                )
                continue
            for field in (
                "session_id",
                "split",
                "source_id",
                "image_path",
                "label_path",
                "annotation_status",
            ):
                expected = str(item.get(field) or "")
                if str(row.get(field) or "") != expected:
                    errors.append(
                        f"manifest.csv, item {item['id']}: "
                        f"{field} incohérent."
                    )
        return len(rows)

    def _validate_splits(
        self,
        items: list[dict[str, Any]],
        trainable_items: list[dict[str, Any]],
        errors: list[str],
    ) -> dict[str, Any]:
        from visionsort.datasets.pipeline import validate_dataset_splits

        legacy = validate_dataset_splits(self.db, self.dataset_id)
        if not legacy["valid"]:
            errors.append(f"Fuite entre splits: {legacy['leaks']}")
        split_counts = {
            split: sum(
                str(item.get("split")) == split
                for item in trainable_items
            )
            for split in ("train", "val", "test")
        }
        linked = [
            dict(row)
            for row in self.db.fetch_all(
                """
                SELECT ds.split, cs.demo_mode
                FROM dataset_sessions ds
                LEFT JOIN capture_sessions cs ON cs.id = ds.session_id
                WHERE ds.dataset_id = ?
                """,
                (self.dataset_id,),
            )
        ]
        if any(
            row.get("demo_mode") is not None
            and not bool(row["demo_mode"])
            for row in linked
        ):
            required_splits = ["train", "val", "test"]
        else:
            required_splits = sorted(
                {
                    str(row["split"])
                    for row in linked
                    if str(row.get("split") or "")
                    in {"train", "val", "test"}
                }
                or {
                    str(item.get("split"))
                    for item in items
                    if str(item.get("split") or "")
                    in {"train", "val", "test"}
                }
            )
        if not required_splits:
            errors.append("Aucun split requis n'est défini.")
        for split in required_splits:
            if split_counts.get(split, 0) <= 0:
                errors.append(
                    f"Le split requis {split} ne contient aucun item "
                    "entraînable."
                )
        return {
            "split_counts": split_counts,
            "required_splits": required_splits,
        }

    def _validate_sessions(
        self, items: list[dict[str, Any]], errors: list[str]
    ) -> None:
        referenced = {
            str(row["session_id"])
            for row in self.db.fetch_all(
                """
                SELECT session_id FROM dataset_sessions
                WHERE dataset_id = ?
                """,
                (self.dataset_id,),
            )
        }
        referenced.update(
            str(item["session_id"])
            for item in items
            if item.get("session_id")
        )
        for session_id in sorted(referenced):
            if self.db.fetch_one(
                "SELECT id FROM capture_sessions WHERE id = ?",
                (session_id,),
            ) is None:
                errors.append(
                    f"CaptureSession référencée absente: {session_id}."
                )

    def _validate_task_artifacts(
        self,
        task: str,
        root: Path,
        errors: list[str],
        warnings: list[str],
        missing_files: list[str],
    ) -> None:
        if task == "local_tracking":
            path = root / "tracking_manifest.jsonl"
            if not path.is_file():
                missing_files.append(str(path))
                errors.append("tracking_manifest.jsonl est absent.")
                return
            rows, parse_errors = _json_lines(path)
            errors.extend(parse_errors)
            if not rows:
                errors.append("tracking_manifest.jsonl est vide.")
            for index, row in enumerate(rows, start=1):
                identity = row.get("track_identity")
                if (
                    not isinstance(identity, list)
                    or len(identity) != 2
                    or identity[0] in {None, ""}
                    or identity[1] is None
                ):
                    errors.append(
                        f"tracking_manifest ligne {index}: identité absente."
                    )
        elif task == "reid_multicamera":
            manifest = root / "reid_manifest.jsonl"
            pairs = root / "reid_pairs.jsonl"
            for path in (manifest, pairs):
                if not path.is_file():
                    missing_files.append(str(path))
                    errors.append(f"{path.name} est absent.")
            if not manifest.is_file() or not pairs.is_file():
                return
            rows, manifest_errors = _json_lines(manifest)
            pair_rows, pair_errors = _json_lines(pairs)
            errors.extend(manifest_errors)
            errors.extend(pair_errors)
            identities = {
                str(row.get("global_parcel_id") or "") for row in rows
            }
            if "" in identities:
                errors.append(
                    "reid_manifest contient une identité globale absente."
                )
            crops_by_path: dict[str, dict[str, Any]] = {}
            for row in rows:
                crop = row.get("crop_path")
                if not crop or not _absolute(str(crop)).is_file():
                    missing_files.append(str(crop or "crop_absent"))
                    errors.append("Un crop ReID référencé est absent.")
                elif crop:
                    crops_by_path[str(crop)] = row
            positive = 0
            negative = 0
            for row in pair_rows:
                label = row.get("label")
                if label == 1:
                    positive += 1
                elif label == 0:
                    negative += 1
                else:
                    errors.append("Une paire ReID possède un label invalide.")
                for field in ("left_crop", "right_crop"):
                    if not row.get(field) or not _absolute(
                        str(row[field])
                    ).is_file():
                        errors.append(
                            f"Paire ReID: {field} absent."
                        )
                left = crops_by_path.get(str(row.get("left_crop") or ""))
                right = crops_by_path.get(
                    str(row.get("right_crop") or "")
                )
                if left is None or right is None:
                    errors.append(
                        "Une paire ReID référence un crop hors manifest."
                    )
                elif label == 1 and (
                    left.get("global_parcel_id")
                    != right.get("global_parcel_id")
                    or left.get("camera_id") == right.get("camera_id")
                ):
                    errors.append(
                        "Une paire ReID positive est incohérente."
                    )
                elif label == 0 and (
                    left.get("global_parcel_id")
                    == right.get("global_parcel_id")
                ):
                    errors.append(
                        "Une paire ReID négative est incohérente."
                    )
            if positive <= 0 or negative <= 0:
                errors.append(
                    "Les paires ReID positives et négatives sont requises."
                )
        elif task not in {"detection", "segmentation", "pose"}:
            warnings.append(
                f"Aucun validateur d'artefact spécialisé pour {task}."
            )
