from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from visionsort.core.paths import ROOT_DIR


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT_DIR / value


@dataclass(slots=True)
class PoseValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_instances: int = 0
    kpt_shape: list[int] | None = None
    label_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_pose_detections(
    detections: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    kpt_shape: tuple[int, int] = (17, 3),
) -> list[str]:
    errors: list[str] = []
    if not detections:
        return ["Aucune instance Pose n'est présente."]
    expected_keypoints, expected_components = kpt_shape
    for instance_index, detection in enumerate(detections):
        prefix = f"Instance {instance_index + 1}"
        if str(detection.get("class_name") or "") != "person":
            errors.append(f"{prefix}: la classe Pose doit être 'person'.")
        bbox = detection.get("bbox") or []
        if len(bbox) != 4:
            errors.append(f"{prefix}: bbox absente ou incomplète.")
        else:
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox]
            except (TypeError, ValueError):
                errors.append(f"{prefix}: bbox non numérique.")
            else:
                if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                    errors.append(f"{prefix}: bbox non finie.")
                elif (
                    x2 <= x1
                    or y2 <= y1
                    or x1 < 0
                    or y1 < 0
                    or x2 > width
                    or y2 > height
                ):
                    errors.append(
                        f"{prefix}: bbox hors limites ou de taille nulle."
                    )
        keypoints = detection.get("keypoints") or []
        if len(keypoints) != expected_keypoints:
            errors.append(
                f"{prefix}: {len(keypoints)} keypoint(s), "
                f"{expected_keypoints} requis."
            )
            continue
        indices = detection.get("keypoint_indices")
        if indices is not None and list(indices) != list(
            range(expected_keypoints)
        ):
            errors.append(
                f"{prefix}: ordre COCO incomplet ou incorrect."
            )
        visible = 0
        for keypoint_index, point in enumerate(keypoints):
            point_prefix = (
                f"{prefix}, keypoint {keypoint_index} "
                f"({COCO_KEYPOINT_NAMES[keypoint_index]})"
            )
            if len(point) != expected_components:
                errors.append(
                    f"{point_prefix}: {len(point)} composante(s), "
                    f"{expected_components} requises."
                )
                continue
            try:
                x, y, visibility = [float(value) for value in point]
            except (TypeError, ValueError):
                errors.append(f"{point_prefix}: valeur non numérique.")
                continue
            if not all(
                math.isfinite(value) for value in (x, y, visibility)
            ):
                errors.append(f"{point_prefix}: valeur non finie.")
                continue
            if x < 0 or y < 0 or x > width or y > height:
                errors.append(f"{point_prefix}: coordonnées hors limites.")
            if visibility < 0 or visibility > 2:
                errors.append(
                    f"{point_prefix}: visibilité/confiance hors de [0, 2]."
                )
            if visibility > 0:
                visible += 1
        if visible == 0:
            errors.append(f"{prefix}: aucun keypoint visible.")
    return errors


class PoseLabelValidator:
    """Validate YOLO Pose labels against the dataset data.yaml contract."""

    def __init__(self, data_yaml_path: str | Path):
        self.data_yaml_path = _absolute(data_yaml_path)

    def _configuration(
        self,
    ) -> tuple[list[int] | None, set[int], list[str]]:
        errors: list[str] = []
        if not self.data_yaml_path.is_file():
            return None, set(), [
                f"data.yaml introuvable: {self.data_yaml_path}"
            ]
        try:
            config = yaml.safe_load(
                self.data_yaml_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            return None, set(), [f"data.yaml illisible: {exc}"]
        if not isinstance(config, dict):
            return None, set(), ["data.yaml doit contenir un objet YAML."]
        raw_shape = config.get("kpt_shape")
        try:
            shape = (
                [int(value) for value in raw_shape]
                if isinstance(raw_shape, (list, tuple))
                and len(raw_shape) == 2
                else None
            )
        except (TypeError, ValueError):
            shape = None
        if shape != [17, 3]:
            errors.append(
                "kpt_shape doit être exactement [17, 3] "
                "pour le squelette COCO."
            )
        names = config.get("names")
        if isinstance(names, dict):
            try:
                classes = {int(key) for key in names}
            except (TypeError, ValueError):
                classes = set()
                errors.append(
                    "data.yaml contient des indices de classe invalides."
                )
        elif isinstance(names, list):
            classes = set(range(len(names)))
        else:
            classes = set()
            errors.append("data.yaml ne définit pas correctement names.")
        return shape, classes, errors

    def validate(
        self,
        label_path: str | Path,
        *,
        expected_instances: int | None = None,
    ) -> PoseValidationReport:
        path = _absolute(label_path)
        if not path.is_file():
            return PoseValidationReport(
                valid=False,
                errors=[f"Fichier label Pose introuvable: {path}"],
                label_path=str(path),
            )
        return self.validate_content(
            path.read_text(encoding="utf-8"),
            expected_instances=expected_instances,
            label_path=str(path),
        )

    def validate_content(
        self,
        content: str,
        *,
        expected_instances: int | None = None,
        label_path: str | None = None,
    ) -> PoseValidationReport:
        shape, classes, errors = self._configuration()
        lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]
        if not lines:
            errors.append("Le fichier label Pose est vide.")
        if expected_instances is not None and len(lines) != int(
            expected_instances
        ):
            errors.append(
                f"{len(lines)} instance(s) écrite(s), "
                f"{int(expected_instances)} attendue(s)."
            )
        expected_keypoints, components = (
            tuple(shape) if shape and len(shape) == 2 else (17, 3)
        )
        expected_values = 5 + expected_keypoints * components
        for line_index, line in enumerate(lines, start=1):
            tokens = line.split()
            if len(tokens) != expected_values:
                errors.append(
                    f"Ligne {line_index}: {len(tokens)} valeur(s), "
                    f"{expected_values} requises par kpt_shape."
                )
                continue
            try:
                class_value = float(tokens[0])
                values = [float(token) for token in tokens[1:]]
            except ValueError:
                errors.append(
                    f"Ligne {line_index}: valeur non numérique."
                )
                continue
            if not class_value.is_integer() or int(class_value) not in classes:
                errors.append(
                    f"Ligne {line_index}: classe {tokens[0]} invalide."
                )
            if not all(math.isfinite(value) for value in values):
                errors.append(
                    f"Ligne {line_index}: valeur non finie."
                )
                continue
            x_center, y_center, box_width, box_height = values[:4]
            if (
                not 0 <= x_center <= 1
                or not 0 <= y_center <= 1
                or box_width <= 0
                or box_height <= 0
                or box_width > 1
                or box_height > 1
                or x_center - box_width / 2 < 0
                or x_center + box_width / 2 > 1
                or y_center - box_height / 2 < 0
                or y_center + box_height / 2 > 1
            ):
                errors.append(
                    f"Ligne {line_index}: bbox normalisée invalide."
                )
            visible = 0
            keypoint_values = values[4:]
            for keypoint_index in range(expected_keypoints):
                offset = keypoint_index * components
                x, y = keypoint_values[offset : offset + 2]
                if not 0 <= x <= 1 or not 0 <= y <= 1:
                    errors.append(
                        f"Ligne {line_index}, keypoint "
                        f"{keypoint_index}: coordonnées hors de [0, 1]."
                    )
                if components == 3:
                    visibility = keypoint_values[offset + 2]
                    if visibility not in {0.0, 1.0, 2.0}:
                        errors.append(
                            f"Ligne {line_index}, keypoint "
                            f"{keypoint_index}: visibilité invalide."
                        )
                    if visibility > 0:
                        visible += 1
            if components == 3 and visible == 0:
                errors.append(
                    f"Ligne {line_index}: aucun keypoint visible."
                )
        return PoseValidationReport(
            valid=not errors,
            errors=errors,
            checked_instances=len(lines),
            kpt_shape=shape,
            label_path=label_path,
        )
