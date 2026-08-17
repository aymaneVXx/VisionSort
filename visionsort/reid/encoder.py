from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Protocol

import cv2
import numpy as np

from visionsort.core.paths import ROOT_DIR


BACKBONE_NAME = "mobilenet_v3_small_imagenet1k_v1"
BACKBONE_DIMENSION = 576


class ParcelReIDBackbone(Protocol):
    """Minimal contract consumed by the keyframe pipeline.

    Keeping this structural interface small lets a stronger frozen backbone be
    evaluated later without coupling it to tracking, scoring, or adaptation.
    """

    model_version: str

    def encode(self, crops: Iterable[np.ndarray]) -> np.ndarray:
        ...


def l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    denominator = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(denominator, 1.0e-12)


class ParcelReIDEncoder:
    """Frozen generic object encoder; it never downloads weights at runtime."""

    def __init__(self, weights_path: str | Path, *, device: str = "cpu") -> None:
        import torch
        from torchvision.models import mobilenet_v3_small

        path = Path(weights_path)
        if not path.is_absolute():
            path = ROOT_DIR / path
        path = path.resolve()
        if not path.is_file():
            raise RuntimeError(
                f"Poids ReID locaux introuvables: {path}. Aucun téléchargement runtime n'est autorisé."
            )
        self.weights_path = path
        resolved_device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        self.device = torch.device(resolved_device)
        if self.device.type == "cpu":
            # Camera workers run in separate processes. One thread per encoder
            # prevents severe oversubscription with two or three live sources.
            torch.set_num_threads(1)
        model = mobilenet_v3_small(weights=None)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.classifier = torch.nn.Identity()
        model.eval().to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        self.model_version = f"{BACKBONE_NAME}:{digest}"

    def encode(self, crops: Iterable[np.ndarray]) -> np.ndarray:
        import torch

        tensors = [self._preprocess(crop) for crop in crops]
        if not tensors:
            return np.empty((0, BACKBONE_DIMENSION), dtype=np.float32)
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            embeddings = self.model(batch)
        return l2_normalize(embeddings.detach().cpu().numpy())

    @staticmethod
    def _preprocess(crop: np.ndarray):
        import torch

        if crop is None or crop.size == 0:
            raise ValueError("Crop ReID vide.")
        letterboxed = ParcelReIDEncoder._letterbox(crop, size=224)
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return torch.from_numpy(array.transpose(2, 0, 1)).float()

    @staticmethod
    def _letterbox(crop: np.ndarray, *, size: int = 224) -> np.ndarray:
        """Resize without geometric distortion, then pad to a square canvas."""
        if crop is None or crop.size == 0:
            raise ValueError("Crop ReID vide.")
        height, width = crop.shape[:2]
        scale = min(float(size) / max(width, 1), float(size) / max(height, 1))
        resized_width = max(1, min(size, int(round(width * scale))))
        resized_height = max(1, min(size, int(round(height * scale))))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(
            crop,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
        canvas = np.full((size, size, crop.shape[2]), 114, dtype=crop.dtype)
        left = (size - resized_width) // 2
        top = (size - resized_height) // 2
        canvas[top : top + resized_height, left : left + resized_width] = resized
        return canvas


class ProjectionHead:
    """Versioned residual adapter over frozen backbone descriptors.

    ``matrix`` remains readable for backwards compatibility with the first PR3
    artifacts. New automatic adaptations use only the low-rank ``adapter_*``
    tensors and never train a dense D x D projection.
    """

    def __init__(
        self,
        matrix: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        *,
        adapter_down: np.ndarray | None = None,
        adapter_up: np.ndarray | None = None,
        version: str = "reid-projection-identity-v1",
    ) -> None:
        self.matrix = None if matrix is None else np.asarray(matrix, dtype=np.float32)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        self.adapter_down = (
            None if adapter_down is None else np.asarray(adapter_down, dtype=np.float32)
        )
        self.adapter_up = (
            None if adapter_up is None else np.asarray(adapter_up, dtype=np.float32)
        )
        if self.matrix is not None and self.adapter_down is not None:
            raise ValueError("Une projection dense et un residual adapter sont exclusifs.")
        if (self.adapter_down is None) != (self.adapter_up is None):
            raise ValueError("Les deux matrices du residual adapter sont obligatoires.")
        if self.adapter_down is not None:
            if self.adapter_down.ndim != 2 or self.adapter_up.ndim != 2:
                raise ValueError("Les matrices du residual adapter doivent etre 2D.")
            if self.adapter_up.shape != (
                self.adapter_down.shape[1],
                self.adapter_down.shape[0],
            ):
                raise ValueError("Dimensions incompatibles pour le residual adapter.")
            if self.bias is not None and self.bias.shape != (self.adapter_down.shape[1],):
                raise ValueError("Dimension de biais incompatible avec le residual adapter.")
        self.version = str(version)

    @classmethod
    def load(cls, path: str | Path, *, version: str | None = None) -> "ProjectionHead":
        with np.load(Path(path), allow_pickle=False) as artifact:
            stored_version = (
                str(artifact["version"].item()) if "version" in artifact else ""
            )
            bias = (
                np.asarray(artifact["bias"], dtype=np.float32)
                if "bias" in artifact
                else None
            )
            if "adapter_down" in artifact and "adapter_up" in artifact:
                return cls(
                    bias=bias,
                    adapter_down=np.asarray(artifact["adapter_down"], dtype=np.float32),
                    adapter_up=np.asarray(artifact["adapter_up"], dtype=np.float32),
                    version=version or stored_version or Path(path).stem,
                )
            matrix = np.asarray(artifact["matrix"], dtype=np.float32)
        return cls(matrix, bias, version=version or stored_version or Path(path).stem)

    @property
    def adapter_rank(self) -> int:
        return int(self.adapter_down.shape[0]) if self.adapter_down is not None else 0

    @property
    def trainable_parameter_count(self) -> int:
        if self.adapter_down is None or self.adapter_up is None:
            return 0
        bias_count = int(self.bias.size) if self.bias is not None else 0
        return int(self.adapter_down.size + self.adapter_up.size + bias_count)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if self.adapter_down is not None and self.adapter_up is not None:
            if values.shape[1] != self.adapter_down.shape[1]:
                raise ValueError(
                    "Dimension ReID incompatible: "
                    f"{values.shape[1]} != {self.adapter_down.shape[1]}."
                )
            hidden = np.tanh(values @ self.adapter_down.T)
            residual = hidden @ self.adapter_up.T
            if self.bias is not None:
                residual = residual + self.bias
            return l2_normalize(values + residual)
        if self.matrix is None:
            return l2_normalize(values)
        if values.shape[1] != self.matrix.shape[1]:
            raise ValueError(
                f"Dimension ReID incompatible: {values.shape[1]} != {self.matrix.shape[1]}."
            )
        projected = values @ self.matrix.T
        if self.bias is not None:
            projected = projected + self.bias
        return l2_normalize(projected)


def descriptor_embeddings(summary: dict[str, Any]) -> np.ndarray | None:
    descriptor = summary.get("appearance_descriptor")
    if isinstance(descriptor, dict):
        views = descriptor.get("embeddings") or []
        aggregate = descriptor.get("aggregate_embedding") or []
        values = list(views)
        if aggregate:
            values.append(aggregate)
        if values:
            return np.asarray(values, dtype=np.float32)
    legacy = summary.get("appearance_embedding")
    if legacy:
        return np.asarray([legacy], dtype=np.float32)
    return None


def descriptor_similarity(
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    projection: ProjectionHead,
) -> float | None:
    if not descriptor_is_reliable(left_summary) or not descriptor_is_reliable(
        right_summary
    ):
        return None
    left = descriptor_embeddings(left_summary)
    right = descriptor_embeddings(right_summary)
    if left is None or right is None:
        return None
    expected_dimension = (
        projection.adapter_down.shape[1]
        if projection.adapter_down is not None
        else projection.matrix.shape[1]
        if projection.matrix is not None
        else None
    )
    if left.shape[1] != right.shape[1]:
        if expected_dimension is not None:
            return None
        width = min(left.shape[1], right.shape[1])
        left, right = left[:, :width], right[:, :width]
    if expected_dimension is not None and (
        left.shape[1] != expected_dimension
        or right.shape[1] != expected_dimension
    ):
        # Legacy detector hints are not in the frozen-backbone feature space.
        return None
    left = projection.transform(left)
    right = projection.transform(right)
    pairwise = left @ right.T
    # Median of the best correspondence for each view is robust to one bad crop.
    directional = np.concatenate((pairwise.max(axis=1), pairwise.max(axis=0)))
    cosine = float(np.median(directional))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def descriptor_is_reliable(summary: dict[str, Any]) -> bool:
    descriptor = summary.get("appearance_descriptor")
    if not isinstance(descriptor, dict):
        return bool(summary.get("appearance_embedding"))
    try:
        view_count = int(descriptor.get("view_count") or 0)
        min_views = max(1, int(descriptor.get("min_views") or 3))
    except (TypeError, ValueError):
        return False
    declared = str(descriptor.get("descriptor_quality") or "").upper()
    return view_count >= min_views and declared != "LOW"
