from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from visionsort.core.paths import ROOT_DIR


BACKBONE_NAME = "mobilenet_v3_small_imagenet1k_v1"
BACKBONE_DIMENSION = 576


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
        resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return torch.from_numpy(array.transpose(2, 0, 1)).float()


class ProjectionHead:
    """Small versioned linear head applied to frozen backbone descriptors."""

    def __init__(
        self,
        matrix: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        *,
        version: str = "reid-projection-identity-v1",
    ) -> None:
        self.matrix = None if matrix is None else np.asarray(matrix, dtype=np.float32)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        self.version = str(version)

    @classmethod
    def load(cls, path: str | Path, *, version: str | None = None) -> "ProjectionHead":
        with np.load(Path(path), allow_pickle=False) as artifact:
            stored_version = (
                str(artifact["version"].item()) if "version" in artifact else ""
            )
            matrix = np.asarray(artifact["matrix"], dtype=np.float32)
            bias = np.asarray(artifact["bias"], dtype=np.float32)
        return cls(matrix, bias, version=version or stored_version or Path(path).stem)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
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
    left = descriptor_embeddings(left_summary)
    right = descriptor_embeddings(right_summary)
    if left is None or right is None:
        return None
    if left.shape[1] != right.shape[1]:
        if projection.matrix is not None:
            return None
        width = min(left.shape[1], right.shape[1])
        left, right = left[:, :width], right[:, :width]
    if projection.matrix is not None and (
        left.shape[1] != projection.matrix.shape[1]
        or right.shape[1] != projection.matrix.shape[1]
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
