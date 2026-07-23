from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from visionsort.core.paths import CONFIG_DIR, ROOT_DIR, ensure_project_dirs


DEFAULT_CONFIG = {
    "app": {
        "name": "VisionSort",
        "demo_mode": False,
        "site_validated": False,
        "timezone": "UTC",
    },
    "runtime": {
        "model_selection": "active_registry",
        "poll_interval_seconds": 1.0,
        "max_buffer_size": 3,
        "preview_jpeg_quality": 82,
        "recording_segment_seconds": 10,
        "details_flush_every": 10,
        "max_inference_queue": 8,
        "inference_result_ttl_seconds": 5.0,
    },
    "gpu": {
        "allow_training_while_inference": False,
        "max_concurrent_live_sources": 3,
        "device": "auto",
        "training_policy": "queue",
    },
    "model_promotion": {
        "require_frozen_test": True,
        "criteria": {
            "precision_min": 0.50,
            "recall_min": 0.50,
            "map50_min": 0.50,
            "count_accuracy_min": 0.80,
            "merge_rate_max": 0.15,
            "fps_min": 5.0,
        },
    },
    "tracking": {
        "site_topology": {
            "edges": [
                {"from_role": "C1", "to_role": "C2", "min_transit_s": 0.5, "max_transit_s": 10.0},
                {"from_role": "C2", "to_role": "C3", "min_transit_s": 0.2, "max_transit_s": 12.0},
            ]
        },
        "zones": {
            "C1": [{"zone_id": "c1_exit", "kind": "exit", "x1": 0.82, "y1": 0.0, "x2": 1.0, "y2": 1.0}],
            "C2": [
                {"zone_id": "c2_entry", "kind": "entry", "x1": 0.0, "y1": 0.0, "x2": 0.18, "y2": 1.0},
                {"zone_id": "c2_pick", "kind": "pick", "x1": 0.55, "y1": 0.1, "x2": 0.95, "y2": 0.9},
            ],
            "C3": [
                {"zone_id": "c3_entry", "kind": "entry", "x1": 0.0, "y1": 0.0, "x2": 0.18, "y2": 1.0},
                {"zone_id": "zone_A", "kind": "destination", "x1": 0.50, "y1": 0.15, "x2": 0.75, "y2": 0.50},
                {"zone_id": "zone_B", "kind": "destination", "x1": 0.62, "y1": 0.55, "x2": 0.93, "y2": 0.92},
            ],
        },
    },
}


@dataclass(slots=True)
class AppConfig:
    values: dict[str, Any] = field(default_factory=dict)

    def get(self, *path: str, default: Any = None) -> Any:
        cursor: Any = self.values
        for part in path:
            if not isinstance(cursor, dict):
                return default
            cursor = cursor.get(part)
            if cursor is None:
                return default
        return cursor

    @property
    def demo_mode(self) -> bool:
        env_flag = os.getenv("DEMO_MODE")
        if env_flag is not None:
            return env_flag.strip().lower() in {"1", "true", "yes", "on"}
        return bool(self.get("app", "demo_mode", default=False))


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            result[key] = merge_dict(base[key], value)
        else:
            result[key] = value
    return result


def ensure_default_config() -> Path:
    ensure_project_dirs()
    path = CONFIG_DIR / "default.yaml"
    if not path.exists():
        path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")
    return path


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else ensure_default_config()
    values = DEFAULT_CONFIG
    if config_path.exists():
        values = merge_dict(DEFAULT_CONFIG, yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    return AppConfig(values=values)


def relative_to_root(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT_DIR))
    except Exception:
        return str(path)
