from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
DEMO_DIR = DATA_DIR / "demo"
RUNTIME_DIR = DATA_DIR / "runtime"
PREVIEWS_DIR = RUNTIME_DIR / "previews"
DETAILS_DIR = RUNTIME_DIR / "details"
RECORDINGS_DIR = DATA_DIR / "recordings"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
DB_PATH = DATA_DIR / "visionsort.db"


def ensure_project_dirs() -> None:
    for path in [
        CONFIG_DIR,
        DATA_DIR,
        DEMO_DIR,
        RUNTIME_DIR,
        PREVIEWS_DIR,
        DETAILS_DIR,
        RECORDINGS_DIR,
        DATASETS_DIR,
        MODELS_DIR,
        LOGS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
        keep = path / ".gitkeep"
        if path.name in {"runtime", "recordings", "datasets", "models"} and not keep.exists():
            keep.write_text("", encoding="utf-8")
