from __future__ import annotations

from dataclasses import dataclass

from visionsort.core.config import load_config
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import ControlRepository


@dataclass(slots=True)
class UIContext:
    db: VisionSortDB
    repo: ControlRepository
    config_demo_mode: bool


def create_ui_context() -> UIContext:
    config = load_config()
    db = VisionSortDB()
    db.initialize()
    return UIContext(db=db, repo=ControlRepository(db), config_demo_mode=config.demo_mode)
