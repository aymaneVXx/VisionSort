from __future__ import annotations

import os
import sys
from pathlib import Path

# Ajoute le répertoire racine du projet au sys.path pour l'exécution en standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

from visionsort.core.config import load_config
from visionsort.core.paths import ensure_project_dirs
from visionsort.database.db import VisionSortDB
from visionsort.database.repositories import ControlRepository
from visionsort.runtime.demo_assets import ensure_demo_assets


def main() -> int:
    ensure_project_dirs()
    config = load_config()
    db = VisionSortDB()
    db.initialize()
    if config.demo_mode:
        assets = ensure_demo_assets()
        repo = ControlRepository(db)
        if not repo.list_sources():
            for role, uri in assets.items():
                repo.upsert_source(
                    {
                        "name": f"Replay {role}",
                        "role": role,
                        "source_type": "REPLAY",
                        "uri": uri,
                        "model_id": "demo_synth_det",
                        "tracker_id": "greedy_iou",
                        "enabled": True,
                    }
                )
        print(f"DEMO_MODE actif: assets générés pour {', '.join(sorted(assets))}")
    else:
        print("Initialisation terminée sans bootstrap démo. Activez DEMO_MODE=1 pour créer les replays de démonstration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
