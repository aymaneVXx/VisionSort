from __future__ import annotations

from visionsort.database.db import VisionSortDB, utc_now


def activate_model(db: VisionSortDB, model_id: str) -> None:
    with db.connect() as conn:
        conn.execute("UPDATE model_registry SET is_active = 0, updated_at = ?", (utc_now(),))
        conn.execute("UPDATE model_registry SET is_active = 1, updated_at = ? WHERE id = ?", (utc_now(), model_id))


def rollback_to_previous_active(db: VisionSortDB) -> str | None:
    row = db.fetch_one("SELECT id FROM model_registry WHERE status = 'CHAMPION' ORDER BY updated_at DESC LIMIT 1")
    if row:
        activate_model(db, row["id"])
        return row["id"]
    return None
