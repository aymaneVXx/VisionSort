from __future__ import annotations

import json
import sqlite3
from typing import Any

from visionsort.calibration.models import (
    DEFAULT_WORLD_CONVENTION,
    CalibrationProfile,
    CalibrationStatus,
    canonical_json,
)
from visionsort.calibration.geometry import WorldGeometry
from visionsort.core.site_config import validate_site_config
from visionsort.database.db import VisionSortDB, utc_now


class CalibrationRepository:
    def __init__(self, db: VisionSortDB):
        self.db = db

    def next_version(self, source_id: str) -> int:
        row = self.db.fetch_one(
            "SELECT COALESCE(MAX(version), 0) AS version FROM calibration_profiles WHERE source_id = ?",
            (str(source_id),),
        )
        return int(row["version"] if row else 0) + 1

    def save_profile(self, profile: CalibrationProfile | dict[str, Any]) -> str:
        value = (
            profile
            if isinstance(profile, CalibrationProfile)
            else CalibrationProfile.from_dict(profile)
        )
        source = self.db.fetch_one(
            "SELECT id FROM sources WHERE id = ?", (value.source_id,)
        )
        if source is None:
            raise RuntimeError(f"Source de calibration introuvable: {value.source_id}")
        payload = canonical_json(value.to_dict())
        try:
            self.db.execute(
                """
                INSERT INTO calibration_profiles
                (id, source_id, version, image_width, image_height, status,
                 fingerprint_sha256, profile_json, validated_on_site, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    value.profile_id,
                    value.source_id,
                    value.version,
                    value.image_width,
                    value.image_height,
                    value.status.value,
                    value.fingerprint_sha256,
                    payload,
                    value.created_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RuntimeError(
                "Profil immuable refuse: identifiant ou version deja utilise."
            ) from exc
        return value.profile_id

    @staticmethod
    def _from_row(row: Any) -> CalibrationProfile:
        profile = CalibrationProfile.from_dict(json.loads(row["profile_json"]))
        if profile.fingerprint_sha256 != str(row["fingerprint_sha256"]):
            raise RuntimeError(
                f"Profil {profile.profile_id} corrompu: hash SQLite incoherent."
            )
        return profile

    def get_profile(self, profile_id: str) -> CalibrationProfile | None:
        row = self.db.fetch_one(
            "SELECT * FROM calibration_profiles WHERE id = ?", (profile_id,)
        )
        return self._from_row(row) if row else None

    def list_profiles(self, source_id: str | None = None) -> list[CalibrationProfile]:
        if source_id is None:
            rows = self.db.fetch_all(
                "SELECT * FROM calibration_profiles ORDER BY source_id, version DESC"
            )
        else:
            rows = self.db.fetch_all(
                "SELECT * FROM calibration_profiles WHERE source_id = ? ORDER BY version DESC",
                (str(source_id),),
            )
        return [self._from_row(row) for row in rows]

    def get_active_profile(self, source_id: str) -> CalibrationProfile | None:
        row = self.db.fetch_one(
            """
            SELECT cp.*
            FROM source_calibration_assignments sca
            JOIN calibration_profiles cp ON cp.id = sca.calibration_profile_id
            WHERE sca.source_id = ?
            """,
            (str(source_id),),
        )
        return self._from_row(row) if row else None

    def activate_profile(self, source_id: str, profile_id: str) -> CalibrationProfile:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM calibration_profiles WHERE id = ?", (profile_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError("Profil de calibration introuvable.")
            profile = self._from_row(row)
            if profile.source_id != str(source_id):
                raise RuntimeError("Le profil appartient a une autre source.")
            if profile.status is not CalibrationStatus.VALID:
                raise RuntimeError(
                    "Seul un profil de calibration VALID peut etre active."
                )
            now = utc_now()
            conn.execute(
                """
                INSERT INTO source_calibration_assignments
                (source_id, calibration_profile_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    calibration_profile_id = excluded.calibration_profile_id,
                    updated_at = excluded.updated_at
                """,
                (str(source_id), profile.profile_id, now),
            )
            site_row = conn.execute(
                "SELECT config_json FROM site_config WHERE id = 'default'"
            ).fetchone()
            site_config = json.loads(site_row["config_json"] or "{}") if site_row else {}
            site_config["schema_version"] = max(
                2, int(site_config.get("schema_version") or 1)
            )
            site_config.setdefault(
                "world_coordinate_convention", DEFAULT_WORLD_CONVENTION
            )
            refs = site_config.setdefault("calibration_profiles", {}).setdefault(
                "active_by_source", {}
            )
            refs[str(source_id)] = profile.profile_id
            validated = validate_site_config(site_config)
            conn.execute(
                """
                INSERT INTO site_config (id, config_json, updated_at)
                VALUES ('default', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    config_json = excluded.config_json,
                    updated_at = excluded.updated_at
                """,
                (canonical_json(validated), now),
            )
        return profile

    def profile_snapshot(self, source_id: str) -> dict[str, Any]:
        profile = self.get_active_profile(source_id)
        return profile.to_dict() if profile else {}

    def active_geometry(self, source_id: str) -> WorldGeometry | None:
        profile = self.get_active_profile(source_id)
        return WorldGeometry(profile) if profile else None

    def session_geometry(
        self, session_id: str, source_id: str
    ) -> WorldGeometry | None:
        row = self.db.fetch_one(
            """
            SELECT calibration_profile_json, calibration_profile_hash
            FROM capture_session_sources
            WHERE session_id = ? AND source_id = ?
            """,
            (str(session_id), str(source_id)),
        )
        if row is None:
            raise RuntimeError("Source absente de la CaptureSession.")
        payload = json.loads(row["calibration_profile_json"] or "{}")
        if not payload:
            return None
        profile = CalibrationProfile.from_dict(payload)
        if profile.fingerprint_sha256 != str(
            row["calibration_profile_hash"] or ""
        ):
            raise RuntimeError("Snapshot de calibration de session corrompu.")
        return WorldGeometry(profile)
