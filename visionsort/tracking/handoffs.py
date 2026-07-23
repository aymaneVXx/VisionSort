from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

from visionsort.database.db import VisionSortDB


class PendingHandoffBuffer:
    """Persistent, bounded event-time window for multicamera tracklets."""

    def __init__(
        self,
        db: VisionSortDB,
        topology_edges: list[dict[str, Any]],
        *,
        window_seconds: float = 0.75,
        max_items: int = 1000,
        expiry_seconds: float = 30.0,
    ):
        self.db = db
        self.topology_edges = topology_edges
        self.window_seconds = max(0.0, float(window_seconds))
        self.max_items = max(1, int(max_items))
        self.expiry_seconds = max(self.window_seconds, float(expiry_seconds))
        self._component_by_role = self._build_components(topology_edges)

    @staticmethod
    def _build_components(
        topology_edges: list[dict[str, Any]],
    ) -> dict[str, str]:
        neighbors: dict[str, set[str]] = defaultdict(set)
        for edge in topology_edges:
            left = str(edge["from_role"])
            right = str(edge["to_role"])
            neighbors[left].add(right)
            neighbors[right].add(left)
        components: dict[str, str] = {}
        for role in list(neighbors):
            if role in components:
                continue
            pending = [role]
            members: set[str] = set()
            while pending:
                current = pending.pop()
                if current in members:
                    continue
                members.add(current)
                pending.extend(neighbors[current] - members)
            key = "->".join(sorted(members))
            for member in members:
                components[member] = key
        return components

    def add(
        self, payload: dict[str, Any], *, received_at: float | None = None
    ) -> list[dict[str, Any]]:
        now = float(received_at if received_at is not None else time.time())
        role = str(payload.get("camera_role") or payload.get("camera_id") or "")
        component = self._component_by_role.get(role, role)
        link_key = f"{component}:{role}"
        self.db.execute(
            """
            INSERT OR IGNORE INTO pending_handoffs
            (tracklet_id, session_id, link_key, event_timestamp, received_at,
             expires_at, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(payload["tracklet_id"]),
                str(payload["session_id"]),
                link_key,
                float(payload["ended_at_global"]),
                now,
                now + self.expiry_seconds,
                json.dumps(payload),
            ),
        )
        count = self.pending_count()
        if count <= self.max_items:
            return []
        overflow_rows = self.db.fetch_all(
            """
            SELECT tracklet_id, payload_json FROM pending_handoffs
            ORDER BY received_at, event_timestamp, tracklet_id
            LIMIT ?
            """,
            (count - self.max_items,),
        )
        self.db.execute_many(
            "DELETE FROM pending_handoffs WHERE tracklet_id = ?",
            [(row["tracklet_id"],) for row in overflow_rows],
        )
        return [json.loads(row["payload_json"]) for row in overflow_rows]

    def pending_count(self, session_id: str | None = None) -> int:
        if session_id is None:
            row = self.db.fetch_one("SELECT COUNT(*) AS count FROM pending_handoffs")
        else:
            row = self.db.fetch_one(
                "SELECT COUNT(*) AS count FROM pending_handoffs WHERE session_id = ?",
                (session_id,),
            )
        return int((row["count"] if row else 0) or 0)

    def pop_ready_batches(
        self,
        *,
        now: float | None = None,
        force: bool = False,
        session_id: str | None = None,
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        wall_time = float(now if now is not None else time.time())
        if session_id is None:
            rows = [
                dict(row)
                for row in self.db.fetch_all(
                    "SELECT * FROM pending_handoffs ORDER BY received_at, event_timestamp"
                )
            ]
        else:
            rows = [
                dict(row)
                for row in self.db.fetch_all(
                    """
                    SELECT * FROM pending_handoffs
                    WHERE session_id = ? ORDER BY received_at, event_timestamp
                    """,
                    (session_id,),
                )
            ]
        if not rows:
            return []

        session_watermarks: dict[str, float] = defaultdict(float)
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            session = str(row["session_id"])
            session_watermarks[session] = max(
                session_watermarks[session], float(row["event_timestamp"])
            )
            groups[(session, str(row["link_key"]))].append(row)

        overflow = max(0, len(rows) - self.max_items)
        ready_ids: set[str] = set()
        ready_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for (session, _), grouped_rows in groups.items():
            oldest_received = min(float(row["received_at"]) for row in grouped_rows)
            newest_event = max(float(row["event_timestamp"]) for row in grouped_rows)
            expired = any(
                float(row["expires_at"]) <= wall_time for row in grouped_rows
            )
            event_window_closed = (
                session_watermarks[session] - newest_event >= self.window_seconds
            )
            wall_window_closed = (
                wall_time - oldest_received >= self.window_seconds
            )
            overflow_group = overflow > 0
            if (
                force
                or expired
                or event_window_closed
                or wall_window_closed
                or overflow_group
            ):
                for row in grouped_rows:
                    ready_ids.add(str(row["tracklet_id"]))
                    ready_by_session[session].append(
                        json.loads(row["payload_json"])
                    )
                overflow = max(0, overflow - len(grouped_rows))

        if ready_ids:
            self.db.execute_many(
                "DELETE FROM pending_handoffs WHERE tracklet_id = ?",
                [(tracklet_id,) for tracklet_id in sorted(ready_ids)],
            )
        return [
            (
                session,
                sorted(
                    payloads,
                    key=lambda payload: (
                        float(payload["ended_at_global"]),
                        str(payload["tracklet_id"]),
                    ),
                ),
            )
            for session, payloads in sorted(ready_by_session.items())
        ]
