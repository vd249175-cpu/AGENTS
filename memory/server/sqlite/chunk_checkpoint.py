"""Checkpoint storage for migrated chunk runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SQLiteChunkCheckpoint:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_checkpoint (
                run_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, thread_id)
            )
            """
        )
        self._connection.commit()

    def save(self, *, run_id: str, thread_id: str, payload: dict[str, Any]) -> None:
        self._connection.execute(
            """
            INSERT INTO chunk_checkpoint (run_id, thread_id, payload_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id, thread_id) DO UPDATE SET
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at
            """,
            (
                run_id,
                thread_id,
                json.dumps(payload, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._connection.commit()

    def load(self, *, run_id: str, thread_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload_json FROM chunk_checkpoint WHERE run_id = ? AND thread_id = ?",
            (run_id, thread_id),
        ).fetchone()
        if row is None:
            return None
        return json.loads(str(row[0]))

    def close(self) -> None:
        self._connection.close()
