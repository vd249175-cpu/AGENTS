"""SQLite resume cache for chunk processing."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SQLiteChunkCache:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_cache (
                source_path TEXT NOT NULL,
                run_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                state_json TEXT NOT NULL,
                completed INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, document_name, thread_id)
            )
            """
        )
        self._connection.commit()

    def save(
        self,
        *,
        source_path: str,
        run_id: str,
        document_name: str,
        thread_id: str,
        state: dict[str, Any],
        completed: bool,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO chunk_cache (
                source_path, run_id, document_name, thread_id, state_json, completed, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, document_name, thread_id) DO UPDATE SET
                source_path = excluded.source_path,
                state_json = excluded.state_json,
                completed = excluded.completed,
                updated_at = excluded.updated_at
            """,
            (
                source_path,
                run_id,
                document_name,
                thread_id,
                json.dumps(state, ensure_ascii=False),
                1 if completed else 0,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._connection.commit()

    def load(self, *, document_name: str, run_id: str, thread_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            """
            SELECT source_path, state_json, completed, updated_at
            FROM chunk_cache
            WHERE document_name = ? AND run_id = ? AND thread_id = ?
            """,
            (document_name, run_id, thread_id),
        ).fetchone()
        if row is None:
            return None
        return {
            "source_path": str(row[0]),
            "state": json.loads(str(row[1])),
            "completed": bool(row[2]),
            "updated_at": str(row[3]),
        }

    def list_records(self, *, document_name: str, run_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT source_path, thread_id, state_json, completed, updated_at
            FROM chunk_cache
            WHERE document_name = ? AND run_id = ?
            ORDER BY updated_at
            """,
            (document_name, run_id),
        ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            records.append(
                {
                    "source_path": str(row[0]),
                    "thread_id": str(row[1]),
                    "state": json.loads(str(row[2])),
                    "completed": bool(row[3]),
                    "updated_at": str(row[4]),
                }
            )
        return records

    def close(self) -> None:
        self._connection.close()
