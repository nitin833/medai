"""SQLite patient record storage."""

import json
import sqlite3
from datetime import datetime, timezone

from app.config import PATIENT_DB


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(PATIENT_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_patient(patient_info: dict) -> int:
    init_db()
    conn = _connect()
    now = datetime.now(timezone.utc).isoformat()
    cursor = conn.execute(
        "INSERT INTO patients (data, created_at) VALUES (?, ?)",
        (json.dumps(patient_info), now),
    )
    conn.commit()
    patient_id = cursor.lastrowid
    conn.close()
    return patient_id


def get_patient(patient_id: int) -> dict | None:
    init_db()
    conn = _connect()
    row = conn.execute("SELECT data FROM patients WHERE id = ?", (patient_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return json.loads(row["data"])


def list_patients(limit: int = 50) -> list[dict]:
    init_db()
    conn = _connect()
    rows = conn.execute(
        "SELECT id, data, created_at FROM patients ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {"id": row["id"], "created_at": row["created_at"], **json.loads(row["data"])}
        for row in rows
    ]
