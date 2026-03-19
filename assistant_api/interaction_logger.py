import csv
import logging
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from .paths import TELEGRAM_LOG_DB_PATH
except ImportError:
    from paths import TELEGRAM_LOG_DB_PATH


logger = logging.getLogger(__name__)


class InteractionLogger:
    """Stores Telegram interactions in SQLite and exports them to CSV."""

    def __init__(self, db_path: str | Path = TELEGRAM_LOG_DB_PATH):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS telegram_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                username TEXT,
                source TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                from_cache INTEGER NOT NULL,
                response_time_ms INTEGER NOT NULL,
                created TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def log_interaction(
        self,
        user_id: str,
        username: str,
        source: str,
        query: str,
        response: str,
        from_cache: bool,
        response_time_ms: int,
    ) -> None:
        timestamp = datetime.now().isoformat(timespec="microseconds")
        created = timestamp

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO telegram_logs (
                timestamp,
                user_id,
                username,
                source,
                query,
                response,
                from_cache,
                response_time_ms,
                created
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                user_id,
                username,
                source,
                query,
                response,
                1 if from_cache else 0,
                response_time_ms,
                created,
            ),
        )
        conn.commit()
        conn.close()

    def export_csv(self, user_id: Optional[str] = None) -> Optional[Path]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if user_id:
            cursor.execute(
                """
                SELECT id, timestamp, user_id, username, source, query, response, from_cache, response_time_ms, created
                FROM telegram_logs
                WHERE user_id = ?
                ORDER BY id DESC
                """,
                (user_id,),
            )
        else:
            cursor.execute(
                """
                SELECT id, timestamp, user_id, username, source, query, response, from_cache, response_time_ms, created
                FROM telegram_logs
                ORDER BY id DESC
                """
            )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            suffix=".csv",
            delete=False,
        )

        with temp_file:
            writer = csv.writer(temp_file)
            writer.writerow(
                [
                    "id",
                    "timestamp",
                    "user_id",
                    "username",
                    "source",
                    "query",
                    "response",
                    "from_cache",
                    "response_time_ms",
                    "created",
                ]
            )
            writer.writerows(rows)

        return Path(temp_file.name)
