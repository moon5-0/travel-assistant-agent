"""基于 SQLite 的长期记忆仓库。

SQLite 是 Python 标准库自带的嵌入式关系型数据库，不需要单独启动服务。
本实现把多个用户的数据放在同一个 ``memory.sqlite3`` 文件中，并在每条
SQL 查询中使用 ``user_id`` 隔离数据。
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

from .memory_repository import LongTermMemoryRepository


logger = logging.getLogger(__name__)

COLLECTION_PREFERENCE_TYPES = frozenset({"hotel_brands", "airlines"})
PREFERENCE_TYPE_ALIASES = {"default_departure": "home_location"}


class SQLiteMemoryRepository(LongTermMemoryRepository):
    """将用户偏好、聊天记录和历史行程持久化到 SQLite。"""

    DATABASE_NAME = "memory.sqlite3"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        user_id: str,
        storage_path: str = "data/memory",
        db_path: Optional[str] = None,
    ):
        self.user_id = user_id
        storage_dir = Path(storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(Path(db_path) if db_path else storage_dir / self.DATABASE_NAME)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
        logger.info("SQLite memory initialized for user: %s", user_id)

    @contextmanager
    def _connect(
        self,
        *,
        immediate: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        """为一次仓库操作创建连接，并统一设置 SQLite 行为。"""
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            # sqlite3.Connection 的上下文会自动提交；发生异常则自动回滚。
            with connection:
                # 需要“先读后写”的操作提前取得写锁，避免两个进程同时读取
                # 旧值后相互覆盖。普通只读和单条写入无需使用该模式。
                if immediate:
                    connection.execute("BEGIN IMMEDIATE")
                yield connection
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        """创建第一版表结构；重复执行不会覆盖已有数据。"""
        with self._connect() as connection:
            # WAL 允许读取和写入更好地并发，适合多个会话共享同一数据库。
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT NOT NULL,
                    preference_type TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, preference_type)
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chat_user_id
                ON chat_messages (user_id, id);

                CREATE INDEX IF NOT EXISTS idx_chat_user_session_id
                ON chat_messages (user_id, session_id, id);

                CREATE TABLE IF NOT EXISTS trip_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    trip_id TEXT NOT NULL,
                    identity_key TEXT,
                    origin TEXT,
                    destination TEXT,
                    start_date TEXT,
                    trip_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    UNIQUE (user_id, trip_id),
                    UNIQUE (user_id, identity_key)
                );

                CREATE INDEX IF NOT EXISTS idx_trip_user_id
                ON trip_history (user_id, id);

                CREATE INDEX IF NOT EXISTS idx_trip_destination
                ON trip_history (user_id, destination);

                CREATE TABLE IF NOT EXISTS user_statistics (
                    user_id TEXT PRIMARY KEY,
                    total_queries INTEGER NOT NULL DEFAULT 0
                );

                """
            )
            connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _dump_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _load_json(value: str) -> Any:
        return json.loads(value)

    @staticmethod
    def _normalize_preference_type(pref_type: str) -> str:
        return PREFERENCE_TYPE_ALIASES.get(pref_type, pref_type)

    @staticmethod
    def _normalize_preference_value(pref_type: str, value: Any) -> Any:
        if pref_type not in COLLECTION_PREFERENCE_TYPES:
            return value

        values = value if isinstance(value, list) else [value]
        normalized = []
        for item in values:
            if item in (None, "") or item in normalized:
                continue
            normalized.append(item)
        return normalized

    @staticmethod
    def _trip_identity(trip_info: Dict[str, Any]) -> Optional[str]:
        """用出发地、目的地和出发日期生成行程业务唯一键。"""
        values = []
        for field in ("origin", "destination", "start_date"):
            value = trip_info.get(field)
            if value in (None, ""):
                return None
            values.append(str(value).strip().casefold())
        return json.dumps(values, ensure_ascii=False, separators=(",", ":"))

    def _upsert_preference(
        self,
        connection: sqlite3.Connection,
        pref_type: str,
        value: Any,
    ) -> None:
        connection.execute(
            """
            INSERT INTO preferences (
                user_id, preference_type, value_json, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, preference_type) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            (self.user_id, pref_type, self._dump_json(value), self._now()),
        )

    def save_preference(self, pref_type: str, value: Any) -> bool:
        pref_type = self._normalize_preference_type(pref_type)
        value = self._normalize_preference_value(pref_type, value)

        with self._connect(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT value_json FROM preferences
                WHERE user_id = ? AND preference_type = ?
                """,
                (self.user_id, pref_type),
            ).fetchone()
            if row is not None and self._load_json(row["value_json"]) == value:
                return False
            self._upsert_preference(connection, pref_type, value)
        return True

    def get_preference(self, pref_type: Optional[str] = None) -> Any:
        with self._connect() as connection:
            if pref_type is not None:
                pref_type = self._normalize_preference_type(pref_type)
                row = connection.execute(
                    """
                    SELECT value_json FROM preferences
                    WHERE user_id = ? AND preference_type = ?
                    """,
                    (self.user_id, pref_type),
                ).fetchone()
                return self._load_json(row["value_json"]) if row else None

            rows = connection.execute(
                """
                SELECT preference_type, value_json
                FROM preferences
                WHERE user_id = ?
                ORDER BY rowid
                """,
                (self.user_id,),
            ).fetchall()
        return {
            row["preference_type"]: self._load_json(row["value_json"])
            for row in rows
        }

    def _append_collection_preference(self, pref_type: str, value: str) -> None:
        """在一个事务内完成读取、追加和写回，避免产生重复值。"""
        with self._connect(immediate=True) as connection:
            row = connection.execute(
                """
                SELECT value_json FROM preferences
                WHERE user_id = ? AND preference_type = ?
                """,
                (self.user_id, pref_type),
            ).fetchone()
            current = self._load_json(row["value_json"]) if row else []
            current = self._normalize_preference_value(pref_type, current)
            if value not in current:
                current.append(value)
                self._upsert_preference(connection, pref_type, current)

    def add_hotel_brand(self, brand: str) -> None:
        self._append_collection_preference("hotel_brands", brand)

    def add_airline(self, airline: str) -> None:
        self._append_collection_preference("airlines", airline)

    def add_chat_message(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_messages (
                    user_id, session_id, role, content, timestamp
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (self.user_id, session_id, role, content, self._now()),
            )

    def get_chat_history(
        self,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        filters = ["user_id = ?"]
        params: List[Any] = [self.user_id]
        if session_id is not None:
            filters.append("session_id = ?")
            params.append(session_id)

        if limit:
            sql = f"""
                SELECT role, content, timestamp, session_id
                FROM chat_messages
                WHERE {' AND '.join(filters)}
                ORDER BY id DESC
                LIMIT ?
            """
            params.append(limit)
        else:
            sql = f"""
                SELECT role, content, timestamp, session_id
                FROM chat_messages
                WHERE {' AND '.join(filters)}
                ORDER BY id ASC
            """

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        if limit:
            rows = list(reversed(rows))
        return [dict(row) for row in rows]

    def _insert_trip(
        self,
        connection: sqlite3.Connection,
        trip_info: Dict[str, Any],
        *,
        preserve_metadata: bool = False,
    ) -> None:
        record = dict(trip_info)
        if preserve_metadata and record.get("trip_id"):
            trip_id = str(record["trip_id"])
        else:
            trip_id = f"trip_{uuid4().hex[:12]}"
        if preserve_metadata and record.get("timestamp"):
            timestamp = str(record["timestamp"])
        else:
            timestamp = self._now()
        record["trip_id"] = trip_id
        record["timestamp"] = timestamp

        connection.execute(
            """
            INSERT INTO trip_history (
                user_id, trip_id, identity_key, origin, destination,
                start_date, trip_json, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.user_id,
                trip_id,
                self._trip_identity(record),
                record.get("origin"),
                record.get("destination"),
                record.get("start_date"),
                self._dump_json(record),
                timestamp,
            ),
        )

    def _save_trip_in_transaction(
        self,
        connection: sqlite3.Connection,
        trip_info: Dict[str, Any],
        *,
        preserve_metadata: bool = False,
    ) -> bool:
        identity_key = self._trip_identity(trip_info)
        if identity_key is not None:
            row = connection.execute(
                """
                SELECT id, trip_json FROM trip_history
                WHERE user_id = ? AND identity_key = ?
                """,
                (self.user_id, identity_key),
            ).fetchone()
            if row is not None:
                existing = self._load_json(row["trip_json"])
                changed = False
                for key, value in trip_info.items():
                    if key in {"trip_id", "timestamp"}:
                        continue
                    if value not in (None, "", []) and existing.get(key) != value:
                        existing[key] = value
                        changed = True
                if changed:
                    connection.execute(
                        """
                        UPDATE trip_history SET
                            origin = ?, destination = ?, start_date = ?,
                            trip_json = ?
                        WHERE id = ?
                        """,
                        (
                            existing.get("origin"),
                            existing.get("destination"),
                            existing.get("start_date"),
                            self._dump_json(existing),
                            row["id"],
                        ),
                    )
                return False

        self._insert_trip(
            connection,
            trip_info,
            preserve_metadata=preserve_metadata,
        )
        return True

    def save_trip_history(self, trip_info: Dict[str, Any]) -> bool:
        with self._connect(immediate=True) as connection:
            return self._save_trip_in_transaction(connection, trip_info)

    def get_trip_history(
        self,
        limit: Optional[int] = 10,
    ) -> List[Dict[str, Any]]:
        params: List[Any] = [self.user_id]
        if limit:
            sql = """
                SELECT trip_json FROM trip_history
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
            """
            params.append(limit)
        else:
            sql = """
                SELECT trip_json FROM trip_history
                WHERE user_id = ?
                ORDER BY id ASC
            """

        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        if limit:
            rows = list(reversed(rows))
        return [self._load_json(row["trip_json"]) for row in rows]

    def get_frequent_destinations(self, top_n: int = 5) -> List[tuple]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT destination, COUNT(*) AS trip_count, MIN(id) AS first_seen
                FROM trip_history
                WHERE user_id = ?
                  AND destination IS NOT NULL
                  AND destination != ''
                GROUP BY destination
                ORDER BY trip_count DESC, first_seen ASC
                LIMIT ?
                """,
                (self.user_id, top_n),
            ).fetchall()
        return [(row["destination"], row["trip_count"]) for row in rows]

    def increment_query_count(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_statistics (user_id, total_queries)
                VALUES (?, 1)
                ON CONFLICT(user_id) DO UPDATE SET
                    total_queries = total_queries + 1
                """,
                (self.user_id,),
            )

    def get_statistics(self) -> Dict[str, Any]:
        with self._connect() as connection:
            total_messages = connection.execute(
                "SELECT COUNT(*) FROM chat_messages WHERE user_id = ?",
                (self.user_id,),
            ).fetchone()[0]
            total_trips = connection.execute(
                "SELECT COUNT(*) FROM trip_history WHERE user_id = ?",
                (self.user_id,),
            ).fetchone()[0]
            row = connection.execute(
                "SELECT total_queries FROM user_statistics WHERE user_id = ?",
                (self.user_id,),
            ).fetchone()

        return {
            "total_trips": total_trips,
            "total_messages": total_messages,
            "frequent_destinations": dict(self.get_frequent_destinations(top_n=100)),
            "total_queries": row["total_queries"] if row else 0,
        }

    def clear_history(self) -> None:
        """清除当前用户的聊天与行程；偏好继续保留。"""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM chat_messages WHERE user_id = ?",
                (self.user_id,),
            )
            connection.execute(
                "DELETE FROM trip_history WHERE user_id = ?",
                (self.user_id,),
            )

    def delete_all(self) -> None:
        """清除当前用户的所有数据，不删除其他用户共享的数据库文件。"""
        with self._connect() as connection:
            for table in (
                "preferences",
                "chat_messages",
                "trip_history",
                "user_statistics",
            ):
                connection.execute(
                    f"DELETE FROM {table} WHERE user_id = ?",
                    (self.user_id,),
                )
