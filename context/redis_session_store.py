"""基于 Redis 的短期会话存储。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from redis import Redis
from redis.exceptions import RedisError

from .session_store import SessionStore


logger = logging.getLogger(__name__)


class SessionStoreUnavailableError(ConnectionError):
    """配置为 Redis，但 Redis 服务无法连接。"""


class RedisSessionStore(SessionStore):
    """使用 Redis List 和 String 保存一个用户的临时会话状态。"""

    def __init__(
        self,
        redis_client: Any,
        *,
        user_id: str,
        ttl_seconds: int = 3600,
        key_prefix: str = "travel-agent",
    ) -> None:
        if not user_id:
            raise ValueError("user_id 不能为空")
        if ttl_seconds <= 0:
            raise ValueError("Redis 会话 TTL 必须大于0")

        normalized_prefix = str(key_prefix).strip().strip(":")
        if not normalized_prefix:
            raise ValueError("Redis key_prefix 不能为空")

        self.redis = redis_client
        self.user_id = str(user_id)
        self.ttl_seconds = int(ttl_seconds)
        self.key_prefix = normalized_prefix

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        user_id: str,
        ttl_seconds: int = 3600,
        key_prefix: str = "travel-agent",
        socket_connect_timeout_sec: float = 2.0,
        socket_timeout_sec: float = 2.0,
        validate_connection: bool = True,
    ) -> "RedisSessionStore":
        """根据 URL 创建带连接池的客户端，并可在启动时验证连接。"""
        client = Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=socket_connect_timeout_sec,
            socket_timeout=socket_timeout_sec,
        )
        store = cls(
            client,
            user_id=user_id,
            ttl_seconds=ttl_seconds,
            key_prefix=key_prefix,
        )
        if validate_connection:
            store.ensure_available()
        return store

    def ensure_available(self) -> None:
        """启动阶段快速验证 Redis，不允许静默退回进程内存。"""
        try:
            self.redis.ping()
        except (RedisError, OSError) as exc:
            raise SessionStoreUnavailableError(
                "无法连接 Redis 短期记忆服务，请检查 REDIS_URL 和 Redis 进程"
            ) from exc

    @staticmethod
    def _key_component(value: str) -> str:
        """转义冒号等分隔符，避免不同用户或会话拼出相同 Key。"""
        return quote(str(value), safe="")

    def _session_base_key(self, session_id: str) -> str:
        if not session_id:
            raise ValueError("session_id 不能为空")
        return (
            f"{self.key_prefix}:user:{self._key_component(self.user_id)}:"
            f"session:{self._key_component(session_id)}"
        )

    def _messages_key(self, session_id: str) -> str:
        return f"{self._session_base_key(session_id)}:messages"

    def _pending_trip_key(self, session_id: str) -> str:
        return f"{self._session_base_key(session_id)}:pending_trip"

    @staticmethod
    def _dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _load(value: Any) -> Any:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value)

    def _refresh_session_ttl(self, pipeline: Any, session_id: str) -> None:
        """一次会话活动同时刷新消息和待补全行程的滑动过期时间。"""
        pipeline.expire(self._messages_key(session_id), self.ttl_seconds)
        pipeline.expire(self._pending_trip_key(session_id), self.ttl_seconds)

    def add_message(
        self,
        session_id: str,
        message: Dict[str, Any],
        *,
        max_messages: int,
    ) -> None:
        if max_messages <= 0:
            raise ValueError("max_messages 必须大于0")

        key = self._messages_key(session_id)
        with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.rpush(key, self._dump(dict(message)))
            pipeline.ltrim(key, -max_messages, -1)
            self._refresh_session_ttl(pipeline, session_id)
            pipeline.execute()

    def get_recent_messages(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if limit is not None and limit < 0:
            raise ValueError("limit 不能小于0")

        start = -limit if limit else 0
        with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.lrange(self._messages_key(session_id), start, -1)
            self._refresh_session_ttl(pipeline, session_id)
            results = pipeline.execute()

        messages = []
        for raw_message in results[0] or []:
            try:
                message = self._load(raw_message)
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                logger.warning("Skipped malformed Redis session message")
                continue
            if isinstance(message, dict):
                messages.append(message)
        return messages

    def clear_messages(self, session_id: str) -> None:
        self.redis.delete(self._messages_key(session_id))

    def get_pending_trip(self, session_id: str) -> Dict[str, Any]:
        key = self._pending_trip_key(session_id)
        with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.get(key)
            self._refresh_session_ttl(pipeline, session_id)
            results = pipeline.execute()

        raw_trip = results[0]
        if raw_trip is None:
            return {}
        try:
            trip = self._load(raw_trip)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            logger.warning("Cleared malformed Redis pending trip")
            self.redis.delete(key)
            return {}
        return trip if isinstance(trip, dict) else {}

    def save_pending_trip(
        self,
        session_id: str,
        trip_data: Dict[str, Any],
    ) -> None:
        normalized = dict(trip_data or {})
        if not normalized:
            self.clear_pending_trip(session_id)
            return

        with self.redis.pipeline(transaction=True) as pipeline:
            pipeline.set(
                self._pending_trip_key(session_id),
                self._dump(normalized),
                ex=self.ttl_seconds,
            )
            pipeline.expire(self._messages_key(session_id), self.ttl_seconds)
            pipeline.execute()

    def clear_pending_trip(self, session_id: str) -> None:
        self.redis.delete(self._pending_trip_key(session_id))

    def clear_session(self, session_id: str) -> None:
        """一条 DEL 命令清除当前会话全部临时状态。"""
        self.redis.delete(
            self._messages_key(session_id),
            self._pending_trip_key(session_id),
        )

    def close(self) -> None:
        """关闭客户端连接池；测试和显式生命周期管理时使用。"""
        close = getattr(self.redis, "close", None)
        if callable(close):
            close()
