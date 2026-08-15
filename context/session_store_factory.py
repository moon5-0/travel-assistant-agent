"""短期会话存储的创建入口。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from config import REDIS_SESSION_CONFIG

from .redis_session_store import RedisSessionStore
from .session_store import SessionStore


def create_session_store(
    user_id: str,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> SessionStore:
    """根据项目配置创建正式 Redis SessionStore。"""
    settings = dict(REDIS_SESSION_CONFIG)
    if config:
        settings.update(config)
    return RedisSessionStore.from_url(
        settings["url"],
        user_id=user_id,
        ttl_seconds=settings["ttl_seconds"],
        key_prefix=settings["key_prefix"],
        socket_connect_timeout_sec=settings["socket_connect_timeout_sec"],
        socket_timeout_sec=settings["socket_timeout_sec"],
    )
