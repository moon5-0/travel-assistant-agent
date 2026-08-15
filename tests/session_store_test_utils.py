"""Redis SessionStore 测试辅助工具。"""

from __future__ import annotations

import fakeredis

from context.redis_session_store import RedisSessionStore


def create_fake_redis_client():
    return fakeredis.FakeRedis(decode_responses=True)


def create_test_session_store(
    user_id: str = "test-user",
    *,
    redis_client=None,
    ttl_seconds: int = 3600,
) -> RedisSessionStore:
    if redis_client is None:
        redis_client = create_fake_redis_client()
    return RedisSessionStore(
        redis_client,
        user_id=user_id,
        ttl_seconds=ttl_seconds,
        key_prefix="test-travel-agent",
    )
