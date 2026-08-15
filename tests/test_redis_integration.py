"""真实 Redis 的可选集成测试。

默认跳过；启动 Redis 后执行：
TEST_REDIS_URL=redis://localhost:6379/0 \
python -m unittest tests.test_redis_integration -v
"""

from __future__ import annotations

import os
import unittest
from uuid import uuid4

from context.redis_session_store import RedisSessionStore


TEST_REDIS_URL = os.getenv("TEST_REDIS_URL")


@unittest.skipUnless(TEST_REDIS_URL, "未配置 TEST_REDIS_URL")
class TestRealRedisSessionStore(unittest.TestCase):
    def test_two_clients_share_and_clear_session_state(self):
        prefix = f"travel-agent-integration-{uuid4().hex}"
        first = RedisSessionStore.from_url(
            TEST_REDIS_URL,
            user_id="integration-user",
            ttl_seconds=30,
            key_prefix=prefix,
        )
        second = RedisSessionStore.from_url(
            TEST_REDIS_URL,
            user_id="integration-user",
            ttl_seconds=30,
            key_prefix=prefix,
        )
        session_id = "integration-session"

        try:
            first.add_message(
                session_id,
                {"role": "user", "content": "测试消息"},
                max_messages=20,
            )
            first.save_pending_trip(
                session_id,
                {"destination": "北京"},
            )

            self.assertEqual(
                second.get_recent_messages(session_id)[0]["content"],
                "测试消息",
            )
            self.assertEqual(
                second.get_pending_trip(session_id),
                {"destination": "北京"},
            )

            second.clear_session(session_id)
            self.assertEqual(first.get_recent_messages(session_id), [])
            self.assertEqual(first.get_pending_trip(session_id), {})
        finally:
            first.clear_session(session_id)
            first.close()
            second.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
