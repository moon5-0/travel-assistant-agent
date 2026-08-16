#!/usr/bin/env python3
"""Redis SessionStore 和 MemoryManager 注入边界的离线测试。"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from unittest.mock import Mock, patch

from redis.exceptions import ConnectionError as RedisConnectionError

from context.memory_manager import MemoryManager
from context.memory_repository import LongTermMemoryRepository
from context.redis_session_store import (
    RedisSessionStore,
    SessionStoreUnavailableError,
)
from context.session_store import SessionStore
from context.sqlite_memory_repository import SQLiteMemoryRepository
from tests.session_store_test_utils import (
    create_fake_redis_client,
    create_test_session_store,
)


class TestRedisSessionStore(unittest.TestCase):
    def setUp(self):
        self.redis = create_fake_redis_client()
        self.store = create_test_session_store(
            "user-a",
            redis_client=self.redis,
            ttl_seconds=60,
        )

    def test_implements_session_store_contract(self):
        self.assertIsInstance(self.store, SessionStore)

    def test_messages_are_bounded_and_sessions_are_isolated(self):
        for index in range(4):
            self.store.add_message(
                "session-a",
                {"role": "user", "content": f"message-{index}"},
                max_messages=2,
            )
        self.store.add_message(
            "session-b",
            {"role": "user", "content": "other-session"},
            max_messages=2,
        )

        self.assertEqual(
            [item["content"] for item in self.store.get_recent_messages("session-a")],
            ["message-2", "message-3"],
        )
        self.assertEqual(
            self.store.get_recent_messages("session-b")[0]["content"],
            "other-session",
        )

    def test_users_are_isolated_even_when_session_id_matches(self):
        other_user_store = create_test_session_store(
            "user-b",
            redis_client=self.redis,
            ttl_seconds=60,
        )
        self.store.save_pending_trip("same-session", {"destination": "北京"})
        other_user_store.save_pending_trip(
            "same-session",
            {"destination": "上海"},
        )

        self.assertEqual(
            self.store.get_pending_trip("same-session")["destination"],
            "北京",
        )
        self.assertEqual(
            other_user_store.get_pending_trip("same-session")["destination"],
            "上海",
        )

    def test_pending_trip_is_serialized_and_clear_session_removes_all_state(self):
        source = {"destination": "北京"}
        self.store.save_pending_trip("session-a", source)
        source["destination"] = "杭州"

        returned = self.store.get_pending_trip("session-a")
        returned["destination"] = "上海"
        self.assertEqual(
            self.store.get_pending_trip("session-a"),
            {"destination": "北京"},
        )

        self.store.add_message(
            "session-a",
            {"role": "user", "content": "test"},
            max_messages=20,
        )
        self.store.clear_session("session-a")
        self.assertEqual(self.store.get_recent_messages("session-a"), [])
        self.assertEqual(self.store.get_pending_trip("session-a"), {})

    def test_activity_sets_and_refreshes_ttl(self):
        self.store.save_pending_trip("session-a", {"destination": "北京"})
        self.store.add_message(
            "session-a",
            {"role": "user", "content": "继续"},
            max_messages=20,
        )

        messages_ttl = self.redis.ttl(self.store._messages_key("session-a"))
        pending_ttl = self.redis.ttl(self.store._pending_trip_key("session-a"))
        self.assertGreater(messages_ttl, 0)
        self.assertLessEqual(messages_ttl, 60)
        self.assertGreater(pending_ttl, 0)
        self.assertLessEqual(pending_ttl, 60)

    def test_two_store_instances_share_the_same_redis_state(self):
        another_instance = create_test_session_store(
            "user-a",
            redis_client=self.redis,
            ttl_seconds=60,
        )
        self.store.save_pending_trip("session-a", {"destination": "北京"})

        self.assertEqual(
            another_instance.get_pending_trip("session-a"),
            {"destination": "北京"},
        )

    def test_malformed_pending_trip_is_cleared(self):
        key = self.store._pending_trip_key("session-a")
        self.redis.set(key, "{invalid-json")

        with self.assertLogs("context.redis_session_store", level="WARNING"):
            self.assertEqual(self.store.get_pending_trip("session-a"), {})
        self.assertFalse(self.redis.exists(key))

    def test_connection_failure_is_explicit_instead_of_falling_back(self):
        unavailable_client = Mock()
        unavailable_client.ping.side_effect = RedisConnectionError("offline")

        with patch(
            "context.redis_session_store.Redis.from_url",
            return_value=unavailable_client,
        ):
            with self.assertRaises(SessionStoreUnavailableError):
                RedisSessionStore.from_url(
                    "redis://localhost:6379/0",
                    user_id="user-a",
                )


class TestMemoryManagerStorageInjection(unittest.TestCase):
    def create_store(self, user_id="user"):
        return create_test_session_store(user_id)

    def test_default_sqlite_storage_implements_repository_contract(self):
        with tempfile.TemporaryDirectory() as storage_path:
            manager = MemoryManager(
                user_id="user",
                session_id="session",
                storage_path=storage_path,
                session_store=self.create_store(),
            )

            self.assertIsInstance(manager.repository, LongTermMemoryRepository)
            self.assertIsInstance(manager.repository, SQLiteMemoryRepository)

    def test_memory_manager_creates_redis_store_by_default(self):
        fake_redis = create_fake_redis_client()
        with patch(
            "context.redis_session_store.Redis.from_url",
            return_value=fake_redis,
        ):
            with tempfile.TemporaryDirectory() as storage_path:
                manager = MemoryManager(
                    user_id="user",
                    session_id="session",
                    storage_path=storage_path,
                )

        self.assertIsInstance(manager.session_store, RedisSessionStore)

    def test_injected_repository_receives_long_term_message_write(self):
        repository = Mock(spec=LongTermMemoryRepository)
        with tempfile.TemporaryDirectory() as storage_path:
            manager = MemoryManager(
                user_id="user",
                session_id="session",
                storage_path=storage_path,
                repository=repository,
                session_store=self.create_store(),
            )

            manager.add_message("user", "测试Repository注入")

        repository.add_chat_message.assert_called_once_with(
            "user",
            "测试Repository注入",
            "session",
        )

    def test_shared_store_keeps_state_separate_by_session_id(self):
        store = self.create_store("same-user")
        with tempfile.TemporaryDirectory() as storage_path:
            first = MemoryManager(
                user_id="same-user",
                session_id="session-a",
                storage_path=storage_path,
                session_store=store,
            )
            second = MemoryManager(
                user_id="same-user",
                session_id="session-b",
                storage_path=storage_path,
                session_store=store,
            )
            first.save_pending_trip({"destination": "北京"})
            first.short_term.add_message("user", "第一会话")

            self.assertEqual(first.get_pending_trip()["destination"], "北京")
            self.assertEqual(
                first.get_full_context()["short_term"]["pending_trip"],
                {"destination": "北京"},
            )
            self.assertEqual(second.get_pending_trip(), {})
            self.assertEqual(second.short_term.get_recent_context(), [])

    def test_end_session_clears_messages_and_pending_trip_together(self):
        with tempfile.TemporaryDirectory() as storage_path:
            manager = MemoryManager(
                user_id="user",
                session_id="session",
                storage_path=storage_path,
                session_store=self.create_store(),
            )
            manager.short_term.add_message("user", "当前问题")
            manager.save_pending_trip({"destination": "北京"})

            asyncio.run(manager.end_session())

            self.assertEqual(manager.short_term.get_recent_context(), [])
            self.assertEqual(manager.get_pending_trip(), {})

    def test_clear_session_state_preserves_long_term_repository(self):
        with tempfile.TemporaryDirectory() as storage_path:
            manager = MemoryManager(
                user_id="user",
                session_id="session",
                storage_path=storage_path,
                session_store=self.create_store(),
            )
            manager.add_message("user", "需要长期保留的原始消息")
            manager.save_pending_trip({"destination": "北京"})

            manager.clear_session_state()

            self.assertEqual(manager.short_term.get_recent_context(), [])
            self.assertEqual(manager.get_pending_trip(), {})
            self.assertEqual(
                manager.long_term.get_chat_history()[0]["content"],
                "需要长期保留的原始消息",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
