#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SessionStore 抽象和默认内存实现的离线测试。"""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import Mock

from context.long_term_memory import LongTermMemory
from context.memory_manager import MemoryManager
from context.memory_repository import LongTermMemoryRepository
from context.session_store import InMemorySessionStore, SessionStore


class TestInMemorySessionStore(unittest.TestCase):
    def test_implements_session_store_contract(self):
        self.assertIsInstance(InMemorySessionStore(), SessionStore)

    def test_messages_are_bounded_and_sessions_are_isolated(self):
        store = InMemorySessionStore()
        for index in range(4):
            store.add_message(
                "session-a",
                {"role": "user", "content": f"message-{index}"},
                max_messages=2,
            )
        store.add_message(
            "session-b",
            {"role": "user", "content": "other-session"},
            max_messages=2,
        )

        self.assertEqual(
            [item["content"] for item in store.get_recent_messages("session-a")],
            ["message-2", "message-3"],
        )
        self.assertEqual(
            store.get_recent_messages("session-b")[0]["content"],
            "other-session",
        )

    def test_pending_trip_is_copied_and_clear_session_removes_all_state(self):
        store = InMemorySessionStore()
        source = {"destination": "北京"}
        store.save_pending_trip("session-a", source)
        source["destination"] = "杭州"

        returned = store.get_pending_trip("session-a")
        returned["destination"] = "上海"
        self.assertEqual(
            store.get_pending_trip("session-a"),
            {"destination": "北京"},
        )

        store.add_message(
            "session-a",
            {"role": "user", "content": "test"},
            max_messages=20,
        )
        store.clear_session("session-a")
        self.assertEqual(store.get_recent_messages("session-a"), [])
        self.assertEqual(store.get_pending_trip("session-a"), {})


class TestMemoryManagerStorageInjection(unittest.TestCase):
    def test_default_json_storage_implements_repository_contract(self):
        with tempfile.TemporaryDirectory() as storage_path:
            manager = MemoryManager(
                user_id="user",
                session_id="session",
                storage_path=storage_path,
            )

            self.assertIsInstance(manager.repository, LongTermMemoryRepository)
            self.assertIsInstance(manager.repository, LongTermMemory)

    def test_injected_repository_receives_long_term_message_write(self):
        repository = Mock(spec=LongTermMemoryRepository)
        with tempfile.TemporaryDirectory() as storage_path:
            manager = MemoryManager(
                user_id="user",
                session_id="session",
                storage_path=storage_path,
                repository=repository,
            )

            manager.add_message("user", "测试Repository注入")

        repository.add_chat_message.assert_called_once_with(
            "user",
            "测试Repository注入",
            "session",
        )

    def test_shared_store_keeps_state_separate_by_session_id(self):
        store = InMemorySessionStore()
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
            )
            manager.short_term.add_message("user", "当前问题")
            manager.save_pending_trip({"destination": "北京"})

            manager.end_session()

            self.assertEqual(manager.short_term.get_recent_context(), [])
            self.assertEqual(manager.get_pending_trip(), {})

    def test_clear_session_state_preserves_long_term_repository(self):
        with tempfile.TemporaryDirectory() as storage_path:
            manager = MemoryManager(
                user_id="user",
                session_id="session",
                storage_path=storage_path,
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
