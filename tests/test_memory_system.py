#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两层记忆系统的离线自动化测试。

测试不依赖 AgentScope、真实 LLM 或网络，并使用临时目录隔离数据：

1. 短期记忆滑动窗口；
2. MemoryManager 对短期和长期聊天历史的双写；
3. 偏好新增、覆盖和列表去重；
4. 行程历史、目的地统计与 clear_history() 语义；
5. 相同用户在不同 session_id 下的长期记忆共享；
6. 使用 FakeModel 验证旧会话 LLM 摘要输入与失败降级。

运行：python3 tests/test_memory_system.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from context.memory_manager import MemoryManager
from context.short_term_memory import ShortTermMemory
from tests.session_store_test_utils import (
    create_fake_redis_client,
    create_test_session_store,
)


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeModel:
    """记录 Prompt 并返回固定摘要的异步模型替身。"""

    def __init__(self, summary: str = "用户曾从上海前往杭州出差。", error=None):
        self.summary = summary
        self.error = error
        self.calls = []

    async def __call__(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return FakeResponse(self.summary)


class TemporaryMemoryTestCase(unittest.TestCase):
    """为每个同步测试提供独立临时存储目录。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = self.temp_dir.name
        self.redis_client = create_fake_redis_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_manager(self, session_id="session_1", user_id="user_1"):
        return MemoryManager(
            user_id=user_id,
            session_id=session_id,
            storage_path=self.storage_path,
            session_store=create_test_session_store(
                user_id,
                redis_client=self.redis_client,
            ),
        )


class TestShortTermMemory(unittest.TestCase):
    def test_sliding_window_keeps_only_latest_turns(self):
        memory = ShortTermMemory(
            max_turns=2,
            session_store=create_test_session_store(),
        )

        for index in range(6):
            role = "user" if index % 2 == 0 else "assistant"
            memory.add_message(role, f"message-{index + 1}")

        self.assertEqual(len(memory.messages), 4)
        self.assertEqual(
            [message["content"] for message in memory.messages],
            ["message-3", "message-4", "message-5", "message-6"],
        )
        self.assertEqual(
            [message["content"] for message in memory.get_recent_context(1)],
            ["message-5", "message-6"],
        )
        self.assertIn("用户: message-5", memory.get_context_string(1))
        self.assertIn("助手: message-6", memory.get_context_string(1))


class TestMemoryManager(TemporaryMemoryTestCase):
    def test_add_message_writes_both_layers_and_end_session_keeps_long_term(self):
        manager = self.create_manager()

        manager.add_message("user", "我想去杭州")
        manager.add_message("assistant", "准备什么时候出发？")

        self.assertEqual(len(manager.short_term.messages), 2)
        self.assertEqual(len(manager.long_term.get_chat_history()), 2)
        self.assertEqual(
            {message["session_id"] for message in manager.long_term.get_chat_history()},
            {"session_1"},
        )
        self.assertTrue(Path(manager.long_term.db_path).exists())

        manager.end_session()

        self.assertEqual(manager.short_term.get_recent_context(), [])
        self.assertEqual(len(manager.long_term.get_chat_history()), 2)

    def test_preferences_support_replace_append_and_deduplication(self):
        manager = self.create_manager()
        memory = manager.long_term

        memory.save_preference("home_location", "北京")
        memory.save_preference("home_location", "上海")
        memory.add_hotel_brand("如家")
        memory.add_hotel_brand("汉庭")
        memory.add_hotel_brand("如家")

        self.assertEqual(memory.get_preference("home_location"), "上海")
        self.assertEqual(memory.get_preference("hotel_brands"), ["如家", "汉庭"])
        self.assertEqual(
            memory.get_preference(),
            {"home_location": "上海", "hotel_brands": ["如家", "汉庭"]},
        )

    def test_collection_preference_keeps_list_shape_and_skips_duplicate(self):
        memory = self.create_manager().long_term

        first_changed = memory.save_preference("hotel_brands", "汉庭")
        second_changed = memory.save_preference("hotel_brands", "汉庭")
        third_changed = memory.save_preference(
            "hotel_brands",
            ["汉庭", "汉庭", "如家"],
        )

        self.assertTrue(first_changed)
        self.assertFalse(second_changed)
        self.assertTrue(third_changed)
        self.assertEqual(
            memory.get_preference("hotel_brands"),
            ["汉庭", "如家"],
        )

        memory.save_preference("home_location", "苏州")
        self.assertEqual(memory.get_preference("home_location"), "苏州")

    def test_default_departure_alias_updates_home_location(self):
        memory = self.create_manager().long_term
        memory.save_preference("home_location", "苏州")

        changed = memory.save_preference("default_departure", "上海")

        self.assertTrue(changed)
        self.assertEqual(memory.get_preference("home_location"), "上海")
        self.assertNotIn("default_departure", memory.get_preference())

    def test_trip_statistics_and_clear_history_preserve_preferences(self):
        manager = self.create_manager()
        memory = manager.long_term
        memory.save_preference("seat_preference", "靠窗")
        manager.add_message("user", "记录一条聊天")

        for destination in ["杭州", "北京", "杭州"]:
            memory.save_trip_history(
                {
                    "origin": "上海",
                    "destination": destination,
                    "purpose": "出差",
                }
            )

        self.assertEqual(memory.get_statistics()["total_trips"], 3)
        self.assertEqual(memory.get_frequent_destinations(2), [("杭州", 2), ("北京", 1)])

        memory.clear_history()

        self.assertEqual(memory.get_chat_history(), [])
        self.assertEqual(memory.get_trip_history(limit=None), [])
        self.assertEqual(memory.get_statistics()["total_trips"], 0)
        self.assertEqual(memory.get_preference("seat_preference"), "靠窗")

    def test_duplicate_trip_history_is_idempotent(self):
        memory = self.create_manager().long_term
        initial_trip = {
            "origin": "苏州",
            "destination": "北京",
            "start_date": "2026-08-10",
            "purpose": "出差",
        }
        completed_trip = {
            **initial_trip,
            "end_date": "2026-08-12",
            "duration_days": 3,
        }

        first_created = memory.save_trip_history(initial_trip)
        second_created = memory.save_trip_history(completed_trip)
        third_created = memory.save_trip_history(dict(completed_trip))

        self.assertTrue(first_created)
        self.assertFalse(second_created)
        self.assertFalse(third_created)
        self.assertEqual(len(memory.get_trip_history(limit=None)), 1)
        saved_trip = memory.get_trip_history(limit=None)[0]
        self.assertEqual(saved_trip["end_date"], "2026-08-12")
        self.assertEqual(saved_trip["duration_days"], 3)
        self.assertEqual(memory.get_statistics()["total_trips"], 1)
        self.assertEqual(memory.get_frequent_destinations(), [("北京", 1)])

    def test_same_user_can_read_long_term_memory_in_a_new_session(self):
        first_session = self.create_manager(session_id="session_a", user_id="same_user")
        first_session.add_message("user", "旧会话消息")
        first_session.long_term.save_preference("home_location", "苏州")
        first_session.long_term.save_trip_history(
            {"origin": "苏州", "destination": "杭州", "purpose": "出差"}
        )

        second_session = self.create_manager(session_id="session_b", user_id="same_user")

        self.assertEqual(second_session.short_term.get_recent_context(), [])
        self.assertEqual(second_session.long_term.get_preference("home_location"), "苏州")
        self.assertEqual(
            second_session.long_term.get_chat_history()[0]["content"],
            "旧会话消息",
        )
        self.assertEqual(
            second_session.long_term.get_trip_history()[0]["destination"],
            "杭州",
        )

    def test_full_context_and_agent_context_combine_both_layers(self):
        manager = self.create_manager()
        manager.long_term.save_preference("seat_preference", "靠窗")
        manager.add_message("user", "当前会话问题")

        full_context = manager.get_full_context()
        agent_context = manager.get_context_for_agent("旧会话摘要")

        self.assertEqual(
            full_context["long_term"]["preferences"]["seat_preference"],
            "靠窗",
        )
        self.assertEqual(len(full_context["short_term"]["recent_dialogue"]), 1)
        self.assertIn("旧会话摘要", agent_context)
        self.assertIn("seat_preference: 靠窗", agent_context)
        self.assertIn("用户: 当前会话问题", agent_context)


class TestLongTermSummary(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = self.temp_dir.name
        self.redis_client = create_fake_redis_client()

    async def asyncTearDown(self):
        self.temp_dir.cleanup()

    async def test_summary_uses_other_sessions_and_trip_history(self):
        old_session = MemoryManager(
            user_id="summary_user",
            session_id="old_session",
            storage_path=self.storage_path,
            session_store=create_test_session_store(
                "summary_user",
                redis_client=self.redis_client,
            ),
        )
        old_session.add_message("user", "旧会话里我问过杭州出差")
        old_session.long_term.save_trip_history(
            {
                "origin": "上海",
                "destination": "杭州",
                "start_date": "2026-07-01",
                "end_date": "2026-07-03",
                "purpose": "出差",
            }
        )

        model = FakeModel()
        current_session = MemoryManager(
            user_id="summary_user",
            session_id="current_session",
            storage_path=self.storage_path,
            llm_model=model,
            session_store=create_test_session_store(
                "summary_user",
                redis_client=self.redis_client,
            ),
        )
        current_session.add_message("user", "当前会话消息不应进入旧会话摘要")

        summary = await current_session.get_long_term_summary_async()

        self.assertEqual(summary, "用户曾从上海前往杭州出差。")
        self.assertEqual(len(model.calls), 1)
        prompt = model.calls[0][0]["content"]
        self.assertIn("旧会话里我问过杭州出差", prompt)
        self.assertNotIn("当前会话消息不应进入旧会话摘要", prompt)
        self.assertIn("上海 → 杭州", prompt)

    async def test_summary_failure_returns_empty_string(self):
        manager = MemoryManager(
            user_id="summary_error_user",
            session_id="current_session",
            storage_path=self.storage_path,
            llm_model=FakeModel(error=RuntimeError("model unavailable")),
            session_store=create_test_session_store(
                "summary_error_user",
                redis_client=self.redis_client,
            ),
        )
        manager.long_term.save_trip_history(
            {"origin": "苏州", "destination": "杭州", "purpose": "出差"}
        )

        with self.assertLogs("context.memory_manager", level="ERROR"):
            summary = await manager.get_long_term_summary_async()

        self.assertEqual(summary, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
