#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OrchestrationAgent 的离线自动化测试。

测试只关注协调器本身，不调用真实 LLM、网络搜索或 RAG：

1. 同优先级 Agent 是否并行，不同优先级是否顺序执行；
2. 后序批次能否收到前序批次结果；
3. 单个 Agent 失败时，成功结果是否保留；
4. 未注册 Agent 和空调度计划能否友好处理；
5. 偏好与行程结果能否写入长期记忆。

运行：python3 tests/test_orchestration.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import types
import unittest
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# 让协调器单元测试在未安装 AgentScope 的环境中也能离线运行。
# 如果项目环境已经安装 AgentScope，则直接使用真实的基类和消息类型。
try:
    from agentscope.agent import AgentBase
    from agentscope.message import Msg
except ModuleNotFoundError:
    agentscope_module = types.ModuleType("agentscope")
    agent_module = types.ModuleType("agentscope.agent")
    message_module = types.ModuleType("agentscope.message")

    class AgentBase:  # type: ignore[no-redef]
        """测试所需的最小 AgentScope AgentBase 替身。"""

    class Msg:  # type: ignore[no-redef]
        """测试所需的最小 AgentScope Msg 替身。"""

        def __init__(
            self,
            name: str,
            content: Any,
            role: str,
            metadata: Optional[Dict[str, Any]] = None,
        ):
            self.name = name
            self.content = content
            self.role = role
            self.metadata = metadata or {}

    agent_module.AgentBase = AgentBase
    message_module.Msg = Msg
    agentscope_module.agent = agent_module
    agentscope_module.message = message_module
    sys.modules["agentscope"] = agentscope_module
    sys.modules["agentscope.agent"] = agent_module
    sys.modules["agentscope.message"] = message_module


from agents.orchestration_agent import OrchestrationAgent


class FakeAgent(AgentBase):
    """可控制耗时、返回值和异常的离线 Agent。"""

    def __init__(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        delay: float = 0.0,
        exception: Optional[Exception] = None,
    ):
        super().__init__()
        self.name = name
        self.payload = payload or {"value": name}
        self.delay = delay
        self.exception = exception
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.received_input: Optional[Dict[str, Any]] = None

    async def reply(self, x: Msg) -> Msg:
        self.started_at = time.perf_counter()
        self.received_input = json.loads(x.content)
        await asyncio.sleep(self.delay)

        if self.exception:
            raise self.exception

        self.finished_at = time.perf_counter()
        return Msg(
            name=self.name,
            content=json.dumps(self.payload, ensure_ascii=False),
            role="assistant",
        )


class FakeShortTermMemory:
    def __init__(self):
        self.recent = [{"role": "user", "content": "最近一条对话"}]

    def get_recent_context(self, _turns: int):
        return list(self.recent)


class FakeLongTermMemory:
    def __init__(self):
        self.preferences = {"hotel_brands": ["如家"]}
        self.saved_trips = []

    def get_preference(self):
        return self.preferences

    def save_preference(self, pref_type: str, value: Any):
        self.preferences[pref_type] = value

    def save_trip_history(self, trip: Dict[str, Any]):
        self.saved_trips.append(trip)


class FakeMemoryManager:
    def __init__(self):
        self.short_term = FakeShortTermMemory()
        self.long_term = FakeLongTermMemory()


def intention_message(
    schedule,
    original_user_input: str = "2026年7月24日从苏州前往杭州",
):
    """构造协调器实际接收的 IntentionAgent 输出。"""
    data = {
        "reasoning": "测试调度",
        "intents": [{"type": "itinerary_planning", "confidence": 0.95}],
        "key_entities": {"origin": "苏州", "destination": "杭州"},
        "rewritten_query": "从苏州前往杭州",
        "agent_schedule": schedule,
    }
    return Msg(
        name="IntentionAgent",
        content=json.dumps(data, ensure_ascii=False),
        role="assistant",
        metadata={"original_user_input": original_user_input},
    )


class TestOrchestrationAgent(unittest.IsolatedAsyncioTestCase):
    async def test_unmentioned_start_date_is_rejected_before_planning(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": "北京",
                # 模拟模型在用户未说日期时擅自选择“明天”。
                "start_date": "2026-07-28",
                "end_date": "2026-07-30",
                "duration_days": 3,
                "missing_info": [],
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(
            agent_registry={
                "event_collection": event_agent,
                "itinerary_planning": plan_agent,
            }
        )
        schedule = [
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "itinerary_planning", "priority": 2},
        ]

        response = await orchestrator.reply(
            intention_message(
                schedule,
                original_user_input="从苏州去北京出差3天，请帮我规划行程",
            )
        )
        result = json.loads(response.content)

        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["missing_fields"], ["start_date"])
        self.assertIsNone(plan_agent.started_at)
        self.assertNotIn("start_date", orchestrator.get_pending_trip())

    async def test_relative_date_from_user_can_be_normalized_and_planned(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": "北京",
                "start_date": "2026-07-29",
                "end_date": "2026-07-31",
                "duration_days": 3,
                "missing_info": [],
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(
            agent_registry={
                "event_collection": event_agent,
                "itinerary_planning": plan_agent,
            }
        )
        schedule = [
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "itinerary_planning", "priority": 2},
        ]

        response = await orchestrator.reply(
            intention_message(
                schedule,
                original_user_input="明天从苏州去北京出差3天",
            )
        )
        result = json.loads(response.content)

        self.assertEqual(result["status"], "completed")
        self.assertIsNotNone(plan_agent.started_at)

    async def test_same_priority_runs_in_parallel_and_next_priority_waits(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": "杭州",
                "start_date": "2026-07-24",
                "duration_days": 2,
                "missing_info": [],
            },
            delay=0.08,
        )
        info_agent = FakeAgent("information_query", delay=0.08)
        plan_agent = FakeAgent("itinerary_planning", delay=0.01)
        orchestrator = OrchestrationAgent(
            agent_registry={
                "event_collection": event_agent,
                "information_query": info_agent,
                "itinerary_planning": plan_agent,
            }
        )

        schedule = [
            {"agent_name": "itinerary_planning", "priority": 2},
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "information_query", "priority": 1},
        ]
        response = await orchestrator.reply(intention_message(schedule))
        result = json.loads(response.content)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["agents_executed"], 3)

        self.assertIsNotNone(event_agent.started_at)
        self.assertIsNotNone(event_agent.finished_at)
        self.assertIsNotNone(info_agent.started_at)
        self.assertIsNotNone(info_agent.finished_at)
        self.assertIsNotNone(plan_agent.started_at)

        # 两个 Priority 1 Agent 的执行时间区间发生重叠，证明不是串行。
        self.assertLess(
            max(event_agent.started_at, info_agent.started_at),
            min(event_agent.finished_at, info_agent.finished_at),
        )
        # Priority 2 必须等 Priority 1 全部结束后才能启动。
        self.assertGreaterEqual(
            plan_agent.started_at,
            max(event_agent.finished_at, info_agent.finished_at),
        )

        previous_results = plan_agent.received_input["previous_results"]
        self.assertEqual(len(previous_results), 2)
        self.assertEqual(
            {item["agent_name"] for item in previous_results},
            {"event_collection", "information_query"},
        )

    async def test_missing_trip_info_skips_itinerary_and_requests_clarification(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": None,
                "start_date": None,
                "duration_days": None,
                "missing_info": [
                    "destination",
                    "start_date",
                    "duration_days",
                ],
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(
            agent_registry={
                "event_collection": event_agent,
                "itinerary_planning": plan_agent,
            }
        )

        schedule = [
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "itinerary_planning", "priority": 2},
        ]

        response = await orchestrator.reply(intention_message(schedule))
        result = json.loads(response.content)

        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(
            result["missing_fields"],
            ["destination", "start_date", "duration_days"],
        )
        self.assertIn("请补充", result["message"])

        # 信息不完整时，行程规划Agent不应该被调用。
        self.assertIsNone(plan_agent.started_at)

    async def test_failed_event_collection_skips_itinerary(self):
        event_agent = FakeAgent(
            "event_collection",
            exception=RuntimeError("信息提取失败"),
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(
            agent_registry={
                "event_collection": event_agent,
                "itinerary_planning": plan_agent,
            }
        )

        schedule = [
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "itinerary_planning", "priority": 2},
        ]

        response = await orchestrator.reply(intention_message(schedule))
        result = json.loads(response.content)

        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("请补充", result["message"])
        self.assertIsNotNone(event_agent.started_at)
        self.assertIsNone(plan_agent.started_at)

    async def test_next_turn_merges_pending_trip_and_resumes_planning(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": None,
                "destination": "北京",
                "start_date": None,
                "end_date": None,
                "duration_days": None,
                "missing_info": [
                    "origin",
                    "start_date",
                    "duration_days",
                ],
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(
            agent_registry={
                "event_collection": event_agent,
                "itinerary_planning": plan_agent,
            }
        )

        schedule = [
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "itinerary_planning", "priority": 2},
        ]

        first_response = await orchestrator.reply(
            intention_message(schedule)
        )
        first_result = json.loads(first_response.content)

        self.assertEqual(
            first_result["status"],
            "needs_clarification",
        )
        self.assertIsNone(plan_agent.started_at)

        # 模拟用户第二轮补充的信息。
        event_agent.payload = {
            "origin": "苏州",
            "destination": None,
            "start_date": "2026-07-24",
            "end_date": "2026-07-26",
            "duration_days": 3,
            # EventCollectionAgent 只看本轮输入，因此仍会认为目的地缺失。
            # 调度器合并上一轮“北京”后必须重新计算，不能保留这个旧状态。
            "missing_info": ["destination"],
        }

        second_response = await orchestrator.reply(
            intention_message(schedule)
        )
        second_result = json.loads(second_response.content)

        self.assertEqual(second_result["status"], "completed")
        self.assertIsNotNone(plan_agent.started_at)

        previous_results = plan_agent.received_input[
            "previous_results"
        ]
        event_result = next(
            item
            for item in previous_results
            if item["agent_name"] == "event_collection"
        )
        merged_data = event_result["result"]["data"]

        self.assertEqual(merged_data["origin"], "苏州")
        self.assertEqual(merged_data["destination"], "北京")
        self.assertEqual(
            merged_data["start_date"],
            "2026-07-24",
        )
        self.assertEqual(merged_data["duration_days"], 3)
        self.assertEqual(merged_data["missing_info"], [])

    async def test_multiple_turns_keep_asking_until_trip_is_complete(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "destination": "北京",
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(
            agent_registry={
                "event_collection": event_agent,
                "itinerary_planning": plan_agent,
            }
        )

        schedule = [
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "itinerary_planning", "priority": 2},
        ]

        # 第一轮：只有目的地。
        first_response = await orchestrator.reply(
            intention_message(schedule)
        )
        first_result = json.loads(first_response.content)

        self.assertEqual(
            first_result["status"],
            "needs_clarification",
        )
        self.assertIsNone(plan_agent.started_at)

        # 第二轮：只补充出发地，信息仍然不完整。
        event_agent.payload = {
            "origin": "苏州",
        }

        second_response = await orchestrator.reply(
            intention_message(schedule)
        )
        second_result = json.loads(second_response.content)

        self.assertEqual(
            second_result["status"],
            "needs_clarification",
        )
        self.assertEqual(
            second_result["missing_fields"],
            ["start_date", "duration_days"],
        )
        self.assertIsNone(plan_agent.started_at)

        # 第三轮：补充日期和天数，信息完整。
        event_agent.payload = {
            "start_date": "2026-07-25",
            "duration_days": 3,
        }

        third_response = await orchestrator.reply(
            intention_message(schedule)
        )
        third_result = json.loads(third_response.content)

        self.assertEqual(third_result["status"], "completed")
        self.assertIsNotNone(plan_agent.started_at)

        # 行程规划完成后，临时状态应该被清空。
        self.assertEqual(orchestrator._pending_trip_data, {})

    async def test_clear_pending_trip_discards_previous_fields(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={"destination": "北京"},
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(
            agent_registry={
                "event_collection": event_agent,
                "itinerary_planning": plan_agent,
            }
        )

        schedule = [
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "itinerary_planning", "priority": 2},
        ]

        first_response = await orchestrator.reply(
            intention_message(schedule)
        )
        first_result = json.loads(first_response.content)

        self.assertEqual(
            first_result["status"],
            "needs_clarification",
        )
        self.assertEqual(
            orchestrator._pending_trip_data["destination"],
            "北京",
        )

        # 模拟用户执行clear命令。
        orchestrator.clear_pending_trip()

        event_agent.payload = {
            "origin": "苏州",
            "start_date": "2026-07-25",
            "duration_days": 3,
        }

        second_response = await orchestrator.reply(
            intention_message(schedule)
        )
        second_result = json.loads(second_response.content)

        # 北京已经被清除，所以仍然缺少目的地。
        self.assertEqual(
            second_result["status"],
            "needs_clarification",
        )
        self.assertEqual(
            second_result["missing_fields"],
            ["destination"],
        )
        self.assertIsNone(plan_agent.started_at)

    async def test_partial_failure_keeps_successful_results(self):
        good_agent = FakeAgent("event_collection", payload={"destination": "杭州"})
        bad_agent = FakeAgent(
            "information_query",
            exception=RuntimeError("weather service unavailable"),
        )
        orchestrator = OrchestrationAgent(
            agent_registry={
                "event_collection": good_agent,
                "information_query": bad_agent,
            }
        )

        schedule = [
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "information_query", "priority": 1},
        ]
        response = await orchestrator.reply(intention_message(schedule))
        result = json.loads(response.content)

        self.assertEqual(result["status"], "partial_failure")
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["agents_executed"], 2)
        status_by_agent = {
            item["agent_name"]: item["status"] for item in result["results"]
        }
        self.assertEqual(status_by_agent["event_collection"], "success")
        self.assertEqual(status_by_agent["information_query"], "error")

    async def test_unregistered_agent_is_reported_without_crashing(self):
        orchestrator = OrchestrationAgent(agent_registry={})
        schedule = [{"agent_name": "missing_agent", "priority": 1}]

        response = await orchestrator.reply(intention_message(schedule))
        result = json.loads(response.content)

        self.assertEqual(result["status"], "partial_failure")
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["results"][0]["agent_name"], "missing_agent")
        self.assertEqual(result["results"][0]["status"], "error")

    async def test_empty_schedule_returns_no_agents(self):
        orchestrator = OrchestrationAgent(agent_registry={})

        response = await orchestrator.reply(intention_message([]))
        result = json.loads(response.content)

        self.assertEqual(result["status"], "no_agents")
        self.assertEqual(result["message"], "没有需要调度的智能体")

    async def test_preference_and_trip_results_update_long_term_memory(self):
        memory = FakeMemoryManager()
        preference_agent = FakeAgent(
            "preference",
            payload={
                "preferences": [
                    {
                        "type": "hotel_brands",
                        "value": "汉庭",
                        "action": "append",
                    }
                ]
            },
        )
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": "杭州",
                "start_date": "2026-07-14",
                "end_date": "2026-07-16",
                "trip_purpose": "出差",
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": [{"day": 1}]}},
        )
        orchestrator = OrchestrationAgent(
            agent_registry={
                "preference": preference_agent,
                "event_collection": event_agent,
                "itinerary_planning": plan_agent,
            },
            memory_manager=memory,
        )

        schedule = [
            {"agent_name": "preference", "priority": 1},
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "itinerary_planning", "priority": 2},
        ]
        response = await orchestrator.reply(intention_message(schedule))
        result = json.loads(response.content)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(memory.long_term.preferences["hotel_brands"], ["如家", "汉庭"])
        self.assertEqual(
            memory.long_term.saved_trips,
            [
                {
                    "origin": "苏州",
                    "destination": "杭州",
                    "start_date": "2026-07-14",
                    "end_date": "2026-07-16",
                    "purpose": "出差",
                }
            ],
        )
        self.assertEqual(
            plan_agent.received_input["context"]["recent_dialogue"],
            [{"role": "user", "content": "最近一条对话"}],
        )
        self.assertEqual(
            plan_agent.received_input["context"]["user_preferences"]["hotel_brands"],
            ["如家"],
        )
        # 公共 context 在执行前生成；本轮新偏好通过 previous_results 传给 Priority 2。
        previous_by_agent = {
            item["agent_name"]: item for item in plan_agent.received_input["previous_results"]
        }
        self.assertEqual(
            previous_by_agent["preference"]["result"]["data"]["preferences"][0]["value"],
            "汉庭",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
