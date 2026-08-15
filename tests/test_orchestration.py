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


from agents.orchestration_agent import OrchestrationAgent as ProductionOrchestrationAgent


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
        self._pending_trip: Dict[str, Any] = {}

    def get_pending_trip(self) -> Dict[str, Any]:
        return dict(self._pending_trip)

    def save_pending_trip(
        self,
        trip_data: Dict[str, Any],
    ) -> None:
        self._pending_trip = dict(trip_data)

    def clear_pending_trip(self) -> None:
        self._pending_trip.clear()


class OrchestrationAgent(ProductionOrchestrationAgent):
    """为单元测试注入可控的会话存储。

    生产代码由 MemoryManager 提供 Redis 会话状态；测试不连接
    真实 Redis，因此在这里显式使用 FakeMemoryManager。
    """

    def __init__(self, *args, memory_manager=None, **kwargs):
        super().__init__(
            *args,
            memory_manager=memory_manager or FakeMemoryManager(),
            **kwargs,
        )


def reference_planning_context():
    """与测试目标无关时使用的完整参考规划上下文。"""
    return {
        "departure_time_window": "上午",
        "return_time_window": "下午",
        "outbound_booking_status": "reference",
        "return_booking_status": "reference",
        "hotel_booking_status": "reference",
    }


def intention_message(
    schedule,
    original_user_input: str = (
        "2026年7月24日从苏州前往杭州，上午出发、下午返程，"
        "交通和酒店都没预订，先看参考方案"
    ),
    planning_signals=None,
):
    """构造协调器实际接收的 IntentionAgent 输出。"""
    data = {
        "reasoning": "测试调度",
        "intents": [{"type": "itinerary_planning", "confidence": 0.95}],
        "key_entities": {"origin": "苏州", "destination": "杭州"},
        "rewritten_query": "从苏州前往杭州",
        "agent_schedule": schedule,
    }
    if planning_signals is not None:
        data["planning_signals"] = planning_signals
    return Msg(
        name="IntentionAgent",
        content=json.dumps(data, ensure_ascii=False),
        role="assistant",
        metadata={"original_user_input": original_user_input},
    )


class TestOrchestrationAgent(unittest.IsolatedAsyncioTestCase):
    async def test_planning_signals_are_forwarded_to_planning_agent(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": "北京",
                "start_date": "2026-08-10",
                "duration_days": 2,
                "trip_purpose": "客户沟通",
                **reference_planning_context(),
                "missing_info": [],
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": [{"day": 1}]}},
        )
        orchestrator = OrchestrationAgent(agent_registry={
            "event_collection": event_agent,
            "itinerary_planning": plan_agent,
        })
        signals = {
            "trip_type": "business",
            "leisure_preference": "forbidden",
            "explicit_constraints": ["办完即返"],
        }

        await orchestrator.reply(intention_message(
            [
                {"agent_name": "event_collection", "priority": 1},
                {"agent_name": "itinerary_planning", "priority": 2},
            ],
            original_user_input=(
                "8月10日从苏州去北京两天，上午出发、下午返程，"
                "交通和酒店都没预订，办完就回来"
            ),
            planning_signals=signals,
        ))

        self.assertEqual(
            plan_agent.received_input["context"]["planning_signals"],
            signals,
        )

    async def test_unmentioned_trip_purpose_is_not_used_or_persisted(self):
        memory = FakeMemoryManager()
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": "北京",
                "start_date": "2026-08-10",
                "end_date": "2026-08-12",
                "duration_days": 3,
                # 模拟模型在用户没有说明目的时擅自补成“旅游”。
                "trip_purpose": "旅游",
                **reference_planning_context(),
                "missing_info": [],
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": [{"day": 1}]}},
        )
        orchestrator = OrchestrationAgent(
            agent_registry={
                "event_collection": event_agent,
                "itinerary_planning": plan_agent,
            },
            memory_manager=memory,
        )
        schedule = [
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "itinerary_planning", "priority": 2},
        ]

        response = await orchestrator.reply(
            intention_message(
                schedule,
                original_user_input=(
                    "8月10日从苏州去北京3天，上午出发、下午返程，"
                    "交通和酒店都没预订"
                ),
            )
        )
        result = json.loads(response.content)

        self.assertEqual(result["status"], "completed")
        previous_results = plan_agent.received_input["previous_results"]
        event_data = next(
            item["result"]["data"]
            for item in previous_results
            if item["agent_name"] == "event_collection"
        )
        self.assertIsNone(event_data.get("trip_purpose"))
        self.assertIsNone(memory.long_term.saved_trips[0]["purpose"])

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
                **reference_planning_context(),
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
                original_user_input=(
                    "明天上午从苏州去北京出差3天，最后一天下午返程，"
                    "交通和酒店都没预订"
                ),
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
                **reference_planning_context(),
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

    async def test_complete_core_trip_asks_for_time_and_booking_context(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": "北京",
                "start_date": "2026-08-10",
                "end_date": "2026-08-12",
                "duration_days": 3,
                "missing_info": [],
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(agent_registry={
            "event_collection": event_agent,
            "itinerary_planning": plan_agent,
        })

        response = await orchestrator.reply(intention_message(
            [
                {"agent_name": "event_collection", "priority": 1},
                {"agent_name": "itinerary_planning", "priority": 2},
            ],
            original_user_input="8月10日从苏州去北京三天",
        ))
        result = json.loads(response.content)

        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(
            result["missing_fields"],
            [
                "departure_time_window",
                "return_time_window",
                "outbound_booking_status",
                "return_booking_status",
                "hotel_booking_status",
            ],
        )
        self.assertIn("大致去程和返程时段", result["message"])
        self.assertIn("是否已预订", result["message"])
        self.assertIsNone(plan_agent.started_at)

    async def test_reference_context_allows_itinerary_planning(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": "北京",
                "start_date": "2026-08-10",
                "end_date": "2026-08-12",
                "duration_days": 3,
                "departure_time_window": "上午",
                "return_time_window": "下午",
                "outbound_booking_status": "reference",
                "return_booking_status": "reference",
                "hotel_booking_status": "reference",
                "missing_info": [],
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(agent_registry={
            "event_collection": event_agent,
            "itinerary_planning": plan_agent,
        })

        response = await orchestrator.reply(intention_message(
            [
                {"agent_name": "event_collection", "priority": 1},
                {"agent_name": "itinerary_planning", "priority": 2},
            ],
            original_user_input=(
                "8月10日从苏州去北京三天，上午出发，最后一天下午返程，"
                "去程、返程和酒店都没预订，先看参考方案"
            ),
        ))
        result = json.loads(response.content)

        self.assertEqual(result["status"], "completed")
        self.assertIsNotNone(plan_agent.started_at)

    async def test_unmentioned_return_time_and_transport_status_are_rejected(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": "北京",
                "start_date": "2026-08-10",
                "end_date": "2026-08-12",
                "duration_days": 3,
                "departure_time_window": "上午",
                # 模拟模型擅自补充用户没说的返程时段和交通状态。
                "return_time_window": "下午",
                "outbound_booking_status": "reference",
                "return_booking_status": "reference",
                "hotel_booking_status": "confirmed",
                "hotel_booking_details": "北京国贸全季酒店",
                "missing_info": [],
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(agent_registry={
            "event_collection": event_agent,
            "itinerary_planning": plan_agent,
        })

        response = await orchestrator.reply(intention_message(
            [
                {"agent_name": "event_collection", "priority": 1},
                {"agent_name": "itinerary_planning", "priority": 2},
            ],
            original_user_input=(
                "8月10日上午从苏州出发去北京，"
                "酒店已经订好北京国贸全季酒店"
            ),
        ))
        result = json.loads(response.content)

        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(
            result["missing_fields"],
            [
                "return_time_window",
                "outbound_booking_status",
                "return_booking_status",
            ],
        )
        pending = orchestrator.get_pending_trip()
        self.assertEqual(pending["departure_time_window"], "上午")
        self.assertNotIn("return_time_window", pending)
        self.assertEqual(pending["hotel_booking_status"], "confirmed")
        self.assertNotIn("outbound_booking_status", pending)
        self.assertIsNone(plan_agent.started_at)

    async def test_two_tokens_in_one_departure_time_do_not_imply_return_time(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": "北京",
                "start_date": "2026-08-10",
                "end_date": "2026-08-12",
                "duration_days": 3,
                "departure_time_window": "上午10点",
                # 模拟模型把同一个去程时间误当成返程时间依据。
                "return_time_window": "下午",
                "outbound_booking_status": "reference",
                "return_booking_status": "reference",
                "hotel_booking_status": "reference",
                "missing_info": [],
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(agent_registry={
            "event_collection": event_agent,
            "itinerary_planning": plan_agent,
        })

        response = await orchestrator.reply(intention_message(
            [
                {"agent_name": "event_collection", "priority": 1},
                {"agent_name": "itinerary_planning", "priority": 2},
            ],
            original_user_input=(
                "8月10日上午10点从苏州出发去北京三天，"
                "车票酒店都没订，先看参考方案"
            ),
        ))
        result = json.loads(response.content)

        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("return_time_window", result["missing_fields"])
        self.assertNotIn(
            "return_time_window",
            orchestrator.get_pending_trip(),
        )
        self.assertIsNone(plan_agent.started_at)

    async def test_return_booking_statement_does_not_authorize_outbound_status(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": "北京",
                "start_date": "2026-08-10",
                "duration_days": 3,
                "departure_time_window": "上午",
                "return_time_window": "下午",
                # 模拟模型根据返程描述擅自补出相同的去程状态。
                "outbound_booking_status": "reference",
                "return_booking_status": "reference",
                "hotel_booking_status": "confirmed",
                "hotel_booking_details": "北京国贸全季酒店",
                "missing_info": [],
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(agent_registry={
            "event_collection": event_agent,
            "itinerary_planning": plan_agent,
        })

        response = await orchestrator.reply(intention_message(
            [
                {"agent_name": "event_collection", "priority": 1},
                {"agent_name": "itinerary_planning", "priority": 2},
            ],
            original_user_input=(
                "8月10日上午从苏州出发去北京三天，最后一天下午回来，"
                "返程车票没订，酒店已经订好北京国贸全季酒店"
            ),
        ))
        result = json.loads(response.content)

        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["missing_fields"], ["outbound_booking_status"])
        pending = orchestrator.get_pending_trip()
        self.assertNotIn("outbound_booking_status", pending)
        self.assertEqual(pending["return_booking_status"], "reference")
        self.assertEqual(pending["hotel_booking_status"], "confirmed")
        self.assertIsNone(plan_agent.started_at)

    async def test_partial_confirmed_booking_context_merges_across_turns(self):
        event_agent = FakeAgent(
            "event_collection",
            payload={
                "origin": "苏州",
                "destination": "北京",
                "start_date": "2026-08-10",
                "end_date": "2026-08-12",
                "duration_days": 3,
                "missing_info": [],
            },
        )
        plan_agent = FakeAgent(
            "itinerary_planning",
            payload={"itinerary": {"days": []}},
        )
        orchestrator = OrchestrationAgent(agent_registry={
            "event_collection": event_agent,
            "itinerary_planning": plan_agent,
        })
        schedule = [
            {"agent_name": "event_collection", "priority": 1},
            {"agent_name": "itinerary_planning", "priority": 2},
        ]

        first_response = await orchestrator.reply(intention_message(
            schedule,
            original_user_input="8月10日从苏州去北京三天",
        ))
        self.assertEqual(
            json.loads(first_response.content)["status"],
            "needs_clarification",
        )

        event_agent.payload = {
            "departure_time_window": "上午",
            "return_time_window": "下午",
            "outbound_booking_status": "reference",
            "return_booking_status": "reference",
            "hotel_booking_status": "confirmed",
            "hotel_booking_details": "北京国贸全季酒店",
            "missing_info": [],
        }
        second_response = await orchestrator.reply(intention_message(
            schedule,
            original_user_input=(
                "上午出发，最后一天下午返程，车票没订；"
                "酒店已经订好北京国贸全季酒店"
            ),
        ))
        result = json.loads(second_response.content)

        self.assertEqual(result["status"], "completed")
        previous_results = plan_agent.received_input["previous_results"]
        event_data = next(
            item["result"]["data"]
            for item in previous_results
            if item["agent_name"] == "event_collection"
        )
        self.assertEqual(event_data["origin"], "苏州")
        self.assertEqual(event_data["departure_time_window"], "上午")
        self.assertEqual(event_data["outbound_booking_status"], "reference")
        self.assertEqual(event_data["hotel_booking_status"], "confirmed")
        self.assertEqual(
            event_data["hotel_booking_details"],
            "北京国贸全季酒店",
        )

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
            **reference_planning_context(),
            # EventCollectionAgent 只看本轮输入，因此仍会认为目的地缺失。
            # 调度器合并上一轮“北京”后必须重新计算，不能保留这个旧状态。
            "missing_info": ["destination"],
        }

        second_response = await orchestrator.reply(intention_message(
            schedule,
            original_user_input=(
                "7月24日上午从苏州出发，7月26日下午返程，"
                "交通和酒店都没预订，先看参考方案"
            ),
        ))
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

        # 第三轮：基础信息完整，但仍应进入时间与预订状态澄清。
        event_agent.payload = {
            "start_date": "2026-07-25",
            "duration_days": 3,
        }

        third_response = await orchestrator.reply(
            intention_message(schedule)
        )
        third_result = json.loads(third_response.content)

        self.assertEqual(third_result["status"], "needs_clarification")
        self.assertEqual(
            third_result["missing_fields"],
            [
                "departure_time_window",
                "return_time_window",
                "outbound_booking_status",
                "return_booking_status",
                "hotel_booking_status",
            ],
        )
        self.assertIsNone(plan_agent.started_at)

        # 第四轮：补充时间范围并选择参考方案后才执行规划。
        event_agent.payload = reference_planning_context()
        fourth_response = await orchestrator.reply(intention_message(
            schedule,
            original_user_input=(
                "上午出发，最后一天下午返程，交通和酒店都没预订，"
                "先看参考方案"
            ),
        ))
        fourth_result = json.loads(fourth_response.content)

        self.assertEqual(fourth_result["status"], "completed")
        self.assertIsNotNone(plan_agent.started_at)

        # 行程规划完成后，临时状态应该被清空。
        self.assertEqual(orchestrator.get_pending_trip(), {})

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
            orchestrator.get_pending_trip()["destination"],
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
                **reference_planning_context(),
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
        response = await orchestrator.reply(
            intention_message(
                schedule,
                original_user_input=(
                    "2026年7月14日上午从苏州前往杭州出差，"
                    "7月16日下午返程，交通和酒店都没预订"
                ),
            )
        )
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
                    "duration_days": None,
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
