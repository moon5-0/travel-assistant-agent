"""ItineraryPlanningAgent 结构化输出可靠性的离线测试。"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

from agentscope.agent import AgentBase
from agentscope.message import Msg


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_itinerary_planning_agent():
    agent_path = (
        PROJECT_ROOT / ".claude" / "skills" / "plan-trip" / "script" / "agent.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plan_trip_skill_agent",
        agent_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 ItineraryPlanningAgent: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ItineraryPlanningAgent


ItineraryPlanningAgent = load_itinerary_planning_agent()


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return FakeResponse(self.responses.pop(0))


def make_agent(model):
    agent = object.__new__(ItineraryPlanningAgent)
    AgentBase.__init__(agent)
    agent.name = "ItineraryPlanningAgent"
    agent.model = model
    agent.skill_loader = types.SimpleNamespace(
        get_skill_content=lambda _name: "生成结构化行程。"
    )
    return agent


def make_input():
    content = {
        "context": {
            "rewritten_query": "2026年8月10日从苏州去北京3天",
            "user_preferences": {},
        },
        "previous_results": [
            {
                "agent_name": "event_collection",
                "result": {
                    "data": {
                        "origin": "苏州",
                        "destination": "北京",
                        "start_date": "2026-08-10",
                        "duration_days": 3,
                    }
                },
            }
        ],
    }
    return Msg(
        name="Orchestrator",
        content=json.dumps(content, ensure_ascii=False),
        role="user",
    )


class TestItineraryPlanningAgentOffline(unittest.IsolatedAsyncioTestCase):
    async def test_valid_json_does_not_trigger_repair(self):
        valid = json.dumps({
            "itinerary": {
                "title": "北京3日游",
                "duration": "3天",
                "daily_plans": [],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        model = FakeModel([valid])
        agent = make_agent(model)

        response = await agent.reply(make_input())
        result = json.loads(response.content)

        self.assertTrue(result["planning_complete"])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            model.calls[0]["kwargs"]["response_format"],
            {"type": "json_object"},
        )
        self.assertIn(
            "活动时间框、描述中的交通时间或耗时、下一项活动开始时间",
            model.calls[0]["messages"][0]["content"],
        )

    async def test_repairs_malformed_json_once(self):
        malformed = """{
          "itinerary": {
            "title": "北京3日游",
            "duration": "3天",
            "daily_plans": [{
              "day": 1,
              "activities": [{"description": "参观"故宫""}]
            }]
          },
          "planning_complete": true
        }"""
        repaired = json.dumps({
            "itinerary": {
                "title": "北京3日游",
                "duration": "3天",
                "daily_plans": [
                    {
                        "day": 1,
                        "activities": [
                            {"description": "参观故宫"},
                        ],
                    }
                ],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        model = FakeModel([malformed, repaired])
        agent = make_agent(model)

        response = await agent.reply(make_input())
        result = json.loads(response.content)

        self.assertTrue(result["planning_complete"])
        self.assertEqual(result["itinerary"]["title"], "北京3日游")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(
            model.calls[0]["kwargs"]["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            model.calls[1]["kwargs"]["response_format"],
            {"type": "json_object"},
        )

    async def test_second_invalid_response_returns_existing_error_result(self):
        model = FakeModel(["{invalid", "still invalid"])
        agent = make_agent(model)

        response = await agent.reply(make_input())
        result = json.loads(response.content)

        self.assertFalse(result["planning_complete"])
        self.assertIn("error", result)
        self.assertEqual(len(model.calls), 2)

    async def test_time_conflict_triggers_one_targeted_repair(self):
        conflicting = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "1天",
                "daily_plans": [
                    {
                        "day": 1,
                        "date": "2026-08-10",
                        "activities": [
                            {
                                "time": "07:00-08:00",
                                "location": "苏州北站至北京南站",
                                "description": (
                                    "乘坐G4次高铁（07:00-11:25）前往北京。"
                                ),
                            },
                            {
                                "time": "12:00-13:00",
                                "location": "酒店",
                                "description": "办理入住。",
                            },
                        ],
                    }
                ],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        repaired = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "1天",
                "daily_plans": [
                    {
                        "day": 1,
                        "date": "2026-08-10",
                        "activities": [
                            {
                                "time": "07:00-11:30",
                                "location": "苏州北站至北京南站",
                                "description": (
                                    "乘坐G4次高铁（07:00-11:25）前往北京。"
                                ),
                            },
                            {
                                "time": "12:00-13:00",
                                "location": "酒店",
                                "description": "办理入住。",
                            },
                        ],
                    }
                ],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        model = FakeModel([conflicting, repaired])
        agent = make_agent(model)

        response = await agent.reply(make_input())
        result = json.loads(response.content)

        self.assertEqual(
            result["itinerary"]["daily_plans"][0]["activities"][0]["time"],
            "07:00-11:30",
        )
        self.assertEqual(len(model.calls), 2)
        self.assertIn(
            "transport_time_outside_activity",
            model.calls[1]["messages"][1]["content"],
        )
        self.assertIn(
            "允许调整、缩短、重新排序或删除",
            model.calls[1]["messages"][0]["content"],
        )
        self.assertIn(
            "普通用户偏好属于软约束",
            model.calls[1]["messages"][0]["content"],
        )

    async def test_failed_time_repair_keeps_original_itinerary(self):
        conflicting = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "1天",
                "daily_plans": [
                    {
                        "day": 1,
                        "activities": [
                            {
                                "time": "09:00-10:00",
                                "location": "宁波站至苏州站",
                                "description": "乘坐高铁，车程约2小时。",
                            }
                        ],
                    }
                ],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        model = FakeModel([conflicting, "{invalid"])
        agent = make_agent(model)

        response = await agent.reply(make_input())
        result = json.loads(response.content)

        self.assertFalse(result["planning_complete"])
        self.assertEqual(
            result["itinerary"]["daily_plans"][0]["activities"][0]["time"],
            "09:00-10:00",
        )
        self.assertEqual(
            result["time_consistency"]["status"],
            "unresolved",
        )
        self.assertIn(
            "存在未能自动解决的时间冲突",
            result["itinerary"]["notes"][0],
        )
        self.assertEqual(len(model.calls), 2)

    async def test_remaining_conflict_after_repair_is_marked_unresolved(self):
        conflicting = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "1天",
                "daily_plans": [
                    {
                        "day": 1,
                        "activities": [
                            {
                                "time": "09:00-10:00",
                                "location": "宁波站至苏州站",
                                "description": "乘坐高铁，车程约2小时。",
                            }
                        ],
                    }
                ],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        model = FakeModel([conflicting, conflicting])
        agent = make_agent(model)

        response = await agent.reply(make_input())
        result = json.loads(response.content)

        self.assertFalse(result["planning_complete"])
        self.assertEqual(
            result["time_consistency"]["issues"][0]["category"],
            "transport_duration_exceeds_activity",
        )
        self.assertEqual(len(model.calls), 2)


if __name__ == "__main__":
    unittest.main()
