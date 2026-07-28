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


if __name__ == "__main__":
    unittest.main()
