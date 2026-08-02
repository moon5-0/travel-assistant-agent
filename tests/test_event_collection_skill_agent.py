"""EventCollection Skill Agent新增规划上下文字段的离线测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from agentscope.agent import AgentBase
from agentscope.message import Msg


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_event_collection_agent():
    agent_path = (
        PROJECT_ROOT
        / ".claude"
        / "skills"
        / "event-collection"
        / "script"
        / "agent.py"
    )
    spec = importlib.util.spec_from_file_location(
        "event_collection_skill_agent",
        agent_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 EventCollectionAgent: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EventCollectionAgent


EventCollectionAgent = load_event_collection_agent()


class FakeModel:
    def __init__(self, output):
        self.output = output
        self.calls = []

    async def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return SimpleNamespace(content=self.output)


def make_agent(model):
    agent = object.__new__(EventCollectionAgent)
    AgentBase.__init__(agent)
    agent.name = "EventCollectionAgent"
    agent.model = model
    return agent


def make_input(query):
    return Msg(
        name="Orchestrator",
        content=json.dumps({
            "context": {
                "rewritten_query": query,
                "user_preferences": {"hotel_brands": ["汉庭"]},
            }
        }, ensure_ascii=False),
        role="user",
    )


class TestEventCollectionSkillAgent(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_extracts_time_windows_and_booking_context(self):
        model = FakeModel(json.dumps({
            "origin": "苏州",
            "destination": "北京",
            "start_date": "2026-08-10",
            "end_date": "2026-08-12",
            "duration_days": 3,
            "departure_time_window": "上午",
            "return_time_window": "下午",
            "outbound_booking_status": "reference",
            "return_booking_status": "reference",
            "hotel_booking_status": "confirmed",
            "hotel_booking_details": "北京国贸全季酒店",
            "missing_info": [],
        }, ensure_ascii=False))
        agent = make_agent(model)

        response = await agent.reply(make_input(
            "上午出发，最后一天下午返程，车票没订，酒店已订国贸全季"
        ))
        result = json.loads(response.content)
        prompt = model.calls[0]["messages"][0]["content"]

        self.assertEqual(result["hotel_booking_status"], "confirmed")
        self.assertEqual(result["hotel_booking_details"], "北京国贸全季酒店")
        self.assertIn("departure_time_window", prompt)
        self.assertIn("return_time_window", prompt)
        self.assertIn("偏好不等于已经预订", prompt)
        self.assertIn("时间不限", prompt)
        self.assertEqual(
            model.calls[0]["kwargs"]["response_format"],
            {"type": "json_object"},
        )

    async def test_reference_status_discards_hallucinated_booking_details(self):
        model = FakeModel(json.dumps({
            "outbound_booking_status": "reference",
            "outbound_booking_details": "G123次高铁",
            "return_booking_status": "unknown",
            "return_booking_details": "G456次高铁",
            "hotel_booking_status": "reference",
            "hotel_booking_details": "汉庭酒店",
        }, ensure_ascii=False))
        agent = make_agent(model)

        response = await agent.reply(make_input("都没预订，先看参考方案"))
        result = json.loads(response.content)

        self.assertIsNone(result["outbound_booking_details"])
        self.assertIsNone(result["return_booking_status"])
        self.assertIsNone(result["return_booking_details"])
        self.assertIsNone(result["hotel_booking_details"])


if __name__ == "__main__":
    unittest.main()
