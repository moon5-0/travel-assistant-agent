#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IntentionAgent JSON 接入与重试边界的离线测试。"""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


try:
    from agentscope.message import Msg
except ModuleNotFoundError:
    agentscope_module = types.ModuleType("agentscope")
    agent_module = types.ModuleType("agentscope.agent")
    message_module = types.ModuleType("agentscope.message")

    class AgentBase:
        pass

    class Msg:
        def __init__(self, name, content, role):
            self.name = name
            self.content = content
            self.role = role

    agent_module.AgentBase = AgentBase
    message_module.Msg = Msg
    agentscope_module.agent = agent_module
    agentscope_module.message = message_module
    sys.modules["agentscope"] = agentscope_module
    sys.modules["agentscope.agent"] = agent_module
    sys.modules["agentscope.message"] = message_module


from agents.intention_agent import IntentionAgent


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FailingStreamResponse:
    """模拟模型已建立连接，但读取流时发生网络中断。"""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise ConnectionError("stream interrupted")


class FakeModel:
    def __init__(self, content=None, error=None, responses=None):
        self.content = content
        self.error = error
        self.responses = list(responses or [])
        self.calls = []

    async def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.error:
            raise self.error
        if self.responses:
            return FakeResponse(self.responses.pop(0))
        return FakeResponse(self.content)


class FailingStreamModel:
    def __init__(self):
        self.calls = []

    async def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return FailingStreamResponse()


class FakeSkillLoader:
    def get_skill_prompt(self, _mapping):
        return "information_query: 查询实时信息"


def make_agent(model):
    agent = IntentionAgent(name="IntentionAgent", model=model)
    agent.skill_loader = FakeSkillLoader()
    return agent


class TestIntentionAgentOffline(unittest.IsolatedAsyncioTestCase):
    async def test_preserves_planning_signals_for_uncommon_expression(self):
        model_output = json.dumps({
            "reasoning": "用户要求纯商务出行",
            "intents": [
                {"type": "itinerary_planning", "confidence": 0.98},
            ],
            "key_entities": {"destination": "北京"},
            "rewritten_query": "去北京处理工作，办完即返",
            "planning_signals": {
                "trip_type": "business",
                "leisure_preference": "forbidden",
                "explicit_constraints": ["办完即返"],
            },
            "agent_schedule": [
                {"agent_name": "event_collection", "priority": 1},
                {"agent_name": "itinerary_planning", "priority": 2},
            ],
        }, ensure_ascii=False)
        model = FakeModel(content=model_output)
        agent = make_agent(model)

        response = await agent.reply(Msg(
            name="User",
            content="我不想把这趟差事搞得跟度假一样，事情办完就回来",
            role="user",
        ))
        result = json.loads(response.content)

        self.assertEqual(result["planning_signals"]["trip_type"], "business")
        self.assertEqual(
            result["planning_signals"]["leisure_preference"],
            "forbidden",
        )
        prompt = model.calls[0]["messages"][1]["content"]
        self.assertIn("不直接决定最终 planning_mode", prompt)

    async def test_old_output_gets_safe_default_planning_signals(self):
        model_output = json.dumps({
            "reasoning": "用户查询天气",
            "intents": [
                {"type": "information_query", "confidence": 0.95},
            ],
            "key_entities": {"city": "杭州"},
            "rewritten_query": "查询杭州天气",
            "agent_schedule": [
                {"agent_name": "information_query", "priority": 1},
            ],
        }, ensure_ascii=False)
        agent = make_agent(FakeModel(content=model_output))

        response = await agent.reply(
            Msg(name="User", content="杭州天气怎么样", role="user")
        )
        result = json.loads(response.content)

        self.assertEqual(result["planning_signals"], {
            "trip_type": "unknown",
            "leisure_preference": "unspecified",
            "explicit_constraints": [],
        })

    async def test_explicit_previous_trip_reuse_adds_memory_query(self):
        model_output = json.dumps({
            "reasoning": "用户希望按上次行程继续规划",
            "intents": [
                {"type": "itinerary_planning", "confidence": 0.95},
            ],
            "key_entities": {
                "destination": "北京",
                "start_date": "2026-08-20",
            },
            "rewritten_query": "2026年8月20日从苏州去北京3天",
            "agent_schedule": [
                {"agent_name": "event_collection", "priority": 1},
                {"agent_name": "itinerary_planning", "priority": 2},
            ],
        }, ensure_ascii=False)
        agent = make_agent(FakeModel(content=model_output))

        response = await agent.reply(
            Msg(
                name="User",
                content=(
                    "2026年8月20日再去北京，出发地和行程天数"
                    "都按上次一样，帮我规划行程"
                ),
                role="user",
            )
        )
        result = json.loads(response.content)

        self.assertEqual(
            [item["agent_name"] for item in result["agent_schedule"]],
            ["memory_query", "event_collection", "itinerary_planning"],
        )

    async def test_history_preference_trip_adds_collection_and_removes_write_agent(self):
        model_output = json.dumps({
            "reasoning": "用户希望引用历史酒店偏好规划行程",
            "intents": [
                {"type": "memory_query", "confidence": 0.95},
                {"type": "itinerary_planning", "confidence": 0.95},
            ],
            "key_entities": {"destination": "北京"},
            "rewritten_query": "按照历史酒店偏好规划北京行程",
            "agent_schedule": [
                {"agent_name": "memory_query", "priority": 1},
                {"agent_name": "preference", "priority": 1},
                {"agent_name": "itinerary_planning", "priority": 2},
            ],
        }, ensure_ascii=False)
        agent = make_agent(FakeModel(content=model_output))

        response = await agent.reply(
            Msg(
                name="User",
                content="按照我之前的酒店偏好规划北京行程",
                role="user",
            )
        )
        result = json.loads(response.content)

        self.assertEqual(
            [item["agent_name"] for item in result["agent_schedule"]],
            ["memory_query", "event_collection", "itinerary_planning"],
        )

    async def test_saved_preference_question_routes_to_memory_only(self):
        model_output = json.dumps({
            "reasoning": "用户在查询信息",
            "intents": [
                {"type": "information_query", "confidence": 0.8},
            ],
            "key_entities": {},
            "rewritten_query": "查询保存的酒店品牌偏好",
            "agent_schedule": [
                {"agent_name": "information_query", "priority": 1},
            ],
        }, ensure_ascii=False)
        agent = make_agent(FakeModel(content=model_output))

        response = await agent.reply(
            Msg(
                name="User",
                content="我之前保存了哪些酒店品牌偏好？",
                role="user",
            )
        )
        result = json.loads(response.content)

        self.assertEqual(
            [item["agent_name"] for item in result["agent_schedule"]],
            ["memory_query"],
        )
        self.assertEqual(
            [item["type"] for item in result["intents"]],
            ["memory_query"],
        )

    async def test_explicit_preference_update_keeps_preference_agent(self):
        model_output = json.dumps({
            "reasoning": "用户正在追加酒店偏好",
            "intents": [
                {"type": "preference", "confidence": 0.98},
            ],
            "key_entities": {"other": "汉庭"},
            "rewritten_query": "追加汉庭酒店偏好",
            "agent_schedule": [
                {"agent_name": "preference", "priority": 1},
            ],
        }, ensure_ascii=False)
        agent = make_agent(FakeModel(content=model_output))

        response = await agent.reply(
            Msg(name="User", content="我还喜欢汉庭酒店", role="user")
        )
        result = json.loads(response.content)

        self.assertEqual(
            [item["agent_name"] for item in result["agent_schedule"]],
            ["preference"],
        )

    async def test_uses_robust_parser_for_markdown_and_common_format_issues(self):
        model_output = """模型分析：
```json
{'reasoning': '查询天气', 'intents': [], 'key_entities': {'city': '杭州'},
 'rewritten_query': '杭州天气', 'agent_schedule': [],}
```"""
        model = FakeModel(content=model_output)
        agent = make_agent(model)

        response = await agent.reply(
            Msg(name="User", content="杭州天气怎么样？", role="user")
        )
        result = json.loads(response.content)

        self.assertEqual(result["rewritten_query"], "杭州天气")
        self.assertEqual(result["key_entities"]["city"], "杭州")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            model.calls[0]["kwargs"]["response_format"],
            {"type": "json_object"},
        )

    async def test_invalid_json_is_repaired_once_and_keeps_original_intent(self):
        invalid_output = """{
  "reasoning": "用户说"帮我规划北京行程"，属于行程规划",
  "intents": [{"type": "itinerary_planning", "confidence": 0.98}],
  "key_entities": {"destination": "北京"},
  "rewritten_query": "帮我规划北京行程",
  "agent_schedule": [{"agent_name": "itinerary_planning", "priority": 2}]
}"""
        repaired_output = json.dumps({
            "reasoning": "用户需要规划北京行程",
            "intents": [
                {"type": "itinerary_planning", "confidence": 0.98},
            ],
            "key_entities": {"destination": "北京"},
            "rewritten_query": "帮我规划北京行程",
            "agent_schedule": [
                {"agent_name": "event_collection", "priority": 1},
                {"agent_name": "itinerary_planning", "priority": 2},
            ],
        }, ensure_ascii=False)
        model = FakeModel(responses=[invalid_output, repaired_output])
        agent = make_agent(model)

        response = await agent.reply(
            Msg(name="User", content="帮我规划北京行程", role="user")
        )
        result = json.loads(response.content)

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(
            [item["agent_name"] for item in result["agent_schedule"]],
            ["event_collection", "itinerary_planning"],
        )
        self.assertIn("校验错误", model.calls[1]["messages"][1]["content"])

    async def test_valid_json_with_invalid_schema_is_repaired_once(self):
        invalid_schema = json.dumps({
            "reasoning": "用户需要规划行程",
            "intents": [],
            "key_entities": {"destination": "北京"},
            "rewritten_query": "规划北京行程",
            "agent_schedule": "itinerary_planning",
        }, ensure_ascii=False)
        repaired_output = json.dumps({
            "reasoning": "用户需要规划行程",
            "intents": [],
            "key_entities": {"destination": "北京"},
            "rewritten_query": "规划北京行程",
            "agent_schedule": [
                {"agent_name": "event_collection", "priority": 1},
            ],
        }, ensure_ascii=False)
        model = FakeModel(responses=[invalid_schema, repaired_output])
        agent = make_agent(model)

        response = await agent.reply(
            Msg(name="User", content="规划北京行程", role="user")
        )
        result = json.loads(response.content)

        self.assertEqual(len(model.calls), 2)
        self.assertEqual(
            result["agent_schedule"][0]["agent_name"],
            "event_collection",
        )

    async def test_invalid_model_format_uses_default_schedule(self):
        model = FakeModel(content="无法输出结构化结果")
        agent = make_agent(model)

        response = await agent.reply(
            Msg(name="User", content="帮我查一下", role="user")
        )
        result = json.loads(response.content)

        self.assertEqual(result["agent_schedule"][0]["agent_name"], "information_query")
        self.assertEqual(result["rewritten_query"], "帮我查一下")
        self.assertEqual(len(model.calls), 2)

    async def test_model_connection_error_propagates_to_retry_layer(self):
        agent = make_agent(FakeModel(error=ConnectionError("LLM unavailable")))

        with self.assertRaises(ConnectionError):
            await agent.reply(
                Msg(name="User", content="杭州天气怎么样？", role="user")
            )

    async def test_stream_connection_error_propagates_without_format_repair(self):
        model = FailingStreamModel()
        agent = make_agent(model)

        with self.assertRaisesRegex(ConnectionError, "stream interrupted"):
            await agent.reply(
                Msg(name="User", content="杭州天气怎么样？", role="user")
            )

        self.assertEqual(len(model.calls), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
