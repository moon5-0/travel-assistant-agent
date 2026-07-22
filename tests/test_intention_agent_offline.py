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


class FakeModel:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error

    async def __call__(self, _messages):
        if self.error:
            raise self.error
        return FakeResponse(self.content)


class FakeSkillLoader:
    def get_skill_prompt(self, _mapping):
        return "information_query: 查询实时信息"


def make_agent(model):
    agent = IntentionAgent(name="IntentionAgent", model=model)
    agent.skill_loader = FakeSkillLoader()
    return agent


class TestIntentionAgentOffline(unittest.IsolatedAsyncioTestCase):
    async def test_uses_robust_parser_for_markdown_and_common_format_issues(self):
        model_output = """模型分析：
```json
{'reasoning': '查询天气', 'intents': [], 'key_entities': {'city': '杭州'},
 'rewritten_query': '杭州天气', 'agent_schedule': [],}
```"""
        agent = make_agent(FakeModel(content=model_output))

        response = await agent.reply(
            Msg(name="User", content="杭州天气怎么样？", role="user")
        )
        result = json.loads(response.content)

        self.assertEqual(result["rewritten_query"], "杭州天气")
        self.assertEqual(result["key_entities"]["city"], "杭州")

    async def test_invalid_model_format_uses_default_schedule(self):
        agent = make_agent(FakeModel(content="无法输出结构化结果"))

        response = await agent.reply(
            Msg(name="User", content="帮我查一下", role="user")
        )
        result = json.loads(response.content)

        self.assertEqual(result["agent_schedule"][0]["agent_name"], "information_query")
        self.assertEqual(result["rewritten_query"], "帮我查一下")

    async def test_model_connection_error_propagates_to_retry_layer(self):
        agent = make_agent(FakeModel(error=ConnectionError("LLM unavailable")))

        with self.assertRaises(ConnectionError):
            await agent.reply(
                Msg(name="User", content="杭州天气怎么样？", role="user")
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
