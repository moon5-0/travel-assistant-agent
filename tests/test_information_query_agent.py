#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""InformationQueryAgent 的离线单元测试。

测试使用固定返回值替代 wttr.in、DDGS 和真实 LLM，主要验证天气/网络
搜索路由、协调器 JSON 输入解析、城市提取和可疑 URL 过滤。

运行：python3 tests/test_information_query_agent.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# 让测试在未安装 AgentScope 的环境中也能运行。
try:
    from agentscope.message import Msg
except ModuleNotFoundError:
    agentscope_module = types.ModuleType("agentscope")
    agent_module = types.ModuleType("agentscope.agent")
    message_module = types.ModuleType("agentscope.message")

    class AgentBase:
        """测试所需的最小 AgentBase 替身。"""

    class Msg:
        """测试所需的最小 Msg 替身。"""

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


def load_information_query_module():
    """从 query-info Skill 的真实位置加载待测模块。"""
    agent_path = (
        PROJECT_ROOT / ".claude" / "skills" / "query-info" / "script" / "agent.py"
    )
    spec = importlib.util.spec_from_file_location("query_info_skill_agent", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 InformationQueryAgent: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QUERY_INFO_MODULE = load_information_query_module()
InformationQueryAgent = QUERY_INFO_MODULE.InformationQueryAgent
is_suspicious_url = QUERY_INFO_MODULE._is_suspicious_url


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    """返回固定摘要并记录 Prompt 的离线模型。"""

    def __init__(self, summary="杭州近期有人工智能展会。"):
        self.summary = summary
        self.calls = []

    async def __call__(self, messages):
        self.calls.append(messages)
        return FakeResponse(self.summary)


def make_agent(model=None):
    agent = object.__new__(InformationQueryAgent)
    agent.name = "InformationQueryAgent"
    agent.model = model
    agent.skill_loader = types.SimpleNamespace(
        get_skill_content=lambda _name: "请基于查询结果简洁回答。"
    )
    return agent


class TestInformationQueryAgent(unittest.IsolatedAsyncioTestCase):
    async def test_weather_query_uses_weather_service_only(self):
        agent = make_agent()
        weather_result = {
            "query_type": "天气查询",
            "query_success": True,
            "results": {"summary": "杭州晴，气温 28°C。"},
        }
        agent._weather_query = AsyncMock(return_value=weather_result)
        agent._web_search = AsyncMock()

        response = await agent.reply(
            Msg(name="User", content="杭州明天天气怎么样？", role="user")
        )
        result = json.loads(response.content)

        self.assertEqual(result, weather_result)
        agent._weather_query.assert_awaited_once_with("杭州明天天气怎么样？")
        agent._web_search.assert_not_awaited()

    async def test_non_weather_query_uses_web_search(self):
        agent = make_agent()
        web_result = {
            "query_type": "网络搜索",
            "query_success": True,
            "results": {"summary": "杭州近期有人工智能展会。"},
        }
        agent._weather_query = AsyncMock()
        agent._web_search = AsyncMock(return_value=web_result)
        message = Msg(
            name="OrchestrationAgent",
            content=json.dumps(
                {"context": {"rewritten_query": "搜索杭州最近的展会活动"}},
                ensure_ascii=False,
            ),
            role="user",
        )

        response = await agent.reply(message)
        result = json.loads(response.content)

        self.assertEqual(result, web_result)
        agent._weather_query.assert_not_awaited()
        agent._web_search.assert_awaited_once_with("搜索杭州最近的展会活动")

    async def test_failed_weather_query_falls_back_to_web_search(self):
        agent = make_agent()
        agent._weather_query = AsyncMock(
            return_value={
                "query_type": "天气查询",
                "query_success": False,
                "results": {"message": "天气接口暂时不可用"},
            }
        )
        fallback_result = {
            "query_type": "网络搜索",
            "query_success": True,
            "results": {"summary": "联网搜索到杭州天气信息。"},
        }
        agent._web_search = AsyncMock(return_value=fallback_result)

        response = await agent.reply(
            Msg(name="User", content="杭州明天天气怎么样？", role="user")
        )
        result = json.loads(response.content)

        self.assertEqual(result, fallback_result)
        agent._weather_query.assert_awaited_once()
        agent._web_search.assert_awaited_once_with("杭州明天天气怎么样？")

    async def test_none_input_returns_explicit_failure(self):
        agent = make_agent()

        response = await agent.reply(None)
        result = json.loads(response.content)

        self.assertFalse(result["query_success"])

    async def test_search_summary_uses_fake_model_and_includes_sources_in_prompt(self):
        model = FakeModel()
        agent = make_agent(model)
        results = [
            {
                "title": "杭州展会信息",
                "snippet": "本周将举办人工智能产业展。",
                "url": "https://example.com/expo",
            }
        ]

        summary = await agent._summarize_search_results("杭州最近的展会", results)

        self.assertEqual(summary, "杭州近期有人工智能展会。")
        prompt = model.calls[0][0]["content"]
        self.assertIn("杭州最近的展会", prompt)
        self.assertIn("人工智能产业展", prompt)


class TestInformationQueryHelpers(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_query_type_and_city_detection(self):
        self.assertTrue(self.agent._is_weather_query("杭州明天下雨吗？"))
        self.assertFalse(self.agent._is_weather_query("杭州最近有什么展会？"))
        self.assertEqual(self.agent._extract_city_from_query("杭州明天天气"), "杭州")

    def test_suspicious_url_filter(self):
        self.assertTrue(is_suspicious_url("https://spam-example.xyz/article"))
        self.assertTrue(is_suspicious_url("not-a-url"))
        self.assertFalse(is_suspicious_url("https://www.gov.cn/policy/index.html"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
