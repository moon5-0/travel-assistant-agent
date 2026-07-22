#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAGKnowledgeAgent 的离线单元测试。

测试使用固定的检索结果和 FakeModel，不加载真实 Embedding 模型、Milvus
数据库或在线 LLM，主要验证：

1. 能否从协调器 JSON 中提取改写后的问题；
2. 能否把检索片段交给模型并返回来源；
3. 无知识与无模型时的降级行为；
4. Milvus 返回结果的格式转换和 top_k 参数。

运行：python3 tests/test_rag_agent.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock


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


def load_rag_agent_class():
    """从当前 Skill 目录加载 RAG Agent，避免使用已经失效的旧导入路径。"""
    agent_path = (
        PROJECT_ROOT / ".claude" / "skills" / "ask-question" / "script" / "agent.py"
    )
    spec = importlib.util.spec_from_file_location("ask_question_skill_agent", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 RAG Agent: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RAGKnowledgeAgent


RAGKnowledgeAgent = load_rag_agent_class()


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    """返回固定内容并记录 Prompt 的离线模型。"""

    def __init__(self, answer="北京住宿标准为每晚 500 元。"):
        self.answer = answer
        self.calls = []

    async def __call__(self, messages):
        self.calls.append(messages)
        return FakeResponse(self.answer)


class FakeSkillLoader:
    def get_skill_content(self, skill_name):
        return f"请执行 {skill_name} Skill 的问答要求。"


class FakeVector:
    def tolist(self):
        return [0.1, 0.2, 0.3]


class FakeEmbeddingModel:
    def __init__(self):
        self.queries = []

    def encode(self, query):
        self.queries.append(query)
        return FakeVector()


class FakeMilvusClient:
    def __init__(self):
        self.search_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [[
            {
                "distance": 0.92,
                "entity": {
                    "id": 7,
                    "content": "一线城市住宿标准为每晚 500 元。",
                    "metadata": json.dumps(
                        {"title": "住宿标准", "category": "travel_policy"},
                        ensure_ascii=False,
                    ),
                },
            }
        ]]

    def close(self):
        pass


def make_agent(model=None):
    """跳过重量级构造过程，只设置 reply() 测试所需状态。"""
    agent = object.__new__(RAGKnowledgeAgent)
    agent.name = "RAGKnowledgeAgent"
    agent.model = model
    agent.initialized = True
    agent.top_k = 3
    agent.similarity_threshold = 0.55
    agent.candidate_multiplier = 3
    agent.dedupe_similarity = 0.92
    agent.skill_loader = FakeSkillLoader()
    return agent


class TestRAGKnowledgeAgent(unittest.IsolatedAsyncioTestCase):
    async def test_structured_query_uses_retrieved_knowledge_and_returns_source(self):
        model = FakeModel()
        agent = make_agent(model)
        documents = [
            {
                "id": 1,
                "content": "北京属于一线城市，住宿标准为每晚 500 元。",
                "metadata": {"title": "住宿标准", "category": "travel_policy"},
                "distance": 0.95,
            }
        ]
        agent.search_knowledge = Mock(return_value=documents)
        message = Msg(
            name="OrchestrationAgent",
            content=json.dumps(
                {"context": {"rewritten_query": "北京出差住宿标准是多少？"}},
                ensure_ascii=False,
            ),
            role="user",
        )

        response = await agent.reply(message)
        result = json.loads(response.content)

        agent.search_knowledge.assert_called_once_with("北京出差住宿标准是多少？")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "北京住宿标准为每晚 500 元。")
        self.assertEqual(
            result["retrieved_documents"][0]["metadata"]["title"],
            "住宿标准",
        )
        self.assertEqual(result["sources"][0]["title"], "住宿标准")
        self.assertEqual(result["sources"][0]["score"], 0.95)
        prompt = model.calls[0][1]["content"]
        self.assertIn("北京出差住宿标准是多少？", prompt)
        self.assertIn("每晚 500 元", prompt)

    async def test_no_retrieved_documents_returns_no_knowledge(self):
        agent = make_agent(FakeModel())
        agent.search_knowledge = Mock(return_value=[])

        response = await agent.reply(
            Msg(name="User", content="火星出差住宿标准是什么？", role="user")
        )
        result = json.loads(response.content)

        self.assertEqual(result["status"], "no_knowledge")
        self.assertEqual(result["retrieved_documents"], [])
        self.assertIn("没有找到", result["answer"])

    async def test_without_llm_returns_retrieved_content_directly(self):
        agent = make_agent(model=None)
        agent.search_knowledge = Mock(
            return_value=[
                {
                    "id": 2,
                    "content": "报销应当在出差结束后 30 天内提交。",
                    "metadata": {"title": "报销规定"},
                    "distance": 0.9,
                }
            ]
        )

        response = await agent.reply(
            Msg(name="User", content="出差后多久报销？", role="user")
        )
        result = json.loads(response.content)

        self.assertEqual(result["status"], "success")
        self.assertIn("30 天", result["answer"])

    async def test_search_formats_milvus_result_and_respects_top_k(self):
        agent = make_agent()
        agent.collection_name = "business_travel_knowledge"
        agent.embedding_model = FakeEmbeddingModel()
        agent.milvus_client = FakeMilvusClient()
        agent._ensure_connection = Mock()

        documents = agent.search_knowledge("杭州住宿标准", top_k=1)

        self.assertEqual(agent.embedding_model.queries, ["杭州住宿标准"])
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["id"], 7)
        self.assertEqual(documents[0]["metadata"]["title"], "住宿标准")
        self.assertEqual(documents[0]["score"], 0.92)
        self.assertEqual(documents[0]["distance"], 0.92)
        self.assertEqual(agent.milvus_client.search_calls[0]["limit"], 3)

    async def test_search_filters_low_scores_and_near_duplicate_chunks(self):
        class FilteringMilvusClient(FakeMilvusClient):
            def search(self, **kwargs):
                self.search_calls.append(kwargs)
                return [[
                    {
                        "distance": 0.93,
                        "entity": {
                            "id": 1,
                            "content": "北京住宿标准为每晚500元。",
                            "metadata": {"title": "住宿标准 Part 1"},
                        },
                    },
                    {
                        "distance": 0.92,
                        "entity": {
                            "id": 2,
                            "content": "北京住宿标准为每晚500元。",
                            "metadata": {"title": "住宿标准重复片段"},
                        },
                    },
                    {
                        "distance": 0.84,
                        "entity": {
                            "id": 3,
                            "content": "杭州住宿标准为每晚400元。",
                            "metadata": {"title": "住宿标准 Part 2"},
                        },
                    },
                    {
                        "distance": 0.30,
                        "entity": {
                            "id": 4,
                            "content": "与问题无关的低分片段。",
                            "metadata": {"title": "无关内容"},
                        },
                    },
                ]]

        agent = make_agent()
        agent.collection_name = "business_travel_knowledge"
        agent.embedding_model = FakeEmbeddingModel()
        agent.milvus_client = FilteringMilvusClient()
        agent._ensure_connection = Mock()

        documents = agent.search_knowledge("住宿标准", top_k=3)

        self.assertEqual([doc["id"] for doc in documents], [1, 3])
        self.assertTrue(all(doc["score"] >= 0.55 for doc in documents))
        self.assertEqual(agent.milvus_client.search_calls[0]["limit"], 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
