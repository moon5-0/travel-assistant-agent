"""单轮业务执行服务的离线测试。"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from agentscope.message import Msg

from services.turn_executor import (
    AgentTurnExecutor,
    InvalidIntentionResultError,
)
from cli import AligoCLI


class FakeShortTermMemory:
    def __init__(self) -> None:
        self.messages = [
            {"role": "user", "content": "上一轮问题"},
            {"role": "assistant", "content": "上一轮回答"},
        ]

    def get_recent_context(self, n_turns: int = 5):
        return self.messages[-n_turns * 2 :]


class FakeLongTermMemory:
    def get_preference(self):
        return {"hotel_brands": ["汉庭"]}

    def get_trip_history(self, limit=None):
        return []


class FakeMemoryManager:
    def __init__(self) -> None:
        self.short_term = FakeShortTermMemory()
        self.long_term = FakeLongTermMemory()
        self.added_messages = []

    def get_previous_session_summaries(self, limit: int = 3):
        return [
            {"session_id": "old", "summary": "用户经常从苏州出发"}
        ]

    def add_message(self, role: str, content: str):
        self.added_messages.append((role, content))


class FakeAgent:
    def __init__(self, response_content: str) -> None:
        self.response_content = response_content
        self.received = None
        self.response = None

    async def reply(self, message):
        self.received = message
        self.response = Msg(
            name="FakeAgent",
            content=self.response_content,
            role="assistant",
        )
        return self.response


class FakeCircuitBreaker:
    def __init__(self) -> None:
        self.open_checks = 0
        self.successes = 0
        self.failures = 0

    def raise_if_open(self):
        self.open_checks += 1

    def record_success(self):
        self.successes += 1

    def record_failure(self):
        self.failures += 1


class TestAgentTurnExecutor(unittest.IsolatedAsyncioTestCase):
    async def test_pending_trip_continuation_restores_itinerary_schedule(self):
        intention_data = {
            "intents": [{"type": "event_collection", "confidence": 0.95}],
            "key_entities": {"origin": "苏州"},
            "rewritten_query": "从苏州出发",
            "agent_schedule": [
                {"agent_name": "event_collection", "priority": 1},
            ],
        }

        class PendingTripOrchestrator(FakeAgent):
            def get_pending_trip(self):
                return {"destination": "北京"}

        intention_agent = FakeAgent(json.dumps(intention_data))
        orchestrator = PendingTripOrchestrator(json.dumps({
            "status": "needs_clarification",
            "missing_fields": ["start_date", "duration_days"],
            "results": [],
        }))
        executor = AgentTurnExecutor(
            intention_agent=intention_agent,
            orchestrator=orchestrator,
            memory_manager=FakeMemoryManager(),
            resilience_config={"max_retries": 0},
        )

        result = await executor.execute_turn("从苏州出发")

        self.assertEqual(
            [
                item["agent_name"]
                for item in result["intention"]["agent_schedule"]
            ],
            ["event_collection", "itinerary_planning"],
        )
        self.assertEqual(
            json.loads(orchestrator.received.content)["agent_schedule"],
            result["intention"]["agent_schedule"],
        )

    async def test_execute_turn_returns_structured_result_and_updates_chat(self):
        intention_data = {
            "intents": ["规划行程"],
            "key_entities": {"destination": "北京"},
            "agent_schedule": [
                {"agent_name": "event_collection", "priority": 1}
            ],
        }
        orchestration_data = {
            "status": "needs_clarification",
            "missing_fields": ["origin", "start_date", "duration_days"],
            "results": [],
        }
        intention_agent = FakeAgent(json.dumps(intention_data))
        orchestrator = FakeAgent(json.dumps(orchestration_data))
        memory = FakeMemoryManager()
        circuit = FakeCircuitBreaker()
        executor = AgentTurnExecutor(
            intention_agent=intention_agent,
            orchestrator=orchestrator,
            memory_manager=memory,
            circuit_breaker=circuit,
            resilience_config={"max_retries": 0},
        )

        result = await executor.execute_turn("帮我规划去北京")

        self.assertEqual(result["intention"], intention_data)
        self.assertEqual(result["orchestration"], orchestration_data)
        self.assertEqual(circuit.open_checks, 1)
        self.assertEqual(circuit.successes, 1)
        self.assertEqual(circuit.failures, 0)
        self.assertEqual(memory.added_messages[0], ("user", "帮我规划去北京"))
        self.assertEqual(memory.added_messages[1][0], "assistant")
        self.assertIn("needs_clarification", memory.added_messages[1][1])

        context_messages = intention_agent.received
        self.assertEqual(context_messages[-1].content, "帮我规划去北京")
        self.assertEqual(context_messages[-1].role, "user")
        self.assertEqual(context_messages[0].role, "system")
        self.assertIn("hotel_brands", context_messages[0].content)
        self.assertIn("用户经常从苏州出发", context_messages[0].content)
        self.assertIs(orchestrator.received, intention_agent.response)
        self.assertEqual(
            orchestrator.received.metadata["original_user_input"],
            "帮我规划去北京",
        )
        self.assertEqual(
            json.loads(orchestrator.received.content),
            intention_data,
        )

    async def test_invalid_intention_json_does_not_write_chat_memory(self):
        memory = FakeMemoryManager()
        executor = AgentTurnExecutor(
            intention_agent=FakeAgent("not-json"),
            orchestrator=FakeAgent("{}"),
            memory_manager=memory,
            resilience_config={"max_retries": 0},
        )

        with self.assertRaises(InvalidIntentionResultError):
            await executor.execute_turn("无法解析的问题")

        self.assertEqual(memory.added_messages, [])

    async def test_orchestration_failure_records_circuit_failure(self):
        class FailingOrchestrator:
            async def reply(self, message):
                raise RuntimeError("orchestration failed")

        memory = FakeMemoryManager()
        circuit = FakeCircuitBreaker()
        executor = AgentTurnExecutor(
            intention_agent=FakeAgent(
                json.dumps({"agent_schedule": []})
            ),
            orchestrator=FailingOrchestrator(),
            memory_manager=memory,
            circuit_breaker=circuit,
            resilience_config={"max_retries": 0},
        )

        with self.assertRaisesRegex(RuntimeError, "orchestration failed"):
            await executor.execute_turn("测试调度失败")

        self.assertEqual(circuit.failures, 1)
        self.assertEqual(circuit.successes, 0)
        self.assertEqual(memory.added_messages, [("user", "测试调度失败")])


class TestCLIExecutionBoundary(unittest.IsolatedAsyncioTestCase):
    async def test_process_query_delegates_business_work_to_turn_executor(self):
        cli = AligoCLI()
        turn_result = {
            "user_input": "帮我规划去北京",
            "intention": {"agent_schedule": []},
            "orchestration": {"status": "completed", "results": []},
        }
        cli.turn_executor = MagicMock()
        cli.turn_executor.execute_turn = AsyncMock(return_value=turn_result)
        cli._display_agents_called = MagicMock()
        cli._display_results = MagicMock()

        result = await cli.process_query("帮我规划去北京")

        cli.turn_executor.execute_turn.assert_awaited_once_with("帮我规划去北京")
        cli._display_agents_called.assert_called_once_with(
            turn_result["orchestration"]
        )
        cli._display_results.assert_called_once_with(
            turn_result["orchestration"]
        )
        self.assertEqual(result, turn_result)


if __name__ == "__main__":
    unittest.main()
