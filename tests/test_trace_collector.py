"""真实执行轨迹采集器的离线测试。"""

import unittest

from evaluation.trace_collector import AgentTraceCollector


class FakeLongTermMemory:
    def __init__(self, preferences=None, trips=None) -> None:
        self.preferences = preferences or {}
        self.trips = trips or []

    def get_preference(self):
        return self.preferences

    def get_trip_history(self, limit=None):
        return self.trips


class FakeMemoryManager:
    def __init__(self, preferences=None, trips=None) -> None:
        self.long_term = FakeLongTermMemory(preferences, trips)


class FakeOrchestrator:
    def __init__(self) -> None:
        self.pending_trip = {}

    def get_pending_trip(self):
        return self.pending_trip


class FakeTurnExecutor:
    def __init__(self, result, preferences=None, trips=None) -> None:
        self.result = result
        self.memory_manager = FakeMemoryManager(preferences, trips)
        self.orchestrator = FakeOrchestrator()

    async def execute_turn(self, user_input):
        self.memory_manager.long_term.preferences.setdefault(
            "hotel_brands",
            [],
        ).append("汉庭")
        self.orchestrator.pending_trip = {"destination": "北京"}
        return self.result


class TestAgentTraceCollector(unittest.IsolatedAsyncioTestCase):
    async def test_collects_route_entities_memory_and_pending_trip(self):
        result = {
            "intention": {
                "key_entities": {"destination": "北京"},
                "agent_schedule": [
                    {"agent_name": "event_collection", "priority": 1},
                    {"agent_name": "itinerary_planning", "priority": 2},
                ],
            },
            "orchestration": {
                "status": "needs_clarification",
                "missing_fields": ["start_date", "duration_days"],
                "message": "请补充出发日期、行程天数",
                "results": [
                    {
                        "agent_name": "event_collection",
                        "status": "success",
                        "data": {
                            "origin": "苏州",
                            "destination": "北京",
                        },
                    }
                ],
            },
        }
        executor = FakeTurnExecutor(result, preferences={}, trips=[])
        collector = AgentTraceCollector(executor)

        trace = await collector.execute_turn("从苏州去北京")

        self.assertEqual(
            trace["scheduled_agents"],
            ["event_collection", "itinerary_planning"],
        )
        self.assertEqual(trace["executed_agents"], ["event_collection"])
        self.assertEqual(
            trace["entities"],
            {"origin": "苏州", "destination": "北京"},
        )
        self.assertEqual(trace["memory_before"]["preferences"], {})
        self.assertEqual(
            trace["memory_after"]["preferences"],
            {"hotel_brands": ["汉庭"]},
        )
        self.assertEqual(trace["pending_trip"], {"destination": "北京"})
        self.assertIn("出发日期", trace["response"])
        self.assertEqual(trace["history_usage"], "not_used")

    def test_detects_unconfirmed_history_auto_fill(self):
        usage = AgentTraceCollector._detect_history_usage(
            user_input="按照我之前的酒店偏好规划北京行程",
            entities={
                "origin": "苏州",
                "destination": "北京",
                "start_date": "2026-07-25",
            },
            status="completed",
            trip_history=[
                {
                    "origin": "苏州",
                    "destination": "北京",
                    "start_date": "2026-07-25",
                }
            ],
            response="已生成行程",
        )

        self.assertEqual(usage, "auto_filled")

    def test_detects_history_confirmation_requirement(self):
        usage = AgentTraceCollector._detect_history_usage(
            user_input="按照我之前的酒店偏好规划北京行程",
            entities={"destination": "北京"},
            status="needs_clarification",
            trip_history=[
                {
                    "origin": "苏州",
                    "destination": "北京",
                    "start_date": "2026-07-25",
                }
            ],
            response="请补充出发地、出发日期和行程天数",
        )

        self.assertEqual(usage, "confirmation_required")


if __name__ == "__main__":
    unittest.main()
