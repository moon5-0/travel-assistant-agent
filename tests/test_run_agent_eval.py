"""真实 Agent 评估入口中隔离环境初始化逻辑的离线测试。"""

from tempfile import TemporaryDirectory
import unittest

from context.memory_manager import MemoryManager
from evaluation.run_agent_eval import (
    build_execution_summary,
    seed_initial_state,
    select_cases,
)


class FakeOrchestrator:
    def __init__(self) -> None:
        self.pending_trip = None

    def restore_pending_trip(self, trip_data):
        self.pending_trip = dict(trip_data)


class TestRealAgentEvaluationSetup(unittest.TestCase):
    def test_seed_initial_state_uses_isolated_memory(self):
        with TemporaryDirectory() as storage_path:
            memory = MemoryManager(
                user_id="eval-user",
                session_id="eval-session",
                storage_path=storage_path,
            )
            orchestrator = FakeOrchestrator()

            seed_initial_state(
                memory,
                orchestrator,
                {
                    "preferences": {"hotel_brands": ["汉庭"]},
                    "trip_history": [
                        {
                            "origin": "苏州",
                            "destination": "北京",
                            "start_date": "2026-08-10",
                        }
                    ],
                    "pending_trip": {"destination": "杭州"},
                },
            )

            self.assertEqual(
                memory.long_term.get_preference(),
                {"hotel_brands": ["汉庭"]},
            )
            self.assertEqual(
                memory.long_term.get_trip_history()[0]["destination"],
                "北京",
            )
            self.assertEqual(
                orchestrator.pending_trip,
                {"destination": "杭州"},
            )

    def test_select_cases_rejects_unknown_id(self):
        dataset = {
            "dataset_version": "test",
            "cases": [{"id": "known", "turns": []}],
        }

        with self.assertRaisesRegex(ValueError, "unknown"):
            select_cases(dataset, ["unknown"])

    def test_execution_summary_counts_turns_and_repeats(self):
        dataset = {"dataset_version": "test"}
        cases = [
            {"id": "one", "turns": [{}, {}]},
            {"id": "two", "turns": [{}]},
        ]

        summary = build_execution_summary(dataset, cases, runs=3)

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["turns_per_run"], 3)
        self.assertEqual(summary["total_turns"], 9)


if __name__ == "__main__":
    unittest.main()
