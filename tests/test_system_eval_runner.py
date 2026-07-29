"""System Evaluation 场景运行器的离线测试。"""

from contextlib import asynccontextmanager
import unittest

from evaluation.system.system_eval_runner import SystemEvaluationRunner


def expected():
    return {
        "required_scheduled_agents": [],
        "forbidden_scheduled_agents": [],
        "required_executed_agents": [],
        "forbidden_executed_agents": [],
        "forbidden_entity_fields": [],
        "status": "completed",
        "entities": {},
        "missing_fields": [],
        "memory": {
            "preferences_change": "none",
            "trip_history_change": "none",
        },
    }


def trace(response: str):
    return {
        "scheduled_agents": [],
        "executed_agents": [],
        "status": "completed",
        "entities": {},
        "missing_fields": [],
        "memory_before": {"preferences": {}, "trip_history": []},
        "memory_after": {"preferences": {}, "trip_history": []},
        "pending_trip": {},
        "response": response,
    }


class StatefulCollector:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.turn_number = 0

    async def execute_turn(self, user_input: str):
        self.turn_number += 1
        return trace(f"{self.case_id}-turn-{self.turn_number}")


class TestSystemEvaluationRunner(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_runtime_within_case_and_isolates_different_cases(self):
        dataset = {
            "dataset_version": "test",
            "default_context": {"runs_per_case": 1},
            "cases": [
                {
                    "id": "multi_turn",
                    "severity": "critical",
                    "turns": [
                        {"user_input": "第一轮", "expected": expected()},
                        {"user_input": "第二轮", "expected": expected()},
                    ],
                },
                {
                    "id": "new_case",
                    "severity": "major",
                    "turns": [
                        {"user_input": "新场景", "expected": expected()},
                    ],
                },
            ],
        }
        collectors = []

        @asynccontextmanager
        async def runtime_factory(case, run_index):
            collector = StatefulCollector(case["id"])
            collectors.append(collector)
            yield collector

        report = await SystemEvaluationRunner(
            dataset,
            runtime_factory,
        ).run()

        self.assertEqual(len(collectors), 2)
        self.assertEqual(collectors[0].turn_number, 2)
        self.assertEqual(collectors[1].turn_number, 1)
        self.assertEqual(report["summary"]["total_runs"], 2)
        self.assertEqual(report["summary"]["passed_runs"], 2)
        self.assertEqual(report["summary"]["pass_rate"], 1.0)

    async def test_one_execution_error_does_not_stop_remaining_cases(self):
        dataset = {
            "dataset_version": "test",
            "default_context": {"runs_per_case": 1},
            "cases": [
                {
                    "id": "broken",
                    "severity": "critical",
                    "turns": [
                        {"user_input": "失败", "expected": expected()}
                    ],
                },
                {
                    "id": "healthy",
                    "severity": "critical",
                    "turns": [
                        {"user_input": "成功", "expected": expected()}
                    ],
                },
            ],
        }

        class BrokenCollector:
            async def execute_turn(self, user_input):
                raise RuntimeError("model unavailable")

        @asynccontextmanager
        async def runtime_factory(case, run_index):
            if case["id"] == "broken":
                yield BrokenCollector()
            else:
                yield StatefulCollector(case["id"])

        report = await SystemEvaluationRunner(
            dataset,
            runtime_factory,
        ).run()

        self.assertEqual(report["summary"]["total_runs"], 2)
        self.assertEqual(report["summary"]["passed_runs"], 1)
        self.assertEqual(
            report["runs"][0]["failures"],
            ["execution_error"],
        )
        self.assertTrue(report["runs"][1]["passed"])


if __name__ == "__main__":
    unittest.main()
