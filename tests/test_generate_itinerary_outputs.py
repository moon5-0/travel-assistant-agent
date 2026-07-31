"""行程输出生成入口的离线测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from evaluation.itinerary_quality.hard_rule_evaluator import load_dataset
from evaluation.itinerary_quality.generate_itinerary_outputs import (
    build_agent_input,
    build_execution_summary,
    run_cases,
    select_cases,
)
from tests.test_itinerary_hard_rule_evaluator import valid_standard_output
from utils.planning_policy import determine_planning_mode


class FakeItineraryAgent:
    def __init__(self, output):
        self.output = output
        self.inputs = []

    async def reply(self, message):
        self.inputs.append(message)
        return SimpleNamespace(
            content=json.dumps(self.output, ensure_ascii=False)
        )


class TestItineraryQualityRunSetup(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dataset = load_dataset()
        self.case = self.dataset["cases"][0]

    def test_build_agent_input_matches_orchestrator_shape(self):
        message = build_agent_input(self.case)
        payload = json.loads(message.content)

        self.assertEqual(
            payload["context"]["rewritten_query"],
            self.case["input"]["user_query"],
        )
        self.assertEqual(
            payload["context"]["user_preferences"],
            self.case["input"]["user_preferences"],
        )
        self.assertEqual(
            payload["context"]["planning_signals"],
            self.case["input"]["planning_signals"],
        )
        event_result = payload["previous_results"][0]
        self.assertEqual(event_result["agent_name"], "event_collection")
        self.assertEqual(
            event_result["result"]["data"],
            self.case["input"]["trip_info"],
        )

    def test_unknown_case_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            select_cases(self.dataset, ["unknown"])

    def test_dataset_signals_produce_expected_planning_modes(self):
        expected_modes = {
            "standard_three_day_business_trip":
                "business_with_optional_leisure",
            "same_day_round_trip": "business_only",
        }

        for case in self.dataset["cases"]:
            payload = json.loads(build_agent_input(case).content)
            mode = determine_planning_mode(
                payload["context"]["rewritten_query"],
                {
                    "context": payload["context"],
                    "event_collection": case["input"]["trip_info"],
                },
            )
            self.assertEqual(
                mode,
                expected_modes.get(case["id"], "business_first"),
                case["id"],
            )

    def test_execution_summary_counts_model_calls(self):
        summary = build_execution_summary(
            self.dataset,
            self.dataset["cases"][:2],
            runs=3,
        )

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["total_agent_calls"], 6)
        self.assertEqual(summary["evaluation_stage"], "hard_rules_only")

    async def test_run_cases_connects_agent_output_to_hard_rules(self):
        agent = FakeItineraryAgent(valid_standard_output())

        report = await run_cases(
            agent,
            self.dataset,
            [self.case],
            runs_per_case=1,
        )

        self.assertEqual(len(agent.inputs), 1)
        self.assertEqual(report["summary"]["total_runs"], 1)
        self.assertEqual(report["summary"]["evaluated_runs"], 1)
        self.assertEqual(report["summary"]["hard_constraint_pass_rate"], 1.0)
        self.assertEqual(report["summary"]["fatal_error_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
