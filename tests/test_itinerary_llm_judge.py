"""Itinerary Quality LLM Judge的离线测试。"""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from evaluation.itinerary_quality.hard_rule_evaluator import load_dataset
from evaluation.itinerary_quality.llm_judge import (
    LLMItineraryJudge,
    build_evidence_catalog,
    build_judge_messages,
    score_judge_output,
    validate_evidence_grounding,
)
from evaluation.itinerary_quality.run_itinerary_llm_judge import run_judging
from tests.test_itinerary_hard_rule_evaluator import valid_standard_output


def valid_judge_output():
    catalog = build_evidence_catalog(valid_standard_output())

    def evidence_id(text):
        return next(
            key
            for key, item in catalog.items()
            if item["text"] == text
        )

    return {
        "time_route_feasibility": {
            "score": 4,
            "reason": "时间顺序合理，交通有基本缓冲。",
            "evidence": [evidence_id("09:00-12:00")],
        },
        "business_personalization": {
            "score": 4,
            "reason": "以客户拜访为主。",
            "evidence": [
                evidence_id("客户拜访"),
                evidence_id("商务沟通"),
            ],
        },
        "completeness_usability": {
            "score": 3,
            "reason": "每日安排完整，但住宿信息较少。",
            "evidence": [evidence_id("返程返回苏州")],
        },
        "factual_groundedness": {
            "score": 4,
            "reason": "提示用户核实车次。",
            "evidence": [evidence_id("具体车次请通过12306核实")],
        },
        "semantic_fatal_errors": [],
        "overall_summary": "整体可执行，存在少量可完善项。",
    }


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return SimpleNamespace(content=self.responses.pop(0))


class TestJudgeScoring(unittest.TestCase):
    def test_weighted_score_is_computed_by_code(self):
        result = score_judge_output(valid_judge_output())

        self.assertEqual(result["weighted_quality_score"], 76.0)
        self.assertEqual(result["min_dimension_score"], 3)
        self.assertTrue(result["judge_passed"])

    def test_one_low_dimension_cannot_be_hidden_by_high_total(self):
        output = valid_judge_output()
        output["completeness_usability"]["score"] = 2
        output["time_route_feasibility"]["score"] = 5
        output["business_personalization"]["score"] = 5
        output["factual_groundedness"]["score"] = 5

        result = score_judge_output(output)

        self.assertGreater(result["weighted_quality_score"], 70)
        self.assertFalse(result["judge_passed"])

    def test_semantic_fatal_error_forces_failure(self):
        output = valid_judge_output()
        output["semantic_fatal_errors"] = [{
            "category": "meeting_conflict",
            "description": "固定会议与跨城交通冲突。",
            "evidence": "E001",
        }]

        result = score_judge_output(output)

        self.assertFalse(result["judge_passed"])
        self.assertEqual(len(result["semantic_fatal_errors"]), 1)

    def test_prompt_contains_task_context_and_not_only_itinerary(self):
        dataset = load_dataset()
        case = dataset["cases"][0]

        messages = build_judge_messages(case, valid_standard_output())
        prompt = messages[1]["content"]

        self.assertIn(case["input"]["user_query"], prompt)
        self.assertIn("trip_info", prompt)
        self.assertIn("user_preferences", prompt)
        self.assertIn("itinerary_output", prompt)

    def test_ungrounded_evidence_is_rejected(self):
        result = score_judge_output(valid_judge_output())
        result["dimensions"]["time_route_feasibility"]["evidence"] = [
            "E999"
        ]

        with self.assertRaisesRegex(ValueError, "evidence catalog"):
            validate_evidence_grounding(result, valid_standard_output())


class TestLLMItineraryJudge(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dataset = load_dataset()
        self.case = self.dataset["cases"][0]

    async def test_valid_output_uses_one_json_mode_call(self):
        model = FakeModel([
            json.dumps(valid_judge_output(), ensure_ascii=False)
        ])
        judge = LLMItineraryJudge(model)

        result = await judge.evaluate(self.case, valid_standard_output())

        self.assertEqual(result["weighted_quality_score"], 76.0)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            model.calls[0]["kwargs"]["response_format"],
            {"type": "json_object"},
        )

    async def test_invalid_structure_is_repaired_once(self):
        invalid = valid_judge_output()
        invalid.pop("overall_summary")
        model = FakeModel([
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(valid_judge_output(), ensure_ascii=False),
        ])
        judge = LLMItineraryJudge(model)

        result = await judge.evaluate(self.case, valid_standard_output())

        self.assertTrue(result["judge_passed"])
        self.assertEqual(len(model.calls), 2)

    async def test_ungrounded_evidence_is_repaired_once(self):
        invalid = valid_judge_output()
        invalid["time_route_feasibility"]["evidence"] = [
            "E999"
        ]
        model = FakeModel([
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(valid_judge_output(), ensure_ascii=False),
        ])
        judge = LLMItineraryJudge(model)

        result = await judge.evaluate(self.case, valid_standard_output())

        self.assertTrue(result["judge_passed"])
        self.assertEqual(len(model.calls), 2)


class FakeJudge:
    async def evaluate(self, case, itinerary_output):
        return score_judge_output(valid_judge_output())


class TestJudgeRunner(unittest.IsolatedAsyncioTestCase):
    async def test_existing_output_is_reused_and_combined(self):
        dataset = load_dataset()
        source_report = {
            "generated_at": "2026-07-29T00:00:00+00:00",
            "runs": [{
                "case_id": dataset["cases"][0]["id"],
                "run_index": 1,
                "output": json.dumps(
                    valid_standard_output(),
                    ensure_ascii=False,
                ),
            }],
        }

        report = await run_judging(
            FakeJudge(),
            dataset,
            source_report,
            source_report["runs"],
        )

        self.assertEqual(report["summary"]["judged_runs"], 1)
        self.assertEqual(
            report["summary"]["average_itinerary_quality_score"],
            76.0,
        )
        self.assertEqual(report["summary"]["hard_constraint_pass_rate"], 1.0)
        self.assertEqual(report["summary"]["fatal_error_rate"], 0.0)
        self.assertEqual(report["summary"]["qualified_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
