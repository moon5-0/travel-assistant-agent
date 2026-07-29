"""Itinerary Quality硬规则评估器的离线测试。"""

from __future__ import annotations

import copy
import json
import unittest

from evaluation.itinerary_quality.hard_rule_evaluator import (
    _assertive_pattern_matches,
    DatasetValidationError,
    evaluate_case,
    load_dataset,
    summarize_dataset,
    validate_dataset,
)


def valid_standard_output():
    return {
        "itinerary": {
            "title": "苏州到北京3天客户拜访行程",
            "duration": "3天",
            "route": "苏州 -> 北京 -> 苏州",
            "daily_plans": [
                {
                    "day": 1,
                    "date": "2026-08-10",
                    "city": "北京",
                    "activities": [
                        {
                            "time": "09:00-12:00",
                            "location": "客户公司",
                            "description": "客户拜访",
                            "transport": "地铁",
                        }
                    ],
                },
                {
                    "day": 2,
                    "date": "2026-08-11",
                    "city": "北京",
                    "activities": [
                        {
                            "time": "09:00-11:00",
                            "location": "客户公司",
                            "description": "商务沟通",
                            "transport": "地铁",
                        }
                    ],
                },
                {
                    "day": 3,
                    "date": "2026-08-12",
                    "city": "北京",
                    "activities": [
                        {
                            "time": "15:00-20:00",
                            "location": "北京南站",
                            "description": "返程返回苏州",
                            "transport": "高铁",
                        }
                    ],
                },
            ],
            "notes": ["具体车次请通过12306核实"],
        },
        "planning_complete": True,
    }


class TestItineraryQualityDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset()

    def test_project_dataset_is_valid_and_contains_ten_cases(self):
        summary = summarize_dataset(self.dataset)

        self.assertEqual(summary["version"], "0.1.0-draft")
        self.assertEqual(summary["case_count"], 10)

    def test_duplicate_case_id_is_rejected(self):
        invalid = copy.deepcopy(self.dataset)
        invalid["cases"][1]["id"] = invalid["cases"][0]["id"]

        with self.assertRaisesRegex(DatasetValidationError, "duplicate"):
            validate_dataset(invalid)

    def test_required_rule_must_choose_one_matching_mode(self):
        invalid = copy.deepcopy(self.dataset)
        rule = invalid["cases"][0]["expected"]["required_content"][0]
        rule["all_of"] = ["客户"]

        with self.assertRaisesRegex(DatasetValidationError, "exactly one"):
            validate_dataset(invalid)


class TestItineraryHardRuleEvaluator(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset()
        self.case = self.dataset["cases"][0]
        self.global_rules = self.dataset["global_rules"]

    def evaluate(self, output):
        return evaluate_case(self.case, output, self.global_rules)

    def test_valid_itinerary_passes_all_hard_rules(self):
        result = self.evaluate(valid_standard_output())

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["fatal_errors"], [])

    def test_wrong_destination_is_a_fatal_error(self):
        output = valid_standard_output()
        serialized = json.dumps(output, ensure_ascii=False).replace("北京", "上海")
        output = json.loads(serialized)

        result = self.evaluate(output)

        self.assertFalse(result["passed"])
        self.assertIn("required_trip.destination", result["failures"])
        self.assertIn("required_trip.destination", result["fatal_errors"])

    def test_wrong_day_count_and_dates_are_reported(self):
        output = valid_standard_output()
        output["itinerary"]["daily_plans"].pop()

        result = self.evaluate(output)

        self.assertIn("required_trip.duration_days", result["failures"])
        self.assertIn("required_trip.daily_plan_dates", result["failures"])

    def test_missing_required_content_is_reported(self):
        output = valid_standard_output()
        serialized = (
            json.dumps(output, ensure_ascii=False)
            .replace("客户", "内部")
            .replace("商务", "内部")
            .replace("拜访", "沟通")
        )
        output = json.loads(serialized)

        result = self.evaluate(output)

        self.assertIn("required_content.business_first", result["failures"])

    def test_unsupported_confirmation_is_fatal(self):
        output = valid_standard_output()
        output["itinerary"]["notes"].append("高铁余票充足")

        result = self.evaluate(output)

        self.assertIn("global.unsupported_confirmation", result["failures"])
        self.assertIn("global.unsupported_confirmation", result["fatal_errors"])

    def test_negated_confirmation_is_not_a_false_positive(self):
        output = valid_standard_output()
        output["itinerary"]["notes"].append(
            "当前无法确认余票充足，请通过12306核实"
        )

        result = self.evaluate(output)

        self.assertTrue(result["passed"])
        self.assertNotIn("global.unsupported_confirmation", result["failures"])

    def test_generic_unable_phrase_is_not_a_forbidden_recommendation(self):
        matches = _assertive_pattern_matches(
            "由于超过预算，无法入住上海浦东丽思卡尔顿",
            ["入住上海浦东丽思卡尔顿"],
        )

        self.assertEqual(matches, [])

    def test_invalid_json_string_fails_safely(self):
        result = self.evaluate("{invalid")

        self.assertFalse(result["passed"])
        self.assertEqual(result["fatal_errors"], ["output.valid_json"])


if __name__ == "__main__":
    unittest.main()
