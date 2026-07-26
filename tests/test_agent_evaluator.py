#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent 评估数据集加载与校验测试。"""

from __future__ import annotations

import copy
import unittest

from evaluation.agent_evaluator import (
    DatasetValidationError,
    evaluate_case,
    evaluate_turn,
    load_dataset,
    summarize_dataset,
    validate_dataset,
)


class TestAgentEvaluationDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset()

    def test_project_dataset_is_valid_and_has_expected_size(self):
        summary = summarize_dataset(self.dataset)

        self.assertEqual(summary["dataset_version"], "0.1.0")
        self.assertEqual(summary["case_count"], 5)
        self.assertEqual(summary["turn_count"], 7)
        self.assertEqual(summary["severities"], {"critical": 5})

    def test_duplicate_case_id_is_rejected(self):
        invalid = copy.deepcopy(self.dataset)
        invalid["cases"][1]["id"] = invalid["cases"][0]["id"]

        with self.assertRaisesRegex(
            DatasetValidationError,
            "duplicate case id",
        ):
            validate_dataset(invalid)

    def test_invalid_response_pattern_is_rejected(self):
        invalid = copy.deepcopy(self.dataset)
        expected = invalid["cases"][0]["turns"][0]["expected"]
        expected["must_not_contain_patterns"] = ["["]

        with self.assertRaisesRegex(
            DatasetValidationError,
            "must_not_contain_patterns",
        ):
            validate_dataset(invalid)

    def test_agent_cannot_be_required_and_forbidden(self):
        invalid = copy.deepcopy(self.dataset)
        expected = invalid["cases"][0]["turns"][0]["expected"]
        expected["forbidden_executed_agents"].append(
            "itinerary_planning"
        )

        with self.assertRaisesRegex(
            DatasetValidationError,
            "both required and forbidden",
        ):
            validate_dataset(invalid)

    def test_correct_clarification_trace_passes(self):
        case = next(
            item
            for item in self.dataset["cases"]
            if item["id"] == "trip_missing_required_fields"
        )
        actual = {
            "scheduled_agents": [
                "event_collection",
                "itinerary_planning",
            ],
            "executed_agents": ["event_collection"],
            "status": "needs_clarification",
            "entities": {"destination": "北京"},
            "missing_fields": [
                "origin",
                "start_date",
                "duration_days",
            ],
            "memory_before": {
                "preferences": {},
                "trip_history": [],
            },
            "memory_after": {
                "preferences": {},
                "trip_history": [],
            },
            "pending_trip": {"destination": "北京"},
            "response": "请补充出发地、出发日期和行程天数。",
        }

        result = evaluate_case(case, [actual])

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])

    def test_forbidden_agent_and_unverified_train_are_reported(self):
        case = next(
            item
            for item in self.dataset["cases"]
            if item["id"] == "trip_missing_required_fields"
        )
        expected = case["turns"][0]["expected"]
        actual = {
            "scheduled_agents": [
                "event_collection",
                "itinerary_planning",
                "preference",
            ],
            "executed_agents": [
                "event_collection",
                "itinerary_planning",
                "preference",
            ],
            "status": "completed",
            "entities": {"destination": "北京"},
            "missing_fields": [],
            "memory_before": {
                "preferences": {},
                "trip_history": [],
            },
            "memory_after": {
                "preferences": {"hotel_brands": ["汉庭"]},
                "trip_history": [{"destination": "北京"}],
            },
            "response": "已生成完整行程，建议乘坐G123次。",
        }

        result = evaluate_turn(expected, actual)

        self.assertFalse(result["passed"])
        self.assertIn("forbidden_executed_agents", result["failures"])
        self.assertIn("status", result["failures"])
        self.assertIn(
            "response.must_not_contain_patterns",
            result["failures"],
        )
        self.assertIn("memory.preferences_change", result["failures"])
        self.assertIn("memory.trip_history_change", result["failures"])

    def test_expected_preference_append_passes(self):
        case = next(
            item
            for item in self.dataset["cases"]
            if item["id"] == "preference_add_hotel_brand"
        )
        actual = {
            "scheduled_agents": ["preference"],
            "executed_agents": ["preference"],
            "status": "completed",
            "entities": {},
            "missing_fields": [],
            "memory_before": {
                "preferences": {},
                "trip_history": [],
            },
            "memory_after": {
                "preferences": {"hotel_brands": ["汉庭"]},
                "trip_history": [],
            },
            "response": "已记录您喜欢汉庭酒店。",
        }

        result = evaluate_case(case, [actual])

        self.assertTrue(result["passed"])

    def test_wrong_turn_count_is_rejected(self):
        case = next(
            item
            for item in self.dataset["cases"]
            if item["id"] == "trip_multi_turn_completion"
        )

        result = evaluate_case(case, [])

        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"], ["turn_count"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
