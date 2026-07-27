#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""System Evaluation 数据集加载与评分测试。"""

from __future__ import annotations

import copy
import unittest

from evaluation.system_evaluator import (
    DatasetValidationError,
    evaluate_case,
    evaluate_turn,
    load_dataset,
    summarize_dataset,
    validate_dataset,
)


class TestSystemEvaluationDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset()

    def test_project_dataset_is_valid_and_has_expected_size(self):
        summary = summarize_dataset(self.dataset)

        self.assertEqual(summary["dataset_version"], "0.3.1")
        self.assertEqual(summary["case_count"], 15)
        self.assertEqual(summary["turn_count"], 19)
        self.assertEqual(
            summary["severities"],
            {"critical": 10, "major": 5},
        )

    def test_duplicate_case_id_is_rejected(self):
        invalid = copy.deepcopy(self.dataset)
        invalid["cases"][1]["id"] = invalid["cases"][0]["id"]

        with self.assertRaisesRegex(
            DatasetValidationError,
            "duplicate case id",
        ):
            validate_dataset(invalid)

    def test_dataset_has_no_result_wording_rules(self):
        for case in self.dataset["cases"]:
            for turn in case["turns"]:
                expected = turn["expected"]
                self.assertNotIn("must_contain", expected)
                self.assertNotIn("must_not_contain_patterns", expected)

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

    def test_scheduled_agent_cannot_be_required_and_forbidden(self):
        invalid = copy.deepcopy(self.dataset)
        expected = invalid["cases"][0]["turns"][0]["expected"]
        expected["forbidden_scheduled_agents"].append(
            "itinerary_planning"
        )

        with self.assertRaisesRegex(
            DatasetValidationError,
            "both required and forbidden in schedule",
        ):
            validate_dataset(invalid)

    def test_entity_field_cannot_be_required_and_forbidden(self):
        invalid = copy.deepcopy(self.dataset)
        expected = invalid["cases"][0]["turns"][0]["expected"]
        expected["forbidden_entity_fields"].append("origin")

        with self.assertRaisesRegex(
            DatasetValidationError,
            "entity fields as both required and forbidden",
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

    def test_system_failures_are_reported_but_response_wording_is_ignored(self):
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
        self.assertIn("forbidden_scheduled_agents", result["failures"])
        self.assertIn("forbidden_executed_agents", result["failures"])
        self.assertIn("status", result["failures"])
        self.assertIn("memory.preferences_change", result["failures"])
        self.assertIn("memory.trip_history_change", result["failures"])
        self.assertFalse(any(
            failure.startswith("response.")
            for failure in result["failures"]
        ))

    def test_auto_filled_history_entities_are_reported(self):
        case = next(
            item
            for item in self.dataset["cases"]
            if item["id"]
            == "missing_hotel_preference_must_not_reuse_trip_history"
        )
        actual = {
            "scheduled_agents": [
                "memory_query",
                "event_collection",
                "itinerary_planning",
            ],
            "executed_agents": ["memory_query", "event_collection"],
            "status": "needs_clarification",
            "entities": {
                "origin": "苏州（从历史行程推断，但不一定准确）",
                "destination": "北京",
                "trip_purpose": "旅游",
            },
            "missing_fields": [
                "origin",
                "start_date",
                "duration_days",
            ],
            "memory_before": {
                "preferences": {},
                "trip_history": [{"destination": "北京"}],
            },
            "memory_after": {
                "preferences": {},
                "trip_history": [{"destination": "北京"}],
            },
            "pending_trip": {
                "destination": "北京",
                "trip_purpose": "旅游",
            },
            "history_usage": "confirmation_required",
        }

        result = evaluate_case(case, [actual])

        self.assertFalse(result["passed"])
        self.assertIn(
            "turn[0].forbidden_entity_fields",
            result["failures"],
        )

    def test_absence_placeholders_are_not_treated_as_auto_filled_entities(self):
        case = next(
            item
            for item in self.dataset["cases"]
            if item["id"] == "trip_missing_required_fields"
        )
        placeholders = [
            None,
            "",
            "未提供",
            "未指定",
            "未知",
            "待补充",
            "未指定，待收集",
            "未提供（需要收集）",
            "未指定（待确认）",
            "未提供，需要收集",
            "未明确",
            "尚未明确",
            "暂未明确",
            "待明确",
            "未确定",
            "待确定",
            "未明确（需要用户补充）",
            "N/A",
            "not provided",
        ]

        for placeholder in placeholders:
            with self.subTest(placeholder=placeholder):
                actual = {
                    "scheduled_agents": [
                        "event_collection",
                        "itinerary_planning",
                    ],
                    "executed_agents": ["event_collection"],
                    "status": "needs_clarification",
                    "entities": {
                        "origin": placeholder,
                        "destination": "北京",
                        "start_date": placeholder,
                        "duration_days": placeholder,
                    },
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
                }

                result = evaluate_case(case, [actual])

                self.assertTrue(result["passed"], result["failures"])

    def test_parenthetical_note_does_not_hide_a_real_inferred_entity(self):
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
            "entities": {
                "origin": "苏州（从历史推断）",
                "destination": "北京",
            },
            "missing_fields": [
                "origin",
                "start_date",
                "duration_days",
            ],
            "memory_before": {"preferences": {}, "trip_history": []},
            "memory_after": {"preferences": {}, "trip_history": []},
            "pending_trip": {"destination": "北京"},
        }

        result = evaluate_case(case, [actual])

        self.assertFalse(result["passed"])
        self.assertIn(
            "turn[0].forbidden_entity_fields",
            result["failures"],
        )

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
                "preferences": {"hotel_brands": ["如家"]},
                "trip_history": [],
            },
            "memory_after": {
                "preferences": {"hotel_brands": ["如家", "汉庭"]},
                "trip_history": [],
            },
            "response": "已记录您喜欢汉庭酒店。",
        }

        result = evaluate_case(case, [actual])

        self.assertTrue(result["passed"])

    def test_pending_trip_allows_extra_optional_fields(self):
        expected = {
            "required_scheduled_agents": [],
            "forbidden_scheduled_agents": [],
            "required_executed_agents": [],
            "forbidden_executed_agents": [],
            "forbidden_entity_fields": [],
            "status": "needs_clarification",
            "entities": {},
            "missing_fields": [],
            "memory": {
                "preferences_change": "none",
                "trip_history_change": "none",
                "pending_trip": {"destination": "北京"},
            },
        }
        actual = {
            "scheduled_agents": [],
            "executed_agents": [],
            "status": "needs_clarification",
            "entities": {},
            "missing_fields": [],
            "memory_before": {"preferences": {}, "trip_history": []},
            "memory_after": {"preferences": {}, "trip_history": []},
            "pending_trip": {
                "destination": "北京",
                "trip_purpose": "旅游",
            },
        }

        self.assertTrue(evaluate_turn(expected, actual)["passed"])

    def test_expected_empty_pending_trip_requires_actual_state_to_be_empty(self):
        expected = {
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
                "pending_trip": {},
            },
        }
        actual = {
            "scheduled_agents": [],
            "executed_agents": [],
            "status": "completed",
            "entities": {},
            "missing_fields": [],
            "memory_before": {"preferences": {}, "trip_history": []},
            "memory_after": {"preferences": {}, "trip_history": []},
            "pending_trip": {"destination": "北京"},
        }

        result = evaluate_turn(expected, actual)

        self.assertFalse(result["passed"])
        self.assertIn("memory.pending_trip", result["failures"])

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
