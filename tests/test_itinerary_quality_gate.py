"""规划结果统一质量门的离线测试。"""

from __future__ import annotations

import unittest

from utils.booking_context import build_booking_context
from utils.itinerary_quality_gate import (
    collect_itinerary_quality_issues,
    finalize_quality_gate,
    normalize_itinerary_result,
    prepare_event_data_for_planning,
)


def complete_result(days: int = 3):
    plans = []
    cities = ["北京", "北京", "苏州"]
    dates = ["2026-08-10", "2026-08-11", "2026-08-12"]
    for index in range(days):
        plans.append({
            "day": index + 1,
            "date": dates[index],
            "city": cities[index],
            "activities": [{
                "time": "09:00-11:00",
                "type": "general",
                "title": "商务安排",
                "description": "按既定安排开展当天活动。",
            }],
        })
    return {
        "itinerary": {
            "title": "苏州至北京三日商务行程",
            "daily_plans": plans,
            "notes": ["具体交通与住宿信息请在预订时确认。"],
        },
        "planning_complete": True,
    }


class TestItineraryQualityGate(unittest.TestCase):
    def setUp(self):
        self.event_data = {
            "origin": "苏州",
            "destination": "北京",
            "start_date": "2026-08-10",
            "end_date": "2026-08-12",
            "duration_days": 3,
        }

    def test_complete_three_day_itinerary_passes(self):
        result = complete_result()

        issues = collect_itinerary_quality_issues(
            result,
            self.event_data,
            {},
            trusted_context=self.event_data,
        )

        self.assertEqual(issues, [])
        finalized = finalize_quality_gate(result, issues)
        self.assertTrue(finalized["planning_complete"])
        self.assertEqual(finalized["quality_gate"]["status"], "passed")

    def test_missing_days_and_dates_are_blocking(self):
        result = complete_result(days=1)

        categories = {
            issue["category"]
            for issue in collect_itinerary_quality_issues(
                result,
                self.event_data,
                {},
                trusted_context=self.event_data,
            )
        }

        self.assertIn("duration_mismatch", categories)
        self.assertIn("date_coverage_mismatch", categories)

    def test_missing_fixed_event_is_blocking(self):
        result = complete_result()
        event_data = prepare_event_data_for_planning({
            **self.event_data,
            "fixed_events": [{
                "date": "2026-08-11",
                "time": "14:00-16:00",
                "location": "国贸中心",
                "title": "客户会议",
            }],
        })

        categories = {
            issue["category"]
            for issue in collect_itinerary_quality_issues(
                result,
                event_data,
                {},
                trusted_context=event_data,
            )
        }

        self.assertIn("fixed_event_not_referenced", categories)

    def test_normalization_rebuilds_booking_usage_from_trusted_context(self):
        booking_context = build_booking_context({
            "hotel_booking_status": "reference",
            "hotel_time_window": "到达后",
        })
        result = complete_result()
        result["booking_usage"] = {"hotel": "use_confirmed_booking"}
        result["fact_grounding"] = {"status": "unresolved"}

        normalized = normalize_itinerary_result(result, booking_context)

        self.assertEqual(
            normalized["booking_usage"]["hotel"],
            "use_reference_plan",
        )
        self.assertNotIn("fact_grounding", normalized)

    def test_normalization_accepts_day_number_and_fills_safe_fields(self):
        result = complete_result(days=1)
        plan = result["itinerary"]["daily_plans"][0]
        plan["day_number"] = plan.pop("day")
        plan.pop("city")
        result["itinerary"].pop("route", None)

        normalized = normalize_itinerary_result(
            result,
            {},
            self.event_data,
        )
        normalized_plan = normalized["itinerary"]["daily_plans"][0]

        self.assertEqual(normalized_plan["day"], 1)
        self.assertEqual(normalized_plan["city"], "北京")
        self.assertEqual(
            normalized["itinerary"]["route"],
            "苏州 -> 北京",
        )

    def test_normalization_accepts_common_activity_field_aliases(self):
        result = complete_result(days=1)
        plan = result["itinerary"]["daily_plans"][0]
        plan["items"] = [{
            "time": "09:00-11:00",
            "activity": "乘坐参考去程交通",
            "location": "苏州至北京",
            "type": "transport",
            "booking_ref": "outbound",
        }]
        plan.pop("activities")

        normalized = normalize_itinerary_result(result, {}, self.event_data)
        activity = normalized["itinerary"]["daily_plans"][0]["activities"][0]

        self.assertEqual(activity["title"], "乘坐参考去程交通")
        self.assertEqual(activity["type"], "transport_booking")
        self.assertEqual(activity["start_time"], "09:00")
        self.assertEqual(activity["end_time"], "11:00")

    def test_provable_time_problem_is_blocking(self):
        result = complete_result(days=1)
        result["itinerary"]["daily_plans"][0]["activities"] = [{
            "time": "09:00-10:00",
            "location": "宁波站至苏州站",
            "description": "乘坐高铁，车程约2小时。",
        }]

        issues = collect_itinerary_quality_issues(
            result,
            {**self.event_data, "duration_days": 1, "end_date": "2026-08-10"},
            {},
        )
        time_issues = [item for item in issues if item["source"] == "time"]

        self.assertTrue(time_issues)
        self.assertTrue(all(item["severity"] == "blocking" for item in time_issues))
        finalized = finalize_quality_gate(result, issues)
        self.assertFalse(finalized["planning_complete"])
        self.assertEqual(finalized["quality_gate"]["status"], "unresolved")

    def test_empty_activity_content_is_blocking(self):
        result = complete_result(days=1)
        result["itinerary"]["daily_plans"][0]["activities"] = [{
            "time": "09:00-10:00",
            "type": "general",
            "location": None,
            "description": None,
        }]

        issues = collect_itinerary_quality_issues(
            result,
            {**self.event_data, "duration_days": 1, "end_date": "2026-08-10"},
            {},
        )

        self.assertIn(
            "invalid_activity_structure",
            {item["category"] for item in issues},
        )

    def test_reference_return_requires_structured_activity(self):
        result = complete_result(days=1)
        event_data = {
            **self.event_data,
            "duration_days": 1,
            "end_date": "2026-08-10",
            "return_time_window": "下午",
            "return_booking_status": "reference",
        }
        booking_context = build_booking_context(event_data)

        issues = collect_itinerary_quality_issues(
            result,
            event_data,
            booking_context,
        )

        self.assertIn(
            "required_booking_not_referenced",
            {item["category"] for item in issues},
        )

    def test_unresolved_issue_marks_result_incomplete(self):
        result = complete_result()
        issues = [{
            "category": "duration_mismatch",
            "source": "structure",
            "severity": "blocking",
            "message": "天数不一致",
        }]

        finalized = finalize_quality_gate(result, issues)

        self.assertFalse(finalized["planning_complete"])
        self.assertEqual(finalized["quality_gate"]["status"], "unresolved")


if __name__ == "__main__":
    unittest.main()
