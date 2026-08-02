"""结构化预订上下文、引用校验与确定性渲染测试。"""

from __future__ import annotations

import unittest

from utils.booking_context import (
    build_booking_context,
    find_booking_reference_issues,
    render_booking_references,
)


class TestBookingContext(unittest.TestCase):
    def test_confirmed_booking_is_rendered_from_original_context(self):
        context = build_booking_context({
            "outbound_booking_status": "confirmed",
            "outbound_booking_details": "G123，07:30发车",
        })
        result = {
            "itinerary": {
                "daily_plans": [{
                    "activities": [{
                        "time": "09:00-14:00",
                        "type": "transport_booking",
                        "booking_ref": "outbound",
                        "description": "模型错误地写成G999",
                    }],
                }],
            },
            "planning_complete": True,
            "booking_usage": {"outbound": "use_confirmed_booking"},
        }

        self.assertEqual(find_booking_reference_issues(result, context), [])
        rendered = render_booking_references(result, context)
        activity = rendered["itinerary"]["daily_plans"][0]["activities"][0]

        self.assertIn("G123", activity["description"])
        self.assertNotIn("G999", activity["description"])
        self.assertEqual(activity["time"], "07:30出发")
        self.assertIn("G123", rendered["booking_summary"]["outbound"]["text"])

    def test_reference_booking_is_rendered_without_specific_facts(self):
        context = build_booking_context({
            "return_booking_status": "reference",
            "return_booking_details": "G456",
            "return_time_window": "下午",
        })
        result = {
            "itinerary": {
                "daily_plans": [{
                    "activities": [{
                        "time": "15:00-20:00",
                        "type": "transport_booking",
                        "booking_ref": "return",
                    }],
                }],
            },
            "booking_usage": {"return": "use_reference_plan"},
        }

        rendered = render_booking_references(result, context)
        activity = rendered["itinerary"]["daily_plans"][0]["activities"][0]

        self.assertNotIn("G456", str(rendered))
        self.assertEqual(activity["time"], "下午")
        self.assertIn("下午", activity["description"])
        self.assertEqual(activity["transport"], "待确认")

    def test_invalid_usage_and_missing_confirmed_ref_are_reported(self):
        context = build_booking_context({
            "hotel_booking_status": "confirmed",
            "hotel_booking_details": "北京国贸全季酒店",
        })
        result = {
            "itinerary": {"daily_plans": []},
            "booking_usage": {"hotel": "use_reference_plan"},
        }

        categories = {
            issue["category"]
            for issue in find_booking_reference_issues(result, context)
        }

        self.assertEqual(categories, {
            "booking_usage_mismatch",
            "confirmed_booking_not_referenced",
        })

    def test_nested_booking_usage_is_accepted_then_normalized_to_root(self):
        context = build_booking_context({
            "outbound_booking_status": "confirmed",
            "outbound_booking_details": "G123，8月10日07:30发车",
        })
        result = {
            "itinerary": {
                "booking_usage": {
                    "outbound": "use_confirmed_booking",
                },
                "daily_plans": [{
                    "activities": [{
                        "type": "transport_booking",
                        "booking_ref": "outbound",
                    }],
                }],
            },
        }

        self.assertEqual(find_booking_reference_issues(result, context), [])

        rendered = render_booking_references(result, context)
        self.assertEqual(
            rendered["booking_usage"]["outbound"],
            "use_confirmed_booking",
        )
        self.assertNotIn("booking_usage", rendered["itinerary"])


if __name__ == "__main__":
    unittest.main()
