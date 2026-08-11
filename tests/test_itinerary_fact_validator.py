"""行程无来源实时事实检查器的离线测试。"""

from __future__ import annotations

import unittest

from utils.itinerary_fact_validator import (
    find_unsupported_itinerary_facts,
)


def result_with_notes(*notes):
    return {
        "itinerary": {
            "title": "参考行程",
            "daily_plans": [],
            "notes": list(notes),
        },
        "planning_complete": True,
    }


class TestItineraryFactValidator(unittest.TestCase):
    def test_generic_reference_plan_has_no_issues(self):
        result = result_with_notes(
            "建议根据上午出发的时间范围选择合适交通。",
            "抵达后前往之后确定的住宿地点。",
            "具体信息请在正式预订时确认。",
        )

        self.assertEqual(
            find_unsupported_itinerary_facts(result, {}),
            [],
        )

    def test_detects_high_risk_unsupported_realtime_claims(self):
        result = result_with_notes(
            "建议乘坐G123次列车，票价553元。",
            "预计气温32℃。",
            "已为您预订酒店。",
        )

        categories = {
            issue["category"]
            for issue in find_unsupported_itinerary_facts(result, {})
        }

        self.assertEqual(categories, {
            "unsupported_transport_identifier",
            "unsupported_price",
            "unsupported_weather_detail",
            "unsupported_confirmation",
        })

    def test_allows_identifier_and_price_from_user_confirmed_details(self):
        result = result_with_notes(
            "按用户已预订的G123次列车出发，已付票价553元。",
        )
        event_data = {
            "outbound_booking_status": "confirmed",
            "outbound_booking_details": "G123次列车，票价553元",
        }

        self.assertEqual(
            find_unsupported_itinerary_facts(result, event_data),
            [],
        )

    def test_rough_total_budget_is_not_treated_as_ticket_or_hotel_price(self):
        result = {
            "itinerary": {
                "title": "参考行程",
                "daily_plans": [],
                "estimated_budget": "整体预算可暂按约2000元控制",
            },
            "planning_complete": True,
        }

        self.assertEqual(
            find_unsupported_itinerary_facts(result, {}),
            [],
        )

    def test_allows_price_from_explicit_company_policy(self):
        result = result_with_notes("住宿费用需控制在每晚500元以内。")
        trusted_context = {
            "policy_constraints": {
                "hotel_budget_per_night": 500,
            },
        }

        self.assertEqual(
            find_unsupported_itinerary_facts(
                result,
                {},
                trusted_context=trusted_context,
            ),
            [],
        )

    def test_other_price_is_still_rejected_when_policy_has_a_budget(self):
        result = result_with_notes("建议选择每晚800元的酒店。")
        trusted_context = {
            "policy_constraints": {
                "hotel_budget_per_night": 500,
            },
        }

        categories = {
            issue["category"]
            for issue in find_unsupported_itinerary_facts(
                result,
                {},
                trusted_context=trusted_context,
            )
        }

        self.assertIn("unsupported_price", categories)

    def test_confirmed_business_meeting_is_not_treated_as_booking_claim(self):
        result = result_with_notes("已安排客户会议，按固定时间到场。")

        self.assertEqual(
            find_unsupported_itinerary_facts(result, {}),
            [],
        )

if __name__ == "__main__":
    unittest.main()
