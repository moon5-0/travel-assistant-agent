"""规划模式策略层的离线测试。"""

from __future__ import annotations

import unittest

from utils.planning_policy import determine_planning_mode


def make_info(
    *,
    trip_purpose="",
    fixed_events=None,
    trip_type="unknown",
    leisure_preference="unspecified",
    explicit_constraints=None,
):
    return {
        "context": {
            "planning_signals": {
                "trip_type": trip_type,
                "leisure_preference": leisure_preference,
                "explicit_constraints": explicit_constraints or [],
            }
        },
        "event_collection": {
            "trip_purpose": trip_purpose,
            "fixed_events": fixed_events or [],
        },
    }


class TestPlanningPolicy(unittest.TestCase):
    def test_business_trip_defaults_to_business_first(self):
        mode = determine_planning_mode(
            "去北京三天",
            make_info(trip_purpose="客户拜访", trip_type="business"),
        )

        self.assertEqual(mode, "business_first")

    def test_semantic_no_leisure_signal_handles_uncommon_wording(self):
        mode = determine_planning_mode(
            "我不想把这趟差事搞得跟度假一样，事情办完就回来",
            make_info(
                trip_purpose="客户沟通",
                trip_type="business",
                leisure_preference="forbidden",
                explicit_constraints=["办完即返"],
            ),
        )

        self.assertEqual(mode, "business_only")

    def test_semantic_leisure_request_handles_uncommon_wording(self):
        mode = determine_planning_mode(
            "正事结束后如果来得及，可以在酒店附近走走",
            make_info(
                trip_purpose="项目交流",
                trip_type="business",
                leisure_preference="requested",
            ),
        )

        self.assertEqual(mode, "business_with_optional_leisure")

    def test_forbidden_signal_has_priority_over_leisure_words(self):
        mode = determine_planning_mode(
            "这是旅游城市，但本次只办工作",
            make_info(
                trip_type="business",
                leisure_preference="forbidden",
            ),
        )

        self.assertEqual(mode, "business_only")

    def test_personal_trip_uses_general_travel_mode(self):
        mode = determine_planning_mode(
            "带父母去北京玩三天",
            make_info(trip_purpose="家庭出游", trip_type="personal"),
        )

        self.assertEqual(mode, "general_travel")

    def test_old_output_without_signals_still_uses_keyword_fallback(self):
        mode = determine_planning_mode(
            "去上海出差，当天往返，不要安排旅游景点",
            {"event_collection": {"trip_purpose": "商务会议"}},
        )

        self.assertEqual(mode, "business_only")


if __name__ == "__main__":
    unittest.main(verbosity=2)
