"""行程时间一致性检查器的离线测试。"""

from __future__ import annotations

import unittest

from utils.itinerary_time_validator import find_itinerary_time_issues


def make_result(activities):
    return {
        "itinerary": {
            "daily_plans": [
                {
                    "day": 1,
                    "date": "2026-08-10",
                    "activities": activities,
                }
            ]
        },
        "planning_complete": True,
    }


class TestItineraryTimeValidator(unittest.TestCase):
    def test_valid_schedule_has_no_issues(self):
        result = make_result([
            {
                "time": "07:00-11:30",
                "location": "北京南站",
                "description": (
                    "乘坐建议车次G4（07:00-11:25）前往北京。"
                ),
            },
            {
                "time": "12:00-13:00",
                "location": "酒店",
                "description": "办理入住并用餐。",
            },
        ])

        self.assertEqual(find_itinerary_time_issues(result), [])

    def test_detects_overlapping_activities(self):
        result = make_result([
            {"time": "09:00-11:00", "description": "客户会议"},
            {"time": "10:30-12:00", "description": "前往酒店"},
        ])

        categories = {
            issue["category"]
            for issue in find_itinerary_time_issues(result)
        }

        self.assertIn("overlapping_activities", categories)

    def test_detects_transport_time_outside_activity_slot(self):
        result = make_result([
            {
                "time": "07:00-08:00",
                "location": "苏州北站至北京南站",
                "description": (
                    "建议乘坐G4次高铁（07:00-11:25）前往北京。"
                ),
            }
        ])

        categories = {
            issue["category"]
            for issue in find_itinerary_time_issues(result)
        }

        self.assertIn("transport_time_outside_activity", categories)

    def test_detects_transport_duration_longer_than_slot(self):
        result = make_result([
            {
                "time": "09:00-10:00",
                "location": "宁波站至苏州站",
                "description": "乘坐高铁返回苏州，车程约2小时。",
            }
        ])

        categories = {
            issue["category"]
            for issue in find_itinerary_time_issues(result)
        }

        self.assertIn(
            "transport_duration_exceeds_activity",
            categories,
        )

    def test_detects_approximate_transport_duration_longer_than_slot(self):
        result = make_result([
            {
                "time": "07:00-10:00",
                "location": "苏州至北京",
                "description": "乘坐高铁前往北京，约4.5小时。",
            }
        ])

        categories = {
            issue["category"]
            for issue in find_itinerary_time_issues(result)
        }

        self.assertIn(
            "transport_duration_exceeds_activity",
            categories,
        )

    def test_allows_small_rounding_difference_in_transport_duration(self):
        result = make_result([
            {
                "time": "06:58-11:23",
                "location": "苏州站至北京南站",
                "description": (
                    "乘坐高铁（06:58-11:23），约4.5小时抵达。"
                ),
            }
        ])

        self.assertEqual(find_itinerary_time_issues(result), [])

    def test_allows_small_rounding_difference_in_arrival_time(self):
        result = make_result([
            {
                "time": "14:30-19:00",
                "location": "武汉站至上海虹桥站",
                "description": (
                    "乘坐高铁（14:30-19:06），历时4小时36分钟。"
                ),
            }
        ])

        self.assertEqual(find_itinerary_time_issues(result), [])

    def test_does_not_treat_opening_hours_as_transport_schedule(self):
        result = make_result([
            {
                "time": "09:00-12:00",
                "location": "故宫博物院",
                "description": "开放时间08:30-17:00，请提前预约。",
                "transport": "步行",
            }
        ])

        self.assertEqual(find_itinerary_time_issues(result), [])

    def test_does_not_treat_check_in_buffer_as_travel_duration(self):
        result = make_result([
            {
                "time": "08:00-09:00",
                "location": "机场",
                "description": "请提前约1.5小时到达机场办理值机。",
                "transport": "地铁",
            }
        ])

        self.assertEqual(find_itinerary_time_issues(result), [])


if __name__ == "__main__":
    unittest.main()
