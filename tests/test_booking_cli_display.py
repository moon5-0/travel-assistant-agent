"""CLI 使用结构化预订摘要展示可信事实。"""

from __future__ import annotations

import unittest

from tests.test_cli_qa import capture_display_results


class TestBookingCliDisplay(unittest.TestCase):
    def test_booking_summary_is_displayed_before_itinerary(self):
        output = capture_display_results({
            "results": [{
                "agent_name": "itinerary_planning",
                "status": "success",
                "data": {
                    "booking_summary": {
                        "outbound": {
                            "text": "去程：已确认，G123，07:30发车",
                        },
                        "return": {
                            "text": "返程：尚未确认，根据下午时间范围选择合适交通",
                        },
                        "hotel": {
                            "text": "住宿：已确认，北京国贸全季酒店",
                        },
                    },
                    "itinerary": {
                        "title": "北京三日游",
                        "duration": "3天",
                        "daily_plans": [],
                    },
                },
            }],
        })

        self.assertIn("预订信息", output)
        self.assertIn("G123", output)
        self.assertIn("下午时间范围", output)
        self.assertIn("北京国贸全季酒店", output)
        self.assertLess(output.index("预订信息"), output.index("G123"))


if __name__ == "__main__":
    unittest.main()
