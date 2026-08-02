"""ItineraryPlanningAgent 结构化输出可靠性的离线测试。"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

from agentscope.agent import AgentBase
from agentscope.message import Msg


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_itinerary_planning_agent():
    agent_path = (
        PROJECT_ROOT / ".claude" / "skills" / "plan-trip" / "script" / "agent.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plan_trip_skill_agent",
        agent_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 ItineraryPlanningAgent: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ItineraryPlanningAgent


ItineraryPlanningAgent = load_itinerary_planning_agent()


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return FakeResponse(self.responses.pop(0))


def make_agent(model):
    agent = object.__new__(ItineraryPlanningAgent)
    AgentBase.__init__(agent)
    agent.name = "ItineraryPlanningAgent"
    agent.model = model
    agent.skill_loader = types.SimpleNamespace(
        get_skill_content=lambda _name: "生成结构化行程。"
    )
    return agent


def make_input(
    query="2026年8月10日从苏州去北京3天",
    trip_purpose=None,
    fixed_events=None,
    planning_signals=None,
    event_overrides=None,
):
    event_data = {
        "origin": "苏州",
        "destination": "北京",
        "start_date": "2026-08-10",
        "duration_days": 3,
    }
    if trip_purpose is not None:
        event_data["trip_purpose"] = trip_purpose
    if fixed_events is not None:
        event_data["fixed_events"] = fixed_events
    if event_overrides:
        event_data.update(event_overrides)
    context = {
        "rewritten_query": query,
        "user_preferences": {},
    }
    if planning_signals is not None:
        context["planning_signals"] = planning_signals
    content = {
        "context": context,
        "previous_results": [
            {
                "agent_name": "event_collection",
                "result": {
                    "data": event_data,
                },
            }
        ],
    }
    return Msg(
        name="Orchestrator",
        content=json.dumps(content, ensure_ascii=False),
        role="user",
    )


class TestItineraryPlanningAgentOffline(unittest.IsolatedAsyncioTestCase):
    async def test_reference_mode_prompt_forbids_unsupported_realtime_facts(self):
        valid = json.dumps({
            "itinerary": {
                "title": "北京参考行程",
                "duration": "3天",
                "daily_plans": [],
                "notes": ["具体交通和住宿请在预订时确认。"],
            },
            "planning_complete": True,
            "booking_usage": {
                "outbound": "use_reference_plan",
                "return": "use_reference_plan",
                "hotel": "use_reference_plan",
            },
        }, ensure_ascii=False)
        model = FakeModel([valid])
        agent = make_agent(model)

        await agent.reply(make_input(event_overrides={
            "departure_time_window": "上午",
            "return_time_window": "下午",
            "outbound_booking_status": "reference",
            "return_booking_status": "reference",
            "hotel_booking_status": "reference",
        }))
        prompt = model.calls[0]["messages"][0]["content"]

        self.assertIn("无来源实时事实禁止规则", prompt)
        self.assertIn("不得生成具体车次、航班号、票价、房价", prompt)
        self.assertIn("不得生成具体酒店门店", prompt)
        self.assertIn("按之后确定的返程安排", prompt)

    async def test_user_confirmed_booking_details_are_allowed_and_preserved(self):
        valid = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "3天",
                "daily_plans": [{
                    "activities": [
                        {
                            "type": "transport_booking",
                            "booking_ref": "outbound",
                            "description": "模型不负责复制具体车次",
                        },
                        {
                            "type": "hotel_booking",
                            "booking_ref": "hotel",
                            "description": "模型不负责复制具体酒店",
                        },
                    ],
                }],
            },
            "planning_complete": True,
            "booking_usage": {
                "outbound": "use_confirmed_booking",
                "return": "use_reference_plan",
                "hotel": "use_confirmed_booking",
            },
        }, ensure_ascii=False)
        model = FakeModel([valid])
        agent = make_agent(model)

        response = await agent.reply(make_input(event_overrides={
            "departure_time_window": "上午",
            "return_time_window": "下午",
            "outbound_booking_status": "confirmed",
            "outbound_booking_details": "G123次列车，08:00发车",
            "return_booking_status": "reference",
            "hotel_booking_status": "confirmed",
            "hotel_booking_details": "北京国贸全季酒店",
        }))
        result = json.loads(response.content)

        self.assertTrue(result["planning_complete"])
        activities = result["itinerary"]["daily_plans"][0]["activities"]
        self.assertIn("G123", activities[0]["description"])
        self.assertIn("北京国贸全季酒店", activities[1]["description"])
        self.assertEqual(
            result["booking_summary"]["outbound"]["source"],
            "user_confirmed",
        )
        self.assertEqual(len(model.calls), 1)

    async def test_invalid_booking_usage_and_missing_ref_trigger_repair(self):
        invalid_references = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "3天",
                "daily_plans": [],
            },
            "planning_complete": True,
            "booking_usage": {
                "outbound": "use_reference_plan",
                "return": "use_reference_plan",
                "hotel": "use_reference_plan",
            },
        }, ensure_ascii=False)
        repaired = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "3天",
                "daily_plans": [{
                    "activities": [{
                        "type": "transport_booking",
                        "booking_ref": "return",
                        "description": "按固定返程安排",
                    }],
                }],
            },
            "planning_complete": True,
            "booking_usage": {
                "outbound": "use_reference_plan",
                "return": "use_confirmed_booking",
                "hotel": "use_reference_plan",
            },
        }, ensure_ascii=False)
        model = FakeModel([invalid_references, repaired])
        agent = make_agent(model)

        response = await agent.reply(make_input(event_overrides={
            "departure_time_window": "上午",
            "return_time_window": "下午",
            "outbound_booking_status": "reference",
            "return_booking_status": "confirmed",
            "return_booking_details": "G456，8月12日15:30发车",
            "hotel_booking_status": "reference",
        }))
        result = json.loads(response.content)

        self.assertTrue(result["planning_complete"])
        activity = result["itinerary"]["daily_plans"][0]["activities"][0]
        self.assertIn("G456", activity["description"])
        self.assertEqual(
            result["booking_usage"]["return"],
            "use_confirmed_booking",
        )
        self.assertEqual(len(model.calls), 2)
        repair_prompt = model.calls[1]["messages"][1]["content"]
        self.assertIn("booking_usage_mismatch", repair_prompt)
        self.assertIn("confirmed_booking_not_referenced", repair_prompt)
        self.assertIn("G456", repair_prompt)

    async def test_unsupported_realtime_facts_trigger_one_grounding_repair(self):
        unsupported = json.dumps({
            "itinerary": {
                "title": "北京参考行程",
                "duration": "3天",
                "daily_plans": [],
                "notes": [
                    "建议乘坐G123次高铁，二等座票价553元。",
                    "北京明天气温32℃。",
                    "已为您预订酒店。",
                ],
            },
            "planning_complete": True,
            "booking_usage": {
                "outbound": "use_reference_plan",
                "return": "use_reference_plan",
                "hotel": "use_reference_plan",
            },
        }, ensure_ascii=False)
        repaired = json.dumps({
            "itinerary": {
                "title": "北京参考行程",
                "duration": "3天",
                "daily_plans": [],
                "notes": [
                    "建议按照上午出发的时间范围选择合适交通。",
                    "抵达后前往之后确定的住宿地点。",
                    "具体交通、住宿和天气请在出发前确认。",
                ],
            },
            "planning_complete": True,
            "booking_usage": {
                "outbound": "use_reference_plan",
                "return": "use_reference_plan",
                "hotel": "use_reference_plan",
            },
        }, ensure_ascii=False)
        model = FakeModel([unsupported, repaired])
        agent = make_agent(model)

        response = await agent.reply(make_input(event_overrides={
            "departure_time_window": "上午",
            "return_time_window": "下午",
            "outbound_booking_status": "reference",
            "return_booking_status": "reference",
            "hotel_booking_status": "reference",
        }))
        result = json.loads(response.content)

        self.assertTrue(result["planning_complete"])
        self.assertNotIn("G123", json.dumps(result, ensure_ascii=False))
        self.assertEqual(len(model.calls), 2)
        repair_prompt = model.calls[1]["messages"][1]["content"]
        self.assertIn("unsupported_transport_identifier", repair_prompt)
        self.assertIn("unsupported_price", repair_prompt)
        self.assertIn("unsupported_weather_detail", repair_prompt)

    async def test_business_mode_instruction_is_injected_into_prompt(self):
        valid = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "3天",
                "daily_plans": [],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        model = FakeModel([valid])
        agent = make_agent(model)

        await agent.reply(make_input(
            query="事情办完以后直接回来",
            trip_purpose="客户拜访",
            planning_signals={
                "trip_type": "business",
                "leisure_preference": "forbidden",
                "explicit_constraints": ["办完即返"],
            },
        ))

        prompt = model.calls[0]["messages"][0]["content"]
        self.assertIn("企业差旅（纯商务）", prompt)
        self.assertIn("不得添加景点", prompt)
        self.assertIn('"planning_mode": "business_only"', prompt)
        self.assertIn('"leisure_preference": "forbidden"', prompt)
        self.assertIn("daily_plans 必须完整覆盖对应天数和日期", prompt)
        self.assertIn("missing_info 只用于记录仍建议确认的可选细节", prompt)
        self.assertIn("planning_complete 才设为 false", prompt)

    async def test_valid_json_does_not_trigger_repair(self):
        valid = json.dumps({
            "itinerary": {
                "title": "北京3日游",
                "duration": "3天",
                "daily_plans": [],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        model = FakeModel([valid])
        agent = make_agent(model)

        response = await agent.reply(make_input())
        result = json.loads(response.content)

        self.assertTrue(result["planning_complete"])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(
            model.calls[0]["kwargs"]["response_format"],
            {"type": "json_object"},
        )
        self.assertIn(
            "活动时间框、描述中的交通时间或耗时、出发前缓冲和下一项活动开始时间",
            model.calls[0]["messages"][0]["content"],
        )

    async def test_repairs_malformed_json_once(self):
        malformed = """{
          "itinerary": {
            "title": "北京3日游",
            "duration": "3天",
            "daily_plans": [{
              "day": 1,
              "activities": [{"description": "参观"故宫""}]
            }]
          },
          "planning_complete": true
        }"""
        repaired = json.dumps({
            "itinerary": {
                "title": "北京3日游",
                "duration": "3天",
                "daily_plans": [
                    {
                        "day": 1,
                        "activities": [
                            {"description": "参观故宫"},
                        ],
                    }
                ],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        model = FakeModel([malformed, repaired])
        agent = make_agent(model)

        response = await agent.reply(make_input())
        result = json.loads(response.content)

        self.assertTrue(result["planning_complete"])
        self.assertEqual(result["itinerary"]["title"], "北京3日游")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(
            model.calls[0]["kwargs"]["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            model.calls[1]["kwargs"]["response_format"],
            {"type": "json_object"},
        )

    async def test_second_invalid_response_returns_existing_error_result(self):
        model = FakeModel(["{invalid", "still invalid"])
        agent = make_agent(model)

        response = await agent.reply(make_input())
        result = json.loads(response.content)

        self.assertFalse(result["planning_complete"])
        self.assertIn("error", result)
        self.assertEqual(len(model.calls), 2)

    async def test_time_conflict_triggers_one_targeted_repair(self):
        conflicting = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "1天",
                "daily_plans": [
                    {
                        "day": 1,
                        "date": "2026-08-10",
                        "activities": [
                            {
                                "time": "07:00-08:00",
                                "location": "苏州北站至北京南站",
                                "type": "transport_booking",
                                "booking_ref": "outbound",
                                "description": (
                                    "乘坐G4次高铁（07:00-11:25）前往北京。"
                                ),
                            },
                            {
                                "time": "12:00-13:00",
                                "location": "酒店",
                                "description": "办理入住。",
                            },
                        ],
                    }
                ],
            },
            "planning_complete": True,
            "booking_usage": {
                "outbound": "use_confirmed_booking",
                "return": "use_reference_plan",
                "hotel": "use_reference_plan",
            },
        }, ensure_ascii=False)
        repaired = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "1天",
                "daily_plans": [
                    {
                        "day": 1,
                        "date": "2026-08-10",
                        "activities": [
                            {
                                "time": "07:00-11:30",
                                "location": "苏州北站至北京南站",
                                "type": "transport_booking",
                                "booking_ref": "outbound",
                                "description": (
                                    "乘坐G4次高铁（07:00-11:25）前往北京。"
                                ),
                            },
                            {
                                "time": "12:00-13:00",
                                "location": "酒店",
                                "description": "办理入住。",
                            },
                        ],
                    }
                ],
            },
            "planning_complete": True,
            "booking_usage": {
                "outbound": "use_confirmed_booking",
                "return": "use_reference_plan",
                "hotel": "use_reference_plan",
            },
        }, ensure_ascii=False)
        model = FakeModel([conflicting, repaired])
        agent = make_agent(model)

        response = await agent.reply(make_input(event_overrides={
            "outbound_booking_status": "confirmed",
            "outbound_booking_details": "G4次高铁，07:00发车",
            "return_booking_status": "reference",
            "hotel_booking_status": "reference",
        }))
        result = json.loads(response.content)

        self.assertEqual(
            result["itinerary"]["daily_plans"][0]["activities"][0]["time"],
            "07:00出发",
        )
        self.assertEqual(len(model.calls), 2)
        self.assertIn(
            "transport_time_outside_activity",
            model.calls[1]["messages"][1]["content"],
        )
        self.assertIn(
            "允许调整、缩短、重新排序或删除",
            model.calls[1]["messages"][0]["content"],
        )
        self.assertIn(
            "普通用户偏好属于软约束",
            model.calls[1]["messages"][0]["content"],
        )

    async def test_missing_train_buffer_triggers_one_targeted_repair(self):
        conflicting = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "1天",
                "daily_plans": [
                    {
                        "day": 1,
                        "date": "2026-08-10",
                        "activities": [
                            {
                                "time": "07:30-08:00",
                                "location": "苏州市区至苏州北站",
                                "description": "前往苏州北站。",
                                "transport": "地铁/出租车",
                            },
                            {
                                "time": "08:00-12:30",
                                "location": "苏州北站至北京南站",
                                "description": "乘坐高铁前往北京。",
                                "transport": "高铁",
                            },
                        ],
                    }
                ],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        repaired = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "1天",
                "daily_plans": [
                    {
                        "day": 1,
                        "date": "2026-08-10",
                        "activities": [
                            {
                                "time": "07:00-07:30",
                                "location": "苏州市区至苏州北站",
                                "description": "前往苏州北站。",
                                "transport": "地铁/出租车",
                            },
                            {
                                "time": "07:30-08:00",
                                "location": "苏州北站",
                                "description": "完成安检、检票并候车。",
                                "transport": "步行",
                            },
                            {
                                "time": "08:00-12:30",
                                "location": "苏州北站至北京南站",
                                "description": "乘坐高铁前往北京。",
                                "transport": "高铁",
                            },
                        ],
                    }
                ],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        model = FakeModel([conflicting, repaired])
        agent = make_agent(model)

        response = await agent.reply(make_input())
        result = json.loads(response.content)

        self.assertTrue(result["planning_complete"])
        self.assertEqual(len(model.calls), 2)
        self.assertIn(
            "insufficient_departure_buffer",
            model.calls[1]["messages"][1]["content"],
        )
        self.assertIn(
            "铁路发车前必须明确保留至少30分钟",
            model.calls[1]["messages"][0]["content"],
        )

    async def test_failed_time_repair_keeps_original_itinerary(self):
        conflicting = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "1天",
                "daily_plans": [
                    {
                        "day": 1,
                        "activities": [
                            {
                                "time": "09:00-10:00",
                                "location": "宁波站至苏州站",
                                "description": "乘坐高铁，车程约2小时。",
                            }
                        ],
                    }
                ],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        model = FakeModel([conflicting, "{invalid"])
        agent = make_agent(model)

        response = await agent.reply(make_input())
        result = json.loads(response.content)

        self.assertFalse(result["planning_complete"])
        self.assertEqual(
            result["itinerary"]["daily_plans"][0]["activities"][0]["time"],
            "09:00-10:00",
        )
        self.assertEqual(
            result["time_consistency"]["status"],
            "unresolved",
        )
        self.assertIn(
            "存在未能自动解决的时间可行性问题",
            result["itinerary"]["notes"][0],
        )
        self.assertEqual(len(model.calls), 2)

    async def test_remaining_conflict_after_repair_is_marked_unresolved(self):
        conflicting = json.dumps({
            "itinerary": {
                "title": "北京商务行程",
                "duration": "1天",
                "daily_plans": [
                    {
                        "day": 1,
                        "activities": [
                            {
                                "time": "09:00-10:00",
                                "location": "宁波站至苏州站",
                                "description": "乘坐高铁，车程约2小时。",
                            }
                        ],
                    }
                ],
            },
            "planning_complete": True,
        }, ensure_ascii=False)
        model = FakeModel([conflicting, conflicting])
        agent = make_agent(model)

        response = await agent.reply(make_input())
        result = json.loads(response.content)

        self.assertFalse(result["planning_complete"])
        self.assertIn(
            "transport_duration_exceeds_activity",
            {
                issue["category"]
                for issue in result["time_consistency"]["issues"]
            },
        )
        self.assertEqual(len(model.calls), 2)


if __name__ == "__main__":
    unittest.main()
