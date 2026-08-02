"""构建、校验并渲染行程中的结构化预订引用。"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Dict, Iterable, List, Tuple


BOOKING_ITEMS = {
    "outbound": {
        "label": "去程",
        "status_field": "outbound_booking_status",
        "details_field": "outbound_booking_details",
        "time_window_field": "departure_time_window",
        "activity_type": "transport_booking",
    },
    "return": {
        "label": "返程",
        "status_field": "return_booking_status",
        "details_field": "return_booking_details",
        "time_window_field": "return_time_window",
        "activity_type": "transport_booking",
    },
    "hotel": {
        "label": "住宿",
        "status_field": "hotel_booking_status",
        "details_field": "hotel_booking_details",
        "time_window_field": None,
        "activity_type": "hotel_booking",
    },
}

USAGE_BY_STATUS = {
    "confirmed": "use_confirmed_booking",
    "reference": "use_reference_plan",
}

TIME_PATTERN = re.compile(r"(?<!\d)(\d{1,2}:\d{2})(?!\d)")


def _confirmed_display_time(details: Any) -> str:
    """只从原始确认详情提取可展示时间，不采用模型补充的到达时刻。"""
    if isinstance(details, dict):
        departure = details.get("departure_time")
        arrival = details.get("arrival_time")
        if departure and arrival:
            return f"{departure}-{arrival}"
        if departure:
            return f"{departure}出发"
        return "按已确认预订"
    matches = TIME_PATTERN.findall(str(details or ""))
    if len(matches) >= 2:
        return f"{matches[0]}-{matches[1]}"
    if matches:
        return f"{matches[0]}出发"
    return "按已确认预订"


def build_booking_context(event_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """把事项收集结果转换成规划阶段唯一可信的预订上下文。"""
    context: Dict[str, Dict[str, Any]] = {}
    if not isinstance(event_data, dict):
        return context

    for booking_ref, config in BOOKING_ITEMS.items():
        status = event_data.get(config["status_field"])
        if status not in USAGE_BY_STATUS:
            continue
        details = event_data.get(config["details_field"])
        if status == "reference":
            details = None
        context[booking_ref] = {
            "label": config["label"],
            "status": status,
            "source": "user_confirmed" if status == "confirmed" else "user_reference",
            "details": details,
            "time_window": (
                event_data.get(config["time_window_field"])
                if config["time_window_field"]
                else None
            ),
            "activity_type": config["activity_type"],
            "display_time": (
                _confirmed_display_time(details)
                if status == "confirmed"
                and config["activity_type"] == "transport_booking"
                else None
            ),
        }
    return context


def expected_booking_usage(
    booking_context: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """根据可信状态生成规划 Agent 必须声明的使用方式。"""
    return {
        booking_ref: USAGE_BY_STATUS[item["status"]]
        for booking_ref, item in booking_context.items()
        if item.get("status") in USAGE_BY_STATUS
    }


def _iter_activities(result: Dict[str, Any]) -> Iterable[Tuple[int, int, Dict[str, Any]]]:
    itinerary = result.get("itinerary", {})
    if not isinstance(itinerary, dict):
        return
    daily_plans = itinerary.get("daily_plans", [])
    if not isinstance(daily_plans, list):
        return
    for day_index, day_plan in enumerate(daily_plans):
        if not isinstance(day_plan, dict):
            continue
        activities = day_plan.get("activities")
        if not isinstance(activities, list):
            activities = day_plan.get("time_slots")
        if not isinstance(activities, list):
            continue
        for activity_index, activity in enumerate(activities):
            if isinstance(activity, dict):
                yield day_index, activity_index, activity


def find_booking_reference_issues(
    result: Dict[str, Any],
    booking_context: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """检查 Agent 声明的使用方式和活动引用是否与可信上下文一致。"""
    if not booking_context:
        return []

    issues: List[Dict[str, Any]] = []
    expected_usage = expected_booking_usage(booking_context)
    # 模型偶尔会把根字段 booking_usage 放进 itinerary。校验时兼容读取，
    # 但根字段优先；后续渲染仍会统一整理为规范的根层结构。
    actual_usage: Dict[str, Any] = {}
    itinerary = result.get("itinerary")
    if isinstance(itinerary, dict):
        nested_usage = itinerary.get("booking_usage")
        if isinstance(nested_usage, dict):
            actual_usage.update(nested_usage)
    root_usage = result.get("booking_usage")
    if isinstance(root_usage, dict):
        actual_usage.update(root_usage)

    for booking_ref, expected in expected_usage.items():
        actual = actual_usage.get(booking_ref)
        if actual != expected:
            issues.append({
                "category": "booking_usage_mismatch",
                "booking_ref": booking_ref,
                "expected": expected,
                "actual": actual,
                "message": "规划声明的预订使用方式与可信预订状态不一致。",
            })

    referenced: set[str] = set()
    for day_index, activity_index, activity in _iter_activities(result):
        booking_ref = activity.get("booking_ref")
        if booking_ref is None:
            continue
        if booking_ref not in booking_context:
            issues.append({
                "category": "unknown_booking_ref",
                "booking_ref": booking_ref,
                "day_index": day_index,
                "activity_index": activity_index,
                "message": "活动引用了不存在的预订事实。",
            })
            continue
        referenced.add(booking_ref)
        expected_type = booking_context[booking_ref]["activity_type"]
        actual_type = activity.get("type")
        if actual_type != expected_type:
            issues.append({
                "category": "booking_activity_type_mismatch",
                "booking_ref": booking_ref,
                "day_index": day_index,
                "activity_index": activity_index,
                "expected": expected_type,
                "actual": actual_type,
                "message": "预订引用与活动类型不匹配。",
            })

    for booking_ref, item in booking_context.items():
        if item.get("status") == "confirmed" and booking_ref not in referenced:
            issues.append({
                "category": "confirmed_booking_not_referenced",
                "booking_ref": booking_ref,
                "message": "用户确认的预订没有进入任何行程活动。",
            })

    return issues


def _render_summary_item(item: Dict[str, Any]) -> str:
    label = item["label"]
    if item.get("status") == "confirmed":
        return f"{label}：已确认，{item.get('details') or '详情待补充'}"
    if label == "住宿":
        return "住宿：尚未确认，按之后确定的住宿安排"
    time_window = item.get("time_window") or "用户给定"
    return f"{label}：尚未确认，根据{time_window}时间范围选择合适交通"


def _render_activity(activity: Dict[str, Any], item: Dict[str, Any]) -> None:
    """只覆盖预订事实相关字段，保留 Agent 决定的日期和时间顺序。"""
    label = item["label"]
    status = item.get("status")
    details = item.get("details")

    if item["activity_type"] == "transport_booking":
        if status == "confirmed":
            activity["time"] = item.get("display_time") or "按已确认预订"
            activity["location"] = f"已确认{label}交通"
            activity["description"] = f"按用户确认的预订出行：{details}"
            activity["transport"] = str(details)
        else:
            time_window = item.get("time_window") or "用户给定"
            activity["time"] = str(time_window)
            activity["location"] = f"{label}交通（待确认）"
            activity["description"] = (
                f"根据{time_window}时间范围选择合适交通，"
                "具体安排以之后确认结果为准。"
            )
            activity["transport"] = "待确认"
        return

    if status == "confirmed":
        activity["location"] = str(details)
        activity["description"] = f"前往用户确认的住宿：{details}"
    else:
        activity["location"] = "住宿地点（待确认）"
        activity["description"] = "前往之后确定的住宿地点。"


def render_booking_references(
    result: Dict[str, Any],
    booking_context: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """从可信上下文生成摘要并展开合法 booking_ref，不信任模型副本。"""
    rendered = deepcopy(result)
    itinerary = rendered.get("itinerary")
    if isinstance(itinerary, dict):
        # 避免根层和 itinerary 内同时保留两份可能不一致的状态。
        itinerary.pop("booking_usage", None)
    rendered["booking_usage"] = expected_booking_usage(booking_context)
    rendered["booking_summary"] = {
        booking_ref: {
            "label": item["label"],
            "status": item["status"],
            "source": item["source"],
            "text": _render_summary_item(item),
        }
        for booking_ref, item in booking_context.items()
    }

    for _, _, activity in _iter_activities(rendered):
        booking_ref = activity.get("booking_ref")
        item = booking_context.get(booking_ref)
        if item is not None and activity.get("type") == item["activity_type"]:
            _render_activity(activity, item)
    return rendered
