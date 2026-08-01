"""行程活动的时间可行性检查。

只处理能够由结构化时间和明确业务规则确定的问题，不推断真实车次或路线耗时。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


TIME_RANGE_PATTERN = re.compile(
    r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)\s*[-—~～至到]\s*"
    r"([01]?\d|2[0-3]):([0-5]\d)(?!\d)"
)
HOUR_DURATION_PATTERN = re.compile(
    r"(?:历时|车程|飞行时间|耗时|用时)\s*(?:约|大约|预计)?\s*"
    r"(\d+(?:\.\d+)?)\s*小时(?:\s*(\d+)\s*分钟)?"
)
MINUTE_DURATION_PATTERN = re.compile(
    r"(?:历时|车程|飞行时间|耗时|用时)\s*(?:约|大约|预计)?\s*"
    r"(\d+)\s*分钟"
)
APPROXIMATE_HOUR_DURATION_PATTERN = re.compile(
    r"(?:约|大约|预计)\s*(\d+(?:\.\d+)?)\s*小时"
    r"(?:\s*(\d+)\s*分钟)?"
)
TRANSPORT_KEYWORDS = (
    "高铁",
    "火车",
    "列车",
    "动车",
    "航班",
    "飞机",
    "机场",
    "车站",
    "乘坐",
    "出发",
    "抵达",
    "返回",
    "车程",
    "飞行",
)
RAIL_TRANSPORT_KEYWORDS = (
    "高铁",
    "动车",
    "城际列车",
    "火车",
    "列车",
)
RAIL_BOARDING_PATTERN = re.compile(
    r"乘(?:坐)?[^。；，,]{0,8}(?:高铁|动车|城际列车|火车|列车)"
)
DEPARTURE_BUFFER_ACTIVITY_KEYWORDS = (
    "安检",
    "进站",
    "检票",
    "候车",
)
TRAVEL_TO_TERMINAL_KEYWORDS = (
    "前往",
    "赶往",
    "去往",
    "到达",
    "抵达",
    "->",
    "→",
)
EXPLICIT_BUFFER_MINUTES_PATTERN = re.compile(
    r"预留\s*(?:至少)?\s*(\d+)\s*分钟"
)
# 这是项目采用的保守规划规则，不代表对具体车站检票时间的实时查询结果。
RAIL_DEPARTURE_BUFFER_MINUTES = 30
# 行程活动通常按5或10分钟取整；小于等于该范围的差异不升级为重规划。
TIME_TOLERANCE_MINUTES = 10


def _to_minutes(hour: str, minute: str) -> int:
    return int(hour) * 60 + int(minute)


def _parse_primary_range(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, str):
        return None
    match = TIME_RANGE_PATTERN.search(value)
    if not match:
        return None
    return (
        _to_minutes(match.group(1), match.group(2)),
        _to_minutes(match.group(3), match.group(4)),
    )


def _find_ranges(value: Any) -> List[Tuple[int, int, str]]:
    if not isinstance(value, str):
        return []
    ranges = []
    for match in TIME_RANGE_PATTERN.finditer(value):
        ranges.append((
            _to_minutes(match.group(1), match.group(2)),
            _to_minutes(match.group(3), match.group(4)),
            match.group(0),
        ))
    return ranges


def _transport_text(activity: Dict[str, Any]) -> str:
    return " ".join(
        str(activity.get(field, ""))
        for field in ("location", "description", "transport")
    )


def _extract_transport_duration_minutes(text: str) -> Optional[int]:
    if not any(keyword in text for keyword in TRANSPORT_KEYWORDS):
        return None

    hour_match = HOUR_DURATION_PATTERN.search(text)
    if hour_match:
        hours = float(hour_match.group(1))
        extra_minutes = int(hour_match.group(2) or 0)
        return round(hours * 60 + extra_minutes)

    for approximate_match in APPROXIMATE_HOUR_DURATION_PATTERN.finditer(text):
        prefix = text[max(0, approximate_match.start() - 6):approximate_match.start()]
        if "提前" in prefix:
            continue
        hours = float(approximate_match.group(1))
        extra_minutes = int(approximate_match.group(2) or 0)
        return round(hours * 60 + extra_minutes)

    minute_match = MINUTE_DURATION_PATTERN.search(text)
    if minute_match:
        return int(minute_match.group(1))
    return None


def _required_departure_buffer_minutes(
    activity: Dict[str, Any],
) -> Optional[int]:
    """返回固定班次交通的规划缓冲；当前仅覆盖评估确认的铁路场景。"""
    transport = str(activity.get("transport", ""))
    description = str(activity.get("description", ""))
    if any(keyword in transport for keyword in RAIL_TRANSPORT_KEYWORDS):
        return RAIL_DEPARTURE_BUFFER_MINUTES
    if any(
        keyword in description
        for keyword in DEPARTURE_BUFFER_ACTIVITY_KEYWORDS
    ):
        return None
    if RAIL_BOARDING_PATTERN.search(description):
        return RAIL_DEPARTURE_BUFFER_MINUTES
    return None


def _dedicated_departure_buffer_minutes(
    activity: Optional[Dict[str, Any]],
    slot_minutes: int,
) -> Optional[int]:
    """返回上一活动中可确定的安检、检票和候车分钟数。"""
    if not isinstance(activity, dict):
        return None
    text = _transport_text(activity)
    if not any(
        keyword in text
        for keyword in DEPARTURE_BUFFER_ACTIVITY_KEYWORDS
    ):
        return None

    explicit_match = EXPLICIT_BUFFER_MINUTES_PATTERN.search(text)
    if explicit_match:
        return min(slot_minutes, int(explicit_match.group(1)))

    if any(
        keyword in text
        for keyword in TRAVEL_TO_TERMINAL_KEYWORDS
    ):
        return None
    return slot_minutes


def _issue(
    category: str,
    day_index: int,
    activity_index: int,
    message: str,
    **details: Any,
) -> Dict[str, Any]:
    issue = {
        "category": category,
        "day_index": day_index,
        "activity_index": activity_index,
        "message": message,
    }
    issue.update(details)
    return issue


def find_itinerary_time_feasibility_issues(
    itinerary_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """返回可以由结构化时间、明确数字和固定缓冲规则证明的问题。"""
    itinerary = itinerary_result.get("itinerary", {})
    if not isinstance(itinerary, dict):
        return []
    daily_plans = itinerary.get("daily_plans", [])
    if not isinstance(daily_plans, list):
        return []

    issues: List[Dict[str, Any]] = []
    for day_index, day_plan in enumerate(daily_plans):
        if not isinstance(day_plan, dict):
            continue
        activities = day_plan.get("activities", [])
        if not isinstance(activities, list):
            continue

        previous_range: Optional[Tuple[int, int]] = None
        previous_index: Optional[int] = None
        previous_activity: Optional[Dict[str, Any]] = None
        for activity_index, activity in enumerate(activities):
            if not isinstance(activity, dict):
                continue
            slot_text = activity.get("time")
            slot = _parse_primary_range(slot_text)
            if slot is None:
                continue
            slot_start, slot_end = slot

            if slot_end <= slot_start:
                issues.append(_issue(
                    "invalid_time_range",
                    day_index,
                    activity_index,
                    f"活动时间“{slot_text}”的结束时间不晚于开始时间。",
                ))
                previous_range = None
                previous_index = None
                previous_activity = None
                continue

            if (
                previous_range is not None
                and previous_index is not None
                and slot_start < previous_range[1]
            ):
                issues.append(_issue(
                    "overlapping_activities",
                    day_index,
                    activity_index,
                    (
                        f"第{previous_index + 1}项活动结束于"
                        f"{previous_range[1] // 60:02d}:{previous_range[1] % 60:02d}，"
                        f"但第{activity_index + 1}项活动开始于"
                        f"{slot_start // 60:02d}:{slot_start % 60:02d}。"
                    ),
                ))

            required_buffer = _required_departure_buffer_minutes(activity)
            if (
                required_buffer is not None
                and (
                    previous_range is None
                    or slot_start >= previous_range[1]
                )
            ):
                actual_buffer = 0
                if previous_range is not None:
                    actual_buffer = slot_start - previous_range[1]
                    dedicated_buffer = _dedicated_departure_buffer_minutes(
                        previous_activity,
                        previous_range[1] - previous_range[0],
                    )
                    if dedicated_buffer is not None:
                        actual_buffer += dedicated_buffer
                if actual_buffer < required_buffer:
                    issues.append(_issue(
                        "insufficient_departure_buffer",
                        day_index,
                        activity_index,
                        (
                            f"第{activity_index + 1}项铁路交通开始于"
                            f"{slot_start // 60:02d}:{slot_start % 60:02d}，"
                            f"但此前只明确预留了{actual_buffer}分钟用于进站、"
                            f"安检、检票和候车；规划规则要求至少预留"
                            f"{required_buffer}分钟。"
                        ),
                        previous_activity_index=previous_index,
                        required_buffer_minutes=required_buffer,
                        actual_buffer_minutes=actual_buffer,
                        transport_type="rail",
                    ))

            transport_text = _transport_text(activity)
            has_transport_context = any(
                keyword in transport_text
                for keyword in TRANSPORT_KEYWORDS
            )
            if has_transport_context:
                for embedded_start, embedded_end, embedded_text in _find_ranges(
                    transport_text
                ):
                    if embedded_end <= embedded_start:
                        continue
                    if (
                        embedded_start < slot_start - TIME_TOLERANCE_MINUTES
                        or embedded_end > slot_end + TIME_TOLERANCE_MINUTES
                    ):
                        issues.append(_issue(
                            "transport_time_outside_activity",
                            day_index,
                            activity_index,
                            (
                                f"活动时间为“{slot_text}”，但交通描述包含"
                                f"“{embedded_text}”，没有被该时间框完整覆盖。"
                            ),
                        ))
                        break

            duration_minutes = _extract_transport_duration_minutes(
                transport_text
            )
            slot_minutes = slot_end - slot_start
            if (
                duration_minutes is not None
                and duration_minutes > slot_minutes + TIME_TOLERANCE_MINUTES
            ):
                issues.append(_issue(
                    "transport_duration_exceeds_activity",
                    day_index,
                    activity_index,
                    (
                        f"活动时间“{slot_text}”只有{slot_minutes}分钟，"
                        f"但交通描述写明约{duration_minutes}分钟。"
                    ),
                ))

            previous_range = slot
            previous_index = activity_index
            previous_activity = activity

    return issues


def find_itinerary_time_issues(
    itinerary_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """兼容旧调用；新代码应使用更准确的时间可行性入口。"""
    return find_itinerary_time_feasibility_issues(itinerary_result)
