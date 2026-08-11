"""规划结果输出前的统一质量门。

只把无法安全交付的确定性问题作为 blocking issue；偏好体现、措辞和
内容丰富度等软质量留给评估系统，不在生产链路中反复调用模型追求完美。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta
import json
import re
from typing import Any, Dict, Iterable, List, Optional

from utils.booking_context import (
    find_booking_reference_issues,
    render_booking_references,
)
from utils.itinerary_time_validator import (
    find_itinerary_time_feasibility_issues,
)


MINIMUM_DAILY_PLAN_FIELDS = ("day", "date", "city", "activities")
ACTIVITY_TYPES = {
    "general",
    "transport_booking",
    "hotel_booking",
    "fixed_event",
    "business",
    "meal",
    "leisure",
    "buffer",
    "local_transport",
}
ACTIVITY_TYPE_ALIASES = {
    "transport": "local_transport",
    "transportation": "local_transport",
    "hotel": "hotel_booking",
    "meeting": "business",
    "dining": "meal",
    "sightseeing": "leisure",
}
DAILY_ACTIVITY_ALIASES = ("items", "plans", "schedule")
TIME_RANGE_PATTERN = re.compile(
    r"^\s*([01]?\d|2[0-3]):([0-5]\d)\s*[-—~～至到]\s*"
    r"([01]?\d|2[0-3]):([0-5]\d)\s*$"
)
DETERMINISTIC_TIME_BLOCKERS = {
    "invalid_time_range",
    "overlapping_activities",
    "transport_time_outside_activity",
    "transport_duration_exceeds_activity",
}
NEGATION_PATTERN = re.compile(
    r"(?:不|不要|不得|避免|禁止|不会|无需|未|没有|尚未|请勿|"
    r"不建议|不推荐)[^，。；;！？!?]{0,10}$"
)


def _issue(
    category: str,
    source: str,
    message: str,
    severity: str = "blocking",
    **details: Any,
) -> Dict[str, Any]:
    result = {
        "category": category,
        "source": source,
        "severity": severity,
        "message": message,
    }
    result.update(details)
    return result


def _warning(
    category: str,
    source: str,
    message: str,
    **details: Any,
) -> Dict[str, Any]:
    return _issue(
        category,
        source,
        message,
        severity="warning",
        **details,
    )


def _canonical_date(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    for pattern in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",
        "%Y年%m月%d日",
    ):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            continue
    return None


def _expected_dates(event_data: Dict[str, Any]) -> List[str]:
    start_text = event_data.get("start_date")
    duration = event_data.get("duration_days")
    if not isinstance(start_text, str) or not isinstance(duration, int):
        return []
    try:
        start = date.fromisoformat(start_text)
    except ValueError:
        return []
    return [
        (start + timedelta(days=offset)).isoformat()
        for offset in range(duration)
    ]


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _contains(text: str, value: Any) -> bool:
    normalized = _normalized(value)
    return bool(normalized) and normalized in text


def _contains_date(text: str, value: Any) -> bool:
    canonical = _canonical_date(value)
    if not canonical:
        return _contains(text, value)
    parsed = date.fromisoformat(canonical)
    variants = (
        canonical,
        f"{parsed.year}-{parsed.month}-{parsed.day}",
        f"{parsed.year}/{parsed.month}/{parsed.day}",
        f"{parsed.year}年{parsed.month}月{parsed.day}日",
    )
    return any(_contains(text, item) for item in variants)


def _contains_in_order(text: str, values: Iterable[Any]) -> bool:
    cursor = 0
    for value in values:
        token = _normalized(value)
        position = text.find(token, cursor)
        if position < 0:
            return False
        cursor = position + len(token)
    return True


def _assertively_mentions(text: str, token: str) -> bool:
    normalized_token = _normalized(token)
    cursor = 0
    while normalized_token:
        position = text.find(normalized_token, cursor)
        if position < 0:
            return False
        prefix = text[max(0, position - 24):position]
        if not NEGATION_PATTERN.search(prefix):
            return True
        cursor = position + len(normalized_token)
    return False


def _location_matches(text: str, value: Any) -> bool:
    """地点允许省市/区域词分开出现，避免把“上海+浦东”误判为缺失。"""
    normalized = _normalized(value)
    if not normalized:
        return False
    if normalized in text:
        return True
    if len(normalized) >= 4:
        return normalized[:2] in text and normalized[-2:] in text
    return False


def _split_destinations(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [
            item.strip()
            for item in re.split(r"[、,，]", value)
            if item.strip()
        ]
    return []


def prepare_event_data_for_planning(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """为固定事项补充稳定引用ID，供模型引用、代码校验和可信渲染。"""
    prepared = deepcopy(event_data) if isinstance(event_data, dict) else {}
    fixed_events = prepared.get("fixed_events")
    if not isinstance(fixed_events, list):
        return prepared
    for index, item in enumerate(fixed_events):
        if not isinstance(item, dict):
            continue
        if not str(item.get("event_id") or "").strip():
            item["event_id"] = f"fixed_event_{index + 1}"
    return prepared


def _fixed_event_map(event_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item["event_id"]): item
        for item in event_data.get("fixed_events") or []
        if isinstance(item, dict) and item.get("event_id")
    }


def _iter_activity_records(
    result: Dict[str, Any],
) -> Iterable[tuple[int, Dict[str, Any], int, Any]]:
    itinerary = result.get("itinerary", {})
    if not isinstance(itinerary, dict):
        return
    daily_plans = itinerary.get("daily_plans", [])
    if not isinstance(daily_plans, list):
        return
    for day_index, plan in enumerate(daily_plans):
        if not isinstance(plan, dict):
            continue
        activities = plan.get("activities")
        if not isinstance(activities, list):
            continue
        for activity_index, activity in enumerate(activities):
            yield day_index, plan, activity_index, activity


def _normalize_activity(activity: Dict[str, Any]) -> None:
    """只统一可确定字段；不通过关键词猜测活动语义。"""
    activity_type = str(activity.get("type") or "").strip()
    if activity.get("fixed_event_ref"):
        activity_type = "fixed_event"
        activity["type"] = activity_type
    elif activity.get("booking_ref") == "hotel":
        activity_type = "hotel_booking"
        activity["type"] = activity_type
    elif activity.get("booking_ref") in {"outbound", "return"}:
        activity_type = "transport_booking"
        activity["type"] = activity_type
    elif activity_type in ACTIVITY_TYPE_ALIASES:
        activity_type = ACTIVITY_TYPE_ALIASES[activity_type]
        activity["type"] = activity_type

    if not activity_type:
        activity["type"] = "general"

    # activity/name 是模型常见的标题别名，属于字段归一化而非语义猜测。
    if not str(activity.get("title") or "").strip():
        for alias in ("activity", "name"):
            if str(activity.get(alias) or "").strip():
                activity["title"] = activity[alias]
                break

    time_text = activity.get("time")
    match = TIME_RANGE_PATTERN.fullmatch(str(time_text or ""))
    if match:
        activity.setdefault(
            "start_time",
            f"{int(match.group(1)):02d}:{match.group(2)}",
        )
        activity.setdefault(
            "end_time",
            f"{int(match.group(3)):02d}:{match.group(4)}",
        )
    elif (
        not str(time_text or "").strip()
        and activity.get("start_time")
        and activity.get("end_time")
    ):
        activity["time"] = (
            f"{activity['start_time']}-{activity['end_time']}"
        )


def _render_fixed_event_references(
    result: Dict[str, Any],
    event_data: Dict[str, Any],
) -> None:
    """固定事项的时间、地点和标题始终来自事项收集结果。"""
    fixed_events = _fixed_event_map(event_data)
    if not fixed_events:
        return
    for _, _, _, activity in _iter_activity_records(result):
        if not isinstance(activity, dict):
            continue
        event = fixed_events.get(str(activity.get("fixed_event_ref") or ""))
        if event is None:
            continue
        activity["type"] = "fixed_event"
        if event.get("time"):
            activity["time"] = event["time"]
        if event.get("location"):
            activity["location"] = event["location"]
        if event.get("title"):
            activity["title"] = event["title"]
            activity.setdefault("description", event["title"])
        _normalize_activity(activity)


def _infer_plan_city(
    plan: Dict[str, Any],
    event_data: Dict[str, Any],
) -> Optional[str]:
    """只根据结构化行程信息补齐展示字段，不调用模型猜测新城市。"""
    for field in ("city", "destination"):
        value = plan.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    text = _normalized(json.dumps(plan, ensure_ascii=False, sort_keys=True))
    candidates = []
    city_order = event_data.get("city_order")
    if isinstance(city_order, list):
        candidates.extend(str(item) for item in city_order)
    candidates.extend(_split_destinations(event_data.get("destination")))
    origin = event_data.get("origin")
    if isinstance(origin, str):
        candidates.append(origin)

    for candidate in candidates:
        if _contains(text, candidate):
            return candidate
    destinations = _split_destinations(event_data.get("destination"))
    if len(destinations) == 1:
        return destinations[0]
    if isinstance(origin, str) and origin.strip():
        return origin.strip()
    return None


def normalize_itinerary_result(
    result: Dict[str, Any],
    booking_context: Dict[str, Dict[str, Any]],
    event_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """统一常见字段别名，并由可信上下文确定性生成预订字段。"""
    normalized = deepcopy(result)
    for field in ("quality_gate", "fact_grounding", "time_consistency"):
        normalized.pop(field, None)

    event_data = event_data or {}
    itinerary = normalized.get("itinerary")
    if isinstance(itinerary, dict):
        if not str(itinerary.get("route") or "").strip():
            city_order = event_data.get("city_order")
            if isinstance(city_order, list) and city_order:
                route_parts = [str(item) for item in city_order if str(item)]
            else:
                route_parts = []
                origin = event_data.get("origin")
                if origin:
                    route_parts.append(str(origin))
                route_parts.extend(
                    _split_destinations(event_data.get("destination"))
                )
                return_location = event_data.get("return_location")
                if return_location:
                    route_parts.append(str(return_location))
            if route_parts:
                itinerary["route"] = " -> ".join(route_parts)

        daily_plans = itinerary.get("daily_plans")
        expected_dates = _expected_dates(event_data)
        if isinstance(daily_plans, list):
            for index, plan in enumerate(daily_plans):
                if not isinstance(plan, dict):
                    continue
                activities = plan.get("activities")
                if not isinstance(activities, list) or not activities:
                    for alias in DAILY_ACTIVITY_ALIASES:
                        alias_value = plan.get(alias)
                        if isinstance(alias_value, list) and alias_value:
                            # 接受模型常见字段别名，但不凭空补写活动内容。
                            plan["activities"] = alias_value
                            break
                day_value = plan.get("day")
                if day_value in (None, ""):
                    day_value = plan.get("day_number", index + 1)
                if isinstance(day_value, str) and day_value.isdigit():
                    day_value = int(day_value)
                plan["day"] = day_value
                if plan.get("date") in (None, "") and index < len(expected_dates):
                    plan["date"] = expected_dates[index]
                if plan.get("city") in (None, ""):
                    inferred_city = _infer_plan_city(plan, event_data)
                    if inferred_city:
                        plan["city"] = inferred_city
                activities = plan.get("activities")
                if isinstance(activities, list):
                    for activity in activities:
                        if isinstance(activity, dict):
                            _normalize_activity(activity)

        _render_fixed_event_references(normalized, event_data)

    if booking_context:
        normalized = render_booking_references(normalized, booking_context)
    return normalized


def _structure_issues(
    result: Dict[str, Any],
    event_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    itinerary = result.get("itinerary")
    if not isinstance(itinerary, dict):
        return [_issue(
            "invalid_itinerary",
            "structure",
            "根对象中的itinerary必须是对象。",
        )]

    daily_plans = itinerary.get("daily_plans")
    if not isinstance(daily_plans, list) or not daily_plans:
        issues.append(_issue(
            "empty_daily_plans",
            "structure",
            "daily_plans必须是非空数组。",
        ))
        daily_plans = daily_plans if isinstance(daily_plans, list) else []

    expected_duration = event_data.get("duration_days")
    if (
        isinstance(expected_duration, int)
        and len(daily_plans) != expected_duration
    ):
        issues.append(_issue(
            "duration_mismatch",
            "structure",
            f"用户要求{expected_duration}天，但实际生成{len(daily_plans)}天。",
            expected=expected_duration,
            actual=len(daily_plans),
        ))

    invalid_days = []
    invalid_activities = []
    for index, plan in enumerate(daily_plans):
        if not isinstance(plan, dict):
            invalid_days.append({"index": index, "reason": "not_object"})
            continue
        missing = [
            field
            for field in MINIMUM_DAILY_PLAN_FIELDS
            if field not in plan or plan[field] in (None, "", [])
        ]
        if not isinstance(plan.get("activities"), list) or not plan.get("activities"):
            if "activities" not in missing:
                missing.append("activities")
        if plan.get("day") != index + 1:
            missing.append(f"day应为{index + 1}")
        if missing:
            invalid_days.append({"index": index, "missing": missing})
        activities = plan.get("activities")
        if not isinstance(activities, list):
            continue
        for activity_index, activity in enumerate(activities):
            if not isinstance(activity, dict):
                invalid_activities.append({
                    "day_index": index,
                    "activity_index": activity_index,
                    "reason": "not_object",
                })
                continue
            activity_type = activity.get("type")
            has_time = bool(
                str(activity.get("time") or "").strip()
                or (
                    str(activity.get("start_time") or "").strip()
                    and str(activity.get("end_time") or "").strip()
                )
            )
            has_content = any(
                str(activity.get(field) or "").strip()
                for field in ("title", "location", "description")
            )
            reasons = []
            if (
                activity_type not in (None, "")
                and activity_type not in ACTIVITY_TYPES
            ):
                reasons.append("invalid_type")
            if not has_time:
                reasons.append("missing_time")
            if not has_content:
                reasons.append("missing_content")
            if bool(activity.get("start_time")) != bool(activity.get("end_time")):
                reasons.append("incomplete_time_range")
            if reasons:
                invalid_activities.append({
                    "day_index": index,
                    "activity_index": activity_index,
                    "reasons": reasons,
                })
    if invalid_days:
        issues.append(_issue(
            "invalid_daily_plan_structure",
            "structure",
            "部分日期缺少day、date、city或非空activities。",
            invalid_days=invalid_days,
        ))
    if invalid_activities:
        issues.append(_issue(
            "invalid_activity_structure",
            "structure",
            "部分活动缺少有效类型、时间或可执行内容。",
            invalid_activities=invalid_activities,
        ))

    expected_dates = _expected_dates(event_data)
    if expected_dates:
        actual_dates = [
            _canonical_date(plan.get("date"))
            if isinstance(plan, dict)
            else None
            for plan in daily_plans
        ]
        if actual_dates != expected_dates:
            issues.append(_issue(
                "date_coverage_mismatch",
                "structure",
                "daily_plans没有完整覆盖用户要求的日期。",
                expected=expected_dates,
                actual=actual_dates,
            ))

    return issues


def _structured_reference_issues(
    result: Dict[str, Any],
    event_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """只检查ID、日期等结构化引用，不用关键词推断语义。"""
    issues: List[Dict[str, Any]] = []
    fixed_events = _fixed_event_map(event_data)
    referenced = set()
    for day_index, plan, activity_index, activity in _iter_activity_records(result):
        if not isinstance(activity, dict):
            continue
        event_ref = activity.get("fixed_event_ref")
        if event_ref is None:
            continue
        event_ref = str(event_ref)
        event = fixed_events.get(event_ref)
        if event is None:
            issues.append(_issue(
                "unknown_fixed_event_ref",
                "constraint",
                "活动引用了不存在的固定事项。",
                day_index=day_index,
                activity_index=activity_index,
                fixed_event_ref=event_ref,
            ))
            continue
        referenced.add(event_ref)
        if activity.get("type") != "fixed_event":
            issues.append(_issue(
                "fixed_event_type_mismatch",
                "constraint",
                "固定事项引用必须使用type=fixed_event。",
                fixed_event_ref=event_ref,
            ))
        expected_date = _canonical_date(event.get("date"))
        actual_date = _canonical_date(plan.get("date"))
        if expected_date and actual_date != expected_date:
            issues.append(_issue(
                "fixed_event_date_mismatch",
                "constraint",
                "固定事项被安排在错误日期。",
                fixed_event_ref=event_ref,
                expected=expected_date,
                actual=actual_date,
            ))

    for event_ref in fixed_events.keys() - referenced:
        issues.append(_issue(
            "fixed_event_not_referenced",
            "constraint",
            "输入中的固定事项没有进入结构化行程。",
            fixed_event_ref=event_ref,
        ))

    return issues


def collect_itinerary_quality_issues(
    result: Dict[str, Any],
    event_data: Dict[str, Any],
    booking_context: Dict[str, Dict[str, Any]],
    trusted_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """统一收集问题；只有确定导致行程不可交付的问题才标记为阻断。"""
    event_data = prepare_event_data_for_planning(event_data)
    issues = [
        *_structure_issues(result, event_data),
        *_structured_reference_issues(result, event_data),
    ]

    for item in find_booking_reference_issues(result, booking_context):
        issues.append(_issue(
            item.get("category", "booking_reference_issue"),
            "booking",
            item.get("message", "预订引用与可信状态不一致。"),
            details=item,
        ))
    for item in find_itinerary_time_feasibility_issues(result):
        category = item.get("category", "time_feasibility_issue")
        if category in DETERMINISTIC_TIME_BLOCKERS:
            issues.append(_issue(
                category,
                "time",
                item.get("message", "行程存在确定性时间问题。"),
                details=item,
            ))
    return issues


def quality_issue_score(issues: List[Dict[str, Any]]) -> int:
    """用于避免修复结果用更严重的结构问题覆盖原候选。"""
    weights = {
        "structure": 10,
        "constraint": 9,
        "booking": 7,
        "fact": 6,
        "time": 6,
    }
    return sum(
        weights.get(item.get("source"), 5)
        for item in issues
        if item.get("severity") == "blocking"
    )


def finalize_quality_gate(
    result: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """阻断问题决定能否交付；警告只保留给后续质量评估。"""
    finalized = deepcopy(result)
    blocking_issues = [
        item for item in issues if item.get("severity") == "blocking"
    ]
    warnings = [
        item for item in issues if item.get("severity") != "blocking"
    ]
    if not blocking_issues:
        finalized["planning_complete"] = True
        finalized["quality_gate"] = {
            "status": "passed_with_warnings" if warnings else "passed",
            "blocking_issues": [],
            "warnings": warnings,
        }
        fact_issues = [
            item for item in warnings
            if item.get("source") in {"fact", "booking"}
        ]
        time_issues = [
            item for item in warnings if item.get("source") == "time"
        ]
        if fact_issues:
            finalized["fact_grounding"] = {
                "status": "warning",
                "issues": fact_issues,
            }
        if time_issues:
            finalized["time_consistency"] = {
                "status": "warning",
                "issues": time_issues,
            }
        return finalized

    finalized["planning_complete"] = False
    finalized["quality_gate"] = {
        "status": "unresolved",
        "blocking_issues": blocking_issues,
        "warnings": warnings,
    }
    fact_issues = [
        item for item in warnings
        if item.get("source") in {"fact", "booking"}
    ]
    time_issues = [item for item in warnings if item.get("source") == "time"]
    if fact_issues:
        finalized["fact_grounding"] = {
            "status": "warning",
            "issues": fact_issues,
        }
    if time_issues:
        finalized["time_consistency"] = {
            "status": "warning",
            "issues": time_issues,
        }
    itinerary = finalized.get("itinerary")
    if isinstance(itinerary, dict):
        notes = itinerary.get("notes")
        if not isinstance(notes, list):
            notes = []
            itinerary["notes"] = notes
        warning = "行程仍有未解决的必要信息或可执行性问题，请确认后使用。"
        if warning not in notes:
            notes.append(warning)
    return finalized
