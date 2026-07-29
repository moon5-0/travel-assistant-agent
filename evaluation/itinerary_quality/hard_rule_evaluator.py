#!/usr/bin/env python3
"""行程质量第一阶段：使用确定性规则检查硬约束。

这里只检查可以由代码稳定判断的事实和结构，不评价路线是否舒适、
推荐是否精彩等主观质量；这些内容由后续 LLM Judge 负责。
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_CASES_PATH = Path(__file__).with_name("itinerary_quality_cases.json")
FATAL_TRIP_FIELDS = {
    "origin",
    "destination",
    "start_date",
    "end_date",
    "duration_days",
    "city_order",
}


class DatasetValidationError(ValueError):
    """行程质量数据集结构不合法。"""


def _require_dict(value: Any, location: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{location} must be an object")
    return value


def _require_list(value: Any, location: str) -> List[Any]:
    if not isinstance(value, list):
        raise DatasetValidationError(f"{location} must be a list")
    return value


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{location} must be a non-empty string")
    return value


def _require_string_list(value: Any, location: str) -> List[str]:
    values = _require_list(value, location)
    if not all(isinstance(item, str) and item.strip() for item in values):
        raise DatasetValidationError(f"{location} must contain strings")
    return values


def _validate_content_rule(
    rule: Any,
    location: str,
    *,
    forbidden: bool,
) -> None:
    data = _require_dict(rule, location)
    _require_string(data.get("id"), f"{location}.id")
    _require_string(data.get("description"), f"{location}.description")

    if forbidden:
        _require_string_list(data.get("patterns"), f"{location}.patterns")
    else:
        modes = [name for name in ("any_of", "all_of") if name in data]
        if len(modes) != 1:
            raise DatasetValidationError(
                f"{location} must define exactly one of any_of or all_of"
            )
        _require_string_list(data[modes[0]], f"{location}.{modes[0]}")

    if "fatal" in data and not isinstance(data["fatal"], bool):
        raise DatasetValidationError(f"{location}.fatal must be a boolean")


def validate_dataset(dataset: Any) -> Dict[str, Any]:
    """校验测试集结构，防止错误的“标准答案”进入评估。"""
    data = _require_dict(dataset, "dataset")
    _require_string(data.get("version"), "dataset.version")
    global_rules = _require_dict(
        data.get("global_rules"),
        "dataset.global_rules",
    )
    _require_string_list(
        global_rules.get("unsupported_confirmation_patterns"),
        "dataset.global_rules.unsupported_confirmation_patterns",
    )
    _require_string_list(
        global_rules.get("minimum_daily_plan_fields"),
        "dataset.global_rules.minimum_daily_plan_fields",
    )

    cases = _require_list(data.get("cases"), "dataset.cases")
    if not cases:
        raise DatasetValidationError("dataset.cases must not be empty")

    case_ids = []
    for index, raw_case in enumerate(cases):
        location = f"dataset.cases[{index}]"
        case = _require_dict(raw_case, location)
        case_id = _require_string(case.get("id"), f"{location}.id")
        case_ids.append(case_id)
        _require_string(case.get("name"), f"{location}.name")

        case_input = _require_dict(case.get("input"), f"{location}.input")
        _require_string(
            case_input.get("user_query"),
            f"{location}.input.user_query",
        )
        _require_dict(
            case_input.get("trip_info"),
            f"{location}.input.trip_info",
        )
        _require_dict(
            case_input.get("user_preferences"),
            f"{location}.input.user_preferences",
        )
        _require_list(
            case_input.get("external_information"),
            f"{location}.input.external_information",
        )

        expected = _require_dict(
            case.get("expected"),
            f"{location}.expected",
        )
        required_trip = _require_dict(
            expected.get("required_trip"),
            f"{location}.expected.required_trip",
        )
        if not required_trip:
            raise DatasetValidationError(
                f"{location}.expected.required_trip must not be empty"
            )

        required_content = _require_list(
            expected.get("required_content"),
            f"{location}.expected.required_content",
        )
        for rule_index, rule in enumerate(required_content):
            _validate_content_rule(
                rule,
                f"{location}.expected.required_content[{rule_index}]",
                forbidden=False,
            )

        forbidden_content = _require_list(
            expected.get("forbidden_content"),
            f"{location}.expected.forbidden_content",
        )
        for rule_index, rule in enumerate(forbidden_content):
            _validate_content_rule(
                rule,
                f"{location}.expected.forbidden_content[{rule_index}]",
                forbidden=True,
            )

        _require_string_list(
            expected.get("judge_focus"),
            f"{location}.expected.judge_focus",
        )
        if "allowed_confirmation_patterns" in expected:
            _require_string_list(
                expected["allowed_confirmation_patterns"],
                f"{location}.expected.allowed_confirmation_patterns",
            )

    if len(case_ids) != len(set(case_ids)):
        raise DatasetValidationError("dataset contains duplicate case id")
    return data


def load_dataset(path: Path = DEFAULT_CASES_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return validate_dataset(json.load(file))


def summarize_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": dataset["version"],
        "case_count": len(dataset["cases"]),
        "case_ids": [case["id"] for case in dataset["cases"]],
    }


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def _date_variants(value: str) -> List[str]:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return [value]
    return [
        parsed.isoformat(),
        f"{parsed.year}-{parsed.month}-{parsed.day}",
        f"{parsed.year}/{parsed.month}/{parsed.day}",
        f"{parsed.year}.{parsed.month}.{parsed.day}",
        f"{parsed.year}年{parsed.month}月{parsed.day}日",
    ]


def _contains_token(text: str, token: Any) -> bool:
    candidates = _date_variants(token) if isinstance(token, str) else [token]
    return any(_normalized(candidate) in text for candidate in candidates)


def _assertive_pattern_matches(text: str, patterns: Iterable[str]) -> List[str]:
    """只返回肯定出现的禁用表达，忽略“不要预订”等否定说明。"""
    matches = []
    negation_before_pattern = re.compile(
        r"(?:不|不要|不得|避免|禁止|不会|无需|未|没有|尚未|请勿|无法|"
        r"无法确认|不能确认|不建议|不推荐)[^，。；;！？!?]{0,8}$"
    )
    for pattern in patterns:
        normalized_pattern = _normalized(pattern)
        cursor = 0
        while True:
            position = text.find(normalized_pattern, cursor)
            if position < 0:
                break
            prefix = text[max(0, position - 24):position]
            if not negation_before_pattern.search(prefix):
                matches.append(pattern)
                break
            cursor = position + len(normalized_pattern)
    return matches


def _canonical_date(value: Any) -> Optional[str]:
    text = str(value).strip()
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


def _expected_dates(required_trip: Dict[str, Any]) -> List[str]:
    start_text = required_trip.get("start_date")
    duration = required_trip.get("duration_days")
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


def _contains_in_order(text: str, values: Iterable[Any]) -> bool:
    cursor = 0
    for value in values:
        token = _normalized(value)
        position = text.find(token, cursor)
        if position < 0:
            return False
        cursor = position + len(token)
    return True


def _location_parts(value: str) -> List[str]:
    return [
        part.strip()
        for part in re.split(r"[、,，]", value)
        if part.strip()
    ]


def _result_from_invalid_output(message: str) -> Dict[str, Any]:
    check = {
        "id": "output.valid_json",
        "group": "structure",
        "passed": False,
        "fatal": True,
        "details": message,
    }
    return {
        "passed": False,
        "hard_constraints_passed": False,
        "fatal_errors": ["output.valid_json"],
        "failures": ["output.valid_json"],
        "summary": {"passed_checks": 0, "failed_checks": 1, "total_checks": 1},
        "checks": [check],
    }


def _parse_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("output must be a JSON object")


def evaluate_case(
    case: Dict[str, Any],
    output: Any,
    global_rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """对一个行程输出执行确定性硬规则检查。"""
    try:
        result = _parse_output(output)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        evaluation = _result_from_invalid_output(str(exc))
        evaluation["case_id"] = case.get("id")
        return evaluation

    rules = global_rules or {}
    expected = case["expected"]
    required_trip = expected["required_trip"]
    checks: List[Dict[str, Any]] = []

    def add_check(
        check_id: str,
        group: str,
        passed: bool,
        *,
        fatal: bool = False,
        expected_value: Any = None,
        actual_value: Any = None,
        details: Any = None,
    ) -> None:
        check = {
            "id": check_id,
            "group": group,
            "passed": bool(passed),
            "fatal": bool(fatal),
        }
        if expected_value is not None:
            check["expected"] = expected_value
        if actual_value is not None:
            check["actual"] = actual_value
        if details is not None:
            check["details"] = details
        checks.append(check)

    itinerary = result.get("itinerary")
    add_check(
        "output.itinerary_object",
        "structure",
        isinstance(itinerary, dict),
        fatal=True,
        actual_value=type(itinerary).__name__,
    )
    add_check(
        "output.planning_complete",
        "structure",
        result.get("planning_complete") is True,
        fatal=True,
        expected_value=True,
        actual_value=result.get("planning_complete"),
    )

    if not isinstance(itinerary, dict):
        failures = [check["id"] for check in checks if not check["passed"]]
        fatal_errors = [
            check["id"]
            for check in checks
            if not check["passed"] and check["fatal"]
        ]
        return {
            "case_id": case.get("id"),
            "passed": False,
            "hard_constraints_passed": False,
            "fatal_errors": fatal_errors,
            "failures": failures,
            "summary": {
                "passed_checks": len(checks) - len(failures),
                "failed_checks": len(failures),
                "total_checks": len(checks),
            },
            "checks": checks,
        }

    daily_plans = itinerary.get("daily_plans")
    daily_plans_valid = isinstance(daily_plans, list) and bool(daily_plans)
    add_check(
        "output.daily_plans",
        "structure",
        daily_plans_valid,
        fatal=True,
        actual_value=(len(daily_plans) if isinstance(daily_plans, list) else None),
        details="daily_plans必须是非空数组",
    )
    daily_plans = daily_plans if isinstance(daily_plans, list) else []

    expected_duration = required_trip.get("duration_days")
    if isinstance(expected_duration, int):
        add_check(
            "required_trip.duration_days",
            "required_trip",
            len(daily_plans) == expected_duration,
            fatal=True,
            expected_value=expected_duration,
            actual_value=len(daily_plans),
        )

    minimum_fields = rules.get(
        "minimum_daily_plan_fields",
        ["day", "date", "city", "activities"],
    )
    invalid_daily_plans = []
    for index, plan in enumerate(daily_plans):
        if not isinstance(plan, dict):
            invalid_daily_plans.append({"index": index, "reason": "not_object"})
            continue
        missing = [
            field
            for field in minimum_fields
            if field not in plan or plan[field] in (None, "", [])
        ]
        if not isinstance(plan.get("activities"), list) or not plan.get("activities"):
            if "activities" not in missing:
                missing.append("activities")
        if missing:
            invalid_daily_plans.append({"index": index, "missing": missing})
    add_check(
        "output.daily_plan_structure",
        "structure",
        not invalid_daily_plans and bool(daily_plans),
        fatal=True,
        details=invalid_daily_plans or "每一天均包含最小必需字段",
    )

    expected_dates = _expected_dates(required_trip)
    if expected_dates:
        actual_dates = [
            _canonical_date(plan.get("date")) if isinstance(plan, dict) else None
            for plan in daily_plans
        ]
        add_check(
            "required_trip.daily_plan_dates",
            "required_trip",
            actual_dates == expected_dates,
            fatal=True,
            expected_value=expected_dates,
            actual_value=actual_dates,
        )

    itinerary_text = _normalized(
        json.dumps(itinerary, ensure_ascii=False, sort_keys=True)
    )
    route_text = _normalized(itinerary.get("route", "")) or itinerary_text

    for field, value in required_trip.items():
        if field == "duration_days":
            continue
        if field in {"start_date", "end_date"}:
            actual_dates = [
                _canonical_date(plan.get("date")) if isinstance(plan, dict) else None
                for plan in daily_plans
            ]
            passed = value in actual_dates
            actual_value = actual_dates
        elif field == "city_order" and isinstance(value, list):
            passed = _contains_in_order(route_text, value)
            actual_value = itinerary.get("route")
        elif field == "destination" and isinstance(value, str):
            parts = _location_parts(value)
            passed = all(_contains_token(itinerary_text, part) for part in parts)
            actual_value = parts
        else:
            passed = _contains_token(itinerary_text, value)
            actual_value = None
        add_check(
            f"required_trip.{field}",
            "required_trip",
            passed,
            fatal=field in FATAL_TRIP_FIELDS,
            expected_value=value,
            actual_value=actual_value,
        )

    for rule in expected["required_content"]:
        if "any_of" in rule:
            matched = [
                token
                for token in rule["any_of"]
                if _contains_token(itinerary_text, token)
            ]
            passed = bool(matched)
            details = {
                "mode": "any_of",
                "matched": matched,
                "candidates": rule["any_of"],
            }
        else:
            missing = [
                token
                for token in rule["all_of"]
                if not _contains_token(itinerary_text, token)
            ]
            passed = not missing
            details = {
                "mode": "all_of",
                "missing": missing,
                "required": rule["all_of"],
            }
        add_check(
            f"required_content.{rule['id']}",
            "required_content",
            passed,
            fatal=rule.get("fatal", False),
            details=details,
        )

    for rule in expected["forbidden_content"]:
        matched = _assertive_pattern_matches(
            itinerary_text,
            rule["patterns"],
        )
        add_check(
            f"forbidden_content.{rule['id']}",
            "forbidden_content",
            not matched,
            fatal=rule.get("fatal", False),
            details={"matched": matched},
        )

    allowed_confirmations = set(
        expected.get("allowed_confirmation_patterns", [])
    )
    unsupported = [
        pattern
        for pattern in _assertive_pattern_matches(
            itinerary_text,
            rules.get("unsupported_confirmation_patterns", []),
        )
        if pattern not in allowed_confirmations
    ]
    add_check(
        "global.unsupported_confirmation",
        "forbidden_content",
        not unsupported,
        fatal=True,
        details={
            "matched": unsupported,
            "allowed": sorted(allowed_confirmations),
        },
    )

    failures = [check["id"] for check in checks if not check["passed"]]
    fatal_errors = [
        check["id"]
        for check in checks
        if not check["passed"] and check["fatal"]
    ]
    return {
        "case_id": case.get("id"),
        "passed": not failures,
        "hard_constraints_passed": not failures,
        "fatal_errors": fatal_errors,
        "failures": failures,
        "summary": {
            "passed_checks": len(checks) - len(failures),
            "failed_checks": len(failures),
            "total_checks": len(checks),
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate or run deterministic itinerary quality checks",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case", help="要评分的case id")
    parser.add_argument("--output", type=Path, help="待评分的行程JSON")
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.cases)
    except (DatasetValidationError, OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}")
        return 2

    if not args.output:
        print(json.dumps(summarize_dataset(dataset), ensure_ascii=False, indent=2))
        print("VALID")
        return 0
    if not args.case:
        print("INVALID: 使用--output时必须同时提供--case")
        return 2

    case = next(
        (item for item in dataset["cases"] if item["id"] == args.case),
        None,
    )
    if case is None:
        print(f"INVALID: 未知测试场景 {args.case}")
        return 2

    try:
        with args.output.open("r", encoding="utf-8") as file:
            output = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}")
        return 2

    evaluation = evaluate_case(case, output, dataset["global_rules"])
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    return 0 if evaluation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
