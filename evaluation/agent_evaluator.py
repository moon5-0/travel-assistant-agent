#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent 链路评估器。

v0.1 先实现数据集加载与结构校验，不调用真实 LLM。
后续版本再增加执行轨迹采集、硬性断言和指标汇总。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_CASES_PATH = Path(__file__).with_name("agent_cases.json")
ALLOWED_SEVERITIES = {"critical", "major", "minor"}

REQUIRED_CASE_FIELDS = {
    "id",
    "category",
    "severity",
    "description",
    "initial_state",
    "turns",
}

REQUIRED_EXPECTED_FIELDS = {
    "required_scheduled_agents",
    "required_executed_agents",
    "forbidden_executed_agents",
    "status",
    "entities",
    "missing_fields",
    "memory",
}


class DatasetValidationError(ValueError):
    """评估数据集结构不合法。"""


def _require_dict(value: Any, location: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{location} must be an object")
    return value


def _require_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{location} must be a non-empty string")
    return value


def _require_string_list(value: Any, location: str) -> List[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip()
        for item in value
    ):
        raise DatasetValidationError(f"{location} must be a list of strings")
    return value


def _check_required_fields(
    data: Dict[str, Any],
    required_fields: set[str],
    location: str,
) -> None:
    missing = sorted(required_fields - set(data))
    if missing:
        raise DatasetValidationError(
            f"{location} missing required fields: {', '.join(missing)}"
        )


def _validate_expected(expected: Any, location: str) -> None:
    expected = _require_dict(expected, location)
    _check_required_fields(expected, REQUIRED_EXPECTED_FIELDS, location)

    scheduled = _require_string_list(
        expected["required_scheduled_agents"],
        f"{location}.required_scheduled_agents",
    )
    executed = _require_string_list(
        expected["required_executed_agents"],
        f"{location}.required_executed_agents",
    )
    forbidden = _require_string_list(
        expected["forbidden_executed_agents"],
        f"{location}.forbidden_executed_agents",
    )

    not_scheduled = sorted(set(executed) - set(scheduled))
    if not_scheduled:
        raise DatasetValidationError(
            f"{location} requires executed agents that are not scheduled: "
            f"{', '.join(not_scheduled)}"
        )

    conflict = sorted(set(executed) & set(forbidden))
    if conflict:
        raise DatasetValidationError(
            f"{location} marks agents as both required and forbidden: "
            f"{', '.join(conflict)}"
        )

    _require_string(expected["status"], f"{location}.status")
    _require_dict(expected["entities"], f"{location}.entities")
    _require_string_list(expected["missing_fields"], f"{location}.missing_fields")
    _require_dict(expected["memory"], f"{location}.memory")

    for optional_field in ("must_contain", "must_not_contain_patterns"):
        if optional_field in expected:
            _require_string_list(
                expected[optional_field],
                f"{location}.{optional_field}",
            )

    for index, pattern in enumerate(
        expected.get("must_not_contain_patterns", [])
    ):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise DatasetValidationError(
                f"{location}.must_not_contain_patterns[{index}] "
                f"is invalid: {exc}"
            ) from exc


def validate_dataset(dataset: Any) -> Dict[str, Any]:
    """校验 Agent 评估数据集，并返回原数据。"""
    dataset = _require_dict(dataset, "dataset")
    _require_string(dataset.get("dataset_version"), "dataset.dataset_version")

    default_context = _require_dict(
        dataset.get("default_context"),
        "dataset.default_context",
    )
    _require_string(
        default_context.get("current_date"),
        "dataset.default_context.current_date",
    )
    _require_string(
        default_context.get("timezone"),
        "dataset.default_context.timezone",
    )
    runs_per_case = default_context.get("runs_per_case")
    if not isinstance(runs_per_case, int) or runs_per_case < 1:
        raise DatasetValidationError(
            "dataset.default_context.runs_per_case must be a positive integer"
        )

    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise DatasetValidationError("dataset.cases must be a non-empty list")

    seen_ids = set()
    for case_index, raw_case in enumerate(cases):
        location = f"dataset.cases[{case_index}]"
        case = _require_dict(raw_case, location)
        _check_required_fields(case, REQUIRED_CASE_FIELDS, location)

        case_id = _require_string(case["id"], f"{location}.id")
        if case_id in seen_ids:
            raise DatasetValidationError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)

        _require_string(case["category"], f"{location}.category")
        severity = _require_string(case["severity"], f"{location}.severity")
        if severity not in ALLOWED_SEVERITIES:
            raise DatasetValidationError(
                f"{location}.severity must be one of: "
                f"{', '.join(sorted(ALLOWED_SEVERITIES))}"
            )
        _require_string(case["description"], f"{location}.description")
        _require_dict(case["initial_state"], f"{location}.initial_state")

        turns = case["turns"]
        if not isinstance(turns, list) or not turns:
            raise DatasetValidationError(f"{location}.turns must be non-empty")

        for turn_index, raw_turn in enumerate(turns):
            turn_location = f"{location}.turns[{turn_index}]"
            turn = _require_dict(raw_turn, turn_location)
            _require_string(
                turn.get("user_input"),
                f"{turn_location}.user_input",
            )
            _validate_expected(
                turn.get("expected"),
                f"{turn_location}.expected",
            )

        if "tags" in case:
            _require_string_list(case["tags"], f"{location}.tags")

    return dataset


def load_dataset(path: Path = DEFAULT_CASES_PATH) -> Dict[str, Any]:
    """从 JSON 文件加载并校验评估数据集。"""
    try:
        dataset = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetValidationError(f"dataset file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(
            f"invalid JSON in {path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    return validate_dataset(dataset)


def summarize_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """生成不调用模型的数据集摘要。"""
    cases = dataset["cases"]
    return {
        "dataset_version": dataset["dataset_version"],
        "case_count": len(cases),
        "turn_count": sum(len(case["turns"]) for case in cases),
        "categories": dict(sorted(Counter(
            case["category"] for case in cases
        ).items())),
        "severities": dict(sorted(Counter(
            case["severity"] for case in cases
        ).items())),
    }


def _check_expected_preferences(
    expected_preferences: Dict[str, Any],
    actual_preferences: Dict[str, Any],
) -> bool:
    """检查期望偏好是否存在于实际偏好中。"""
    for key, expected_value in expected_preferences.items():
        actual_value = actual_preferences.get(key)
        if isinstance(expected_value, list):
            if not isinstance(actual_value, list):
                return False
            if not set(expected_value).issubset(set(actual_value)):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _evaluate_memory(
    expected_memory: Dict[str, Any],
    actual: Dict[str, Any],
) -> Dict[str, bool]:
    """评估偏好、行程历史和待补全状态变化。"""
    before = actual.get("memory_before", {}) or {}
    after = actual.get("memory_after", {}) or {}
    checks: Dict[str, bool] = {}

    preference_change = expected_memory.get("preferences_change")
    before_preferences = before.get("preferences", {}) or {}
    after_preferences = after.get("preferences", {}) or {}

    if preference_change == "none":
        checks["memory.preferences_change"] = (
            before_preferences == after_preferences
        )
    elif preference_change in {"append", "replace"}:
        expected_preferences = expected_memory.get(
            "expected_preferences",
            {},
        )
        checks["memory.preferences_change"] = (
            _check_expected_preferences(
                expected_preferences,
                after_preferences,
            )
            and before_preferences != after_preferences
        )

    trip_history_change = expected_memory.get("trip_history_change")
    before_trips = before.get("trip_history", []) or []
    after_trips = after.get("trip_history", []) or []

    if trip_history_change == "none":
        checks["memory.trip_history_change"] = before_trips == after_trips
    elif trip_history_change == "append_one":
        checks["memory.trip_history_change"] = (
            len(after_trips) == len(before_trips) + 1
        )

    if "pending_trip" in expected_memory:
        checks["memory.pending_trip"] = (
            actual.get("pending_trip", {})
            == expected_memory["pending_trip"]
        )

    return checks


def evaluate_turn(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
) -> Dict[str, Any]:
    """将一轮模拟或真实执行轨迹与期望结果比较。"""
    scheduled_agents = set(actual.get("scheduled_agents", []))
    executed_agents = set(actual.get("executed_agents", []))
    required_scheduled = set(expected["required_scheduled_agents"])
    required_executed = set(expected["required_executed_agents"])
    forbidden_executed = set(expected["forbidden_executed_agents"])

    actual_entities = actual.get("entities", {}) or {}
    expected_entities = expected.get("entities", {})
    response = str(actual.get("response", ""))

    checks: Dict[str, bool] = {
        "required_scheduled_agents": required_scheduled.issubset(
            scheduled_agents
        ),
        "required_executed_agents": required_executed.issubset(
            executed_agents
        ),
        "forbidden_executed_agents": not bool(
            forbidden_executed & executed_agents
        ),
        "status": actual.get("status") == expected.get("status"),
        "entities": all(
            actual_entities.get(key) == value
            for key, value in expected_entities.items()
        ),
        "missing_fields": set(actual.get("missing_fields", []))
        == set(expected.get("missing_fields", [])),
        "response.must_contain": all(
            text in response
            for text in expected.get("must_contain", [])
        ),
        "response.must_not_contain_patterns": not any(
            re.search(pattern, response)
            for pattern in expected.get(
                "must_not_contain_patterns",
                [],
            )
        ),
    }

    checks.update(_evaluate_memory(expected["memory"], actual))

    if "history_usage" in expected:
        checks["history_usage"] = (
            actual.get("history_usage") == expected["history_usage"]
        )

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
    }


def evaluate_case(
    case: Dict[str, Any],
    actual_turns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """评估一个单轮或多轮场景。"""
    expected_turns = case["turns"]
    if len(actual_turns) != len(expected_turns):
        return {
            "case_id": case["id"],
            "passed": False,
            "failures": ["turn_count"],
            "turns": [],
        }

    turn_results = [
        evaluate_turn(expected_turn["expected"], actual_turn)
        for expected_turn, actual_turn in zip(expected_turns, actual_turns)
    ]
    failures = [
        f"turn[{index}].{failure}"
        for index, result in enumerate(turn_results)
        for failure in result["failures"]
    ]
    return {
        "case_id": case["id"],
        "passed": not failures,
        "failures": failures,
        "turns": turn_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the Agent evaluation dataset",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to agent_cases.json",
    )
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.cases)
    except DatasetValidationError as exc:
        print(f"INVALID: {exc}")
        return 1

    print(json.dumps(summarize_dataset(dataset), ensure_ascii=False, indent=2))
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
