"""检查行程中缺少用户确认来源的高风险实时事实。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


TRANSPORT_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[GDCZTK]\d{1,4}|[A-Z]{2}\d{3,4})(?!\d)"
)
PRICE_PATTERN = re.compile(
    r"(?:票价|车票|机票|房价|每晚|住宿)"
    r"[^。；，,\n]{0,20}?(?:[¥￥]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*元)"
)
PRICE_VALUE_PATTERN = re.compile(
    r"[¥￥]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*元"
)
WEATHER_DETAIL_PATTERN = re.compile(
    r"-?\d+(?:\.\d+)?\s*(?:℃|摄氏度)"
)
CONFIRMED_CLAIM_PATTERN = re.compile(
    r"(?:已为您|已经为您|您已|用户已|已经|已)"
    r"[^。；，,\n]{0,10}(?:预订|安排|确认)"
)
BOOKING_CONTEXT_PATTERN = re.compile(
    r"(?:酒店|住宿|房间|车票|机票|车次|航班|高铁|火车|列车|去程|返程)"
)


def _iter_text_values(
    value: Any,
    path: str = "itinerary",
) -> Iterable[Tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _iter_text_values(child, child_path)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_text_values(child, f"{path}[{index}]")
        return
    if isinstance(value, str) and value.strip():
        yield path, value.strip()


def _detail_text(event_data: Dict[str, Any], field: str) -> str:
    value = event_data.get(field)
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _add_issue(
    issues: List[Dict[str, Any]],
    category: str,
    path: str,
    claim: str,
    message: str,
) -> None:
    key = (category, path, claim)
    if any(
        (item["category"], item["path"], item["claim"]) == key
        for item in issues
    ):
        return
    issues.append({
        "category": category,
        "path": path,
        "claim": claim,
        "message": message,
    })


def _trusted_price_values(value: Any, parent_key: str = "") -> Set[str]:
    """提取用户输入、企业政策或外部查询中明确给出的价格/预算数字。"""
    values: Set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            values.update(_trusted_price_values(child, str(key)))
        return values
    if isinstance(value, list):
        for child in value:
            values.update(_trusted_price_values(child, parent_key))
        return values

    key = parent_key.lower()
    if any(token in key for token in ("budget", "price", "fare", "cost")):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.add(str(value).rstrip("0").rstrip(".") if isinstance(value, float) else str(value))
        elif isinstance(value, str):
            values.update(re.findall(r"\d+(?:\.\d+)?", value))
    return values


def find_unsupported_itinerary_facts(
    itinerary_result: Dict[str, Any],
    event_data: Dict[str, Any],
    *,
    trusted_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """返回无用户确认来源的车次、价格、温度和确认性预订措辞。"""
    itinerary = itinerary_result.get("itinerary", {})
    if not isinstance(itinerary, dict):
        return []

    outbound_confirmed = (
        event_data.get("outbound_booking_status") == "confirmed"
    )
    return_confirmed = (
        event_data.get("return_booking_status") == "confirmed"
    )
    hotel_confirmed = event_data.get("hotel_booking_status") == "confirmed"
    transport_details = " ".join(filter(None, (
        _detail_text(event_data, "outbound_booking_details"),
        _detail_text(event_data, "return_booking_details"),
    )))
    all_confirmed_details = " ".join(filter(None, (
        transport_details,
        _detail_text(event_data, "hotel_booking_details"),
    )))
    trusted_prices = _trusted_price_values(trusted_context or event_data)

    issues: List[Dict[str, Any]] = []
    for path, text in _iter_text_values(itinerary):
        for match in TRANSPORT_IDENTIFIER_PATTERN.finditer(text):
            identifier = match.group(0)
            if identifier not in transport_details:
                _add_issue(
                    issues,
                    "unsupported_transport_identifier",
                    path,
                    identifier,
                    "具体车次或航班号没有出现在用户确认的预订信息中。",
                )

        for match in PRICE_PATTERN.finditer(text):
            price_match = PRICE_VALUE_PATTERN.search(match.group(0))
            price = price_match.group(0) if price_match else match.group(0)
            numeric_price = "".join(re.findall(r"\d+(?:\.\d+)?", price))
            if (
                price not in all_confirmed_details
                and numeric_price not in trusted_prices
            ):
                _add_issue(
                    issues,
                    "unsupported_price",
                    path,
                    match.group(0),
                    "票价或住宿价格没有用户确认来源。",
                )

        for match in WEATHER_DETAIL_PATTERN.finditer(text):
            _add_issue(
                issues,
                "unsupported_weather_detail",
                path,
                match.group(0),
                "当前范围未接入天气事实来源，不能输出具体温度。",
            )

        confirmation = CONFIRMED_CLAIM_PATTERN.search(text)
        if not confirmation:
            continue
        # “已安排客户会议”是正常的固定活动，不属于交通或住宿预订事实。
        if (
            "安排" in confirmation.group(0)
            and not BOOKING_CONTEXT_PATTERN.search(text)
        ):
            continue
        claim_is_supported = False
        if "酒店" in text or "住宿" in text:
            claim_is_supported = hotel_confirmed
        elif "返程" in text:
            claim_is_supported = return_confirmed
        elif any(word in text for word in ("去程", "车次", "航班", "高铁", "机票", "车票")):
            claim_is_supported = outbound_confirmed or return_confirmed
        else:
            claim_is_supported = (
                outbound_confirmed or return_confirmed or hotel_confirmed
            )
        if not claim_is_supported:
            _add_issue(
                issues,
                "unsupported_confirmation",
                path,
                confirmation.group(0),
                "参考方案不能把未确认项目表述为已经预订或安排。",
            )

    return issues
