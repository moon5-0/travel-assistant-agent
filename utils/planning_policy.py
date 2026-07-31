"""根据语义信号和结构化行程事实，确定行程规划模式。"""

from __future__ import annotations

from typing import Any


BUSINESS_KEYWORDS = (
    "出差", "差旅", "商务", "会议", "客户", "拜访",
    "办公", "项目", "培训", "工作", "考察",
)

# 这些规则只用于旧输出或模型漏提 planning_signals 时的兼容兜底。
NO_LEISURE_PATTERNS = (
    "不要旅游", "不需要旅游", "不安排旅游", "不安排景点",
    "不要安排景点", "不安排旅游景点", "不要安排旅游景点",
    "不需要景点", "不要景点", "纯商务", "只安排商务",
)

REQUESTED_LEISURE_PATTERNS = (
    "适量空闲", "空闲活动", "少量休闲", "适当休闲",
    "顺便逛", "顺便游", "自由活动", "游览", "旅游", "景点",
)


def determine_planning_mode(
    user_query: str,
    all_info: dict[str, Any],
) -> str:
    """用确定性优先级把 LLM 语义信号和行程事实解析为规划模式。"""
    event_data = all_info.get("event_collection", {})
    if not isinstance(event_data, dict):
        event_data = {}

    context = all_info.get("context", {})
    if not isinstance(context, dict):
        context = {}
    signals = context.get("planning_signals", {})
    if not isinstance(signals, dict):
        signals = {}

    trip_type = signals.get("trip_type", "unknown")
    leisure_preference = signals.get("leisure_preference", "unspecified")
    constraints = signals.get("explicit_constraints", [])
    if not isinstance(constraints, list):
        constraints = []

    purpose = str(event_data.get("trip_purpose", ""))
    combined_text = " ".join(
        [user_query, purpose, *(str(item) for item in constraints)]
    )

    is_business = (
        trip_type == "business"
        or bool(event_data.get("fixed_events"))
        or any(keyword in combined_text for keyword in BUSINESS_KEYWORDS)
    )
    if not is_business:
        return "general_travel"

    # 明确拒绝休闲的语义信号优先级最高；关键词只作为兼容兜底。
    if leisure_preference == "forbidden":
        return "business_only"
    if any(pattern in combined_text for pattern in NO_LEISURE_PATTERNS):
        return "business_only"

    if leisure_preference == "requested":
        return "business_with_optional_leisure"
    if any(pattern in combined_text for pattern in REQUESTED_LEISURE_PATTERNS):
        return "business_with_optional_leisure"

    return "business_first"


def planning_mode_instruction(mode: str) -> str:
    """返回可直接放入规划 Prompt 的业务约束。"""
    instructions = {
        "business_only": (
            "企业差旅（纯商务）。用户明确不需要旅游活动。只安排固定商务活动、"
            "必要交通、会议准备、用餐和休息；不得添加景点或以自由活动名义变相"
            "加入旅游内容，也不得虚构未提供的会议和客户拜访。"
        ),
        "business_first": (
            "企业差旅（商务优先）。优先保证固定商务活动、交通缓冲、资料准备、"
            "用餐和休息。用户未要求旅游时，不主动安排景点；空余时间可标为机动、"
            "休息或工作准备，但不得虚构未提供的会议和客户拜访。"
        ),
        "business_with_optional_leisure": (
            "企业差旅（允许少量可选休闲）。商务活动、交通缓冲和休息始终优先。"
            "整段行程最多安排1至2项、每项不超过2小时、靠近酒店或商务地点的"
            "可选休闲活动，并明确标为可取消；不得为了景点压缩商务准备或跨城折返。"
        ),
        "general_travel": (
            "普通旅行。根据用户兴趣合理安排游览、交通、用餐和休息，避免活动"
            "堆砌并保留必要缓冲。"
        ),
    }
    return instructions.get(mode, instructions["general_travel"])
