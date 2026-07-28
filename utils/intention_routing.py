"""IntentionAgent 输出后的确定性业务路由约束。"""

from __future__ import annotations

from typing import Any


HISTORY_REFERENCE_MARKERS = (
    "之前",
    "以前",
    "过去",
    "历史",
    "上次",
    "保存了",
    "记得我",
    "去过",
)
PREFERENCE_MARKERS = (
    "偏好",
    "喜欢",
    "酒店品牌",
    "常坐",
    "座位",
)
PREFERENCE_MUTATION_MARKERS = (
    "还喜欢",
    "也喜欢",
    "另外喜欢",
    "改成",
    "换成",
    "搬家",
    "设置",
    "请保存",
    "帮我保存",
    "记住",
    "删除",
    "取消",
    "不喜欢",
    "新增",
    "添加",
    "修改",
    "更新",
)
EXTERNAL_INFORMATION_MARKERS = (
    "天气",
    "搜索",
    "实时",
    "航班",
    "车次",
    "展会",
)
AGENT_ORDER = {
    "memory_query": 0,
    "event_collection": 1,
    "preference": 2,
    "information_query": 3,
    "rag_knowledge": 4,
    "itinerary_planning": 5,
}


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _ensure_agent(
    schedule: list[dict[str, Any]],
    agent_name: str,
    priority: int,
    reason: str,
) -> None:
    if any(item.get("agent_name") == agent_name for item in schedule):
        return
    schedule.append({
        "agent_name": agent_name,
        "priority": priority,
        "reason": reason,
        "expected_output": "提供后续任务所需的结构化结果",
    })


def normalize_intention_routing(
    result: dict[str, Any],
    user_query: str,
) -> dict[str, Any]:
    """对已通过结构校验的模型调度表应用核心业务不变量。"""
    normalized = dict(result)
    schedule = [dict(item) for item in result.get("agent_schedule", [])]
    intents = [dict(item) for item in result.get("intents", [])]
    query = user_query or ""

    is_history_reference = _contains_any(query, HISTORY_REFERENCE_MARKERS)
    is_preference_context = _contains_any(query, PREFERENCE_MARKERS)
    is_preference_mutation = _contains_any(
        query,
        PREFERENCE_MUTATION_MARKERS,
    )
    is_external_query = _contains_any(query, EXTERNAL_INFORMATION_MARKERS)

    # 查询或引用已保存偏好属于读操作，不应调用会写长期记忆的 PreferenceAgent。
    if (
        is_history_reference
        and is_preference_context
        and not is_preference_mutation
    ):
        schedule = [
            item
            for item in schedule
            if item.get("agent_name") != "preference"
        ]
        if not is_external_query:
            schedule = [
                item
                for item in schedule
                if item.get("agent_name") != "information_query"
            ]
        _ensure_agent(
            schedule,
            "memory_query",
            1,
            "用户正在查询或引用已保存的历史偏好",
        )

        # 用户意图和实际调度要保持一致，避免出现“意图是联网查询，
        # 实际却执行记忆查询”这类难以追踪的矛盾结果。
        blocked_intents = {"preference"}
        if not is_external_query:
            blocked_intents.add("information_query")
        intents = [
            item
            for item in intents
            if item.get("type") not in blocked_intents
        ]
        if not any(item.get("type") == "memory_query" for item in intents):
            intents.append({
                "type": "memory_query",
                "confidence": 1.0,
                "description": "查询或引用用户已保存的历史偏好",
                "reason": "用户明确引用了自己的历史偏好",
            })

    # 行程规划必须先经过事项收集，不能直接把意图实体交给规划 Agent。
    if any(
        item.get("agent_name") == "itinerary_planning"
        for item in schedule
    ):
        _ensure_agent(
            schedule,
            "event_collection",
            1,
            "行程规划前必须收集并校验必填行程字段",
        )

    # 统一关键 Agent 的执行批次，并让同批次顺序稳定、便于测试和追踪。
    for item in schedule:
        if item.get("agent_name") == "itinerary_planning":
            item["priority"] = 2
        elif item.get("agent_name") in AGENT_ORDER:
            item["priority"] = 1
    schedule.sort(
        key=lambda item: (
            item.get("priority", 999),
            AGENT_ORDER.get(item.get("agent_name"), 999),
        )
    )

    normalized["intents"] = intents
    normalized["agent_schedule"] = schedule
    return normalized
