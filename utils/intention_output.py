"""IntentionAgent 结构化输出的数据合同与校验入口。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


AgentName = Literal[
    "memory_query",
    "itinerary_planning",
    "preference",
    "information_query",
    "rag_knowledge",
    "event_collection",
]


class IntentDecision(BaseModel):
    """模型识别到的一项用户意图。"""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    type: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    description: str = ""
    reason: str = ""


class AgentScheduleItem(BaseModel):
    """一项可交给 OrchestrationAgent 执行的调度任务。"""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    agent_name: AgentName
    priority: int = Field(ge=1)
    reason: str = ""
    expected_output: str = ""


class PlanningSignals(BaseModel):
    """与行程规划有关的开放语义信号，不直接决定最终规划模式。"""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    trip_type: Literal["business", "personal", "unknown"] = "unknown"
    leisure_preference: Literal[
        "forbidden",
        "requested",
        "unspecified",
    ] = "unspecified"
    explicit_constraints: list[str] = Field(default_factory=list)


class IntentionResult(BaseModel):
    """IntentionAgent 与后续调度链路之间的稳定数据合同。"""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    reasoning: str
    intents: list[IntentDecision]
    key_entities: dict[str, Any]
    rewritten_query: str = Field(min_length=1)
    agent_schedule: list[AgentScheduleItem]
    planning_signals: PlanningSignals = Field(default_factory=PlanningSignals)


def validate_intention_result(value: dict[str, Any]) -> dict[str, Any]:
    """校验并归一化模型结果，返回可直接 JSON 序列化的字典。"""
    return IntentionResult.model_validate(value).model_dump()
