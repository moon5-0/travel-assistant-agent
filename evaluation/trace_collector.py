"""真实 Agent 单轮执行轨迹采集。

采集器包裹 AgentTurnExecutor，记录执行前后的可观察状态，但不负责评分。
"""

from copy import deepcopy
import json
from typing import Any, Dict, Iterable


TRIP_ENTITY_FIELDS = (
    "origin",
    "destination",
    "start_date",
    "end_date",
    "duration_days",
    "return_location",
    "trip_purpose",
)


class AgentTraceCollector:
    """执行真实业务链路，并转换为评分器需要的 actual_turn。"""

    def __init__(self, turn_executor: Any) -> None:
        self.turn_executor = turn_executor

    async def execute_turn(self, user_input: str) -> Dict[str, Any]:
        """采集执行前后记忆，并返回一轮标准化执行轨迹。"""
        memory_before = self._snapshot_memory()
        turn_result = await self.turn_executor.execute_turn(user_input)
        memory_after = self._snapshot_memory()

        return self.build_trace(
            user_input=user_input,
            turn_result=turn_result,
            memory_before=memory_before,
            memory_after=memory_after,
        )

    def build_trace(
        self,
        user_input: str,
        turn_result: Dict[str, Any],
        memory_before: Dict[str, Any],
        memory_after: Dict[str, Any],
    ) -> Dict[str, Any]:
        """把业务层返回值转换成 evaluate_turn() 接受的字段。"""
        intention = turn_result.get("intention", {}) or {}
        orchestration = turn_result.get("orchestration", {}) or {}
        schedule = intention.get("agent_schedule", []) or []
        results = orchestration.get("results", []) or []
        entities = self._collect_entities(intention, results)
        response = self._collect_response_text(orchestration)

        return {
            "scheduled_agents": [
                item.get("agent_name")
                for item in schedule
                if item.get("agent_name")
            ],
            "executed_agents": [
                item.get("agent_name")
                for item in results
                if item.get("agent_name")
            ],
            "status": orchestration.get("status"),
            "entities": entities,
            "missing_fields": orchestration.get("missing_fields", []) or [],
            "memory_before": deepcopy(memory_before),
            "memory_after": deepcopy(memory_after),
            "pending_trip": self._get_pending_trip(),
            "history_usage": self._detect_history_usage(
                user_input=user_input,
                entities=entities,
                status=orchestration.get("status"),
                trip_history=memory_before.get("trip_history", []),
                response=response,
            ),
            "response": response,
            # 保留原始结果，后续排查失败时不用重新运行大模型。
            "raw": deepcopy(turn_result),
        }

    def _snapshot_memory(self) -> Dict[str, Any]:
        """复制偏好和行程历史，防止后续原地修改污染 before 快照。"""
        memory = self.turn_executor.memory_manager.long_term
        return {
            "preferences": deepcopy(memory.get_preference()),
            "trip_history": deepcopy(memory.get_trip_history(limit=None)),
        }

    def _get_pending_trip(self) -> Dict[str, Any]:
        orchestrator = self.turn_executor.orchestrator
        if hasattr(orchestrator, "get_pending_trip"):
            return deepcopy(orchestrator.get_pending_trip())
        return deepcopy(getattr(orchestrator, "_pending_trip_data", {}))

    @staticmethod
    def _collect_entities(
        intention: Dict[str, Any],
        results: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """以意图实体为基础，用事项收集结果补充或覆盖行程实体。"""
        entities = deepcopy(intention.get("key_entities", {}) or {})
        if not isinstance(entities, dict):
            entities = {}

        for item in results:
            if item.get("agent_name") != "event_collection":
                continue
            data = item.get("data", {}) or {}
            if not isinstance(data, dict):
                continue
            for field in TRIP_ENTITY_FIELDS:
                value = data.get(field)
                if value not in (None, "", []):
                    entities[field] = value
        return entities

    @staticmethod
    def _collect_response_text(orchestration: Dict[str, Any]) -> str:
        """汇总调度消息及各子 Agent 原始输出，供文本规则初步检查。

        TODO(评估精度): 当前文本不等于 CLI 最终展示内容。后续应抽取统一的
        ResponsePresenter，让 CLI 和评估器消费同一份用户可见回复。
        """
        parts = []
        message = orchestration.get("message")
        if message:
            parts.append(str(message))

        for result in orchestration.get("results", []) or []:
            data = result.get("data", {})
            if data not in (None, "", {}, []):
                if isinstance(data, str):
                    parts.append(data)
                else:
                    parts.append(json.dumps(data, ensure_ascii=False))

        if not parts:
            parts.append(json.dumps(orchestration, ensure_ascii=False))
        return "\n".join(parts)

    @staticmethod
    def _detect_history_usage(
        user_input: str,
        entities: Dict[str, Any],
        status: Any,
        trip_history: Iterable[Dict[str, Any]],
        response: str,
    ) -> str:
        """根据实体来源迹象区分未使用、等待确认和自动填充历史。

        TODO(评估精度): 当前属于启发式推断，可能将巧合相同的实体误判为
        历史自动填充。后续应由 Agent 输出 source/confirmed 等实体来源字段。
        """
        trips = list(trip_history or [])
        if not trips:
            return "not_used"

        history_references = ("之前", "历史", "上次", "以前", "曾经")
        refers_to_history = any(word in user_input for word in history_references)

        for field in TRIP_ENTITY_FIELDS:
            value = entities.get(field)
            if value in (None, "", []):
                continue
            value_text = str(value)
            if value_text in user_input:
                continue
            if any(str(trip.get(field)) == value_text for trip in trips):
                return "auto_filled"

        asks_for_confirmation = "确认" in response
        if refers_to_history and (
            status == "needs_clarification" or asks_for_confirmation
        ):
            return "confirmation_required"
        return "not_used"
