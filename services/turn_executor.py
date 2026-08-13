"""单轮 Agent 业务执行服务。

这个模块只负责完成一轮用户请求，不负责终端打印。CLI 和后续评估器
可以共享同一条真实执行链路，避免各自重复实现意图识别和调度逻辑。
"""

import json
from typing import Any, Dict, Optional

from agentscope.message import Msg

from utils.circuit_breaker import CircuitOpenError
from utils.intention_routing import normalize_intention_routing
from utils.llm_resilience import retry_with_backoff


class InvalidIntentionResultError(ValueError):
    """IntentionAgent 没有返回可解析的 JSON。"""


class AgentTurnExecutor:
    """执行一轮完整的 Agent 业务流程，并返回结构化结果。"""

    def __init__(
        self,
        intention_agent: Any,
        orchestrator: Any,
        memory_manager: Any,
        circuit_breaker: Optional[Any] = None,
        resilience_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.intention_agent = intention_agent
        self.orchestrator = orchestrator
        self.memory_manager = memory_manager
        self.circuit_breaker = circuit_breaker
        self.resilience_config = resilience_config or {}

    async def execute_turn(self, user_input: str) -> Dict[str, Any]:
        """执行“读取上下文 → 意图识别 → 调度 → 写入对话记忆”。"""
        if self.circuit_breaker:
            self.circuit_breaker.raise_if_open()

        context_messages = await self._build_intention_context(user_input)
        intention_result = await self._execute_intention(context_messages)

        try:
            intention_data = json.loads(intention_result.content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise InvalidIntentionResultError(
                "IntentionAgent 返回结果不是有效 JSON"
            ) from exc

        # IntentionAgent 只知道对话文本，不直接持有协调器中的待补全状态。
        # 在业务入口补充这层状态感知，避免用户续接行程时只执行事项收集，
        # 却跳过后续的合并、缺失字段检查和继续规划流程。
        get_pending_trip = getattr(self.memory_manager, "get_pending_trip", None)
        if not callable(get_pending_trip):
            # 兼容评估器中的轻量 FakeMemoryManager。
            get_pending_trip = getattr(self.orchestrator, "get_pending_trip", None)
        pending_trip = get_pending_trip() if callable(get_pending_trip) else {}
        if pending_trip:
            intention_data = normalize_intention_routing(
                intention_data,
                user_input,
                pending_trip=pending_trip,
            )
            intention_result.content = json.dumps(
                intention_data,
                ensure_ascii=False,
            )

        # 原始输入属于本轮运行时上下文，不写进 LLM 的五字段输出合同；
        # 调度器用它判断子 Agent 提取的日期是否真的来自用户表达。
        intention_result.metadata = {
            **(getattr(intention_result, "metadata", {}) or {}),
            "original_user_input": user_input,
        }

        # 意图识别完成后再记录当前输入，供子 Agent 读取最新的会话上下文。
        self.memory_manager.add_message("user", user_input)

        orchestration_result = await self._execute_orchestration(intention_result)
        try:
            result_data = json.loads(orchestration_result.content)
        except (json.JSONDecodeError, TypeError):
            result_data = {"error": "解析结果失败"}

        # 一次完整请求成功走完，才重置熔断器的连续失败计数。
        # TODO(监控): record_success 这里只表示下游服务可用；如果上面的
        # 调度结果解析失败，还应另行记录 orchestration_parse_error 业务指标。
        if self.circuit_breaker:
            self.circuit_breaker.record_success()

        self.memory_manager.add_message(
            "assistant",
            json.dumps(result_data, ensure_ascii=False),
        )

        return {
            "user_input": user_input,
            "intention": intention_data,
            "orchestration": result_data,
        }

    async def _execute_intention(self, context_messages: list[Msg]) -> Msg:
        """调用 IntentionAgent，并在可重试错误发生时进行退避重试。"""
        config = self.resilience_config
        try:
            return await retry_with_backoff(
                lambda: self.intention_agent.reply(context_messages),
                max_retries=config.get("max_retries", 3),
                base_delay_sec=config.get("retry_base_delay_sec", 1.0),
                max_delay_sec=config.get("retry_max_delay_sec", 30.0),
            )
        except CircuitOpenError:
            raise
        except Exception:
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            raise

    async def _execute_orchestration(self, intention_result: Msg) -> Msg:
        """调用调度器，并在可重试错误发生时进行退避重试。"""
        config = self.resilience_config
        try:
            return await retry_with_backoff(
                lambda: self.orchestrator.reply(intention_result),
                max_retries=config.get("max_retries", 3),
                base_delay_sec=config.get("retry_base_delay_sec", 1.0),
                max_delay_sec=config.get("retry_max_delay_sec", 30.0),
            )
        except CircuitOpenError:
            raise
        except Exception:
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            raise

    async def _build_intention_context(self, user_input: str) -> list[Msg]:
        """组装 IntentionAgent 需要的长期摘要、近期对话和当前问题。"""
        long_term_summary = await self._get_long_term_summary(user_input)
        recent_context = self.memory_manager.short_term.get_recent_context(
            n_turns=5,
        )

        messages = []
        if long_term_summary:
            messages.append(
                Msg(
                    name="system",
                    content=long_term_summary,
                    role="system",
                )
            )
        for message in recent_context:
            messages.append(
                Msg(
                    name=message["role"],
                    content=message["content"],
                    role=message["role"],
                )
            )
        messages.append(Msg(name="user", content=user_input, role="user"))
        return messages

    async def _get_long_term_summary(self, user_input: str = "") -> str:
        """为意图识别整理偏好、历史会话摘要和相关历史行程。"""
        summary_parts = []

        preferences = self.memory_manager.long_term.get_preference()
        if preferences:
            preference_lines = [
                "【用户背景信息】（来自长期记忆，可用于推断缺失信息）"
            ]
            for preference_key, preference_value in preferences.items():
                if not preference_value:
                    continue
                if isinstance(preference_value, list):
                    preference_lines.append(
                        f"• {preference_key}: {', '.join(preference_value)}"
                    )
                else:
                    preference_lines.append(
                        f"• {preference_key}: {preference_value}"
                    )

            if len(preference_lines) > 1:
                summary_parts.extend(preference_lines)

        chat_summary = await self.memory_manager.get_long_term_summary_async(
            max_messages=50,
        )
        if chat_summary:
            summary_parts.append("\n【历史会话总结】")
            summary_parts.append(chat_summary)

        all_trips = self.memory_manager.long_term.get_trip_history(limit=None)
        if all_trips:
            relevant_trips = []
            other_trips = []

            for trip in all_trips:
                origin = trip.get("origin", "") or ""
                destination = trip.get("destination", "") or ""
                if (
                    (origin and origin in user_input)
                    or (destination and destination in user_input)
                ):
                    relevant_trips.append(trip)
                else:
                    other_trips.append(trip)

            trips_to_show = relevant_trips[:2] + other_trips[:1]
            if trips_to_show:
                summary_parts.append("\n【历史行程】")
                for index, trip in enumerate(trips_to_show[:3], 1):
                    origin = trip.get("origin", "未知")
                    destination = trip.get("destination", "未知")
                    start_date = trip.get("start_date", "")
                    purpose = trip.get("purpose", "")
                    relevance_mark = "✦ " if trip in relevant_trips else ""
                    summary_parts.append(
                        f"{index}. {relevance_mark}{origin} → {destination} "
                        f"({start_date}) - {purpose}"
                    )

        return "\n".join(summary_parts) if summary_parts else ""
