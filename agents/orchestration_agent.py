"""
协调器智能体 OrchestrationAgent
职责：根据意图识别结果，协调调度多个子智能体完成任务

核心功能：
1. 接收 IntentionAgent 的调度决策
2. 按照优先级顺序执行子智能体
3. 管理智能体之间的消息传递
4. 聚合多个智能体的结果
5. 与三层记忆系统集成

执行模式：
- Sequential (顺序执行): 按优先级依次执行，前一个的输出作为后一个的输入
- Parallel (并行执行): 同时执行多个智能体（暂不实现）
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Union, List, Dict, Any
import json
import logging
import asyncio

logger = logging.getLogger(__name__)


class OrchestrationAgent(AgentBase):
    """协调器智能体 - 调度和协调多个子智能体"""

    def __init__(
        self,
        name: str = "OrchestrationAgent",
        agent_registry: Dict[str, AgentBase] = None,
        memory_manager = None,
        **kwargs
    ):
        """
        初始化协调器

        Args:
            name: 智能体名称
            agent_registry: 子智能体注册表 {agent_name: agent_instance}
            memory_manager: 记忆管理器
        """
        super().__init__()
        self.name = name
        self.agent_registry = agent_registry or {}
        self.memory_manager = memory_manager
        # 保存当前会话中尚未补全的结构化行程信息。
        self._pending_trip_data: Dict[str, Any] = {}

    def register_agent(self, agent_name: str, agent: AgentBase):
        """注册子智能体"""
        self.agent_registry[agent_name] = agent
        logger.info(f"Registered agent: {agent_name}")

    def unregister_agent(self, agent_name: str):
        """注销子智能体"""
        if agent_name in self.agent_registry:
            del self.agent_registry[agent_name]
            logger.info(f"Unregistered agent: {agent_name}")

    def clear_pending_trip(self) -> None:
        """清除当前会话中尚未完成的行程信息。"""
        self._pending_trip_data.clear()

    def get_pending_trip(self) -> Dict[str, Any]:
        """返回待补全行程的副本，供状态展示和执行轨迹采集使用。"""
        return dict(self._pending_trip_data)

    def restore_pending_trip(self, trip_data: Dict[str, Any]) -> None:
        """恢复待补全行程，供会话恢复和隔离评估环境初始化使用。"""
        self._pending_trip_data = dict(trip_data or {})

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """
        协调执行流程

        Args:
            x: 输入消息，应包含 IntentionAgent 的输出

        Returns:
            Msg: 执行结果
        """
        if x is None:
            return Msg(
                name=self.name,
                content=json.dumps({"error": "No input provided"}),
                role="assistant"
            )

        # 解析输入
        if isinstance(x, list):
            intention_output = x[-1].content if x else "{}"
        else:
            intention_output = x.content

        # 解析意图识别结果
        try:
            intention_data = json.loads(intention_output) if isinstance(intention_output, str) else intention_output
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse intention output: {e}")
            return Msg(
                name=self.name,
                content=json.dumps({"error": "Invalid intention format"}),
                role="assistant"
            )

        # 获取智能体调度计划
        agent_schedule = intention_data.get("agent_schedule", [])
        if not agent_schedule:
            return Msg(
                name=self.name,
                content=json.dumps({
                    "status": "no_agents",
                    "message": "没有需要调度的智能体"
                }),
                role="assistant"
            )

        # 按优先级排序
        sorted_schedule = sorted(agent_schedule, key=lambda x: x.get("priority", 999))

        logger.info(f"Orchestrating {len(sorted_schedule)} agents")

        # 准备上下文信息
        context = self._prepare_context(intention_data)

        # 并行执行智能体（按优先级分组）
        results = []
        current_priority = None
        parallel_tasks = []

        for task in sorted_schedule:
            priority = task.get("priority", 0)

            # 如果优先级变化，先执行当前批次
            if current_priority is not None and priority != current_priority:
                # 并行执行当前优先级的所有任务
                if parallel_tasks:
                    batch_results = await self._execute_parallel_agents(parallel_tasks, context, results)
                    results.extend(batch_results)
                    # 下一批包含行程规划时，先检查前序事项信息是否完整。
                    next_batch_has_itinerary = any(
                        scheduled_task.get("priority", 0) == priority
                        and scheduled_task.get("agent_name") == "itinerary_planning"
                        for scheduled_task in sorted_schedule
                    )

                    if next_batch_has_itinerary:
                        self._merge_pending_trip_data(results)

                    missing_fields = self._get_missing_trip_fields(results)

                    if next_batch_has_itinerary and missing_fields:
                        field_labels = {
                            "origin": "出发地",
                            "destination": "目的地",
                            "start_date": "出发日期",
                            "duration_days": "行程天数",
                        }
                        missing_labels = "、".join(
                            field_labels.get(field, field)
                            for field in missing_fields
                        )

                        final_result = self._aggregate_results(results, intention_data)
                        final_result.update({
                            "status": "needs_clarification",
                            "missing_fields": missing_fields,
                            "message": f"行程信息不完整，请补充：{missing_labels}",
                        })

                        # 提前返回前，也要保存Priority 1中已经识别成功的偏好。
                        if self.memory_manager:
                            self._update_memory(intention_data, results)

                        return Msg(
                            name=self.name,
                            content=json.dumps(final_result, ensure_ascii=False),
                            role="assistant",
                        )
                    parallel_tasks = []

            current_priority = priority
            parallel_tasks.append(task)

        # 执行最后一批
        if parallel_tasks:
            batch_results = await self._execute_parallel_agents(parallel_tasks, context, results)
            results.extend(batch_results)

        # 行程规划成功后，本次待补全状态已经完成。
        itinerary_completed = any(
            item.get("agent_name") == "itinerary_planning"
            and item.get("result", {}).get("status") == "success"
            for item in results
        )
        if itinerary_completed:
            self._pending_trip_data.clear()

        # 聚合结果
        final_result = self._aggregate_results(results, intention_data)

        # 更新记忆
        if self.memory_manager:
            self._update_memory(intention_data, results)

        return Msg(
            name=self.name,
            content=json.dumps(final_result, ensure_ascii=False),
            role="assistant"
        )

    def _merge_pending_trip_data(
        self,
        results: List[Dict[str, Any]],
    ) -> None:
        """将本轮事项信息与上一轮待补全信息合并。"""
        trip_fields = (
            "origin",
            "destination",
            "start_date",
            "end_date",
            "duration_days",
            "return_location",
            "trip_purpose",
        )

        for item in results:
            if item.get("agent_name") != "event_collection":
                continue

            result = item.get("result", {})
            if result.get("status") != "success":
                continue

            current_data = result.get("data", {})
            if not isinstance(current_data, dict):
                continue

            # 先保留上一轮的信息，再使用本轮非空字段覆盖。
            merged_data = dict(self._pending_trip_data)

            for field in trip_fields:
                value = current_data.get(field)
                if value not in (None, "", []):
                    merged_data[field] = value

            # 保留summary、missing_info等非行程字段。
            for key, value in current_data.items():
                if key not in trip_fields:
                    merged_data[key] = value

            # 后面的行程规划Agent将从results中获得合并后的结果。
            result["data"] = merged_data

            # 保存下来，供下一轮继续补充。
            self._pending_trip_data = {
                field: merged_data.get(field)
                for field in trip_fields
                if merged_data.get(field) not in (None, "", [])
            }
            break

    def _get_missing_trip_fields(
        self,
        results: List[Dict[str, Any]],
    ) -> List[str]:
        """从事项收集结果中检查行程规划所需的必填字段。"""
        event_data = None

        for item in results:
            if item.get("agent_name") != "event_collection":
                continue

            result = item.get("result", {})
            if result.get("status") != "success":
                continue

            event_data = result.get("data", {})
            break

        # 没有可靠的事项收集结果时，不能继续规划行程。
        if event_data is None:
            return [
                "origin",
                "destination",
                "start_date",
                "duration_days",
            ]

        missing_fields = []

        # 行程规划需要明确的出发地、目的地和出发日期。
        for field in ("origin", "destination", "start_date"):
            if not event_data.get(field):
                missing_fields.append(field)

        # duration_days和end_date有一个即可表达行程长度。
        if not event_data.get("duration_days") and not event_data.get("end_date"):
            missing_fields.append("duration_days")

        return missing_fields

    def _prepare_context(self, intention_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备上下文信息，供子智能体使用

        Args:
            intention_data: 意图识别结果

        Returns:
            上下文字典
        """
        context = {
            "reasoning": intention_data.get("reasoning", ""),
            "intents": intention_data.get("intents", []),
            "key_entities": intention_data.get("key_entities", {}),
            "rewritten_query": intention_data.get("rewritten_query", "")
        }

        # 这里会再次读取记忆，但用途不同于 CLI 给 IntentionAgent 的历史摘要：
        # IntentionAgent 使用“摘要 + 最近对话”理解和路由；子 Agent 需要最新的
        # 原始对话与结构化偏好执行任务（此时当前用户消息也已写入记忆）。
        # TODO(优化): 后续可在请求入口构建统一 RequestContext 并向下传递，减少重复读取。
        if self.memory_manager:
            # 短期记忆：最近对话
            recent_context = self.memory_manager.short_term.get_recent_context(3)
            context["recent_dialogue"] = recent_context

            # 长期记忆：用户偏好
            preferences = self.memory_manager.long_term.get_preference()
            context["user_preferences"] = preferences

        return context

    async def _execute_parallel_agents(
        self,
        tasks: List[Dict],
        context: Dict[str, Any],
        previous_results: List[Dict]
    ) -> List[Dict]:
        """
        并行执行多个智能体

        Args:
            tasks: 任务列表，每个任务包含 agent_name, priority, reason, expected_output
            context: 上下文信息
            previous_results: 前序智能体的结果

        Returns:
            执行结果列表
        """
        if not tasks:
            return []

        # 如果只有一个任务，直接执行
        if len(tasks) == 1:
            task = tasks[0]
            result = await self._execute_agent(
                agent_name=task.get("agent_name"),
                context=context,
                reason=task.get("reason", ""),
                expected_output=task.get("expected_output", ""),
                previous_results=previous_results
            )
            return [{
                "agent_name": task.get("agent_name"),
                "priority": task.get("priority", 0),
                "result": result
            }]

        # 多个任务并行执行
        logger.info(f"Executing {len(tasks)} agents in parallel")

        # 创建并行任务
        parallel_coroutines = []
        for task in tasks:
            agent_name = task.get("agent_name")
            priority = task.get("priority", 0)
            reason = task.get("reason", "")
            expected_output = task.get("expected_output", "")

            logger.info(f"Parallel executing agent: {agent_name} (priority={priority})")

            # 创建协程
            coroutine = self._execute_agent(
                agent_name=agent_name,
                context=context,
                reason=reason,
                expected_output=expected_output,
                previous_results=previous_results
            )
            parallel_coroutines.append((agent_name, priority, coroutine))

        # 使用 asyncio.gather 并行执行
        execution_results = await asyncio.gather(
            *[coro for _, _, coro in parallel_coroutines],
            return_exceptions=True
        )

        # 整理结果
        results = []
        for (agent_name, priority, _), exec_result in zip(parallel_coroutines, execution_results):
            if isinstance(exec_result, Exception):
                logger.error(f"Parallel agent execution failed: {agent_name}, error: {exec_result}")
                result = {
                    "status": "error",
                    "agent_name": agent_name,
                    "data": {"error": str(exec_result)},
                    "message": f"并行执行失败: {str(exec_result)}"
                }
            else:
                result = exec_result

            results.append({
                "agent_name": agent_name,
                "priority": priority,
                "result": result
            })

        return results

    async def _execute_agent(
        self,
        agent_name: str,
        context: Dict[str, Any],
        reason: str,
        expected_output: str,
        previous_results: List[Dict]
    ) -> Dict[str, Any]:
        """
        执行单个智能体

        Args:
            agent_name: 智能体名称
            context: 上下文信息
            reason: 调用原因
            expected_output: 期望输出
            previous_results: 前序智能体的结果

        Returns:
            执行结果
        """
        # 检查智能体是否注册
        if agent_name not in self.agent_registry:
            logger.warning(f"Agent not registered: {agent_name}")
            return {
                "status": "error",
                "message": f"智能体未注册: {agent_name}"
            }

        agent = self.agent_registry[agent_name]

        # 构建输入消息
        input_msg = Msg(
            name="Orchestrator",
            content=json.dumps({
                "context": context,
                "reason": reason,
                "expected_output": expected_output,
                "previous_results": previous_results
            }, ensure_ascii=False),
            role="user"
        )

        try:
            # 调用智能体
            response = await agent.reply(input_msg)

            # 解析响应
            if isinstance(response.content, str):
                try:
                    result = json.loads(response.content)
                except json.JSONDecodeError:
                    result = {"output": response.content}
            else:
                result = response.content

            # 检查 result 中是否有 error 字段
            # 如果有，说明智能体内部执行失败了
            if isinstance(result, dict) and "error" in result:
                error_msg = result.get("error", "未知错误")
                return {
                    "status": "error",
                    "agent_name": agent_name,
                    "data": result,
                    "message": error_msg
                }

            return {
                "status": "success",
                "agent_name": agent_name,
                "data": result
            }

        except Exception as e:
            logger.error(f"Agent execution failed: {agent_name}, error: {e}")
            # 返回友好的错误信息，但不中断流程
            return {
                "status": "error",
                "agent_name": agent_name,
                "data": {"error": str(e)},
                "message": f"智能体执行失败: {str(e)}"
            }

    def _aggregate_results(
        self,
        results: List[Dict],
        intention_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        聚合多个智能体的结果

        Args:
            results: 所有智能体的执行结果
            intention_data: 原始意图识别结果

        Returns:
            聚合后的最终结果
        """
        aggregated = {
            "status": "completed",
            "intention": {
                "intents": intention_data.get("intents", []),
                "key_entities": intention_data.get("key_entities", {})
            },
            "agents_executed": len(results),
            "results": []
        }

        # 收集每个智能体的结果
        for result in results:
            aggregated["results"].append({
                "agent_name": result["agent_name"],
                "priority": result["priority"],
                "status": result["result"].get("status", "unknown"),
                "data": result["result"].get("data", {})
            })

        # 检查是否有错误
        errors = [r for r in results if r["result"].get("status") == "error"]
        if errors:
            aggregated["status"] = "partial_failure"
            aggregated["errors"] = len(errors)

        return aggregated

    def _update_memory(self, intention_data: Dict[str, Any], results: List[Dict]):
        """
        更新记忆系统

        Args:
            intention_data: 意图识别结果
            results: 智能体执行结果
        """
        if not self.memory_manager:
            return

        # 提取并保存信息到长期记忆
        for result in results:
            agent_name = result["agent_name"]
            data = result["result"].get("data", {})

            # 如果是偏好智能体，保存偏好信息到长期记忆
            if agent_name == "preference" and isinstance(data, dict):
                preferences_data = data.get("preferences", {})

                # 新格式：preferences 是列表，包含 {type, value, action}
                if isinstance(preferences_data, list):
                    for pref_item in preferences_data:
                        if not isinstance(pref_item, dict):
                            continue

                        pref_type = pref_item.get("type")
                        pref_value = pref_item.get("value")
                        pref_action = pref_item.get("action", "replace")  # 默认覆盖

                        if not pref_type or not pref_value:
                            continue

                        # 根据 action 决定操作
                        if pref_action == "append":
                            # 追加模式：获取现有值并追加
                            current_prefs = self.memory_manager.long_term.get_preference()
                            existing_value = current_prefs.get(pref_type)

                            # 如果现有值是列表，追加
                            if isinstance(existing_value, list):
                                if pref_value not in existing_value:
                                    existing_value.append(pref_value)
                                self.memory_manager.long_term.save_preference(pref_type, existing_value)
                                logger.info(f"Appended to {pref_type}: {pref_value}, total: {existing_value}")
                            else:
                                # 如果现有值不是列表，创建新列表
                                new_list = [existing_value, pref_value] if existing_value else [pref_value]
                                self.memory_manager.long_term.save_preference(pref_type, new_list)
                                logger.info(f"Created list for {pref_type}: {new_list}")
                        else:
                            # 覆盖模式：直接保存新值
                            self.memory_manager.long_term.save_preference(pref_type, pref_value)
                            logger.info(f"Replaced {pref_type}: {pref_value}")

                # 旧格式兼容：preferences 是字典
                elif isinstance(preferences_data, dict):
                    for pref_type, value in preferences_data.items():
                        if value and pref_type != "has_preferences" and pref_type != "error":
                            self.memory_manager.long_term.save_preference(pref_type, value)
                            logger.info(f"Updated {pref_type}: {value} (legacy format)")

            # 如果是行程规划智能体，保存行程到长期记忆
            if agent_name == "itinerary_planning" and isinstance(data, dict):
                itinerary = data.get("itinerary", {})

                # 只要有行程信息就保存（不管是否完全规划好）
                if itinerary:
                    # 提取事项收集的信息（出发地、目的地等）
                    event_data = {}
                    for r in results:
                        if r["agent_name"] == "event_collection":
                            event_data = r["result"].get("data", {})
                            break

                    # 从 event_data 获取行程信息
                    origin = event_data.get("origin")
                    destination = event_data.get("destination")
                    start_date = event_data.get("start_date")
                    end_date = event_data.get("end_date")
                    purpose = event_data.get("trip_purpose", "旅游")

                    # 保存到长期记忆（只要有目的地就保存）
                    if destination:
                        self.memory_manager.long_term.save_trip_history({
                            "origin": origin,
                            "destination": destination,
                            "start_date": start_date,
                            "end_date": end_date,
                            "purpose": purpose
                        })
                        logger.info(f"Saved trip to long-term memory: {origin} -> {destination}")

        logger.info("Memory updated after orchestration")
