"""
行程规划智能体
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Union, List, Dict, Any
import json
import logging
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
# project_root = Path(__file__).parent.parent # Removed old logic
# sys.path.insert(0, str(project_root))

from utils.json_parser import robust_json_parse, extract_json_from_async_response
from utils.itinerary_time_validator import find_itinerary_time_issues
from utils.planning_policy import (
    determine_planning_mode,
    planning_mode_instruction,
)

logger = logging.getLogger(__name__)


class ItineraryPlanningAgent(AgentBase):
    """
    行程规划智能体（主协调）
    职责：协调事项收集、路线规划、酒店规划等多个子任务

    整合三层编排智能体的结果，生成完整行程计划
    """

    def __init__(self, name: str = "ItineraryPlanningAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        from utils.skill_loader import SkillLoader
        self.skill_loader = SkillLoader()

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        if x is None:
            return Msg(name=self.name, content={}, role="assistant")

        # 解析输入内容
        content = x.content if not isinstance(x, list) else x[-1].content

        # 初始化变量
        user_query = ""
        context_info = {}
        previous_results = []
        user_preferences = {}

        # 如果content是JSON字符串，解析它（来自OrchestrationAgent）
        if isinstance(content, str):
            try:
                data = json.loads(content)
                context_info = data.get("context", {})
                user_query = context_info.get("rewritten_query", "")
                previous_results = data.get("previous_results", [])
                user_preferences = context_info.get("user_preferences", {})
            except json.JSONDecodeError:
                user_query = content
        elif isinstance(content, dict):
            context_info = content
            user_query = content.get("rewritten_query", str(content))
            user_preferences = content.get("user_preferences", {})

        # 整合所有可用信息
        all_info = {
            "user_query": user_query,
            "context": context_info,
        }

        # 从previous_results中提取其他agent的数据
        for prev in previous_results:
            agent_name = prev.get("agent_name", "")
            result_data = prev.get("result", {}).get("data", {})
            if result_data and agent_name:
                all_info[agent_name] = result_data

        planning_mode = determine_planning_mode(user_query, all_info)
        all_info["planning_mode"] = planning_mode
        mode_instruction = planning_mode_instruction(planning_mode)

        # 构建用户偏好信息
        preferences_info = ""
        if user_preferences:
            pref_parts = ["【用户偏好】（规划时优先考虑）"]
            if user_preferences.get("home_location"):
                pref_parts.append(f"• 家庭住址: {user_preferences['home_location']}")
            if user_preferences.get("hotel_brands"):
                pref_parts.append(f"• 酒店偏好: {', '.join(user_preferences['hotel_brands'])}")
            if user_preferences.get("airlines"):
                pref_parts.append(f"• 航空偏好: {', '.join(user_preferences['airlines'])}")
            if user_preferences.get("seat_preference"):
                pref_parts.append(f"• 座位偏好: {user_preferences['seat_preference']}")

            if len(pref_parts) > 1:
                preferences_info = "\n".join(pref_parts) + "\n\n"

        # 获取当前时间
        from datetime import datetime
        current_date = datetime.now().strftime("%Y年%m月%d日")
        current_month = datetime.now().month
        current_season = "冬季" if current_month in [12, 1, 2] else \
                        "春季" if current_month in [3, 4, 5] else \
                        "夏季" if current_month in [6, 7, 8] else "秋季"
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][datetime.now().weekday()]

        # 尝试从 SKILL.md 动态读取详细指令 (Progressive Disclosure)
        skill_instruction = self.skill_loader.get_skill_content("plan-trip")
        if not skill_instruction:
            # Fallback: 如果读取失败，使用默认的简单指令
            skill_instruction = "请根据用户需求和偏好生成行程规划。"

        prompt = f"""你是一个高级行程规划专家。

【当前时间】
{current_date} {weekday}，当前季节是{current_season}

【用户需求】
{user_query}

{preferences_info}【所有收集的信息】
{json.dumps(all_info, ensure_ascii=False, indent=2)}

【任务说明与指南】
{skill_instruction}

【本次规划模式】
{mode_instruction}

【行程完整性硬性要求】
1. 如果事项信息提供了 duration_days、start_date 或 end_date，daily_plans 必须完整覆盖对应天数和日期，不得少生成或多生成某一天。
2. missing_info 只用于记录仍建议确认的可选细节。酒店门店、普通商务活动地点等非必要细节待确认时，仍应先给出可执行方案并将 planning_complete 设为 true。
3. 只有无法覆盖必需日期、存在未解决的硬约束冲突或时间冲突、或者无法生成可执行主体方案时，planning_complete 才设为 false。

【时间一致性硬性要求】
1. 每项活动的 time 必须完整覆盖描述中出现的明确出发时间、到达时间和交通耗时。
2. 同一天的活动必须按时间顺序排列且不能重叠，后一项开始时间不得早于前一项结束时间。
3. 固定会议、客户拜访等不可移动事件必须原样保留，并在前后安排必要的交通和缓冲。
4. 输出前逐项核对活动时间框、描述中的交通时间或耗时、下一项活动开始时间，发现矛盾必须先修正。

请直接输出 JSON 格式的行程规划，不要输出 Markdown、注释或额外解释。
JSON 字符串内部出现双引号时必须正确转义。
"""

        try:
            # JSON mode 先降低格式错误概率；如果内容仍损坏，下面只修复一次。
            response = await self.model(
                [{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            # 获取响应文本
            text = await extract_json_from_async_response(response)

            try:
                result = self._parse_and_validate(text)
            except ValueError as first_error:
                logger.warning(
                    "Itinerary output invalid, attempting one repair: %s",
                    first_error,
                )
                result = await self._repair_output(text, first_error)

            time_issues = find_itinerary_time_issues(result)
            if time_issues:
                logger.warning(
                    "Itinerary time conflicts detected, attempting one repair: %s",
                    time_issues,
                )
                try:
                    result = await self._repair_time_consistency(
                        result,
                        time_issues,
                        all_info,
                    )
                    remaining_issues = find_itinerary_time_issues(result)
                    if remaining_issues:
                        logger.warning(
                            "Itinerary still has time conflicts after repair: %s",
                            remaining_issues,
                        )
                        result = self._mark_time_conflicts_unresolved(
                            result,
                            remaining_issues,
                        )
                except Exception as repair_error:
                    # 保留原行程内容，但不能把已知有冲突的结果标成规划完成。
                    logger.warning(
                        "Itinerary time repair failed; keeping original result: %s",
                        repair_error,
                    )
                    result = self._mark_time_conflicts_unresolved(
                        result,
                        time_issues,
                    )

        except Exception as e:
            logger.error(f"Itinerary planning failed: {e}")
            # Ensure text is defined for logging even if extraction failed
            # 使用 locals().get 安全获取 text，防止 UnboundLocalError
            raw_text = locals().get('text', 'N/A')
            logger.error(f"Raw response text (first 500 chars): {str(raw_text)[:500]}")

            # 构建用户友好的错误消息
            error_detail = str(e)
            if "JSON" in error_detail or "parse" in error_detail.lower():
                user_message = "抱歉，模型返回的数据格式有误，无法解析行程信息。请稍后重试或简化您的需求描述。"
            else:
                user_message = f"行程规划过程中出现问题：{error_detail}"

            result = {
                "itinerary": {
                    "title": "行程规划",
                    "duration": "待完善",
                    "daily_plans": []
                },
                "planning_complete": False,
                "error": user_message,
                "technical_error": str(e)  # 保留技术细节用于调试
            }

        # 返回JSON字符串格式
        return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")

    @staticmethod
    def _parse_and_validate(text: str) -> Dict[str, Any]:
        """容错解析后校验下游展示和记忆更新依赖的最小结构。"""
        result = robust_json_parse(text, fallback=None)
        itinerary = result.get("itinerary")
        if not isinstance(itinerary, dict):
            raise ValueError("itinerary must be an object")
        if not isinstance(itinerary.get("daily_plans"), list):
            raise ValueError("itinerary.daily_plans must be a list")
        if not isinstance(result.get("planning_complete"), bool):
            raise ValueError("planning_complete must be a boolean")
        return result

    @staticmethod
    def _mark_time_conflicts_unresolved(
        result: Dict[str, Any],
        issues: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """保留已有方案并显式暴露未解决冲突，避免假装行程可执行。"""
        result["planning_complete"] = False
        result["time_consistency"] = {
            "status": "unresolved",
            "issues": issues,
        }
        itinerary = result.get("itinerary")
        if isinstance(itinerary, dict):
            notes = itinerary.get("notes")
            if not isinstance(notes, list):
                notes = []
                itinerary["notes"] = notes
            warning = "存在未能自动解决的时间冲突，请调整交通或可选活动后确认。"
            if warning not in notes:
                notes.append(warning)
        return result

    async def _repair_output(
        self,
        invalid_text: str,
        validation_error: Exception,
    ) -> Dict[str, Any]:
        """只修复一次 JSON 格式或最小字段结构，不重新规划行程。"""
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "你是JSON格式修复器。只修复格式和字段结构，保留原行程内容，"
                    "不要新增景点、日期或其他事实。只输出一个合法JSON对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "根对象必须包含 itinerary 和 planning_complete。"
                    "itinerary必须是对象，daily_plans必须是数组，"
                    "planning_complete必须是布尔值。"
                    "不要输出Markdown、注释或解释。\n"
                    f"校验错误：{validation_error}\n"
                    "待修复内容：\n"
                    f"{invalid_text[:12000]}"
                ),
            },
        ]
        response = await self.model(
            repair_messages,
            response_format={"type": "json_object"},
        )
        repaired_text = await extract_json_from_async_response(response)
        return self._parse_and_validate(repaired_text)

    async def _repair_time_consistency(
        self,
        original_result: Dict[str, Any],
        issues: List[Dict[str, Any]],
        planning_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """保护硬约束并对冲突部分执行一次最小范围的重新规划。"""
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "你是企业差旅行程的时间一致性修复器，需要对冲突部分执行最小"
                    "范围的重新规划。必须保护以下硬约束：出发和返程日期、城市顺序、"
                    "固定会议与客户拜访、企业预算和差旅政策、用户明确声明的必须或"
                    "禁止要求，以及上下文中已有的外部查询事实。普通用户偏好属于软"
                    "约束，只有在不违反上述硬约束时优先满足。优先移动可调整活动的"
                    "时间；仍无法解决时，允许调整、缩短、重新排序或删除景点、休闲、"
                    "用餐、休息等非必要活动。不得新增或编造车次、票价、天气、景点"
                    "及其他外部事实。修复后确保同日活动按时间顺序排列且不重叠，活动"
                    "时间框完整覆盖描述中的交通时刻和耗时。如果现有信息下无法同时"
                    "满足所有硬约束，保留硬约束并将planning_complete设为false，不得"
                    "通过移动固定活动伪造可行性。只输出合法JSON对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "【原规划上下文】\n"
                    f"{json.dumps(planning_context, ensure_ascii=False, indent=2)}\n\n"
                    "【确定性检查发现的问题】\n"
                    f"{json.dumps(issues, ensure_ascii=False, indent=2)}\n\n"
                    "【待修复行程】\n"
                    f"{json.dumps(original_result, ensure_ascii=False, indent=2)}"
                ),
            },
        ]
        response = await self.model(
            repair_messages,
            response_format={"type": "json_object"},
        )
        repaired_text = await extract_json_from_async_response(response)
        return self._parse_and_validate(repaired_text)
