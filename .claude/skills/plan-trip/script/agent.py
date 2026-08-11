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
from utils.booking_context import (
    build_booking_context,
    expected_booking_usage,
)
from utils.itinerary_quality_gate import (
    collect_itinerary_quality_issues,
    finalize_quality_gate,
    normalize_itinerary_result,
    prepare_event_data_for_planning,
    quality_issue_score,
)
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
        event_data = all_info.get("event_collection", {})
        if not isinstance(event_data, dict):
            event_data = {}
        event_data = prepare_event_data_for_planning(event_data)
        all_info["event_collection"] = event_data
        booking_context = build_booking_context(event_data)
        booking_fact_instruction = self._booking_fact_instruction(
            event_data,
            booking_context,
        )

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

【预订状态与事实来源】
{booking_fact_instruction}

【行程完整性硬性要求】
1. 如果事项信息提供了 duration_days、start_date 或 end_date，daily_plans 必须完整覆盖对应天数和日期，不得少生成或多生成某一天。
2. missing_info 只用于记录仍建议确认的可选细节。酒店门店、普通商务活动地点等非必要细节待确认时，仍应先给出可执行方案并将 planning_complete 设为 true。
3. 只有无法覆盖必需日期、存在未解决的硬约束冲突或时间可行性问题、或者无法生成可执行主体方案时，planning_complete 才设为 false。

【时间可行性硬性要求】
1. 每项活动的 time 必须完整覆盖描述中出现的明确出发时间、到达时间和交通耗时。
2. 同一天的活动必须按时间顺序排列且不能重叠，后一项开始时间不得早于前一项结束时间。
3. 固定会议、客户拜访等不可移动事件必须原样保留，并在前后安排必要的交通和缓冲。
4. 铁路交通活动的开始时间视为发车时间；此前必须明确留出至少30分钟用于进站、安检、检票和候车。可以单独安排候车活动，但前往车站或在车站用餐不能代替该缓冲。
5. 输出前逐项核对活动时间框、描述中的交通时间或耗时、出发前缓冲和下一项活动开始时间，发现矛盾必须先修正。

【最小结构化活动契约】
1. 每个activity必须输出type，取值只能是general、transport_booking、hotel_booking、fixed_event、business、meal、leisure、buffer、local_transport。
2. 每个activity必须有time，并且title、location、description至少有一个非空值，不得生成只有null的空活动。
3. time为HH:MM-HH:MM时，同时输出start_time和end_time；模糊时段可只写time=上午、下午或flexible。
4. 事项信息中的每个fixed_event都带event_id；必须生成type=fixed_event且fixed_event_ref等于对应event_id的活动。具体时间、地点和标题由代码使用原始事项统一渲染。
5. booking_context中存在outbound或return时，无论confirmed还是reference，都必须分别生成booking_ref=outbound或return的transport_booking活动；引用只表示去返程活动，不代表已经购票。

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

            # 先由代码统一生成可信预订字段，再一次性检查结构、硬约束、
            # 预订引用、事实来源和时间可行性，避免多次修复互相覆盖。
            result = normalize_itinerary_result(
                result,
                booking_context,
                event_data,
            )
            initial_issues = collect_itinerary_quality_issues(
                result,
                event_data,
                booking_context,
                trusted_context=all_info,
            )
            final_issues = initial_issues
            initial_blocking = [
                issue
                for issue in initial_issues
                if issue.get("severity") == "blocking"
            ]
            if initial_blocking:
                logger.warning(
                    "Itinerary quality gate found blocking issues; "
                    "attempting one unified repair: %s",
                    initial_blocking,
                )
                original_result = result
                try:
                    repaired_result = await self._repair_quality_gate(
                        original_result,
                        initial_blocking,
                        all_info,
                        booking_context,
                    )
                    repaired_result = normalize_itinerary_result(
                        repaired_result,
                        booking_context,
                        event_data,
                    )
                    repaired_issues = collect_itinerary_quality_issues(
                        repaired_result,
                        event_data,
                        booking_context,
                        trusted_context=all_info,
                    )
                    if quality_issue_score(repaired_issues) < quality_issue_score(
                        initial_blocking
                    ):
                        result = repaired_result
                        final_issues = repaired_issues
                    else:
                        logger.warning(
                            "Unified repair did not improve the candidate; "
                            "keeping the normalized original result."
                        )
                except Exception as repair_error:
                    logger.warning(
                        "Unified itinerary repair failed; keeping the "
                        "normalized original result: %s",
                        repair_error,
                    )

            result = finalize_quality_gate(result, final_issues)

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
    def _booking_fact_instruction(
        event_data: Dict[str, Any],
        booking_context: Dict[str, Dict[str, Any]],
    ) -> str:
        """根据用户确认状态生成事实使用白名单和参考模式约束。"""
        confirmed_items = []
        reference_items = []
        item_fields = (
            ("去程交通", "outbound_booking_status", "outbound_booking_details"),
            ("返程交通", "return_booking_status", "return_booking_details"),
            ("住宿", "hotel_booking_status", "hotel_booking_details"),
        )
        for label, status_field, detail_field in item_fields:
            if event_data.get(status_field) == "confirmed":
                confirmed_items.append({
                    "item": label,
                    "details": event_data.get(detail_field),
                })
            else:
                reference_items.append(label)

        lines = [
            "只有下方‘用户确认信息’可以作为已预订事实，必须原样保留，不得擅自修改。",
            "用户给出的去程和返程时段只是规划范围，不等于真实车次时刻。",
            f"用户确认信息：{json.dumps(confirmed_items, ensure_ascii=False)}",
            f"参考规划项目：{json.dumps(reference_items, ensure_ascii=False)}",
            "【无来源实时事实禁止规则】",
            "1. 参考规划项目不得生成具体车次、航班号、票价、房价、天气或温度。",
            "2. 住宿未确认时不得生成具体酒店门店；用户偏好只能作为选型原则。",
            "3. 不得把参考内容写成已预订、已确认或已安排。",
            "4. 交通未确认时使用‘根据时间范围选择合适交通’；住宿未确认时使用‘前往之后确定的住宿地点’；返程未确认时使用‘按之后确定的返程安排’。",
            "5. 可以生成会议、用餐和休息等规划时间段，但不得把这些规划时间冒充实时班次。",
            "6. 用户已确认的项目不得再写成‘未预订’‘待确认’或‘需要重新购买’，输出前必须逐项核对预订状态。",
            "【结构化预订引用契约】",
            f"可信booking_context：{json.dumps(booking_context, ensure_ascii=False)}",
            f"booking_usage必须等于：{json.dumps(expected_booking_usage(booking_context), ensure_ascii=False)}",
            "7. 根对象必须输出booking_usage；不得复制或改写可信预订详情。",
            "8. 使用去程或返程预订的活动必须输出type=transport_booking，并设置booking_ref=outbound或return。",
            "9. 使用住宿预订的活动必须输出type=hotel_booking，并设置booking_ref=hotel。",
            "10. 每个confirmed项目必须至少被一个活动引用；outbound和return即使是reference也必须创建对应活动，但不得生成具体事实。reference住宿只有确实需要住宿时才创建活动。",
            "11. 预订活动的location、description和transport只写规划意图，系统会根据booking_ref使用原始事实统一渲染。",
        ]
        return "\n".join(lines)

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

    async def _repair_quality_gate(
        self,
        original_result: Dict[str, Any],
        issues: List[Dict[str, Any]],
        planning_context: Dict[str, Any],
        booking_context: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """一次性修复统一质量门发现的所有阻断问题。"""
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "你是企业差旅行程统一修复器。根据问题清单修复整份行程，但只能"
                    "处理确定性阻断问题，不追求措辞完美。必须完整返回根对象和全部"
                    "daily_plans，不能只返回修改片段，不能删除原本正确的日期、城市、"
                    "固定会议、客户拜访、企业政策或用户明确禁止项。普通偏好是软"
                    "约束，不能覆盖企业政策。不得新增或编造车次、航班、票价、房价、"
                    "天气和温度；用户或企业政策明确给出的预算数字可以原样保留。"
                    "同日活动不得重叠，铁路发车前应明确预留至少30分钟进站、安检、"
                    "检票和候车。booking_usage必须与可信booking_context一致，已确认"
                    "预订必须使用合法booking_ref。修复完成且主体行程可执行时将"
                    "planning_complete设为true。只输出一个完整合法JSON对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "【可信规划上下文】\n"
                    f"{json.dumps(planning_context, ensure_ascii=False, indent=2)}\n\n"
                    "【可信预订上下文】\n"
                    f"{json.dumps(booking_context, ensure_ascii=False, indent=2)}\n\n"
                    "【统一质量门发现的阻断问题】\n"
                    f"{json.dumps(issues, ensure_ascii=False, indent=2)}\n\n"
                    "【必须完整保留并修复的候选行程】\n"
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
