"""
事项收集智能体
职责：收集用户的出发地/事项地点/事项时间/返程地

核心功能：
- 提取出发地、目的地、时间、返程地等基础信息
- 识别缺失信息并提示
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Union, List
import json
import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from utils.json_parser import extract_json_from_async_response, robust_json_parse

logger = logging.getLogger(__name__)

BOOKING_STATUS_FIELDS = (
    "outbound_booking_status",
    "return_booking_status",
    "hotel_booking_status",
)
BOOKING_DETAIL_FIELDS = (
    "outbound_booking_details",
    "return_booking_details",
    "hotel_booking_details",
)
ALLOWED_BOOKING_STATUSES = {"confirmed", "reference"}


class EventCollectionAgent(AgentBase):
    """事项收集智能体"""

    def __init__(self, name: str = "EventCollectionAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        if x is None:
            return Msg(name=self.name, content={}, role="assistant")

        # 解析输入内容
        content = x.content if not isinstance(x, list) else x[-1].content

        # 如果content是JSON字符串，解析它
        if isinstance(content, str):
            try:
                data = json.loads(content)
                context = data.get("context", {})
                # TODO(优化): IntentionAgent 已在 context["key_entities"] 中做过一次粗粒度实体提取，
                # 当前仍只从 rewritten_query 重新提取。后续可复用并校验 key_entities，减少重复 LLM
                # 工作和两次提取结果不一致的风险；行程字段仍以本 Agent 的标准化结果为准。
                user_query = context.get("rewritten_query", "") or str(data)
                user_preferences = context.get("user_preferences", {})
            except json.JSONDecodeError:
                user_query = content
                user_preferences = {}
        else:
            user_query = str(content)
            user_preferences = {}

        # 构建用户背景信息
        background_info = ""
        if user_preferences:
            bg_parts = ["【用户背景信息】（只可按下方规则补全，不可用于猜测日期）"]
            if user_preferences.get("home_location"):
                bg_parts.append(f"• 家庭住址: {user_preferences['home_location']}")
            if user_preferences.get("hotel_brands"):
                bg_parts.append(f"• 酒店偏好: {', '.join(user_preferences['hotel_brands'])}")
            if user_preferences.get("airlines"):
                bg_parts.append(f"• 航空偏好: {', '.join(user_preferences['airlines'])}")

            if len(bg_parts) > 1:
                background_info = "\n".join(bg_parts) + "\n\n"

        # 获取当前时间
        from datetime import datetime
        current_date = datetime.now().strftime("%Y年%m月%d日")
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][datetime.now().weekday()]

        prompt = f"""你是事项收集专家，负责提取旅行的基础信息。

【当前时间】
{current_date} {weekday}

{background_info}【用户输入】
{user_query}

【提取要求】
请尽可能提取以下信息：
1. origin - 出发地
2. destination - 目的地
3. start_date - 出发日期（YYYY-MM-DD格式）
4. end_date - 返程日期
5. duration_days - 行程天数
6. return_location - 返程地
7. trip_purpose - 行程目的
8. departure_time_window - 去程大概时段，例如“上午”“10点前”“flexible”
9. return_time_window - 返程大概时段，例如“下午”“会议结束后”“flexible”
10. outbound_booking_status - 去程预订状态，只能是confirmed、reference或null
11. return_booking_status - 返程预订状态，只能是confirmed、reference或null
12. hotel_booking_status - 酒店预订状态，只能是confirmed、reference或null
13. outbound_booking_details - 用户明确提供的已订去程信息，否则为null
14. return_booking_details - 用户明确提供的已订返程信息，否则为null
15. hotel_booking_details - 用户明确提供的已订酒店信息，否则为null

【日期处理规则】（重要）
- 当前时间是{current_date}
- 用户说"2月27日"或"2.27"等相对时间，请根据当前时间推断完整日期（年月日）
- 用户说"明天"、"后天"、"下周"等相对时间，请根据当前时间计算具体日期
- 如果用户完全没有提到日期或相对时间，start_date和end_date必须设为null
- 不得因为“尽快出行”或当前日期而默认填入今天、明天或其他日期
- 所有日期必须输出完整的YYYY-MM-DD格式

【特殊处理】
- 对于"北京一日游"这类：destination和origin都设为北京
- 对于"一日游"：duration_days设为1
- 如果用户没说出发地，但有家庭住址信息，可推断出发地为家庭住址

【实体可信度规则】（重要）
- 只提取用户明确说出的行程目的，不要根据“规划行程”等表达猜测为旅游或出差
- 用户没有说明行程目的时，trip_purpose必须设为null
- trip_purpose是可选字段，值为null时不需要加入missing_info
- 用户没有明确说明去程或返程的大概时段时，对应time_window必须设为null；“时间不限”“几点都行”统一输出flexible
- 用户明确说已经预订时，对应booking_status才设为confirmed，并原样提取其提供的详情
- 用户明确说未预订、不需要推荐或先看参考方案时，对应booking_status设为reference
- 用户没有说明某一项预订状态时，该booking_status必须设为null，不得自行猜测
- 酒店品牌偏好、航空偏好和历史偏好不等于已经预订，不得据此设置confirmed
- booking_status为reference或null时，对应booking_details必须设为null

【输出格式】(严格JSON)
{{
    "origin": "北京",
    "destination": "北京",
    "start_date": "2026-02-27",
    "end_date": "2026-02-27",
    "duration_days": 1,
    "return_location": "北京",
    "trip_purpose": null,
    "departure_time_window": "上午",
    "return_time_window": "下午",
    "outbound_booking_status": "reference",
    "return_booking_status": "reference",
    "hotel_booking_status": "confirmed",
    "outbound_booking_details": null,
    "return_booking_details": null,
    "hotel_booking_details": "北京国贸全季酒店",
    "missing_info": [],
    "extracted_count": 12,
    "summary": "北京一日游，2月27日"
}}

缺失的信息在missing_info中列出，对应字段设为null。
"""

        try:
            # 实体抽取必须返回结构化对象；JSON mode 能避免长推理后正文为空，
            # 也与 IntentionAgent、ItineraryPlanningAgent 的调用方式保持一致。
            response = await self.model(
                [{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            text = await extract_json_from_async_response(response)
            result = robust_json_parse(text)
            for status_field, detail_field in zip(
                BOOKING_STATUS_FIELDS,
                BOOKING_DETAIL_FIELDS,
            ):
                status = result.get(status_field)
                if status not in ALLOWED_BOOKING_STATUSES:
                    result[status_field] = None
                    result[detail_field] = None
                elif status == "reference":
                    result[detail_field] = None
        except Exception as e:
            logger.error(f"Event collection failed: {e}")
            result = {
                "missing_info": ["所有信息"],
                "extracted_count": 0,
                "error": str(e)
            }

        # 返回JSON字符串格式
        return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")
