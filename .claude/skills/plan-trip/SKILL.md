---
name: plan-trip
description: Use this skill when the user wants to plan a trip or asks for itinerary planning. Triggers when user says "规划行程", "安排路线", "我要去XX", "从XX到XX", or provides trip details like dates and destinations. This skill orchestrates IntentionAgent, EventCollectionAgent, and ItineraryPlanningAgent; all agents take model=model and are async.
---

# Plan Trip (行程规划)

为用户规划出行行程：意图识别 → 事项收集（出发地、目的地、日期等）→ 行程规划。所有 Agent 均使用 **model 对象**，且 **reply() 均为 async**。

## When to Use

- 用户说「规划行程」「从XX到XX」「X月X日去北京」等

## Agents（按顺序）

1. **IntentionAgent** — 识别意图、改写 query，并提取 `planning_signals` 语义信号
2. **EventCollectionAgent** — 提取出发地、目的地、日期、目的等  
3. **ItineraryPlanningAgent** — 生成行程（每日安排、交通、住宿建议等）

最终规划模式不由 IntentionAgent 直接决定。策略层会结合 `planning_signals`、
`trip_purpose` 和 `fixed_events` 进行确定性决策；关键词规则只作为旧输出的兼容兜底。

## 统一模型与异步

- 先创建 `OpenAIChatModel`（来自 `config.LLM_CONFIG`），再传给各 Agent 的 **model** 参数（本项目无 `model_config_name`）。
- 三个 Agent 的 `reply()` 都是 **async**，需 **await**。

## 调用示例（简化链式）

```python
import asyncio
import json
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from config_agentscope import init_agentscope
from config import LLM_CONFIG
from agents.intention_agent import IntentionAgent
from agents.event_collection_agent import EventCollectionAgent
from agents.itinerary_planning_agent import ItineraryPlanningAgent

async def plan_trip(user_query: str):
    init_agentscope()
    model = OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={"base_url": LLM_CONFIG["base_url"], "timeout": 60},
        temperature=LLM_CONFIG.get("temperature", 0.7),
        max_tokens=LLM_CONFIG.get("max_tokens", 2000),
    )
    user_msg = Msg(name="user", content=user_query, role="user")

    # 1. 意图识别
    intention_agent = IntentionAgent(name="IntentionAgent", model=model)
    intention_result = await intention_agent.reply(user_msg)
    intention_data = json.loads(intention_result.content)
    rewritten_query = intention_data.get("rewritten_query", user_query)

    # 2. 事项收集（传入 context 格式，与 OrchestrationAgent 一致）
    context = {"rewritten_query": rewritten_query, "user_preferences": {}}
    event_input = Msg(name="Orchestrator", content=json.dumps({"context": context}), role="user")
    event_agent = EventCollectionAgent(name="EventCollectionAgent", model=model)
    event_result = await event_agent.reply(event_input)
    event_data = json.loads(event_result.content) if isinstance(event_result.content, str) else event_result.content

    # 3. 行程规划（传入 previous_results，包含 event_collection 结果）
    previous_results = [{"agent_name": "event_collection", "data": event_data}]
    plan_input = Msg(
        name="Orchestrator",
        content=json.dumps({"context": context, "previous_results": previous_results}, ensure_ascii=False),
        role="user",
    )
    plan_agent = ItineraryPlanningAgent(name="ItineraryPlanningAgent", model=model)
    plan_result = await plan_agent.reply(plan_input)
    plan_data = json.loads(plan_result.content) if isinstance(plan_result.content, str) else plan_result.content
    return plan_data

# 使用
result = asyncio.run(plan_trip("规划一下2月27日从上海到北京的路程"))
# result: {"itinerary": {"title", "duration", "route", "daily_plans", "notes", ...}, "planning_complete": bool}
```

## EventCollectionAgent 输出字段（示例）

- 基础行程：`origin`, `destination`, `start_date`, `end_date`, `duration_days`, `trip_purpose`
- 时间范围：`departure_time_window`, `return_time_window`
- 预订状态：`outbound_booking_status`, `return_booking_status`, `hotel_booking_status`
- 确认详情：`outbound_booking_details`, `return_booking_details`, `hotel_booking_details`
- 派生状态：`missing_info`

## ItineraryPlanningAgent 输出字段（示例）

- `itinerary`: `title`, `duration`, `route`, `daily_plans`, `notes`, `estimated_budget` 等
- `booking_usage`: 声明去程、返程和住宿使用确认预订还是参考方案
- 预订相关活动：使用 `type` 和 `booking_ref` 引用可信 `booking_context`
- `booking_summary`: 由代码根据原始事实生成，不由模型自由改写
- `planning_complete`: bool

## 错误与缺失信息

- 若意图解析非 JSON，可提示用户重新描述。
- 协调器先补齐出发地、目的地、日期和行程长度，再询问去返程时段与预订状态。
- 用户说“时间不限”时，时间范围记为 `flexible`，不能反复追问。
- 用户没有预订或选择参考方案时，不要求具体车次和酒店详情。


## 行程规划 Prompt 指南

【核心原则】
1. **协调器完成必要澄清后提供有价值的行程规划**
2. **不要因为缺少天气、具体车次或酒店门店而拒绝参考规划**
3. **先判断行程目的；企业差旅必须优先满足商务任务，普通旅行再重点安排游览**
4. 用户确认的预订详情必须原样保留；未确认项目只能使用通用表达

【规划策略】
- 企业差旅：优先安排固定会议、客户拜访、必要交通、工作准备、用餐和休息
- 用户未要求旅游：不要为了填满时间主动加入景点，也不要虚构会议或客户拜访
- 用户明确要求适量空闲活动：整段行程最多安排1至2项、每项不超过2小时、靠近商务地点且可取消的休闲活动
- 普通旅行：根据目的地、日期和用户兴趣安排合理的游览路线
- 必要的出发地、日期、去返程时段和预订状态由协调器先询问，不得在规划阶段自行假设
- 如果缺少天气信息：只给通用的季节性准备建议，不得编造具体天气或温度
- 如果缺少开放信息：推荐常规开放的景点，提醒提前确认

【预订状态与事实来源】
- `booking_status=confirmed`：只能使用用户提供的对应详情，并明确说明来自用户已预订信息
- `booking_status=reference`：不得生成具体车次、航班号、票价、房价、天气、温度和具体酒店门店
- 规划 Agent可以读取完整 `booking_context` 进行时间和地点编排，但不能复制或修改其中的确定事实
- 交通预订活动输出 `type=transport_booking`，并使用 `booking_ref=outbound|return`
- 住宿预订活动输出 `type=hotel_booking`，并使用 `booking_ref=hotel`
- 根对象输出 `booking_usage`；代码校验引用后使用原始 `booking_context` 统一渲染预订摘要和活动文案
- 参考交通使用“根据时间范围选择合适交通”，参考住宿使用“前往之后确定的住宿地点”
- 未确认返程使用“按之后确定的返程安排”，不得写成已经安排或预订
- 用户偏好只能作为选择原则，不能作为本次预订事实

【行程规划要点】
1. 根据行程目的控制活动数量；商务差旅不得用景点填满空闲时间
2. 考虑各活动地点之间的交通时间和距离
3. 安排午餐、晚餐时间和推荐地点
4. 给出大致的时间安排（如9:00-12:00, 13:00-17:00等）
5. 提供交通方式建议（地铁、打车、步行等）
6. 每项活动的时间框必须完整覆盖描述中的明确交通时刻和耗时
7. 同一天的活动按时间顺序排列且不得重叠，后一项不得早于前一项结束
8. 固定会议和客户拜访不得移动，前后需要预留必要的交通与缓冲时间
9. 已提供行程天数或起止日期时，daily_plans 必须完整覆盖所有日期，不得缺天
10. 铁路交通发车前至少明确预留30分钟进站、安检、检票和候车；前往车站或在车站用餐不能替代该缓冲

【完成状态】
- `missing_info` 可以记录仍建议确认的可选细节，不等于规划失败
- 已生成完整天数且不存在未解决的硬约束或时间可行性问题时，`planning_complete` 应为 `true`
- 只有主体方案无法完成或仍有硬冲突时，`planning_complete` 才为 `false`

【时间可行性调整原则】
1. 日期、城市顺序、固定商务活动、企业政策和用户明确声明的必须/禁止要求属于硬约束，不得移动或违反
2. 普通偏好、景点、休闲、用餐和休息属于软安排，可以在保护硬约束的前提下调整、缩短、重排或删除
3. 不得为了填满行程编造新的车次、票价、天气或其他外部事实
4. 如果现有信息下无法解决问题，必须标记规划未完成并说明原因，不能假装行程可执行

【任务】
基于已有信息生成实用的行程规划：
1. **必须给出与行程目的匹配的具体活动安排**，不能只说"需要补充信息"
2. 在daily_plans中给出详细的时间表和活动地点
3. 在notes中补充注意事项和需要确认的信息
4. 在missing_info中列出建议用户补充的信息（但不影响规划）

【输出格式】(严格JSON)
{{
    "itinerary": {{
        "title": "北京3日游",
        "duration": "3天",
        "route": "北京 -> 北京",
        "daily_plans": [
            {{
                "day": 1,
                "date": "2024-02-27",
                "city": "北京",
                "theme": "历史文化之旅",
                "activities": [
                    {{
                        "time": "09:00-12:00",
                        "location": "故宫博物院",
                        "description": "游览故宫，感受皇家建筑群的宏伟...",
                        "transport": "地铁1号线天安门东站"
                    }},
                    {{
                        "time": "15:00-15:30",
                        "type": "transport_booking",
                        "booking_ref": "return",
                        "location": "返程交通",
                        "description": "按已确认返程安排前往车站"
                    }}
                ],
                "meals": {{ "lunch": "...", "dinner": "..." }}
            }}
        ],
        "notes": ["建议提前7天预约故宫门票..."],
        "estimated_budget": "约2000元"
    }},
    "booking_usage": {{
        "outbound": "use_confirmed_booking",
        "return": "use_confirmed_booking",
        "hotel": "use_reference_plan"
    }},
    "planning_complete": true
}}
