"""
意图识别智能体 IntentionRecognitionAgent
职责：准确识别用户意图，并进行智能体调度

核心功能：
1. 多意图识别和分类：融合上下文对模糊意图进行消歧
2. 智能体调度决策：基于预定义的触发条件和业务规则，根据识别结果决定调用哪些子智能体
3. Query改写：标准化用户口语化的query输入，补全上下文信息，提取和重组关键信息
4. 显示推理：输出的两段式结构（推理过程 + JSON决策），提升意图识别准确度

架构：
- 使用单一LLM（用户配置的模型）
- 输入：用户query（自然语言）
- 输出：推理过程生成（包含reasoning+原因） + 多意图识别（原因） + 智能Query改写 + 构建结构化决策
"""
from agentscope.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Union, List
import json
import logging
from utils.skill_loader import SkillLoader
from utils.json_parser import extract_json_from_async_response, robust_json_parse
from utils.intention_output import validate_intention_result
from utils.intention_routing import normalize_intention_routing

logger = logging.getLogger(__name__)


class IntentionAgent(AgentBase):
    """意图识别智能体（IntentionRecognitionAgent）"""

    def __init__(self, name: str = "IntentionRecognitionAgent", model=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        self.conversation_history = []
        self.skill_loader = SkillLoader()

    async def reply(self, x: Optional[Union[Msg, List[Msg]]] = None) -> Msg:
        """
        意图识别主流程
        1. 推理过程生成
        2. 多意图识别
        3. 智能Query改写
        4. 构建结构化决策
        """
        if x is None:
            return Msg(name=self.name, content=json.dumps({}), role="assistant")

        # 获取用户查询
        if isinstance(x, list):
            user_query = x[-1].content if x else ""
            # 提取历史对话，保留角色信息
            self.conversation_history = []
            for msg in x[:-1]:
                if hasattr(msg, 'content') and hasattr(msg, 'role'):
                    # 区分处理不同角色的消息
                    if msg.role == "system":
                        # 长期记忆（system）- 完整保留，不截断
                        self.conversation_history.append(f"[系统记忆]\n{msg.content}")
                    else:
                        # 对话历史（user/assistant）- 适当截断但保留更多信息
                        role_name = "用户" if msg.role == "user" else "助手"
                        content = msg.content[:800] if len(msg.content) > 800 else msg.content
                        if len(msg.content) > 800:
                            content += "..."
                        self.conversation_history.append(f"{role_name}: {content}")
        else:
            user_query = x.content

        # 构建上下文
        # 策略：长期记忆始终保留，短期对话全部保留（已在 cli.py 控制数量）
        context_parts = []
        system_memory = None
        dialogue_history = []

        for item in self.conversation_history:
            if item.startswith("[系统记忆]"):
                system_memory = item  # 保存长期记忆
            else:
                dialogue_history.append(item)  # 保存对话历史

        # 组装上下文：长期记忆 + 全部对话
        if system_memory:
            context_parts.append(system_memory)
        if dialogue_history:
            context_parts.extend(dialogue_history) 

        context_str = "\n".join(context_parts) if context_parts else "无历史对话"

        # 获取当前时间
        from datetime import datetime
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][datetime.now().weekday()]

        # 动态获取 Skills 描述
        skill_mapping = {
            "memory-query": "memory_query",
            "plan-trip": "itinerary_planning", 
            "preference": "preference",
            "query-info": "information_query",
            "ask-question": "rag_knowledge",
            "event-collection": "event_collection"
        }
        
        dynamic_skills_prompt = self.skill_loader.get_skill_prompt(skill_mapping)
        
        # 构建意图识别Prompt
        prompt = f"""你是一个高级意图识别专家（IntentionRecognitionAgent）。请分析用户查询，识别意图并输出结构化的决策。
        
【当前时间】
{current_time} {weekday}
（重要：当用户说"2月28日"或"明天"等相对时间时，请根据当前时间进行推断完整日期）

【用户Query】
{user_query}

【对话历史上下文】
{context_str}

【可调度的子智能体 (Skills)】
{dynamic_skills_prompt}

【重要 - 意图区分原则】
请基于语义理解判断意图，不要机械匹配关键词。同一个词在不同语境下可能对应不同意图：
- "我去过北京吗？" → memory_query（询问自己的历史）
- "北京怎么样？" / "北京有什么好玩的？" → information_query（询问客观信息）
- "我想去北京" → itinerary_planning（规划未来行程）

优先级规则：
- memory_query 优先于 information_query（当问题涉及用户自己的历史时）
- 如果用户明确询问"我的"、"我过去的"，必须识别为 memory_query
- 查询或引用已保存的偏好属于 memory_query，只有新增、修改、删除偏好才调用 preference
- 只要调用 itinerary_planning，就必须先调用 event_collection 收集并校验行程字段

【任务要求】
请按以下步骤进行分析：

**第1步：生成简要决策依据**
- 分析用户query的核心诉求
- 识别query中的关键实体和意图信号
- 判断是否需要结合对话历史进行消歧
- reasoning 只用一句话说明调度依据，不复述用户原文，不换行

**第2步：多意图识别（原因）**
- 识别所有可能的用户意图（可以是多个）
- 为每个意图分配置信度（0-1之间）
- 说明为什么识别出该意图的原因

**第3步：智能Query改写**
- 识别口语化表达，进行标准化
- 补全省略的上下文信息
- 提取和重组关键信息

**第4步：构建结构化决策**
- 基于识别的意图，决定调用哪些子智能体
- 说明调用顺序和优先级
- 输出结构化的调用策略

【输出格式要求】
必须严格按照以下JSON格式输出（**只输出JSON，不要有其他文本**）：

{{
    "reasoning": "一句话说明意图判断和调度依据，不复述用户原文",

    "intents": [
        {{
            "type": "意图类型（使用 itinerary_planning、preference、information_query、memory_query、rag_knowledge、event_collection）",
            "confidence": 0.95,
            "description": "该意图的具体说明",
            "reason": "为什么识别出该意图的原因"
        }}
    ],

    "key_entities": {{
        "origin": "出发地（如果有）",
        "destination": "目的地（如果有）",
        "date": "日期（如果有）",
        "duration": "时长（如果有）",
        "other": "其他关键信息"
    }},

    "rewritten_query": "标准化、补全后的查询内容",

    "agent_schedule": [
        {{
            "agent_name": "子智能体名称",
            "priority": 1,
            "reason": "调用该智能体的原因和依据",
            "expected_output": "期望该智能体提供什么输出"
        }}
    ]
}}

【重要提示 - 优先级设置规则】
优先级数字相同的智能体会**并行执行**，不同优先级按顺序批次执行。

**所有智能体优先级分组：**

**Priority 1（并行执行）- 信息收集类：**
- memory_query: 记忆查询智能体
- event_collection: 事项收集智能体
- preference: 偏好管理智能体
- information_query: 信息查询智能体（联网搜索）
- rag_knowledge: RAG知识库智能体（查询企业知识库）

**Priority 2（依赖 Priority 1）- 行程规划类：**
- itinerary_planning: 行程规划智能体（需要事项收集的结果）

**说明：**
- Priority 1 的智能体都是信息获取，互不依赖，可并行执行提升速度
- Priority 2 的智能体需要使用 Priority 1 收集的信息
- 示例：用户说"我要从天津去北京，喜欢住汉庭"
  → Priority 1: preference + event_collection（并行）
  → Priority 2: itinerary_planning（使用 Priority 1 的结果）

请开始分析，直接输出JSON：
"""

        # 调用 LLM。连接、超时等异常继续抛给上层 retry_with_backoff；
        # 只有响应格式或结构异常才在本 Agent 内尝试一次定向修复。
        messages = [
            {"role": "system", "content": "你是一个高级意图识别专家。只输出JSON格式的结果，不要输出其他文本。"},
            {"role": "user", "content": prompt}
        ]
        try:
            # JSON mode 仅作用于 IntentionAgent，不影响其他需要自然语言输出的 Agent。
            response = await self.model(
                messages,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.error(f"Intent model call failed: {e}")
            raise

        # 消费流时发生的网络异常不属于格式问题，应继续抛给上层退避重试。
        text = await extract_json_from_async_response(response)
        should_normalize_routing = True
        try:
            result = self._parse_and_validate(text)
        except ValueError as first_error:
            logger.warning(
                "Intent response invalid, attempting one repair: %s",
                first_error,
            )
            result, should_normalize_routing = await self._repair_or_default(
                user_query=user_query,
                invalid_text=text,
                validation_error=first_error,
            )

        if should_normalize_routing:
            result = normalize_intention_routing(result, user_query)

        # 将结果转换为JSON字符串，因为Msg的content必须是字符串
        return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")

    @staticmethod
    def _parse_and_validate(text: str) -> dict:
        """先做 JSON 容错解析，再校验调度链路真正依赖的数据结构。"""
        return validate_intention_result(robust_json_parse(text))

    async def _repair_or_default(
        self,
        user_query: str,
        invalid_text: str,
        validation_error: Exception,
    ) -> tuple[dict, bool]:
        """只请求模型修复一次；仍失败时保留项目原有降级行为。"""
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "你是JSON格式修复器。只修复格式和字段结构，不改变原有意图。"
                    "只输出一个合法JSON对象，不要输出解释或Markdown。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始用户问题：{user_query}\n"
                    "必须包含 reasoning、intents、key_entities、rewritten_query、"
                    "agent_schedule 五个字段。agent_schedule 必须是数组，"
                    "其中 agent_name 只能使用 memory_query、itinerary_planning、"
                    "preference、information_query、rag_knowledge、event_collection，"
                    "priority 必须是大于等于1的整数。\n"
                    f"校验错误：{validation_error}\n"
                    "待修复内容：\n"
                    f"{invalid_text[:4000]}"
                ),
            },
        ]

        # 模型连接异常仍交给上层重试；这里只处理第二次结果仍无法解析的情况。
        response = await self.model(
            repair_messages,
            response_format={"type": "json_object"},
        )
        repaired_text = await extract_json_from_async_response(response)
        try:
            result = self._parse_and_validate(repaired_text)
            logger.info("Intent response repaired successfully")
            return result, True
        except ValueError as repair_error:
            logger.error("Intent response repair failed: %s", repair_error)
            return self._default_result(user_query, repair_error), False

    @staticmethod
    def _default_result(user_query: str, error: Exception) -> dict:
        """连续两次结构化输出失败时，沿用现有 information_query 降级。"""
        return {
            "reasoning": f"意图识别出错，使用默认策略。错误: {str(error)}",
            "intents": [
                {
                    "type": "information_query",
                    "confidence": 0.5,
                    "description": "默认查询意图",
                    "reason": "无法解析用户意图，使用默认策略",
                }
            ],
            "key_entities": {},
            "rewritten_query": user_query,
            "agent_schedule": [
                {
                    "agent_name": "information_query",
                    "priority": 1,
                    "reason": "默认查询",
                    "expected_output": "查询结果",
                }
            ],
        }
