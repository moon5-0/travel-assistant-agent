"""
偏好智能体
职责：收集用户的长期偏好
如"我的家在XXX"、"我喜欢XXX酒店"
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


class PreferenceAgent(AgentBase):
    """偏好智能体"""

    def __init__(self, name: str = "PreferenceAgent", model=None, memory_manager=None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model
        self.memory_manager = memory_manager
        from utils.skill_loader import SkillLoader
        self.skill_loader = SkillLoader()

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
                user_query = context.get("rewritten_query", "") or str(data)
            except json.JSONDecodeError:
                user_query = content
        else:
            user_query = str(content)

        # 获取当前已保存的偏好
        current_preferences = {}
        if self.memory_manager:
            current_preferences = self.memory_manager.long_term.get_preference()

        # 格式化当前偏好，便于展示
        current_prefs_str = json.dumps(current_preferences, ensure_ascii=False, indent=2)

        # 动态读取 Prompt 指令 (Progressive Disclosure)
        skill_instruction = self.skill_loader.get_skill_content("preference")
        if not skill_instruction:
            skill_instruction = "请分析用户的偏好。"

        prompt = f"""你是用户偏好分析专家，负责提取用户的长期偏好信息。

【当前已保存的用户偏好】
{current_prefs_str}

【新的用户输入】
{user_query}

【任务说明】
{skill_instruction}

请直接输出JSON：
"""

        try:
            # 调用模型
            response = await self.model([
                {"role": "user", "content": prompt}
            ])

            text = await extract_json_from_async_response(response)
            result = robust_json_parse(text)
        except Exception as e:
            logger.error(f"Preference collection failed: {e}")
            result = {"has_preferences": False, "error": str(e)}

        return Msg(name=self.name, content=json.dumps(result, ensure_ascii=False), role="assistant")
