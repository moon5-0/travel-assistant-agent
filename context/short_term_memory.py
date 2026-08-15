"""
短期记忆 (Short-term Memory)
存储当前会话最近的对话历史，用于理解上下文和消歧
"""
from typing import List, Dict, Any
from datetime import datetime
import logging

from .session_store import SessionStore

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    短期记忆：存储最近的对话历史
    - 存储最近 5-10 轮对话
    - 自动淘汰旧消息
    - 用于上下文理解
    """

    def __init__(
        self,
        max_turns: int = 10,
        *,
        session_id: str = "default_session",
        session_store: SessionStore = None,
    ):
        """
        初始化短期记忆

        Args:
            max_turns: 最大保存轮数（一轮 = 一对用户-助手消息）
        """
        self.max_turns = max_turns
        self.session_id = session_id
        if session_store is None:
            raise ValueError("ShortTermMemory 必须注入 SessionStore")
        self.session_store = session_store

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """兼容旧调用；真实状态由 SessionStore 管理。"""
        return self.session_store.get_recent_messages(self.session_id)

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """
        添加消息到短期记忆

        Args:
            role: 角色 (user/assistant)
            content: 消息内容
            metadata: 额外的元数据
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        max_messages = self.max_turns * 2
        self.session_store.add_message(
            self.session_id,
            message,
            max_messages=max_messages,
        )

        logger.debug(f"Added message to short-term memory: {role}")

    def get_recent_context(self, n_turns: int = None) -> List[Dict[str, Any]]:
        """
        获取最近 n 轮对话

        Args:
            n_turns: 获取轮数，默认为全部

        Returns:
            最近的消息列表
        """
        if n_turns is None:
            return self.session_store.get_recent_messages(self.session_id)

        # n轮 = 2n条消息
        n_messages = n_turns * 2
        return self.session_store.get_recent_messages(
            self.session_id,
            limit=n_messages,
        )

    def get_context_string(self, n_turns: int = 5) -> str:
        """
        获取最近对话的字符串表示

        Args:
            n_turns: 获取轮数

        Returns:
            格式化的对话字符串
        """
        messages = self.get_recent_context(n_turns)
        if not messages:
            return "无历史对话"

        lines = []
        for msg in messages:
            role_name = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role_name}: {msg['content']}")

        return "\n".join(lines)

    def clear(self):
        """清空短期记忆"""
        self.session_store.clear_messages(self.session_id)
        logger.info("Short-term memory cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 只读取一次 Redis，避免为了一份统计信息发起多次网络往返。
        messages = self.messages
        return {
            "total_messages": len(messages),
            "max_turns": self.max_turns,
            "oldest_message_time": messages[0]["timestamp"] if messages else None,
            "newest_message_time": messages[-1]["timestamp"] if messages else None
        }
