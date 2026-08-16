"""
记忆管理器 (Memory Manager)
统一管理两层记忆，提供简单的API
"""
from typing import Dict, Any, List, Optional
from .short_term_memory import ShortTermMemory
from .memory_repository import LongTermMemoryRepository
from .session_store import SessionStore
from .session_store_factory import create_session_store
from .sqlite_memory_repository import SQLiteMemoryRepository
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    记忆管理器：统一管理两层记忆
    - 短期记忆：最近对话（会话级）
    - 长期记忆：用户偏好和历史（跨会话）
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        storage_path: str = "data/memory",
        llm_model=None,
        session_store: SessionStore = None,
        repository: LongTermMemoryRepository = None,
    ):
        """
        初始化记忆管理器

        Args:
            user_id: 用户ID
            session_id: 会话ID
            storage_path: 长期记忆存储路径
            llm_model: LLM模型实例（用于总结长期记忆）
        """
        self.user_id = user_id
        self.session_id = session_id
        self.llm_model = llm_model

        # SessionStore 管理当前会话状态。正式运行默认使用 Redis；测试可注入
        # 遵循同一接口的隔离客户端，不允许连接失败后静默切换本地内存。
        self.session_store = session_store or create_session_store(user_id)
        self.short_term = ShortTermMemory(
            max_turns=10,
            session_id=session_id,
            session_store=self.session_store,
        )

        # Repository 管理跨会话数据。默认使用 SQLite；调用方仍可注入测试
        # Repository，所以上层 Agent 不需要知道具体数据库实现。
        self.long_term = repository or SQLiteMemoryRepository(
            user_id,
            storage_path,
        )
        self.repository = self.long_term

        logger.info(f"Memory manager initialized for user {user_id}, session {session_id}")

    # ========== 短期记忆操作 ==========

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """
        添加消息到短期记忆和长期记忆

        Args:
            role: 角色 (user/assistant)
            content: 消息内容
            metadata: 元数据
        """
        # 添加到短期记忆（当前会话）
        self.short_term.add_message(role, content, metadata)

        # 同时添加到长期记忆（跨会话持久化）
        self.long_term.add_chat_message(role, content, self.session_id)

    # ========== 长期记忆操作 ==========
    # 注意：大部分方法直接使用 self.short_term 和 self.long_term 即可，无需封装

    # ========== 当前任务状态 ==========

    def get_pending_trip(self) -> Dict[str, Any]:
        """读取当前会话尚未补全的行程草稿。"""
        return self.session_store.get_pending_trip(self.session_id)

    def save_pending_trip(self, trip_data: Dict[str, Any]) -> None:
        """覆盖保存当前会话的行程草稿。"""
        self.session_store.save_pending_trip(self.session_id, trip_data)

    def clear_pending_trip(self) -> None:
        """清除当前会话的行程草稿。"""
        self.session_store.clear_pending_trip(self.session_id)

    def clear_session_state(self) -> None:
        """清除最近对话和待补全任务，不影响长期记忆。"""
        self.session_store.clear_session(self.session_id)

    # ========== 综合查询 ==========

    def get_full_context(self) -> Dict[str, Any]:
        """
        获取完整上下文（两层记忆）

        Returns:
            完整上下文字典
        """
        return {
            "short_term": {
                "recent_dialogue": self.short_term.get_recent_context(5),
                "context_string": self.short_term.get_context_string(5),
                "pending_trip": self.get_pending_trip(),
                "statistics": self.short_term.get_statistics(),
            },
            "long_term": {
                "preferences": self.long_term.get_preference(),
                "chat_history": self.long_term.get_chat_history(10),
                "session_summaries": self.long_term.get_session_summaries(
                    limit=3,
                    exclude_session_id=self.session_id,
                ),
                "trip_history": self.long_term.get_trip_history(5),
                "frequent_destinations": self.long_term.get_frequent_destinations(3),
                "statistics": self.long_term.get_statistics()
            }
        }

    def get_context_for_agent(self, long_term_summary: str = None) -> str:
        """
        获取用于Agent的上下文字符串

        Args:
            long_term_summary: 已持久化的历史会话摘要（可选）

        Returns:
            格式化的上下文字符串
        """
        lines = []

        # 长期记忆总结（历史会话）
        if long_term_summary:
            lines.append("【历史会话总结】")
            lines.append(long_term_summary)
            lines.append("")

        # 用户偏好
        prefs = self.long_term.get_preference()
        has_prefs = any(v for v in prefs.values() if v)
        if has_prefs:
            lines.append("【用户偏好】")
            for key, value in prefs.items():
                if value:
                    lines.append(f"- {key}: {value}")
            lines.append("")

        # 短期记忆（当前会话）
        context_str = self.short_term.get_context_string(3)
        if context_str != "无历史对话":
            lines.append("【当前会话对话】")
            lines.append(context_str)
            lines.append("")

        return "\n".join(lines) if lines else "无上下文信息"

    # ========== 会话管理 ==========

    async def _extract_model_text(self, response: Any) -> str:
        """统一提取普通响应和异步流式响应中的文本。"""
        summary = ""
        if hasattr(response, "__aiter__"):
            async for chunk in response:
                if isinstance(chunk, str):
                    summary = chunk
                elif hasattr(chunk, "content"):
                    content = chunk.content
                    if isinstance(content, str):
                        summary = content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                summary = item.get("text", "")
        elif hasattr(response, "content"):
            summary = str(response.content)
        else:
            summary = str(response)
        return summary.strip()

    async def generate_current_session_summary_async(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        max_messages: int = 50,
    ) -> str:
        """只总结当前会话聊天，不重复混入结构化偏好和行程。"""
        if max_messages <= 0:
            raise ValueError("max_messages 必须大于0")
        if not self.llm_model:
            return ""

        if messages is None:
            messages = self.long_term.get_chat_history(
                limit=None,
                session_id=self.session_id,
            )
        selected_messages = messages[-max_messages:]
        if not selected_messages:
            return ""

        dialogue_lines = []
        for message in selected_messages:
            role = message.get("role", "unknown")
            content = str(message.get("content", ""))[:1500]
            timestamp = message.get("timestamp", "")
            dialogue_lines.append(f"[{timestamp}] {role}: {content}")

        prompt = f"""请为以下一次已结束的对话生成会话摘要。
只总结对后续会话有价值的事实：
1. 用户想完成的任务或提出的核心问题
2. 用户已明确提供的关键信息和做出的决定
3. 尚未解决、之后可能需要继续的事项

不要推测未出现的信息，不要复述Agent调度过程，不要重复大段回答。
请用不超过200字的中文完成摘要。

【当前会话】
{chr(10).join(dialogue_lines)}
"""

        try:
            response = await self.llm_model(
                [{"role": "user", "content": prompt}]
            )
            summary = await self._extract_model_text(response)
            logger.info(
                "Generated session summary for %s (%d chars)",
                self.session_id,
                len(summary),
            )
            return summary
        except Exception as exc:
            logger.error(
                "Failed to generate summary for session %s: %s",
                self.session_id,
                exc,
            )
            return ""

    def get_previous_session_summaries(
        self,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """直接读取历史会话已持久化的摘要，不调用LLM。"""
        return self.long_term.get_session_summaries(
            limit=limit,
            exclude_session_id=self.session_id,
        )

    async def end_session(self, max_messages: int = 50) -> str:
        """生成并持久化当前会话摘要，最后清除Redis临时状态。"""
        summary = ""
        try:
            messages = self.long_term.get_chat_history(
                limit=None,
                session_id=self.session_id,
            )
            if not messages:
                return ""

            existing = self.long_term.get_session_summary(self.session_id)
            if (
                existing
                and existing.get("message_count") == len(messages)
            ):
                return str(existing.get("summary", ""))

            summary = await self.generate_current_session_summary_async(
                messages,
                max_messages=max_messages,
            )
            if summary:
                self.long_term.save_session_summary(
                    self.session_id,
                    summary,
                    len(messages),
                )
            return summary
        except Exception as exc:
            # 摘要是可重建的派生数据；失败时保留SQLite原始聊天，
            # 不让用户因为摘要存储异常无法退出。
            logger.error(
                "Failed to persist summary for session %s: %s",
                self.session_id,
                exc,
            )
            return ""
        finally:
            self.clear_session_state()
            logger.info("Session ended: %s", self.session_id)
