"""当前会话状态的存储接口。

SessionStore 只保存具有会话生命周期的数据：最近对话和待补全行程。
正式实现使用 Redis，Agent 与 MemoryManager 不感知底层读写命令。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SessionStore(ABC):
    """短期会话状态的最小存储合同。"""

    @abstractmethod
    def add_message(
        self,
        session_id: str,
        message: Dict[str, Any],
        *,
        max_messages: int,
    ) -> None:
        """追加消息，并只保留指定数量的最新消息。"""

    @abstractmethod
    def get_recent_messages(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """读取最近消息；不允许调用方原地修改内部状态。"""

    @abstractmethod
    def clear_messages(self, session_id: str) -> None:
        """清除当前会话对话。"""

    @abstractmethod
    def get_pending_trip(self, session_id: str) -> Dict[str, Any]:
        """读取待补全行程。"""

    @abstractmethod
    def save_pending_trip(
        self,
        session_id: str,
        trip_data: Dict[str, Any],
    ) -> None:
        """覆盖保存待补全行程。"""

    @abstractmethod
    def clear_pending_trip(self, session_id: str) -> None:
        """清除待补全行程。"""

    def clear_session(self, session_id: str) -> None:
        """清除当前会话的全部临时状态。"""
        self.clear_messages(session_id)
        self.clear_pending_trip(session_id)
