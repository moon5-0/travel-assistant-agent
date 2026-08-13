"""当前会话状态的存储接口与默认内存实现。

SessionStore 只保存具有会话生命周期的数据：最近对话和待补全行程。
Redis 接入时实现相同接口即可，Agent 与 MemoryManager 不需要感知底层变化。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
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


class InMemorySessionStore(SessionStore):
    """使用 Python 容器保存会话状态的默认实现。

    适用于本地运行、单元测试和 Redis 不可用时的降级。进程退出后数据会
    消失，因此它不是长期记忆实现。
    """

    def __init__(self) -> None:
        self._messages: Dict[str, List[Dict[str, Any]]] = {}
        self._pending_trips: Dict[str, Dict[str, Any]] = {}

    def add_message(
        self,
        session_id: str,
        message: Dict[str, Any],
        *,
        max_messages: int,
    ) -> None:
        messages = self._messages.setdefault(session_id, [])
        messages.append(deepcopy(message))
        if len(messages) > max_messages:
            self._messages[session_id] = messages[-max_messages:]

    def get_recent_messages(
        self,
        session_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        messages = self._messages.get(session_id, [])
        selected = messages[-limit:] if limit else messages
        return deepcopy(selected)

    def clear_messages(self, session_id: str) -> None:
        self._messages.pop(session_id, None)

    def get_pending_trip(self, session_id: str) -> Dict[str, Any]:
        return deepcopy(self._pending_trips.get(session_id, {}))

    def save_pending_trip(
        self,
        session_id: str,
        trip_data: Dict[str, Any],
    ) -> None:
        normalized = deepcopy(dict(trip_data or {}))
        if normalized:
            self._pending_trips[session_id] = normalized
        else:
            self.clear_pending_trip(session_id)

    def clear_pending_trip(self, session_id: str) -> None:
        self._pending_trips.pop(session_id, None)

