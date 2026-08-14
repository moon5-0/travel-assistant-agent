"""长期记忆的数据访问接口。

业务层只依赖这个合同，不直接读写具体数据库。当前正式实现是 SQLite；
未来如果迁移 PostgreSQL，只需新增另一个 Repository 实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LongTermMemoryRepository(ABC):
    """当前项目真正使用的长期记忆存取合同。"""

    @abstractmethod
    def save_preference(self, pref_type: str, value: Any) -> bool:
        pass

    @abstractmethod
    def get_preference(self, pref_type: Optional[str] = None) -> Any:
        pass

    @abstractmethod
    def add_hotel_brand(self, brand: str) -> None:
        pass

    @abstractmethod
    def add_airline(self, airline: str) -> None:
        pass

    @abstractmethod
    def add_chat_message(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
    ) -> None:
        pass

    @abstractmethod
    def get_chat_history(
        self,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def save_trip_history(self, trip_info: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def get_trip_history(
        self,
        limit: Optional[int] = 10,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_frequent_destinations(self, top_n: int = 5) -> List[tuple]:
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def increment_query_count(self) -> None:
        pass

    @abstractmethod
    def clear_history(self) -> None:
        pass

    @abstractmethod
    def delete_all(self) -> None:
        pass
