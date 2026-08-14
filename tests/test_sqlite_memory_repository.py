#!/usr/bin/env python3
"""SQLite 长期记忆和数据隔离的离线测试。"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest

from context.sqlite_memory_repository import SQLiteMemoryRepository


class TestSQLiteMemoryRepository(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_repository(self, user_id="user-a"):
        return SQLiteMemoryRepository(user_id, self.storage_path)

    def test_schema_contains_queryable_business_tables(self):
        repository = self.create_repository()

        with sqlite3.connect(repository.db_path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }

        self.assertTrue(
            {
                "preferences",
                "chat_messages",
                "trip_history",
                "user_statistics",
            }.issubset(tables)
        )

    def test_users_share_database_file_but_data_is_isolated(self):
        first = self.create_repository("user-a")
        second = self.create_repository("user-b")

        first.save_preference("home_location", "苏州")
        first.add_chat_message("user", "只属于A", "session-a")
        second.save_preference("home_location", "上海")
        second.add_chat_message("user", "只属于B", "session-b")

        self.assertEqual(first.db_path, second.db_path)
        self.assertEqual(first.get_preference("home_location"), "苏州")
        self.assertEqual(second.get_preference("home_location"), "上海")
        self.assertEqual(first.get_chat_history()[0]["content"], "只属于A")
        self.assertEqual(second.get_chat_history()[0]["content"], "只属于B")

    def test_delete_all_removes_only_current_user(self):
        first = self.create_repository("user-a")
        second = self.create_repository("user-b")
        first.save_preference("home_location", "苏州")
        second.save_preference("home_location", "上海")

        first.delete_all()

        self.assertEqual(first.get_preference(), {})
        self.assertEqual(second.get_preference("home_location"), "上海")


if __name__ == "__main__":
    unittest.main(verbosity=2)
