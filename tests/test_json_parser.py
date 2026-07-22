#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSON 响应提取与容错解析的离线测试。"""

from __future__ import annotations

import unittest
import sys
from types import SimpleNamespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.json_parser import (
    extract_json_from_async_response,
    extract_json_from_response,
    robust_json_parse,
)


class AsyncChunks:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self.chunks:
            yield chunk


class TestRobustJsonParse(unittest.TestCase):
    def test_extracts_markdown_json_from_explanatory_text(self):
        text = '模型回答如下：\n```json\n{"city": "杭州", "days": 3}\n```\n完成。'

        result = robust_json_parse(text)

        self.assertEqual(result, {"city": "杭州", "days": 3})

    def test_combines_single_quote_and_trailing_comma_repairs(self):
        result = robust_json_parse("{'city': '杭州', 'days': 3,}")

        self.assertEqual(result, {"city": "杭州", "days": 3})

    def test_escapes_literal_newline_inside_string(self):
        result = robust_json_parse('{"reasoning": "先查天气\n再规划行程"}')

        self.assertEqual(result["reasoning"], "先查天气\n再规划行程")

    def test_invalid_text_returns_fallback(self):
        fallback = {"intents": [], "agent_schedule": []}

        self.assertEqual(robust_json_parse("没有结构化结果", fallback), fallback)

    def test_invalid_json_without_fallback_raises_value_error(self):
        with self.assertRaises(ValueError):
            robust_json_parse('{"city": }')


class TestResponseExtraction(unittest.IsolatedAsyncioTestCase):
    async def test_extracts_content_from_regular_response(self):
        response = SimpleNamespace(content='{"city": "杭州"}')

        self.assertEqual(
            extract_json_from_response(response),
            '{"city": "杭州"}',
        )

    async def test_async_delta_chunks_are_concatenated(self):
        response = AsyncChunks(['{"city":', ' "杭州"}'])

        text = await extract_json_from_async_response(response)

        self.assertEqual(text, '{"city": "杭州"}')

    async def test_async_cumulative_chunks_are_not_duplicated(self):
        response = AsyncChunks(['{"city":', '{"city": "杭州"}'])

        text = await extract_json_from_async_response(response)

        self.assertEqual(text, '{"city": "杭州"}')


if __name__ == "__main__":
    unittest.main(verbosity=2)
