#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量 RAG 评估器的离线测试。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.rag.rag_evaluator import RAGEvaluator, load_cases


CASES = [
    {
        "id": "positive_hit",
        "question": "命中问题",
        "expected_keywords": ["500"],
        "expected_sources": ["policy_a.txt"],
        "should_answer": True,
    },
    {
        "id": "positive_miss",
        "question": "未命中问题",
        "expected_keywords": ["30"],
        "expected_sources": ["policy_b.txt"],
        "should_answer": True,
    },
    {
        "id": "negative_rejected",
        "question": "应拒答且已拒答",
        "expected_keywords": [],
        "expected_sources": [],
        "should_answer": False,
    },
    {
        "id": "negative_not_rejected",
        "question": "应拒答但错误召回",
        "expected_keywords": [],
        "expected_sources": [],
        "should_answer": False,
    },
]


def document(source, score=0.9):
    return {
        "content": "知识内容",
        "metadata": {"parent_doc": source, "title": source},
        "score": score,
    }


class FakeRAGAgent:
    def search_knowledge(self, question):
        mapping = {
            "命中问题": [document("policy_a.txt")],
            "未命中问题": [document("wrong_source.txt")],
            "应拒答且已拒答": [],
            "应拒答但错误召回": [document("wrong_source.txt", 0.56)],
        }
        return mapping[question]

    async def reply(self, message):
        mapping = {
            "命中问题": {
                "status": "success",
                "answer": "住宿标准为500元",
                "retrieved_documents": [document("policy_a.txt")],
            },
            "未命中问题": {
                "status": "success",
                "answer": "没有包含预期关键点",
                "retrieved_documents": [document("wrong_source.txt")],
            },
            "应拒答且已拒答": {
                "status": "no_knowledge",
                "answer": "没有找到相关信息",
                "retrieved_documents": [],
            },
            "应拒答但错误召回": {
                "status": "success",
                "answer": "错误回答",
                "retrieved_documents": [document("wrong_source.txt")],
            },
        }
        return SimpleNamespace(content=json.dumps(mapping[message.content], ensure_ascii=False))


class TestRAGEvaluator(unittest.IsolatedAsyncioTestCase):
    def test_project_dataset_contains_positive_and_negative_cases(self):
        cases = load_cases()

        self.assertGreaterEqual(len(cases), 20)
        self.assertTrue(any(case["should_answer"] for case in cases))
        self.assertTrue(any(not case["should_answer"] for case in cases))

    def test_retrieval_metrics_are_computed_separately(self):
        report = RAGEvaluator(CASES).evaluate_retrieval(FakeRAGAgent())

        self.assertEqual(report["hit_rate_at_k"], 0.5)
        self.assertEqual(report["precision_at_k"], 0.5)
        self.assertEqual(report["negative_rejection_rate"], 0.5)

    async def test_generation_metrics_cover_keywords_sources_and_rejection(self):
        report = await RAGEvaluator(CASES).evaluate_generation(FakeRAGAgent())

        self.assertEqual(report["answer_keyword_accuracy"], 0.5)
        self.assertEqual(report["source_hit_rate"], 0.5)
        self.assertEqual(report["negative_rejection_rate"], 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
