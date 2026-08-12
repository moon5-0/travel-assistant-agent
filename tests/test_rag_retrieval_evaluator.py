#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG 证据级检索评估器的离线测试。"""

from __future__ import annotations

import copy
import unittest

from evaluation.rag.rag_retrieval_evaluator import (
    DatasetValidationError,
    RAGRetrievalEvaluator,
    document_matches_evidence,
    load_dataset,
    validate_dataset,
)


def evidence(source, *, chunk_index=1):
    return {
        "source": source,
        "chunk_index": chunk_index,
    }


def case(case_id, question, *, match="all", gold=None, should_answer=True):
    gold = list(gold or [])
    return {
        "id": case_id,
        "category": "test",
        "question": question,
        "should_answer": should_answer,
        "risk_type": "test",
        "expected_sources": sorted({item["source"] for item in gold}),
        "evidence_match": match,
        "gold_evidence": gold,
    }


def dataset(cases):
    return {
        "dataset_version": "test",
        "knowledge_base": {
            "document_directory": ".claude/skills/ask-question/data/documents"
        },
        "case_count": len(cases),
        "cases": cases,
    }


def document(source, content, score=0.9, *, chunk_index=1):
    return {
        "content": content,
        "metadata": {
            "parent_doc": source,
            "title": f"{source} (Part 1)",
            "chunk_index": chunk_index,
        },
        "score": score,
    }


class FakeRetriever:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def search_knowledge(self, question, top_k=None):
        self.calls.append((question, top_k))
        value = self.mapping[question]
        if isinstance(value, Exception):
            raise value
        return value


class StepClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class TestRAGRetrievalDataset(unittest.TestCase):
    def test_project_development_dataset_is_valid(self):
        data = load_dataset()

        self.assertEqual(data["case_count"], 15)
        self.assertEqual(sum(c["should_answer"] for c in data["cases"]), 12)
        self.assertEqual(sum(not c["should_answer"] for c in data["cases"]), 3)

    def test_duplicate_case_id_is_rejected(self):
        data = load_dataset()
        broken = copy.deepcopy(data)
        broken["cases"][1]["id"] = broken["cases"][0]["id"]

        with self.assertRaisesRegex(DatasetValidationError, "duplicate case id"):
            validate_dataset(broken)

    def test_negative_case_cannot_have_gold_evidence(self):
        broken_case = case(
            "negative",
            "无关问题",
            match="none",
            gold=[evidence("01_travel_standards.txt")],
            should_answer=False,
        )

        with self.assertRaisesRegex(DatasetValidationError, "negative case"):
            validate_dataset(dataset([broken_case]))

class TestEvidenceMatching(unittest.TestCase):
    def test_parent_document_and_chunk_index_must_both_match(self):
        gold = evidence("01_travel_standards.txt", chunk_index=2)

        self.assertTrue(document_matches_evidence(
            document(
                "01_travel_standards.txt",
                "正文不参与匹配",
                chunk_index=2,
            ),
            gold,
        ))
        self.assertFalse(document_matches_evidence(
            document(
                "01_travel_standards.txt",
                "即使正文相同，块编号错误也不能命中",
                chunk_index=1,
            ),
            gold,
        ))
        self.assertFalse(document_matches_evidence(
            document(
                "02_reimbursement_policy.txt",
                "即使块编号相同，来源错误也不能命中",
                chunk_index=2,
            ),
            gold,
        ))


class TestRAGRetrievalMetrics(unittest.TestCase):
    def test_all_any_negative_and_metric_denominators(self):
        first = evidence("01_travel_standards.txt", chunk_index=1)
        second = evidence("02_reimbursement_policy.txt", chunk_index=1)
        alternative_a = evidence(
            "02_reimbursement_policy.txt",
            chunk_index=2,
        )
        alternative_b = evidence(
            "02_reimbursement_policy.txt",
            chunk_index=5,
        )
        cases = [
            case("all_hit", "all hit", gold=[first, second]),
            case("all_partial", "all partial", gold=[first, second]),
            case(
                "any_hit",
                "any hit",
                match="any",
                gold=[alternative_a, alternative_b],
            ),
            case(
                "negative_rejected",
                "negative rejected",
                match="none",
                should_answer=False,
            ),
            case(
                "negative_false_positive",
                "negative false positive",
                match="none",
                should_answer=False,
            ),
        ]
        retriever = FakeRetriever({
            "all hit": [
                document("unrelated.txt", "无关内容"),
                document(
                    "01_travel_standards.txt",
                    "国际长途航线（4小时以上）可预订高端经济舱",
                    chunk_index=1,
                ),
                document(
                    "02_reimbursement_policy.txt",
                    "报销期限为30个自然日",
                    chunk_index=1,
                ),
            ],
            "all partial": [
                document(
                    "01_travel_standards.txt",
                    "国际长途航线（4小时以上）可预订高端经济舱",
                    chunk_index=1,
                ),
                document("unrelated.txt", "无关内容"),
            ],
            "any hit": [
                document(
                    "02_reimbursement_policy.txt",
                    "遗失发票：需出具书面说明",
                    chunk_index=5,
                ),
            ],
            "negative rejected": [],
            "negative false positive": [
                document("01_travel_standards.txt", "相似但不应回答"),
            ],
        })
        # 5 个 Case，每个时钟读取两次；分别耗时 1、2、3、4、5 毫秒。
        clock = StepClock([
            0.000, 0.001,
            0.010, 0.012,
            0.020, 0.023,
            0.030, 0.034,
            0.040, 0.045,
        ])

        report = RAGRetrievalEvaluator(
            dataset(cases),
            top_k=4,
            clock=clock,
        ).evaluate(retriever)

        # 三个正例中 all_hit 与 any_hit 完整命中。
        self.assertEqual(report["evidence_recall_at_k"], 0.6667)
        # 首个相关证据分别排第2、第1、第1，MRR=(1/2+1+1)/3。
        self.assertEqual(report["mrr_at_k"], 0.8333)
        # 正例共返回 6 个文档，其中 4 个文档至少命中一条证据。
        self.assertEqual(report["evidence_precision_at_k"], 0.6667)
        self.assertEqual(report["negative_rejection_rate"], 0.5)
        self.assertEqual(report["latency_ms"]["p50"], 3.0)
        self.assertEqual(report["latency_ms"]["p95"], 4.8)
        self.assertEqual(retriever.calls[0], ("all hit", 4))

        details = {item["id"]: item for item in report["details"]}
        self.assertTrue(details["all_hit"]["complete_evidence_hit"])
        self.assertFalse(details["all_partial"]["complete_evidence_hit"])
        self.assertTrue(details["any_hit"]["complete_evidence_hit"])
        self.assertTrue(details["negative_rejected"]["negative_rejected"])

    def test_search_error_is_not_counted_as_rejection(self):
        negative = case(
            "negative",
            "negative",
            match="none",
            should_answer=False,
        )
        report = RAGRetrievalEvaluator(dataset([negative])).evaluate(
            FakeRetriever({"negative": RuntimeError("Milvus unavailable")})
        )

        self.assertEqual(report["execution_error_count"], 1)
        self.assertEqual(report["negative_rejection_rate"], 0.0)
        self.assertIn(
            "Milvus unavailable",
            report["details"][0]["execution_error"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
