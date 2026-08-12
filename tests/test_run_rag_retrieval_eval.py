#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实 RAG 检索评估入口的离线单元测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from evaluation.rag.rag_retrieval_evaluator import load_dataset
from evaluation.rag.run_rag_retrieval_eval import (
    build_report,
    expected_evidence_keys,
    parse_args,
    select_cases,
    validate_collection_contract,
    warm_up_retriever,
)


class FakeMilvusClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return self.rows


def fake_agent(rows, total_documents=66):
    return SimpleNamespace(
        collection_name="business_travel_knowledge",
        milvus_client=FakeMilvusClient(rows),
        get_stats=lambda: {
            "status": "success",
            "total_documents": total_documents,
        },
    )


class TestRAGRetrievalRunSetup(unittest.TestCase):
    def test_select_cases_preserves_requested_order(self):
        dataset = load_dataset()
        selected = select_cases(dataset, [
            "negative_programming_question",
            "hotel_limit_beijing_paraphrase",
        ])

        self.assertEqual(selected["case_count"], 2)
        self.assertEqual(
            [case["id"] for case in selected["cases"]],
            ["negative_programming_question", "hotel_limit_beijing_paraphrase"],
        )

    def test_unknown_case_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知 RAG Case"):
            select_cases(load_dataset(), ["missing_case"])

    def test_collection_contract_accepts_all_required_chunk_ids(self):
        dataset = select_cases(load_dataset(), [
            "hotel_limit_beijing_paraphrase",
            "international_long_flight_cabin",
        ])
        rows = [
            {"metadata": {
                "parent_doc": "01_travel_standards.txt",
                "chunk_index": 1,
            }},
            {"metadata": {
                "parent_doc": "01_travel_standards.txt",
                "chunk_index": 2,
            }},
        ]
        agent = fake_agent(rows)

        result = validate_collection_contract(agent, dataset)

        self.assertEqual(result["required_gold_chunk_count"], 2)
        self.assertEqual(result["missing_gold_chunks"], [])
        self.assertEqual(agent.milvus_client.calls[0]["limit"], 10000)

    def test_collection_contract_rejects_old_metadata_without_chunk_index(self):
        dataset = select_cases(load_dataset(), [
            "hotel_limit_beijing_paraphrase",
        ])
        agent = fake_agent([{
            "metadata": {"parent_doc": "01_travel_standards.txt"},
        }])

        with self.assertRaisesRegex(RuntimeError, "没有 parent_doc \+ chunk_index"):
            validate_collection_contract(agent, dataset)

    def test_build_report_separates_summary_from_details(self):
        evaluation = {
            "dataset_version": "test",
            "evaluation_scope": "retrieval_only",
            "top_k": 4,
            "case_count": 1,
            "answerable_count": 1,
            "negative_count": 0,
            "execution_error_count": 0,
            "evidence_recall_at_k": 1.0,
            "mrr_at_k": 1.0,
            "evidence_precision_at_k": 0.25,
            "negative_rejection_rate": 0.0,
            "latency_ms": {"p50": 1.0, "p95": 1.0, "average": 1.0},
            "details": [{"id": "case"}],
        }

        report = build_report(
            evaluation,
            dataset_path=load_dataset.__globals__["DEFAULT_CASES_PATH"],
            collection={"document_chunk_count": 66},
            top_k=4,
            startup_ms=1200.0,
            warmup_ms=300.0,
        )

        self.assertNotIn("details", report["summary"])
        self.assertEqual(report["details"], [{"id": "case"}])
        self.assertIn("development", report["evaluation_type"])
        self.assertEqual(report["runtime"]["warmup_search_ms"], 300.0)

    def test_warm_up_is_not_part_of_evaluation_and_surfaces_errors(self):
        class Retriever:
            last_search_error = None

            def __init__(self):
                self.calls = []

            def search_knowledge(self, question, top_k=None):
                self.calls.append((question, top_k))
                return []

        retriever = Retriever()
        elapsed = warm_up_retriever(retriever, "warm up", 4)

        self.assertEqual(retriever.calls, [("warm up", 4)])
        self.assertGreaterEqual(elapsed, 0.0)

    def test_cli_rejects_non_positive_top_k(self):
        with self.assertRaises(SystemExit):
            parse_args(["--top-k", "0"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
