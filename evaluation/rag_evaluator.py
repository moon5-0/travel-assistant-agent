#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量 RAG 评估器：分开评估检索、生成与知识库外拒答。"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = Path(__file__).with_name("rag_cases.json")


def load_cases(path: Path = DEFAULT_CASES_PATH) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list) or not cases:
        raise ValueError("RAG evaluation cases must be a non-empty list")
    return cases


def _metadata_text(document: Dict[str, Any]) -> str:
    metadata = document.get("metadata", {}) or {}
    values = [
        metadata.get("parent_doc", ""),
        metadata.get("title", ""),
        metadata.get("category", ""),
        Path(str(metadata.get("file_path", ""))).name,
    ]
    return " ".join(str(value).lower() for value in values if value)


def _source_matches(document: Dict[str, Any], expected_sources: Iterable[str]) -> bool:
    source_text = _metadata_text(document)
    return any(str(source).lower() in source_text for source in expected_sources)


def _round_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


class RAGEvaluator:
    def __init__(self, cases: List[Dict[str, Any]]):
        self.cases = cases

    def evaluate_retrieval(self, agent) -> Dict[str, Any]:
        """评估 Hit Rate@K、Precision@K、拒答率和平均检索耗时。"""
        answerable = [case for case in self.cases if case.get("should_answer", True)]
        negatives = [case for case in self.cases if not case.get("should_answer", True)]
        source_hits = 0
        relevant_documents = 0
        retrieved_documents = 0
        rejected_negatives = 0
        total_ms = 0.0
        details = []

        for case in self.cases:
            start = time.perf_counter()
            documents = agent.search_knowledge(case["question"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            total_ms += elapsed_ms

            expected_sources = case.get("expected_sources", [])
            matching_documents = [
                doc for doc in documents if _source_matches(doc, expected_sources)
            ]
            should_answer = case.get("should_answer", True)
            hit = bool(matching_documents) if should_answer else not documents

            if should_answer:
                source_hits += int(bool(matching_documents))
                relevant_documents += len(matching_documents)
                retrieved_documents += len(documents)
            else:
                rejected_negatives += int(not documents)

            details.append({
                "id": case.get("id"),
                "should_answer": should_answer,
                "retrieved_count": len(documents),
                "source_hit_or_rejected": hit,
                "top_score": documents[0].get("score", documents[0].get("distance")) if documents else None,
                "elapsed_ms": round(elapsed_ms, 3),
            })

        return {
            "case_count": len(self.cases),
            "answerable_count": len(answerable),
            "negative_count": len(negatives),
            "hit_rate_at_k": _round_ratio(source_hits, len(answerable)),
            "precision_at_k": _round_ratio(relevant_documents, retrieved_documents),
            "negative_rejection_rate": _round_ratio(rejected_negatives, len(negatives)),
            "average_retrieval_ms": round(total_ms / len(self.cases), 3),
            "details": details,
        }

    async def evaluate_generation(self, agent) -> Dict[str, Any]:
        """评估答案关键点、返回来源和知识库外拒答。"""
        answerable_total = 0
        keyword_hits = 0
        source_hits = 0
        negative_total = 0
        negative_rejections = 0
        details = []

        for case in self.cases:
            response = await agent.reply(
                SimpleNamespace(content=case["question"])
            )
            result = json.loads(response.content)
            should_answer = case.get("should_answer", True)
            answer = str(result.get("answer", ""))

            if should_answer:
                answerable_total += 1
                expected_keywords = case.get("expected_keywords", [])
                keyword_hit = all(keyword in answer for keyword in expected_keywords)
                returned_documents = result.get("retrieved_documents", [])
                source_hit = any(
                    _source_matches(doc, case.get("expected_sources", []))
                    for doc in returned_documents
                )
                keyword_hits += int(keyword_hit)
                source_hits += int(source_hit)
            else:
                negative_total += 1
                keyword_hit = False
                source_hit = False
                negative_rejections += int(result.get("status") == "no_knowledge")

            details.append({
                "id": case.get("id"),
                "status": result.get("status"),
                "keyword_hit": keyword_hit,
                "source_hit": source_hit,
            })

        return {
            "answerable_count": answerable_total,
            "negative_count": negative_total,
            "answer_keyword_accuracy": _round_ratio(keyword_hits, answerable_total),
            "source_hit_rate": _round_ratio(source_hits, answerable_total),
            "negative_rejection_rate": _round_ratio(negative_rejections, negative_total),
            "details": details,
        }


def _load_rag_agent_class():
    agent_path = (
        PROJECT_ROOT / ".claude" / "skills" / "ask-question" / "script" / "agent.py"
    )
    spec = importlib.util.spec_from_file_location("rag_evaluation_agent", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load RAG Agent: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RAGKnowledgeAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight RAG evaluation")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--include-generation",
        action="store_true",
        help="Also call the configured LLM and evaluate generated answers",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    cases = load_cases(args.cases)
    agent_class = _load_rag_agent_class()

    model = None
    if args.include_generation:
        from agentscope.model import OpenAIChatModel
        from config import LLM_CONFIG

        model = OpenAIChatModel(
            model_name=LLM_CONFIG["model_name"],
            api_key=LLM_CONFIG["api_key"],
            client_kwargs={"base_url": LLM_CONFIG["base_url"]},
            generate_kwargs={
                "temperature": 0,
                "max_tokens": LLM_CONFIG.get("max_tokens", 2000),
            },
        )

    agent = agent_class(model=model)
    if not agent.initialized:
        print("RAG Agent 初始化失败，请先安装依赖并初始化知识库。", file=sys.stderr)
        return 1

    try:
        evaluator = RAGEvaluator(cases)
        report = {"retrieval": evaluator.evaluate_retrieval(agent)}
        if args.include_generation:
            report["generation"] = asyncio.run(evaluator.evaluate_generation(agent))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
