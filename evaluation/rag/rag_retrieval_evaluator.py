#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RAG 证据级检索评估器。

本模块只评价 ``search_knowledge(question, top_k=K)`` 的检索结果，
不调用 LLM，也不判断最终回答措辞。Gold Evidence 使用入库 metadata
中的 ``parent_doc + chunk_index`` 唯一标识一个知识块。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = Path(__file__).with_name("rag_dev_cases_v0.2.json")
SUPPORTED_EVIDENCE_MATCH = {"all", "any", "none"}


class DatasetValidationError(ValueError):
    """RAG 检索数据集不符合评估合同。"""


def load_dataset(path: Path | str = DEFAULT_CASES_PATH) -> Dict[str, Any]:
    """读取并校验一份证据级检索数据集。"""
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    validate_dataset(dataset, project_root=PROJECT_ROOT)
    return dataset


def validate_dataset(
    dataset: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> None:
    """校验结构、Case ID、正负例和 Gold Evidence 来源。"""
    if not isinstance(dataset, Mapping):
        raise DatasetValidationError("RAG dataset must be a JSON object")

    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise DatasetValidationError("dataset.cases must be a non-empty list")
    if dataset.get("case_count") != len(cases):
        raise DatasetValidationError(
            f"case_count={dataset.get('case_count')!r} does not match "
            f"len(cases)={len(cases)}"
        )

    knowledge_base = dataset.get("knowledge_base")
    if not isinstance(knowledge_base, Mapping):
        raise DatasetValidationError("dataset.knowledge_base must be an object")
    document_directory = knowledge_base.get("document_directory")
    if not isinstance(document_directory, str) or not document_directory.strip():
        raise DatasetValidationError(
            "knowledge_base.document_directory must be a non-empty string"
        )

    root = project_root or PROJECT_ROOT
    document_root = root / document_directory
    seen_ids = set()
    required_fields = {
        "id",
        "question",
        "should_answer",
        "risk_type",
        "expected_sources",
        "evidence_match",
        "gold_evidence",
    }

    for index, case in enumerate(cases):
        label = f"cases[{index}]"
        if not isinstance(case, Mapping):
            raise DatasetValidationError(f"{label} must be an object")
        missing = sorted(required_fields - set(case))
        if missing:
            raise DatasetValidationError(f"{label} missing fields: {missing}")

        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise DatasetValidationError(f"{label}.id must be a non-empty string")
        if case_id in seen_ids:
            raise DatasetValidationError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)
        label = case_id

        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise DatasetValidationError(f"{label}.question must be non-empty")
        if not isinstance(case.get("should_answer"), bool):
            raise DatasetValidationError(f"{label}.should_answer must be boolean")

        evidence_match = case.get("evidence_match")
        if evidence_match not in SUPPORTED_EVIDENCE_MATCH:
            raise DatasetValidationError(
                f"{label}.evidence_match must be one of "
                f"{sorted(SUPPORTED_EVIDENCE_MATCH)}"
            )

        expected_sources = case.get("expected_sources")
        gold_evidence = case.get("gold_evidence")
        if not isinstance(expected_sources, list) or not all(
            isinstance(source, str) and source for source in expected_sources
        ):
            raise DatasetValidationError(
                f"{label}.expected_sources must be a list of strings"
            )
        if not isinstance(gold_evidence, list):
            raise DatasetValidationError(f"{label}.gold_evidence must be a list")

        if case["should_answer"]:
            if evidence_match not in {"all", "any"}:
                raise DatasetValidationError(
                    f"{label}: answerable case must use evidence_match=all/any"
                )
            if not expected_sources or not gold_evidence:
                raise DatasetValidationError(
                    f"{label}: answerable case requires sources and Gold Evidence"
                )
        elif evidence_match != "none" or expected_sources or gold_evidence:
            raise DatasetValidationError(
                f"{label}: negative case must use none and have no evidence"
            )

        for evidence_index, evidence in enumerate(gold_evidence):
            evidence_label = f"{label}.gold_evidence[{evidence_index}]"
            if not isinstance(evidence, Mapping):
                raise DatasetValidationError(f"{evidence_label} must be an object")
            source = evidence.get("source")
            chunk_index = evidence.get("chunk_index")
            if source not in expected_sources:
                raise DatasetValidationError(
                    f"{evidence_label}.source is not in expected_sources"
                )
            if not isinstance(chunk_index, int) or chunk_index < 1:
                raise DatasetValidationError(
                    f"{evidence_label}.chunk_index must be a positive integer"
                )
            source_path = document_root / source
            if not source_path.is_file():
                raise DatasetValidationError(
                    f"{evidence_label}.source does not exist: {source}"
                )


def document_matches_evidence(
    document: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> bool:
    """严格按 ``parent_doc + chunk_index`` 判断是否为同一知识块。"""
    metadata = document.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        return False
    parent_doc = metadata.get("parent_doc")
    chunk_index = metadata.get("chunk_index")
    try:
        chunk_index = int(chunk_index)
    except (TypeError, ValueError):
        return False
    return (
        Path(str(parent_doc or "")).name == Path(str(evidence.get("source", ""))).name
        and chunk_index == evidence.get("chunk_index")
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    """线性插值百分位数；空输入返回 0。"""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


class RAGRetrievalEvaluator:
    """对实现 ``search_knowledge`` 的检索器执行证据级评估。"""

    def __init__(
        self,
        dataset: Mapping[str, Any],
        *,
        top_k: int = 4,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        validate_dataset(dataset, project_root=PROJECT_ROOT)
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.dataset = dict(dataset)
        self.cases = list(dataset["cases"])
        self.top_k = int(top_k)
        self.clock = clock

    @staticmethod
    def _search(retriever: Any, question: str, top_k: int) -> List[Dict[str, Any]]:
        documents = retriever.search_knowledge(question, top_k=top_k)
        search_error = getattr(retriever, "last_search_error", None)
        if search_error:
            raise RuntimeError(f"retrieval failed: {search_error}")
        if documents is None:
            return []
        if not isinstance(documents, list):
            documents = list(documents)
        return [dict(document) for document in documents[:top_k]]

    def evaluate(self, retriever: Any) -> Dict[str, Any]:
        """运行全部 Case 并返回汇总指标及逐题证据诊断。"""
        answerable_count = sum(case["should_answer"] for case in self.cases)
        negative_count = len(self.cases) - answerable_count
        complete_hits = 0
        reciprocal_rank_sum = 0.0
        relevant_documents = 0
        retrieved_positive_documents = 0
        rejected_negatives = 0
        execution_errors = 0
        latencies_ms: List[float] = []
        details: List[Dict[str, Any]] = []

        for case in self.cases:
            started_at = self.clock()
            error = None
            try:
                documents = self._search(
                    retriever,
                    case["question"],
                    self.top_k,
                )
            except Exception as exc:  # 保留单题错误，不中断整份评估。
                documents = []
                error = f"{type(exc).__name__}: {exc}"
                execution_errors += 1
            elapsed_ms = max(0.0, (self.clock() - started_at) * 1000)
            latencies_ms.append(elapsed_ms)

            gold_evidence = case["gold_evidence"]
            matched_evidence_indexes = set()
            relevant_ranks: List[int] = []
            result_documents = []

            for rank, document in enumerate(documents, start=1):
                matched_indexes = [
                    index
                    for index, evidence in enumerate(gold_evidence)
                    if document_matches_evidence(document, evidence)
                ]
                if matched_indexes:
                    relevant_ranks.append(rank)
                    matched_evidence_indexes.update(matched_indexes)
                metadata = document.get("metadata") or {}
                result_documents.append({
                    "rank": rank,
                    "source": (
                        metadata.get("parent_doc")
                        if isinstance(metadata, Mapping)
                        else None
                    ),
                    "title": (
                        metadata.get("title")
                        if isinstance(metadata, Mapping)
                        else None
                    ),
                    "chunk_index": (
                        metadata.get("chunk_index")
                        if isinstance(metadata, Mapping)
                        else None
                    ),
                    "score": document.get("score", document.get("distance")),
                    "matched_evidence_indexes": matched_indexes,
                    "content_preview": str(document.get("content", ""))[:160],
                })

            should_answer = case["should_answer"]
            if should_answer:
                if case["evidence_match"] == "all":
                    complete_hit = len(matched_evidence_indexes) == len(gold_evidence)
                else:
                    complete_hit = bool(matched_evidence_indexes)
                complete_hits += int(complete_hit and error is None)
                first_relevant_rank = min(relevant_ranks) if relevant_ranks else None
                if first_relevant_rank is not None and error is None:
                    reciprocal_rank_sum += 1 / first_relevant_rank
                relevant_documents += len(relevant_ranks)
                retrieved_positive_documents += len(documents)
                rejected = False
            else:
                complete_hit = False
                first_relevant_rank = None
                rejected = not documents and error is None
                rejected_negatives += int(rejected)

            details.append({
                "id": case["id"],
                "category": case.get("category"),
                "risk_type": case.get("risk_type"),
                "should_answer": should_answer,
                "evidence_match": case["evidence_match"],
                "retrieved_count": len(documents),
                "matched_evidence_count": len(matched_evidence_indexes),
                "gold_evidence_count": len(gold_evidence),
                "complete_evidence_hit": complete_hit,
                "first_relevant_rank": first_relevant_rank,
                "negative_rejected": rejected,
                "latency_ms": round(elapsed_ms, 3),
                "execution_error": error,
                "documents": result_documents,
            })

        return {
            "dataset_version": self.dataset.get("dataset_version"),
            "evaluation_scope": "retrieval_only",
            "top_k": self.top_k,
            "case_count": len(self.cases),
            "answerable_count": answerable_count,
            "negative_count": negative_count,
            "execution_error_count": execution_errors,
            # Case 级 Recall：all 题必须找齐所有证据，any 题命中任一替代证据。
            "evidence_recall_at_k": _ratio(complete_hits, answerable_count),
            "mrr_at_k": round(
                reciprocal_rank_sum / answerable_count,
                4,
            ) if answerable_count else 0.0,
            "evidence_precision_at_k": _ratio(
                relevant_documents,
                retrieved_positive_documents,
            ),
            "negative_rejection_rate": _ratio(
                rejected_negatives,
                negative_count,
            ),
            "latency_ms": {
                "p50": round(_percentile(latencies_ms, 0.50), 3),
                "p95": round(_percentile(latencies_ms, 0.95), 3),
                "average": round(
                    sum(latencies_ms) / len(latencies_ms),
                    3,
                ) if latencies_ms else 0.0,
            },
            "details": details,
        }


__all__ = [
    "DatasetValidationError",
    "RAGRetrievalEvaluator",
    "document_matches_evidence",
    "load_dataset",
    "validate_dataset",
]
