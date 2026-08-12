#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在真实 BGE + Milvus Lite 链路上运行 RAG 证据级检索评估。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = EVALUATION_DIR / "rag_dev_cases_v0.2.json"
DEFAULT_RESULTS_DIR = EVALUATION_DIR / "results"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RAG_CONFIG
from evaluation.rag.rag_retrieval_evaluator import (
    RAGRetrievalEvaluator,
    load_dataset,
    validate_dataset,
)


EvidenceKey = Tuple[str, int]


def _load_rag_agent_class():
    """从实际 Skill 加载 RAGKnowledgeAgent，避免依赖失效的旧导入路径。"""
    agent_path = (
        PROJECT_ROOT
        / ".claude"
        / "skills"
        / "ask-question"
        / "script"
        / "agent.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rag_retrieval_evaluation_agent",
        agent_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 RAG Agent: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RAGKnowledgeAgent


def select_cases(
    dataset: Mapping[str, Any],
    case_ids: Sequence[str] | None,
) -> Dict[str, Any]:
    """按 ID 选择 Case，并保留可被评估器校验的数据集结构。"""
    if not case_ids:
        return dict(dataset)
    requested = list(dict.fromkeys(case_ids))
    case_map = {case["id"]: case for case in dataset["cases"]}
    unknown = [case_id for case_id in requested if case_id not in case_map]
    if unknown:
        raise ValueError(f"未知 RAG Case: {', '.join(unknown)}")
    selected = dict(dataset)
    selected["cases"] = [case_map[case_id] for case_id in requested]
    selected["case_count"] = len(selected["cases"])
    validate_dataset(selected, project_root=PROJECT_ROOT)
    return selected


def expected_evidence_keys(dataset: Mapping[str, Any]) -> Set[EvidenceKey]:
    """提取数据集引用的全部 ``source + chunk_index``。"""
    return {
        (Path(evidence["source"]).name, int(evidence["chunk_index"]))
        for case in dataset["cases"]
        for evidence in case["gold_evidence"]
    }


def _decode_metadata(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return dict(parsed) if isinstance(parsed, Mapping) else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return {}


def collection_evidence_keys(agent: Any) -> Set[EvidenceKey]:
    """读取真实 Collection metadata，确认评估使用的是新版知识库。"""
    rows = agent.milvus_client.query(
        collection_name=agent.collection_name,
        filter="id >= 0",
        output_fields=["metadata"],
        limit=10000,
    )
    keys: Set[EvidenceKey] = set()
    for row in rows or []:
        metadata = _decode_metadata(row.get("metadata"))
        source = metadata.get("parent_doc")
        chunk_index = metadata.get("chunk_index")
        if not source or chunk_index is None:
            continue
        try:
            keys.add((Path(str(source)).name, int(chunk_index)))
        except (TypeError, ValueError):
            continue
    return keys


def validate_collection_contract(
    agent: Any,
    dataset: Mapping[str, Any],
) -> Dict[str, Any]:
    """阻止用旧库或不完整知识库生成看似正常的评估分数。"""
    stats = agent.get_stats()
    if stats.get("status") != "success":
        raise RuntimeError(f"读取 Milvus 统计失败: {stats}")
    actual_keys = collection_evidence_keys(agent)
    if not actual_keys:
        raise RuntimeError(
            "Milvus Collection 中没有 parent_doc + chunk_index；"
            "请先重新运行 init_knowledge_base.py"
        )
    required_keys = expected_evidence_keys(dataset)
    missing = sorted(required_keys - actual_keys)
    if missing:
        formatted = ", ".join(f"{source}#{index}" for source, index in missing)
        raise RuntimeError(f"Milvus 缺少数据集所需 Gold Evidence: {formatted}")
    return {
        "collection_name": agent.collection_name,
        "document_chunk_count": stats.get("total_documents"),
        "identified_chunk_count": len(actual_keys),
        "required_gold_chunk_count": len(required_keys),
        "missing_gold_chunks": [],
    }


def warm_up_retriever(agent: Any, question: str, top_k: int) -> float:
    """执行一次不计分检索，使后续延迟不混入首次搜索初始化。"""
    started_at = time.perf_counter()
    agent.search_knowledge(question, top_k=top_k)
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    search_error = getattr(agent, "last_search_error", None)
    if search_error:
        raise RuntimeError(f"RAG warm-up 检索失败: {search_error}")
    return round(elapsed_ms, 3)


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RESULTS_DIR / f"rag-retrieval-dev-{timestamp}.json"


def build_report(
    evaluation: Mapping[str, Any],
    *,
    dataset_path: Path,
    collection: Mapping[str, Any],
    top_k: int,
    startup_ms: float,
    warmup_ms: float,
) -> Dict[str, Any]:
    return {
        "evaluation_type": "rag_evidence_retrieval_development",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path.resolve()),
        "configuration": {
            "embedding_model": RAG_CONFIG.get("embedding_model"),
            "retrieval_mode": RAG_CONFIG.get("retrieval_mode", "vector"),
            "top_k": top_k,
            "similarity_threshold": RAG_CONFIG.get("similarity_threshold"),
            "candidate_multiplier": RAG_CONFIG.get("candidate_multiplier"),
            "vector_candidate_k": RAG_CONFIG.get("vector_candidate_k"),
            "keyword_candidate_k": RAG_CONFIG.get("keyword_candidate_k"),
            "rrf_k": RAG_CONFIG.get("rrf_k"),
            "hybrid_similarity_floor": RAG_CONFIG.get("hybrid_similarity_floor"),
            "dedupe_similarity": RAG_CONFIG.get("dedupe_similarity"),
        },
        "collection": dict(collection),
        "runtime": {
            "agent_startup_ms": round(startup_ms, 3),
            "warmup_search_ms": round(warmup_ms, 3),
            "measured_latency_scope": "warm_retrieval_only",
        },
        "summary": {
            key: value
            for key, value in evaluation.items()
            if key != "details"
        },
        "details": evaluation["details"],
        "limitations": [
            "当前为开发集结果，可用于排错和参数调试，不作为最终测试成绩。",
            "当前只评估证据检索，不评价 LLM 最终回答质量。",
            "延迟为本机热检索耗时，不包含首次 BGE 模型加载时间。",
        ],
    }


def print_summary(report: Mapping[str, Any], output_path: Path) -> None:
    summary = report["summary"]
    print(json.dumps({
        "case_count": summary["case_count"],
        "answerable_count": summary["answerable_count"],
        "negative_count": summary["negative_count"],
        f"evidence_recall_at_{summary['top_k']}": summary["evidence_recall_at_k"],
        f"mrr_at_{summary['top_k']}": summary["mrr_at_k"],
        f"evidence_precision_at_{summary['top_k']}": summary["evidence_precision_at_k"],
        "negative_rejection_rate": summary["negative_rejection_rate"],
        "execution_error_count": summary["execution_error_count"],
        "latency_ms": summary["latency_ms"],
    }, ensure_ascii=False, indent=2))

    failures = []
    for detail in report["details"]:
        failed = (
            detail["execution_error"]
            or (
                detail["should_answer"]
                and not detail["complete_evidence_hit"]
            )
            or (
                not detail["should_answer"]
                and not detail["negative_rejected"]
            )
        )
        if failed:
            failures.append({
                "id": detail["id"],
                "retrieved_count": detail["retrieved_count"],
                "matched_evidence_count": detail["matched_evidence_count"],
                "gold_evidence_count": detail["gold_evidence_count"],
                "execution_error": detail["execution_error"],
            })
    if failures:
        print("\n失败 Case：")
        print(json.dumps(failures, ensure_ascii=False, indent=2))
    else:
        print("\n全部 Case 通过当前检索合同。")
    print(f"\nREPORT: {output_path.resolve()}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evidence-level RAG retrieval evaluation on Milvus Lite",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        help="只运行指定 Case；可重复传入",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(RAG_CONFIG.get("top_k", 4)),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验数据集和打印执行规模，不加载 BGE/Milvus",
    )
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k 必须大于等于1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset = select_cases(load_dataset(args.cases), args.case_ids)
    except (OSError, ValueError) as exc:
        print(f"数据集错误: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps({
            "dataset_version": dataset.get("dataset_version"),
            "case_ids": [case["id"] for case in dataset["cases"]],
            "case_count": dataset["case_count"],
            "answerable_count": sum(case["should_answer"] for case in dataset["cases"]),
            "negative_count": sum(not case["should_answer"] for case in dataset["cases"]),
            "top_k": args.top_k,
            "evaluation_scope": "retrieval_only",
            "model_calls": 0,
        }, ensure_ascii=False, indent=2))
        return 0

    agent = None
    try:
        startup_started_at = time.perf_counter()
        agent_class = _load_rag_agent_class()
        agent = agent_class(model=None, top_k=args.top_k)
        if not getattr(agent, "initialized", False):
            raise RuntimeError("RAG Agent 初始化失败，请检查 BGE、Milvus 依赖和知识库")
        collection = validate_collection_contract(agent, dataset)
        startup_ms = (time.perf_counter() - startup_started_at) * 1000
        warmup_ms = warm_up_retriever(
            agent,
            dataset["cases"][0]["question"],
            args.top_k,
        )
        evaluation = RAGRetrievalEvaluator(
            dataset,
            top_k=args.top_k,
        ).evaluate(agent)
        report = build_report(
            evaluation,
            dataset_path=args.cases,
            collection=collection,
            top_k=args.top_k,
            startup_ms=startup_ms,
            warmup_ms=warmup_ms,
        )
        output_path = args.output or default_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print_summary(report, output_path)
        return 1 if evaluation["execution_error_count"] else 0
    except Exception as exc:
        print(f"RAG 检索评估失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if agent is not None:
            agent.close()


if __name__ == "__main__":
    raise SystemExit(main())
