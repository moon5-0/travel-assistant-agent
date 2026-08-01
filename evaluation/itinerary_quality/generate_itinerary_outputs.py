#!/usr/bin/env python3
"""生成 ItineraryPlanningAgent 输出，并执行行程质量硬规则评估。

第一版只生成 Hard Constraint Pass Rate 和 Fatal Error Rate；
LLM Judge 的主观质量分将在后续阶段接入。
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from getpass import getpass
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from config import LLM_CONFIG, SYSTEM_CONFIG
from config_agentscope import init_agentscope
from evaluation.itinerary_quality.hard_rule_evaluator import (
    DEFAULT_CASES_PATH,
    evaluate_case,
    load_dataset,
)


DEFAULT_OUTPUT_DIR = Path(__file__).with_name("results")


def load_itinerary_agent_class():
    """从 plan-trip Skill 加载真实 ItineraryPlanningAgent。"""
    agent_path = (
        PROJECT_ROOT
        / ".claude"
        / "skills"
        / "plan-trip"
        / "script"
        / "agent.py"
    )
    spec = importlib.util.spec_from_file_location(
        "itinerary_quality_eval_agent",
        agent_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载行程规划Agent: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ItineraryPlanningAgent


def build_agent_input(case: Dict[str, Any]) -> Msg:
    """把固定评估输入转换为调度器实际发送给规划Agent的格式。"""
    case_input = case["input"]
    context = {
        "rewritten_query": case_input["user_query"],
        "user_preferences": case_input["user_preferences"],
        # 行程质量评估隔离规划 Agent，因此这里使用测试集提供的
        # IntentionAgent 结构化语义信号，不在本阶段重复做意图识别。
        "planning_signals": case_input["planning_signals"],
    }
    previous_results = [
        {
            "agent_name": "event_collection",
            "status": "success",
            "result": {
                "status": "success",
                "data": case_input["trip_info"],
            },
        }
    ]
    if case_input["external_information"]:
        previous_results.append({
            "agent_name": "information_query",
            "status": "success",
            "result": {
                "status": "success",
                "data": {
                    "items": case_input["external_information"],
                },
            },
        })

    payload = {
        "context": context,
        "previous_results": previous_results,
    }
    return Msg(
        name="ItineraryQualityEvaluator",
        content=json.dumps(payload, ensure_ascii=False),
        role="user",
    )


def select_cases(
    dataset: Dict[str, Any],
    case_ids: Iterable[str],
) -> List[Dict[str, Any]]:
    requested = list(case_ids)
    if not requested:
        return list(dataset["cases"])

    available = {case["id"] for case in dataset["cases"]}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"未知测试场景: {', '.join(unknown)}")
    return [case for case in dataset["cases"] if case["id"] in requested]


def build_execution_summary(
    dataset: Dict[str, Any],
    cases: List[Dict[str, Any]],
    runs: int,
) -> Dict[str, Any]:
    return {
        "dataset_version": dataset["version"],
        "case_ids": [case["id"] for case in cases],
        "case_count": len(cases),
        "runs_per_case": runs,
        "total_agent_calls": len(cases) * runs,
        "evaluation_stage": "hard_rules_only",
        "note": (
            "每次通常调用规划模型1次；格式异常或检测到明确时间可行性问题时，"
            "对应修复步骤会各额外调用1次。"
        ),
    }


def build_model(temperature: float) -> OpenAIChatModel:
    timeout_sec = SYSTEM_CONFIG.get("timeout", 60)
    return OpenAIChatModel(
        model_name=LLM_CONFIG["model_name"],
        api_key=LLM_CONFIG["api_key"],
        client_kwargs={
            "base_url": LLM_CONFIG["base_url"],
            "timeout": float(timeout_sec),
        },
        generate_kwargs={
            "temperature": temperature,
            "max_tokens": LLM_CONFIG.get("max_tokens", 8192),
        },
    )


async def run_cases(
    agent: Any,
    dataset: Dict[str, Any],
    cases: List[Dict[str, Any]],
    runs_per_case: int,
    progress: bool = False,
) -> Dict[str, Any]:
    """逐场景调用真实规划Agent，并保留原始输出和硬规则结果。"""
    run_results = []
    total_runs = len(cases) * runs_per_case
    completed_runs = 0
    for case in cases:
        for run_index in range(1, runs_per_case + 1):
            if progress:
                print(
                    f"[{completed_runs + 1}/{total_runs}] "
                    f"正在运行 {case['id']}（第{run_index}次）...",
                    flush=True,
                )
            started = time.perf_counter()
            try:
                response = await agent.reply(build_agent_input(case))
                raw_output = response.content
                evaluation = evaluate_case(
                    case,
                    raw_output,
                    dataset["global_rules"],
                )
                run_results.append({
                    "case_id": case["id"],
                    "run_index": run_index,
                    "elapsed_ms": round(
                        (time.perf_counter() - started) * 1000,
                        3,
                    ),
                    "output": raw_output,
                    "evaluation": evaluation,
                })
                if progress:
                    status = "PASS" if evaluation["passed"] else "FAIL"
                    print(f"    {status}", flush=True)
            except Exception as exc:
                run_results.append({
                    "case_id": case["id"],
                    "run_index": run_index,
                    "elapsed_ms": round(
                        (time.perf_counter() - started) * 1000,
                        3,
                    ),
                    "execution_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                })
                if progress:
                    print(
                        f"    ERROR: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
            completed_runs += 1

    evaluated = [run for run in run_results if "evaluation" in run]
    passed = sum(
        run["evaluation"]["hard_constraints_passed"]
        for run in evaluated
    )
    fatal_runs = sum(
        bool(run["evaluation"]["fatal_errors"])
        for run in evaluated
    )
    evaluated_count = len(evaluated)
    return {
        "dataset_version": dataset["version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_runs": len(run_results),
            "evaluated_runs": evaluated_count,
            "execution_error_runs": len(run_results) - evaluated_count,
            "hard_constraint_passed_runs": passed,
            "hard_constraint_pass_rate": (
                passed / evaluated_count if evaluated_count else 0.0
            ),
            "fatal_error_runs": fatal_runs,
            "fatal_error_rate": (
                fatal_runs / evaluated_count if evaluated_count else 0.0
            ),
        },
        "runs": run_results,
    }


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"itinerary-outputs-{timestamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate itinerary outputs and run hard constraints",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="只运行指定case id；可重复传入",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验数据并显示执行规模，不调用模型",
    )
    return parser.parse_args()


async def run_real_evaluation(
    dataset: Dict[str, Any],
    selected_cases: List[Dict[str, Any]],
    runs: int,
    temperature: float,
) -> Dict[str, Any]:
    model = build_model(temperature)
    agent_class = load_itinerary_agent_class()
    agent = agent_class(name="ItineraryPlanningAgent", model=model)
    report = await run_cases(
        agent,
        dataset,
        selected_cases,
        runs,
        progress=True,
    )
    report["configuration"] = {
        "model_name": LLM_CONFIG["model_name"],
        "temperature": temperature,
        "evaluation_stage": "hard_rules_only",
    }
    return report


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        print("INVALID: --runs必须大于等于1")
        return 2

    try:
        dataset = load_dataset(args.cases)
        selected_cases = select_cases(dataset, args.case)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}")
        return 2

    print(json.dumps(
        build_execution_summary(dataset, selected_cases, args.runs),
        ensure_ascii=False,
        indent=2,
    ))
    if args.dry_run:
        print("DRY RUN: 未调用模型")
        return 0

    api_key = str(os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key and sys.stdin.isatty():
        api_key = getpass(
            "DeepSeek API Key（输入内容不会显示）: "
        ).strip()
    if not api_key:
        print("ERROR: DeepSeek API Key不能为空")
        return 2
    LLM_CONFIG["api_key"] = api_key

    init_agentscope()
    report = asyncio.run(run_real_evaluation(
        dataset,
        selected_cases,
        args.runs,
        args.temperature,
    ))
    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"REPORT: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
