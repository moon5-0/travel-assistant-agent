#!/usr/bin/env python3
"""运行真实 System Evaluation 并生成系统行为基线报告。

先检查但不调用模型：
    python evaluation/run_system_eval.py --dry-run

只运行一个场景一次：
    python evaluation/run_system_eval.py --case trip_missing_required_fields --runs 1
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from contextlib import asynccontextmanager
from datetime import datetime
from getpass import getpass
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.intention_agent import IntentionAgent
from agents.lazy_agent_registry import LazyAgentRegistry
from agents.orchestration_agent import OrchestrationAgent
from agentscope.model import OpenAIChatModel
from config import LLM_CONFIG, RESILIENCE_CONFIG, SYSTEM_CONFIG
from config_agentscope import init_agentscope
from context.memory_manager import MemoryManager
from evaluation.system_eval_runner import SystemEvaluationRunner
from evaluation.system_evaluator import DEFAULT_CASES_PATH, load_dataset
from evaluation.system_trace_collector import SystemTraceCollector
from services.turn_executor import AgentTurnExecutor
from utils.circuit_breaker import CircuitBreaker


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"


def seed_initial_state(
    memory_manager: MemoryManager,
    orchestrator: OrchestrationAgent,
    initial_state: Dict[str, Any],
) -> None:
    """把测试用例声明的偏好、行程和待补全状态写入隔离环境。"""
    for preference_type, value in initial_state.get("preferences", {}).items():
        memory_manager.long_term.save_preference(
            preference_type,
            deepcopy(value),
        )

    for trip in initial_state.get("trip_history", []):
        memory_manager.long_term.save_trip_history(deepcopy(trip))

    orchestrator.restore_pending_trip(
        deepcopy(initial_state.get("pending_trip", {}))
    )


class RealRuntimeFactory:
    """为每次场景运行创建并清理一套真实、隔离的 Agent 环境。"""

    def __init__(self, model: Any) -> None:
        self.model = model

    # 让创建和清理可以写成 async with
    @asynccontextmanager
    # __call__ 让工厂实例可以像函数一样被运行器调用。
    async def __call__(self, case: Dict[str, Any], run_index: int):
        with TemporaryDirectory(prefix="aligo-system-eval-") as storage_path:
            safe_case_id = case["id"].replace("/", "_")
            memory_manager = MemoryManager(
                user_id=f"eval-{safe_case_id}-{run_index}",
                session_id=f"eval-session-{safe_case_id}-{run_index}",
                storage_path=storage_path,
                llm_model=self.model,
            )
            intention_agent = IntentionAgent(
                name="IntentionAgent",
                model=self.model,
            )
            agent_cache = {}
            registry = LazyAgentRegistry(
                model=self.model,
                cache=agent_cache,
                memory_manager=memory_manager,
            )
            orchestrator = OrchestrationAgent(
                name="OrchestrationAgent",
                agent_registry=registry,
                memory_manager=memory_manager,
            )
            circuit_breaker = CircuitBreaker(
                failure_threshold=RESILIENCE_CONFIG.get(
                    "circuit_failure_threshold",
                    5,
                ),
                recovery_timeout_sec=RESILIENCE_CONFIG.get(
                    "circuit_recovery_timeout_sec",
                    60.0,
                ),
                half_open_successes=RESILIENCE_CONFIG.get(
                    "circuit_half_open_successes",
                    2,
                ),
            )
            executor = AgentTurnExecutor(
                intention_agent=intention_agent,
                orchestrator=orchestrator,
                memory_manager=memory_manager,
                circuit_breaker=circuit_breaker,
                resilience_config=RESILIENCE_CONFIG,
            )
            seed_initial_state(
                memory_manager,
                orchestrator,
                case["initial_state"],
            )
            # 把 Collector 暂时交出去使用，环境先不要销毁
            yield SystemTraceCollector(executor)


def select_cases(
    dataset: Dict[str, Any],
    case_ids: Iterable[str],
) -> list[Dict[str, Any]]:
    """选择指定场景，并在名称写错时直接报错。"""
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
    cases: list[Dict[str, Any]],
    runs: int,
) -> Dict[str, Any]:
    """在不调用模型的情况下，展示预计执行规模。"""
    turns_per_run = sum(len(case["turns"]) for case in cases)
    return {
        "dataset_version": dataset["dataset_version"],
        "case_ids": [case["id"] for case in cases],
        "case_count": len(cases),
        "runs_per_case": runs,
        "turns_per_run": turns_per_run,
        "total_turns": turns_per_run * runs,
        "note": "total_turns 是用户对话轮数，实际模型调用次数会更多。",
    }


def build_model(temperature: float) -> OpenAIChatModel:
    """使用项目当前 DeepSeek 配置创建评估模型。"""
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


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"system-baseline-{timestamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real Aligo System Evaluation",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="只运行指定 case id；可重复传入",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="每个场景重复次数；首次基线建议1，稳定性评估建议3",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验配置并估算执行量，不调用模型",
    )
    return parser.parse_args()


async def run_real_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    dataset = load_dataset(args.cases)
    selected_cases = select_cases(dataset, args.case)
    selected_ids = [case["id"] for case in selected_cases]

    model = build_model(args.temperature)
    runner = SystemEvaluationRunner(dataset, RealRuntimeFactory(model))
    report = await runner.run(
        case_ids=selected_ids,
        runs_per_case=args.runs,
    )
    report["configuration"].update({
        "model_name": LLM_CONFIG["model_name"],
        "temperature": args.temperature,
        "dataset_current_date": dataset["default_context"]["current_date"],
        "timezone": dataset["default_context"]["timezone"],
        # TODO(可复现性): Agent 内部目前读取系统时间，后续应注入固定时钟。
        "clock_source": "system_clock",
    })
    return report


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        print("INVALID: --runs 必须大于等于1")
        return 2

    try:
        dataset = load_dataset(args.cases)
        selected_cases = select_cases(dataset, args.case)
    except (ValueError, OSError) as exc:
        print(f"INVALID: {exc}")
        return 2

    execution_summary = build_execution_summary(
        dataset,
        selected_cases,
        args.runs,
    )
    print(json.dumps(execution_summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("DRY RUN: 未调用模型")
        return 0

    api_key = str(os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not api_key and sys.stdin.isatty():
        api_key = getpass(
            "DeepSeek API Key（输入内容不会显示）: "
        ).strip()
    if not api_key:
        print("ERROR: DeepSeek API Key 不能为空")
        return 2
    LLM_CONFIG["api_key"] = api_key

    init_agentscope()
    report = asyncio.run(run_real_evaluation(args))
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
