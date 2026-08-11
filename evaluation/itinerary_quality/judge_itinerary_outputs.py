#!/usr/bin/env python3
"""复用已生成的行程报告，运行 LLM Judge 并汇总完整质量指标。"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from getpass import getpass
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import LLM_CONFIG
from config_agentscope import init_agentscope
from evaluation.itinerary_quality.hard_rule_evaluator import (
    DEFAULT_CASES_PATH,
    evaluate_case,
    load_dataset,
)
from evaluation.itinerary_quality.llm_judge import (
    DIMENSION_WEIGHTS,
    LLMItineraryJudge,
    validate_evaluation_substance,
)
from evaluation.itinerary_quality.generate_itinerary_outputs import build_model


RESULTS_DIR = Path(__file__).with_name("results")


def latest_generated_report() -> Path | None:
    reports = sorted(
        [
            *RESULTS_DIR.glob("itinerary-outputs-*.json"),
            # 兼容重命名前生成的本地报告。
            *RESULTS_DIR.glob("itinerary-quality-hard-*.json"),
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def load_source_report(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    if not isinstance(report, dict) or not isinstance(report.get("runs"), list):
        raise ValueError("source report must contain a runs list")
    return report


def select_runs(
    source_report: Dict[str, Any],
    case_ids: Iterable[str],
    run_indices: Iterable[int] = (),
) -> List[Dict[str, Any]]:
    requested = set(case_ids)
    requested_indices = set(run_indices)
    runs = list(source_report["runs"])
    available = {run.get("case_id") for run in runs}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"源报告中不存在场景: {', '.join(unknown)}")
    return [
        run
        for run in runs
        if (not requested or run.get("case_id") in requested)
        and (
            not requested_indices
            or run.get("run_index") in requested_indices
        )
    ]


def build_execution_summary(
    dataset: Dict[str, Any],
    source_path: Path,
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    judgeable = sum("output" in run for run in runs)
    return {
        "dataset_version": dataset["version"],
        "source_report": str(source_path),
        "selected_runs": len(runs),
        "judge_calls": judgeable,
        "planning_agent_calls": 0,
        "note": "只评价已有行程；格式修复发生时Judge会额外调用1次。",
    }


def _run_key(run: Dict[str, Any]) -> tuple[Any, Any]:
    return run.get("case_id"), run.get("run_index")


def judge_run_needs_retry(run: Dict[str, Any] | None) -> bool:
    """超时、缺失以及结构合法但内容为空的旧结果都需要补评。"""
    if not isinstance(run, dict) or "judge_evaluation" not in run:
        return True
    try:
        validate_evaluation_substance(run["judge_evaluation"])
    except (KeyError, TypeError, ValueError):
        return True
    return False


def select_resume_runs(
    selected_source_runs: List[Dict[str, Any]],
    previous_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    previous_by_key = {
        _run_key(run): run
        for run in previous_report.get("runs", [])
    }
    return [
        run
        for run in selected_source_runs
        if judge_run_needs_retry(previous_by_key.get(_run_key(run)))
    ]


def summarize_judge_runs(combined_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """只将真正通过Judge校验的结果纳入质量指标分母。"""
    judged_runs = [
        run for run in combined_runs if "judge_evaluation" in run
    ]
    judged_count = len(judged_runs)
    hard_passed = sum(
        run["hard_evaluation"]["hard_constraints_passed"]
        for run in judged_runs
    )
    fatal_runs = sum(run["fatal_error"] for run in judged_runs)
    qualified_runs = sum(run["passed"] for run in judged_runs)
    quality_scores = [
        run["judge_evaluation"]["weighted_quality_score"]
        for run in judged_runs
    ]
    dimension_averages = {
        name: round(sum(
            run["judge_evaluation"]["dimensions"][name]["score"]
            for run in judged_runs
        ) / judged_count, 3) if judged_count else 0.0
        for name in DIMENSION_WEIGHTS
    }
    return {
        "total_runs": len(combined_runs),
        "judged_runs": judged_count,
        "judge_error_runs": len(combined_runs) - judged_count,
        "hard_constraint_passed_runs": hard_passed,
        "hard_constraint_pass_rate": (
            hard_passed / judged_count if judged_count else 0.0
        ),
        "average_itinerary_quality_score": (
            round(sum(quality_scores) / judged_count, 2)
            if judged_count else 0.0
        ),
        "fatal_error_runs": fatal_runs,
        "fatal_error_rate": (
            fatal_runs / judged_count if judged_count else 0.0
        ),
        "qualified_runs": qualified_runs,
        "qualified_rate": (
            qualified_runs / judged_count if judged_count else 0.0
        ),
        "dimension_average_scores_1_to_5": dimension_averages,
    }


def merge_judge_reports(
    previous_report: Dict[str, Any],
    retry_report: Dict[str, Any],
    source_report: Dict[str, Any],
) -> Dict[str, Any]:
    """用补评结果替换旧错误，同时保留之前已经成功的评价。"""
    previous_by_key = {
        _run_key(run): run
        for run in previous_report.get("runs", [])
    }
    retry_by_key = {
        _run_key(run): run
        for run in retry_report.get("runs", [])
    }
    merged_runs = []
    for source_run in source_report.get("runs", []):
        key = _run_key(source_run)
        merged = retry_by_key.get(key, previous_by_key.get(key))
        if merged is None:
            merged = {
                "case_id": key[0],
                "run_index": key[1],
                "judge_error": {
                    "type": "MissingJudgeResult",
                    "message": "该行程尚未执行Judge评价",
                },
            }
        merged_runs.append(merged)
    return {
        "dataset_version": retry_report.get(
            "dataset_version",
            previous_report.get("dataset_version"),
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report_generated_at": source_report.get("generated_at"),
        "summary": summarize_judge_runs(merged_runs),
        "runs": merged_runs,
    }


async def run_judging(
    judge: Any,
    dataset: Dict[str, Any],
    source_report: Dict[str, Any],
    selected_runs: List[Dict[str, Any]],
    *,
    progress: bool = False,
    judge_timeout_seconds: float = 360.0,
    max_attempts: int = 2,
    retry_delay_seconds: float = 2.0,
) -> Dict[str, Any]:
    cases = {case["id"]: case for case in dataset["cases"]}
    combined_runs = []
    total = len(selected_runs)

    for index, source_run in enumerate(selected_runs, 1):
        case_id = source_run.get("case_id")
        if progress:
            print(f"[{index}/{total}] 正在评价 {case_id}...", flush=True)

        if case_id not in cases or "output" not in source_run:
            combined_runs.append({
                "case_id": case_id,
                "run_index": source_run.get("run_index"),
                "judge_error": {
                    "type": "MissingSourceOutput",
                    "message": "源报告没有可评价的行程输出",
                },
            })
            if progress:
                print("    ERROR: 没有可评价的行程输出", flush=True)
            continue

        case = cases[case_id]
        hard_evaluation = evaluate_case(
            case,
            source_run["output"],
            dataset["global_rules"],
        )
        judge_evaluation = None
        judge_error = None
        attempts_used = 0
        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            try:
                # 每次都新建evaluate协程；协程被超时取消后不能再次等待。
                judge_evaluation = await asyncio.wait_for(
                    judge.evaluate(
                        case,
                        source_run["output"],
                    ),
                    timeout=judge_timeout_seconds,
                )
                judge_error = None
                break
            except Exception as exc:
                judge_error = exc
                if attempt < max_attempts:
                    delay = retry_delay_seconds * (2 ** (attempt - 1))
                    if progress:
                        print(
                            f"    Judge失败，{delay:g}秒后重试 "
                            f"({attempt}/{max_attempts})："
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    if delay > 0:
                        await asyncio.sleep(delay)

        if judge_evaluation is not None:
            combined_fatal = bool(
                hard_evaluation["fatal_errors"]
                or judge_evaluation["semantic_fatal_errors"]
            )
            passed = bool(
                hard_evaluation["hard_constraints_passed"]
                and judge_evaluation["judge_passed"]
                and not combined_fatal
            )
            combined_runs.append({
                "case_id": case_id,
                "run_index": source_run.get("run_index"),
                "hard_evaluation": hard_evaluation,
                "judge_evaluation": judge_evaluation,
                "judge_attempts": attempts_used,
                "fatal_error": combined_fatal,
                "passed": passed,
            })
            if progress:
                print(
                    f"    {'PASS' if passed else 'FAIL'} "
                    f"quality={judge_evaluation['weighted_quality_score']}",
                    flush=True,
                )
        else:
            assert judge_error is not None
            error_message = str(judge_error)
            if not error_message and isinstance(judge_error, TimeoutError):
                error_message = "Judge evaluation timed out"
            combined_runs.append({
                "case_id": case_id,
                "run_index": source_run.get("run_index"),
                "hard_evaluation": hard_evaluation,
                "judge_attempts": attempts_used,
                "judge_error": {
                    "type": type(judge_error).__name__,
                    "message": error_message,
                },
            })
            if progress:
                print(
                    f"    ERROR after {attempts_used} attempts: "
                    f"{type(judge_error).__name__}: {error_message}",
                    flush=True,
                )

    return {
        "dataset_version": dataset["version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report_generated_at": source_report.get("generated_at"),
        "summary": summarize_judge_runs(combined_runs),
        "runs": combined_runs,
    }


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return RESULTS_DIR / f"itinerary-quality-judged-{timestamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge existing itinerary outputs without regenerating them",
    )
    parser.add_argument("--input-report", type=Path)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--run-index",
        action="append",
        type=int,
        default=[],
        help="只评价源报告中的指定重复轮次；可重复传入",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=360.0,
        help="单份Judge评价（含内部修复）的最长等待时间",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=180.0,
        help="Judge单次底层模型请求的网络超时时间",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="单份Judge评价失败后的最大完整尝试次数",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=2.0,
        help="完整重试前的初始等待时间（指数增长）",
    )
    parser.add_argument(
        "--resume-report",
        type=Path,
        help="从旧Judge报告断点续评，只补跑错误、缺失或占位结果",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.input_report or latest_generated_report()
    if source_path is None:
        print("INVALID: 未找到已生成的行程输出报告")
        return 2

    try:
        dataset = load_dataset(args.cases)
        source_report = load_source_report(source_path)
        selected_runs = select_runs(
            source_report,
            args.case,
            args.run_index,
        )
        previous_report = None
        if args.resume_report:
            previous_report = load_source_report(args.resume_report)
            selected_runs = select_resume_runs(
                selected_runs,
                previous_report,
            )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: {exc}")
        return 2

    print(json.dumps(
        build_execution_summary(dataset, source_path, selected_runs),
        ensure_ascii=False,
        indent=2,
    ))
    if args.dry_run:
        print("DRY RUN: 未调用Judge模型，也未重新生成行程")
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
    judge = LLMItineraryJudge(build_model(
        args.temperature,
        request_timeout_seconds=args.request_timeout_seconds,
    ))
    report = asyncio.run(run_judging(
        judge,
        dataset,
        source_report,
        selected_runs,
        progress=True,
        judge_timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
    ))
    if previous_report is not None:
        report = merge_judge_reports(
            previous_report,
            report,
            source_report,
        )
    report["configuration"] = {
        "judge_model_name": LLM_CONFIG["model_name"],
        "judge_temperature": args.temperature,
        "source_report": str(source_path),
        "planning_outputs_reused": True,
        "request_timeout_seconds": args.request_timeout_seconds,
        "judge_timeout_seconds": args.timeout_seconds,
        "max_attempts": args.max_attempts,
        "resume_report": (
            str(args.resume_report) if args.resume_report else None
        ),
    }

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
