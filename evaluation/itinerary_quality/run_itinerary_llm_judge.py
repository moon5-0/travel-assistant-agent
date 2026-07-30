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
)
from evaluation.itinerary_quality.run_itinerary_quality_eval import build_model


RESULTS_DIR = Path(__file__).with_name("results")


def latest_hard_report() -> Path | None:
    reports = sorted(
        RESULTS_DIR.glob("itinerary-quality-hard-*.json"),
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
) -> List[Dict[str, Any]]:
    requested = set(case_ids)
    runs = list(source_report["runs"])
    available = {run.get("case_id") for run in runs}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"源报告中不存在场景: {', '.join(unknown)}")
    return [
        run
        for run in runs
        if not requested or run.get("case_id") in requested
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


async def run_judging(
    judge: Any,
    dataset: Dict[str, Any],
    source_report: Dict[str, Any],
    selected_runs: List[Dict[str, Any]],
    *,
    progress: bool = False,
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
        try:
            judge_evaluation = await judge.evaluate(
                case,
                source_run["output"],
            )
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
                "fatal_error": combined_fatal,
                "passed": passed,
            })
            if progress:
                print(
                    f"    {'PASS' if passed else 'FAIL'} "
                    f"quality={judge_evaluation['weighted_quality_score']}",
                    flush=True,
                )
        except Exception as exc:
            combined_runs.append({
                "case_id": case_id,
                "run_index": source_run.get("run_index"),
                "hard_evaluation": hard_evaluation,
                "judge_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            })
            if progress:
                print(
                    f"    ERROR: {type(exc).__name__}: {exc}",
                    flush=True,
                )

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
        "dataset_version": dataset["version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_report_generated_at": source_report.get("generated_at"),
        "summary": {
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
        },
        "runs": combined_runs,
    }


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return RESULTS_DIR / f"itinerary-quality-combined-{timestamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge existing itinerary outputs without regenerating them",
    )
    parser.add_argument("--input-report", type=Path)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.input_report or latest_hard_report()
    if source_path is None:
        print("INVALID: 未找到itinerary-quality-hard报告")
        return 2

    try:
        dataset = load_dataset(args.cases)
        source_report = load_source_report(source_path)
        selected_runs = select_runs(source_report, args.case)
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
    judge = LLMItineraryJudge(build_model(args.temperature))
    report = asyncio.run(run_judging(
        judge,
        dataset,
        source_report,
        selected_runs,
        progress=True,
    ))
    report["configuration"] = {
        "judge_model_name": LLM_CONFIG["model_name"],
        "judge_temperature": args.temperature,
        "source_report": str(source_path),
        "planning_outputs_reused": True,
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
