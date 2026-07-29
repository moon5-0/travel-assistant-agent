"""System Evaluation 场景评估运行器。

运行器负责场景隔离、多轮顺序执行和结果汇总；具体使用真实模型还是
离线假实现，由 runtime_factory 决定。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Optional

from evaluation.system.system_evaluator import evaluate_case


class SystemEvaluationRunner:
    """运行已校验的数据集，并汇总每个场景的评分结果。"""

    def __init__(
        self,
        dataset: Dict[str, Any],
        runtime_factory: Callable[[Dict[str, Any], int], Any],
    ) -> None:
        self.dataset = dataset
        self.runtime_factory = runtime_factory

    async def run(
        self,
        case_ids: Optional[Iterable[str]] = None,
        runs_per_case: Optional[int] = None,
    ) -> Dict[str, Any]:
        """逐场景、逐轮执行；不同场景由 runtime_factory 隔离状态。"""
        selected_ids = set(case_ids or [])
        cases = [
            case
            for case in self.dataset["cases"]
            if not selected_ids or case["id"] in selected_ids
        ]
        run_count = runs_per_case or self.dataset["default_context"][
            "runs_per_case"
        ]

        run_results = []
        for case in cases:
            for run_index in range(run_count):
                run_results.append(
                    await self._run_case(case, run_index + 1)
                )

        return self._build_report(run_results, run_count)

    async def _run_case(
        self,
        case: Dict[str, Any],
        run_index: int,
    ) -> Dict[str, Any]:
        """在一个独立运行环境中顺序执行同一场景的全部轮次。"""
        traces = []
        try:
            # runtime_factory 返回异步上下文管理器，负责创建和清理隔离环境。
            async with self.runtime_factory(case, run_index) as collector:
                for turn in case["turns"]:
                    traces.append(
                        await collector.execute_turn(turn["user_input"])
                    )

            evaluation = evaluate_case(case, traces)
            return {
                "case_id": case["id"],
                "run_index": run_index,
                "passed": evaluation["passed"],
                "failures": evaluation["failures"],
                "turns": evaluation["turns"],
                "traces": traces,
            }
        except Exception as exc:
            # 单个场景执行失败不应阻断整份基线报告。
            return {
                "case_id": case["id"],
                "run_index": run_index,
                "passed": False,
                "failures": ["execution_error"],
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "turns": [],
                "traces": traces,
            }

    def _build_report(
        self,
        run_results: list[Dict[str, Any]],
        runs_per_case: int,
    ) -> Dict[str, Any]:
        """汇总总体通过率以及每个场景的稳定通过率。"""
        case_summaries = []
        for case in self.dataset["cases"]:
            case_runs = [
                result
                for result in run_results
                if result["case_id"] == case["id"]
            ]
            if not case_runs:
                continue
            passed_runs = sum(result["passed"] for result in case_runs)
            case_summaries.append({
                "case_id": case["id"],
                "severity": case["severity"],
                "runs": len(case_runs),
                "passed_runs": passed_runs,
                "pass_rate": passed_runs / len(case_runs),
            })

        passed_runs = sum(result["passed"] for result in run_results)
        total_runs = len(run_results)
        critical_runs = [
            result
            for result in run_results
            if self._severity_for(result["case_id"]) == "critical"
        ]
        critical_passed = sum(result["passed"] for result in critical_runs)

        return {
            "dataset_version": self.dataset["dataset_version"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "runs_per_case": runs_per_case,
                "case_count": len(case_summaries),
            },
            "summary": {
                "total_runs": total_runs,
                "passed_runs": passed_runs,
                "pass_rate": passed_runs / total_runs if total_runs else 0.0,
                "critical_runs": len(critical_runs),
                "critical_passed_runs": critical_passed,
                "critical_pass_rate": (
                    critical_passed / len(critical_runs)
                    if critical_runs
                    else 0.0
                ),
            },
            "cases": case_summaries,
            "runs": run_results,
        }

    def _severity_for(self, case_id: str) -> Optional[str]:
        for case in self.dataset["cases"]:
            if case["id"] == case_id:
                return case["severity"]
        return None
