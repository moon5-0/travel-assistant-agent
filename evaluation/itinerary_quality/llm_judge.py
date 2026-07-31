"""使用结构化 LLM Judge 评价行程的主观质量。"""

from __future__ import annotations

import json
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils.json_parser import (
    extract_json_from_async_response,
    robust_json_parse,
)


DIMENSION_WEIGHTS = {
    "time_route_feasibility": 35,
    "business_personalization": 30,
    "completeness_usability": 20,
    "factual_groundedness": 15,
}
QUALITY_PASS_THRESHOLD = 70.0
MIN_DIMENSION_SCORE = 3


class DimensionEvaluation(BaseModel):
    """一个质量维度的分数、理由和行程内证据。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    score: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1, max_length=3)


class SemanticFatalError(BaseModel):
    """必须理解语义才能发现、且足以令行程不可执行的错误。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: str = Field(min_length=1)
    description: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1, max_length=3)

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value: Any) -> Any:
        """兼容旧版单个证据字符串，内部统一使用证据数组。"""
        if isinstance(value, str):
            return [value]
        return value


class JudgeModelOutput(BaseModel):
    """模型必须遵守的原始输出合同；总分不让模型自行计算。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    time_route_feasibility: DimensionEvaluation
    business_personalization: DimensionEvaluation
    completeness_usability: DimensionEvaluation
    factual_groundedness: DimensionEvaluation
    semantic_fatal_errors: list[SemanticFatalError] = Field(default_factory=list)
    overall_summary: str = Field(min_length=1)


JUDGE_OUTPUT_CONTRACT = {
    "time_route_feasibility": {
        "score": 1,
        "reason": "评分理由",
        "evidence": ["E001"],
    },
    "business_personalization": {
        "score": 1,
        "reason": "评分理由",
        "evidence": ["E001"],
    },
    "completeness_usability": {
        "score": 1,
        "reason": "评分理由",
        "evidence": ["E001"],
    },
    "factual_groundedness": {
        "score": 1,
        "reason": "评分理由",
        "evidence": ["E001"],
    },
    "semantic_fatal_errors": [
        {
            "category": "错误类型",
            "description": "为什么足以导致不可执行",
            "evidence": ["E001"],
        }
    ],
    "overall_summary": "整体评价",
}


def score_judge_output(value: Dict[str, Any]) -> Dict[str, Any]:
    """校验Judge结构，并由代码计算加权分和通过状态。"""
    validated = JudgeModelOutput.model_validate(value)
    raw = validated.model_dump()
    dimensions = {
        name: raw[name]
        for name in DIMENSION_WEIGHTS
    }
    weighted_score = round(sum(
        dimensions[name]["score"] / 5 * weight
        for name, weight in DIMENSION_WEIGHTS.items()
    ), 2)
    min_score = min(
        dimension["score"]
        for dimension in dimensions.values()
    )
    fatal_errors = raw["semantic_fatal_errors"]
    return {
        "dimensions": dimensions,
        "weighted_quality_score": weighted_score,
        "min_dimension_score": min_score,
        "semantic_fatal_errors": fatal_errors,
        "overall_summary": raw["overall_summary"],
        "judge_passed": (
            weighted_score >= QUALITY_PASS_THRESHOLD
            and min_score >= MIN_DIMENSION_SCORE
            and not fatal_errors
        ),
    }


def build_evidence_catalog(itinerary_output: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """把行程中的原始标量字段编号，供Judge引用而不自行改写证据。"""
    catalog: Dict[str, Dict[str, str]] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
            return
        if isinstance(value, bool) or value is None:
            return
        text = str(value).strip()
        if not text:
            return
        evidence_id = f"E{len(catalog) + 1:03d}"
        catalog[evidence_id] = {"path": path, "text": text}

    visit(itinerary_output, "")
    return catalog


def validate_evidence_grounding(
    scored_result: Dict[str, Any],
    itinerary_output: Dict[str, Any],
) -> None:
    """验证证据编号存在，并把编号解析回原字段，方便人工复核。"""
    catalog = build_evidence_catalog(itinerary_output)
    invalid_references = []
    for dimension_name, dimension in scored_result["dimensions"].items():
        details = []
        for evidence_id in dimension["evidence"]:
            if evidence_id not in catalog:
                invalid_references.append({
                    "location": f"dimensions.{dimension_name}.evidence",
                    "evidence_id": evidence_id,
                })
                continue
            details.append({"id": evidence_id, **catalog[evidence_id]})
        dimension["evidence_details"] = details

    for fatal_error in scored_result["semantic_fatal_errors"]:
        details = []
        for evidence_id in fatal_error["evidence"]:
            if evidence_id not in catalog:
                invalid_references.append({
                    "location": "semantic_fatal_errors.evidence",
                    "evidence_id": evidence_id,
                })
                continue
            details.append({"id": evidence_id, **catalog[evidence_id]})
        fatal_error["evidence_details"] = details
    if invalid_references:
        raise ValueError(
            "Judge evidence IDs must exist in evidence catalog: "
            f"{invalid_references}"
        )


def build_judge_messages(
    case: Dict[str, Any],
    itinerary_output: Dict[str, Any],
) -> list[Dict[str, str]]:
    """提供完整任务上下文、评分锚点和严格JSON合同。"""
    evaluation_input = {
        "user_query": case["input"]["user_query"],
        "trip_info": case["input"]["trip_info"],
        "user_preferences": case["input"]["user_preferences"],
        "external_information": case["input"]["external_information"],
        "judge_focus": case["expected"]["judge_focus"],
        "itinerary_output": itinerary_output,
        "evidence_catalog": build_evidence_catalog(itinerary_output),
    }
    system_prompt = """你是严格、中立的企业差旅行程质量评审员。

只依据提供的用户需求、结构化信息和行程内容评分，不补充外部事实，不因文字长或景点多而加分。
每个维度按1到5分：
- 1分：严重不可用或核心方面明显错误；
- 2分：存在重大问题，需要明显修改；
- 3分：基本可用，但有清楚可见的不足；
- 4分：整体良好，仅有少量轻微问题；
- 5分：优秀，几乎没有可指出的问题。

四个维度：
1. time_route_feasibility：检查时间先后、交通耗时、缓冲、固定活动冲突、折返和每日节奏。必须逐项比较活动时间段、描述中的耗时和下一活动开始时间：存在一处非关键的明确时间矛盾时最高3分；多个矛盾、关键交通无法完成或固定活动冲突时应为1到2分，并按严重程度判断是否属于语义致命错误。
2. business_personalization：检查商务目标优先级、用户偏好、预算和政策约束，避免把出差写成景点堆砌。
3. completeness_usability：检查每日安排、交通、住宿、用餐、返程和提醒是否清楚且可直接使用。不要因为用户输入本来没有提供会议地址、联系人、酒店名称或餐厅名称而扣分；如果行程明确提示确认、保留调整空间，就是合理处理。只评价规划Agent对已有信息的利用和缺失信息的处理。
4. factual_groundedness：检查未查询的车次、票价、房价、天气、营业时间等是否被当成确定事实；候选信息应提醒核实。该维度使用以下具体锚点：
   - 1分：把未查询信息说成已确认、已预订，或大量关键事实明显无依据；
   - 2分：给出多项精确实时信息，却没有候选措辞和核实提醒；
   - 3分：把精确信息标为建议或候选，并提醒通过官方渠道或按实际情况核实，但仍有较多未经查询的具体数字；
   - 4分：主要使用时间段、范围或候选方案，清楚区分建议与实时事实；
   - 5分：关键事实均有提供的外部信息支持，或完全避免无依据的精确事实。

semantic_fatal_errors只记录需要语义理解才能发现、且足以令行程不可执行的严重错误，例如固定会议冲突、关键交通在时间上不可能、明确违反硬政策、把未查询信息说成已确认或已预订。普通啰嗦、次要信息缺失、景点选择一般不能列为致命错误。

每个维度必须提供1到3个evidence_catalog中真实存在的证据编号，例如E001。只能输出编号，不能把原文或自行概括的句子放入evidence。只输出合法JSON，不输出Markdown或额外解释。"""
    user_prompt = (
        "请评价以下企业差旅行程。没有语义致命错误时，"
        "semantic_fatal_errors必须输出空数组。\n\n"
        f"【评估材料】\n{json.dumps(evaluation_input, ensure_ascii=False, indent=2)}\n\n"
        f"【输出JSON结构】\n{json.dumps(JUDGE_OUTPUT_CONTRACT, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class LLMItineraryJudge:
    """调用Judge模型，并对格式失败只执行一次结构修复。"""

    def __init__(self, model: Any):
        self.model = model

    async def evaluate(
        self,
        case: Dict[str, Any],
        itinerary_output: Any,
    ) -> Dict[str, Any]:
        parsed_output = robust_json_parse(itinerary_output, fallback=None)
        response = await self.model(
            build_judge_messages(case, parsed_output),
            response_format={"type": "json_object"},
        )
        text = await extract_json_from_async_response(response)
        try:
            return self._parse_and_score(text, parsed_output)
        except Exception as first_error:
            return await self._repair_once(
                text,
                first_error,
                parsed_output,
            )

    @staticmethod
    def _parse_and_score(
        text: str,
        itinerary_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        scored = score_judge_output(
            robust_json_parse(text, fallback=None)
        )
        validate_evidence_grounding(scored, itinerary_output)
        return scored

    async def _repair_once(
        self,
        invalid_text: str,
        validation_error: Exception,
        itinerary_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是JSON结构修复器。只修复字段和类型，保留原评分、理由、"
                    "证据和错误判断，不重新评价行程。只输出合法JSON对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "根对象字段名必须且只能是time_route_feasibility、"
                    "business_personalization、completeness_usability、"
                    "factual_groundedness、semantic_fatal_errors、overall_summary。"
                    "禁止改成accuracy、feasibility、completeness、consistency、"
                    "rationality或personalization等其他名称。"
                    "每个维度包含1到5的整数score、非空reason和"
                    "1到3条evidence。evidence只能填写下方证据目录中存在的E编号，"
                    "不能填写原文或概括。semantic_fatal_errors必须是数组。\n"
                    "semantic_fatal_errors中的evidence也必须是包含1到3个E编号的数组。\n"
                    f"唯一合法的JSON结构：\n"
                    f"{json.dumps(JUDGE_OUTPUT_CONTRACT, ensure_ascii=False, indent=2)}\n"
                    f"校验错误：{validation_error}\n"
                    f"证据目录：\n{json.dumps(build_evidence_catalog(itinerary_output), ensure_ascii=False)}\n"
                    f"待修复内容：\n{invalid_text[:12000]}"
                ),
            },
        ]
        response = await self.model(
            messages,
            response_format={"type": "json_object"},
        )
        repaired_text = await extract_json_from_async_response(response)
        return self._parse_and_score(repaired_text, itinerary_output)
