# Evaluation 目录说明

评估按业务问题分为三个独立模块，避免把系统流程、最终行程和 RAG 问答混成一个总分。

```text
evaluation/
├── system/             # 路由、执行、状态与记忆副作用
├── itinerary_quality/  # 最终行程的硬约束与内容质量
└── rag/                # 知识库检索与回答质量
```

## System Evaluation

```bash
python evaluation/system/run_system_eval.py --dry-run
python evaluation/system/run_system_eval.py --runs 1
```

人工归档报告放在 `system/reports/`，自动生成的原始结果放在 `system/results/`。

## Itinerary Quality Evaluation

只校验数据集，不调用模型：

```bash
python -m evaluation.itinerary_quality.hard_rule_evaluator
```

查看真实评估的执行规模，不调用模型：

```bash
python evaluation/itinerary_quality/run_itinerary_quality_eval.py --dry-run
```

运行全部场景一次：

```bash
python evaluation/itinerary_quality/run_itinerary_quality_eval.py --runs 1
```

复用已有行程运行LLM Judge，不重新生成行程：

```bash
python evaluation/itinerary_quality/run_itinerary_llm_judge.py --dry-run
python evaluation/itinerary_quality/run_itinerary_llm_judge.py
```

人工归档报告放在 `itinerary_quality/reports/`，自动生成的原始结果放在 `itinerary_quality/results/`。

使用指定场景评价一份已保存的行程 JSON：

```bash
python -m evaluation.itinerary_quality.hard_rule_evaluator \
  --case standard_three_day_business_trip \
  --output /path/to/itinerary.json
```

## RAG Evaluation

```bash
python evaluation/rag/run_local_retrieval_eval.py
```

三个模块分别维护自己的数据集、评估器和结果，不合并成一个项目总分。
