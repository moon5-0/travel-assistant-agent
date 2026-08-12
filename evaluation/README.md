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
python evaluation/itinerary_quality/generate_itinerary_outputs.py --dry-run
```

运行全部场景一次：

```bash
python evaluation/itinerary_quality/generate_itinerary_outputs.py --runs 1
```

复用已有行程运行LLM Judge，不重新生成行程：

```bash
python evaluation/itinerary_quality/judge_itinerary_outputs.py --dry-run
python evaluation/itinerary_quality/judge_itinerary_outputs.py \
  --input-report /path/to/generated-itinerary-report.json
```

人工归档报告放在 `itinerary_quality/reports/`，自动生成的原始结果放在 `itinerary_quality/results/`。

使用指定场景评价一份已保存的行程 JSON：

```bash
python -m evaluation.itinerary_quality.hard_rule_evaluator \
  --case standard_three_day_business_trip \
  --output /path/to/itinerary.json
```

## RAG Evaluation

旧版 RAG 评估原型已移除。v0.2 当前只建设检索质量评估；回答质量与端到端评估暂列为后续方向。新版评估边界、数据集设计和指标定义见：

```text
evaluation/rag/rag_evaluation_spec.md
```

当前已实现证据级检索开发集、真实 Milvus Lite 运行入口，以及 BGE 向量 + BM25 + RRF 混合检索；独立测试集将在开发集校准后另行冻结。

校验 RAG 检索开发集和执行规模，不加载 BGE/Milvus：

```bash
python evaluation/rag/run_rag_retrieval_eval.py --dry-run
```

在真实 BGE + Milvus Lite 链路上运行15道开发题：

```bash
python evaluation/rag/run_rag_retrieval_eval.py
```

只运行一个指定 Case：

```bash
python evaluation/rag/run_rag_retrieval_eval.py \
  --case hotel_limit_beijing_paraphrase
```

原始报告写入 `rag/results/` 并由 Git 忽略。当前结果只作为开发集基线，不能作为独立测试集最终成绩。

可提交、可追踪的阶段结果归档在：

```text
evaluation/rag/reports/v0.1.0-vector-retrieval-baseline.md
evaluation/rag/reports/v0.2.0-hybrid-retrieval-optimization.md
```

三个模块分别维护自己的数据集、评估器和结果，不合并成一个项目总分。
