# Aligo 最终回归与评估收尾报告 v1.0.0

## 1. 目的与口径

本报告用于冻结当前简历项目版本的可运行性和评估结论，不再根据单个模型输出追加 Case 特判。

评估分为两类：

1. **2026-08-16 当前分支实际重跑**：单元测试、数据集合同、固定行程输出的硬规则复算、本地 BGE + Milvus 混合检索。
2. **已归档的真实 LLM 评估**：System Evaluation 与 Itinerary LLM Judge 的固定数据集多次运行结果。本次终端未配置 `DEEPSEEK_API_KEY`，因此不伪造新的真模型运行。

基础代码版本：`75da5b3` (`main`, 已包含 Redis 短期记忆、SQLite 长期记忆和持久化会话摘要)。

## 2. 当前分支回归结果

### 2.1 完整自动化测试

执行：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

结果：

```text
Ran 248 tests in 0.343s
OK (skipped=1)
```

唯一跳过项是需要 `TEST_REDIS_URL` 的真实 Redis 可选集成测试。本次尝试启动 `redis:7-alpine` 时 Docker Hub 镜像拉取长时间无进展，主动终止；Redis 命令语义、TTL、用户/会话隔离和多客户端共享仍由 `fakeredis` 回归覆盖。

### 2.2 评估数据集合同

| 模块 | 校验结果 |
| --- | --- |
| System Evaluation | v0.3.1，15 个场景，19 个对话轮次，通过 |
| Itinerary Quality | v0.3.0，10 个企业差旅场景，通过 |
| RAG Retrieval | v0.2.0-dev，15 题（12 正例 + 3 负例），通过 |

这一步证明“试卷的字段、Case ID、预期结果和执行规模仍符合评估器合同”，不代表真实 LLM 已被重新调用。

### 2.3 行程硬规则复算

使用当前版本 `hard_rule_evaluator` 重新评价已归档的 30 份固定行程输出：

| 指标 | 结果 |
| --- | ---: |
| 行程数 | 30 |
| 硬规则通过 | 28 |
| 硬规则通过率 | **93.3%** |

失败保留为真实模型波动：

- `corporate_budget_and_hotel_policy` 第 3 次；
- `preference_conflicts_with_budget` 第 1 次。

### 2.4 RAG 真实本地检索

执行：

```bash
.venv/bin/python evaluation/rag/run_rag_retrieval_eval.py \
  --output evaluation/rag/results/rag-retrieval-final-regression-v1.0.0.json
```

| 指标 | 结果 |
| --- | ---: |
| Evidence Recall@4 | **91.67%** |
| MRR@4 | **70.14%** |
| Evidence Precision@4 | **25.00%** |
| Negative Rejection Rate | **100%** |
| 执行异常 | 0 |
| 热检索 P50 / P95 | 12.47 ms / 19.45 ms |

唯一未找齐 Gold Evidence 的题目仍是 `hotel_minibar_reimbursement`。本次结果与 v0.2.0 归档一致，说明 Redis/SQLite/会话摘要改造没有影响 RAG 检索链路。延迟为当前本机热检索，不包含 BGE 首次加载。

## 3. 已归档的真实 LLM 评估

### 3.1 System Evaluation v0.4.0

| 配置 | 值 |
| --- | --- |
| 数据集 | 15 个场景，每个 3 次 |
| 总运行 | 45 |
| 总通过率 | **93.3% (42/45)** |
| Critical Pass Rate | **90% (27/30)** |

原报告：`evaluation/system/reports/v0.4.0-optimized-system-evaluation.md`。

### 3.2 Itinerary Quality v0.6.0

| 指标 | 结果 |
| --- | ---: |
| 规划运行 | 30 |
| 有效 Judge 结果 | 28 |
| Judge 基础设施错误 | 2 |
| 平均行程质量分 | **87.93 / 100** |
| 有效样本合格率 | **85.7% (24/28)** |
| 保守端到端合格率 | **80.0% (24/30)** |

| 质量维度 | 平均分（1～5） |
| --- | ---: |
| 时间与路线可行性 | 4.36 |
| 商务与个性化 | 4.29 |
| 完整性与可用性 | 4.43 |
| 事实可靠性 | 4.57 |

原报告：`evaluation/itinerary_quality/reports/v0.6.0-unified-quality-gate-three-run-evaluation.md`。

## 4. 最终结论

- 当前版本的代码级回归通过，记忆存储改造未导致行程硬规则或 RAG 检索指标回退。
- 简历与面试可使用 93.3% 系统通过率、87.93 行程平均质量分和 91.67% RAG Recall@4，但必须同时说明数据集规模与 LLM Judge / 开发集边界。
- 项目已达到实习简历项目的工程收尾条件；后续应优先做演示、README 与面试表达，而不是继续为单个输出追加硬编码规则。

## 5. 边界声明

- 本次未重新调用 DeepSeek 生成 System / Itinerary / Judge 结果，对应数字来自已归档的固定数据集真实运行。
- 未进行真实票务、酒店或企业费控 API 集成测试，因为当前项目没有实现这些外部系统。
- RAG 数据集是开发集；相关指标不得表述为“问答准确率”。
- 2 份 Judge 空响应被单独记录，没有被伪造成低分或从保守端到端口径中删除。

