# 差旅出行助手 System Evaluation 规范（v0.3）

## 1. 目的与阶段边界

项目评估分成两个独立阶段：

1. **System Evaluation**：判断系统有没有理解并按正确流程完成任务；
2. **Task Result Quality Evaluation**：判断最终行程内容是否正确、合理、完整和好用。

当前只实现第一阶段，扩展数据集版本为 v0.3.1。最终回复的措辞、行程安排质量、推荐合理性和表达效果不参与本阶段通过率，第二阶段边界见 `../itinerary_quality/itinerary_quality_evaluation_spec.md`。

这样拆分后，可以明确回答两类问题：

- 系统有没有调用正确的 Agent、补全正确字段并正确更新记忆；
- 在系统流程正确的前提下，生成的行程究竟好不好。

## 2. System Evaluation 评估什么

问题不是凭感觉编写，而是从“业务合同”和“状态变化”反推。第一批场景采用五个角度：

1. **正常路径**：信息完整时，系统能否走完整规划链路；
2. **输入边界**：信息不足时，系统能否阻止后续规划并准确追问；
3. **状态迁移**：用户多轮补充时，待补全状态能否保存、合并和清空；
4. **读写边界**：读取历史时不得偷偷修改记忆，也不得自动填充未经确认的字段；
5. **副作用正确性**：用户明确修改偏好时，是否只执行对应 Agent 并正确写入。

这五类分别覆盖成功路径、失败保护、多轮状态、记忆安全和写入结果。后续新增场景也应先说明它验证哪条业务合同，而不是只堆不同说法的用户问题。

v0.3.0 在原有 5 个核心场景上新增 10 个合同场景：

| 新增场景 | 主要系统合同 |
| --- | --- |
| 单独缺少目的地 | 只追问目的地，不丢失其他字段 |
| 单独缺少出发日期 | 不擅自推断日期，不提前规划 |
| 单独缺少行程时长 | 不擅自生成结束日期，不提前规划 |
| 多轮修改目的地 | 新值覆盖待补全旧值，且不误调用偏好 Agent |
| 新行程覆盖待补全任务 | 明确重新规划时不残留上一任务字段 |
| 查询已保存偏好 | 只读记忆，不产生写入副作用 |
| 替换常驻地 | 使用 replace 覆盖旧值，不追加成列表 |
| 重复追加相同偏好 | 写入操作保持幂等，不产生重复值 |
| 明确授权沿用上次行程 | 区分用户授权复用与系统擅自填充 |
| 重复规划完全相同行程 | 历史记录写入保持幂等 |

当前评估以下六类系统行为：

1. **计划路由**：IntentionAgent 是否把必要 Agent 放进调度计划；
2. **实际执行**：OrchestrationAgent 是否执行必要 Agent、阻止不应执行的 Agent；
3. **实体状态**：出发地、目的地、日期、时长等结构化字段是否正确；
4. **澄清流程**：信息不足时是否暂停规划，并返回正确的 `missing_fields`；
5. **多轮状态**：待补全行程是否跨轮保留、合并，并在完成后清空；
6. **记忆副作用**：偏好和行程历史是否按预期保持、追加或修改。

本阶段明确不评估：

- 回复是否包含某句话；
- 回复是否过长、重复或表达自然；
- 景点顺序、交通方案和酒店推荐是否合理；
- 具体车次、天气、票价等生成内容是否可信；
- 行程的个性化程度和可执行性。

## 3. 核心系统合同

### 3.1 行程规划

规划前必须具备：

- `origin`；
- `destination`；
- `start_date`；
- `duration_days` 或 `end_date` 中至少一个。

缺少必填字段时：

- 状态必须为 `needs_clarification`；
- `itinerary_planning` 不得实际执行；
- 已知字段写入待补全状态；
- 后续轮次继续合并用户补充的信息。

规划成功后：

- 状态为 `completed`；
- 待补全状态清空；
- 成功行程写入一次历史记录。

`return_location` 和 `trip_purpose` 是可选字段，不得单独阻止规划。

### 3.2 偏好与历史

- `memory_query` 只读取记忆，不修改偏好；
- `preference` 只在用户明确新增、追加、替换或删除偏好时调用；
- “按照我之前的偏好规划”是读取，不是修改；
- 历史行程只能作为候选信息，用户未确认时不得自动填充当前行程；
- 不同评估场景使用独立临时记忆，互不污染。

### 3.3 调度依赖

- 事项收集等前置任务先于 `itinerary_planning` 完成；
- 同优先级且无依赖的任务允许并行；
- 单个 Agent 失败不应丢失其他 Agent 的成功结果；
- 计划调度和实际执行必须分别记录。

## 4. 数据与评分口径

系统场景保存在 `system_cases.json`。每轮期望只包含结构化系统断言：

```json
{
  "required_scheduled_agents": ["event_collection", "itinerary_planning"],
  "forbidden_scheduled_agents": ["preference"],
  "required_executed_agents": ["event_collection"],
  "forbidden_executed_agents": ["itinerary_planning", "preference"],
  "forbidden_entity_fields": ["origin", "start_date", "duration_days"],
  "status": "needs_clarification",
  "entities": {"destination": "北京"},
  "missing_fields": ["origin", "start_date", "duration_days"],
  "memory": {
    "preferences_change": "none",
    "trip_history_change": "none",
    "pending_trip": {"destination": "北京"}
  }
}
```

不再使用 `must_contain` 和 `must_not_contain_patterns`。原始回复仍保存在执行轨迹中，仅用于排查问题和未来第二阶段复用。

`forbidden_entity_fields` 用于检查系统是否自动补出了用户没有提供、也没有确认的实体。字段不存在、值为 `null`，或使用“未提供、未指定、未明确、未确定、未知、待补充、N/A”等缺失信息占位表达时通过；真正出现实体值才失败。评估器按“缺失语义”处理同义表达及其解释文本，不只针对某一个固定词；但“苏州（从历史推断）”这类包含真实实体的值仍判为已填写。

`required_scheduled_agents` 和 `forbidden_scheduled_agents` 检查 IntentionAgent 的计划路由；`required_executed_agents` 和 `forbidden_executed_agents` 检查 OrchestrationAgent 的真实执行。这样即使多余 Agent 最终被拦截，也能发现上游路由误判。

v0.2 对待补全状态作如下校准：

- 期望包含字段时，按字段子集比较，允许系统额外提取可选字段；
- 期望为 `{}` 时，实际状态也必须为空，防止任务完成后残留脏状态。

偏好的列表值按包含关系检查，允许记忆中存在与本次任务无关的其他合法偏好。

## 5. 当前指标

对外汇报优先使用三个简单指标：

- **System Case Pass Rate**：完整满足全部系统断言的运行次数占比；
- **Critical Pass Rate**：严重场景的完整通过率；
- **Repeated-run Stability**：同一场景重复运行时的通过比例。

失败报告保留具体诊断项，例如 `status`、`entities`、`memory.pending_trip`，用于定位是路由、状态还是记忆问题。当前 v0.2 不把这些诊断项包装成一堆独立的简历指标。

## 6. 文件职责

```text
evaluation/
├── system/                               # 系统行为评估
│   ├── system_evaluation_spec.md
│   ├── system_cases.json
│   ├── system_evaluator.py
│   ├── system_trace_collector.py
│   ├── system_eval_runner.py
│   ├── run_system_eval.py
│   ├── reports/                          # 人工归档的版本报告
│   └── results/                          # 自动生成报告（Git 忽略）
├── itinerary_quality/                    # 行程结果质量评估
│   ├── itinerary_quality_evaluation_spec.md
│   └── itinerary_quality_cases.json
└── rag/                                  # RAG 检索与生成评估
    ├── rag_cases.json
    ├── rag_evaluator.py
    └── run_local_retrieval_eval.py
```

## 7. 使用方式

只校验数据集和预计执行量，不调用模型：

```bash
python evaluation/system/run_system_eval.py --dry-run
```

真实运行全部系统场景一次：

```bash
python evaluation/system/run_system_eval.py --runs 1
```

真实运行一个指定场景：

```bash
python evaluation/system/run_system_eval.py \
  --case trip_missing_required_fields \
  --runs 1
```

评估报告记录数据集版本、模型配置、每轮轨迹、失败断言和汇总通过率。自动生成的原始 JSON 不提交到 Git；每个重要版本只提交人工整理后的基线总结。
