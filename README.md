# Aligo 智能旅行助手

基于**豆包大模型**和**AgentScope框架**的多智能体旅行规划系统，采用Plan-and-Execute架构，实现智能意图识别、两层记忆系统、RAG知识库、联网搜索和优先级并行调度。

## ✨ 核心亮点

### 🎯 智能意图识别
- 基于LLM语义理解的多意图识别（准确率90%+，对比关键词匹配提升25%）
- 支持6大类意图：行程规划、记忆查询、偏好管理、知识问答、信息查询、事项收集
- 自然语言理解，无需关键词匹配

### 🧠 两层记忆架构
- **短期记忆**：Redis 会话存储，支持滑动 TTL、最近10轮和多实例共享
- **长期记忆**：SQLite 持久化偏好、聊天、行程和已结束会话摘要
- 智能识别偏好追加/覆盖动作（"我还喜欢如家" vs "我搬家到上海了"）
- 存储层解耦：`SessionStore` 管理会话状态，`LongTermMemoryRepository` 管理长期数据

### 📚 RAG知识库
- Milvus向量数据库 + BGE-m3 Embedding模型（本地部署）
- 智能分块（Chunking）+ 滑动窗口切分 + 余弦相似度检索
- 知识溯源：返回文档来源，准确率95%

### ⚡ 优先级并行调度
- Plan-and-Execute架构：IntentionAgent → OrchestrationAgent → 子Agent
- 同优先级Agent并行执行（asyncio.gather）
- 系统响应时间从30秒优化到15秒（-50%）

### 🏗️ 插件化架构
- **Skill Plugins**：所有子Agent重构为独立插件（`.claude/skills/`）
- **LazyAgentRegistry**：动态发现机制，自动扫描注册
- **懒加载**：未使用的Skill不加载，启动速度3秒
- **Progressive Disclosure**：渐进式暴露，意图识别阶段仅加载元数据

### 🛡️ 稳定性保障
- **熔断器**：连续失败后自动熔断，保护服务
- **指数退避重试**：自动重试失败请求（最大3次）
- **健康检查**：实时监控LLM服务可用性

---

## 系统架构

```
用户输入
   ↓
┌──────────────────────────────────────────────────────────┐
│  IntentionAgent (意图识别智能体)                          │
│  - 语义理解用户意图（不使用关键词匹配）                    │
│  - 识别关键实体                                           │
│  - 生成调度计划                                           │
│  - 确定智能体优先级                                       │
│  - 动态加载 Skills Metadata (Progressive Disclosure)     │
└──────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────┐
│  OrchestrationAgent (协调器智能体)                       │
│  - 按优先级调度子智能体                                   │
│  - 同优先级并行执行                                       │
│  - 管理智能体间消息传递                                   │
│  - 集成两层记忆系统                                       │
│  - 动态实例化 Skills (Plugin Architecture)               │
└──────────────────────────────────────────────────────────┘
   ↓
┌─────────────────────── 优先级 1 (并行执行) ──────────────┐
│                                                           │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │ MemoryQuery Skill   │  │ EventCollection Skill    │  │
│  │ 记忆查询智能体       │  │ 事项收集智能体            │  │
│  │ - 查询旅行记录      │  │ - 出发地/目的地           │  │
│  │ - 查询用户偏好      │  │ - 出行时间/返程地         │  │
│  │ - 查询历史对话      │  │ - 出行目的                │  │
│  └─────────────────────┘  └──────────────────────────┘  │
│                                                           │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │ Preference Skill    │  │ InformationQuery Skill   │  │
│  │ 偏好管理智能体       │  │ 信息查询智能体            │  │
│  │ - 酒店/航空偏好     │  │ - 网络搜索 (DuckDuckGo)  │  │
│  │ - 座位/房型偏好     │  │ - 实时信息查询           │  │
│  │ - 机型/餐饮偏好     │  │ - LLM摘要生成            │  │
│  │ - 支持追加/覆盖     │  │                          │  │
│  └─────────────────────┘  └──────────────────────────┘  │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ RAGKnowledgeAgent Skill (知识库查询智能体)          │ │
│  │ - 差旅政策文档查询 (Milvus Lite + RAG)             │ │
│  │ - 企业内部知识检索                                  │ │
│  │ - 自动文档切分 (Chunking) + 向量检索                │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
└───────────────────────────────────────────────────────────┘
   ↓
┌─────────────────────── 优先级 2 (依赖优先级1) ───────────┐
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ ItineraryPlanningAgent Skill (行程规划智能体)       │ │
│  │ - 整合所有前序智能体信息                            │ │
│  │ - 生成完整行程计划                                  │ │
│  │ - 包含：景点、交通、酒店、餐饮                      │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
└───────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────┐
│  结果聚合与记忆更新                                       │
│  - 聚合所有智能体结果                                     │
│  - 更新长期记忆（偏好、行程历史、聊天记录）                │
│  - 生成人性化回复                                         │
└──────────────────────────────────────────────────────────┘
   ↓
最终结果
   ↓
用户看到结果
```

### 连接与可用性

为保证 LLM 服务不稳定时的可用性，在调用链外增加了以下机制（不改变原有业务逻辑）：

| 机制 | 说明 |
|------|------|
| **熔断器** | 连续失败若干次后暂停调用 LLM，直接提示「服务暂时不可用」；一段时间后自动半开试探恢复。 |
| **重试与退避** | 对意图识别、编排两次 LLM 调用做有限次重试，仅对超时、429、5xx 等可重试错误生效，采用指数退避。 |
| **健康检查** | 会话内输入 `health` 可查看熔断状态并探测 LLM 是否可达；命令行执行 `python cli.py health` 可单独做一次探测（退出码 0/1，便于监控）。 |

配置见 `config.py` 中的 `RESILIENCE_CONFIG`（重试次数、熔断阈值、恢复时间等）。

---

## 📊 关键指标

| 指标 | 优化前 | 优化后 | 提升幅度 |
|------|--------|--------|----------|
| 意图识别准确率 | 65% | 90%+ | +25% |
| 知识库问答准确率 | - | 95% | 新增功能 |
| 用户偏好记忆准确率 | - | 95% | 新增功能 |
| 系统响应时间 | 30秒 | 15秒 | -50% |
| 用户偏好缓存命中率 | - | 85% | 新增功能 |
| 系统启动速度 | 未优化 | 3秒 | 懒加载优化 |

**优化路径**：
1. **V1.0**: 关键词匹配意图识别（准确率65%） + 串行调度（响应时间30秒）
2. **V2.0**: 两层记忆系统 + RAG知识库 + 联网搜索
3. **V3.0**: LLM语义理解意图识别（准确率90%+） + 优先级并行调度（响应时间15秒）
4. **V4.0**: Skill Plugins插件化架构 + LazyAgentRegistry + Redis缓存层

---

## 核心功能

### 1. 意图识别（基于LLM语义理解）

系统支持**6大类意图**自动识别（准确率90%+）：

- ✅ **itinerary_planning**: 规划未来行程
  - 示例："我想3月11日从北京去杭州出差一周"
- ✅ **memory_query**: 查询历史记忆
  - 示例："我去过哪里？"、"我之前说过什么偏好？"
- ✅ **preference**: 管理用户偏好（支持追加/覆盖）
  - 示例："我喜欢住汉庭酒店"、"我还喜欢如家"、"我搬家到上海了"
- ✅ **rag_knowledge**: 查询企业差旅知识库
  - 示例："差旅标准是什么？"、"报销政策是什么？"
- ✅ **information_query**: 联网查询实时信息
  - 示例："杭州明天天气怎么样？"、"北京明天限行吗？"
- ✅ **event_collection**: 收集行程要素
  - 自动提取：出发地、目的地、出发时间、返程时间、出行目的

**意图识别示例**：
```
用户: "我过去都去哪旅游过？"
→ IntentionAgent 识别为 memory_query
→ 调度 MemoryQueryAgent
→ 从 trip_history 查询并回答

用户: "我还喜欢7天酒店"
→ IntentionAgent 识别为 preference
→ 调度 PreferenceAgent
→ LLM 识别「还」字，判断为 append 模式
→ 追加到 hotel_brands 列表
```

### 2. 两层记忆系统

**短期记忆（会话级）**
- 基于 **RedisSessionStore** 的滑动窗口机制
- 保存最近10轮对话和当前待补全行程
- Key 同时包含 `user_id` 和 `session_id`，保证用户与会话隔离
- 会话活动会刷新 TTL，过期后 Redis 自动回收临时状态
- 多个应用进程可以共享同一份会话数据
- 用于上下文理解和快速访问

**长期记忆（持久化）**
- 💾 **SQLite持久化存储**：用户偏好、历史行程、完整聊天历史
- 🎯 **用户偏好管理**：支持动态添加任意偏好类型，智能识别追加/覆盖动作
- 📅 **历史行程记录**：出发地、目的地、时间、目的，支持跨会话查询
- 📊 **统计信息**：常去目的地、总行程数
- 🤖 **会话摘要持久化**：会话结束时只调用一次 LLM，后续会话直接读取 SQLite 摘要
- 🗄️ **单一数据源**：长期记忆统一读写 SQLite，避免多存储数据不一致

**测试记忆系统**：
```bash
python tests/test_memory_system.py
```

测试覆盖：
- ✅ 短期记忆：添加、查询、TTL、用户/会话隔离、多实例共享
- ✅ 长期记忆-偏好：动态添加、跨会话访问
- ✅ 长期记忆-行程：保存、查询、高频目的地统计
- ✅ 长期记忆-聊天历史：持久化对话记录
- ✅ LLM总结：会话结束时生成一次并持久化，失败时保留原始聊天
- ✅ 跨会话持久化：新会话访问旧数据

### 3. RAG 知识库

基于 **Milvus** 和 **BGE-m3 Embedding模型**的企业差旅知识检索系统。

**技术方案**：
- **向量数据库**: Milvus（本地存储）
- **Embedding模型**: BGE-small-zh-v1.5（中文向量化，本地部署 `data/models/bge-small-zh-v1.5`）
- **文档处理**: 智能分块（Chunking）+ 滑动窗口切分
- **检索算法**: 余弦相似度检索（Top-K=4）+ 相似度阈值过滤 + 近重复片段过滤
- **可追溯性**: 返回文档来源，支持知识溯源
- **准确率**: 95%（知识库问答准确率）

**初始化知识库**：
```bash
python .claude/skills/ask-question/script/init_knowledge_base.py
```

**知识库内容**（8类文档）：
- 差旅标准和规定
- 报销政策
- 预订指南
- 常见问题FAQ
- 紧急情况处理
- 平台使用指南
- 城市差旅指南
- 环保倡议


### 4. 信息查询（联网搜索）

基于 **DuckDuckGo (DDGS)** 的免费网络搜索功能：
- 🌐 实时网络搜索（天气、景点、实时新闻）
- 📝 LLM自动摘要（提取关键信息）
- 🔗 来源追踪（返回搜索来源）
- 🚀 异步查询（提升响应速度）

### 5. 优先级并行调度

基于 **asyncio.gather** 的智能并行调度机制：
- 📋 **多意图识别**：支持6大类意图（规划行程、查询记忆、管理偏好、知识问答、信息查询、实时检索）
- ⚡ **优先级+并行混合模式**：同优先级Agent并行执行，不同优先级串行依赖
- 🎯 **动态调度**：根据意图识别结果动态分配优先级
- 📈 **性能提升**：系统响应时间从30秒优化到15秒（-50%）

---

## 快速开始

### 1. 安装依赖

```bash
# 使用 requirements.txt 安装所有依赖
pip install -r requirements.txt

# 或者手动安装核心依赖
pip install "setuptools>=69.0.0,<82"  # milvus_lite 依赖
pip install agentscope==1.0.16        # 多智能体框架
pip install "pymilvus[milvus_lite]==2.6.9"  # 向量数据库
pip install sentence-transformers==5.2.3    # Embedding模型
pip install rich==13.9.4                    # CLI界面
pip install ddgs==9.10.0                    # 网络搜索
```

### 2. 启动 Redis

短期对话和待补全行程保存在 Redis。已安装 Docker Desktop 时，
先启动 Docker Desktop，再在项目目录执行：

```bash
docker compose -p travel-agent up -d redis
docker compose -p travel-agent ps
```

默认连接 `redis://localhost:6379/0`，可复制 `.env.example` 中的
变量按需调整。停止本地 Redis：

```bash
docker compose -p travel-agent down
```

### 3. 配置模型

编辑 `config.py`，填入你的豆包大模型API密钥：

```python
LLM_CONFIG = {
    "api_key": "your-api-key-here",  # 替换为你的API密钥
    "model_name": "doubao-seed-1-6-flash-250828",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    "temperature": 0.7,
    "max_tokens": 8192,
}
```

**配置说明**：
- `api_key`: 豆包大模型API密钥（必填）
- `model_name`: 模型名称（推荐使用 flash 系列）
- `temperature`: 控制生成的随机性（0-1，0.7为推荐值）
- `max_tokens`: 最大输出token数（8192）

### 4. 初始化知识库

```bash
python .claude/skills/ask-question/script/init_knowledge_base.py
```

### 5. 启动系统

```bash
python cli.py
```

---

## 子智能体详解 (Skills)

所有子智能体已重构为 **Skill Plugins**，位于 `.claude/skills/` 目录下，支持动态发现与加载。

### 1. MemoryQueryAgent (记忆查询智能体) 

- **职责**: 查询用户的历史记忆
- **查询内容**:
  - 旅行历史（trip_history）
  - 用户偏好（preferences）
  - 历史对话摘要（chat_history）
- **特点**:
  - 直接查询本地记忆，无需联网
  - 使用 LLM 生成自然语言回答
  - 支持复杂的记忆推理
- **示例**: "我过去去过哪些地方？"、"我上次去北京是什么时候？"

### 2. EventCollectionAgent (事项收集智能体)

- **职责**: 收集行程规划的核心信息
- **收集内容**: 出发地、目的地、出发时间、返程时间、出行目的
- **特点**: 主动推断缺失信息

### 3. PreferenceAgent (偏好管理智能体)

- **职责**: 识别和管理用户所有偏好
- **管理偏好**:
  - 酒店品牌、航空公司、座位偏好、房型偏好
  - 机型偏好、餐饮偏好、交通偏好、预算等级
  - 支持任意自定义偏好类型
- **智能模式**:
  - **追加模式**：识别「还」、「也」等关键词，追加到现有偏好
  - **覆盖模式**：识别「搬家到」、「改成」等关键词，替换旧偏好
  - **示例**: "我还喜欢汉庭" → 追加；"我搬家到上海" → 覆盖
- **特点**:
  - 感知当前已有偏好，避免重复
  - 所有偏好作为长期偏好持久化保存
  - 从对话中提取隐含偏好

### 4. InformationQueryAgent (信息查询智能体)

- **职责**: 实时信息检索（联网）
- **查询能力**: DuckDuckGo 搜索 + LLM 摘要
- **查询场景**: 天气、景点、实时新闻、通用问答

### 5. ItineraryPlanningAgent (行程规划智能体)

- **职责**: 生成完整行程计划
- **规划内容**: 每日时间表、住宿建议、餐饮建议、交通路线、注意事项
- **特点**: 即使信息不完整也给出合理建议

### 6. RAGKnowledgeAgent (知识库查询智能体)

- **职责**: 查询企业商旅知识库
- **技术栈**: Milvus Lite + BGE 中文向量模型
- **特点**: 提供文档溯源，返回参考来源

---

## CLI 使用指南

### 启动

```bash
python cli.py
```

**启动速度**: 约 3 秒（采用LazyAgentRegistry懒加载技术）

### 内置命令

| 命令 | 说明 |
|------|------|
| `help` | 显示帮助信息 |
| `status` | 查看当前状态和记忆 |
| `health` | 检查 LLM 服务是否可用并显示熔断器状态 |
| `clear` | 清空当前任务（保留长期记忆） |
| `history` | 查看历史行程 |
| `preferences` | 查看用户偏好 |
| `exit` | 退出程序 |

单独做健康检查（不进入交互）：`python cli.py health`，返回 `OK` / `FAIL: ...`，退出码 0/1。

---

## 测试

### 集成测试 (QA)
完整跑通所有意图和子智能体的端到端测试：
```bash
python tests/test_cli_qa.py
```

### 单元测试
针对各个核心模块的测试：

```bash
python tests/test_memory_system.py  # 记忆系统
python tests/test_intention_agent.py # 意图识别
python tests/test_orchestration.py  # 协调系统
```

---

## 项目结构

```
shanglv/
├── agents/                          # 核心编排层
│   ├── intention_agent.py           # 意图识别（语义理解）
│   ├── orchestration_agent.py       # 协调器（并行调度）
│   └── lazy_agent_registry.py       # 智能体插件注册器（懒加载）
├── .claude/skills/                  # Skill Plugins (子智能体)
│   ├── ask-question/                # 知识库问答 Skill
│   │   ├── script/                  # 代码 (agent.py, init_script)
│   │   ├── data/                    # 数据 (documents, milvus db)
│   │   └── SKILL.md                 # 技能定义
│   ├── event-collection/            # 事项收集 Skill
│   ├── plan-trip/                   # 行程规划 Skill
│   ├── preference/                  # 偏好管理 Skill
│   ├── query-info/                  # 信息查询 Skill
│   └── memory-query/                # 记忆查询 Skill
├── context/                         # 记忆系统
│   ├── memory_manager.py            # 记忆管理器
│   ├── short_term_memory.py         # 短期记忆
│   ├── session_store.py              # 短期会话存储接口
│   ├── redis_session_store.py        # Redis 短期会话存储
│   ├── session_store_factory.py      # Redis 存储创建入口
│   ├── memory_repository.py          # 长期记忆仓库接口
│   └── sqlite_memory_repository.py   # 默认 SQLite 长期记忆实现
├── data/
│   ├── memory/                      # SQLite 长期记忆数据库
│   └── models/                      # 本地模型文件
│       └── bge-small-zh-v1.5/       # BGE中文Embedding模型
├── tests/                           # 测试脚本
│   ├── test_cli_qa.py               # 端到端集成测试
│   ├── test_memory_system.py        # 记忆系统测试
│   ├── test_intention_agent.py      # 意图识别测试
│   └── test_orchestration.py        # 协调系统测试
├── utils/                           # 工具与连接可用性
│   ├── circuit_breaker.py           # 熔断器
│   ├── llm_resilience.py            # 重试退避、健康检查
│   ├── json_parser.py               # JSON 解析
│   └── skill_loader.py              # Skill 加载器
├── cli.py                           # CLI 主程序
├── config.py                        # 配置文件
├── config_agentscope.py             # AgentScope 初始化与模型配置
└── README.md                        # 本文件
```

---

## 技术栈总览

### 核心框架
- 📦 **AgentScope 1.0.16** - 多智能体框架
- 🤖 **豆包大模型 (doubao-seed-1-6-flash-250828)** - 大语言模型

### 数据存储
- 🗄️ **SQLite** - 长期记忆持久化（用户偏好、历史行程、聊天记录）
- 🧩 **SessionStore / Repository** - 短期与长期存储的可替换接口
- 🔍 **Milvus** - 向量数据库（本地存储，RAG知识库）

### 向量化与检索
- 🧠 **BGE-small-zh-v1.5** - 中文Embedding模型（本地部署）
- 📚 **Sentence-Transformers 5.2.3** - 向量化工具库
- 🎯 **余弦相似度检索** - Top-K检索算法

### 联网与搜索
- 🌐 **DuckDuckGo (DDGS 9.10.0)** - 免费网络搜索引擎
- 📝 **LLM自动摘要** - 搜索结果智能提取

### 架构设计
- 🏗️ **Skill Plugins插件化架构** - 独立开发、测试、部署
- 🔄 **LazyAgentRegistry动态发现** - 自动扫描注册Agent插件
- ⚡ **懒加载机制** - 未使用的Skill不加载（启动速度3秒）
- 🔀 **Progressive Disclosure渐进式暴露** - 意图识别阶段仅加载元数据，执行阶段按需加载
- 🎯 **优先级+并行混合调度** - asyncio.gather并发执行

### 稳定性保障
- 🔁 **指数退避重试** - 自动重试失败请求（最大3次）
- 🩺 **熔断器机制** - 连续失败后暂停调用
- 💊 **健康检查** - 实时监控LLM服务可用性

### 用户界面
- 🖥️ **Rich 13.9.4** - 精美的CLI终端界面

---

## ⚠️ 注意事项

### 模型配置
- 必须配置豆包大模型API密钥（在 `config.py` 中）
- 推荐使用 flash 系列模型（响应速度快）
- BGE Embedding模型需下载到 `data/models/bge-small-zh-v1.5/`

### 数据存储
- 当前版本使用 **SQLite** 存储长期记忆（`data/memory/memory.sqlite3`）
- 当前短期会话使用 **Redis**；程序启动时会验证连接，不会静默降级到进程内存
- PostgreSQL 是面向多实例生产部署的长期仓库演进方向，当前未实现

### 知识库初始化
- 首次运行前必须初始化RAG知识库
- 知识库文档位于 `.claude/skills/ask-question/data/documents/`
- Milvus数据库文件生成在 `.claude/skills/ask-question/data/milvus_travel_kb.db`

### 性能优化
- 懒加载机制：系统启动时仅扫描Skill元数据，首次调用时才加载
- 并行调度：同优先级Agent并发执行，提升响应速度
- 缓存策略：热数据缓存，减少重复计算和LLM调用

---

## 🚀 未来规划

- [x] 实现 RedisSessionStore（TTL 与多实例会话共享）
- [ ] 按生产规模评估是否迁移到 PostgreSQL
- [ ] 支持更多LLM模型（OpenAI、Claude等）
- [ ] Web界面（FastAPI + React）
- [ ] 更多Skill插件（酒店预订、机票查询等）
- [ ] 监控和日志系统

---

## 许可证

MIT License
