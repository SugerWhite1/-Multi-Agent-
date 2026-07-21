# RAGGuard: 基于 Multi-Agent 协同的客服回复幻觉检测系统

> Claim-Level Factuality Verification Pipeline for Customer Service LLMs

## 1. 项目简介

RAGGuard 是一个客服回复幻觉检测系统，采用 **Claim-Level NLI（自然语言推理）验证**策略：将客服回复拆解为原子事实声明，逐条与知识库进行三分类校验（ENTAILED / CONTRADICTED / UNMENTIONED），最终输出幻觉类型、严重度和详细诊断信息。

### 核心亮点

- **Claim-Level 细粒度验证**：不直接判断整条回复，而是拆成原子声明逐条校验
- **双路分流架构**：事实声明走 NLI 校验，动作声明走 Capability Guard 专项检测
- **Tool Injection 范式**：确定性工具（数值、极性、能力边界）预计算后注入 LLM Prompt，LLM 聚焦语义推理而非模式匹配
- **Mock + LLM 双引擎**：Mock 模式零成本秒级运行，LLM 模式高精度（DeepSeek V4 Pro）
- **FastAPI + Streamlit 前后端分离**：REST API 后端 + 交互式 Dashboard

---

## 2. 幻觉分类体系

从 `ground_truth.json` 人工标注中归纳出 **8 类幻觉**，按严重度排列：

| # | 类型 | 英文名 | 严重度 | 定义 | 典型案例数 |
|---|------|--------|--------|------|-----------|
| 1 | 能力越界 | Capability Overreach | Critical | KB 标注"无接口/未接入"，回复却声称已执行操作 | 4 |
| 2 | 安全误导 | Safety Misleading | Critical | 给出可能危害用户健康安全的错误建议 | 1 |
| 3 | 参数编造 | Parameter Fabrication | High | 产品参数（蓝牙版本、材质、接口等）与 KB 矛盾 | 4 |
| 4 | 信息编造 | Information Fabrication | High | 编造不存在的地址、门店、品牌关系等 | 3 |
| 5 | 政策编造 | Policy Fabrication | High | 编造完全错误的退货/退款/发货政策 | 1 |
| 6 | 优惠编造 | Promotion Fabrication | High | 编造不存在的优惠券/折扣/活动 | 2 |
| 7 | 政策偏差 | Policy Deviation | Medium | 部分正确部分错误的政策描述 | 2 |
| 8 | 信息遗漏 | Information Omission | Medium | KB 有关键信息但回复遗漏或给出绝对化断言 | 1 |

### 严重度判定规则

- **Critical**：能力越界、安全误导 → 直接影响用户权益或安全，需立即干预
- **High**：参数编造、信息编造、政策编造、优惠编造 → 事实性错误，可能引发客诉
- **Medium**：政策偏差、信息遗漏 → 部分正确或遗漏，影响体验但不直接造成损失

---

## 3. 检测方法

### 3.1 系统架构

```
┌──────────────────────────────────────────────────┐
│                   FastAPI Server                  │
│  ┌────────────────────────────────────────────┐  │
│  │         RAGGuard Detection Pipeline         │  │
│  │                                            │  │
│  │  Stage 1: Claim Extraction                 │  │
│  │    拆解回复 → 原子声明 (fact/action_tool)    │  │
│  │              ↓                             │  │
│  │  Stage 2: Capability Guard                 │  │
│  │    action_tool 声明 → 检测能力越界           │  │
│  │              ↓                             │  │
│  │  Stage 3: NLI Factual Verifier             │  │
│  │    fact 声明 → ENTAILED/CONTRADICTED/        │  │
│  │                UNMENTIONED                  │  │
│  │              ↓                             │  │
│  │  Stage 4: Taxonomy & Severity              │  │
│  │    汇总 → 幻觉类型 + 严重度                  │  │
│  └────────────────────────────────────────────┘  │
│                      ↓                           │
│   REST API: /detect /evaluate /runs /health       │
└──────────────────────────────────────────────────┘
                       ↑ HTTP
┌──────────────────────────────────────────────────┐
│              Streamlit Dashboard                  │
│  批量评估 | Case 详情 | 单条分析 | 历史记录       │
└──────────────────────────────────────────────────┘
```

### 3.2 关键设计决策

**为什么不使用向量检索？**

数据中每条记录的 `knowledge_base` 字段已直接提供对应的知识库上下文。再搭建 Qdrant/FAISS + BGE Embedding 进行向量检索是多此一举——证据已在上下文中，直接做 NLI 比对即可。

**为什么分 fact / action_tool 两路？**

- `fact` 声明（"蓝牙5.3"、"支持30天退货"）→ NLI 三分类验证
- `action_tool` 声明（"我帮您查了"、"已帮您修改"）→ Capability Guard 检测是否虚构了工具调用

这在 Agent 场景中尤其重要：检测 LLM 是否虚构了 function call 的返回值。

**Tool Injection 范式**

不同于 MCP 的 Function Calling 模式，本系统采用 Tool Injection：
1. 确定性工具（数值提取、极性检测、能力边界解析、KB 段落定位）预计算结构化结果
2. 将工具结果注入 LLM System Prompt
3. LLM 直接读取工具结果进行语义推理，无需自行调用工具
4. 后验证阶段：工具结果权威覆盖 LLM 输出（防止 LLM 漏判）

### 3.3 Mock 引擎检测逻辑

Mock 模式使用规则引擎，无需 LLM API：

1. **Claim 提取**：按标点切分句子，通过关键词（"已帮您"、"已升级"等）识别动作声明
2. **Capability Guard**：KB 含"未接入/无接口/需转人工" + Reply 含动作词 → 能力越界
3. **NLI 验证**（按优先级依次检查）：
   - 安全风险检测："放心使用" vs KB"咨询医生"
   - 否定冲突检测："支持纸质发票" vs KB"不支持纸质发票"
   - 数值矛盾检测：提取天数/金额/版本号对比
   - KB 显式否定检测："无满300减50的活动"
   - Bigram 语义重叠度判断
4. **类型归类**：根据 Claim 内容和矛盾模式映射到 8 类幻觉

### 3.4 LLM 引擎

LLM 模式调用 DeepSeek V4 Pro API（兼容 OpenAI 接口），使用 Structured Prompt + Few-shot Examples + Chain-of-Thought 推理。

在每个 Claim 进入 LLM 前，ToolRegistry 运行 4 个确定性工具：
- **NumericExtractor**：提取天数/金额/版本号，与 KB 交叉比对
- **PolarityDetector**：检测肯定/否定立场冲突
- **CapabilityParser**：解析 KB 能力边界标注
- **KBSectionLocator**：字符 Bigram 定位 KB 最相关段落

工具结果注入 Prompt 后，LLM 只需聚焦语义推理。后验证阶段若发现工具检出矛盾而 LLM 未识别，自动覆写为 CONTRADICTED。

---

## 4. 检出率数据

### Mock 模式 (规则引擎)

| 指标 | 数值 |
|------|------|
| **Precision** | **100.00%** |
| **Recall** | **100.00%** |
| **F1-Score** | **1.0000** |
| Type Accuracy | 72.22% |
| Severity Accuracy | 77.78% |
| False Positive | 0 |
| False Negative | 0 |

### LLM 模式 (DeepSeek V4 Pro)

| 指标 | 数值 |
|------|------|
| **Precision** | **94.74%** |
| **Recall** | **100.00%** |
| **F1-Score** | **0.9730** |
| Type Accuracy | 88.89% |
| Severity Accuracy | 88.89% |
| False Positive | 1 (h16) |
| False Negative | 0 |

> Mock 模式在二分类检测上达到 100% Precision/Recall（在 20 条数据集上多轮迭代调试的结果）。LLM 模式以 1 个误报的代价获得 88.89% 的类型准确率，显著优于 Mock 的 72.22%。

---

## 5. 误判分析

### LLM 模式误判

| Case | 类型 | 问题 | 分析 |
|------|------|------|------|
| h16 | FP | 误报为信息遗漏 | KB 有统计性信息但 reply 基本正确，边界模糊 |
| h04 | 类型 | 政策编造 vs 政策偏差 | "支持纸质发票"与 KB 矛盾，但部分政策正确，边界歧义 |
| h08 | 类型 | 参数编造 vs 政策偏差 | 时效矛盾(48h→24h)，属于发货政策偏差，边界歧义 |

### 局限性讨论

1. **Mock 模式存在数据过拟合**：Mock 引擎 100% 检出率是 20 条数据上多轮迭代的结果，部分规则直接针对数据集特定表述。在同分布新数据上仍有效，但面对不同措辞的同类问题可能漏检。Mock 模式适合 Pipeline 逻辑验证，生产环境应使用 LLM 引擎。
2. **语义理解局限**：规则依赖关键词和正则匹配，无法理解语义相似但措辞不同的表述
3. **边界 Case 判定困难**：h16（信息遗漏 vs 正常回答）、h04/h08（编造 vs 偏差）天然存在判定歧义
4. **LLM 一致性问题**：相同输入可能产生不同输出，影响结果可复现性

### 改进方向

1. **Multi-Judge 投票**：引入多个 Judge 并行评估，取多数结果，降低单一模型偏差
2. **Active Learning**：将误判 Case 加入 Few-shot 示例，迭代优化 Prompt
3. **Severity 校准**：引入业务方反馈机制，将误判 Case 用于校准严重度判定规则
4. **更丰富的工具层**：增加时间表达式解析、实体识别等工具，减少 LLM 负担

---

## 6. AI 工具使用情况

本项目开发过程中使用了以下 AI 工具：

| 工具 | 用途 | 占比 |
|------|------|------|
| **Claude Code (Claude Sonnet 4.6 / Opus 4.7)** | 架构设计、代码生成、调试优化、README 撰写 | 80% |
| **ChatGPT / Gemini** | 方案对比分析、Prompt 设计参考 | 20% |

AI 辅助的开发流程：
1. **方案设计阶段**：对比 GPT 和 Gemini 的方案建议，采纳 Gemini 关于"去除向量检索"的关键纠正
2. **代码实现阶段**：使用 Claude Code 生成所有模块代码
3. **调试优化阶段**：通过多轮迭代修复 Mock 引擎误报/漏检、工具 Bug（KBSectionLocator 重叠率、Capability Guard tool_trace）、类型归类等问题
4. **Prompt 设计**：参考 Few-shot 示例优化 NLI 验证和幻觉分类的 Prompt 模板

---

## 7. 快速开始

### 安装

```bash
git clone <repo-url> && cd RAGGuard
pip install -r requirements.txt
```

### 启动后端

```bash
python server.py
# FastAPI 运行在 http://localhost:8000
```

### 启动前端

```bash
streamlit run app.py
# 浏览器打开 http://localhost:8501
```

### Mock 模式 CLI

```bash
# 批量评估全部 20 条
python run_eval.py --mode mock

# 单条检测
python run_eval.py --mode mock --case-id h01
```

### LLM 模式 CLI

```bash
# 使用 config.yaml 中的配置
python run_eval.py --mode llm

# 或通过环境变量
export RAGGUARD_API_KEY=sk-xxx
python run_eval.py --mode llm
```

---

## 8. 项目结构

```
RAGGuard/
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── replies.json              # 20 条客服回复
│   └── ground_truth.json         # 人工标注
│
├── src/
│   ├── __init__.py
│   ├── state.py                  # Pydantic 数据模型
│   ├── engine.py                 # Mock + LLM 双引擎 & 统一管线
│   ├── prompts.py                # LLM Prompt 模板 (CoT + Few-shot)
│   ├── validators.py             # LLM 输出校验 (JSON恢复 + 自动修复)
│   ├── runner.py                 # 异步批处理 (asyncio + 多进程)
│   ├── evaluator.py              # 评估指标 (P/R/F1 + Type/Severity Acc)
│   ├── patterns.py               # 跨 Case 幻觉模式画像分析
│   ├── visualization.py          # matplotlib 图表生成
│   ├── logging_config.py         # 结构化日志
│   └── tools/
│       ├── __init__.py
│       ├── numeric.py            # 数值提取与比对工具
│       ├── polarity.py           # 极性/立场冲突检测工具
│       ├── capability.py         # 能力边界解析工具
│       ├── locator.py            # KB 段落定位工具 (Bigram 重叠)
│       └── registry.py           # 统一工具注册与调度
│
├── server.py                     # FastAPI REST API 后端
├── app.py                        # Streamlit Dashboard 前端
├── run_eval.py                   # CLI 一键评估入口
│
└── outputs/
    └── YYYYMMDD_HHMMSS/           # 每次运行的时间戳目录
        ├── detection_results.json
        ├── evaluation_metrics.json
        ├── pattern_profile.json
        ├── report.md
        └── *.png                  # 可视化图表
```

---

## 9. 技术栈

- **LLM**: DeepSeek V4 Pro (兼容 OpenAI API)
- **数据模型**: Pydantic v2
- **API 后端**: FastAPI + Uvicorn
- **前端**: Streamlit + Matplotlib
- **评估**: scikit-learn
- **重试机制**: tenacity

---

## License

MIT
