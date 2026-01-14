# RAG 系统评估项目 - Week 4

## 🎯 项目简介

本项目是一个模块化的 RAG（检索增强生成）系统，采用模块化设计，每个模块负责 RAG 系统的不同方面。

## 📁 项目结构

```
rag-eva-week4/
├── app.py              # 【展示层】Streamlit 主程序，负责成果展示与交互
├── rag_engine.py       # 【逻辑层】封装 Module 01-08，核心 RAG 流水线实现
├── evaluator.py        # 【评估层】封装 Module 09，基于 JSON 的自动评分逻辑
├── requirements.txt    # 【依赖管理】项目运行所需的库列表
│
├── /data
│   └── q2sql_pairs.json    # 【数据集】作为基础知识库和评估的 Ground Truth
│
├── /modules            # 【可选】若逻辑复杂，可将 05-07 等高级模块独立拆分
│   ├── retrieval_opt.py    # 对应 Module 05 & 07（查询扩展与重排序）
│   └── indexing_opt.py     # 对应 Module 06（索引构建与优化）
│
├── /vector_db          # 【存储层】Module 04 生成的本地向量数据库文件夹
│
└── README.md           # 【说明文档】项目背景、模块对应关系及运行指南
```

## 📚 模块对应关系

| 模块 | 功能 | 技术栈 | 依赖 |
|------|------|--------|------|
| 00-简单RAG-SimpleRAG | 基础 RAG 系统实现 | LangChain/LlamaIndex | 基础环境 |
| 01-数据导入-DataLoading | 数据加载和预处理 | pandas, PyPDF2 | 文档解析库 |
| 02-文本切块-DocChunking | 文档分块策略 | LangChain Splitters | NLP 工具 |
| 03-向量嵌入-Embedding | 文本向量化 | HuggingFace, BGE | GPU 支持(可选) |
| 04-向量存储-VectorDB | 向量数据库操作 | Milvus, Chroma | 向量数据库 |
| 05-检索前处理-PreRetrieval | 检索优化 | Query Expansion | NLP 工具 |
| 06-索引优化-Indexing | 索引构建和优化 | 层次索引, 关键词索引 | 搜索引擎 |
| 07-检索后处理-PostRetrieval | 检索结果优化 | 重排序, 过滤 | ML 模型 |
| 08-响应生成-Generation | 答案生成 | LLM 集成 | GPU 推荐 |
| 09-系统评估-Evaluation | 系统性能评估 | RAGAS, TruLens | 评估框架 |
| 10-高级RAG-AdvanceRAG | 高级 RAG 技术实现 | Graph RAG, Multi-Agent | 高级框架 |

## 🚀 快速开始

### 步骤 1: 环境准备

确保已安装 Python 3.9+ 环境，推荐使用虚拟环境：

```bash

# 激活虚拟环境使用UV进行包管理
uv venv

### 步骤 2: 安装依赖

```bash
# 安装项目依赖
uv pip install -r requirements.txt
```

> ⚠️ **注意**: 首次安装可能需要较长时间，特别是 PyTorch 和 Sentence Transformers

### 步骤 3: 启动应用

```bash
# 运行 Streamlit 应用
uv run streamlit run app.py
```

### 步骤 4: 访问界面

应用启动后，在浏览器中打开：

- **本地访问**: `http://localhost:8501`
- **网络访问**: `http://<your-ip>:8501`

### 步骤 5: 使用系统

1. **查询功能**: 在"🔍 查询"标签页输入自然语言问题，点击"搜索"获取相关 SQL
2. **评估功能**: 在"📊 评估"标签页点击"运行评估"，系统将自动评估 RAG 性能
3. **数据管理**: 在"📂 数据"标签页查看和管理测试数据集

### 步骤 6: 配置 DeepSeek LLM

本项目使用 **DeepSeek** 作为 LLM。复制 `.env.example` 创建 `.env` 文件并填入 API Key：

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
DEEPSEEK_API_KEY=your-deepseek-api-key-here
```

> 💡 **获取 API Key**: 访问 [DeepSeek 开放平台](https://platform.deepseek.com/) 注册并获取 API Key

如果不配置 API Key，系统将使用模板生成模式（基于检索结果直接返回）。

## 🔧 功能说明

### 查询功能
- 输入自然语言问题
- 系统自动检索最相关的 SQL 示例
- 支持查询扩展和结果重排序

### 评估功能
- 自动评估 RAG 系统性能
- 计算精确匹配率、相似度等指标
- 生成详细评估报告

### 数据管理
- 查看和管理测试数据集
- 支持 JSON 格式的 Q2SQL 数据

---

## 🔬 核心模块实现详解

### Module 07: 检索后处理 (PostRetrieval)

**文件**: `modules/retrieval_opt.py`

#### ResultReranker - 结果重排序器

支持三种重排序策略：

| 方法 | 说明 | 实现细节 |
|------|------|----------|
| `similarity` | 基于相似度排序 | 使用向量检索返回的原始分数排序 |
| `cross_encoder` | Cross-Encoder 重排序 | 使用 `ms-marco-MiniLM-L-6-v2` 模型计算 query-doc 相关性 |
| `hybrid` | 混合重排序 | 综合原始分数(50%) + 长度分数(20%) + 关键词匹配(30%) |

```python
# 混合重排序公式
hybrid_score = 0.5 * original_score + 0.2 * length_score + 0.3 * keyword_score
```

#### ResultFilter - 结果过滤器

- **分数过滤**: 过滤低于阈值的结果
- **长度过滤**: 过滤过长/过短的内容
- **去重**: 基于 Jaccard 相似度去除重复文档

---

### Module 08: 响应生成 (Generation)

**文件**: `rag_engine.py` - `AnswerGenerator` 类

#### 双模式生成

| 模式 | 触发条件 | 说明 |
|------|----------|------|
| **LLM 模式** | 配置了 `DEEPSEEK_API_KEY` | 调用 DeepSeek API 智能生成 SQL |
| **模板模式** | 未配置 API Key | 直接返回最相似的检索结果 |

#### LLM 生成流程

```
1. 构建上下文 (取 Top 3 检索结果)
   ↓
2. 构建 Prompt (包含参考示例 + 用户问题)
   ↓
3. 调用 DeepSeek API (deepseek-chat 模型)
   ↓
4. 返回生成的 SQL + 解释
```

#### Prompt 模板

```
你是一个SQL专家。根据以下参考示例，为用户的问题生成合适的SQL查询。

参考示例:
{context}  # Top 3 检索结果

用户问题: {query}

请生成SQL查询并解释你的思路。
```

---

### Module 09: 系统评估 (Evaluation)

**文件**: `evaluator.py`

#### SQLEvaluator - SQL 评估器

| 评估指标 | 计算方法 | 权重 |
|----------|----------|------|
| **Exact Match** | SQL 标准化后完全匹配 | 30% |
| **Similarity** | SequenceMatcher 字符串相似度 | 30% |
| **Context Relevance** | 问题与上下文词汇重叠度 | 20% |
| **Faithfulness** | 预测与真实 SQL 词汇重叠度 | 20% |

```python
# 整体得分计算公式
overall_score = 0.3 * exact_match + 0.3 * similarity + 0.2 * context_relevance + 0.2 * faithfulness
```

#### RAGEvaluator - 端到端评估器

**检索质量评估**:
- **Precision**: 检索结果中相关文档的比例
- **Recall**: 相关文档被检索到的比例
- **MRR** (Mean Reciprocal Rank): 第一个相关结果的排名倒数

**端到端评估流程**:
```
1. 加载测试数据 (q2sql_pairs.json)
   ↓
2. 对每个问题执行 RAG 查询
   ↓
3. 从答案中提取 SQL
   ↓
4. 与 Ground Truth 对比评分
   ↓
5. 生成评估报告
```

## 📊 评估指标

- **精确匹配率 (Exact Match)**: 生成的 SQL 与标准答案完全一致的比例
- **相似度得分 (Similarity)**: 基于字符串相似度的评分
- **上下文相关性 (Context Relevance)**: 检索内容与问题的相关程度
- **答案忠实度 (Faithfulness)**: 生成答案对检索内容的忠实程度
- **整体得分 (Overall Score)**: 综合以上指标的加权平均

## 🛠️ 技术栈

- **前端**: Streamlit
- **向量数据库**: ChromaDB
- **Embedding**: Sentence Transformers (BGE)
- **LLM**: DeepSeek (deepseek-chat)
- **文档处理**: LangChain
- **评估**: 自定义评估框架

## 📝 注意事项

1. 首次运行会自动下载 Embedding 模型，需要网络连接
2. 向量数据库会保存在 `vector_db/` 目录下
3. 如需使用 LLM 生成功能，需配置 `DEEPSEEK_API_KEY` 环境变量
4. DeepSeek API 文档: https://platform.deepseek.com/api-docs

## 📄 License

MIT License
