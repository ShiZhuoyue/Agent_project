# DeepResearch-LangGraph 智能科研Agent

## 项目简介
本项目是**基于 LangGraph 构建的全自动科研文献研究智能 Agent**，专为学术调研、文献综述、论文检索、前沿资讯整理设计。

系统支持**两种用户输入模式自动适配**：
1. **精准输入（带论文标题）** → 直接精准定位、单篇精读总结
2. **模糊需求（无标题科研提问）** → 自动关键词扩搜、多源检索、筛选、复盘迭代补全

具备完整的：**规划 → 执行 → 观测复盘 → 答案合成 → 记忆持久化** 闭环工作流，支持前端流式输出、多轮对话记忆、自动重试重规划，纯CPU可部署，无需GPU。

## 系统整体架构
整体采用 **LangGraph 状态机多节点分层架构**，完全解耦规划、执行、工具、校验、记忆模块。

### 架构分层
1. **前端交互层（Streamlit UI）**
- 可视化对话界面
- 支持打字机流式输出
- 对接后端 FastAPI 流式接口

2. **API服务层（FastAPI）**
- 请求统一预检、参数校验
- SSE 流式响应推送
- 启动并调度 LangGraph Agent 工作流

3. **LangGraph 核心节点（核心业务逻辑）**
- **Memory Router 记忆路由节点**
  读取 Chroma 历史对话向量记忆，为本轮提问提供上下文，路由进入规划节点。
- **Planner 规划节点（核心）**
  根据用户输入**智能生成两种执行计划**：
  - 有明确论文标题：**单步精准查询计划**
  - 模糊科研需求：**三段式检索计划（泛检索→筛选→总结）**
- **Executor 执行节点**
  解析 Planner 计划，分步调度工具执行任务。
- **Tool Node 工具节点**
  统一封装所有学术检索与文献处理工具。
- **Observer 观测复盘节点（自动纠错核心）**
  判断检索文献是否充足、匹配度是否达标。
  - 信息不足 → **自动回流 Planner 重规划、换关键词扩搜**
  - 信息充足 → 进入答案合成
- **Synthesizer 答案合成节点**
  整合多源文献、多轮检索结果，输出结构化科研综述。
- **Memory Writer 记忆写入节点**
  将问答、文献信息向量化存入向量库 + 结构化存入SQL，实现长期记忆复用。

4. **工具能力层**
- 多源学术并行检索（arXiv / Semantic Scholar / OpenAlex / Crossref）
- 混合RAG语义筛选
- 批量文献摘要精读与结构化总结

5. **RAG引擎层**
- 嵌入模型：BGE-M3
- 语义分块 + 关键词/向量混合检索 + 重排
- 向量数据库：ChromaDB

6. **持久化存储层**
- ChromaDB：对话记忆、文献片段向量存储
- SQLlite：结构化对话日志存储

## 核心工作流（完整运行逻辑）
### 场景一：用户输入【明确论文标题】
1. Memory Router 加载历史记忆
2. Planner 判定为**精准查询场景**，生成**单步执行计划**
3. Executor 调用工具直接标题匹配检索
4. 一次性获取论文信息，无需迭代
5. Synthesizer 生成解读
6. Memory Writer 持久化本轮对话

### 场景二：用户输入【模糊科研需求、无标题】（项目最大亮点）
1. Memory Router 召回历史上下文
2. Planner 生成**标准三段式科研计划**：
   - 第一步：多源学术关键词泛检索，批量捞取候选论文
   - 第二步：RAG 语义过滤、重排、筛选高相关文献
   - 第三步：批量摘要精读、结构化整理
3. Executor 驱动工具批量检索
4. Observer 校验结果质量
   - 文献少/不相关 → **回流 Planner 重新改写关键词、二次扩搜**
5. 信息充足后合成综述答案
6. 落地记忆持久化

> 本架构独有：**模糊提问全自动迭代检索 + 自我纠错重规划能力**

## 项目目录结构
```
DeepResearch/
├── streamlit_ui.py              # 前端流式UI界面
├── main.py                      # FastAPI 服务入口
├── research_pipeline.py         # Agent 总调度、请求预检
├── agent/
│   ├── agent.py                 # LangGraph 状态图定义
│   └── nodes/                   # 七大核心节点源码
│       ├── memory_router.py
│       ├── planner.py
│       ├── executor.py
│       ├── tool_node.py
│       ├── observer.py
│       ├── synthesizer.py
│       └── memory_writer.py
├── tools/                       # 所有检索&文献工具
├── rag/                         # RAG检索、分块、重排逻辑
├── database/                    # 向量库 & SQL存储封装
├── .env                         # 环境变量配置
├── requirements.txt              # 依赖列表
└── README.md
```

## 环境依赖安装
### 环境要求
- Python >= 3.10
- 纯CPU运行，无需GPU

### 依赖安装
```bash
pip install -r requirements.txt
```

### 环境变量 .env 配置
```env
# LLM
OPENAI_API_KEY=xxx
OPENAI_BASE_URL=xxx

# 服务端口
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501

# 模型与存储
EMBEDDING_MODEL=BAAI/bge-m3
CHROMA_DB_PATH=./chroma_storage
SQL_DB_PATH=./conversation.db
```

## 项目启动方式
### 1. 启动后端 FastAPI
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. 启动前端界面
```bash
streamlit run streamlit_ui.py
```

访问：`http://localhost:8501`

## 核心功能亮点
1. **双模式智能自适应规划**
自动区分「精准标题查询」和「模糊科研调研」，生成不同执行链路。

2. **AI 自主复盘迭代机制**
Observer 自动检测资料不足，**主动回流重搜、优化关键词**，模拟人工科研试错过程。

3. **四大学术源并行检索**
arXiv、Semantic Scholar、OpenAlex、Crossref 同时拉取，覆盖最新顶会/预印本。

4. **BGE-M3 混合 RAG 精准筛选**
过滤噪声文献，只保留与用户问题高度相关内容。

5. **长期对话记忆能力**
自动记忆历史提问与调研结果，多轮对话上下文连贯，避免重复检索。

6. **全流式输出体验**
大段科研总结逐字输出，交互流畅。

## 使用示例
### 精准模式（有标题）
```
帮我解读论文《XXX Survey 2026》
```

### 模糊调研模式（无标题，核心使用场景）
```
帮我梳理2026年轻量化RAG优化的最新研究进展
```
系统自动：扩搜 → 筛选 → 精读 → 复盘补全 → 输出综述

## 常见问题
1. **检索结果太少**
Agent 会**自动重规划、替换关键词再次检索**，无需人工干预。

2. **历史对话消失**
请勿删除 `chroma_storage` 向量库文件夹。

3. **运行卡顿**
可在代码内限制单次检索论文数量、调小分块尺寸，降低CPU计算压力。

## 许可证
MIT License
仅供学术科研学习使用，可自由二次修改与部署。
