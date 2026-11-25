# ResearchAgent

基于 **LangGraph 有环状态机** 与 **混合 RAG（Hybrid RAG）** 的自主科研文献调研与综述生成智能体系统。

> 本地核心包名与命令行入口为 `reacher-agent` / `reacher`，项目已严格实现 `core/`（核心引擎）、`api/`（服务层）与 `frontend/`（前端工作台）三层解耦。

---

## 一、本项目有什么用？

传统大模型直接撰写文献综述容易出现**虚构参考文献（幻觉）**、**单轮检索视角片面**、**流水账式堆砌而缺乏横向对比**等问题。**ResearchAgent** 将科研文献调研建模为可观察、可人工干预（HITL）、可自我反思修正的多阶段工作流，适用于课程科研调研、开题文献梳理、方向追踪以及本地 PDF 与在线文献的联合对比分析。

### 核心功能与特点

1. **研究计划拆解与人机协同（HITL）**
   - 输入中文或英文研究主题后，自动拆解为多个子问题（`SQ1..SQn`）及对应的英文学术检索词。
   - 基于 `SqliteSaver` 支持在规划完成后暂停，用户可在 Web 界面或终端中增删子问题、修改检索词后再恢复执行。
2. **多学术源并发检索 + 本地 PDF 混合入库**
   - 支持 **arXiv**、**OpenAlex** 与可选的 **Semantic Scholar** 多源并行检索，自动去重并补全引用数、期刊/会议与开放获取 PDF 链接。
   - 支持拖拽/批量上传本地 PDF 文献，与在线检索文献混合入库并优先保留。
3. **全文解析与混合检索（Dense + BM25 + RRF）**
   - 使用 PyMuPDF（`pymupdf`）解析 PDF 全文并自动截断尾部 `References` 列表防止分块膨胀。
   - 结合 **ChromaDB 稠密向量检索** 与 **BM25 关键词检索**，通过 **倒数排序融合（RRF）** 与单篇文献最大分块上限（`max_per_paper`）约束，抽取跨文献结构化证据矩阵。
4. **动态覆盖度评估与反思闭环**
   - `evaluator` 节点评估每个子问题的证据覆盖度；当存在薄弱子问题时，由 `query_rewriter` 生成补充检索词再次检索，并内置最大轮次上限与边际收益熔断机制防止死循环。
5. **双层引用核验与定点改写（Verifier-Guided Revision）**
   - 合成带 `[P01]`、`[P02]` 等内联引用的 Markdown 综述后，由 `Citation Verifier` 执行引用编号真实率检查与原文 Chunk 语义支撑度核验，并可触发 `reviser` 节点对问题语句进行定点修正与复检。
6. **现代化科研工作台与原文溯源**
   - Web 前端支持实时 SSE 节点轨迹时间线、**点击正文 `[Pxx]` 内联引用弹出右侧原文分块与页码溯源抽屉**、一键导出 `.md` 综述与 `.bib` 参考文献，以及在线修改模型与接口配置。

---

## 二、目录结构

```text
ResearchAgent/
├── core/                  # 核心引擎层（纯 Python，零 UI 依赖）
│   ├── cli.py             # Typer CLI 命令行入口
│   ├── config.py          # Pydantic Settings 配置加载
│   ├── schemas.py         # 状态图与证据/核验数据结构
│   ├── graph/             # LangGraph 工作流、节点实现与 Prompt
│   ├── llm/               # OpenAI 兼容 LLM / Embedding 适配与离线 Mock
│   ├── retrieval/         # arXiv / OpenAlex / Semantic Scholar / PDF 解析 / Hybrid RAG
│   └── verifier/          # 引用真实率与语义支撑度核验器
├── api/                   # FastAPI 服务层（REST + SSE + 运行时设置 + 静态前端托管）
├── frontend/              # React + TypeScript + Vite + Tailwind CSS v4 中文工作台
├── ui/                    # 备用轻量 Streamlit 客户端 (ui/app.py)
├── eval/                  # 自动化基线对比与消融实验框架
├── scripts/               # 案例复现与重核验辅助脚本
├── tests/                 # pytest 自动化测试套件
├── .env.example           # 环境变量配置模板
├── pyproject.toml         # Python 依赖与项目元数据
└── start_*.ps1 / reacher.ps1  # Windows PowerShell 快捷启动脚本
```

---

## 三、环境准备与配置

### 1. 基础环境
- **Python**：`>= 3.11, < 3.14`（推荐使用 [`uv`](https://docs.astral.sh/uv/) 管理环境）
- **Node.js**：`>= 18`（仅在需要开发或重新构建 `frontend/` 时需要）

### 2. 安装依赖

```powershell
# 1. 安装 Python 依赖
uv sync

# 2. 安装并构建前端静态资源（供 FastAPI 直接托管）
cd frontend
npm install
npm run build
cd ..
```

### 3. 配置模型与检索环境变量

将 [.env.example](.env.example) 复制为 `.env`，并填入你自己的 OpenAI 兼容接口密钥：

```powershell
Copy-Item .env.example .env
```

常见配置项说明：
- `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`：兼容 OpenAI Chat Completions 接口的大模型服务（如 DeepSeek、OpenAI 等）。
- `EMBEDDING_PROVIDER`：
  - `openai`：使用外部 Embedding 接口（需配置 `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`，例如 SiliconFlow 的 `BAAI/bge-m3`）；
  - `local_hash`：纯本地确定性哈希向量（无需外部 Embedding API，适合离线测试）。
- `ACADEMIC_SOURCES`：默认 `arxiv,openalex`，可选加入 `semantic_scholar`。
- `OPENALEX_EMAIL` / `SEMANTIC_SCHOLAR_API_KEY`：可选，用于提升对应学术接口速率限制。

> 启动 Web 工作台后，也可以直接在页面左下角**“设置”**弹窗中动态修改模型、Base URL 与 API Key（保存于本地 `data/runtime_settings.json`，已被 `.gitignore` 忽略）。

---

## 四、使用方式

### 方式 1：Web 科研工作台（推荐）

#### 一体化启动（后端 + 已构建前端）
```powershell
.\start_api.ps1
# 或手动执行：
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```
启动后在浏览器访问 `http://127.0.0.1:8000` 即可使用完整中文科研工作台。

#### 前后端分离热重载开发模式
在两个终端中分别运行：
```powershell
# 终端 1：启动 FastAPI 后端 (127.0.0.1:8000)
.\start_api.ps1

# 终端 2：启动 Vite 前端开发服务器 (127.0.0.1:5173，自动代理 /api)
.\start_frontend.ps1
```

#### 备用轻量 Streamlit 界面
```powershell
.\start_ui.ps1
# 或：uv run streamlit run ui/app.py --server.port 8501
```

---

### 方式 2：命令行（CLI）

```powershell
# 查看当前生效的非敏感配置
uv run python -m core.cli check-config

# 离线 Mock 模式快速验证完整工作流（不消耗 API 额度）
uv run python -m core.cli run "Agentic RAG for Scientific Fact Verification" --mock --no-pdf-download

# 真实在线调研（开启 HITL 计划确认 + 指定本地 PDF 目录）
uv run python -m core.cli run "Multi-Agent Literature Review Systems" --top-k 8 --max-iterations 2 --hitl --local-pdfs path/to/pdfs
```

运行完成后，综述报告（`report.md`）与完整状态（`state.json`）会保存在 `outputs/<run_id>/` 目录下。

---

### 方式 3：运行自动化测试与评测框架

```powershell
# 运行全量自动化测试
uv run pytest -v

# 运行评测框架（离线 Mock 干跑单主题 T01）
uv run python -m eval.runner --topics T01 --mock --no-pdf-download

# 运行全量 10 个基准主题评测（需配置有效 API Key）
uv run python -m eval.runner --all --resume
```
