# 飞书生态智能助手

基于 **LangGraph** 编排的企业级智能助手，深度集成飞书生态：在飞书里用自然语言完成「数据查询 / 知识库问答 / 联网检索 / 任务与文档自动化」，并内置部门级权限隔离与工具超时兜底。

- 完整架构设计与模块说明见 [飞书生态智能助手.md](飞书生态智能助手.md)
- 规模：19 个 Python 模块（约 8.6k 行）、29 个 FastAPI 路由、前端单页 547 行、5 个 pytest 用例文件

## 核心能力

| 能力 | 说明 | 实现要点 |
|------|------|----------|
| 意图路由 | 自动识别用户意图并分发到对应工具子图 | 关键词评分（web / sql / rag 三组词表打分取最大） |
| 自然语言转 SQL | LLM 生成 SQL → 校验 → 执行 → 格式化返回 | `sqlparse` AST 语句白名单、强制 `LIMIT`、部门级数据过滤、敏感字段脱敏 |
| 知识库 RAG | 多格式文档解析 → 分块 → 向量入库 → 检索 | FAISS（Flat / IVF / HNSW）、向量 + 关键词加权融合、启发式重排序、低相关度自动降级联网检索 |
| 联网搜索 | 实时信息检索 | Tavily / Bing Search API |
| 飞书集成 | 消息、任务、云文档、多维表格、审批、通讯录、会议与录制转写 | 飞书开放平台 API + 事件回调签名校验 |
| 权限管控 | 部门级知识库隔离与角色校验 | `permissions.py`（部门管理员 / 知识库归属 / RAG 权限守卫） |
| 状态与容错 | 多轮上下文、超时兜底、失败重试 | 文件式会话检查点（JSON）+ `timeout_fallback.py` 超时/重试/降级 |
| 实时通信 | SSE 流式输出、WebSocket 会话管理 | `StreamingResponse` + `websocket_service.py` |

## 技术栈

- **Web**：FastAPI 0.110 + Uvicorn、Pydantic v2
- **编排**：LangChain + LangGraph（`StateGraph` / 条件边 / `TypedDict` 状态）
- **模型**：阿里云百炼 DashScope（OpenAI 兼容模式），`qwen3.6-flash` / `qwen3.6-plus`，嵌入 `text-embedding-v4`
- **检索**：FAISS（Flat / IVF / HNSW）、jieba 分词、加权融合 + 启发式重排序
- **文档解析**：pdfminer / PyMuPDF / python-docx / openpyxl
- **数据库**：MySQL（PyMySQL / aiomysql），并支持 SQLite（aiosqlite）与 PostgreSQL（asyncpg）
- **安全**：JWT（python-jose，HS256）、`fastapi-limiter` + Redis 限流、`sqlparse` SQL 安全校验、敏感字段脱敏
- **测试**：pytest / pytest-asyncio / pytest-mock

## 目录结构

```
.
├── main.py                  # FastAPI 入口：路由 / 鉴权 / SSE / 静态页面
├── config.py                # 配置加载（读取 .env）
├── feishu.py                # 飞书开放平台 API 封装 + 回调验签
├── feishu_subgraph.py       # 飞书事件处理子图 + 后台任务队列
├── tool_subgraphs.py        # SQL / RAG / Web 三个子图与 6 个工具封装
├── agent.py                 # 会话检查点与消息历史修剪
├── checkpoint.py            # 文件式检查点（JSON 落盘、过期清理、版本迁移）
├── document_parser.py       # 多格式文档解析器
├── document_subgraph.py     # 文档摄取子图（解析 → 分块 → 嵌入 → FAISS 入库）
├── vector_store.py          # FAISS 向量库 + 混合检索 + 重排序
├── permissions.py           # 部门级权限与知识库隔离
├── meeting_agent.py         # 会议纪要
├── data_analysis_agent.py   # 数据分析
├── automation_agent.py      # 自动化工作流
├── database.py              # 数据访问层
├── websocket_service.py     # WebSocket 会话与消息分发
├── timeout_fallback.py      # 工具超时 / 重试 / 降级
├── trace_id.py              # 全链路 TraceID
├── static/index.html        # 前端页面
├── knowledge_base/          # 知识库文档（示例语料）
└── test/                    # pytest 用例
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

copy .env.example .env            # 按实际情况填写
python main.py
```

服务监听 `SERVER_PORT`（代码默认 5000，`.env.example` 示例为 5001），浏览器打开 `http://localhost:<port>/` 使用前端页面，`/docs` 为接口文档。

## 配置说明

配置全部来自 `.env`，模板见 [.env.example](.env.example)：

- DashScope / OpenAI 兼容接口的 API Key 与模型名（对话、推理、嵌入）
- 飞书自建应用 `APP_ID` / `APP_SECRET` / `VERIFICATION_TOKEN` / `ENCRYPT_KEY`
- 数据库连接（MySQL）、服务端口、JWT 密钥
- Tavily / Bing 搜索 API Key

> `.env` 已在 `.gitignore` 中忽略，请勿提交真实密钥。

## 测试

```bash
pytest
```

## 已知边界（后续改进清单）

说明：以下为本仓库当前的实现范围，避免过度描述。

- **意图识别**为关键词评分，暂不支持同义改写与复合意图；计划改为小模型分类或 function calling 打分 + 置信度阈值。
- **重排序**为启发式规则（命中位置 / 文档长度 / 词覆盖率），未接入 cross-encoder 模型。
- **RAG 默认链路**使用 `knowledge_base/` 示例语料；接入真实语料需注入真实嵌入模型并重建 FAISS 索引（`document_parser.py` / `vector_store.py` 已提供可复用组件）。
- **文档上传**在请求内同步完成解析与入库，大文件建议改为后台任务队列。
- **飞书回调**当前实现了签名校验，尚未实现 `url_verification`（challenge）应答与 `encrypt_key` 解密，事件也未做 `event_id` 幂等去重。
- **登录口令**当前为明文比较（源码已标注），生产环境应改为哈希存储（werkzeug / bcrypt / Argon2）。
- 未提供 Dockerfile，部署方式为直接运行 Python 服务。
