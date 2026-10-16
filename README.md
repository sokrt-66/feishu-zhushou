# 飞书生态智能助手

基于 **LangGraph** 构建的企业级智能助手，深度集成飞书生态，提供多工具协同、持久化对话状态、权限管控等能力。

完整的架构设计与模块说明见 [飞书生态智能助手.md](飞书生态智能助手.md)。

## 核心能力

- 意图识别：关键词评分机制自动识别用户意图，并路由到对应工具子图
- SQL 查询：自然语言转 SQL，支持部门级数据隔离
- 知识库检索：基于 FAISS 向量库的 RAG 问答（`knowledge_base/`）
- 联网搜索：Tavily / Bing 实时检索
- 飞书集成：消息、会议、任务、文档、日历等开放平台能力
- 状态持久化：LangGraph Checkpointer 保持多轮对话上下文
- 权限管控：角色与工具级权限校验
- 实时通信：FastAPI + WebSocket 服务

## 目录结构

```
.
├── main.py                  # FastAPI 入口，注册路由 / WebSocket / 静态页面
├── config.py                # 配置加载（读取 .env）
├── agent.py                 # LangGraph 主图
├── tool_subgraphs.py        # 各工具子图实现
├── feishu.py                # 飞书开放平台 API 封装
├── feishu_subgraph.py       # 飞书相关子图
├── document_parser.py       # 文档解析
├── document_subgraph.py     # 文档问答子图
├── meeting_agent.py         # 会议相关 Agent
├── data_analysis_agent.py   # 数据分析 Agent
├── automation_agent.py      # 自动化任务 Agent
├── database.py              # 数据库访问层
├── vector_store.py          # 向量库封装（FAISS）
├── checkpoint.py            # 状态持久化
├── permissions.py           # 权限校验
├── websocket_service.py     # WebSocket 服务
├── static/index.html        # 前端页面
├── knowledge_base/          # 知识库文档
├── test/                    # pytest 测试
└── requirements.txt
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

copy .env.example .env         # 然后按实际情况填写配置
python main.py
```

服务默认监听 `0.0.0.0:5001`（可通过 `.env` 中的 `SERVER_PORT` 修改），浏览器访问 `http://localhost:5001/` 打开前端页面。

## 配置说明

所有配置项都在 `.env` 中，模板见 [.env.example](.env.example)：

- 阿里云百炼 DashScope 的 API Key 与模型名（对话、推理、Embedding）
- 飞书应用 `APP_ID` / `APP_SECRET` / 事件校验 Token
- MySQL 连接信息、服务端口
- Tavily / Bing 搜索 API Key

> `.env` 已被 `.gitignore` 忽略，请勿提交真实密钥。

## 测试

```bash
pytest
```
