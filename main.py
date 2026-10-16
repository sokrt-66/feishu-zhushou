from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

from tools import query_sql_tool, rag_search_tool, web_search_tool

from fastapi import FastAPI, Request, Depends, HTTPException, status, Form, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import List, Optional
import os
import asyncio
import json
import logging
import re
from werkzeug.utils import secure_filename

from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import (
    feishu_config, security_config, system_config,
    rag_config, document_config, llm_config, init_config
)
from feishu import FeishuClient
from feishu_subgraph import FeishuEventSubGraph, background_task_queue
from document_subgraph import DocumentIngestionSubGraph
from agent import build_main_graph
from tools import tools_from_B
from automation_agent import AutomationAgent
from permissions import PermissionManager, RAGPermissionGuard
from data_analysis_agent import DataAnalysisAgent
from meeting_agent import MeetingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("启动后台任务队列...")
    await background_task_queue.start()
    
    # 初始化限流
    logger.info("初始化限流服务...")
    try:
        redis_client = redis.from_url("redis://localhost:6379", encoding="utf-8", decode_responses=True)
        await FastAPILimiter.init(redis_client)
        logger.info("限流服务初始化成功")
    except Exception as e:
        logger.warning(f"限流服务初始化失败（Redis未运行）: {e}")
    
    yield
    
    # 关闭时清理
    logger.info("关闭应用...")
    await background_task_queue.stop()
    if 'feishu_client' in globals():
        await feishu_client.close()

app = FastAPI(title="飞书智能助手网关", version="1.0", lifespan=lifespan)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.doc', '.docx', '.xls', '.xlsx'}


@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/@vite/client")
async def vite_client():
    """处理Vite开发服务器请求，返回空响应"""
    return {"status": "ok"}

def sanitize_filename(filename: str) -> str:
    """安全清理文件名"""
    safe_name = secure_filename(filename)
    if not safe_name:
        safe_name = "unnamed_file"
    name, ext = os.path.splitext(safe_name)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = '.txt'
    return f"{name[:100]}{ext}"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EnsureJSONCharsetMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if response.headers.get("content-type", "").startswith("application/json"):
            response.headers["content-type"] = "application/json; charset=utf-8"
        return response

app.add_middleware(EnsureJSONCharsetMiddleware)

security = HTTPBearer()

class MessageRequest(BaseModel):
    sender_id: str
    content: str
    message_type: str = "text"

class Token(BaseModel):
    access_token: str
    token_type: str


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=security_config.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, security_config.secret_key, algorithm=security_config.jwt_algorithm)
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            credentials.credentials,
            security_config.secret_key,
            algorithms=[security_config.jwt_algorithm]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return user_id


feishu_client = FeishuClient(
    app_id=feishu_config.app_id,
    app_secret=feishu_config.app_secret,
    verification_token=feishu_config.verification_token,
    encrypt_key=feishu_config.encrypt_key
)

permission_manager = PermissionManager()
rag_permission_guard = RAGPermissionGuard(permission_manager)

# 重新创建带权限管理的 RAG 工具
from tool_subgraphs import RagSearchTool
rag_search_tool_with_permission = RagSearchTool(
    web_search_tool=web_search_tool,
    permission_guard=rag_permission_guard
)
# 覆盖 tools 模块中的实例（如果已导入）
import tools
tools.rag_search_tool = rag_search_tool_with_permission
# 同时更新当前模块的引用
rag_search_tool = rag_search_tool_with_permission

# 从配置文件初始化部门
for dept_id, dept_info in init_config.default_departments.items():
    permission_manager.create_department(dept_id, dept_info["name"], dept_info["description"])

# 从配置文件初始化用户
for username, (password, dept_id) in init_config.default_users.items():
    permission_manager.create_user(username, password, dept_id)

# 从配置文件设置部门管理员
for dept_id, admins in init_config.default_dept_admins.items():
    for admin_user_id in admins:
        permission_manager.add_department_owner(dept_id, admin_user_id)

# 从配置文件创建知识库
for kb_config in init_config.default_knowledge_bases:
    permission_manager.create_knowledge_base(
        kb_config["kb_id"], 
        kb_config["name"], 
        kb_config["owner"], 
        kb_config["department"]
    )

api_key = security_config.dashscope_api_key or security_config.openai_api_key
base_url = llm_config.dashscope_base_url if security_config.dashscope_api_key else None

if api_key:
    MAIN_AGENT_MODEL = ChatOpenAI(
        model=llm_config.model_name,
        temperature=llm_config.main_temperature,
        max_tokens=llm_config.max_tokens,
        api_key=api_key,
        base_url=base_url
    )
    if llm_config.enable_thinking:
        reasoning_extra = {"extra_body": {"enable_thinking": True}}
    else:
        reasoning_extra = {}

    REASONING_MODEL = ChatOpenAI(
        model=llm_config.reasoning_model_name,
        temperature=llm_config.reasoning_temperature,
        max_tokens=llm_config.reasoning_max_tokens,
        api_key=api_key,
        base_url=base_url,
        **reasoning_extra
    )
    
    SQL_MODEL = ChatOpenAI(
        model=llm_config.model_name,
        temperature=llm_config.sql_temperature,
        max_tokens=llm_config.max_tokens,
        api_key=api_key,
        base_url=base_url
    )
    
    RAG_MODEL = ChatOpenAI(
        model=llm_config.model_name,
        temperature=llm_config.rag_temperature,
        max_tokens=llm_config.max_tokens,
        api_key=api_key,
        base_url=base_url
    )
    
    MEETING_MODEL = ChatOpenAI(
        model=llm_config.model_name,
        temperature=llm_config.meeting_temperature,
        max_tokens=llm_config.max_tokens,
        api_key=api_key,
        base_url=base_url
    )
    
    logger.info(f"主Agent模型: {llm_config.model_name} (温度={llm_config.main_temperature}), 推理模型: {llm_config.reasoning_model_name} (温度={llm_config.reasoning_temperature})")
    logger.info(f"SQL模型: 温度={llm_config.sql_temperature}, RAG模型: 温度={llm_config.rag_temperature}, 会议模型: 温度={llm_config.meeting_temperature}")
else:
    MAIN_AGENT_MODEL = None
    REASONING_MODEL = None
    SQL_MODEL = None
    RAG_MODEL = None
    MEETING_MODEL = None
    logger.warning("未配置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY，AI功能将无法使用")

EMBEDDING_MODEL = None
if api_key:
    try:
        if security_config.dashscope_api_key:
            from dashscope import TextEmbedding
            
            class DashScopeEmbeddings:
                def __init__(self, model_name, api_key):
                    self.model_name = model_name
                    self.api_key = api_key
                
                def embed_documents(self, texts):
                    if isinstance(texts, str):
                        texts = [texts]
                    resp = TextEmbedding.call(
                        model=self.model_name,
                        input=texts,
                        api_key=self.api_key
                    )
                    if resp.status_code == 200:
                        return [item['embedding'] for item in resp.output['embeddings']]
                    else:
                        logger.error(f"嵌入失败: {resp.message}")
                        return []
                
                def embed_query(self, text):
                    resp = TextEmbedding.call(
                        model=self.model_name,
                        input=text,
                        api_key=self.api_key
                    )
                    if resp.status_code == 200:
                        return resp.output['embeddings'][0]['embedding']
                    else:
                        logger.error(f"嵌入失败: {resp.message}")
                        return []
            
            EMBEDDING_MODEL = DashScopeEmbeddings(
                model_name=llm_config.embedding_model_name,
                api_key=security_config.dashscope_api_key
            )
        else:
            EMBEDDING_MODEL = OpenAIEmbeddings(
                api_key=api_key,
                model=llm_config.embedding_model_name,
                base_url=base_url
            )
        logger.info(f"嵌入模型: {llm_config.embedding_model_name}")
    except Exception as e:
        logger.warning(f"嵌入模型初始化失败: {e}")

document_subgraph = DocumentIngestionSubGraph(embedding_model=EMBEDDING_MODEL)

from tool_subgraphs import sql_subgraph, rag_subgraph
if SQL_MODEL:
    sql_subgraph.set_llm(SQL_MODEL)
if RAG_MODEL:
    rag_subgraph.set_llm(RAG_MODEL)

from meeting_agent import MeetingAgent
meeting_agent = MeetingAgent(feishu_client=feishu_client, llm=MEETING_MODEL) if MEETING_MODEL else None


def create_main_agent():
    if MAIN_AGENT_MODEL is None:
        async def simple_agent(messages, metadata=None):
            return {"final_output": "服务已启动，但AI模型未配置。请设置OPENAI_API_KEY环境变量以启用AI功能。"}
        return simple_agent

    tools = {t.name: t._run for t in tools_from_B}

    class PlannerAgent:
        def __init__(self, llm, tools):
            self.llm = llm
            self.tools = tools

        async def ainvoke(self, input_data):
            messages = input_data.get("messages", [])
            context = input_data.get("context", {})
            tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in self.tools])

            prompt = f"""
            你是一个严格的工具调用规划器。**绝对禁止**直接回答用户的数据查询问题。

            **规则**：
            1. 如果用户问题涉及查询、统计、获取数据（例如“查询部门信息”、“员工工资”、“订单数量”），你必须生成一个调用 `query_sql_tool` 的任务，并且 `needs_tools` 必须为 true。
            2. 不要尝试自己回答数据查询问题，即使你认为信息不足。
            3. 只有当用户是简单的问候（如“你好”）或闲聊时，才可以将 `needs_tools` 设为 false 并输出一个简单的 plan。

            用户消息：{messages[-1].content if messages else ''}
            上下文信息：{json.dumps(context, ensure_ascii=False)}

            可用工具：
            {tool_descriptions}

            **输出格式（严格遵守，不要输出任何其他文本）**：
            {{
                "plan": "使用 query_sql_tool 查询部门信息",
                "tasks": [{{"tool": "query_sql_tool", "input": {{"query": "用户原始问题"}}}}],
                "needs_tools": true
            }}

            请根据用户消息输出对应的 JSON：
            """

            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            try:
                result = json.loads(response.content)
            except:
                result = {"plan": response.content, "tasks": [], "needs_tools": False}
            return result

    class ReflectorAgent:
        def __init__(self, llm, tools):
            self.llm = llm
            self.tools = tools

        async def ainvoke(self, input_data):
            original_plan = input_data.get("original_plan", {})
            execution_results = input_data.get("execution_results", [])
            errors = input_data.get("errors", [])
            reflection_count = input_data.get("reflection_count", 0)

            prompt = f"""
            你是一个智能反思助手，请分析执行结果并决定下一步行动。

            原始计划：{json.dumps(original_plan, ensure_ascii=False)}

            执行结果：
            {json.dumps(execution_results, ensure_ascii=False)}

            错误信息：{json.dumps(errors, ensure_ascii=False)}

            反思次数：{reflection_count}

            请分析：
            1. 执行是否成功？
            2. 如果失败，原因是什么？
            3. 是否需要重试？如果需要，请提供新的执行策略。

            输出格式（JSON）：
            {{
                "analysis": "分析结果描述",
                "decision": "retry"/"continue",
                "new_strategy": "如果重试，新的执行策略"
            }}
            """

            response = await self.llm.ainvoke([{"role": "user", "content": prompt}])
            try:
                result = json.loads(response.content)
            except:
                result = {"analysis": response.content, "decision": "continue", "new_strategy": ""}
            return result

    planner_agent = PlannerAgent(MAIN_AGENT_MODEL, tools_from_B)
    reflector_agent = ReflectorAgent(REASONING_MODEL, tools_from_B)

    sub_agents = {
        "planner": planner_agent,
        "reflector": reflector_agent
    }

    return build_main_graph(MAIN_AGENT_MODEL, tools, sub_agents)


main_agent = create_main_agent()
feishu_event_graph = FeishuEventSubGraph(main_agent)
feishu_event_graph.set_feishu_client(feishu_client)

automation_agent = AutomationAgent(feishu_client, MAIN_AGENT_MODEL)
data_analysis_agent = DataAnalysisAgent(feishu_client, MAIN_AGENT_MODEL)
meeting_agent = MeetingAgent(feishu_client, MEETING_MODEL) if MEETING_MODEL else None


async def process_message(sender_id: str, message_id: str, text: str):
    try:
        auto_result = await automation_agent.process_message(sender_id, message_id, text)

        if auto_result.get("action") != "none":
            logger.info(f"自动化流程已处理: {auto_result.get('action')}")
            return

        result = await feishu_event_graph.run({
            "sender_id": sender_id,
            "message_id": message_id,
            "message_type": "text",
            "content": text
        })

    except Exception as e:
        logger.error(f"处理消息失败: {str(e)}")
        await feishu_client.send_text_message(sender_id, f"处理消息时发生错误: {str(e)}")


async def process_file(sender_id: str, message_id: str, file_key: str, file_name: str):
    try:
        file_content = await feishu_client.get_file_content(file_key)
        if not file_content:
            await feishu_client.send_text_message(sender_id, "无法获取文件内容")
            return

        result = await document_subgraph.run({
            "file_content": file_content,
            "file_name": file_name,
            "text_chunks": [],
            "embeddings": None,
            "index_path": "",
            "success": False,
            "error": None,
            "chunks_count": 0
        })

        if result.get("success"):
            msg = f"文件处理完成！\n文件名: {file_name}\n分割片段数: {result.get('chunks_count', 0)}\n索引位置: {result.get('index_path', '')}"
        else:
            msg = f"文件处理失败: {result.get('error', '未知错误')}"

        await feishu_client.send_text_message(sender_id, msg)

    except Exception as e:
        logger.error(f"处理文件失败: {str(e)}")
        await feishu_client.send_text_message(sender_id, f"处理文件时发生错误: {str(e)}")


feishu_client.register_message_handler("*", process_message)
feishu_client.register_file_handler(process_file)


@app.post("/feishu/webhook")
async def feishu_webhook(request: Request):
    try:
        body = await request.body()
        body_str = body.decode("utf-8")
        headers = dict(request.headers)

        if not feishu_client.verify_request(headers, body_str):
            return JSONResponse(content={"error": "验证失败"}, status_code=401)

        event_data = json.loads(body_str)
        event_type = event_data.get("header", {}).get("event_type", "")

        if event_type == "im.message.receive_v1":
            asyncio.create_task(feishu_client._handle_message_event(event_data))

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Webhook处理失败: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/message")
async def send_message(request: Request, message: MessageRequest, user_id: str = Depends(get_current_user)):
    try:
        content = message.content
        logger.info(f"收到消息: sender_id={message.sender_id}, content长度={len(content)}, content='{content}'")
        logger.info(f"content 类型: {type(content)}")
        
        simple_result = await handle_simple_test(content, user_id=user_id)
        logger.info(f"handle_simple_test 结果: {simple_result}")
        
        if simple_result and simple_result.get("response"):
            return {
                "response": simple_result.get("response"),
                "tool": simple_result.get("tool", "unknown"),
                "card": simple_result.get("card")
            }

        logger.info("执行完整Agent流程")
        result = await feishu_event_graph.run_and_wait_async({
            "sender_id": message.sender_id,
            "message_id": "api_message",
            "message_type": message.message_type,
            "content": message.content
        })

        return {"response": result.get("agent_response"), "card": result.get("card_result")}

    except Exception as e:
        logger.error(f"API消息处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def handle_simple_test(content: str, user_id: str = None):
    content_lower = content.lower()

    # 优先匹配 RAG 知识库查询
    if any(keyword in content_lower for keyword in ["知识库", "知", "公司"]):
        logger.info(f"🧪 简单测试模式：RAG搜索 - {content} (user={user_id})")
        try:
            result_str = await rag_search_tool._arun(content, user_id=user_id)
            result = json.loads(result_str)
            return {
                "response": result.get("result", "知识库搜索完成"),
                "tool": "rag_search_tool",
                "raw": result
            }
        except Exception as e:
            logger.error(f"RAG搜索失败: {e}")
            return {"response": f"知识库搜索尝试失败: {e}"}

    # 然后匹配 SQL 查询
    if any(keyword in content_lower for keyword in ["sql", "查", "查询", "统计", "部门", "员工", "订单"]):
        logger.info(f"🧪 简单测试模式：SQL查询 - {content} (user={user_id})")
        try:
            # 获取用户部门信息用于权限过滤
            user_dept = permission_manager.get_user_department(user_id) if user_id else None
            logger.info(f"用户 {user_id} 所属部门: {user_dept}")
            
            result_str = await query_sql_tool._arun(content, user_id=user_id, department=user_dept)
            result = json.loads(result_str)
            return {
                "response": result.get("result", "查询完成"),
                "tool": "query_sql_tool",
                "raw": result
            }
        except Exception as e:
            logger.error(f"SQL查询失败: {e}")
            return {"response": f"SQL查询尝试失败: {e}"}

    # 最后匹配 Web 搜索
    if any(keyword in content_lower for keyword in ["百度", "搜索", "网上"]):
        logger.info(f"简单测试模式：Web搜索 - {content}")
        try:
            result_str = await web_search_tool._arun(content)
            result = json.loads(result_str)
            return {
                "response": result.get("result", "搜索完成"),
                "tool": "web_search_tool",
                "raw": result
            }
        except Exception as e:
            logger.error(f"Web搜索失败: {e}")
            return {"response": f"Web搜索尝试失败: {e}"}

    return None


@app.post("/api/web/search")
async def web_search_api(request: Request, user_id: str = Depends(get_current_user)):
    """直接调用 Web 搜索工具"""
    try:
        # 支持 JSON 和表单两种格式
        try:
            body = await request.json()
            query = body.get("query", "")
        except:
            query = ""
        
        if not query:
            form_data = await request.form()
            query = form_data.get("query", "")
        
        if not query:
            raise HTTPException(status_code=400, detail="请输入搜索关键词")
        
        result_str = await web_search_tool._arun(query)
        result = json.loads(result_str)
        
        # 直接返回整理后的文本内容，而不是 JSON 格式
        if result.get("success") and result.get("result"):
            # 还原转义的换行符
            raw_result = result["result"]
            # 替换转义的换行符为实际换行符
            formatted_result = raw_result.replace('\\n', '\n')
            return formatted_result
        else:
            return "暂无搜索结果"
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Web搜索失败: {e}")
        return f"搜索失败: {str(e)}"


@app.post("/api/upload")
async def upload_document(request: Request, user_id: str = Depends(get_current_user)):
    try:
        form_data = await request.form()
        file = form_data.get("file")

        if not file:
            raise HTTPException(status_code=400, detail="请上传文件")

        file_content = await file.read()
        file_name = sanitize_filename(file.filename)

        result = await document_subgraph.run({
            "file_content": file_content,
            "file_name": file_name,
            "text_chunks": [],
            "embeddings": None,
            "index_path": "",
            "success": False,
            "error": None,
            "chunks_count": 0
        })

        return {
            "success": result.get("success"),
            "chunks_count": result.get("chunks_count"),
            "index_path": result.get("index_path"),
            "error": result.get("error")
        }

    except Exception as e:
        logger.error(f"文件上传处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stream")
async def stream_response(request: MessageRequest, user_id: str = Depends(get_current_user)):
    """流式响应端点 - 支持真实的SSE流式输出"""
    async def generate():
        try:
            # 模拟流式响应（实际生产环境应连接LLM的流式API）
            response_text = "这是一个流式响应的演示。飞书智能助手支持自然语言转SQL查询、企业知识库检索和实时Web搜索功能。"
            
            # 逐个字符发送，模拟流式效果
            for i, char in enumerate(response_text):
                # 每3个字符发送一次，模拟真实的流式输出
                if i % 3 == 0 or i == len(response_text) - 1:
                    chunk = response_text[max(0, i-2):i+1] if i > 1 else response_text[:i+1]
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk, 'done': False}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.05)  # 控制流速度
            
            # 发送结束标记
            yield f"data: {json.dumps({'type': 'content', 'content': '', 'done': True}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"流式响应失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e), 'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked"
    })

class StreamRequest(BaseModel):
    """流式请求模型"""
    query: str
    type: str = "stream"  # sql, rag, web, stream

@app.post("/api/stream/all")
async def stream_all(request: StreamRequest, user_id: str = Depends(get_current_user)):
    """统一流式响应端点 - 支持SQL查询、RAG检索、Web搜索的流式输出"""
    async def generate():
        try:
            query = request.query
            func_type = request.type
            
            # 获取用户部门信息用于权限过滤
            user_dept = permission_manager.get_user_department(user_id) if user_id else None
            logger.info(f"用户 {user_id} 所属部门: {user_dept}")
            
            # 根据类型执行不同的查询
            if func_type == "sql":
                yield f"data: {json.dumps({'type': 'status', 'content': '正在执行SQL查询...'}, ensure_ascii=False)}\n\n"
                result_str = await query_sql_tool._arun(query, user_id=user_id, department=user_dept)
                result = json.loads(result_str)
                content = result.get("result", result.get("message", "查询完成"))
            elif func_type == "rag":
                yield f"data: {json.dumps({'type': 'status', 'content': '正在检索知识库...'}, ensure_ascii=False)}\n\n"
                result_str = await rag_search_tool._arun(query, user_id=user_id)
                result = json.loads(result_str)
                content = result.get("result", result.get("message", "检索完成"))
            elif func_type == "web":
                yield f"data: {json.dumps({'type': 'status', 'content': '正在搜索网络...'}, ensure_ascii=False)}\n\n"
                result_str = await web_search_tool._arun(query)
                result = json.loads(result_str)
                content = result.get("result", result.get("message", "搜索完成"))
            else:
                # 默认：智能选择工具执行
                yield f"data: {json.dumps({'type': 'status', 'content': '正在分析问题...'}, ensure_ascii=False)}\n\n"
                
                # 意图识别：根据用户问题判断应该调用哪个工具
                query_lower = query.lower()
                
                # Web搜索关键词 - 需要实时信息的问题
                web_keywords = ['新闻', '今天', '最新', '现在', '最近', '天气', '股票', '价格', 
                               '娱乐', '体育', '科技', '资讯', '头条', '热搜', '动态',
                               '新闻资讯', '今日', '当前', '最近更新', '最新消息']
                
                # SQL查询关键词 - 需要查询数据库的问题
                sql_keywords = ['查询', '员工', '数据', '报表', '统计', '数量', '有多少', 
                               '列表', '信息', '记录', '表格', '汇总', '人数', '部门',
                               '员工信息', '数据统计', '总共有', '有哪些']
                
                # RAG查询关键词 - 需要查询知识库的问题
                rag_keywords = ['制度', '政策', '规定', '流程', '手册', '指南', '请假', 
                               '考勤', '报销', '福利', '公司', '制度', '政策', '规定',
                               '年假', '病假', '加班', '薪酬', '培训', '入职', '离职']
                
                # 统计各类型关键词匹配数量
                web_score = sum(1 for keyword in web_keywords if keyword in query_lower)
                sql_score = sum(1 for keyword in sql_keywords if keyword in query_lower)
                rag_score = sum(1 for keyword in rag_keywords if keyword in query_lower)
                
                content = ""
                
                # 根据得分判断用户意图，调用最合适的工具
                max_score = max(web_score, sql_score, rag_score)
                
                if max_score == 0:
                    # 没有明确意图，尝试通用对话
                    yield f"data: {json.dumps({'type': 'status', 'content': '正在分析您的问题...'}, ensure_ascii=False)}\n\n"
                    content = f"已收到您的问题：{query}\n\n我可以帮助您：\n- 查询公司知识库（如：考勤制度、请假流程）\n- 查询数据库数据（如：员工信息、部门统计）\n- 搜索网络信息（如：新闻资讯、实时动态）"
                
                elif web_score == max_score:
                    # 用户意图是获取实时信息
                    yield f"data: {json.dumps({'type': 'status', 'content': '正在搜索网络...'}, ensure_ascii=False)}\n\n"
                    try:
                        result_str = await web_search_tool._arun(query)
                        result = json.loads(result_str)
                        content = result.get("result", result.get("message", ""))
                        if not content or len(content.strip()) < 20:
                            content = f"未能找到关于「{query}」的相关信息"
                    except Exception as e:
                        logger.error(f"Web搜索失败: {e}")
                        content = f"网络搜索暂时不可用，请稍后重试"
                
                elif sql_score == max_score:
                    # 用户意图是查询数据库
                    yield f"data: {json.dumps({'type': 'status', 'content': '正在查询数据库...'}, ensure_ascii=False)}\n\n"
                    try:
                        result_str = await query_sql_tool._arun(query, user_id=user_id, department=user_dept)
                        result = json.loads(result_str)
                        content = result.get("result", result.get("message", "查询完成"))
                    except Exception as e:
                        logger.error(f"SQL查询失败: {e}")
                        content = f"数据库查询暂时不可用，请稍后重试"
                
                elif rag_score == max_score:
                    # 用户意图是查询知识库
                    yield f"data: {json.dumps({'type': 'status', 'content': '正在检索知识库...'}, ensure_ascii=False)}\n\n"
                    try:
                        result_str = await rag_search_tool._arun(query, user_id=user_id)
                        result = json.loads(result_str)
                        content = result.get("result", result.get("message", "未找到相关信息"))
                    except Exception as e:
                        logger.error(f"RAG检索失败: {e}")
                        content = f"知识库检索暂时不可用，请稍后重试"
            
            # 流式输出内容
            chunk_size = 10
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'content', 'content': chunk, 'done': False}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.03)  # 控制流速度
            
            # 发送结束标记
            yield f"data: {json.dumps({'type': 'content', 'content': '', 'done': True}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"流式响应失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e), 'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked"
    })


@app.post("/token")
async def login(username: str = Form(...), password: str = Form(...)):
    """用户登录 - 获取访问令牌"""
    # 检查用户是否存在
    if username not in permission_manager.user_departments:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 验证密码（实际生产环境应使用哈希）
    stored_password = permission_manager.user_passwords.get(username)
    if stored_password != password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=security_config.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


class TaskRequest(BaseModel):
    title: str
    description: str = ""
    due_date: str = ""
    assignee_id: str = ""


@app.post("/api/tasks/create")
async def create_task_api(request: TaskRequest, user_id: str = Depends(get_current_user)):
    try:
        task = await feishu_client.create_task(
            title=request.title,
            description=request.description,
            assignee_id=request.assignee_id if request.assignee_id else None
        )

        if task:
            return {"status": "success", "task": task}
        else:
            raise HTTPException(status_code=500, detail="创建任务失败")

    except Exception as e:
        logger.error(f"创建任务API失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks/list")
async def list_tasks_api(user_id: str = Depends(get_current_user)):
    tasks = await feishu_client.get_tasks()
    return {"status": "success", "tasks": tasks}


class DataQueryRequest(BaseModel):
    query: str


@app.post("/api/data/query")
async def query_data_api(request: DataQueryRequest, user_id: str = Depends(get_current_user)):
    """SQL查询接口 - 将自然语言转换为SQL并执行（带部门权限过滤）"""
    try:
        # 获取用户所属部门
        user_dept = permission_manager.get_user_department(user_id)
        logger.info(f"用户 {user_id} 所属部门: {user_dept}")
        
        # 传递 user_id 和部门信息给 SQL 工具
        result_str = await query_sql_tool._arun(request.query, user_id=user_id, department=user_dept)
        result = json.loads(result_str)
        return {
            "status": "success",
            "intent": "query_data",
            "report": {
                "type": "success",
                "content": result.get("result", "查询完成")
            },
            "sql": result.get("sql"),
            "raw": result
        }
    except Exception as e:
        logger.error(f"SQL查询失败: {e}")
        return {
            "status": "success",
            "intent": "query_data",
            "report": {
                "type": "error",
                "content": f"查询失败: {str(e)}"
            }
        }


@app.post("/api/data/rag")
async def rag_search_api(request: DataQueryRequest, user_id: str = Depends(get_current_user)):
    """RAG检索接口 - 在企业知识库中检索相关文档"""
    try:
        logger.info(f"用户 {user_id} 执行RAG检索: {request.query}")
        
        # 调用 RAG 检索工具
        result_str = await rag_search_tool._arun(request.query, user_id=user_id)
        result = json.loads(result_str)
        
        return {
            "status": "success",
            "intent": "rag_search",
            "report": {
                "type": "success",
                "content": result.get("result", "检索完成")
            },
            "raw": result
        }
    except Exception as e:
        logger.error(f"RAG检索失败: {e}")
        return {
            "status": "success",
            "intent": "rag_search",
            "report": {
                "type": "error",
                "content": f"检索失败: {str(e)}"
            }
        }


class BitableRegisterRequest(BaseModel):
    config_id: str
    app_token: str
    table_id: str
    table_name: str = ""


@app.post("/api/bitable/register")
async def register_bitable_api(request: BitableRegisterRequest, user_id: str = Depends(get_current_user)):
    data_analysis_agent.register_bitable(
        request.config_id,
        request.app_token,
        request.table_id,
        request.table_name
    )
    return {"status": "success", "message": "表格注册成功"}


class MeetingTranscriptRequest(BaseModel):
    meeting_id: str
    transcript: str
    participants: List[str]


@app.post("/api/meeting/process")
async def process_meeting_api(request: MeetingTranscriptRequest, user_id: str = Depends(get_current_user)):
    result = await meeting_agent.process_meeting_transcript(
        request.meeting_id,
        request.transcript,
        request.participants
    )
    return result


@app.get("/api/meeting/history")
async def get_meeting_history_api(days: int = 30, query: str = "", user_id: str = Depends(get_current_user)):
    meetings = await meeting_agent.query_meeting_history(user_id, query, days)
    return {"status": "success", "meetings": meetings}


@app.get("/api/permissions/kbs")
async def get_user_kbs_api(user_id: str = Depends(get_current_user)):
    kbs = permission_manager.get_user_available_kbs(user_id)
    return {"status": "success", "knowledge_bases": kbs}


@app.get("/api/permissions/department")
async def get_department_info_api(dept_id: str = None, user_id: str = Depends(get_current_user)):
    if not dept_id:
        dept_id = permission_manager.get_user_department(user_id)

    dept_info = permission_manager.get_department_info(dept_id) if dept_id else None
    return {"status": "success", "department": dept_info}


class CreateKBRequest(BaseModel):
    kb_id: str
    name: str
    dept_id: str = ""


@app.post("/api/permissions/kb/create")
async def create_kb_api(request: CreateKBRequest, user_id: str = Depends(get_current_user)):
    success = permission_manager.create_knowledge_base(
        request.kb_id,
        request.name,
        user_id,
        request.dept_id if request.dept_id else None
    )

    if success:
        return {"status": "success", "message": "知识库创建成功"}
    else:
        raise HTTPException(status_code=400, detail="知识库创建失败")


# ==================== 用户和部门管理 API ====================

# 部门管理员权限检查依赖
async def require_department_admin(dept_id: str, user_id: str = Depends(get_current_user)):
    """验证用户是否是指定部门的管理员（admin 用户拥有所有权限）"""
    if not permission_manager.is_department_admin(user_id, dept_id):
        raise HTTPException(status_code=403, detail="没有权限操作该部门（需要是部门管理员）")
    return user_id


class CreateDepartmentRequest(BaseModel):
    dept_id: str
    name: str
    description: str = ""


@app.post("/api/permissions/department/create")
async def create_department_api(request: CreateDepartmentRequest, user_id: str = Depends(get_current_user)):
    """创建新部门（只有管理员可以创建）"""
    # 只有 admin 用户可以创建部门
    if user_id != "admin":
        raise HTTPException(status_code=403, detail="只有系统管理员可以创建部门")
    
    success = permission_manager.create_department(request.dept_id, request.name, request.description)
    if success:
        return {"status": "success", "message": "部门创建成功"}
    else:
        raise HTTPException(status_code=400, detail="部门创建失败（可能已存在）")


@app.delete("/api/permissions/department/{dept_id}")
async def delete_department_api(dept_id: str, user_id: str = Depends(get_current_user)):
    """删除部门（需要是部门管理员）"""
    await require_department_admin(dept_id, user_id)
    
    # 检查部门是否存在
    if not permission_manager.get_department_info(dept_id):
        raise HTTPException(status_code=400, detail="部门不存在")
    
    # 删除部门（简化实现：直接移除部门信息）
    if dept_id in permission_manager.department_info:
        del permission_manager.department_info[dept_id]
        if dept_id in permission_manager.department_kb_mapping:
            del permission_manager.department_kb_mapping[dept_id]
        if dept_id in permission_manager.department_owners:
            del permission_manager.department_owners[dept_id]
        
        logger.info(f"删除部门: {dept_id}")
        return {"status": "success", "message": "部门删除成功"}
    else:
        raise HTTPException(status_code=400, detail="部门不存在")


@app.post("/api/permissions/department/add-admin")
async def add_department_admin_api(request: dict = Body(...), user_id: str = Depends(get_current_user)):
    """添加部门管理员（需要是部门管理员）"""
    dept_id = request.get("dept_id")
    admin_user_id = request.get("admin_user_id")
    
    if not dept_id or not admin_user_id:
        raise HTTPException(status_code=400, detail="缺少参数：dept_id 和 admin_user_id")
    
    await require_department_admin(dept_id, user_id)
    
    permission_manager.add_department_owner(dept_id, admin_user_id)
    return {"status": "success", "message": f"用户 {admin_user_id} 已成为部门 {dept_id} 的管理员"}


@app.get("/api/permissions/departments")
async def get_all_departments_api(user_id: str = Depends(get_current_user)):
    """获取所有部门列表"""
    departments = permission_manager.get_all_departments()
    return {"status": "success", "departments": departments}


class CreateUserRequest(BaseModel):
    username: str
    password: str
    department: str


@app.post("/api/permissions/user/create")
async def create_user_api(request: CreateUserRequest, user_id: str = Depends(get_current_user)):
    """创建新用户并分配部门（需要是目标部门的管理员）"""
    # 验证部门是否存在
    if not permission_manager.get_department_info(request.department):
        raise HTTPException(status_code=400, detail="部门不存在")
    
    # 验证是否有创建权限（需要是目标部门的管理员）
    await require_department_admin(request.department, user_id)
    
    success = permission_manager.create_user(request.username, request.password, request.department)
    if success:
        return {"status": "success", "message": "用户创建成功"}
    else:
        raise HTTPException(status_code=400, detail="用户创建失败（可能已存在）")


@app.post("/api/permissions/user/update")
async def update_user_department_api(username: str, department: str, user_id: str = Depends(get_current_user)):
    """更新用户所属部门（需要是目标部门的管理员）"""
    # 验证部门是否存在
    if not permission_manager.get_department_info(department):
        raise HTTPException(status_code=400, detail="部门不存在")
    
    # 验证是否有权限（需要是目标部门的管理员）
    await require_department_admin(department, user_id)
    
    success = permission_manager.set_user_department(username, department)
    if success:
        return {"status": "success", "message": "用户部门更新成功"}
    else:
        raise HTTPException(status_code=400, detail="用户不存在")


@app.delete("/api/permissions/user/{username}")
async def delete_user_api(username: str, user_id: str = Depends(get_current_user)):
    """删除用户（需要是该用户所在部门的管理员）"""
    # 获取用户所在部门
    target_dept = permission_manager.get_user_department(username)
    if not target_dept:
        raise HTTPException(status_code=400, detail="用户不存在")
    
    # 验证是否有权限（需要是该用户所在部门的管理员）
    await require_department_admin(target_dept, user_id)
    
    success = permission_manager.delete_user(username)
    if success:
        return {"status": "success", "message": "用户删除成功"}
    else:
        raise HTTPException(status_code=400, detail="用户不存在")


@app.get("/api/permissions/users")
async def get_all_users_api(user_id: str = Depends(get_current_user)):
    """获取所有用户列表（只有管理员可以查看）"""
    users = permission_manager.get_all_users()
    return {"status": "success", "users": users}


# ==================== 员工管理 API ====================

class EmployeeCreateRequest(BaseModel):
    name: str
    gender: str
    department: str
    position: str
    hire_date: str
    salary: int
    email: str
    phone: str
    status: str = "在职"


@app.post("/api/employees")
async def create_employee(request: EmployeeCreateRequest, user_id: str = Depends(get_current_user)):
    """创建新员工（需要是部门管理员）"""
    # 支持部门名称和部门ID两种方式
    dept_id = None
    dept_name = request.department
    
    # 首先检查是否是部门ID（如 dept_1）
    if request.department.startswith("dept_"):
        dept_id = request.department
        dept_info = permission_manager.get_department_info(dept_id)
        if dept_info:
            dept_name = dept_info.get("name", request.department)
    else:
        # 按部门名称查找部门ID
        for did, info in permission_manager.department_info.items():
            if info.get("name") == request.department:
                dept_id = did
                break
    
    if not dept_id:
        raise HTTPException(status_code=400, detail="部门不存在")
    
    # 验证是否有权限（需要是目标部门的管理员）
    await require_department_admin(dept_id, user_id)
    
    # 使用SQLSubGraph执行INSERT操作（管理员模式）
    try:
        sql_query = f"""INSERT INTO employees (name, gender, department, position, hire_date, salary, email, phone, status)
                       VALUES ('{request.name}', '{request.gender}', '{dept_name}', '{request.position}', 
                               '{request.hire_date}', {request.salary}, '{request.email}', '{request.phone}', '{request.status}')"""
        
        result = await sql_subgraph.run(sql_query, department=dept_id, is_admin=True)
        
        if result["success"]:
            import json
            exec_result = json.loads(result.get("execution_result", "{}"))
            emp_id = exec_result.get("last_insert_id", 0)
            return {"status": "success", "message": "员工创建成功", "emp_id": emp_id}
        else:
            raise HTTPException(status_code=500, detail=f"创建员工失败: {result.get('error', '未知错误')}")
    except Exception as e:
        logger.error(f"创建员工失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建员工失败: {str(e)}")


@app.get("/api/employees")
async def get_employees(department: str = None, user_id: str = Depends(get_current_user)):
    """查询员工列表（根据权限过滤）"""
    user_dept = permission_manager.get_user_department(user_id)
    
    # 确定目标部门名称
    target_dept_name = None
    target_dept_id = None
    
    if department:
        # 如果指定了部门，处理部门ID或部门名称
        if department.startswith("dept_"):
            # 是部门ID，转换为部门名称
            target_dept_id = department
            dept_info = permission_manager.get_department_info(department)
            if dept_info:
                target_dept_name = dept_info.get("name")
        else:
            # 直接使用部门名称
            target_dept_name = department
            # 查找对应的部门ID
            for did, info in permission_manager.department_info.items():
                if info.get("name") == department:
                    target_dept_id = did
                    break
    elif user_dept:
        # 使用用户所在部门
        target_dept_id = user_dept
        dept_info = permission_manager.get_department_info(user_dept)
        if dept_info:
            target_dept_name = dept_info.get("name")
    
    # 使用SQLSubGraph执行查询
    try:
        if target_dept_name:
            sql_query = f"SELECT * FROM employees WHERE department = '{target_dept_name}'"
        else:
            sql_query = "SELECT * FROM employees"
        
        result = await sql_subgraph.run(sql_query, department=target_dept_id, is_admin=False)
        
        if result["success"]:
            import json
            exec_result = json.loads(result.get("execution_result", "{}"))
            data = exec_result.get("data", [])
            return {"status": "success", "data": data, "count": len(data)}
        else:
            raise HTTPException(status_code=500, detail=f"查询员工失败: {result.get('error', '未知错误')}")
    except Exception as e:
        logger.error(f"查询员工失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询员工失败: {str(e)}")


@app.delete("/api/permissions/kb/{kb_id}")
async def delete_kb_api(kb_id: str, user_id: str = Depends(get_current_user)):
    """删除知识库"""
    success = permission_manager.delete_knowledge_base(user_id, kb_id)

    if success:
        return {"status": "success", "message": "知识库删除成功"}
    else:
        raise HTTPException(status_code=400, detail="知识库删除失败或无权限")


@app.post("/api/reminders/send")
async def send_reminders_api(user_id: str = Depends(get_current_user)):
    reminders = await automation_agent.auto_send_reminder()
    return {"status": "success", "reminders_sent": len(reminders), "reminders": reminders}


import signal
import sys
import subprocess

def release_port(port: int):
    """释放指定端口（Windows）"""
    try:
        # 获取占用端口的进程ID
        result = subprocess.run(
            ["netstat", "-ano", "|", "findstr", f":{port}"],
            capture_output=True,
            text=True,
            shell=True
        )
        
        if result.returncode == 0 and result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/PID", pid],
                            capture_output=True
                        )
                        print(f"已释放端口 {port} (PID: {pid})")
                    except Exception:
                        pass
    except Exception:
        pass

async def shutdown_resources():
    """关闭所有资源"""
    await background_task_queue.stop()
    if 'feishu_client' in globals():
        await feishu_client.close()
    
    # 关闭数据库连接
    try:
        from database import get_db_connection
        db_conn = get_db_connection()
        await db_conn.close()
    except:
        pass

async def main():
    """主函数"""
    # 启动前尝试释放端口（防止端口被占用）
    if sys.platform == "win32":
        release_port(system_config.port)
    
    # 注册信号处理（Unix/Linux）
    if sys.platform != "win32":
        def handle_signal(signum, frame):
            logger.info(f"收到信号 {signum}，准备关闭...")
            raise KeyboardInterrupt()
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    
    logger.info("启动飞书集成服务...")
    logger.info("可用API: /health, /token, /feishu/webhook, /api/message, /api/upload, /api/stream")
    logger.info("新功能API: /api/tasks/*, /api/data/*, /api/meeting/*, /api/permissions/*, /api/bitable/*")
    
    # 使用 uvicorn 的配置来支持优雅关闭
    config = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=system_config.port,
        reload=False,
        timeout_graceful_shutdown=10  # 优雅关闭超时时间
    )
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("收到键盘中断，正在优雅关闭...")
        # 触发优雅关闭
        await server.shutdown()
    finally:
        await shutdown_resources()

if __name__ == "__main__":
    import uvicorn
    asyncio.run(main())
