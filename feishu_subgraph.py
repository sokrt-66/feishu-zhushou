from typing import TypedDict, Optional, Any, Dict, List
from langchain_core.messages import HumanMessage
import asyncio
import re
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
import uuid

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackgroundTask:
    task_id: str
    task_type: str
    params: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


class BackgroundTaskQueue:
    """简单的后台任务队列"""

    def __init__(self, max_concurrent: int = 5):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._tasks: Dict[str, BackgroundTask] = {}
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False

    async def start(self):
        """启动任务处理器"""
        self._running = True
        for _ in range(self._max_concurrent):
            asyncio.create_task(self._process_tasks())

    async def stop(self):
        """停止任务处理器"""
        self._running = False

    async def _process_tasks(self):
        """处理后台任务"""
        while self._running:
            try:
                async with self._semaphore:
                    task_info = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    task_id, task_func, params = task_info
                    await self._execute_task(task_id, task_func, params)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"处理后台任务失败: {e}")

    async def _execute_task(self, task_id: str, task_func: Callable, params: Dict[str, Any]):
        """执行单个任务"""
        task = self._tasks.get(task_id)
        if not task:
            return

        task.status = TaskStatus.RUNNING

        try:
            if asyncio.iscoroutinefunction(task_func):
                result = await task_func(**params)
            else:
                result = task_func(**params)

            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            logger.info(f"后台任务完成: {task_id}")

        except Exception as e:
            task.error = str(e)
            task.retry_count += 1

            if task.retry_count < task.max_retries:
                task.status = TaskStatus.PENDING
                self._queue.put_nowait((task_id, task_func, params))
                logger.warning(f"任务 {task_id} 失败，将重试 ({task.retry_count}/{task.max_retries})")
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                logger.error(f"任务 {task_id} 最终失败: {e}")

    def submit_task(self, task_type: str, task_func: Callable, params: Dict[str, Any]) -> str:
        """提交任务"""
        task_id = f"{task_type}_{uuid.uuid4().hex[:8]}"
        task = BackgroundTask(
            task_id=task_id,
            task_type=task_type,
            params=params
        )
        self._tasks[task_id] = task
        self._queue.put_nowait((task_id, task_func, params))
        logger.info(f"任务已提交: {task_id}")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[BackgroundTask]:
        """获取任务状态"""
        return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[BackgroundTask]:
        """列出任务"""
        if status:
            return [t for t in self._tasks.values() if t.status == status]
        return list(self._tasks.values())


background_task_queue = BackgroundTaskQueue(max_concurrent=3)


class FeishuEventState(TypedDict):
    sender_id: str
    message_id: str
    message_type: str
    content: str
    intent: Optional[str]
    routing_key: Optional[str]
    agent_response: Optional[str]
    card_result: Optional[Dict[str, Any]]
    error: Optional[str]
    background_task: Optional[Any]


class FeishuEventSubGraph:
    def __init__(self, main_agent, llm_model: str = "gpt-4o-mini"):
        self.main_agent = main_agent
        self.feishu_client = None
        self.llm = None
        try:
            from langchain_openai import ChatOpenAI
            import os
            openai_api_key = os.getenv("OPENAI_API_KEY", "")
            if openai_api_key:
                self.llm = ChatOpenAI(model=llm_model, temperature=0, api_key=openai_api_key)
        except Exception as e:
            logger.warning(f"LLM初始化失败: {e}")

    def set_feishu_client(self, feishu_client):
        self.feishu_client = feishu_client

    async def _intent_recognition(self, text: str) -> str:
        intent_patterns = {
            "file_upload": r"(上传|文件|文档|pdf|附件)",
            "knowledge_query": r"(查询|搜索|知识库|了解|是什么|为什么|如何)",
            "task_request": r"(帮我|请|需要|想)",
            "greeting": r"(你好|您好|hi|hello|早|晚)",
            "unknown": r".*"
        }

        for intent_name, pattern in intent_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return intent_name
        return "unknown"

    def _route_by_intent(self, intent: str) -> str:
        routing_map = {
            "file_upload": "document_processing",
            "knowledge_query": "rag_search",
            "task_request": "task_execution",
            "greeting": "direct_response",
            "unknown": "general_query"
        }
        return routing_map.get(intent, "general_query")

    async def _call_agent(self, routing_key: str, content: str) -> str:
        if routing_key == "direct_response":
            responses = {
                "greeting": "您好！我是您的智能助手，请问有什么可以帮助您的？"
            }
            return responses.get(routing_key, "收到，我来帮您处理。")

        messages = [HumanMessage(content=content)]
        result = await self.main_agent(messages, {"routing_key": routing_key})

        if isinstance(result, dict) and "final_output" in result:
            return result["final_output"]
        elif hasattr(result, 'content'):
            return result.content
        else:
            return str(result)

    def _generate_card(self, response: str) -> Dict[str, Any]:
        return {
            "config": {"wide_screen_mode": True},
            "elements": [
                {
                    "tag": "div",
                    "text": {"content": response, "tag": "lark_md"}
                },
                {
                    "tag": "div",
                    "text": {"content": "\n---\n*以上内容由智能助手生成*", "tag": "lark_md"}
                }
            ]
        }

    def _generate_processing_card(self) -> Dict[str, Any]:
        return {
            "config": {"wide_screen_mode": True},
            "elements": [
                {
                    "tag": "div",
                    "text": {"content": "🔄 正在处理您的请求，请稍候...", "tag": "lark_md"}
                }
            ]
        }

    async def _background_process(self, state: FeishuEventState):
        """后台处理实际逻辑"""
        try:
            state["intent"] = await self._intent_recognition(state["content"])
            state["routing_key"] = self._route_by_intent(state["intent"])
            state["agent_response"] = await self._call_agent(state["routing_key"], state["content"])
            state["card_result"] = self._generate_card(state["agent_response"])

            if self.feishu_client:
                await self.feishu_client.send_card_message(state["sender_id"], state["card_result"])
                await self.feishu_client.send_text_message(
                    state["sender_id"],
                    f"📝 处理完成：\n{state['agent_response']}"
                )

        except Exception as e:
            logger.error(f"后台处理失败: {str(e)}")
            state["error"] = str(e)
            state["card_result"] = {
                "config": {"wide_screen_mode": True},
                "elements": [{
                    "tag": "div",
                    "text": {"content": f"处理消息时发生错误: {str(e)}", "tag": "lark_md"}
                }]
            }
            if self.feishu_client:
                await self.feishu_client.send_text_message(
                    state["sender_id"],
                    f"❌ 处理失败: {str(e)}"
                )

    async def run(self, state: FeishuEventState) -> FeishuEventState:
        state["card_result"] = self._generate_processing_card()
        state["background_task"] = asyncio.create_task(self._background_process(state))
        return state

    def run_and_wait(self, state: FeishuEventState) -> FeishuEventState:
        """同步包装方法 - 等待执行完成"""
        return asyncio.run(self._run_sync(state))

    async def _run_sync(self, state: FeishuEventState) -> FeishuEventState:
        try:
            state["intent"] = await self._intent_recognition(state["content"])
            state["routing_key"] = self._route_by_intent(state["intent"])
            state["agent_response"] = await self._call_agent(state["routing_key"], state["content"])
            state["card_result"] = self._generate_card(state["agent_response"])
            return state

        except Exception as e:
            state["error"] = str(e)
            state["card_result"] = {
                "config": {"wide_screen_mode": True},
                "elements": [{
                    "tag": "div",
                    "text": {"content": f"处理消息时发生错误: {str(e)}", "tag": "lark_md"}
                }]
            }
            return state

    async def run_and_wait_async(self, state: FeishuEventState) -> FeishuEventState:
        try:
            state["intent"] = await self._intent_recognition(state["content"])
            state["routing_key"] = self._route_by_intent(state["intent"])
            state["agent_response"] = await self._call_agent(state["routing_key"], state["content"])
            state["card_result"] = self._generate_card(state["agent_response"])
            return state

        except Exception as e:
            state["error"] = str(e)
            state["card_result"] = {
                "config": {"wide_screen_mode": True},
                "elements": [{
                    "tag": "div",
                    "text": {"content": f"处理消息时发生错误: {str(e)}", "tag": "lark_md"}
                }]
            }
            return state
