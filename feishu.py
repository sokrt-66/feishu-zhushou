import httpx
import json
import hashlib
import base64
from typing import Optional, Dict, Any, Callable, List
import logging
import time
import asyncio
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on_codes: tuple = (429, 500, 502, 503, 504)
):
    """指数退避重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    last_exception = e
                    if e.response.status_code in retry_on_codes:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"请求失败 (attempt {attempt + 1}/{max_retries + 1}), 状态码: {e.response.status_code}, {delay:.1f}秒后重试")
                        await asyncio.sleep(delay)
                    else:
                        raise
                except httpx.ConnectError as e:
                    last_exception = e
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"连接失败 (attempt {attempt + 1}/{max_retries + 1}), {delay:.1f}秒后重试")
                    await asyncio.sleep(delay)
            if last_exception:
                raise last_exception
        return wrapper
    return decorator


class FeishuClient:
    def __init__(self, app_id: str, app_secret: str, verification_token: str, encrypt_key: str = ""):
        self.app_id = app_id
        self.app_secret = app_secret
        self.verification_token = verification_token
        self.encrypt_key = encrypt_key
        self.access_token = None
        self.token_expire_time = 0

        self.message_handlers: Dict[str, Callable] = {}
        self.file_handlers: Dict[str, Callable] = {}

        self._http_client: Optional[httpx.AsyncClient] = None
        self._user_open_ids: Dict[str, str] = {}

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def _get_access_token(self) -> Optional[str]:
        if self.access_token and time.time() < self.token_expire_time:
            return self.access_token

        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        data = {
            "app_id": self.app_id,
            "app_secret": self.app_secret
        }

        try:
            client = await self._get_http_client()
            resp = await client.post(url, json=data)
            result = resp.json()
            if result.get("code") == 0:
                self.access_token = result.get("tenant_access_token")
                self.token_expire_time = time.time() + result.get("expire", 7200) - 60
                return self.access_token
            else:
                logger.error(f"获取token失败: {result}")
                return None
        except Exception as e:
            logger.error(f"获取token异常: {str(e)}")
            return None

    async def _handle_message_event(self, event_data: dict):
        try:
            message = event_data.get("event", {}).get("message", {})
            sender = event_data.get("event", {}).get("sender", {})

            message_id = message.get("message_id")
            message_type = message.get("message_type")
            content = message.get("content", "{}")
            sender_id = sender.get("sender_id", {}).get("user_id", "")

            self._user_open_ids[sender_id] = sender_id

            try:
                content_dict = json.loads(content)
            except:
                content_dict = {"text": content}

            logger.info(f"收到消息: type={message_type}, sender={sender_id}, content={content}")

            if message_type == "text":
                text = content_dict.get("text", "")
                await self._dispatch_text_message(sender_id, message_id, text)
            elif message_type == "file":
                file_key = content_dict.get("file_key", "")
                file_name = content_dict.get("file_name", "")
                await self._dispatch_file_message(sender_id, message_id, file_key, file_name)

        except Exception as e:
            logger.error(f"处理消息事件失败: {str(e)}")

    async def _dispatch_text_message(self, sender_id: str, message_id: str, text: str):
        for pattern, handler in self.message_handlers.items():
            if pattern == "*" or (callable(pattern) and pattern(text)) or (isinstance(pattern, str) and pattern in text):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(sender_id, message_id, text)
                    else:
                        await asyncio.to_thread(handler, sender_id, message_id, text)
                except Exception as e:
                    logger.error(f"消息处理器执行失败: {str(e)}")

    async def _dispatch_file_message(self, sender_id: str, message_id: str, file_key: str, file_name: str):
        for handler in self.file_handlers.values():
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(sender_id, message_id, file_key, file_name)
                else:
                    await asyncio.to_thread(handler, sender_id, message_id, file_key, file_name)
            except Exception as e:
                logger.error(f"文件处理器执行失败: {str(e)}")

    def register_message_handler(self, pattern: str, handler: Callable):
        self.message_handlers[pattern] = handler

    def register_file_handler(self, handler: Callable):
        self.file_handlers["default"] = handler

    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0)
    async def _post_with_retry(self, url: str, headers: Dict[str, str], json_data: Dict[str, Any]) -> Dict[str, Any]:
        """带指数退避重试的 POST 请求"""
        client = await self._get_http_client()
        resp = await client.post(url, headers=headers, json=json_data)
        result = resp.json()
        if result.get("code") != 0:
            error_code = result.get("code", 0)
            if error_code in (99991663, 99991664):
                self.access_token = None
                token = await self._get_access_token()
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                    resp = await client.post(url, headers=headers, json=json_data)
                    result = resp.json()
        return result

    async def send_text_message(self, user_id: str, text: str) -> bool:
        try:
            token = await self._get_access_token()
            if not token:
                return False

            url = "https://open.feishu.cn/open-apis/im/v1/messages"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            data = {
                "receive_id_type": "user_id",
                "receive_id": user_id,
                "content": json.dumps({"text": text}),
                "msg_type": "text"
            }

            result = await self._post_with_retry(url, headers, data)
            return result.get("code") == 0
        except Exception as e:
            logger.error(f"发送文本消息失败: {str(e)}")
            return False

    async def send_card_message(self, user_id: str, card: Dict[str, Any]) -> bool:
        try:
            token = await self._get_access_token()
            if not token:
                return False

            url = "https://open.feishu.cn/open-apis/im/v1/messages"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            data = {
                "receive_id_type": "user_id",
                "receive_id": user_id,
                "content": json.dumps(card),
                "msg_type": "interactive"
            }

            result = await self._post_with_retry(url, headers, data)
            return result.get("code") == 0
        except Exception as e:
            logger.error(f"发送卡片消息失败: {str(e)}")
            return False

    async def get_file_content(self, file_key: str) -> Optional[bytes]:
        try:
            token = await self._get_access_token()
            if not token:
                return None

            url = f"https://open.feishu.cn/open-apis/drive/v1/files/{file_key}/download"
            headers = {"Authorization": f"Bearer {token}"}

            client = await self._get_http_client()
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.content
            else:
                logger.error(f"获取文件失败: {resp.text}")
                return None
        except Exception as e:
            logger.error(f"获取文件内容失败: {str(e)}")
            return None

    def verify_request(self, headers: Dict[str, str], body: str) -> bool:
        try:
            signature = headers.get("X-Lark-Signature", "") or headers.get("x-lark-signature", "")
            timestamp = headers.get("X-Lark-Request-Timestamp", "") or headers.get("x-lark-request-timestamp", "")
            nonce = headers.get("X-Lark-Request-Nonce", "") or headers.get("x-lark-request-nonce", "")

            if not signature or not timestamp or not nonce:
                return False

            sign_str = timestamp + nonce + self.verification_token + body
            hash_obj = hashlib.sha256(sign_str.encode("utf-8"))
            expected_signature = base64.b64encode(hash_obj.digest()).decode("utf-8")

            return signature == expected_signature
        except Exception as e:
            logger.error(f"验证请求失败: {str(e)}")
            return False

    async def create_task(self, title: str, description: str = "",
                    due_date: Optional[int] = None, assignee_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            token = await self._get_access_token()
            if not token:
                return None

            url = "https://open.feishu.cn/open-apis/task/v2/tasks"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }

            task_data = {
                "summary": title,
                "description": description,
                "tasklist_id": "0"
            }

            if due_date:
                task_data["due_time"] = due_date

            if assignee_id:
                task_data["assignee_id"] = assignee_id

            client = await self._get_http_client()
            resp = await client.post(url, headers=headers, json=task_data)
            result = resp.json()

            if result.get("code") == 0:
                return result.get("data", {}).get("task")
            else:
                logger.error(f"创建任务失败: {result}")
                return None
        except Exception as e:
            logger.error(f"创建任务异常: {str(e)}")
            return None

    async def get_tasks(self) -> List[Dict[str, Any]]:
        try:
            token = await self._get_access_token()
            if not token:
                return []

            url = "https://open.feishu.cn/open-apis/task/v2/tasks"
            headers = {"Authorization": f"Bearer {token}"}

            client = await self._get_http_client()
            resp = await client.get(url, headers=headers)
            result = resp.json()

            if result.get("code") == 0:
                return result.get("data", {}).get("items", [])
            else:
                logger.error(f"获取任务失败: {result}")
                return []
        except Exception as e:
            logger.error(f"获取任务异常: {str(e)}")
            return []

    async def get_bitable_records(self, app_token: str, table_id: str) -> List[Dict[str, Any]]:
        try:
            token = await self._get_access_token()
            if not token:
                return []

            url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
            headers = {"Authorization": f"Bearer {token}"}

            client = await self._get_http_client()
            resp = await client.get(url, headers=headers)
            result = resp.json()

            if result.get("code") == 0:
                return result.get("data", {}).get("items", [])
            else:
                logger.error(f"获取多维表格记录失败: {result}")
                return []
        except Exception as e:
            logger.error(f"获取多维表格记录异常: {str(e)}")
            return []

    async def create_bitable_record(self, app_token: str, table_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            token = await self._get_access_token()
            if not token:
                return None

            url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            data = {"fields": fields}

            client = await self._get_http_client()
            resp = await client.post(url, headers=headers, json=data)
            result = resp.json()

            if result.get("code") == 0:
                return result.get("data", {}).get("record")
            else:
                logger.error(f"创建多维表格记录失败: {result}")
                return None
        except Exception as e:
            logger.error(f"创建多维表格记录异常: {str(e)}")
            return None

    async def get_approval_list(self) -> List[Dict[str, Any]]:
        try:
            token = await self._get_access_token()
            if not token:
                return []

            url = "https://open.feishu.cn/open-apis/approval/v4/instances"
            headers = {"Authorization": f"Bearer {token}"}

            client = await self._get_http_client()
            resp = await client.get(url, headers=headers)
            result = resp.json()

            if result.get("code") == 0:
                return result.get("data", {}).get("data", [])
            else:
                logger.error(f"获取审批列表失败: {result}")
                return []
        except Exception as e:
            logger.error(f"获取审批列表异常: {str(e)}")
            return []

    async def create_docx(self, title: str, content: str = "") -> Optional[Dict[str, Any]]:
        try:
            token = await self._get_access_token()
            if not token:
                return None

            url = "https://open.feishu.cn/open-apis/docx/v1/documents"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            data = {"title": title}

            client = await self._get_http_client()
            resp = await client.post(url, headers=headers, json=data)
            result = resp.json()

            if result.get("code") == 0:
                return result.get("data", {}).get("document")
            else:
                logger.error(f"创建文档失败: {result}")
                return None
        except Exception as e:
            logger.error(f"创建文档异常: {str(e)}")
            return None

    async def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        try:
            token = await self._get_access_token()
            if not token:
                return None

            url = f"https://open.feishu.cn/open-apis/contact/v3/users/{user_id}"
            headers = {"Authorization": f"Bearer {token}"}

            client = await self._get_http_client()
            resp = await client.get(url, headers=headers)
            result = resp.json()

            if result.get("code") == 0:
                return result.get("data", {}).get("user")
            else:
                logger.error(f"获取用户信息失败: {result}")
                return None
        except Exception as e:
            logger.error(f"获取用户信息异常: {str(e)}")
            return None

    # ==================== 会议相关 API ====================

    async def create_meeting(self, topic: str, start_time: str, end_time: str, 
                             attendees: List[str] = None, description: str = "") -> Optional[Dict[str, Any]]:
        """创建飞书会议"""
        try:
            token = await self._get_access_token()
            if not token:
                return None

            url = "https://open.feishu.cn/open-apis/vc/v1/meetings"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

            data = {
                "topic": topic,
                "start_time": start_time,
                "end_time": end_time,
                "description": description,
                "attendees": [{"id": attendee, "type": "user_id"} for attendee in (attendees or [])],
                "settings": {
                    "allow_join_before_host": True,
                    "auto_record": True,
                    "record_quality": "high"
                }
            }

            client = await self._get_http_client()
            resp = await client.post(url, headers=headers, json=data)
            result = resp.json()

            if result.get("code") == 0:
                logger.info(f"会议创建成功: {result}")
                return result.get("data", {})
            else:
                logger.error(f"创建会议失败: {result}")
                return None
        except Exception as e:
            logger.error(f"创建会议异常: {str(e)}")
            return None

    async def get_meeting_info(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """获取会议详情"""
        try:
            token = await self._get_access_token()
            if not token:
                return None

            url = f"https://open.feishu.cn/open-apis/vc/v1/meetings/{meeting_id}"
            headers = {"Authorization": f"Bearer {token}"}

            client = await self._get_http_client()
            resp = await client.get(url, headers=headers)
            result = resp.json()

            if result.get("code") == 0:
                return result.get("data", {})
            else:
                logger.error(f"获取会议信息失败: {result}")
                return None
        except Exception as e:
            logger.error(f"获取会议信息异常: {str(e)}")
            return None

    async def get_meeting_transcript(self, meeting_id: str) -> Optional[str]:
        """获取会议转录内容（需要会议已开启录制）"""
        try:
            token = await self._get_access_token()
            if not token:
                return None

            url = f"https://open.feishu.cn/open-apis/vc/v1/meetings/{meeting_id}/records"
            headers = {"Authorization": f"Bearer {token}"}

            client = await self._get_http_client()
            resp = await client.get(url, headers=headers)
            result = resp.json()

            if result.get("code") == 0:
                records = result.get("data", {}).get("records", [])
                if records:
                    # 获取最新的录制记录
                    latest_record = sorted(records, key=lambda x: x.get("create_time", 0), reverse=True)[0]
                    record_id = latest_record.get("id")
                    
                    # 获取转录内容
                    transcript_url = f"https://open.feishu.cn/open-apis/vc/v1/records/{record_id}/transcript"
                    resp = await client.get(transcript_url, headers=headers)
                    transcript_result = resp.json()
                    
                    if transcript_result.get("code") == 0:
                        return transcript_result.get("data", {}).get("content", "")
            else:
                logger.error(f"获取会议录制失败: {result}")
                return None
        except Exception as e:
            logger.error(f"获取会议转录异常: {str(e)}")
            return None

    async def get_meeting_participants(self, meeting_id: str) -> Optional[List[Dict[str, Any]]]:
        """获取会议参与者列表"""
        try:
            token = await self._get_access_token()
            if not token:
                return None

            url = f"https://open.feishu.cn/open-apis/vc/v1/meetings/{meeting_id}/participants"
            headers = {"Authorization": f"Bearer {token}"}

            client = await self._get_http_client()
            resp = await client.get(url, headers=headers)
            result = resp.json()

            if result.get("code") == 0:
                return result.get("data", {}).get("participants", [])
            else:
                logger.error(f"获取会议参与者失败: {result}")
                return None
        except Exception as e:
            logger.error(f"获取会议参与者异常: {str(e)}")
            return None

    async def search_user_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """根据姓名搜索用户"""
        try:
            token = await self._get_access_token()
            if not token:
                return None

            url = "https://open.feishu.cn/open-apis/contact/v3/users/search"
            headers = {"Authorization": f"Bearer {token}"}
            params = {"query": name, "page_size": 1}

            client = await self._get_http_client()
            resp = await client.get(url, headers=headers, params=params)
            result = resp.json()

            if result.get("code") == 0:
                users = result.get("data", {}).get("items", [])
                if users:
                    return users[0]
            else:
                logger.error(f"搜索用户失败: {result}")
                return None
        except Exception as e:
            logger.error(f"搜索用户异常: {str(e)}")
            return None
