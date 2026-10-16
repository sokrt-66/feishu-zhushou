import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MeetingAgent:
    """飞书会议智能体 - 集成真实飞书会议系统"""
    
    def __init__(self, feishu_client, llm, db_connector=None, permission_manager=None):
        self.feishu = feishu_client
        self.llm = llm
        self.db_connector = db_connector
        self.permission_manager = permission_manager
        
        # 内存缓存
        self.meeting_transcripts = {}
        self.user_name_to_id_cache = {}
        
        # 注册飞书事件监听
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """注册飞书会议事件处理器"""
        # 监听会议结束事件
        if hasattr(self.feishu, 'message_handlers'):
            self.feishu.message_handlers['meeting_ended'] = self._handle_meeting_ended
            self.feishu.message_handlers['meeting_record_ready'] = self._handle_record_ready
    
    async def _handle_meeting_ended(self, event_data: Dict[str, Any]):
        """处理会议结束事件"""
        try:
            meeting_id = event_data.get("event", {}).get("meeting_id")
            if not meeting_id:
                return
            
            logger.info(f"检测到会议结束: {meeting_id}")
            
            # 获取会议信息和参与者
            meeting_info = await self.feishu.get_meeting_info(meeting_id)
            participants = await self.feishu.get_meeting_participants(meeting_id)
            
            if participants:
                participant_ids = [p.get("user_id") for p in participants if p.get("user_id")]
            else:
                participant_ids = []
            
            # 等待录制完成（需要一定时间）
            import asyncio
            await asyncio.sleep(30)  # 等待30秒让录制完成
            
            # 处理会议转录
            await self.process_meeting_transcript(meeting_id, "", participant_ids)
            
        except Exception as e:
            logger.error(f"处理会议结束事件失败: {str(e)}")
    
    async def _handle_record_ready(self, event_data: Dict[str, Any]):
        """处理会议录制完成事件"""
        try:
            meeting_id = event_data.get("event", {}).get("meeting_id")
            record_id = event_data.get("event", {}).get("record_id")
            
            logger.info(f"会议录制完成: {meeting_id}, 录制ID: {record_id}")
            
            # 获取会议参与者
            participants = await self.feishu.get_meeting_participants(meeting_id)
            participant_ids = [p.get("user_id") for p in participants if p.get("user_id")]
            
            # 处理会议转录
            await self.process_meeting_transcript(meeting_id, "", participant_ids)
            
        except Exception as e:
            logger.error(f"处理录制完成事件失败: {str(e)}")
    
    async def _init_user_cache(self):
        """初始化用户姓名到ID的映射缓存"""
        if self.permission_manager:
            for username in self.permission_manager.user_departments.keys():
                try:
                    user_info = await self.feishu.get_user_info(username)
                    if user_info:
                        user_id = user_info.get("user_id") or user_info.get("id")
                        if user_id:
                            self.user_name_to_id_cache[username] = user_id
                            name = user_info.get("name") or user_info.get("display_name")
                            if name:
                                self.user_name_to_id_cache[name] = user_id
                except Exception as e:
                    logger.debug(f"获取用户 {username} 信息失败: {e}")
    
    async def _get_user_id_by_name(self, name: str) -> Optional[str]:
        """根据姓名获取用户ID"""
        if name in self.user_name_to_id_cache:
            return self.user_name_to_id_cache[name]
        
        try:
            user_info = await self.feishu.search_user_by_name(name)
            if user_info:
                user_id = user_info.get("user_id") or user_info.get("id")
                if user_id:
                    self.user_name_to_id_cache[name] = user_id
                    return user_id
        except Exception as e:
            logger.debug(f"搜索用户 {name} 失败: {e}")
        
        if self.permission_manager and name in self.permission_manager.user_departments:
            return name
        
        return None
    
    async def _save_meeting_to_db(self, meeting_id: str, data: Dict[str, Any]) -> bool:
        """保存会议数据到数据库"""
        if not self.db_connector:
            return False
        
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS meeting_transcripts (
                meeting_id VARCHAR(64) PRIMARY KEY,
                transcript TEXT,
                participants TEXT,
                minutes TEXT,
                todos TEXT,
                created_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            await self.db_connector.execute_update(create_table_sql)
            
            participants_json = json.dumps(data.get("participants", []))
            minutes_json = json.dumps(data.get("minutes", {}))
            todos_json = json.dumps(data.get("todos", []))
            
            upsert_sql = f"""
            INSERT OR REPLACE INTO meeting_transcripts 
            (meeting_id, transcript, participants, minutes, todos, created_time)
            VALUES ('{meeting_id}', '{data.get("transcript", "")}', 
                    '{participants_json}', '{minutes_json}', '{todos_json}', 
                    '{data.get("created_time", datetime.now().isoformat())}')
            """
            result = await self.db_connector.execute_update(upsert_sql)
            return result.get("success", False)
        except Exception as e:
            logger.error(f"保存会议到数据库失败: {e}")
            return False
    
    async def _load_meeting_from_db(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """从数据库加载会议数据"""
        if not self.db_connector:
            return None
        
        try:
            query_sql = f"SELECT * FROM meeting_transcripts WHERE meeting_id = '{meeting_id}'"
            result = await self.db_connector.execute_query(query_sql)
            
            if result.get("success") and result.get("data"):
                row = result["data"][0]
                return {
                    "meeting_id": row["meeting_id"],
                    "transcript": row["transcript"],
                    "participants": json.loads(row["participants"]) if row["participants"] else [],
                    "minutes": json.loads(row["minutes"]) if row["minutes"] else {},
                    "todos": json.loads(row["todos"]) if row["todos"] else [],
                    "created_time": row["created_time"]
                }
        except Exception as e:
            logger.error(f"从数据库加载会议失败: {e}")
        
        return None
    
    async def process_meeting_transcript(self, meeting_id: str, transcript: str = "", 
                                          participants: List[str] = None) -> Dict[str, Any]:
        """处理会议转录内容（集成真实飞书会议）"""
        try:
            if not self.user_name_to_id_cache:
                await self._init_user_cache()
            
            participants = participants or []
            
            # 如果没有传入转录内容，从飞书获取
            if not transcript:
                transcript = await self.feishu.get_meeting_transcript(meeting_id)
                if not transcript:
                    logger.warning(f"无法获取会议 {meeting_id} 的转录内容")
            
            # 获取会议详情用于标题
            meeting_info = await self.feishu.get_meeting_info(meeting_id)
            meeting_topic = meeting_info.get("topic", "") if meeting_info else ""
            
            # 保存到内存
            self.meeting_transcripts[meeting_id] = {
                "transcript": transcript,
                "participants": participants,
                "topic": meeting_topic,
                "created_time": datetime.now().isoformat()
            }
            
            # 异步生成会议纪要
            minutes = await self._generate_meeting_minutes(meeting_id, transcript, meeting_topic)
            
            # 异步提取待办事项
            todos = await self._extract_action_items(meeting_id, transcript)
            
            # 将负责人姓名映射为用户ID
            for todo in todos:
                if todo.get("assignee") and not todo.get("assignee_id"):
                    assignee_name = todo["assignee"]
                    assignee_id = await self._get_user_id_by_name(assignee_name)
                    if assignee_id:
                        todo["assignee_id"] = assignee_id
                    else:
                        logger.warning(f"无法找到负责人 '{assignee_name}' 的用户ID")
            
            # 异步发送纪要给参与者
            send_tasks = []
            for participant in participants:
                send_tasks.append(self._send_meeting_summary(participant, minutes, todos))
            
            if send_tasks:
                import asyncio
                await asyncio.gather(*send_tasks, return_exceptions=True)
            
            # 异步创建飞书任务
            if todos:
                create_tasks = []
                for todo in todos:
                    if todo.get("assignee_id"):
                        create_tasks.append(self.feishu.create_task(
                            title=todo.get("title", "会议待办"),
                            description=todo.get("description", ""),
                            assignee_id=todo.get("assignee_id")
                        ))
                
                if create_tasks:
                    import asyncio
                    await asyncio.gather(*create_tasks, return_exceptions=True)
            
            # 保存到数据库
            await self._save_meeting_to_db(meeting_id, {
                "transcript": transcript,
                "participants": participants,
                "minutes": minutes,
                "todos": todos,
                "created_time": datetime.now().isoformat()
            })
            
            logger.info(f"会议纪要处理完成: {meeting_id}")
            
            return {
                "status": "success",
                "meeting_id": meeting_id,
                "topic": meeting_topic,
                "minutes": minutes,
                "todos": todos
            }
            
        except Exception as e:
            logger.error(f"处理会议转录失败: {str(e)}", exc_info=True)
            return {"status": "error", "message": str(e)}
    
    async def _generate_meeting_minutes(self, meeting_id: str, transcript: str, topic: str = "") -> Dict[str, Any]:
        """生成会议纪要（异步版本）"""
        prompt = f"""
基于以下会议转录内容，生成结构化会议纪要，返回JSON格式：

会议主题: {topic}

会议转录:
{transcript}

需要生成以下字段：
- title: 会议标题（如果没有主题则自动生成）
- date: 会议日期 (YYYY-MM-DD)
- summary: 会议要点总结 (200字以内)
- key_points: 关键要点列表
- decisions: 会议决议列表
- next_steps: 下一步计划

只返回JSON，不要其他内容
"""
        
        try:
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(prompt)
            else:
                import asyncio
                response = await asyncio.to_thread(self.llm.invoke, prompt)
            
            content = response.content if hasattr(response, "content") else str(response)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # 如果没有标题，使用传入的主题
                if not result.get("title") and topic:
                    result["title"] = topic
                return result
        except Exception as e:
            logger.warning(f"生成会议纪要失败: {str(e)}")
        
        return {
            "title": topic or "会议纪要",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": "会议已完成",
            "key_points": [],
            "decisions": [],
            "next_steps": []
        }
    
    async def _extract_action_items(self, meeting_id: str, transcript: str) -> List[Dict[str, Any]]:
        """提取待办事项（异步版本）"""
        prompt = f"""
从以下会议转录中提取待办事项，返回JSON数组：

会议转录:
{transcript}

每个待办事项包含：
- title: 任务标题
- description: 任务描述
- assignee: 负责人姓名（如果有）
- deadline: 截止时间（如果有，YYYY-MM-DD格式）

只返回JSON数组，不要其他内容
"""
        
        try:
            if hasattr(self.llm, 'ainvoke'):
                response = await self.llm.ainvoke(prompt)
            else:
                import asyncio
                response = await asyncio.to_thread(self.llm.invoke, prompt)
            
            content = response.content if hasattr(response, "content") else str(response)
            
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except Exception as e:
            logger.warning(f"提取待办事项失败: {str(e)}")
        
        return []
    
    async def _send_meeting_summary(self, user_id: str, minutes: Dict[str, Any], 
                                     todos: List[Dict[str, Any]]):
        """发送会议摘要给用户"""
        summary_text = f"📋 {minutes.get('title', '会议纪要')}\n\n"
        summary_text += f"📅 日期: {minutes.get('date', '')}\n"
        summary_text += f"📝 要点: {minutes.get('summary', '')}\n\n"
        
        if minutes.get("key_points"):
            summary_text += "🔑 关键要点:\n"
            for point in minutes["key_points"]:
                summary_text += f"  • {point}\n"
            summary_text += "\n"
        
        if minutes.get("decisions"):
            summary_text += "✅ 会议决议:\n"
            for decision in minutes["decisions"]:
                summary_text += f"  • {decision}\n"
            summary_text += "\n"
        
        if todos:
            summary_text += "📋 待办事项:\n"
            for i, todo in enumerate(todos, 1):
                summary_text += f"  {i}. {todo.get('title', '')}"
                if todo.get("assignee"):
                    summary_text += f" (负责人: {todo.get('assignee')})"
                if todo.get("deadline"):
                    summary_text += f" (截止: {todo.get('deadline')})"
                summary_text += "\n"
        
        await self.feishu.send_text_message(user_id, summary_text)
    
    async def schedule_meeting(self, topic: str, start_time: str, end_time: str, 
                               attendees: List[str], description: str = "") -> Dict[str, Any]:
        """创建飞书会议（真实会议）"""
        try:
            # 创建会议
            result = await self.feishu.create_meeting(
                topic=topic,
                start_time=start_time,
                end_time=end_time,
                attendees=attendees,
                description=description
            )
            
            if result:
                meeting_id = result.get("meeting_id")
                logger.info(f"会议创建成功: {meeting_id}")
                
                return {
                    "status": "success",
                    "meeting_id": meeting_id,
                    "topic": topic,
                    "start_time": start_time,
                    "end_time": end_time,
                    "attendees": attendees,
                    "join_url": result.get("join_url", "")
                }
            else:
                return {"status": "error", "message": "创建会议失败"}
        
        except Exception as e:
            logger.error(f"创建会议失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def query_meeting_history(self, user_id: str, query: str = "", 
                                    days: int = 30) -> List[Dict[str, Any]]:
        """查询会议历史记录"""
        if self.db_connector:
            db_meetings = await self._load_all_meetings_from_db()
            self.meeting_transcripts.update(db_meetings)
        
        cutoff_time = datetime.now() - timedelta(days=days)
        results = []
        
        for meeting_id, data in self.meeting_transcripts.items():
            created_time_str = data.get("created_time")
            if created_time_str:
                try:
                    created_time = datetime.fromisoformat(created_time_str.replace("Z", "+00:00"))
                    if created_time >= cutoff_time:
                        participants = data.get("participants", [])
                        if user_id in participants:
                            results.append({
                                "meeting_id": meeting_id,
                                "topic": data.get("topic", ""),
                                "created_time": data.get("created_time"),
                                "participants": participants,
                                "preview": data.get("transcript", "")[:100]
                            })
                except Exception as e:
                    logger.debug(f"解析会议时间失败: {e}")
        
        results.sort(key=lambda x: x.get("created_time", ""), reverse=True)
        return results
    
    async def _load_all_meetings_from_db(self) -> Dict[str, Any]:
        """从数据库加载所有会议数据"""
        if not self.db_connector:
            return {}
        
        try:
            query_sql = "SELECT * FROM meeting_transcripts"
            result = await self.db_connector.execute_query(query_sql)
            
            if result.get("success"):
                meetings = {}
                for row in result["data"]:
                    meeting_id = row["meeting_id"]
                    meetings[meeting_id] = {
                        "transcript": row["transcript"],
                        "participants": json.loads(row["participants"]) if row["participants"] else [],
                        "created_time": row["created_time"]
                    }
                return meetings
        except Exception as e:
            logger.error(f"从数据库加载所有会议失败: {e}")
        
        return {}