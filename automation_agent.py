import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AutomationAgent:
    """飞书自动办公流程智能体"""
    
    def __init__(self, feishu_client, llm):
        self.feishu = feishu_client
        self.llm = llm
        self.report_tasks = []
    
    async def process_message(self, sender_id: str, message_id: str, text: str) -> Dict[str, Any]:
        """处理消息并触发自动化流程"""
        try:
            # 识别消息意图
            intent = await self._detect_intent(text)
            
            if intent == "create_task":
                return await self._handle_create_task(sender_id, text)
            elif intent == "report_summary":
                return await self._handle_report_summary(sender_id, text)
            elif intent == "query_data":
                return await self._handle_query_data(sender_id, text)
            else:
                return {"status": "success", "action": "none", "message": "未触发自动化流程"}
                
        except Exception as e:
            logger.error(f"自动化流程处理失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _detect_intent(self, text: str) -> str:
        """识别消息意图"""
        text = text.lower()
        
        if any(keyword in text for keyword in ["任务", "待办", "todo", "task", "帮我创建"]):
            return "create_task"
        elif any(keyword in text for keyword in ["日报", "周报", "总结", "汇总"]):
            return "report_summary"
        elif any(keyword in text for keyword in ["查询", "数据", "报表", "统计"]):
            return "query_data"
        
        return "none"
    
    async def _handle_create_task(self, sender_id: str, text: str) -> Dict[str, Any]:
        """处理创建任务请求"""
        try:
            # 使用LLM提取任务信息
            task_info = await self._extract_task_info(text)
            
            if not task_info.get("title"):
                return {
                    "status": "error", 
                    "message": "无法提取任务标题，请明确描述任务"
                }
            
            # 创建飞书任务
            task = await self.feishu.create_task(
                title=task_info["title"],
                description=task_info.get("description", ""),
                due_date=task_info.get("due_date"),
                assignee_id=task_info.get("assignee_id")
            )
            
            if task:
                # 回复确认消息
                reply_text = f"✅ 任务创建成功！\n"
                reply_text += f"📋 标题: {task_info['title']}\n"
                if task_info.get("description"):
                    reply_text += f"📝 描述: {task_info['description']}\n"
                if task_info.get("due_date"):
                    due_date = datetime.fromtimestamp(task_info["due_date"])
                    reply_text += f"⏰ 截止时间: {due_date.strftime('%Y-%m-%d %H:%M')}\n"
                
                await self.feishu.send_text_message(sender_id, reply_text)
                
                return {
                    "status": "success",
                    "action": "create_task",
                    "task": task
                }
            else:
                return {
                    "status": "error",
                    "message": "任务创建失败，请检查飞书API配置"
                }
                
        except Exception as e:
            logger.error(f"创建任务失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _extract_task_info(self, text: str) -> Dict[str, Any]:
        """使用LLM提取任务信息"""
        prompt = f"""
从以下消息中提取任务信息，返回JSON格式：
消息内容: {text}

需要提取的字段：
- title: 任务标题（必填）
- description: 任务描述（可选）
- due_date: 截止时间（时间戳，可选）
- assignee_id: 负责人用户ID（可选）

如果没有明确的截止时间，due_date为null
如果没有明确的负责人，assignee_id为null
只返回JSON，不要其他内容
"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            
            # 提取JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except Exception as e:
            logger.warning(f"LLM提取任务信息失败: {str(e)}")
        
        # 降级方案：简单提取
        title = text.strip()
        if len(title) > 100:
            title = title[:100] + "..."
        
        return {
            "title": title,
            "description": "",
            "due_date": None,
            "assignee_id": None
        }
    
    async def _handle_report_summary(self, sender_id: str, text: str) -> Dict[str, Any]:
        """处理日报/周报汇总"""
        try:
            # 保存报告内容
            report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_tasks.append({
                "sender_id": sender_id,
                "content": text,
                "time": report_time
            })
            
            # 创建汇总文档（简化版）
            doc = await self.feishu.create_docx(
                title=f"工作汇报汇总 - {report_time}"
            )
            
            reply_text = f"✅ 已收到您的工作汇报！\n"
            if doc:
                reply_text += f"📄 已保存到文档\n"
            
            await self.feishu.send_text_message(sender_id, reply_text)
            
            return {
                "status": "success",
                "action": "report_summary",
                "document": doc
            }
            
        except Exception as e:
            logger.error(f"处理汇报失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_query_data(self, sender_id: str, text: str) -> Dict[str, Any]:
        """处理数据查询请求"""
        # 这里需要对接多维表格，先返回基础响应
        reply_text = "🔍 数据查询功能正在开发中...\n"
        reply_text += "请配置多维表格app_token和table_id来使用此功能"
        
        await self.feishu.send_text_message(sender_id, reply_text)
        
        return {
            "status": "success",
            "action": "query_data",
            "message": "数据查询功能待完善"
        }
    
    async def auto_send_reminder(self) -> List[Dict[str, Any]]:
        """自动发送审批提醒"""
        try:
            # 获取待处理审批
            approvals = await self.feishu.get_approval_list()
            
            reminders = []
            for approval in approvals:
                # 检查是否超时
                create_time = approval.get("create_time", 0)
                if create_time:
                    create_datetime = datetime.fromtimestamp(create_time / 1000)
                    if datetime.now() - create_datetime > timedelta(days=1):
                        # 超时，发送提醒
                        approver_id = approval.get("approver_id")
                        if approver_id:
                            await self.feishu.send_text_message(
                                approver_id,
                                f"⏰ 审批超时提醒！\n审批单号: {approval.get('instance_code', '')}\n请及时处理"
                            )
                            reminders.append(approval)
            
            return reminders
            
        except Exception as e:
            logger.error(f"发送审批提醒失败: {str(e)}")
            return []
