"""
WebSocket 服务模块
实现实时双向通信，支持流式对话和实时状态更新
"""
import json
import asyncio
import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from collections import defaultdict

logger = logging.getLogger(__name__)

class WebSocketConnectionManager:
    """WebSocket 连接管理器"""
    
    def __init__(self):
        # 活跃连接：{user_id: {session_id: websocket}}
        self.active_connections: Dict[str, Dict[str, WebSocket]] = defaultdict(dict)
        
        # 连接元数据：{session_id: {user_id, connected_at, trace_id}}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # 广播订阅：{topic: {session_id}}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
    
    async def connect(self, websocket: WebSocket, user_id: str, session_id: str) -> bool:
        """建立 WebSocket 连接"""
        try:
            await websocket.accept()
            
            # 存储连接
            self.active_connections[user_id][session_id] = websocket
            
            # 存储元数据
            self.connection_metadata[session_id] = {
                "user_id": user_id,
                "connected_at": datetime.now().isoformat(),
                "trace_id": None,
                "status": "connected"
            }
            
            logger.info(f"WebSocket 连接建立: user={user_id}, session={session_id}")
            
            # 发送连接成功消息
            await self.send_to_session(session_id, {
                "type": "connection",
                "status": "connected",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"WebSocket 连接失败: {e}")
            return False
    
    async def disconnect(self, user_id: str, session_id: str):
        """断开 WebSocket 连接"""
        try:
            # 移除连接
            if user_id in self.active_connections:
                if session_id in self.active_connections[user_id]:
                    del self.active_connections[user_id][session_id]
            
            # 移除元数据
            if session_id in self.connection_metadata:
                del self.connection_metadata[session_id]
            
            # 移除订阅
            for topic in self.subscriptions:
                if session_id in self.subscriptions[topic]:
                    self.subscriptions[topic].remove(session_id)
            
            logger.info(f"WebSocket 连接断开: user={user_id}, session={session_id}")
            
        except Exception as e:
            logger.error(f"WebSocket 断开连接失败: {e}")
    
    async def send_to_session(self, session_id: str, message: Dict[str, Any]) -> bool:
        """向指定会话发送消息"""
        try:
            # 查找会话所属的用户
            user_id = None
            for uid, sessions in self.active_connections.items():
                if session_id in sessions:
                    user_id = uid
                    break
            
            if not user_id:
                logger.warning(f"会话不存在: {session_id}")
                return False
            
            websocket = self.active_connections[user_id].get(session_id)
            if not websocket:
                logger.warning(f"WebSocket 连接不存在: {session_id}")
                return False
            
            await websocket.send_json(message)
            return True
            
        except WebSocketDisconnect:
            logger.warning(f"WebSocket 已断开: {session_id}")
            await self.disconnect(user_id, session_id)
            return False
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]) -> int:
        """向用户的所有会话发送消息"""
        sent_count = 0
        if user_id in self.active_connections:
            for session_id, websocket in self.active_connections[user_id].items():
                try:
                    await websocket.send_json(message)
                    sent_count += 1
                except Exception as e:
                    logger.error(f"向用户 {user_id} 会话 {session_id} 发送消息失败: {e}")
        return sent_count
    
    async def broadcast(self, topic: str, message: Dict[str, Any]) -> int:
        """向订阅了指定主题的所有会话广播消息"""
        sent_count = 0
        if topic in self.subscriptions:
            for session_id in self.subscriptions[topic].copy():
                if await self.send_to_session(session_id, message):
                    sent_count += 1
        return sent_count
    
    def subscribe(self, session_id: str, topic: str):
        """订阅主题"""
        self.subscriptions[topic].add(session_id)
        logger.debug(f"会话 {session_id} 订阅主题: {topic}")
    
    def unsubscribe(self, session_id: str, topic: str):
        """取消订阅主题"""
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(session_id)
            logger.debug(f"会话 {session_id} 取消订阅主题: {topic}")
    
    def set_trace_id(self, session_id: str, trace_id: str):
        """设置会话的 TraceID"""
        if session_id in self.connection_metadata:
            self.connection_metadata[session_id]["trace_id"] = trace_id
    
    def get_trace_id(self, session_id: str) -> Optional[str]:
        """获取会话的 TraceID"""
        if session_id in self.connection_metadata:
            return self.connection_metadata[session_id].get("trace_id")
        return None
    
    def get_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取活跃会话列表"""
        sessions = []
        
        if user_id:
            # 获取指定用户的会话
            if user_id in self.active_connections:
                for session_id in self.active_connections[user_id]:
                    metadata = self.connection_metadata.get(session_id, {})
                    sessions.append({
                        "session_id": session_id,
                        "user_id": user_id,
                        "connected_at": metadata.get("connected_at"),
                        "trace_id": metadata.get("trace_id"),
                        "status": metadata.get("status")
                    })
        else:
            # 获取所有会话
            for session_id, metadata in self.connection_metadata.items():
                sessions.append({
                    "session_id": session_id,
                    "user_id": metadata.get("user_id"),
                    "connected_at": metadata.get("connected_at"),
                    "trace_id": metadata.get("trace_id"),
                    "status": metadata.get("status")
                })
        
        return sessions
    
    def get_connection_count(self) -> Dict[str, int]:
        """获取连接统计"""
        total_sessions = len(self.connection_metadata)
        total_users = len(self.active_connections)
        
        return {
            "total_sessions": total_sessions,
            "total_users": total_users,
            "subscriptions": {topic: len(sessions) for topic, sessions in self.subscriptions.items()}
        }

# 全局 WebSocket 管理器实例
websocket_manager = WebSocketConnectionManager()

class WebSocketMessageHandler:
    """WebSocket 消息处理器"""
    
    def __init__(self, manager: WebSocketConnectionManager):
        self.manager = manager
    
    async def handle_message(self, websocket: WebSocket, user_id: str, session_id: str, message: Dict[str, Any]):
        """处理接收到的消息"""
        message_type = message.get("type")
        
        if message_type == "ping":
            # 心跳检测
            await self.manager.send_to_session(session_id, {
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            })
        
        elif message_type == "subscribe":
            # 订阅主题
            topic = message.get("topic")
            if topic:
                self.manager.subscribe(session_id, topic)
                await self.manager.send_to_session(session_id, {
                    "type": "subscribed",
                    "topic": topic,
                    "timestamp": datetime.now().isoformat()
                })
        
        elif message_type == "unsubscribe":
            # 取消订阅
            topic = message.get("topic")
            if topic:
                self.manager.unsubscribe(session_id, topic)
                await self.manager.send_to_session(session_id, {
                    "type": "unsubscribed",
                    "topic": topic,
                    "timestamp": datetime.now().isoformat()
                })
        
        elif message_type == "chat":
            # 聊天消息
            await self._handle_chat_message(websocket, user_id, session_id, message)
        
        elif message_type == "status":
            # 状态查询
            await self._handle_status_message(session_id, message)
        
        else:
            # 未知消息类型
            await self.manager.send_to_session(session_id, {
                "type": "error",
                "error": f"未知消息类型: {message_type}",
                "timestamp": datetime.now().isoformat()
            })
    
    async def _handle_chat_message(self, websocket: WebSocket, user_id: str, session_id: str, message: Dict[str, Any]):
        """处理聊天消息"""
        content = message.get("content")
        if not content:
            await self.manager.send_to_session(session_id, {
                "type": "error",
                "error": "消息内容为空",
                "timestamp": datetime.now().isoformat()
            })
            return
        
        # 这里应该调用 Agent 处理聊天消息
        # 暂时返回模拟响应
        await self.manager.send_to_session(session_id, {
            "type": "chat_response",
            "content": f"收到消息: {content}",
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_status_message(self, session_id: str, message: Dict[str, Any]):
        """处理状态查询消息"""
        status_type = message.get("status_type", "connection")
        
        if status_type == "connection":
            # 连接状态
            metadata = self.manager.connection_metadata.get(session_id, {})
            await self.manager.send_to_session(session_id, {
                "type": "status_response",
                "status_type": "connection",
                "data": {
                    "session_id": session_id,
                    "user_id": metadata.get("user_id"),
                    "connected_at": metadata.get("connected_at"),
                    "trace_id": metadata.get("trace_id"),
                    "status": metadata.get("status")
                },
                "timestamp": datetime.now().isoformat()
            })
        
        elif status_type == "statistics":
            # 统计信息
            stats = self.manager.get_connection_count()
            await self.manager.send_to_session(session_id, {
                "type": "status_response",
                "status_type": "statistics",
                "data": stats,
                "timestamp": datetime.now().isoformat()
            })
        
        else:
            await self.manager.send_to_session(session_id, {
                "type": "error",
                "error": f"未知状态类型: {status_type}",
                "timestamp": datetime.now().isoformat()
            })

# 全局消息处理器实例
websocket_handler = WebSocketMessageHandler(websocket_manager)