"""
工具调用超时控制与降级兜底模块
实现统一的超时控制、降级策略和错误处理
"""
import asyncio
import logging
from typing import Any, Callable, Optional, Dict, List
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """超时错误"""
    pass

class FallbackStrategy:
    """降级策略枚举"""
    MOCK_DATA = "mock_data"  # 返回模拟数据
    CACHED_RESULT = "cached_result"  # 返回缓存结果
    ERROR_MESSAGE = "error_message"  # 返回错误信息
    RETRY = "retry"  # 重试

class ToolTimeoutManager:
    """工具调用超时管理器"""
    
    def __init__(self):
        self.default_timeout = 30.0  # 默认超时时间（秒）
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 1.0  # 重试延迟（秒）
        
        # 工具特定的超时配置
        self.tool_timeouts = {
            "sql_query": 10.0,
            "web_search": 15.0,
            "rag_search": 8.0,
            "document_parse": 60.0,
        }
        
        # 工具特定的降级策略
        self.tool_fallbacks = {
            "sql_query": FallbackStrategy.MOCK_DATA,
            "web_search": FallbackStrategy.MOCK_DATA,
            "rag_search": FallbackStrategy.ERROR_MESSAGE,
            "document_parse": FallbackStrategy.ERROR_MESSAGE,
        }
        
        # 缓存存储
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 缓存有效期（秒）
    
    def get_timeout(self, tool_name: str) -> float:
        """获取工具的超时时间"""
        return self.tool_timeouts.get(tool_name, self.default_timeout)
    
    def get_fallback_strategy(self, tool_name: str) -> FallbackStrategy:
        """获取工具的降级策略"""
        return self.tool_fallbacks.get(tool_name, FallbackStrategy.ERROR_MESSAGE)
    
    async def execute_with_timeout(
        self,
        tool_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """执行工具调用，带超时控制和降级兜底"""
        timeout = self.get_timeout(tool_name)
        fallback_strategy = self.get_fallback_strategy(tool_name)
        
        # 尝试执行，带重试机制
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                # 成功执行，缓存结果
                cache_key = self._get_cache_key(tool_name, args, kwargs)
                self._set_cache(cache_key, result)
                return result
                
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"工具 {tool_name} 执行超时（{timeout}s）")
                logger.warning(f"工具 {tool_name} 执行超时，尝试 {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    
            except Exception as e:
                last_error = e
                logger.error(f"工具 {tool_name} 执行失败: {e}")
                # 如果是可重试的错误，继续重试
                if self._is_retryable_error(e) and attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    break
        
        # 所有尝试都失败，执行降级策略
        logger.warning(f"工具 {tool_name} 所有尝试均失败，执行降级策略: {fallback_strategy}")
        return await self._execute_fallback(tool_name, fallback_strategy, last_error, args, kwargs)
    
    async def _execute_fallback(
        self,
        tool_name: str,
        strategy: FallbackStrategy,
        error: Optional[Exception],
        args: tuple,
        kwargs: dict
    ) -> Any:
        """执行降级策略"""
        if strategy == FallbackStrategy.MOCK_DATA:
            return self._get_mock_data(tool_name, args, kwargs)
        elif strategy == FallbackStrategy.CACHED_RESULT:
            cache_key = self._get_cache_key(tool_name, args, kwargs)
            cached = self._get_cache(cache_key)
            if cached:
                logger.info(f"使用缓存结果作为降级: {tool_name}")
                return cached
            else:
                logger.warning(f"无缓存可用，降级为错误信息: {tool_name}")
                return {"error": str(error), "success": False}
        elif strategy == FallbackStrategy.ERROR_MESSAGE:
            error_msg = f"工具 {tool_name} 执行失败: {str(error)}"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
        elif strategy == FallbackStrategy.RETRY:
            raise error if error else Exception("工具执行失败")
        else:
            return {"error": "未知降级策略", "success": False}
    
    def _get_mock_data(self, tool_name: str, args: tuple, kwargs: dict) -> Any:
        """获取模拟数据"""
        mock_data = {
            "sql_query": {
                "success": True,
                "data": [
                    {"id": 1, "name": "张三", "department": "技术部", "position": "工程师"},
                    {"id": 2, "name": "李四", "department": "产品部", "position": "产品经理"},
                    {"id": 3, "name": "王五", "department": "技术部", "position": "架构师"},
                ],
                "message": "返回模拟数据（真实数据库不可用）"
            },
            "web_search": {
                "success": True,
                "results": [
                    {"title": "示例搜索结果1", "snippet": "这是模拟的搜索结果内容..."},
                    {"title": "示例搜索结果2", "snippet": "这是模拟的搜索结果内容..."},
                    {"title": "示例搜索结果3", "snippet": "这是模拟的搜索结果内容..."},
                ],
                "message": "返回模拟数据（搜索服务不可用）"
            },
            "rag_search": {
                "success": True,
                "documents": [
                    {"content": "这是模拟的知识库文档内容...", "score": 0.85},
                    {"content": "这是模拟的知识库文档内容...", "score": 0.78},
                ],
                "message": "返回模拟数据（知识库不可用）"
            },
        }
        
        result = mock_data.get(tool_name, {"error": "无模拟数据", "success": False})
        logger.info(f"返回模拟数据: {tool_name}")
        return result
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """判断错误是否可重试"""
        retryable_errors = (
            TimeoutError,
            asyncio.TimeoutError,
            ConnectionError,
        )
        return isinstance(error, retryable_errors)
    
    def _get_cache_key(self, tool_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        import hashlib
        key_str = f"{tool_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _set_cache(self, key: str, data: Any) -> None:
        """设置缓存"""
        self.cache[key] = {
            "data": data,
            "timestamp": datetime.now().timestamp()
        }
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key in self.cache:
            cached = self.cache[key]
            if datetime.now().timestamp() - cached["timestamp"] < self.cache_ttl:
                return cached["data"]
            else:
                del self.cache[key]
        return None
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("缓存已清空")

def with_timeout(tool_name: str, timeout: Optional[float] = None):
    """装饰器：为函数添加超时控制"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = ToolTimeoutManager()
            actual_timeout = timeout or manager.get_timeout(tool_name)
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=actual_timeout)
            except asyncio.TimeoutError:
                logger.error(f"函数 {func.__name__} 执行超时（{actual_timeout}s）")
                raise TimeoutError(f"函数执行超时（{actual_timeout}s）")
        return wrapper
    return decorator

# 全局超时管理器实例
timeout_manager = ToolTimeoutManager()