"""
全链路 TraceID 模块
实现请求追踪、日志关联和性能监控
"""
import uuid
import logging
import time
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger(__name__)

class TraceContext:
    """追踪上下文"""
    
    def __init__(self, trace_id: str, parent_span_id: Optional[str] = None):
        self.trace_id = trace_id
        self.span_id = self._generate_span_id()
        self.parent_span_id = parent_span_id
        self.start_time = time.time()
        self.metadata: Dict[str, Any] = {}
        self.tags: Dict[str, str] = {}
        self.events: List[Dict[str, Any]] = []
        self.children: List['TraceContext'] = []
    
    def _generate_span_id(self) -> str:
        """生成 Span ID"""
        return uuid.uuid4().hex[:16]
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """添加事件"""
        self.events.append({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "attributes": attributes or {}
        })
    
    def add_tag(self, key: str, value: str):
        """添加标签"""
        self.tags[key] = value
    
    def set_metadata(self, key: str, value: Any):
        """设置元数据"""
        self.metadata[key] = value
    
    def get_duration(self) -> float:
        """获取持续时间"""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "duration": self.get_duration(),
            "metadata": self.metadata,
            "tags": self.tags,
            "events": self.events,
            "children_count": len(self.children)
        }

class TraceManager:
    """追踪管理器"""
    
    def __init__(self):
        # 线程本地存储
        self._local = threading.local()
        
        # 追踪存储
        self._traces: Dict[str, TraceContext] = {}
        
        # 启用标志
        self._enabled = True
    
    @property
    def current_trace(self) -> Optional[TraceContext]:
        """获取当前追踪上下文"""
        return getattr(self._local, 'trace', None)
    
    def generate_trace_id(self) -> str:
        """生成 Trace ID"""
        return f"trace_{uuid.uuid4().hex}"
    
    def start_trace(self, trace_id: Optional[str] = None, parent_span_id: Optional[str] = None) -> TraceContext:
        """开始追踪"""
        if not self._enabled:
            return None
        
        trace_id = trace_id or self.generate_trace_id()
        trace = TraceContext(trace_id, parent_span_id)
        
        self._local.trace = trace
        self._traces[trace_id] = trace
        
        logger.info(f"开始追踪: trace_id={trace_id}, span_id={trace.span_id}")
        
        return trace
    
    def end_trace(self, trace_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """结束追踪"""
        if not self._enabled:
            return None
        
        trace = self.current_trace
        if not trace:
            return None
        
        trace_data = trace.to_dict()
        
        logger.info(f"结束追踪: trace_id={trace.trace_id}, span_id={trace.span_id}, duration={trace_data['duration']:.2f}s")
        
        # 清理
        if trace.trace_id in self._traces:
            del self._traces[trace.trace_id]
        self._local.trace = None
        
        return trace_data
    
    def get_trace(self, trace_id: str) -> Optional[TraceContext]:
        """获取追踪上下文"""
        return self._traces.get(trace_id)
    
    def get_current_trace_id(self) -> Optional[str]:
        """获取当前 Trace ID"""
        trace = self.current_trace
        return trace.trace_id if trace else None
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """添加事件到当前追踪"""
        trace = self.current_trace
        if trace:
            trace.add_event(name, attributes)
    
    def add_tag(self, key: str, value: str):
        """添加标签到当前追踪"""
        trace = self.current_trace
        if trace:
            trace.add_tag(key, value)
    
    def set_metadata(self, key: str, value: Any):
        """设置元数据到当前追踪"""
        trace = self.current_trace
        if trace:
            trace.set_metadata(key, value)
    
    def enable(self):
        """启用追踪"""
        self._enabled = True
        logger.info("追踪已启用")
    
    def disable(self):
        """禁用追踪"""
        self._enabled = False
        logger.info("追踪已禁用")
    
    def is_enabled(self) -> bool:
        """检查追踪是否启用"""
        return self._enabled
    
    def get_active_traces(self) -> List[Dict[str, Any]]:
        """获取活跃追踪列表"""
        return [trace.to_dict() for trace in self._traces.values()]
    
    def clear_traces(self):
        """清空所有追踪"""
        self._traces.clear()
        logger.info("所有追踪已清空")

# 全局追踪管理器实例
trace_manager = TraceManager()

def traced(operation_name: str):
    """装饰器：为函数添加追踪"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace = trace_manager.current_trace
            if not trace:
                return await func(*args, **kwargs)
            
            # 创建子 span
            child_trace = TraceContext(trace.trace_id, trace.span_id)
            child_trace.add_tag("operation", operation_name)
            child_trace.add_tag("function", func.__name__)
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                child_trace.add_tag("status", "success")
                child_trace.add_event("operation_completed", {"duration": time.time() - start_time})
                return result
            except Exception as e:
                child_trace.add_tag("status", "error")
                child_trace.add_tag("error", str(e))
                child_trace.add_event("operation_failed", {"error": str(e), "duration": time.time() - start_time})
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            trace = trace_manager.current_trace
            if not trace:
                return func(*args, **kwargs)
            
            # 创建子 span
            child_trace = TraceContext(trace.trace_id, trace.span_id)
            child_trace.add_tag("operation", operation_name)
            child_trace.add_tag("function", func.__name__)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                child_trace.add_tag("status", "success")
                child_trace.add_event("operation_completed", {"duration": time.time() - start_time})
                return result
            except Exception as e:
                child_trace.add_tag("status", "error")
                child_trace.add_tag("error", str(e))
                child_trace.add_event("operation_failed", {"error": str(e), "duration": time.time() - start_time})
                raise
        
        # 根据函数类型返回对应的包装器
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

@contextmanager
def trace_context(trace_id: Optional[str] = None):
    """上下文管理器：创建追踪上下文"""
    trace = trace_manager.start_trace(trace_id)
    try:
        yield trace
    finally:
        trace_manager.end_trace()

class TraceLogger:
    """追踪日志记录器"""
    
    def __init__(self, manager: TraceManager):
        self.manager = manager
    
    def log(self, level: str, message: str, **kwargs):
        """记录带追踪信息的日志"""
        trace_id = self.manager.get_current_trace_id()
        extra = {
            "trace_id": trace_id,
            **kwargs
        }
        
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """记录 INFO 级别日志"""
        self.log("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录 WARNING 级别日志"""
        self.log("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录 ERROR 级别日志"""
        self.log("error", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """记录 DEBUG 级别日志"""
        self.log("debug", message, **kwargs)

# 全局追踪日志记录器实例
trace_logger = TraceLogger(trace_manager)