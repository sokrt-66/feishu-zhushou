"""
LangGraph Checkpointer 实现
支持状态持久化与版本迁移，实现断线恢复
"""
import json
import os
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timedelta
import hashlib
import logging

logger = logging.getLogger(__name__)

class FileCheckpointer:
    """基于文件系统的Checkpointer实现，支持状态持久化与版本迁移"""
    
    # 默认过期时间：7天
    DEFAULT_EXPIRY_DAYS = 7
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpointer initialized with directory: {checkpoint_dir}")
    
    def _get_checkpoint_path(self, thread_id: str) -> str:
        """生成检查点文件路径"""
        hash_id = hashlib.md5(thread_id.encode()).hexdigest()
        return os.path.join(self.checkpoint_dir, f"{hash_id}.json")
    
    def get(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """获取检查点"""
        checkpoint_path = self._get_checkpoint_path(thread_id)
        if not os.path.exists(checkpoint_path):
            logger.debug(f"Checkpoint not found for thread: {thread_id}")
            return None
        
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 版本迁移处理
            data = self._migrate_version(data)
            logger.info(f"Checkpoint loaded for thread: {thread_id}, version: {data.get('_version', 1)}")
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint for thread {thread_id}: {e}")
            return None
    
    def put(self, thread_id: str, state: Dict[str, Any]) -> None:
        """保存检查点"""
        checkpoint_path = self._get_checkpoint_path(thread_id)
        
        # 添加元数据
        state_copy = state.copy()
        state_copy["_version"] = 2
        state_copy["_saved_at"] = datetime.now().isoformat()
        state_copy["_thread_id"] = thread_id
        
        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(state_copy, f, ensure_ascii=False, indent=2)
            logger.info(f"Checkpoint saved for thread: {thread_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint for thread {thread_id}: {e}")
    
    def delete(self, thread_id: str) -> bool:
        """删除检查点"""
        checkpoint_path = self._get_checkpoint_path(thread_id)
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                logger.info(f"Checkpoint deleted for thread: {thread_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete checkpoint for thread {thread_id}: {e}")
                return False
        return False
    
    def list_all(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        results = []
        try:
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(self.checkpoint_dir, filename), "r", encoding="utf-8") as f:
                            data = json.load(f)
                            results.append({
                                "thread_id": data.get("_thread_id", filename[:-5]),
                                "saved_at": data.get("_saved_at"),
                                "version": data.get("_version", 1)
                            })
                    except Exception as e:
                        logger.warning(f"Failed to read checkpoint {filename}: {e}")
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
        
        return results
    
    def cleanup_expired(self, expiry_days: int = None) -> int:
        """
        清理过期的检查点
        
        Args:
            expiry_days: 过期天数，默认为7天
        
        Returns:
            删除的检查点数量
        """
        if expiry_days is None:
            expiry_days = self.DEFAULT_EXPIRY_DAYS
        
        deleted_count = 0
        expiry_time = datetime.now() - timedelta(days=expiry_days)
        
        try:
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.checkpoint_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        saved_at_str = data.get("_saved_at")
                        if saved_at_str:
                            try:
                                saved_at = datetime.fromisoformat(saved_at_str)
                                if saved_at < expiry_time:
                                    os.remove(filepath)
                                    deleted_count += 1
                                    logger.info(f"Deleted expired checkpoint: {filename} (saved at {saved_at_str})")
                            except ValueError:
                                # 日期格式无效，跳过
                                logger.warning(f"Skipping checkpoint with invalid date format: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to check checkpoint {filename}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleanup completed: {deleted_count} expired checkpoints removed")
            else:
                logger.info("Cleanup completed: no expired checkpoints found")
        
        except Exception as e:
            logger.error(f"Failed to cleanup expired checkpoints: {e}")
        
        return deleted_count
    
    def get_expired_threads(self, expiry_days: int = None) -> List[str]:
        """
        获取过期的thread列表
        
        Args:
            expiry_days: 过期天数，默认为7天
        
        Returns:
            过期的thread_id列表
        """
        if expiry_days is None:
            expiry_days = self.DEFAULT_EXPIRY_DAYS
        
        expired_threads = []
        expiry_time = datetime.now() - timedelta(days=expiry_days)
        
        try:
            for filename in os.listdir(self.checkpoint_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.checkpoint_dir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        saved_at_str = data.get("_saved_at")
                        if saved_at_str:
                            try:
                                saved_at = datetime.fromisoformat(saved_at_str)
                                if saved_at < expiry_time:
                                    thread_id = data.get("_thread_id", filename[:-5])
                                    expired_threads.append(thread_id)
                            except ValueError:
                                pass
                    except Exception as e:
                        logger.warning(f"Failed to check checkpoint {filename}: {e}")
        except Exception as e:
            logger.error(f"Failed to get expired threads: {e}")
        
        return expired_threads
    
    def _migrate_version(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """版本迁移处理"""
        version = data.get("_version", 1)
        
        # v1 -> v2: 添加新字段，支持过期清理
        if version == 1:
            if "_saved_at" not in data:
                data["_saved_at"] = datetime.now().isoformat()
            if "_thread_id" not in data:
                data["_thread_id"] = ""
            data["_version"] = 2
        
        # 可继续添加更多版本迁移逻辑
        return data


# 全局Checkpointer实例
checkpointer = FileCheckpointer()