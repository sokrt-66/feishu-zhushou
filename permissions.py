import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)

class PermissionManager:
    """权限管理器 - 部门级数据隔离"""
    
    def __init__(self):
        self.user_departments: Dict[str, str] = {}  # user_id -> dept_id
        self.user_passwords: Dict[str, str] = {}  # user_id -> password (hashed)
        self.department_owners: Dict[str, List[str]] = {}  # dept_id -> [owner_id]
        self.department_kb_mapping: Dict[str, Set[str]] = {}  # dept_id -> {kb_id, ...}
        self.kb_metadata: Dict[str, Dict[str, Any]] = {}  # kb_id -> {name, created_by, created_time}
        self.department_info: Dict[str, Dict[str, Any]] = {}  # dept_id -> {name, description}
    
    def set_user_department(self, user_id: str, dept_id: str):
        """设置用户所属部门"""
        self.user_departments[user_id] = dept_id
        logger.info(f"用户 {user_id} 归属部门: {dept_id}")
    
    def get_user_department(self, user_id: str) -> Optional[str]:
        """获取用户所属部门"""
        return self.user_departments.get(user_id)
    
    def get_department_mapping(self) -> Dict[str, str]:
        """获取部门ID到部门名称的映射"""
        mapping = {}
        for dept_id, info in self.department_info.items():
            mapping[dept_id] = info.get("name", dept_id)
        return mapping
    
    def add_department_owner(self, dept_id: str, user_id: str):
        """添加部门管理员"""
        if dept_id not in self.department_owners:
            self.department_owners[dept_id] = []
        if user_id not in self.department_owners[dept_id]:
            self.department_owners[dept_id].append(user_id)
            logger.info(f"用户 {user_id} 成为部门 {dept_id} 管理员")
    
    def is_department_owner(self, user_id: str, dept_id: Optional[str] = None) -> bool:
        """检查是否是部门管理员"""
        if dept_id:
            owners = self.department_owners.get(dept_id, [])
            return user_id in owners
        
        # 检查用户是否是任何部门的管理员
        user_dept = self.get_user_department(user_id)
        if user_dept:
            owners = self.department_owners.get(user_dept, [])
            return user_id in owners
        
        return False
    
    def create_knowledge_base(self, kb_id: str, name: str, created_by: str, 
                              dept_id: Optional[str] = None) -> bool:
        """创建知识库"""
        if kb_id in self.kb_metadata:
            logger.warning(f"知识库 {kb_id} 已存在")
            return False
        
        # 如果没有指定部门，使用创建者的部门
        if not dept_id:
            dept_id = self.get_user_department(created_by)
            if not dept_id:
                logger.error(f"无法确定知识库 {kb_id} 所属部门")
                return False
        
        # 保存元数据
        self.kb_metadata[kb_id] = {
            "name": name,
            "created_by": created_by,
            "created_time": datetime.now().isoformat(),
            "dept_id": dept_id
        }
        
        # 关联到部门
        if dept_id not in self.department_kb_mapping:
            self.department_kb_mapping[dept_id] = set()
        self.department_kb_mapping[dept_id].add(kb_id)
        
        logger.info(f"创建知识库: {kb_id} ({name})，归属部门: {dept_id}")
        return True
    
    def can_access_kb(self, user_id: str, kb_id: str) -> bool:
        """检查用户是否可以访问知识库"""
        if kb_id not in self.kb_metadata:
            return False

        kb_dept = self.kb_metadata[kb_id]["dept_id"]
        user_dept = self.get_user_department(user_id)

        if user_dept and user_dept == kb_dept:
            return True

        if self.is_department_owner(user_id, kb_dept):
            return True

        return False
    
    def get_user_available_kbs(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户可访问的知识库列表"""
        user_dept = self.get_user_department(user_id)
        available_kbs = []
        
        if user_dept:
            # 获取部门下的知识库
            kb_ids = self.department_kb_mapping.get(user_dept, set())
            for kb_id in kb_ids:
                if kb_id in self.kb_metadata:
                    available_kbs.append({
                        "kb_id": kb_id,
                        "name": self.kb_metadata[kb_id]["name"],
                        "dept_id": user_dept,
                        "created_by": self.kb_metadata[kb_id]["created_by"],
                        "created_time": self.kb_metadata[kb_id]["created_time"]
                    })
        
        return available_kbs
    
    def delete_knowledge_base(self, user_id: str, kb_id: str) -> bool:
        """删除知识库（需要权限）"""
        if kb_id not in self.kb_metadata:
            return False
        
        # 检查权限
        kb_metadata = self.kb_metadata[kb_id]
        is_owner = (user_id == kb_metadata["created_by"])
        is_dept_owner = self.is_department_owner(user_id, kb_metadata["dept_id"])
        
        if not (is_owner or is_dept_owner):
            logger.warning(f"用户 {user_id} 无权限删除知识库 {kb_id}")
            return False
        
        # 从部门映射中移除
        dept_id = kb_metadata["dept_id"]
        if dept_id in self.department_kb_mapping:
            self.department_kb_mapping[dept_id].discard(kb_id)
        
        # 删除元数据
        del self.kb_metadata[kb_id]
        
        logger.info(f"删除知识库: {kb_id}")
        return True
    
    def get_department_users(self, dept_id: str) -> List[str]:
        """获取部门下所有用户"""
        return [user_id for user_id, user_dept in self.user_departments.items() 
                if user_dept == dept_id]
    
    def get_department_info(self, dept_id: str) -> Optional[Dict[str, Any]]:
        """获取部门信息"""
        # 如果部门不存在于部门信息表中，返回None
        if dept_id not in self.department_info:
            return None
        
        return {
            "dept_id": dept_id,
            "name": self.department_info[dept_id].get("name", ""),
            "description": self.department_info[dept_id].get("description", ""),
            "owners": self.department_owners.get(dept_id, []),
            "users": self.get_department_users(dept_id),
            "knowledge_bases": list(self.department_kb_mapping.get(dept_id, set()))
        }
    
    def create_department(self, dept_id: str, name: str, description: str = "") -> bool:
        """创建新部门"""
        if dept_id in self.department_info:
            logger.warning(f"部门 {dept_id} 已存在")
            return False
        
        self.department_info[dept_id] = {
            "name": name,
            "description": description
        }
        
        # 初始化部门的知识库映射
        if dept_id not in self.department_kb_mapping:
            self.department_kb_mapping[dept_id] = set()
        
        logger.info(f"创建部门: {dept_id} ({name})")
        return True
    
    def get_all_departments(self) -> List[Dict[str, Any]]:
        """获取所有部门列表"""
        departments = []
        for dept_id in self.department_info:
            departments.append({
                "dept_id": dept_id,
                "name": self.department_info[dept_id].get("name", ""),
                "description": self.department_info[dept_id].get("description", ""),
                "user_count": len(self.get_department_users(dept_id))
            })
        return departments
    
    def create_user(self, username: str, password: str, department: str) -> bool:
        """创建新用户"""
        if username in self.user_departments:
            logger.warning(f"用户 {username} 已存在")
            return False
        
        # 存储密码（实际生产环境应使用哈希）
        self.user_passwords[username] = password
        self.user_departments[username] = department
        
        logger.info(f"创建用户: {username}，归属部门: {department}")
        return True
    
    def delete_user(self, username: str) -> bool:
        """删除用户"""
        if username not in self.user_departments:
            logger.warning(f"用户 {username} 不存在")
            return False
        
        del self.user_departments[username]
        if username in self.user_passwords:
            del self.user_passwords[username]
        
        logger.info(f"删除用户: {username}")
        return True
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """获取所有用户列表"""
        users = []
        for username, dept_id in self.user_departments.items():
            dept_name = self.department_info.get(dept_id, {}).get("name", dept_id)
            users.append({
                "username": username,
                "department": dept_id,
                "department_name": dept_name
            })
        return users
    
    def is_department_admin(self, user_id: str, dept_id: str) -> bool:
        """检查用户是否是指定部门的管理员"""
        # admin 用户拥有所有权限
        if user_id == "admin":
            return True
        
        owners = self.department_owners.get(dept_id, [])
        return user_id in owners


class RAGPermissionGuard:
    """RAG检索权限守卫"""
    
    def __init__(self, permission_manager: PermissionManager):
        self.perm_manager = permission_manager
    
    async def retrieve_with_permission(self, user_id: str, query: str, 
                                        kb_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """带权限检查的RAG检索"""
        # 获取用户可访问的知识库
        available_kbs = self.perm_manager.get_user_available_kbs(user_id)
        available_kb_ids = {kb["kb_id"] for kb in available_kbs}
        
        # 过滤指定的知识库
        if kb_ids:
            filtered_kb_ids = [kb_id for kb_id in kb_ids if kb_id in available_kb_ids]
        else:
            filtered_kb_ids = list(available_kb_ids)
        
        if not filtered_kb_ids:
            return {
                "status": "error",
                "message": "没有可访问的知识库",
                "results": []
            }
        
        # 这里执行实际的RAG检索（简化版）
        results = []
        for kb_id in filtered_kb_ids:
            results.append({
                "kb_id": kb_id,
                "score": 0.9,
                "content": f"[权限检查通过] {query} 的相关结果"
            })
        
        return {
            "status": "success",
            "message": f"从 {len(filtered_kb_ids)} 个知识库检索",
            "results": results,
            "accessed_kbs": filtered_kb_ids
        }
