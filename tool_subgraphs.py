from __future__ import annotations

"""
工具子图模块 
包含SQL、RAG、Web搜索三个子智能体图的完整实现
使用LangGraph构建，支持重试、反思、熔断、超时机制
"""

from typing import TypedDict, Dict, Any, Optional, List
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
import json
import time
import asyncio
import re
import os
from datetime import datetime
import sqlparse
from sqlparse import tokens as T

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except (ImportError, TypeError) as e:
    # TypeError occurs when langgraph uses Python 3.9+ syntax on Python 3.8
    print(f"LangGraph 不可用（Python版本兼容问题）: {e}")
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from config import sql_config, rag_config, web_search_config, agent_config, init_config
from database import get_db_connection


# ==================== 1. 状态定义 ====================

class SQLState(TypedDict):
    """SQL子图状态"""
    query: str
    intent: Optional[str]
    schema_context: Optional[str]
    generated_sql: Optional[str]
    validated: bool
    validation_error: Optional[str]
    retry_count: int
    execution_result: Optional[str]
    final_answer: Optional[str]
    error: Optional[str]
    success: bool
    start_time: float
    department: Optional[str]
    is_admin: bool
    operation_type: Optional[str]  # 'query' or 'insert'


class RAGState(TypedDict):
    """RAG子图状态"""
    query: str
    rewritten_query: Optional[str]
    retrieved_docs: Optional[List[str]]
    relevance_scores: Optional[List[float]]
    avg_relevance: float
    need_web_search: bool
    context: Optional[str]
    final_answer: Optional[str]
    error: Optional[str]
    success: bool
    start_time: float
    available_kbs: Optional[List[str]]
    has_web_search: bool  # 标记是否使用了Web搜索


class WebSearchState(TypedDict):
    """Web搜索子图状态"""
    query: str
    search_results: Optional[List[Dict[str, Any]]]
    summarized_results: Optional[str]
    trust_score: float
    final_answer: Optional[str]
    error: Optional[str]
    success: bool
    timeout_triggered: bool
    start_time: float


# ==================== 2. SQL子图实现 ====================

class SQLSubGraph:
    """SQL子智能体图 - 实现Text2SQL完整流程"""
    
    def __init__(self, llm=None):
        self.retry_limit = sql_config.sql_internal_retry
        self.timeout = sql_config.sql_exec_timeout
        self.enable_cache = sql_config.enable_sql_cache
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = sql_config.cache_ttl_minutes * 60
        
        self.llm = llm
        
        # 获取数据库连接（异步）
        self.db_connector = get_db_connection()
        
        # 延迟获取数据库表结构，在第一次使用时初始化
        self._schema = None
    
    def set_llm(self, llm):
        """设置LLM实例"""
        self.llm = llm
    
    async def _get_database_schema(self) -> str:
        """从实际数据库获取表结构"""
        try:
            schema = await self.db_connector.get_table_schema()
            if schema and "无法连接" not in schema:
                return schema
        except Exception as e:
            print(f"获取数据库结构失败: {e}")
        
        # 如果无法连接，返回模拟的表结构
        return """
        表结构：
        - users(id, name, email, department, created_at)
        - orders(id, user_id, product, amount, status, created_at)
        - products(id, name, price, category, stock)
        - departments(id, name, manager_id)
        """
    
    def _get_cache(self, key: str) -> Optional[Any]:
        if not self.enable_cache:
            return None
        cached = self.cache.get(key)
        if cached and time.time() - cached["time"] < self.cache_ttl:
            return cached["data"]
        return None
    
    def _set_cache(self, key: str, data: Any):
        if self.enable_cache:
            self.cache[key] = {"data": data, "time": time.time()}
    
    def node_understand_intent(self, state: SQLState) -> SQLState:
        """节点：理解用户查询意图"""
        query = state["query"]
        
        intent_keywords = {
            "查询": ["查", "找", "搜索", "获取", "显示", "列出"],
            "统计": ["统计", "计算", "总数", "平均", "求和", "最大", "最小", "汇总"],
            "比较": ["比较", "对比", "差异", "排名"],
            "更新": ["更新", "修改", "更改"],
            "删除": ["删除", "移除"]
        }
        
        intent = "查询"
        for intent_type, keywords in intent_keywords.items():
            if any(kw in query for kw in keywords):
                intent = intent_type
                break
        
        # 提取时间范围信息
        time_info = self._extract_time_info(query)
        
        # 提取表名信息
        table_info = self._extract_table_info(query)
        
        return {**state, "intent": intent, "time_info": time_info, "table_info": table_info}
    
    def _extract_time_info(self, query: str) -> dict:
        """从查询中提取时间范围信息"""
        import re
        
        time_info = {
            "has_time_filter": False,
            "year": None,
            "month": None,
            "quarter": None,
            "time_range": None,
            "relative_time": None
        }
        
        # 匹配年份
        year_match = re.search(r'(\d{4})年', query)
        if year_match:
            time_info["year"] = int(year_match.group(1))
            time_info["has_time_filter"] = True
        
        # 匹配月份
        month_match = re.search(r'(\d{1,2})月', query)
        if month_match:
            time_info["month"] = int(month_match.group(1))
            time_info["has_time_filter"] = True
        
        # 匹配季度
        quarter_match = re.search(r'([一二三四])季度|Q(\d)', query)
        if quarter_match:
            quarters = {'一': 1, '二': 2, '三': 3, '四': 4}
            if quarter_match.group(1):
                time_info["quarter"] = quarters.get(quarter_match.group(1))
            else:
                time_info["quarter"] = int(quarter_match.group(2))
            time_info["has_time_filter"] = True
        
        # 匹配相对时间
        relative_patterns = [
            (r'最近(\d+)天', 'days'),
            (r'最近(\d+)周', 'weeks'),
            (r'最近(\d+)个月', 'months'),
            (r'本周', 'week'),
            (r'上周', 'last_week'),
            (r'本月', 'month'),
            (r'上月', 'last_month'),
            (r'今年', 'year'),
            (r'去年', 'last_year'),
            (r'昨天', 'yesterday'),
            (r'今天', 'today'),
        ]
        
        for pattern, time_type in relative_patterns:
            match = re.search(pattern, query)
            if match:
                if match.group(1):
                    time_info["relative_time"] = {"type": time_type, "value": int(match.group(1))}
                else:
                    time_info["relative_time"] = {"type": time_type}
                time_info["has_time_filter"] = True
                break
        
        return time_info
    
    def _extract_table_info(self, query: str) -> dict:
        """从查询中提取表名信息"""
        table_keywords = {
            "用户": ["user", "users", "员工", "人员", "成员"],
            "订单": ["order", "orders", "交易", "销售"],
            "产品": ["product", "products", "商品", "货品"],
            "部门": ["department", "departments", "组织", "团队"],
            "日志": ["log", "logs", "记录", "历史"],
            "报表": ["report", "reports", "统计"],
        }
        
        table_info = {
            "hints": [],
            "likely_tables": []
        }
        
        for table_name, keywords in table_keywords.items():
            for kw in keywords:
                if kw in query:
                    table_info["hints"].append(table_name)
                    # 添加可能的表名（考虑分区表）
                    table_info["likely_tables"].append(table_name.lower())
                    # 考虑分区表命名模式
                    table_info["likely_tables"].append(f"{table_name.lower()}s")
                    table_info["likely_tables"].append(f"{table_name.lower()}_history")
                    break
        
        return table_info
    
    async def node_inject_schema(self, state: SQLState) -> SQLState:
        """节点：注入数据库Schema上下文"""
        if self._schema is None:
            self._schema = await self._get_database_schema()
        return {**state, "schema_context": self._schema}
    
    async def node_generate_sql(self, state: SQLState) -> SQLState:
        """节点：LLM生成SQL语句"""
        query = state["query"]
        schema = state["schema_context"]
        intent = state["intent"]
        time_info = state.get("time_info", {})
        table_info = state.get("table_info", {})
        
        # 检查查询是否为空
        if not query or not query.strip():
            return {**state, "generated_sql": "", "validated": False, "error": "请输入查询内容"}
        
        # 构建时间条件提示
        time_hint = ""
        if time_info.get("has_time_filter"):
            time_hint = "\n时间条件："
            if time_info.get("year"):
                time_hint += f"年份={time_info['year']} "
            if time_info.get("month"):
                time_hint += f"月份={time_info['month']} "
            if time_info.get("quarter"):
                time_hint += f"季度=Q{time_info['quarter']} "
            if time_info.get("relative_time"):
                rt = time_info["relative_time"]
                time_units = {'days':'天','weeks':'周','months':'月','week':'周','last_week':'上周','month':'月','last_month':'上月','year':'年','last_year':'去年','yesterday':'昨天','today':'今天'}
                if rt.get("value"):
                    time_hint += f"最近{rt['value']}{time_units.get(rt['type'], '')} "
                else:
                    time_hint += f"{time_units.get(rt['type'], '')} "
        
        # 构建表提示
        table_hint = ""
        if table_info.get("hints"):
            table_hint = f"\n可能的表：{', '.join(table_info['hints'])}"
        
        # 构建意图提示
        intent_hint = ""
        if intent == "统计":
            intent_hint = "\n这是一个统计查询，请使用COUNT/SUM/AVG等聚合函数，并考虑分组统计。"
        elif intent == "比较":
            intent_hint = "\n这是一个比较查询，请考虑使用ORDER BY排序或LIMIT限制结果。"
        
        if self.llm:
            try:
                prompt = f"""
你是一个SQL专家。根据用户问题和数据库结构，生成正确的SQL查询语句。

数据库结构：
{schema}

用户问题：{query}
查询意图：{intent}
{time_hint}
{table_hint}
{intent_hint}

要求：
1. 只生成SELECT语句（安全考虑）
2. 使用标准SQL语法
3. 必须添加 LIMIT 100 限制返回行数（防止返回过多数据）
4. 如果是统计查询，使用合适的聚合函数（COUNT/SUM/AVG/MIN/MAX）
5. 如果涉及时间范围，添加合适的时间条件
6. 只返回SQL语句，不要解释

SQL：
"""
                response = await self.llm.ainvoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)
                sql = content.strip()
                sql = re.sub(r'```sql\s*', '', sql)
                sql = re.sub(r'```\s*', '', sql).strip()
                
                # 确保有 LIMIT 限制
                if not sql.upper().strip().endswith(';'):
                    sql = sql.rstrip() + ';'
                
                if not re.search(r'\bLIMIT\s+\d+\b', sql, re.IGNORECASE):
                    sql = re.sub(r';\s*$', '', sql)
                    sql = f"{sql} LIMIT 100;"
                
                if not sql.upper().startswith('SELECT'):
                    sql = f"SELECT * FROM employees WHERE name LIKE '%{query}%' LIMIT 100;"
            except Exception as e:
                sql = f"SELECT * FROM employees WHERE name LIKE '%{query}%' LIMIT 100;"
        else:
            sql = f"SELECT * FROM employees WHERE name LIKE '%{query}%' LIMIT 100;"
        
        return {**state, "generated_sql": sql}
    
    def node_validate_sql(self, state: SQLState) -> SQLState:
        """节点：SQL语法校验"""
        sql = state["generated_sql"] or ""
        is_admin = state.get("is_admin", False)
        
        # 危险操作（始终禁止）
        dangerous_patterns = [
            (r'\bDROP\b', '删除表'),
            (r'\bTRUNCATE\b', '清空表'),
            (r'\bALTER\b', '修改表结构'),
            (r'\bEXEC\b', '执行存储过程'),
            (r'\bEXECUTE\b', '执行存储过程'),
            (r'\bGRANT\b', '授权'),
            (r'\bREVOKE\b', '撤销权限'),
        ]
        
        # 管理员允许但普通用户禁止的操作
        admin_only_patterns = [
            (r'\bDELETE\b', '删除数据'),
            (r'\bUPDATE\b', '更新数据'),
        ]
        
        validation_error = None
        
        # 检查危险操作
        for pattern, action in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                validation_error = f"禁止执行危险操作：{action}"
                break
        
        # 如果不是危险操作，检查管理员权限
        if not validation_error:
            sql_upper = sql.upper()
            
            # 管理员允许INSERT和UPDATE
            if is_admin:
                # 管理员可以执行INSERT和UPDATE（限制在employees表）
                if sql_upper.startswith('INSERT') or sql_upper.startswith('UPDATE'):
                    # 限制只能操作employees表
                    if 'EMPLOYEES' not in sql_upper:
                        validation_error = "管理员只能对employees表进行写操作"
                elif not sql_upper.startswith('SELECT'):
                    validation_error = "不支持的SQL操作类型"
            else:
                # 普通用户只允许SELECT
                for pattern, action in admin_only_patterns:
                    if re.search(pattern, sql, re.IGNORECASE):
                        validation_error = f"没有权限执行该操作：{action}（需要管理员权限）"
                        break
                if not validation_error and not sql_upper.startswith('SELECT'):
                    validation_error = "没有权限执行该操作：仅允许查询操作"
        
        validated = validation_error is None
        return {**state, "validated": validated, "validation_error": validation_error}
    
    async def node_execute_sql(self, state: SQLState) -> SQLState:
        """节点：执行SQL并返回结果"""
        sql = state["generated_sql"]
        is_admin = state.get("is_admin", False)
        
        # 验证SQL是否为空
        if not sql or not sql.strip():
            return {**state, "error": "SQL语句为空", "success": False}
        
        # 只有SELECT查询才使用缓存
        cache_key = None
        if sql.upper().startswith('SELECT'):
            cache_key = f"sql_{hash(sql)}"
            cached = self._get_cache(cache_key)
            if cached:
                print(f"DEBUG: 使用缓存结果")
                return {**state, "execution_result": cached, "success": True}
        
        elapsed = time.time() - state["start_time"]
        if elapsed > self.timeout:
            error_msg = f"执行超时（{self.timeout}s）"
            print(f"ERROR: {error_msg}")
            return {**state, "error": error_msg, "success": False}
        
        # 使用异步数据库连接执行SQL
        try:
            sql_upper = sql.upper()
            
            # 如果是INSERT操作（管理员），使用execute_update方法
            if (sql_upper.startswith('INSERT') or sql_upper.startswith('UPDATE')) and is_admin:
                result = await self.db_connector.execute_update(sql, timeout=self.timeout)
            else:
                result = await self.db_connector.execute_query(sql, timeout=self.timeout)
            
            if result["success"]:
                if sql_upper.startswith('INSERT'):
                    # INSERT操作返回影响行数和新增ID
                    result_data = {
                        "affected_rows": result.get("affected_rows", 0),
                        "last_insert_id": result.get("last_insert_id", 0),
                        "execution_time": result.get("execution_time", 0)
                    }
                elif sql_upper.startswith('UPDATE'):
                    # UPDATE操作返回影响行数
                    result_data = {
                        "affected_rows": result.get("affected_rows", 0),
                        "execution_time": result.get("execution_time", 0)
                    }
                else:
                    # SELECT查询返回数据
                    result_data = {
                        "data": result["data"],
                        "columns": result["columns"],
                        "row_count": result["row_count"],
                        "execution_time": result["execution_time"]
                    }
                    result_data = self._mask_sensitive_data(result_data)
                
                result_str = json.dumps(result_data, ensure_ascii=False, indent=2)
                
                # SELECT查询才缓存
                if sql_upper.startswith('SELECT'):
                    self._set_cache(cache_key, result_str)
                
                return {**state, "execution_result": result_str, "success": True}
            else:
                return {**state, "error": result.get("error", "执行失败"), "success": False}
        
        except Exception as e:
            # 返回真实错误信息，不使用模拟数据
            error_msg = f"数据库执行异常: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {**state, "error": error_msg, "success": False}
    
    def _mask_sensitive_data(self, result_data: dict) -> dict:
        """脱敏敏感数据"""
        sensitive_fields = ['salary', 'email', 'phone']
        
        if 'data' in result_data and isinstance(result_data['data'], list):
            for row in result_data['data']:
                for field in sensitive_fields:
                    if field in row:
                        if field == 'salary':
                            row[field] = '******'
                        elif field == 'email':
                            email = row[field]
                            if '@' in email:
                                parts = email.split('@')
                                row[field] = parts[0][0] + '****@' + parts[1]
                            else:
                                row[field] = '******'
                        elif field == 'phone':
                            phone = str(row[field])
                            if len(phone) >= 11:
                                row[field] = phone[:3] + '****' + phone[-4:]
                            else:
                                row[field] = '******'
        return result_data
    
    async def _get_department_mapping(self) -> Dict[str, str]:
        """动态获取部门ID到部门名称的映射（异步版本）"""
        # 从数据库查询部门信息
        try:
            if self.db_connector:
                try:
                    result = await self.db_connector.execute_query("SELECT dept_id, name FROM departments")
                    if result.get("success") and result.get("data"):
                        mapping = {}
                        for row in result["data"]:
                            dept_id = row.get("dept_id")
                            name = row.get("name")
                            if dept_id and name:
                                mapping[str(dept_id)] = str(name)
                        return mapping
                except Exception as e:
                    logger.debug(f"查询部门映射失败: {e}")
        except Exception as e:
            logger.debug(f"获取部门映射异常: {e}")
        
        # 默认映射（从配置文件获取）
        return {dept_id: dept_info["name"] for dept_id, dept_info in init_config.default_departments.items()}
    
    async def node_filter_by_department(self, state: SQLState) -> SQLState:
        """节点：按部门过滤查询结果（异步版本）"""
        department = state.get("department")
        
        # 如果没有部门信息，不进行过滤（管理员或系统用户）
        if not department:
            return state
        
        # 如果执行失败，跳过过滤
        if not state["success"] or not state["execution_result"]:
            return state
        
        try:
            result_data = json.loads(state["execution_result"])
            
            # 如果有数据且有 department 字段，进行过滤
            if "data" in result_data and isinstance(result_data["data"], list):
                original_count = len(result_data["data"])
                
                # 动态获取部门映射（异步调用）
                dept_mapping = await self._get_department_mapping()
                target_dept = dept_mapping.get(department, department)
                
                # 检查用户查询的部门是否与自己所在部门一致
                user_query = state.get("query", "").lower()
                user_dept_name = target_dept
                
                # 定义所有部门名称列表（用于检查查询中是否包含部门名称）
                all_dept_names = list(dept_mapping.values())
                
                # 检查用户是否查询了其他部门
                other_dept_found = False
                queried_dept_name = ""
                for dept_id, dept_name in dept_mapping.items():
                    # 如果查询中包含其他部门名称，但不是用户自己的部门
                    if dept_name.lower() in user_query:
                        if dept_name != user_dept_name:
                            other_dept_found = True
                            queried_dept_name = dept_name
                            break
                
                # 如果用户明确查询其他部门，返回权限不足提示
                if other_dept_found:
                    state["success"] = False
                    state["error"] = f"权限不足：您只能查询自己部门（{user_dept_name}）的数据，无法查询其他部门（{queried_dept_name}）的信息。"
                    state["execution_result"] = json.dumps({
                        "status": "error",
                        "message": f"权限不足：您只能查询自己部门（{user_dept_name}）的数据，无法查询其他部门（{queried_dept_name}）的信息。",
                        "data": [],
                        "row_count": 0
                    }, ensure_ascii=False)
                    return state
                
                # 检查用户是否查询了不存在的部门
                # 常见部门名称关键词
                dept_keywords = ["销售部", "销售部门", "市场销售", "业务部", "研发部", "工程部", "财务部", "行政部", "客服部", "运营部"]
                for keyword in dept_keywords:
                    # 直接比较（中文字符不需要lower转换）
                    if keyword in state.get("query", ""):
                        # 检查这个关键词是否是已知部门的一部分
                        is_known_dept = False
                        for dept_name in all_dept_names:
                            if keyword in dept_name or dept_name in keyword:
                                is_known_dept = True
                                break
                        if not is_known_dept:
                            state["success"] = False
                            state["error"] = f"部门不存在：系统中未找到'{keyword}'相关的部门信息。"
                            state["execution_result"] = json.dumps({
                                "status": "error",
                                "message": f"部门不存在：系统中未找到'{keyword}'相关的部门信息。",
                                "data": [],
                                "row_count": 0
                            }, ensure_ascii=False)
                            return state
                
                # 过滤数据
                filtered_data = [
                    row for row in result_data["data"] 
                    if row.get("department") == target_dept
                ]
                
                result_data["data"] = filtered_data
                result_data["row_count"] = len(filtered_data)
                result_data["filtered_by_department"] = target_dept
                
                state["execution_result"] = json.dumps(result_data, ensure_ascii=False)
                
        except Exception as e:
            print(f"部门过滤失败: {e}")
        
        return state
    
    def node_format_result(self, state: SQLState) -> SQLState:
        """节点：格式化最终结果"""
        if state["success"] and state["execution_result"]:
            # 解析结果并脱敏敏感数据
            try:
                result_data = json.loads(state["execution_result"])
                result_data = self._mask_sensitive_data(result_data)
                answer = self._format_large_result(result_data)
            except:
                answer = f"查询结果：\n{state['execution_result']}"
        else:
            # 如果有权限错误，直接返回权限错误信息
            error_msg = state.get('error', '')
            if error_msg and ('权限不足' in error_msg or '无权' in error_msg):
                # 直接返回权限错误，不添加"查询失败："前缀
                answer = error_msg
            else:
                answer = f"查询失败：{error_msg or '未知错误'}"
        
        return {**state, "final_answer": answer}
    
    def _format_large_result(self, result_data: dict) -> str:
        """格式化大数据量查询结果"""
        data = result_data.get("data", [])
        columns = result_data.get("columns", [])
        row_count = result_data.get("row_count", 0)
        execution_time = result_data.get("execution_time", 0)
        
        # 统计类型查询的处理
        is_summary = False
        if len(data) == 1 and columns:
            for col in columns:
                if col.lower() in ['count', 'sum', 'avg', 'min', 'max', 'total', '合计', '总数']:
                    is_summary = True
                    break
        
        if is_summary:
            # 统计结果：简化展示
            lines = ["📊 统计结果："]
            for row in data:
                for col in columns:
                    value = row.get(col, "")
                    lines.append(f"  - {col}: {value}")
            if execution_time:
                lines.append(f"\n⏱️ 执行时间：{execution_time:.2f}秒")
            return "\n".join(lines)
        
        # 大数据量截断处理 - 超过1000行时只保留前50行并添加截断标记
        truncated = False
        max_display_rows = 50
        max_total_rows = 1000
        
        if row_count > max_total_rows:
            truncated = True
            display_data = data[:max_display_rows]
            lines = [f"📋 查询结果（共 {row_count} 条记录，仅显示前{max_display_rows}条）"]
        elif row_count == 0:
            return "📭 未找到匹配的数据"
        else:
            display_data = data[:max_display_rows]
            if len(data) > max_display_rows:
                truncated = True
            lines = [f"📋 查询结果（共 {row_count} 条记录）"]
        
        # 简洁的列表格式输出
        if columns and display_data:
            # 判断是否为员工信息表
            is_employee_table = False
            employee_cols = ['name', '姓名', '部门', 'department', 'position', '职位', 'email', 'phone']
            for col in columns:
                if col.lower() in employee_cols:
                    is_employee_table = True
                    break
            
            if is_employee_table and len(columns) > 4:
                # 员工信息：每行一个人，列出关键信息
                for i, row in enumerate(display_data, 1):
                    name = row.get('name', row.get('姓名', ''))
                    dept = row.get('department', row.get('部门', ''))
                    position = row.get('position', row.get('职位', ''))
                    status = row.get('status', row.get('状态', ''))
                    
                    info_parts = []
                    if name:
                        info_parts.append(f"姓名：{name}")
                    if dept:
                        info_parts.append(f"部门：{dept}")
                    if position:
                        info_parts.append(f"职位：{position}")
                    if status:
                        info_parts.append(f"状态：{status}")
                    
                    # 添加其他字段
                    for col in columns:
                        if col.lower() not in ['name', '姓名', 'department', '部门', 'position', '职位', 'status', '状态']:
                            value = row.get(col, "")
                            if value and str(value) != '******':
                                info_parts.append(f"{col}：{value}")
                    
                    lines.append(f"\n{i}. {'; '.join(info_parts)}")
            else:
                # 普通表格数据：简洁的列表格式
                for i, row in enumerate(display_data, 1):
                    row_items = []
                    for col in columns:
                        value = row.get(col, "")
                        if value and str(value) != '******':
                            row_items.append(f"{col}={value}")
                    if row_items:
                        lines.append(f"\n{i}. {', '.join(row_items)}")
            
            # 添加截断提示
            if truncated:
                lines.append(f"\n⚠️ 结果已截断，如需查看更多请添加筛选条件")
            
            if execution_time:
                lines.append(f"\n⏱️ 执行时间：{execution_time:.2f}秒")
        
        return "\n".join(lines)
    
    def node_handle_error(self, state: SQLState) -> SQLState:
        """节点：错误处理与重试判断"""
        retry_count = state["retry_count"] + 1
        
        if retry_count >= self.retry_limit:
            return {**state, 
                    "retry_count": retry_count,
                    "final_answer": f"SQL执行失败，已重试{retry_count}次，请检查查询语句",
                    "success": False}
        
        return {**state, "retry_count": retry_count}
    
    def route_after_validation(self, state: SQLState) -> str:
        """条件路由：校验后决定下一步"""
        if state["validated"]:
            return "execute"
        return "error"
    
    def route_after_error(self, state: SQLState) -> str:
        """条件路由：错误处理后决定重试或结束"""
        if state["retry_count"] >= self.retry_limit:
            return "end"
        return "retry"
    
    def build_graph(self):
        """构建SQL子图"""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        graph = StateGraph(SQLState)
        
        graph.add_node("understand_intent", self.node_understand_intent)
        graph.add_node("inject_schema", self.node_inject_schema)
        graph.add_node("generate_sql", self.node_generate_sql)
        graph.add_node("validate_sql", self.node_validate_sql)
        graph.add_node("execute_sql", self.node_execute_sql)
        graph.add_node("filter_by_department", self.node_filter_by_department)  # 添加部门过滤节点
        graph.add_node("format_result", self.node_format_result)
        graph.add_node("handle_error", self.node_handle_error)
        
        graph.set_entry_point("understand_intent")
        graph.add_edge("understand_intent", "inject_schema")
        graph.add_edge("inject_schema", "generate_sql")
        graph.add_edge("generate_sql", "validate_sql")
        
        graph.add_conditional_edges(
            "validate_sql",
            self.route_after_validation,
            {"execute": "execute_sql", "error": "handle_error"}
        )
        
        graph.add_edge("execute_sql", "filter_by_department")  # 执行后先过滤部门
        graph.add_edge("filter_by_department", "format_result")  # 过滤后再格式化结果
        graph.add_edge("format_result", END)
        
        graph.add_conditional_edges(
            "handle_error",
            self.route_after_error,
            {"retry": "generate_sql", "end": END}
        )
        
        return graph.compile()
    
    async def run(self, query: str, department: str = None, is_admin: bool = False) -> Dict[str, Any]:
        """执行SQL子图（支持部门权限过滤和管理员模式）"""
        initial_state: SQLState = {
            "query": query,
            "intent": None,
            "schema_context": None,
            "generated_sql": None,
            "validated": False,
            "validation_error": None,
            "retry_count": 0,
            "execution_result": None,
            "final_answer": None,
            "error": None,
            "success": False,
            "start_time": time.time(),
            "department": department,  # 添加部门信息
            "is_admin": is_admin,      # 添加管理员标志
            "operation_type": "query"  # 默认操作类型
        }
        
        if LANGGRAPH_AVAILABLE:
            try:
                graph = self.build_graph()
                if graph:
                    result = graph.invoke(initial_state)
                    return {
                        "success": result.get("success", False),
                        "result": result.get("final_answer"),
                        "sql": result.get("generated_sql"),
                        "error": result.get("error")
                    }
            except Exception as e:
                pass
        
        return await self._run_fallback(query, initial_state)
    
    async def _run_fallback(self, query: str, state: SQLState) -> Dict[str, Any]:
        """降级执行（无LangGraph时）"""
        state = self.node_understand_intent(state)
        state = await self.node_inject_schema(state)  # 添加 await
        
        for attempt in range(self.retry_limit + 1):
            state["retry_count"] = attempt
            state = await self.node_generate_sql(state)
            state = self.node_validate_sql(state)
            
            if state["validated"]:
                state = await self.node_execute_sql(state)  # 添加 await
                state = await self.node_filter_by_department(state)  # 添加部门过滤（异步）
                state = self.node_format_result(state)
                break
            else:
                if attempt >= self.retry_limit:
                    state["final_answer"] = f"SQL生成失败，已重试{attempt}次"
                    state["success"] = False
        
        return {
            "success": state.get("success", False),
            "result": state.get("final_answer"),
            "sql": state.get("generated_sql"),
            "error": state.get("error")
        }


# ==================== 3. RAG子图实现 ====================

class RAGSubGraph:
    """RAG子智能体图 - 实现知识库检索完整流程"""

    def __init__(self, document_subgraph=None, llm=None):
        self.threshold = rag_config.rag_relevance_threshold
        self.max_recall = rag_config.rag_max_recall
        self.enable_rewrite = rag_config.enable_query_rewrite
        self.enable_synthesis = rag_config.enable_answer_synthesis

        self.document_subgraph = document_subgraph
        self.llm = llm

        # 支持的文件扩展名
        self.supported_extensions = ['.txt', '.md', '.json']
        
        # 从知识库目录加载所有文件
        self.knowledge_base_dir = "knowledge_base"
        self.knowledge_base = self._load_knowledge_from_directory(self.knowledge_base_dir)
    
    def _load_knowledge_from_directory(self, dir_path: str) -> list:
        """从目录加载所有支持的文件作为知识库"""
        import os
        import glob
        
        knowledge_base = []
        
        # 检查目录是否存在
        if not os.path.exists(dir_path):
            print(f"DEBUG: 知识库目录 {dir_path} 不存在，创建目录并使用默认知识库")
            os.makedirs(dir_path, exist_ok=True)
            # 创建示例文件
            self._create_sample_files(dir_path)
            return self._load_default_knowledge()
        
        # 查找所有支持的文件
        files = []
        for ext in self.supported_extensions:
            files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
        
        if not files:
            print(f"DEBUG: 知识库目录 {dir_path} 中没有找到支持的文件，使用默认知识库")
            # 创建示例文件
            self._create_sample_files(dir_path)
            return self._load_default_knowledge()
        
        # 加载每个文件
        total_chunks = 0
        for file_path in files:
            file_chunks = self._load_knowledge_from_file(file_path)
            knowledge_base.extend(file_chunks)
            total_chunks += len(file_chunks)
            print(f"DEBUG: 从 {os.path.basename(file_path)} 加载了 {len(file_chunks)} 条内容")
        
        print(f"DEBUG: 共从 {len(files)} 个文件加载了 {total_chunks} 条知识库内容")
        return knowledge_base
    
    def _load_knowledge_from_file(self, file_path: str) -> list:
        """从单个文件加载知识库内容"""
        import os
        
        knowledge_base = []
        file_name = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 根据文件类型进行不同处理
            _, ext = os.path.splitext(file_path)
            
            if ext == '.json':
                # JSON 文件处理
                import json
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and 'content' in item:
                                text = item['content']
                            elif isinstance(item, str):
                                text = item
                            else:
                                text = str(item)
                            knowledge_base.extend(self._chunk_text(text, file_name))
                    elif isinstance(data, dict):
                        for key, value in data.items():
                            text = f"{key}: {value}" if isinstance(value, str) else str(value)
                            knowledge_base.extend(self._chunk_text(text, file_name))
                    else:
                        knowledge_base.extend(self._chunk_text(str(data), file_name))
                except json.JSONDecodeError:
                    # 如果不是有效JSON，当作普通文本处理
                    knowledge_base.extend(self._chunk_text(content, file_name))
            else:
                # TXT 和 MD 文件处理
                # 按段落分割内容（空行分隔）
                paragraphs = content.split('\n\n')
                paragraphs = [p.strip() for p in paragraphs if p.strip()]
                
                for paragraph in paragraphs:
                    knowledge_base.extend(self._chunk_text(paragraph, file_name))
            
        except Exception as e:
            print(f"DEBUG: 加载文件 {file_name} 失败: {e}")
        
        return knowledge_base
    
    def _chunk_text(self, text: str, source_file: str, chunk_size: int = 500) -> list:
        """将文本分割成合适大小的块"""
        chunks = []
        
        # 如果文本为空，返回空列表
        if not text or not text.strip():
            return chunks
        
        # 移除多余空白
        text = ' '.join(text.split())
        
        # 如果文本长度小于块大小，直接返回
        if len(text) <= chunk_size:
            return [{
                "content": text,
                "embedding": [0.1] * 1536,
                "source": source_file
            }]
        
        # 按块大小分割
        start = 0
        chunk_index = 0
        while start < len(text):
            end = start + chunk_size
            
            # 尝试在句子边界处分割
            if end < len(text):
                # 查找最近的句子结束符
                for sep in ['。', '！', '？', '.', '!', '?', '；', ';', '，', ',']:
                    pos = text.rfind(sep, start, end + 50)
                    if pos != -1 and pos > start:
                        end = pos + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({
                    "content": chunk,
                    "embedding": [(0.1 + chunk_index * 0.05) % 1.0] * 1536,
                    "source": source_file
                })
                chunk_index += 1
            
            start = end
        
        return chunks
    
    def _create_sample_files(self, dir_path: str):
        """创建示例知识库文件"""
        import os
        
        # 创建公司管理制度文件
        with open(os.path.join(dir_path, "company_policy.txt"), 'w', encoding='utf-8') as f:
            f.write("""# 公司管理制度

## 考勤制度
员工需在工作日的上午9:00前到达公司，下午6:00后离开。迟到超过30分钟视为旷工半天。
每月迟到不得超过3次，否则将扣除绩效分数。

## 请假制度
员工请假需提前至少1天申请，紧急情况可当天申请但需提供证明。
年假每年15天，病假凭医院证明可无限期休假但需提供病历。

## 报销制度
报销需提供正规发票，每月15日统一处理报销申请。
差旅费报销标准：一线城市每天500元，二线城市每天300元。

## 会议制度
每周一上午9:30召开部门例会，总结上周工作并规划本周任务。
重要会议需提前发送会议议程，并在会后24小时内发布会议纪要。
""")
        
        # 创建产品说明文件
        with open(os.path.join(dir_path, "products.md"), 'w', encoding='utf-8') as f:
            f.write("""# 公司产品介绍

## 智能客服系统
基于大语言模型的智能客服解决方案，支持多轮对话、意图识别、情感分析。
主要功能：自动问答、工单分配、知识库检索、数据分析报告。

## 数据分析平台
面向企业的数据可视化和分析工具，支持实时数据处理和预测分析。
核心模块：数据采集、数据清洗、可视化报表、机器学习预测。

## 知识库系统
企业级文档管理和检索系统，支持多格式文档解析和智能检索。
特点：支持PDF/Word/Excel等多种格式，支持语义搜索和问答功能。
""")
        
        # 创建员工手册文件
        with open(os.path.join(dir_path, "employee_handbook.json"), 'w', encoding='utf-8') as f:
            import json
            data = {
                "公司简介": "成立于2020年，专注于人工智能技术研发，现有员工50人",
                "企业文化": "创新、协作、客户至上",
                "职业发展": "提供完善的培训体系和晋升通道",
                "福利待遇": "五险一金、年度体检、节日福利、团建活动",
                "办公时间": "周一至周五 9:00-18:00，周末双休"
            }
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
    
    def _load_default_knowledge(self) -> list:
        """加载默认知识库"""
        return [
            {"content": "公司成立于2020年，专注于人工智能技术研发。", "embedding": [0.1] * 1536, "source": "default"},
            {"content": "技术团队有50人，分布在北京、上海、深圳。", "embedding": [0.2] * 1536, "source": "default"},
            {"content": "主要产品包括智能客服、数据分析平台、知识库系统。", "embedding": [0.3] * 1536, "source": "default"},
            {"content": "2023年营收增长200%，客户数量超过500家。", "embedding": [0.4] * 1536, "source": "default"},
            {"content": "公司价值观：创新、协作、客户至上。", "embedding": [0.5] * 1536, "source": "default"},
        ]
    
    def reload_knowledge(self):
        """重新加载知识库"""
        self.knowledge_base = self._load_knowledge_from_directory(self.knowledge_base_dir)
        print(f"DEBUG: 知识库已重新加载，共 {len(self.knowledge_base)} 条内容")
    
    def set_llm(self, llm):
        """设置LLM实例"""
        self.llm = llm

    def node_rewrite_query(self, state: RAGState) -> RAGState:
        """节点：查询重写优化"""
        if not self.enable_rewrite:
            return {**state, "rewritten_query": state["query"]}
        
        query = state["query"]
        
        if self.llm:
            try:
                prompt = f"请将以下查询优化为更适合检索的形式，只返回优化后的查询：\n{query}"
                response = self.llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0
                )
                rewritten = response.choices[0].message.content.strip()
            except Exception:
                rewritten = query
        else:
            rewritten = f"优化：{query}"
        
        return {**state, "rewritten_query": rewritten}
    
    def node_retrieve(self, state: RAGState) -> RAGState:
        """节点：向量检索"""
        query = state["rewritten_query"] or state["query"]
        available_kbs = state.get("available_kbs")
        
        # 获取查询中的有意义关键词（去除常见词）
        common_words = ['制度', '规定', '政策', '流程', '公司', '信息', '查询']
        query_keywords = []
        
        # 尝试分词提取关键词
        try:
            import jieba
            words = jieba.lcut(query)
            query_keywords = [w.lower() for w in words if len(w) > 1 and w not in common_words]
        except:
            # 如果没有jieba，使用简单的字符提取
            for char in query:
                if char.strip() and not char.isdigit():
                    query_keywords.append(char.lower())
        
        # 如果分词后没有关键词，回退到字符级别
        if not query_keywords:
            query_keywords = [c.lower() for c in query if c.strip() and not c.isdigit()]
        
        # 确保至少有一些关键词
        if not query_keywords:
            query_keywords = list(query.lower())
        
        if self.document_subgraph:
            try:
                search_results = self.document_subgraph.search(query, top_k=self.max_recall, kb_ids=available_kbs)
                if search_results:
                    # 过滤与查询关键词相关的文档
                    filtered_results = []
                    for doc in search_results:
                        doc_lower = doc.lower()
                        # 计算文档与查询的相关度
                        match_count = sum(1 for kw in query_keywords if kw in doc_lower)
                        
                        # 检查查询中的每个主要词汇是否都在文档中
                        query_vocab = [w for w in query_keywords if len(w) >= 2]
                        all_words_present = all(w in doc_lower for w in query_vocab) if query_vocab else True
                        
                        if match_count >= 1 and all_words_present:
                            filtered_results.append(doc)
                    
                    docs = filtered_results if filtered_results else search_results[:2]
                    scores = [0.9] * len(docs)
                    avg_score = sum(scores) / len(scores) if scores else 0
                    return {**state,
                            "retrieved_docs": docs,
                            "relevance_scores": scores,
                            "avg_relevance": avg_score}
            except Exception as e:
                pass
        
        # 兜底逻辑：对知识库进行简单的关键词匹配过滤
        docs = []
        scores = []
        for doc_info in self.knowledge_base:
            doc_content = doc_info.get("content", "")
            doc_lower = doc_content.lower()
            
            # 计算关键词匹配度
            match_count = sum(1 for kw in query_keywords if kw in doc_lower)
            
            # 检查查询中的每个主要词汇是否都在文档中
            query_vocab = [w for w in query_keywords if len(w) >= 2]
            all_words_present = all(w in doc_lower for w in query_vocab) if query_vocab else True
            
            if match_count >= 1 and all_words_present:
                docs.append(doc_content)
                scores.append(min(0.9, 0.5 + match_count * 0.1))
        
        # 如果没有匹配的文档，返回空的
        if not docs:
            return {**state,
                    "retrieved_docs": [],
                    "relevance_scores": [],
                    "avg_relevance": 0}
        
        # 按相关性排序，取前max_recall条
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        docs, scores = zip(*doc_score_pairs) if doc_score_pairs else ([], [])
        docs = list(docs)[:self.max_recall]
        scores = list(scores)[:self.max_recall]
        
        avg_score = sum(scores) / len(scores) if scores else 0

        return {**state,
                "retrieved_docs": docs,
                "relevance_scores": scores,
                "avg_relevance": avg_score}
    
    def node_evaluate_relevance(self, state: RAGState) -> RAGState:
        """节点：相关度评估"""
        avg_score = state["avg_relevance"]
        need_web = avg_score < self.threshold
        
        return {**state, "need_web_search": need_web, "has_web_search": False}
    
    def node_build_context(self, state: RAGState) -> RAGState:
        """节点：构建上下文"""
        docs = state["retrieved_docs"] or []
        query = state.get("query", "") or ""
        
        # 根据查询关键词过滤相关文档
        query_keywords = []
        for char in query:
            if char.strip():
                query_keywords.append(char)
        
        # 计算每篇文档与查询的相关性
        doc_scores = []
        for doc in docs:
            doc_lower = doc.lower()
            score = sum(1 for kw in query_keywords if kw.lower() in doc_lower)
            # 对于连续匹配的关键词给予更高分数
            for keyword in query_keywords:
                if keyword.lower() in doc_lower:
                    score += 2
            doc_scores.append((doc, score))
        
        # 按相关性排序，只保留分数最高的内容
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 根据相关性阈值过滤，保留与查询最相关的内容
        min_score = 1
        relevant_docs = [doc for doc, score in doc_scores if score >= min_score]
        
        # 如果过滤后内容太多，进一步限制
        if not relevant_docs and docs:
            # 如果没有匹配度足够高的，至少返回最相关的前2条
            relevant_docs = [docs[0]] if docs else []
        
        # 构建上下文，只包含相关部分
        if relevant_docs:
            context_parts = []
            for doc in relevant_docs[:3]:  # 最多只取3条最相关的
                # 清理文档内容，移除markdown格式符号
                doc_clean = doc.strip()
                doc_clean = doc_clean.replace("# ", "").replace("## ", "").replace("**", "")
                context_parts.append(doc_clean)
            context = "\n\n".join(context_parts)
        else:
            context = "\n".join([f"- {doc}" for doc in docs[:2]]) if docs else "未找到相关信息"
        
        return {**state, "context": context}
    
    async def node_synthesize_answer(self, state: RAGState) -> RAGState:
        """节点：答案合成"""
        if not self.enable_synthesis:
            answer = state["context"] or "未找到相关信息"
            return {**state, "final_answer": answer, "success": True}
        
        query = state["query"]
        context = state["context"] or ""
        has_web_search = state.get("has_web_search", False)
        avg_relevance = state.get("avg_relevance", 0)
        
        if self.llm:
            try:
                prompt = f"""基于以下知识库内容回答问题：

知识库内容：
{context}

问题：{query}

请给出简洁准确的回答："""
                response = await self.llm.ainvoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)
                answer = answer.strip()
            except Exception:
                answer = f"基于知识库：{context[:200]}..."
        else:
            answer = f"基于知识库（相关度:{avg_relevance:.2f}）：\n{context}"
        
        # 添加信息来源提示
        if has_web_search:
            # 当使用了Web搜索时，添加提醒
            source_note = f"\n\n⚠️ 注意：由于知识库匹配度较低（相关度: {avg_relevance:.2f}），部分信息来源于网络搜索，建议核实准确性。"
            answer = answer + source_note
        
        return {**state, "final_answer": answer, "success": True}
    
    def route_after_evaluation(self, state: RAGState) -> str:
        """条件路由：相关度评估后决定是否降级"""
        if state["need_web_search"]:
            return "web_search"
        return "build_context"
    
    def build_graph(self, web_search_node=None):
        """构建RAG子图"""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        graph = StateGraph(RAGState)
        
        graph.add_node("rewrite_query", self.node_rewrite_query)
        graph.add_node("retrieve", self.node_retrieve)
        graph.add_node("evaluate_relevance", self.node_evaluate_relevance)
        graph.add_node("build_context", self.node_build_context)
        graph.add_node("synthesize_answer", self.node_synthesize_answer)
        
        if web_search_node:
            graph.add_node("web_search", self._web_search_and_merge)
        
        graph.set_entry_point("rewrite_query")
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("retrieve", "evaluate_relevance")
        
        graph.add_conditional_edges(
            "evaluate_relevance",
            self.route_after_evaluation,
            {"web_search": "web_search" if web_search_node else END, 
             "build_context": "build_context"}
        )
        
        graph.add_edge("build_context", "synthesize_answer")
        graph.add_edge("synthesize_answer", END)
        
        if web_search_node:
            graph.add_edge("web_search", "build_context")
        
        return graph.compile()
    
    def _web_search_and_merge(self, state: RAGState) -> RAGState:
        """执行Web搜索并合并结果到上下文，设置标记"""
        query = state["query"]
        
        try:
            # 执行Web搜索
            from tools import web_search_tool
            result = web_search_tool._run(query)
            
            # 合并搜索结果到上下文
            web_context = result.strip()
            
            # 获取现有的知识库检索结果
            retrieved_docs = state.get("retrieved_docs", [])
            if retrieved_docs:
                # 将Web搜索结果添加到现有文档后面
                retrieved_docs.append(f"【Web搜索结果】\n{web_context}")
            else:
                retrieved_docs = [f"【Web搜索结果】\n{web_context}"]
            
            # 设置标记
            return {**state, 
                    "retrieved_docs": retrieved_docs,
                    "has_web_search": True,
                    "avg_relevance": state.get("avg_relevance", 0)}  # 保持原相关度
        except Exception as e:
            print(f"DEBUG: Web搜索失败: {e}")
            # 如果Web搜索失败，继续使用知识库结果
            return {**state, "has_web_search": False}
    
    async def run(self, query: str, web_search_func=None, available_kbs=None) -> Dict[str, Any]:
        """执行RAG子图"""
        initial_state: RAGState = {
            "query": query,
            "rewritten_query": None,
            "retrieved_docs": None,
            "relevance_scores": None,
            "avg_relevance": 0.0,
            "need_web_search": False,
            "context": None,
            "final_answer": None,
            "error": None,
            "success": False,
            "start_time": time.time(),
            "available_kbs": available_kbs
        }
        
        state = self.node_rewrite_query(initial_state)
        state = self.node_retrieve(state)
        state = self.node_evaluate_relevance(state)
        
        if state["need_web_search"] and web_search_func:
            web_result = web_search_func(query)
            return {
                "success": True,
                "result": web_result.get("result"),
                "degraded": True,
                "relevance": state["avg_relevance"]
            }
        
        state = self.node_build_context(state)
        state = await self.node_synthesize_answer(state)
        
        return {
            "success": state.get("success", False),
            "result": state.get("final_answer"),
            "relevance": state.get("avg_relevance"),
            "degraded": False
        }


# ==================== 4. Web搜索子图实现 ====================

class WebSearchSubGraph:
    """Web搜索子智能体图 - 实现联网搜索完整流程"""

    def __init__(self, api_key: str = None):
        self.timeout = web_search_config.web_search_timeout
        self.results_count = web_search_config.search_results_count
        self.trust_threshold = web_search_config.trust_threshold
        self.enable_cache = web_search_config.enable_search_cache
        self.cache_ttl = web_search_config.cache_ttl_minutes

        self.tavily_api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.bing_api_key = os.getenv("BING_SEARCH_API_KEY")

        self.llm = None
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception:
                pass
        
        # 内存缓存
        self._search_cache = {}
        self._cache_timestamps = {}

    async def _search_tavily(self, query: str) -> list:
        """使用Tavily API搜索"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={"query": query, "max_results": self.results_count},
                    headers={"Authorization": f"Bearer {self.tavily_api_key}"}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return [
                        {"title": r.get("title", ""), "snippet": r.get("content", ""), "url": r.get("url", "")}
                        for r in data.get("results", [])
                    ]
        except Exception as e:
            pass
        return []

    async def _search_bing(self, query: str) -> list:
        """使用Bing Search API搜索"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    params={"q": query, "count": self.results_count},
                    headers={"Ocp-Apim-Subscription-Key": self.bing_api_key}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return [
                        {"title": r.get("name", ""), "snippet": r.get("snippet", ""), "url": r.get("url", "")}
                        for r in data.get("webPages", {}).get("value", [])
                    ]
        except Exception as e:
            pass
        return []

    def _get_cache(self, query: str) -> list:
        """获取缓存结果"""
        if not self.enable_cache:
            return None
        cache_key = query.lower().strip()
        if cache_key in self._search_cache:
            timestamp = self._cache_timestamps[cache_key]
            if (time.time() - timestamp) < self.cache_ttl * 60:
                return self._search_cache[cache_key]
            # 过期缓存清理
            del self._search_cache[cache_key]
            del self._cache_timestamps[cache_key]
        return None

    def _set_cache(self, query: str, results: list):
        """设置缓存结果"""
        if self.enable_cache and results:
            cache_key = query.lower().strip()
            self._search_cache[cache_key] = results
            self._cache_timestamps[cache_key] = time.time()

    def node_search(self, state: WebSearchState) -> WebSearchState:
        """节点：执行搜索（同步包装，内部调用异步实现）"""
        if self.tavily_api_key or self.bing_api_key:
            # 有 API Key，调用真实搜索
            return asyncio.run(self.node_search_async(state))
        
        # 无 API Key，返回模拟结果
        query = state["query"]
        elapsed = time.time() - state["start_time"]

        if elapsed > self.timeout:
            return {**state, "timeout_triggered": True, "error": f"搜索超时（{self.timeout}s）"}

        mock_results = [
            {"title": f"搜索结果{i+1}", "snippet": f"关于{query}的相关信息...", "url": f"https://example.com/{i+1}"}
            for i in range(self.results_count)
        ]

        return {**state, "search_results": mock_results, "timeout_triggered": False}

    async def node_search_async(self, state: WebSearchState) -> WebSearchState:
        """异步节点：执行搜索（支持缓存和并行搜索）"""
        query = state["query"]
        elapsed = time.time() - state["start_time"]

        if elapsed > self.timeout:
            return {**state, "timeout_triggered": True, "error": f"搜索超时（{self.timeout}s）"}

        # 先检查缓存
        cached_results = self._get_cache(query)
        if cached_results:
            return {**state, "search_results": cached_results, "timeout_triggered": False, "cache_hit": True}

        # 并行调用多个搜索API
        tasks = []
        if self.tavily_api_key:
            tasks.append(self._search_tavily(query))
        if self.bing_api_key:
            tasks.append(self._search_bing(query))

        results = []
        if tasks:
            # 并行执行，取第一个返回的结果（最快的API）
            done, _ = await asyncio.wait(tasks, timeout=self.timeout, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    task_results = task.result()
                    if task_results:
                        results = task_results[:self.results_count]
                        break
                except Exception:
                    continue

        if not results:
            results = [
                {"title": f"搜索结果{i+1}", "snippet": f"关于{query}的相关信息...", "url": f"https://example.com/{i+1}"}
                for i in range(self.results_count)
            ]

        # 设置缓存
        self._set_cache(query, results)

        return {**state, "search_results": results, "timeout_triggered": False, "cache_hit": False}
    
    def node_summarize(self, state: WebSearchState) -> WebSearchState:
        """节点：结果摘要"""
        if state["timeout_triggered"]:
            return {**state, "final_answer": state.get("error", "搜索超时"), "success": False}
        
        results = state["search_results"] or []
        
        if self.llm:
            try:
                results_text = "\n".join([f"- {r.get('title', '')}: {r.get('snippet', '')}" for r in results if r.get('snippet')])
                prompt = f"请总结以下搜索结果的关键信息：\n{results_text}"
                response = self.llm.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.3
                )
                summary = response.choices[0].message.content.strip()
            except Exception:
                summary = self._clean_text(" | ".join([r.get("snippet", "") for r in results[:3] if r.get('snippet')]))
        else:
            summary = self._clean_text(" | ".join([r.get("snippet", "") for r in results[:3] if r.get('snippet')]))
        
        return {**state, "summarized_results": summary}
    
    def _clean_text(self, text: str) -> str:
        """清理文本中的乱码、重复和多余字符"""
        import re
        
        # 移除多余的换行符和空格
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # 移除重复的模式
        text = re.sub(r'(关于[\u4e00-\u9fa5]+新闻的相关信息\.\.\.\s*\|?\s*)+', '关于相关新闻信息... ', text)
        text = re.sub(r'(查找[\u4e00-\u9fa5]+的相关信息\.\.\.\s*\|?\s*)+', '', text)
        
        # 移除连续重复的内容
        text = re.sub(r'([\u4e00-\u9fa5]{3,})\1+', r'\1', text)
        
        # 移除无效字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.\,\!\?\:\;\(\)\-\_\|\[\]\{\}【】《》]', '', text)
        
        # 清理首尾空格
        text = text.strip()
        
        return text
    
    def _get_domain_trust(self, url: str) -> float:
        """获取域名可信度分数"""
        import re
        
        # 提取域名
        domain_pattern = r"(?:https?://)?(?:www\.)?([^/]+)"
        match = re.match(domain_pattern, url.lower())
        if not match:
            return 0.5  # 无法识别的域名，默认中等可信度
        
        domain = match.group(1)
        
        # 检查是否在可信域名列表中
        for trusted_domain, score in web_search_config.trusted_domains.items():
            if trusted_domain in domain:
                return score
        
        # 检查是否为 IP 地址
        if re.match(r"^\d+\.\d+\.\d+\.\d+$", domain):
            return 0.4  # IP地址可信度较低
        
        # 检查是否为知名 TLD
        tld = domain.split('.')[-1] if '.' in domain else ''
        if tld in ['gov', 'edu']:
            return 0.9
        elif tld in ['com', 'org', 'net']:
            return 0.65
        else:
            return 0.5  # 其他域名默认中等可信度

    def _calculate_consistency(self, results: list) -> float:
        """计算多源一致性分数"""
        if len(results) < 2:
            return 0.5  # 结果太少，无法评估一致性
        
        snippets = [r.get("snippet", "").lower() for r in results if r.get("snippet")]
        if not snippets:
            return 0.5
        
        # 计算两两相似度
        from difflib import SequenceMatcher
        total_similarity = 0
        pair_count = 0
        
        for i in range(len(snippets)):
            for j in range(i + 1, len(snippets)):
                similarity = SequenceMatcher(None, snippets[i], snippets[j]).ratio()
                total_similarity += similarity
                pair_count += 1
        
        if pair_count == 0:
            return 0.5
        
        avg_similarity = total_similarity / pair_count
        return min(0.5 + avg_similarity * 0.5, 1.0)

    def _estimate_timestamp(self, snippet: str) -> float:
        """从摘要中估计时效性"""
        import re
        
        # 查找年份
        year_pattern = r"(20[0-9]{2}|19[0-9]{2})"
        match = re.search(year_pattern, snippet)
        if match:
            year = int(match.group(1))
            current_year = 2026  # 当前年份
            if year >= current_year:
                return 1.0  # 最新信息
            elif year >= current_year - 2:
                return 0.85  # 较新信息
            elif year >= current_year - 5:
                return 0.65  # 中等时效
            else:
                return 0.4  # 过时信息
        
        # 查找时间关键词
        time_keywords = {
            "最新": 0.95,
            "近日": 0.9,
            "今年": 0.85,
            "去年": 0.7,
            "近日": 0.9,
            "刚刚": 0.95,
            "刚刚发布": 0.95,
            "最新消息": 0.95,
            "最新报道": 0.95,
            "最近": 0.85,
            "近期": 0.85,
            "日前": 0.8,
            "目前": 0.85,
            "现在": 0.85,
            "过去": 0.6,
            "曾经": 0.5,
            "以前": 0.5,
            "历史": 0.4,
            "去年": 0.7,
            "前年": 0.55,
        }
        
        for keyword, score in time_keywords.items():
            if keyword in snippet:
                return score
        
        # 默认中等时效性
        return 0.7

    def node_evaluate_trust(self, state: WebSearchState) -> WebSearchState:
        """节点：可信度评估（基于来源域名、多源一致性、时效性）"""
        if state["timeout_triggered"]:
            return state
        
        results = state["search_results"] or []
        if not results:
            return {**state, "trust_score": 0.3, "trust_details": {"error": "无搜索结果"}}
        
        # 1. 域名可信度评估
        domain_scores = []
        for result in results:
            url = result.get("url", "")
            domain_score = self._get_domain_trust(url)
            domain_scores.append(domain_score)
        
        avg_domain_trust = sum(domain_scores) / len(domain_scores) if domain_scores else 0.5
        
        # 2. 多源一致性评估
        consistency_score = self._calculate_consistency(results)
        
        # 3. 时效性评估
        timestamps = []
        for result in results:
            snippet = result.get("snippet", "")
            timestamp_score = self._estimate_timestamp(snippet)
            timestamps.append(timestamp_score)
        
        avg_timestamp = sum(timestamps) / len(timestamps) if timestamps else 0.7
        
        # 4. 综合评估（加权平均）
        # 域名可信度权重：40%，多源一致性权重：30%，时效性权重：30%
        trust_score = (avg_domain_trust * 0.4) + (consistency_score * 0.3) + (avg_timestamp * 0.3)
        
        # 添加结果数量奖励
        result_bonus = min(len(results) * 0.02, 0.1)
        trust_score = min(trust_score + result_bonus, 1.0)
        
        # 记录评估详情
        trust_details = {
            "domain_trust": round(avg_domain_trust, 2),
            "consistency": round(consistency_score, 2),
            "timeliness": round(avg_timestamp, 2),
            "result_count": len(results),
            "final_score": round(trust_score, 2)
        }
        
        return {**state, "trust_score": trust_score, "trust_details": trust_details}
    
    def node_format_result(self, state: WebSearchState) -> WebSearchState:
        """节点：格式化结果（结构化文本输出）"""
        if state["timeout_triggered"]:
            return {**state, "final_answer": state.get("error", "搜索超时"), "success": False}
        
        results = state.get("search_results", [])
        
        import re
        import urllib.parse
        import html
        
        # 清理单条结果的函数
        def clean_result(result):
            title = result.get("title", "").strip()
            snippet = result.get("snippet", "").strip()
            
            # 1. 解码 HTML 实体
            title = html.unescape(title)
            snippet = html.unescape(snippet)
            
            # 2. 解码 URL 编码字符
            title = urllib.parse.unquote(title)
            snippet = urllib.parse.unquote(snippet)
            
            # 3. 移除 URL
            title = re.sub(r'https?://[\w\-._~:/?#[\]@!$&\'()*+,;=%]+', '', title)
            snippet = re.sub(r'https?://[\w\-._~:/?#[\]@!$&\'()*+,;=%]+', '', snippet)
            
            # 4. 移除格式符号和HTML标签
            title = re.sub(r'[*`>#~|[\](){}<>]', '', title)
            snippet = re.sub(r'[*`>#~|[\](){}<>]', '', snippet)
            
            # 5. 移除控制字符和重复标点
            title = re.sub(r'[\x00-\x1F\x7F]', '', title)
            snippet = re.sub(r'[\x00-\x1F\x7F]', '', snippet)
            title = re.sub(r'([.,;:!?]){2,}', r'\1', title)
            snippet = re.sub(r'([.,;:!?]){2,}', r'\1', snippet)
            title = re.sub(r' +', ' ', title).strip()
            snippet = re.sub(r' +', ' ', snippet).strip()
            
            return title, snippet
        
        # 处理搜索结果
        cleaned_items = []
        seen_snippets = set()
        
        for result in results[:6]:  # 最多处理6条
            title, snippet = clean_result(result)
            
            # 过滤垃圾内容
            if not snippet or len(snippet) < 10:
                continue
            if "Missing:" in snippet or "大 模型" in snippet:
                continue
            
            # 去重
            snippet_hash = snippet.lower()[:80]
            if snippet_hash in seen_snippets:
                continue
            seen_snippets.add(snippet_hash)
            
            # 提取关键内容
            if snippet and len(snippet) > 15:
                # 优先使用snippet，只取前150字符
                content = snippet[:150]
                if len(snippet) > 150:
                    content = content.rsplit('，', 1)[0].rsplit('。', 1)[0] + "..."
                cleaned_items.append(content)
        
        # 构建最终回答
        if cleaned_items:
            lines = ["📰 搜索结果"]
            for i, item in enumerate(cleaned_items, 1):
                lines.append(f"{i}. {item}")
            answer = "\n".join(lines)
        else:
            answer = "暂无相关搜索结果"
        
        return {**state, "final_answer": answer.strip(), "success": True, "trust_level": "高可信度"}
    
    def _extract_timestamp(self, result: dict) -> str:
        """从搜索结果中提取时间信息"""
        snippet = result.get("snippet", "")
        title = result.get("title", "")
        
        import re
        
        # 查找日期模式
        date_patterns = [
            r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})[日号]?",  # YYYY-MM-DD 或中文格式
            r"(\d{1,2})[-/月](\d{1,2})[日号]?",  # MM-DD
            r"(\d{4})年(\d{1,2})月",  # YYYY年MM月
            r"(\d{2})小时前|(\d{1,2})天前|(\d{1,2})周前|(\d{1,2})月前",  # 相对时间
            r"昨天|前天|今日|今日|近日|刚刚|最新",  # 中文时间词
        ]
        
        text = f"{title} {snippet}"
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return ""
    
    def _extract_key_points(self, text: str) -> str:
        """提取文本中的关键信息（简化版）"""
        import re
        # 移除多余字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？、：；（）【】《》]', '', text)
        # 截取前80个字符作为关键点
        return text[:80] + "..." if len(text) > 80 else text
    
    def build_graph(self):
        """构建Web搜索子图"""
        if not LANGGRAPH_AVAILABLE:
            return None
        
        graph = StateGraph(WebSearchState)
        
        graph.add_node("search", self.node_search)
        graph.add_node("summarize", self.node_summarize)
        graph.add_node("evaluate_trust", self.node_evaluate_trust)
        graph.add_node("format_result", self.node_format_result)
        
        graph.set_entry_point("search")
        graph.add_edge("search", "summarize")
        graph.add_edge("summarize", "evaluate_trust")
        graph.add_edge("evaluate_trust", "format_result")
        graph.add_edge("format_result", END)
        
        return graph.compile()
    
    def run(self, query: str) -> Dict[str, Any]:
        """执行Web搜索子图"""
        initial_state: WebSearchState = {
            "query": query,
            "search_results": None,
            "summarized_results": None,
            "trust_score": 0.0,
            "final_answer": None,
            "error": None,
            "success": False,
            "timeout_triggered": False,
            "start_time": time.time()
        }
        
        if LANGGRAPH_AVAILABLE:
            try:
                graph = self.build_graph()
                if graph:
                    result = graph.invoke(initial_state)
                    return {
                        "success": result.get("success", False),
                        "result": result.get("final_answer"),
                        "trust_score": result.get("trust_score"),
                        "error": result.get("error")
                    }
            except Exception as e:
                print(f"LangGraph执行失败: {e}")
        
        return self._run_fallback(query, initial_state)
    
    async def arun(self, query: str) -> Dict[str, Any]:
        """异步执行Web搜索子图"""
        initial_state: WebSearchState = {
            "query": query,
            "search_results": None,
            "summarized_results": None,
            "trust_score": 0.0,
            "final_answer": None,
            "error": None,
            "success": False,
            "timeout_triggered": False,
            "start_time": time.time()
        }
        
        if LANGGRAPH_AVAILABLE:
            try:
                graph = self.build_graph()
                if graph:
                    result = await graph.ainvoke(initial_state)
                    return {
                        "success": result.get("success", False),
                        "result": result.get("final_answer"),
                        "trust_score": result.get("trust_score"),
                        "error": result.get("error")
                    }
            except Exception as e:
                print(f"LangGraph异步执行失败: {e}")
        
        # 异步降级执行
        return await self._arun_fallback(query, initial_state)
    
    async def _arun_fallback(self, query: str, state: WebSearchState) -> Dict[str, Any]:
        """异步降级执行"""
        state = self.node_search(state)
        state = self.node_summarize(state)
        state = self.node_evaluate_trust(state)
        state = self.node_format_result(state)
        
        return {
            "success": state.get("success", False),
            "result": state.get("final_answer"),
            "trust_score": state.get("trust_score"),
            "error": state.get("error")
        }
    
    def _run_fallback(self, query: str, state: WebSearchState) -> Dict[str, Any]:
        """降级执行"""
        state = self.node_search(state)
        state = self.node_summarize(state)
        state = self.node_evaluate_trust(state)
        state = self.node_format_result(state)
        
        return {
            "success": state.get("success", False),
            "result": state.get("final_answer"),
            "trust_score": state.get("trust_score"),
            "error": state.get("error")
        }


# ==================== 5. 标准化Tool封装 ====================

class BaseTool:
    """工具基类 - 所有工具必须继承"""

    name: str = "base_tool"
    description: str = "基础工具类"

    def _validate_input(self, *args, **kwargs) -> bool:
        return True

    def _sanitize_input(self, query: str) -> str:
        if not query:
            return ""
        dangerous_patterns = ['<script>', 'javascript:', 'onerror=', 'onload=']
        sanitized = query
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, '')
        return sanitized.strip()

    def _validate_sql_query(self, query: str) -> tuple:
        """验证SQL查询安全性，使用sqlparse进行AST分析，返回 (是否安全, 错误信息)"""
        if not query:
            return False, "查询不能为空"
        
        # 允许的SQL语句类型
        ALLOWED_STATEMENTS = {'SELECT', 'SHOW', 'DESCRIBE', 'EXPLAIN'}
        
        # 危险的SQL语句类型（白名单之外的都禁止）
        DANGEROUS_STATEMENTS = {
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 
            'UPDATE', 'EXEC', 'EXECUTE', 'GRANT', 'REVOKE', 'DENY', 
            'BACKUP', 'RESTORE', 'RENAME', 'TRANSACTION', 'COMMIT', 'ROLLBACK'
        }
        
        try:
            # 使用sqlparse解析SQL
            parsed = sqlparse.parse(query)
            
            for statement in parsed:
                # 获取语句类型
                stmt_type = None
                for token in statement.tokens:
                    if token.ttype in (T.DML, T.DDL, T.Keyword.DML, T.Keyword.DDL, T.Keyword):
                        token_str = token.normalized.upper()
                        if token_str in ALLOWED_STATEMENTS:
                            stmt_type = token_str
                            break
                        elif token_str in DANGEROUS_STATEMENTS:
                            return False, f"禁止执行危险SQL操作: {token_str}"
                
                # 如果无法识别语句类型，检查是否包含危险模式
                if stmt_type is None:
                    # 检查是否有隐藏的危险操作（如注释中的DROP）
                    query_upper = query.upper()
                    for dangerous in DANGEROUS_STATEMENTS:
                        if dangerous in query_upper:
                            return False, f"检测到危险SQL关键字: {dangerous}"
                    return False, "无法识别的SQL语句类型"
                
                # 检查是否包含多个语句（SQL注入常见手法）
                if len(parsed) > 1:
                    return False, "不允许执行多条SQL语句"
                
                # 检查是否包含注释（可能隐藏恶意代码）
                for token in statement.flatten():
                    if token.ttype in (T.Comment.Single, T.Comment.Multiline):
                        # 检查注释中是否包含危险关键字
                        comment_upper = token.normalized.upper()
                        for dangerous in DANGEROUS_STATEMENTS:
                            if dangerous in comment_upper:
                                return False, f"注释中包含危险SQL操作: {dangerous}"
            
            return True, ""
            
        except Exception as e:
            return False, f"SQL解析失败: {str(e)}"

    def _run(self, *args, **kwargs) -> str:
        raise NotImplementedError

    async def _arun(self, *args, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._run(*args, **kwargs))


class QuerySqlTool(BaseTool):
    """SQL查询工具 - 封装SQL子图"""

    name = "query_sql_tool"
    description = "自然语言转SQL，自动生成、校验、执行并返回结构化结果"

    def __init__(self):
        # 创建LLM实例（使用 langchain 的 ChatOpenAI）
        llm = None
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        # 优先使用DashScope的base_url，如果配置了DASHSCOPE_API_KEY
        if os.getenv("DASHSCOPE_API_KEY"):
            base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        else:
            base_url = os.getenv("OPENAI_API_BASE")
        
        if api_key:
            try:
                from langchain.chat_models import ChatOpenAI
                llm = ChatOpenAI(
                    model=os.getenv("LLM_MODEL_NAME", "qwen3.6-flash"),
                    temperature=float(os.getenv("SQL_TEMPERATURE", "0.1")),
                    max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
                    api_key=api_key,
                    base_url=base_url
                )
            except Exception as e:
                pass
        self.subgraph = SQLSubGraph(llm=llm)

    def _run(self, query: str, user_id: str = None, department: str = None) -> str:
        if not self._validate_input(query):
            return json.dumps({"success": False, "error": "输入验证失败"}, ensure_ascii=False)

        query = self._sanitize_input(query)

        result = asyncio.run(self.subgraph.run(query, department=department))

        return json.dumps({
            "success": result.get("success", False),
            "result": result.get("result"),
            "sql": result.get("sql"),
            "error": result.get("error")
        }, ensure_ascii=False)

    async def _arun(self, query: str, user_id: str = None, department: str = None) -> str:
        """异步执行"""
        if not self._validate_input(query):
            return json.dumps({"success": False, "error": "输入验证失败"}, ensure_ascii=False)

        query = self._sanitize_input(query)

        result = await self.subgraph.run(query, department=department)

        return json.dumps({
            "success": result.get("success", False),
            "result": result.get("result"),
            "sql": result.get("sql"),
            "error": result.get("error")
        }, ensure_ascii=False)


class RagSearchTool(BaseTool):
    """RAG检索工具 - 封装RAG子图"""

    name = "rag_search_tool"
    description = "企业内部知识库检索，查询优化→向量检索→相关度评估→答案生成"

    def __init__(self, web_search_tool=None, permission_guard=None):
        self.subgraph = RAGSubGraph()
        self.web_search_tool = web_search_tool
        self.permission_guard = permission_guard

    def _run(self, query: str, user_id: str = None) -> str:
        if not self._validate_input(query):
            return json.dumps({"success": False, "error": "输入验证失败"}, ensure_ascii=False)

        query = self._sanitize_input(query)

        # 获取用户可访问的知识库列表
        available_kbs = None
        if user_id and self.permission_guard:
            available_kbs = self.permission_guard.perm_manager.get_user_available_kbs(user_id)
            if not available_kbs:
                return json.dumps({
                    "success": False,
                    "error": "您没有权限访问任何知识库",
                    "relevance": 0,
                    "degraded": False
                }, ensure_ascii=False)

        web_func = None
        if self.web_search_tool:
            web_func = lambda q: self.web_search_tool._run(q)

        # 传递可访问的知识库列表给子图
        result = asyncio.run(self.subgraph.run(query, web_func, available_kbs=available_kbs))

        return json.dumps({
            "success": result.get("success", False),
            "result": result.get("result"),
            "relevance": result.get("relevance"),
            "degraded": result.get("degraded", False)
        }, ensure_ascii=False)

    async def _arun(self, query: str, user_id: str = None) -> str:
        """异步执行"""
        if not self._validate_input(query):
            return json.dumps({"success": False, "error": "输入验证失败"}, ensure_ascii=False)

        query = self._sanitize_input(query)

        # 获取用户可访问的知识库列表
        available_kbs = None
        if user_id and self.permission_guard:
            available_kbs = self.permission_guard.perm_manager.get_user_available_kbs(user_id)
            if not available_kbs:
                return json.dumps({
                    "success": False,
                    "error": "您没有权限访问任何知识库",
                    "relevance": 0,
                    "degraded": False
                }, ensure_ascii=False)

        web_func = None
        if self.web_search_tool:
            web_func = lambda q: self.web_search_tool._run(q)

        # 传递可访问的知识库列表给子图
        result = await self.subgraph.run(query, web_func, available_kbs=available_kbs)

        return json.dumps({
            "success": result.get("success", False),
            "result": result.get("result"),
            "relevance": result.get("relevance"),
            "degraded": result.get("degraded", False)
        }, ensure_ascii=False)


class WebSearchTool(BaseTool):
    """Web搜索工具 - 封装Web搜索子图"""

    name = "web_search_tool"
    description = "联网搜索实时信息，自动摘要、可信度评估"

    def __init__(self, api_key: str = None):
        self.subgraph = WebSearchSubGraph()
        self.api_key = api_key or os.getenv("BING_SEARCH_API_KEY") or os.getenv("TAVILY_API_KEY")

    def _run(self, query: str, user_id: str = None) -> str:
        if not self._validate_input(query):
            return json.dumps({"success": False, "error": "输入验证失败"}, ensure_ascii=False)

        query = self._sanitize_input(query)

        result = self.subgraph.run(query)

        return json.dumps({
            "success": result.get("success", False),
            "result": result.get("result"),
            "trust_score": result.get("trust_score"),
            "error": result.get("error")
        }, ensure_ascii=False)

    async def _arun(self, query: str, user_id: str = None) -> str:
        """异步执行 - 使用 asyncio.to_thread 避免阻塞事件循环"""
        return await asyncio.to_thread(self._run, query, user_id)


# ==================== 6. 业务智能体工具包装 ====================

class AutomationTool(BaseTool):
    """自动化工具 - 封装AutomationAgent"""

    name = "automation_tool"
    description = "飞书自动化办公：创建任务、生成日报周报、查询数据"

    def __init__(self, feishu_client=None, llm=None):
        try:
            from automation_agent import AutomationAgent
            self.agent = AutomationAgent(feishu_client, llm)
            self.agent_available = True
        except ImportError:
            self.agent = None
            self.agent_available = False

    def _run(self, query: str, user_id: str = None) -> str:
        if not self._validate_input(query):
            return json.dumps({"success": False, "error": "输入验证失败"}, ensure_ascii=False)

        query = self._sanitize_input(query)

        if not self.agent_available:
            return json.dumps({
                "success": False,
                "error": "自动化代理不可用",
                "degraded": True
            }, ensure_ascii=False)

        try:
            import asyncio
            result = asyncio.run(self.agent.process_message(user_id or "unknown", "msg_001", query))
            return json.dumps({
                "success": result.get("status") == "success",
                "result": result.get("message"),
                "action": result.get("action"),
                "error": result.get("error")
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "degraded": True
            }, ensure_ascii=False)

    async def _arun(self, query: str, user_id: str = None) -> str:
        """异步执行 - 使用 asyncio.to_thread 避免阻塞事件循环"""
        return await asyncio.to_thread(self._run, query, user_id)


class DataAnalysisTool(BaseTool):
    """数据分析工具 - 封装DataAnalysisAgent"""

    name = "data_analysis_tool"
    description = "数据分析与报表：查询多维表格、生成统计报表、添加记录"

    def __init__(self, feishu_client=None, llm=None):
        try:
            from data_analysis_agent import DataAnalysisAgent
            self.agent = DataAnalysisAgent(feishu_client, llm)
            self.agent_available = True
        except ImportError:
            self.agent = None
            self.agent_available = False

    def _run(self, query: str, user_id: str = None) -> str:
        if not self._validate_input(query):
            return json.dumps({"success": False, "error": "输入验证失败"}, ensure_ascii=False)

        query = self._sanitize_input(query)

        if not self.agent_available:
            return json.dumps({
                "success": False,
                "error": "数据分析代理不可用",
                "degraded": True
            }, ensure_ascii=False)

        try:
            import asyncio
            result = asyncio.run(self.agent.analyze(user_id or "unknown", query))
            return json.dumps({
                "success": result.get("status") == "success",
                "result": result.get("result") or result.get("message"),
                "chart_config": result.get("chart_config"),
                "error": result.get("error")
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "degraded": True
            }, ensure_ascii=False)

    async def _arun(self, query: str, user_id: str = None) -> str:
        """异步执行 - 使用 asyncio.to_thread 避免阻塞事件循环"""
        return await asyncio.to_thread(self._run, query, user_id)


class MeetingTool(BaseTool):
    """会议工具 - 封装MeetingAgent"""

    name = "meeting_tool"
    description = "会议管理：处理会议转录、生成会议纪要、提取待办事项"

    def __init__(self, feishu_client=None, llm=None):
        try:
            from meeting_agent import MeetingAgent
            self.agent = MeetingAgent(feishu_client, llm)
            self.agent_available = True
        except ImportError:
            self.agent = None
            self.agent_available = False

    def _run(self, query: str, user_id: str = None) -> str:
        if not self._validate_input(query):
            return json.dumps({"success": False, "error": "输入验证失败"}, ensure_ascii=False)

        query = self._sanitize_input(query)

        if not self.agent_available:
            return json.dumps({
                "success": False,
                "error": "会议代理不可用",
                "degraded": True
            }, ensure_ascii=False)

        try:
            import asyncio
            mock_participants = [user_id] if user_id else []
            result = asyncio.run(self.agent.process_meeting_transcript(
                f"meeting_{int(time.time())}", query, mock_participants
            ))
            return json.dumps({
                "success": result.get("status") == "success",
                "meeting_id": result.get("meeting_id"),
                "minutes": result.get("minutes"),
                "todos": result.get("todos"),
                "error": result.get("error")
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e),
                "degraded": True
            }, ensure_ascii=False)

    async def _arun(self, query: str, user_id: str = None) -> str:
        """异步执行 - 使用 asyncio.to_thread 避免阻塞事件循环"""
        return await asyncio.to_thread(self._run, query, user_id)


# ==================== 7. 工具实例导出 ====================

web_search_tool = WebSearchTool()
rag_search_tool = RagSearchTool(web_search_tool=web_search_tool)
query_sql_tool = QuerySqlTool()
automation_tool = AutomationTool()
data_analysis_tool = DataAnalysisTool()
meeting_tool = MeetingTool()

tools_from_B = [
    query_sql_tool,
    rag_search_tool,
    web_search_tool,
    automation_tool,
    data_analysis_tool,
    meeting_tool,
]

tools_dict = {t.name: t._run for t in tools_from_B}


# ==================== 8. 子图实例导出（供高级使用）====================

sql_subgraph = SQLSubGraph()
rag_subgraph = RAGSubGraph()
web_search_subgraph = WebSearchSubGraph()
