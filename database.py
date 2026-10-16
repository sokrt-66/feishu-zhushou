"""
数据库连接模块 
支持MySQL、PostgreSQL、SQLite的异步操作
"""

import os
import time
import re
from typing import Dict, Any, Optional, List

try:
    import aiomysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import asyncpg
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    import aiosqlite
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

from config import sql_config


class AsyncDatabaseConnector:
    """异步数据库连接器 - 支持多种数据库类型"""

    def __init__(self):
        self.db_type = sql_config.db_type
        self.host = sql_config.db_host
        self.port = sql_config.db_port
        self.user = sql_config.db_user
        self.password = sql_config.db_password
        self.db_name = sql_config.db_name
        self.pool = None

    async def connect(self) -> bool:
        """建立数据库连接池"""
        try:
            if self.db_type == "mysql":
                if not MYSQL_AVAILABLE:
                    raise ImportError("需要安装 aiomysql: pip install aiomysql")
                self.pool = await aiomysql.create_pool(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    db=self.db_name,
                    charset='utf8mb4',
                    minsize=1,
                    maxsize=10
                )
            elif self.db_type == "postgresql":
                if not POSTGRESQL_AVAILABLE:
                    raise ImportError("需要安装 asyncpg: pip install asyncpg")
                self.pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    database=self.db_name,
                    min_size=1,
                    max_size=10
                )
            elif self.db_type == "sqlite":
                if not SQLITE_AVAILABLE:
                    raise ImportError("需要安装 aiosqlite: pip install aiosqlite")
                self.pool = await aiosqlite.connect(self.db_name)
                # 初始化示例表
                await self._init_sample_tables()
            else:
                raise ValueError(f"不支持的数据库类型: {self.db_type}")

            return True
        except Exception as e:
            print(f"数据库连接失败: {str(e)}")
            return False

    async def _init_sample_tables(self):
        """初始化SQLite示例表"""
        if self.db_type != "sqlite" or not self.pool:
            return
        
        try:
            async with self.pool.execute("""
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    gender TEXT,
                    department TEXT,
                    position TEXT,
                    hire_date TEXT,
                    salary INTEGER,
                    email TEXT,
                    phone TEXT,
                    status TEXT DEFAULT '在职'
                )
            """):
                pass
            
            async with self.pool.execute("""
                CREATE TABLE IF NOT EXISTS departments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dept_id TEXT UNIQUE,
                    name TEXT NOT NULL,
                    manager_id TEXT
                )
            """):
                pass
            
            # 插入示例数据（如果表为空）
            async with self.pool.execute("SELECT COUNT(*) FROM employees") as cursor:
                count = await cursor.fetchone()
                if count and count[0] == 0:
                    await self.pool.execute("""
                        INSERT INTO employees (name, gender, department, position, hire_date, salary, email, phone)
                        VALUES 
                        ('张三', '男', '技术部', '高级工程师', '2020-01-15', 15000, 'zhangsan@example.com', '13800138001'),
                        ('李四', '女', '市场部', '市场经理', '2021-03-20', 12000, 'lisi@example.com', '13800138002'),
                        ('王五', '男', '人事部', 'HR主管', '2019-06-10', 10000, 'wangwu@example.com', '13800138003'),
                        ('赵六', '女', '技术部', '工程师', '2022-08-01', 10000, 'zhaoliu@example.com', '13800138004'),
                        ('钱七', '男', '市场部', '销售代表', '2023-01-15', 8000, 'qianqi@example.com', '13800138005')
                    """)
                    await self.pool.commit()
                
            async with self.pool.execute("SELECT COUNT(*) FROM departments") as cursor:
                count = await cursor.fetchone()
                if count and count[0] == 0:
                    await self.pool.execute("""
                        INSERT INTO departments (dept_id, name, manager_id)
                        VALUES 
                        ('dept_1', '技术部', 'admin'),
                        ('dept_2', '市场部', 'admin'),
                        ('dept_3', '人事部', 'admin')
                    """)
                    await self.pool.commit()
                    
        except Exception as e:
            print(f"初始化示例表失败: {str(e)}")

    async def close(self):
        """关闭数据库连接池"""
        if self.pool:
            try:
                # 检查事件循环是否仍然运行
                import asyncio
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    if self.db_type == "sqlite":
                        await self.pool.close()
                    else:
                        self.pool.close()
                        await self.pool.wait_closed()
            except:
                pass
            self.pool = None
    
    def sync_close(self):
        """同步关闭数据库连接池（用于应用关闭时）"""
        if self.pool:
            try:
                if self.db_type == "sqlite":
                    # SQLite不需要特殊处理
                    pass
                else:
                    # 尝试同步关闭，忽略事件循环状态
                    try:
                        import asyncio
                        loop = asyncio.get_event_loop()
                        if not loop.is_closed():
                            asyncio.create_task(self.close())
                    except:
                        pass
            except:
                pass
            self.pool = None

    async def execute_update(self, sql: str, timeout: int = 30) -> Dict[str, Any]:
        """异步执行SQL更新（支持INSERT、UPDATE）"""
        start_time = time.time()

        if not self.pool and not await self.connect():
            return {"success": False, "error": "无法连接数据库"}

        sql_upper = sql.strip().upper()
        
        # 只允许INSERT和UPDATE操作
        if not sql_upper.startswith('INSERT') and not sql_upper.startswith('UPDATE'):
            return {"success": False, "error": "只允许INSERT和UPDATE操作"}

        # 检查是否有多语句
        sql_stripped = sql.strip()
        if sql_stripped.count(';') > 1 or (sql_stripped.count(';') == 1 and not sql_stripped.endswith(';')):
            return {"success": False, "error": "禁止多语句操作"}

        try:
            if self.db_type == "sqlite":
                async with self.pool.execute(sql) as cursor:
                    await self.pool.commit()
                    affected_rows = cursor.rowcount
                    last_insert_id = cursor.lastrowid if hasattr(cursor, 'lastrowid') else None
            else:
                async with self.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute(sql)
                        affected_rows = cursor.rowcount
                        last_insert_id = cursor.lastrowid if hasattr(cursor, 'lastrowid') else None
                        await conn.commit()

            elapsed = time.time() - start_time
            return {
                "success": True,
                "affected_rows": affected_rows,
                "last_insert_id": last_insert_id,
                "execution_time": round(elapsed, 2)
            }

        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "execution_time": round(elapsed, 2)
            }

    async def execute_query(self, sql: str, timeout: int = 30) -> Dict[str, Any]:
        """异步执行SQL查询（仅支持SELECT）"""
        start_time = time.time()

        if not self.pool and not await self.connect():
            return {"success": False, "error": "无法连接数据库"}

        sql_upper = sql.strip().upper()
        if not sql_upper.startswith('SELECT'):
            return {"success": False, "error": "只允许SELECT查询操作"}

        # 检查是否有多语句（分号后还有内容）
        sql_stripped = sql.strip()
        if sql_stripped.count(';') > 1 or (sql_stripped.count(';') == 1 and not sql_stripped.endswith(';')):
            return {"success": False, "error": "禁止多语句查询"}

        dangerous_patterns = [
            r'\bINTO\s+OUTFILE\b', r'\bLOAD_FILE\b',
            r'\bBENCHMARK\b', r'\bSLEEP\b',
            r'\bINFORMATION_SCHEMA\b', r'\bPROCESSLIST\b'
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return {"success": False, "error": f"检测到危险操作: {pattern}"}

        try:
            if self.db_type == "sqlite":
                async with self.pool.execute(sql) as cursor:
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    results = await cursor.fetchall()
            else:
                async with self.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        if self.db_type == "mysql":
                            await cursor.execute(f"SET MAX_EXECUTION_TIME={timeout * 1000}")

                        await cursor.execute(sql)

                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                        results = await cursor.fetchall()

            data = []
            for row in results:
                row_dict = {}
                for i, col in enumerate(columns):
                    value = row[i]
                    # 处理日期类型，转换为字符串
                    if hasattr(value, 'isoformat'):
                        value = value.isoformat()
                    # 处理 Decimal 类型，转换为 float
                    elif hasattr(value, 'to_eng_string'):
                        value = float(value)
                    row_dict[col] = value
                data.append(row_dict)
            
            elapsed = time.time() - start_time
            return {
                "success": True,
                "data": data,
                "columns": columns,
                "row_count": len(data),
                "execution_time": round(elapsed, 2)
            }

        except Exception as e:
            elapsed = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "execution_time": round(elapsed, 2)
            }

    async def get_table_schema(self, table_name: Optional[str] = None) -> str:
        """获取数据库表结构信息"""
        if not self.pool and not await self.connect():
            return "无法连接数据库"

        schema_info = []

        try:
            if self.db_type == "sqlite":
                async with self.pool.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
                    tables = [row[0] for row in await cursor.fetchall()]
            else:
                async with self.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        if self.db_type == "mysql":
                            await cursor.execute("SHOW TABLES")
                            tables = [row[0] for row in await cursor.fetchall()]
                        elif self.db_type == "postgresql":
                            await cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'")
                            tables = [row[0] for row in await cursor.fetchall()]

            schema_info.append("数据库表列表:")
            schema_info.append(", ".join(tables))
            schema_info.append("")

            if not table_name:
                for table in tables[:5]:
                    if self.db_type == "sqlite":
                        async with self.pool.execute(f"PRAGMA table_info({table})") as cursor:
                            cols = await cursor.fetchall()
                    else:
                        async with self.pool.acquire() as conn:
                            async with conn.cursor() as cursor:
                                if self.db_type == "mysql":
                                    await cursor.execute(f"DESCRIBE {table}")
                                elif self.db_type == "postgresql":
                                    await cursor.execute(f"""
                                        SELECT column_name, data_type 
                                        FROM information_schema.columns 
                                        WHERE table_name = $1
                                    """, (table,))
                                cols = await cursor.fetchall()

                    schema_info.append(f"表: {table}")
                    for col in cols:
                        if self.db_type == "sqlite":
                            schema_info.append(f"  {col[1]} ({col[2]})")
                        else:
                            schema_info.append(f"  {col[0]} ({col[1]})")
                    schema_info.append("")

            return "\n".join(schema_info)

        except Exception as e:
            return f"获取表结构失败: {str(e)}"

    async def get_all_tables(self) -> List[str]:
        """获取所有表名"""
        if not self.pool and not await self.connect():
            return []

        try:
            if self.db_type == "sqlite":
                async with self.pool.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
                    tables = [row[0] for row in await cursor.fetchall()]
            else:
                async with self.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        if self.db_type == "mysql":
                            await cursor.execute("SHOW TABLES")
                        elif self.db_type == "postgresql":
                            await cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_type = 'BASE TABLE'")
                        tables = [row[0] for row in await cursor.fetchall()]

            return tables

        except Exception as e:
            print(f"获取表列表失败: {str(e)}")
            return []


# 创建全局异步数据库连接实例
db_connector = AsyncDatabaseConnector()


def get_db_connection() -> AsyncDatabaseConnector:
    """获取数据库连接实例"""
    return db_connector


class AsyncStatePersistence:
    """异步状态持久化类 - 用于存储任务、会议、权限等状态"""

    def __init__(self, db: AsyncDatabaseConnector):
        self.db = db

    async def _init_tables(self):
        """初始化持久化表"""
        if not self.db.pool and not await self.db.connect():
            return

        try:
            if self.db.db_type == "sqlite":
                async with self.db.pool.execute("""
                    CREATE TABLE IF NOT EXISTS report_tasks (
                        task_id TEXT PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        title TEXT,
                        description TEXT,
                        status TEXT DEFAULT 'pending',
                        user_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        result TEXT,
                        error TEXT
                    )
                """):
                    pass
            else:
                async with self.db.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("""
                            CREATE TABLE IF NOT EXISTS report_tasks (
                                task_id VARCHAR(64) PRIMARY KEY,
                                task_type VARCHAR(32) NOT NULL,
                                title VARCHAR(256),
                                description TEXT,
                                status VARCHAR(16) DEFAULT 'pending',
                                user_id VARCHAR(64),
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                                result TEXT,
                                error TEXT
                            )
                        """)

                        await cursor.execute("""
                            CREATE TABLE IF NOT EXISTS meeting_transcripts (
                                meeting_id VARCHAR(64) PRIMARY KEY,
                                transcript TEXT,
                                participants TEXT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                                summary TEXT,
                                action_items TEXT
                            )
                        """)

                        await cursor.execute("""
                            CREATE TABLE IF NOT EXISTS user_kb_permissions (
                                id INTEGER AUTO_INCREMENT PRIMARY KEY,
                                user_id VARCHAR(64) NOT NULL,
                                kb_id VARCHAR(64) NOT NULL,
                                permission_level VARCHAR(16) DEFAULT 'read',
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                UNIQUE KEY unique_user_kb (user_id, kb_id)
                            )
                        """)

                        await cursor.execute("""
                            CREATE TABLE IF NOT EXISTS dept_kb_mappings (
                                id INTEGER AUTO_INCREMENT PRIMARY KEY,
                                dept_id VARCHAR(64) NOT NULL,
                                kb_id VARCHAR(64) NOT NULL,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                UNIQUE KEY unique_dept_kb (dept_id, kb_id)
                            )
                        """)

                        await conn.commit()

        except Exception as e:
            print(f"初始化持久化表失败: {str(e)}")

    async def save_report_task(self, task_id: str, task_type: str, title: str, description: str,
                              status: str, user_id: str, result: str = None, error: str = None):
        """保存报告任务"""
        if not self.db.pool and not await self.db.connect():
            return False

        try:
            if self.db.db_type == "sqlite":
                async with self.db.pool.execute("""
                    INSERT OR REPLACE INTO report_tasks 
                    (task_id, task_type, title, description, status, user_id, result, error, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (task_id, task_type, title, description, status, user_id, result, error)):
                    await self.db.pool.commit()
            else:
                async with self.db.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("""
                            INSERT INTO report_tasks (task_id, task_type, title, description, status, user_id, result, error)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                                title=VALUES(title), description=VALUES(description), status=VALUES(status),
                                result=VALUES(result), error=VALUES(error), updated_at=CURRENT_TIMESTAMP
                        """, (task_id, task_type, title, description, status, user_id, result, error))
                        await conn.commit()
            return True
        except Exception as e:
            print(f"保存报告任务失败: {str(e)}")
            return False

    async def get_report_tasks(self, user_id: str = None, status: str = None) -> List[Dict[str, Any]]:
        """获取报告任务列表"""
        if not self.db.pool and not await self.db.connect():
            return []

        try:
            if self.db.db_type == "sqlite":
                query = "SELECT * FROM report_tasks WHERE 1=1"
                params = []

                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                if status:
                    query += " AND status = ?"
                    params.append(status)

                async with self.db.pool.execute(query, params) as cursor:
                    columns = [desc[0] for desc in cursor.description]
                    results = await cursor.fetchall()
            else:
                async with self.db.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        query = "SELECT * FROM report_tasks WHERE 1=1"
                        params = []

                        if user_id:
                            query += " AND user_id = %s"
                            params.append(user_id)
                        if status:
                            query += " AND status = %s"
                            params.append(status)

                        await cursor.execute(query, params)
                        columns = [desc[0] for desc in cursor.description]
                        results = await cursor.fetchall()

            return [dict(zip(columns, row)) for row in results]
        except Exception as e:
            print(f"获取报告任务失败: {str(e)}")
            return []

    async def save_meeting_transcript(self, meeting_id: str, transcript: str, participants: str,
                                      summary: str = None, action_items: str = None):
        """保存会议纪要"""
        if not self.db.pool and not await self.db.connect():
            return False

        try:
            if self.db.db_type == "sqlite":
                async with self.db.pool.execute("""
                    INSERT OR REPLACE INTO meeting_transcripts 
                    (meeting_id, transcript, participants, summary, action_items, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (meeting_id, transcript, participants, summary, action_items)):
                    await self.db.pool.commit()
            else:
                async with self.db.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("""
                            INSERT INTO meeting_transcripts (meeting_id, transcript, participants, summary, action_items)
                            VALUES (%s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                                transcript=VALUES(transcript), participants=VALUES(participants),
                                summary=VALUES(summary), action_items=VALUES(action_items),
                                updated_at=CURRENT_TIMESTAMP
                        """, (meeting_id, transcript, participants, summary, action_items))
                        await conn.commit()
            return True
        except Exception as e:
            print(f"保存会议纪要失败: {str(e)}")
            return False

    async def get_meeting_transcript(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """获取会议纪要"""
        if not self.db.pool and not await self.db.connect():
            return None

        try:
            if self.db.db_type == "sqlite":
                async with self.db.pool.execute("SELECT * FROM meeting_transcripts WHERE meeting_id = ?", (meeting_id,)) as cursor:
                    columns = [desc[0] for desc in cursor.description]
                    row = await cursor.fetchone()
                    return dict(zip(columns, row)) if row else None
            else:
                async with self.db.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("SELECT * FROM meeting_transcripts WHERE meeting_id = %s", (meeting_id,))
                        columns = [desc[0] for desc in cursor.description]
                        row = await cursor.fetchone()
                        return dict(zip(columns, row)) if row else None
        except Exception as e:
            print(f"获取会议纪要失败: {str(e)}")
            return None

    async def save_user_kb_permission(self, user_id: str, kb_id: str, permission_level: str = "read"):
        """保存用户知识库权限"""
        if not self.db.pool and not await self.db.connect():
            return False

        try:
            if self.db.db_type == "sqlite":
                async with self.db.pool.execute("""
                    INSERT OR REPLACE INTO user_kb_permissions (user_id, kb_id, permission_level, created_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (user_id, kb_id, permission_level)):
                    await self.db.pool.commit()
            else:
                async with self.db.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("""
                            INSERT INTO user_kb_permissions (user_id, kb_id, permission_level)
                            VALUES (%s, %s, %s)
                            ON DUPLICATE KEY UPDATE permission_level=VALUES(permission_level)
                        """, (user_id, kb_id, permission_level))
                        await conn.commit()
            return True
        except Exception as e:
            print(f"保存用户权限失败: {str(e)}")
            return False

    async def get_user_kb_permissions(self, user_id: str) -> List[str]:
        """获取用户可访问的知识库ID列表"""
        if not self.db.pool and not await self.db.connect():
            return []

        try:
            if self.db.db_type == "sqlite":
                async with self.db.pool.execute("SELECT kb_id FROM user_kb_permissions WHERE user_id = ?", (user_id,)) as cursor:
                    results = await cursor.fetchall()
            else:
                async with self.db.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("SELECT kb_id FROM user_kb_permissions WHERE user_id = %s", (user_id,))
                        results = await cursor.fetchall()
            return [r[0] for r in results]
        except Exception as e:
            print(f"获取用户权限失败: {str(e)}")
            return []

    async def save_dept_kb_mapping(self, dept_id: str, kb_id: str):
        """保存部门知识库映射"""
        if not self.db.pool and not await self.db.connect():
            return False

        try:
            if self.db.db_type == "sqlite":
                async with self.db.pool.execute("INSERT OR IGNORE INTO dept_kb_mappings (dept_id, kb_id, created_at) VALUES (?, ?, CURRENT_TIMESTAMP)", (dept_id, kb_id)):
                    await self.db.pool.commit()
            else:
                async with self.db.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("""
                            INSERT IGNORE INTO dept_kb_mappings (dept_id, kb_id) VALUES (%s, %s)
                        """, (dept_id, kb_id))
                        await conn.commit()
            return True
        except Exception as e:
            print(f"保存部门映射失败: {str(e)}")
            return False

    async def get_dept_kbs(self, dept_id: str) -> List[str]:
        """获取部门关联的知识库ID列表"""
        if not self.db.pool and not await self.db.connect():
            return []

        try:
            if self.db.db_type == "sqlite":
                async with self.db.pool.execute("SELECT kb_id FROM dept_kb_mappings WHERE dept_id = ?", (dept_id,)) as cursor:
                    results = await cursor.fetchall()
            else:
                async with self.db.pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("SELECT kb_id FROM dept_kb_mappings WHERE dept_id = %s", (dept_id,))
                        results = await cursor.fetchall()
            return [r[0] for r in results]
        except Exception as e:
            print(f"获取部门知识库失败: {str(e)}")
            return []