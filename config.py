"""系统参数配置文件 - 所有敏感信息必须从环境变量读取"""

import os
import secrets
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_required_env(var_name: str, default: Optional[str] = None, min_length: int = 0) -> str:
    """从环境变量获取值，如果不存在且无默认值则抛出异常"""
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"必需的环境变量 {var_name} 未设置")
    if min_length > 0 and len(value) < min_length:
        raise ValueError(f"环境变量 {var_name} 长度必须至少 {min_length} 字节")
    return value


def _get_optional_env(var_name: str, default: str = "") -> str:
    """从环境变量获取可选值"""
    return os.getenv(var_name, default)


def _get_int_env(var_name: str, default: int) -> int:
    """从环境变量获取整数值"""
    value = os.getenv(var_name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"环境变量 {var_name} 应为整数，使用默认值 {default}")
        return default


def _get_bool_env(var_name: str, default: bool) -> bool:
    """从环境变量获取布尔值"""
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def _get_float_env(var_name: str, default: float) -> float:
    """从环境变量获取浮点数值"""
    value = os.getenv(var_name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"环境变量 {var_name} 应为浮点数，使用默认值 {default}")
        return default


class AgentConfig:
    max_reflection_times: int = _get_int_env("AGENT_MAX_REFLECTION_TIMES", 3)
    max_thinking_steps: int = _get_int_env("AGENT_MAX_THINKING_STEPS", 10)
    enable_reflection: bool = _get_bool_env("AGENT_ENABLE_REFLECTION", True)
    max_history_rounds: int = _get_int_env("AGENT_MAX_HISTORY_ROUNDS", 10)  # 对话历史最大轮数


class LLMConfig:
    dashscope_base_url: str = _get_optional_env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model_name: str = _get_optional_env("LLM_MODEL_NAME", "qwen3.6-flash")
    reasoning_model_name: str = _get_optional_env("LLM_REASONING_MODEL", "qwen3.6-plus")
    embedding_model_name: str = _get_optional_env("LLM_EMBEDDING_MODEL", "text-embedding-v4")
    
    main_temperature: float = _get_float_env("LLM_MAIN_TEMPERATURE", 0.0)
    reasoning_temperature: float = _get_float_env("LLM_REASONING_TEMPERATURE", 0.1)
    sql_temperature: float = _get_float_env("SQL_TEMPERATURE", 0.1)
    rag_temperature: float = _get_float_env("RAG_TEMPERATURE", 0.3)
    meeting_temperature: float = _get_float_env("MEETING_TEMPERATURE", 0.5)
    
    max_tokens: int = _get_int_env("LLM_MAX_TOKENS", 2000)
    reasoning_max_tokens: int = _get_int_env("LLM_REASONING_MAX_TOKENS", 4096)
    enable_thinking: bool = _get_bool_env("LLM_ENABLE_THINKING", False)


class SQLConfig:
    sql_internal_retry: int = _get_int_env("SQL_INTERNAL_RETRY", 2)
    sql_exec_timeout: int = _get_int_env("SQL_EXEC_TIMEOUT", 30)
    enable_sql_cache: bool = _get_bool_env("SQL_ENABLE_CACHE", True)
    cache_ttl_minutes: int = _get_int_env("SQL_CACHE_TTL_MINUTES", 5)

    # 数据库配置改为可选，支持SQLite作为默认配置
    db_type: str = _get_optional_env("DB_TYPE", "sqlite")
    db_host: str = _get_optional_env("DB_HOST", "localhost")
    db_port: int = _get_int_env("DB_PORT", 3306)
    db_user: str = _get_optional_env("DB_USER", "admin")
    db_password: str = _get_optional_env("DB_PASSWORD", "password")
    db_name: str = _get_optional_env("DB_NAME", ":memory:")


class RAGConfig:
    rag_relevance_threshold: float = _get_float_env("RAG_RELEVANCE_THRESHOLD", 0.7)
    rag_max_recall: int = _get_int_env("RAG_MAX_RECALL", 8)
    enable_query_rewrite: bool = _get_bool_env("RAG_ENABLE_QUERY_REWRITE", True)
    enable_answer_synthesis: bool = _get_bool_env("RAG_ENABLE_ANSWER_SYNTHESIS", True)


class WebSearchConfig:
    web_search_timeout: int = _get_int_env("WEB_SEARCH_TIMEOUT", 5)  # 从10秒降低到5秒
    search_results_count: int = _get_int_env("SEARCH_RESULTS_COUNT", 3)  # 从5条减少到3条
    trust_threshold: float = _get_float_env("TRUST_THRESHOLD", 0.7)
    low_trust_threshold: float = _get_float_env("LOW_TRUST_THRESHOLD", 0.5)
    tavily_api_key: Optional[str] = _get_optional_env("TAVILY_API_KEY") or None
    bing_search_key: Optional[str] = _get_optional_env("BING_SEARCH_API_KEY") or None
    enable_search_cache: bool = _get_bool_env("ENABLE_SEARCH_CACHE", True)
    cache_ttl_minutes: int = _get_int_env("SEARCH_CACHE_TTL", 10)
    
    # 域名可信度权重
    trusted_domains = {
        "gov.cn": 1.0,
        "edu.cn": 0.95,
        "news.cn": 0.95,
        "people.com.cn": 0.95,
        "xinhuanet.com": 0.95,
        "央视网": 0.95,
        "qq.com": 0.85,
        "sina.com.cn": 0.85,
        "sohu.com": 0.85,
        "163.com": 0.85,
        "baidu.com": 0.8,
        "zhihu.com": 0.75,
        "weibo.com": 0.7,
        "github.com": 0.85,
        "csdn.net": 0.75,
        "blog.csdn.net": 0.7,
        "medium.com": 0.8,
        "stackexchange.com": 0.9,
        "stackoverflow.com": 0.9,
    }


class DocumentConfig:
    chunk_size: int = _get_int_env("DOC_CHUNK_SIZE", 500)
    chunk_overlap: int = _get_int_env("DOC_CHUNK_OVERLAP", 50)
    enable_title_extraction: bool = _get_bool_env("DOC_ENABLE_TITLE_EXTRACTION", True)
    enable_summary: bool = _get_bool_env("DOC_ENABLE_SUMMARY", True)


class FeishuConfig:
    app_id: str = _get_required_env("FEISHU_APP_ID")
    app_secret: str = _get_required_env("FEISHU_APP_SECRET")
    verification_token: str = _get_required_env("FEISHU_VERIFICATION_TOKEN")
    encrypt_key: str = _get_optional_env("FEISHU_ENCRYPT_KEY", "")
    webhook_endpoint: str = _get_optional_env("FEISHU_WEBHOOK_ENDPOINT", "/api/feishu/webhook")
    message_retry_count: int = _get_int_env("FEISHU_MESSAGE_RETRY_COUNT", 3)
    message_timeout: int = _get_int_env("FEISHU_MESSAGE_TIMEOUT", 30)


class SystemConfig:
    port: int = _get_int_env("SERVER_PORT", 5000)
    debug: bool = _get_bool_env("DEBUG", False)
    log_level: str = _get_optional_env("LOG_LEVEL", "INFO")


class SecurityConfig:
    secret_key: str = _get_required_env("SECRET_KEY", min_length=32)
    jwt_algorithm: str = _get_optional_env("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = _get_int_env("ACCESS_TOKEN_EXPIRE_MINUTES", 30)

    dashscope_api_key: Optional[str] = _get_optional_env("DASHSCOPE_API_KEY") or None
    openai_api_key: Optional[str] = _get_optional_env("OPENAI_API_KEY") or None

    @classmethod
    def validate(cls) -> bool:
        """启动时验证所有安全配置"""
        if len(cls.secret_key) < 32:
            raise ValueError("SECRET_KEY 长度必须至少为 32 字节")

        if not cls.dashscope_api_key and not cls.openai_api_key:
            logger.warning("未设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY，AI功能将无法使用")

        return True


class InitConfig:
    """初始化数据配置 - 首次启动时创建的默认数据"""
    
    # 默认部门配置
    default_departments: dict = {
        "dept_1": {"name": "技术部", "description": "负责技术研发"},
        "dept_2": {"name": "市场部", "description": "负责市场推广"},
        "dept_3": {"name": "人事部", "description": "负责人力资源"}
    }
    
    # 默认用户配置（用户名: (密码, 部门ID)）
    default_users: dict = {
        "admin": ("password", "dept_1"),
        "user1": ("password", "dept_1"),
        "user2": ("password", "dept_2"),
        "user3": ("password", "dept_3")
    }
    
    # 默认部门管理员配置（部门ID: [管理员用户ID列表]）
    default_dept_admins: dict = {
        "dept_1": ["admin"]
    }
    
    # 默认知识库配置
    default_knowledge_bases: list = [
        {"kb_id": "kb_dept1", "name": "技术部知识库", "owner": "admin", "department": "dept_1"},
        {"kb_id": "kb_dept2", "name": "市场部知识库", "owner": "admin", "department": "dept_2"},
        {"kb_id": "kb_dept3", "name": "人事部知识库", "owner": "admin", "department": "dept_3"}
    ]


agent_config = AgentConfig()
llm_config = LLMConfig()
sql_config = SQLConfig()
rag_config = RAGConfig()
web_search_config = WebSearchConfig()
document_config = DocumentConfig()
feishu_config = FeishuConfig()
system_config = SystemConfig()
security_config = SecurityConfig()
init_config = InitConfig()

CONFIG = {
    "agent": agent_config,
    "llm": llm_config,
    "sql": sql_config,
    "rag": rag_config,
    "web_search": web_search_config,
    "document": document_config,
    "feishu": feishu_config,
    "system": system_config,
    "security": security_config,
    "init": init_config
}
