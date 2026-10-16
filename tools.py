"""
工具模块 - 成员B负责
导出工具列表供主智能体调用
"""

from typing import List, Dict, Any
import json

# 优先导入高级工具子图
try:
    from tool_subgraphs import (
        QuerySqlTool,
        RagSearchTool,
        WebSearchTool,
        sql_subgraph,
        rag_subgraph,
        web_search_subgraph
    )
    ADVANCED_TOOLS_AVAILABLE = True
    print("成功加载高级工具子图")
    
    query_sql_tool = QuerySqlTool()
    rag_search_tool = RagSearchTool()
    web_search_tool = WebSearchTool()
    
    tools_from_B = [query_sql_tool, rag_search_tool, web_search_tool]
    tools_dict = {t.name: t._run for t in tools_from_B}
    
except ImportError as e:
    print(f"加载高级工具子图失败: {e}")
    ADVANCED_TOOLS_AVAILABLE = False
    
    from typing import TypedDict
    import time
    from config import sql_config, rag_config, web_search_config

    class QuerySqlTool:
        name = "query_sql_tool"
        description = "自然语言转SQL"

        def _run(self, query: str) -> str:
            return json.dumps({
                "success": True,
                "result": f"SQL查询结果（简化模式）：{query}",
                "sql": f"SELECT * FROM data WHERE content LIKE '%{query}%'",
                "error": None
            }, ensure_ascii=False)

    class RagSearchTool:
        name = "rag_search_tool"
        description = "企业内部知识库检索"

        def _run(self, query: str) -> str:
            return json.dumps({
                "success": True,
                "result": f"知识库检索结果（简化模式）：{query}",
                "relevance": 0.75,
                "degraded": False
            }, ensure_ascii=False)

    class WebSearchTool:
        name = "web_search_tool"
        description = "联网搜索实时信息"

        def _run(self, query: str) -> str:
            return json.dumps({
                "success": True,
                "result": f"Web搜索结果（简化模式）：{query}",
                "trust_score": 0.8,
                "error": None
            }, ensure_ascii=False)

    query_sql_tool = QuerySqlTool()
    rag_search_tool = RagSearchTool()
    web_search_tool = WebSearchTool()

    tools_from_B = [query_sql_tool, rag_search_tool, web_search_tool]
    tools_dict = {t.name: t._run for t in tools_from_B}


def get_tool_by_name(name: str):
    """根据名称获取工具"""
    for tool in tools_from_B:
        if tool.name == name:
            return tool
    return None


def execute_tool(name: str, *args, **kwargs) -> str:
    """执行指定工具"""
    tool = get_tool_by_name(name)
    if tool:
        return tool._run(*args, **kwargs)
    return json.dumps({"success": False, "error": f"工具 {name} 不存在"}, ensure_ascii=False)


__all__ = [
    'QuerySqlTool',
    'RagSearchTool', 
    'WebSearchTool',
    'query_sql_tool',
    'rag_search_tool',
    'web_search_tool',
    'tools_from_B',
    'tools_dict',
    'get_tool_by_name',
    'execute_tool'
]

if ADVANCED_TOOLS_AVAILABLE:
    __all__.extend(['sql_subgraph', 'rag_subgraph', 'web_search_subgraph'])

# 调试信息 - 打印已加载的工具
print("已加载的工具:")
for tool in tools_from_B:
    print(f"  - {tool.name}: {tool.description}")