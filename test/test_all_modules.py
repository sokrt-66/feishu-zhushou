#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试所有模块是否正常导入"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(encoding='utf-8')

def test_module_import(module_name, import_statement):
    """测试单个模块导入"""
    try:
        exec(import_statement)
        print(f"✅ [{module_name}] 导入成功")
        return True
    except Exception as e:
        print(f"❌ [{module_name}] 导入失败: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("开始测试所有模块...")
    print("=" * 60)
    
    modules = [
        ("config", "import config"),
        ("feishu", "from feishu import FeishuClient"),
        ("tools", "from tools import tools_from_B"),
        ("agent", "from agent import build_main_graph, MainGraphNodes, OverallState"),
        ("automation_agent", "from automation_agent import AutomationAgent"),
        ("data_analysis_agent", "from data_analysis_agent import DataAnalysisAgent"),
        ("database", "from database import get_db_connection"),
        ("document_subgraph", "from document_subgraph import DocumentIngestionSubGraph, DocumentState"),
        ("feishu_subgraph", "from feishu_subgraph import FeishuEventSubGraph, background_task_queue"),
        ("meeting_agent", "from meeting_agent import MeetingAgent"),
        ("permissions", "from permissions import PermissionManager, RAGPermissionGuard"),
        ("tool_subgraphs", "from tool_subgraphs import QuerySqlTool, RagSearchTool, WebSearchTool"),
    ]
    
    success_count = 0
    total_count = len(modules)
    
    for name, import_stmt in modules:
        if test_module_import(name, import_stmt):
            success_count += 1
    
    print("=" * 60)
    print(f"测试完成: {success_count}/{total_count} 模块导入成功")
    print("=" * 60)
    
    if success_count == total_count:
        print("\n🎉 所有模块测试通过！")
        return 0
    else:
        print(f"\n⚠️ 有 {total_count - success_count} 个模块导入失败")
        return 1

if __name__ == "__main__":
    exit(main())
