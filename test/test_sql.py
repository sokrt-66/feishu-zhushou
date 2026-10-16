import asyncio
import sys
import os
import json
from dotenv import load_dotenv

# 添加项目路径
sys.path.insert(0, 'f:/python项目/飞书日常功能实现')

# 加载环境变量
load_dotenv()

# 先禁用缓存
os.environ["SQL_ENABLE_CACHE"] = "false"

from tool_subgraphs import query_sql_tool

async def test():
    print("开始测试 SQL 查询工具...")
    
    # 检查缓存是否禁用
    print(f"缓存状态: {'已禁用' if not query_sql_tool.subgraph.enable_cache else '已启用'}")
    
    try:
        result = await query_sql_tool._arun("查询所有员工")
        print("\n查询结果:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        
        # 解析JSON结果
        try:
            result_dict = json.loads(result)
            print("\n结果详情:")
            print(f"成功: {result_dict.get('success')}")
            print(f"执行的SQL: {result_dict.get('sql')}")
            print(f"错误信息: {result_dict.get('error')}")
            
            # 解析查询结果
            if result_dict.get('result'):
                result_str = result_dict['result']
                if '查询结果：' in result_str:
                    data_str = result_str.replace('查询结果：', '').strip()
                    try:
                        data = json.loads(data_str)
                        if isinstance(data, dict) and 'data' in data:
                            # 数据库返回的完整格式
                            print(f"\n返回记录数: {data.get('row_count', '未知')}")
                            print("字段列表:", data.get('columns', []))
                            print("\n记录详情:")
                            for row in data.get('data', []):
                                print(json.dumps(row, ensure_ascii=False))
                        elif isinstance(data, list):
                            # 旧格式（模拟数据）
                            print(f"\n返回记录数: {len(data)}")
                            print("字段列表:", list(data[0].keys()) if data else [])
                    except Exception as e:
                        print(f"解析结果失败: {e}")
                        print(f"结果内容: {result_str[:300]}...")
        except json.JSONDecodeError:
            print("无法解析JSON结果")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test())