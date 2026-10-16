import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataAnalysisAgent:
    """飞书数据分析 & 报表智能体"""
    
    def __init__(self, feishu_client, llm):
        self.feishu = feishu_client
        self.llm = llm
        self.bitable_configs = {}  # 存储多维表格配置
    
    def register_bitable(self, config_id: str, app_token: str, table_id: str, 
                         table_name: str = ""):
        """注册多维表格"""
        self.bitable_configs[config_id] = {
            "app_token": app_token,
            "table_id": table_id,
            "table_name": table_name or config_id
        }
        logger.info(f"注册多维表格: {config_id} ({table_name})")
    
    async def process_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """处理自然语言查询"""
        try:
            # 分析查询意图
            analysis = await self._analyze_query(query)
            
            if analysis["intent"] == "query_data":
                # 执行数据查询
                result = await self._execute_query(analysis)
                
                # 生成报表
                report = await self._generate_report(result, analysis)
                
                # 发送回复
                await self._send_reply(user_id, report)
                
                return {
                    "status": "success",
                    "intent": "query_data",
                    "report": report
                }
            else:
                return {
                    "status": "success",
                    "intent": "unknown",
                    "message": "未能理解您的查询意图"
                }
                
        except Exception as e:
            logger.error(f"数据分析失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询意图和参数"""
        prompt = f"""
分析以下自然语言查询，返回JSON格式：
查询内容: {query}

需要分析：
- intent: 查询意图（query_data / summarize / compare / trend）
- target_table: 目标表格（可选）
- filters: 过滤条件列表（可选，如时间范围、部门等）
- aggregations: 聚合方式（如sum, count, avg, max, min）
- group_by: 分组字段（可选）
- chart_type: 图表类型（bar, line, pie, table）

只返回JSON，不要其他内容
"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except Exception as e:
            logger.warning(f"查询分析失败: {str(e)}")
        
        # 降级方案
        return {
            "intent": "query_data",
            "target_table": None,
            "filters": [],
            "aggregations": ["count"],
            "group_by": None,
            "chart_type": "table"
        }
    
    async def _execute_query(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """执行数据查询"""
        # 获取目标表格配置
        target_table = analysis.get("target_table")
        if not target_table:
            # 使用第一个配置的表格
            if self.bitable_configs:
                target_table = next(iter(self.bitable_configs.keys()))
            else:
                return {"status": "error", "message": "没有配置多维表格"}
        
        config = self.bitable_configs.get(target_table)
        if not config:
            return {"status": "error", "message": f"未找到表格配置: {target_table}"}
        
        # 获取数据
        records = await self.feishu.get_bitable_records(config["app_token"], config["table_id"])
        
        # 简单过滤和聚合
        results = self._apply_filters(records, analysis.get("filters", []))
        aggregated = self._aggregate_data(results, analysis.get("aggregations", ["count"]), 
                                           analysis.get("group_by"))
        
        return {
            "status": "success",
            "table_name": config["table_name"],
            "total_records": len(records),
            "filtered_records": len(results),
            "aggregated": aggregated,
            "raw_data": results[:10]  # 只返回前10条原始数据
        }
    
    def _apply_filters(self, records: List[Dict], filters: List) -> List[Dict]:
        """应用过滤条件（简化版）"""
        return records
    
    def _aggregate_data(self, records: List[Dict], aggregations: List[str], 
                        group_by: Optional[str]) -> Dict[str, Any]:
        """数据聚合（简化版）"""
        result = {
            "count": len(records)
        }
        
        # 按字段分组统计（示例）
        if group_by and records:
            groups = {}
            for record in records:
                fields = record.get("fields", {})
                group_value = fields.get(group_by, "unknown")
                if group_value not in groups:
                    groups[group_value] = 0
                groups[group_value] += 1
            result["groups"] = groups
        
        return result
    
    async def _generate_report(self, query_result: Dict[str, Any], 
                               analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成报表"""
        if query_result.get("status") != "success":
            return {
                "type": "error",
                "content": "查询失败: " + query_result.get("message", "")
            }
        
        # 生成文本报告
        text_report = f"📊 数据查询报告 - {query_result.get('table_name', '')}\n\n"
        text_report += f"📈 总记录数: {query_result.get('total_records', 0)}\n"
        text_report += f"📊 符合条件: {query_result.get('filtered_records', 0)}\n"
        text_report += f"📋 统计: {query_result.get('aggregated', {})}\n"
        
        # 生成图表配置
        chart_config = self._generate_chart_config(query_result, analysis)
        
        return {
            "type": "report",
            "text_report": text_report,
            "chart_config": chart_config,
            "data": query_result
        }
    
    def _generate_chart_config(self, query_result: Dict[str, Any], 
                              analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成图表配置"""
        chart_type = analysis.get("chart_type", "table")
        
        # 简单图表配置
        if chart_type == "pie" and "groups" in query_result.get("aggregated", {}):
            groups = query_result["aggregated"]["groups"]
            return {
                "type": "pie",
                "data": [{"name": k, "value": v} for k, v in groups.items()]
            }
        else:
            return {
                "type": "table",
                "columns": ["字段", "值"],
                "rows": [["记录数", query_result.get("filtered_records", 0)]]
            }
    
    async def _send_reply(self, user_id: str, report: Dict[str, Any]):
        """发送回复消息"""
        if report.get("type") == "error":
            await self.feishu.send_text_message(user_id, "❌ " + report.get("content", ""))
            return
        
        # 发送文本报告
        text_report = report.get("text_report", "")
        await self.feishu.send_text_message(user_id, text_report)
        
        # 如果有图表，发送卡片
        chart_config = report.get("chart_config")
        if chart_config:
            card = self._build_chart_card(chart_config)
            await self.feishu.send_card_message(user_id, card)
    
    def _build_chart_card(self, chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """构建图表卡片"""
        # 简单的飞书卡片结构
        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"content": "📊 数据分析图表", "tag": "plain_text"}
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "content": f"图表类型: {chart_config.get('type', 'table')}",
                        "tag": "plain_text"
                    }
                }
            ]
        }
    
    async def add_record(self, table_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """添加记录到多维表格"""
        config = self.bitable_configs.get(table_id)
        if not config:
            return {"status": "error", "message": f"未找到表格配置: {table_id}"}
        
        record = await self.feishu.create_bitable_record(config["app_token"], config["table_id"], data)
        
        if record:
            return {"status": "success", "record": record}
        else:
            return {"status": "error", "message": "添加记录失败"}
