from typing import TypedDict, Sequence, Any, List, Optional, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import asyncio
from datetime import datetime
import json
import logging
import os
import hashlib

logger = logging.getLogger(__name__)

# 导入配置
from config import AgentConfig

# Checkpointer实现
class FileCheckpointer:
    """基于文件系统的Checkpointer实现，支持状态持久化与版本迁移"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _get_checkpoint_path(self, thread_id: str) -> str:
        hash_id = hashlib.md5(thread_id.encode()).hexdigest()
        return os.path.join(self.checkpoint_dir, f"{hash_id}.json")
    
    def get(self, thread_id: str) -> Optional[Dict]:
        checkpoint_path = self._get_checkpoint_path(thread_id)
        if not os.path.exists(checkpoint_path):
            return None
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None
    
    def put(self, thread_id: str, state: Dict) -> None:
        checkpoint_path = self._get_checkpoint_path(thread_id)
        state["_version"] = 1
        state["_saved_at"] = datetime.now().isoformat()
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

# 全局Checkpointer实例
checkpointer = FileCheckpointer()

def trim_message_history(messages: List[BaseMessage], max_rounds: int = None) -> List[BaseMessage]:
    """
    修剪消息历史，保留最近的对话轮数
    
    Args:
        messages: 原始消息列表
        max_rounds: 最大保留轮数，默认为配置中的 AGENT_MAX_HISTORY_ROUNDS
    
    Returns:
        修剪后的消息列表
    """
    if max_rounds is None:
        max_rounds = AgentConfig.max_history_rounds
    
    if max_rounds <= 0:
        return messages
    
    # 计算需要保留的消息数量（一轮包含用户消息+助手消息）
    max_messages = max_rounds * 2
    
    if len(messages) <= max_messages:
        return messages
    
    # 保留最后 max_messages 条消息
    trimmed = messages[-max_messages:]
    
    # 如果第一条是助手消息，可能需要调整（确保从用户消息开始）
    if trimmed and isinstance(trimmed[0], AIMessage):
        # 如果第一条是AI消息且还有更多消息，移除它
        if len(trimmed) > 1:
            trimmed = trimmed[1:]
    
    logger.debug(f"对话历史已修剪: {len(messages)} -> {len(trimmed)} 条消息")
    return trimmed

class OverallState(TypedDict):
    messages: List[BaseMessage]
    current_step: str
    tool_results: List[dict]
    reflection_count: int
    max_reflection: int
    execution_status: str
    final_output: str
    error_info: str
    stream_queue: asyncio.Queue
    metadata: dict
    tool_list: List[dict]

class MainGraphNodes:
    def __init__(self, llm, tools: dict, sub_agents: dict, tool_descriptions: List[dict] = None):
        self.llm = llm
        self.tools = tools
        self.sub_agents = sub_agents
        self.max_retries = 3
        self.tool_descriptions = tool_descriptions or []
        self.llm_available = self._check_llm_available()

    def _check_llm_available(self) -> bool:
        """检查LLM服务是否可用"""
        if not self.llm:
            return False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在运行中的事件循环中，直接调用异步方法
                return True
            else:
                asyncio.run(self.llm.ainvoke("test"))
                return True
        except Exception:
            return False
    
    async def planning_node(self, state: OverallState) -> dict:
        messages = state["messages"]

        if not self.llm_available:
            await state["stream_queue"].put({
                "type": "error",
                "content": "AI服务暂时不可用，请稍后重试",
                "timestamp": datetime.now().isoformat()
            })
            await state["stream_queue"].put({"type": "done"})
            return {
                "current_step": "error",
                "execution_status": "failed",
                "error_info": "LLM服务不可用",
                "final_output": "AI服务暂时不可用，请稍后重试"
            }

        tool_list = state.get("tool_list", self.tool_descriptions)
        
        tools_info = "\n".join([
            f"- {tool['name']}: {tool['description']}" 
            for tool in tool_list
        ]) if tool_list else ""

        planner_prompt = f"""
你是一个智能任务规划器。根据用户的问题，分析需要调用哪些工具，并生成执行计划。

可用工具：
{tools_info}

用户消息：{messages[-1].content if messages else ""}

请输出JSON格式的计划，包含：
- "tasks": 任务列表，每个任务包含 "tool"（工具名称）和 "input"（工具输入参数）
- "reasoning": 选择这些工具的理由

只返回JSON，不要其他内容。
"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=planner_prompt)])
            content = response.content if hasattr(response, 'content') else str(response)
            
            try:
                plan_result = json.loads(content)
            except json.JSONDecodeError:
                plan_result = {"tasks": [], "reasoning": "无法解析计划"}
            
            if not isinstance(plan_result, dict) or "tasks" not in plan_result:
                plan_result = {"tasks": [], "reasoning": "计划格式错误"}

        except Exception as e:
            logger.error(f"规划节点执行失败: {e}")
            plan_result = {"tasks": [], "reasoning": f"规划失败: {str(e)}"}

        await state["stream_queue"].put({
            "type": "plan",
            "content": plan_result.get("reasoning", "") + "\n任务列表: " + str([t.get("tool") for t in plan_result.get("tasks", [])]),
            "timestamp": datetime.now().isoformat()
        })

        return {
            "current_step": "planning_complete",
            "execution_status": "planning",
            "metadata": {**state.get("metadata", {}), "plan": plan_result}
        }
    
    async def execution_node(self, state: OverallState) -> dict:
        plan = state.get("metadata", {}).get("plan", {})
        tasks = plan.get("tasks", [])
        
        results = []
        for task in tasks:
            tool_name = task.get("tool")
            tool_input = task.get("input", {})
            
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                try:
                    if asyncio.iscoroutinefunction(tool):
                        result = await tool(**tool_input)
                    else:
                        result = tool(**tool_input)
                    
                    results.append({
                        "task": task,
                        "status": "success",
                        "output": result
                    })
                    
                    await state["stream_queue"].put({
                        "type": "tool_result",
                        "tool": tool_name,
                        "content": str(result)[:500],
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    results.append({
                        "task": task,
                        "status": "error",
                        "error": str(e)
                    })
                    
                    await state["stream_queue"].put({
                        "type": "tool_error",
                        "tool": tool_name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
        
        return {
            "tool_results": results,
            "current_step": "execution_complete",
            "execution_status": "executing"
        }
    
    async def reflection_node(self, state: OverallState) -> dict:
        tool_results = state.get("tool_results", [])
        reflection_count = state.get("reflection_count", 0)
        
        errors = [r for r in tool_results if r.get("status") == "error"]
        
        reflector = self.sub_agents.get("reflector")
        reflection_result = await reflector.ainvoke({
            "original_plan": state.get("metadata", {}).get("plan", {}),
            "execution_results": tool_results,
            "errors": errors,
            "reflection_count": reflection_count
        })
        
        await state["stream_queue"].put({
            "type": "reflection",
            "content": reflection_result.get("analysis", ""),
            "decision": reflection_result.get("decision", "continue"),
            "timestamp": datetime.now().isoformat()
        })
        
        new_reflection_count = reflection_count + 1
        
        return {
            "reflection_count": new_reflection_count,
            "current_step": "reflection_complete",
            "execution_status": "reflecting",
            "metadata": {
                **state.get("metadata", {}),
                "reflection": reflection_result
            }
        }
    
    async def synthesis_node(self, state: OverallState) -> dict:
        messages = state["messages"]
        tool_results = state.get("tool_results", [])

        if not self.llm_available:
            tool_outputs = "\n".join([
                f"工具 {r.get('task', {}).get('tool')}: {str(r.get('output', ''))[:200]}"
                for r in tool_results if r.get("status") == "success"
            ])
            final_output = f"AI服务暂时不可用。\n工具执行结果：\n{tool_outputs}" if tool_outputs else "AI服务暂时不可用，请稍后重试"
            
            await state["stream_queue"].put({
                "type": "final",
                "content": final_output,
                "timestamp": datetime.now().isoformat()
            })
            await state["stream_queue"].put({"type": "done"})
            
            return {
                "final_output": final_output,
                "current_step": "synthesis_complete",
                "execution_status": "finished",
                "messages": [AIMessage(content=final_output)]
            }
        
        synthesis_prompt = self._build_synthesis_prompt(messages, tool_results)
        
        try:
            response = await self.llm.ainvoke([
                *messages,
                HumanMessage(content=synthesis_prompt)
            ])
            final_output = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"总结节点执行失败: {e}")
            tool_outputs = "\n".join([
                f"工具 {r.get('task', {}).get('tool')}: {str(r.get('output', ''))[:200]}"
                for r in tool_results if r.get("status") == "success"
            ])
            final_output = f"AI服务暂时不可用。\n工具执行结果：\n{tool_outputs}" if tool_outputs else "AI服务暂时不可用，请稍后重试"
        
        await state["stream_queue"].put({
            "type": "final",
            "content": final_output,
            "timestamp": datetime.now().isoformat()
        })
        
        await state["stream_queue"].put({"type": "done"})
        
        return {
            "final_output": final_output,
            "current_step": "synthesis_complete",
            "execution_status": "finished",
            "messages": [AIMessage(content=final_output)]
        }
    
    async def error_handler_node(self, state: OverallState) -> dict:
        error_info = state.get("error_info", "Unknown error")
        
        await state["stream_queue"].put({
            "type": "error",
            "error": error_info,
            "timestamp": datetime.now().isoformat()
        })
        
        await state["stream_queue"].put({"type": "done"})
        
        return {
            "final_output": f"执行过程中发生错误: {error_info}",
            "execution_status": "error",
            "current_step": "error_handled"
        }
    
    def _build_synthesis_prompt(self, messages, tool_results) -> str:
        successful_results = [r for r in tool_results if r.get("status") == "success"]
        failed_results = [r for r in tool_results if r.get("status") == "error"]
        
        prompt_parts = [
            "基于以下执行结果，生成完整的回答：\n",
            "成功执行的任务："
        ]
        
        for result in successful_results:
            prompt_parts.append(f"- {result['task'].get('tool')}: {str(result['output'])[:200]}")
        
        if failed_results:
            prompt_parts.append("\n失败的任务：")
            for result in failed_results:
                prompt_parts.append(f"- {result['task'].get('tool')}: {result.get('error', 'Unknown error')}")
        
        prompt_parts.append("\n请整合以上信息，给出完整、准确的回答。")
        
        return "\n".join(prompt_parts)

def build_main_graph(llm, tools: dict, sub_agents: dict):
    nodes = MainGraphNodes(llm, tools, sub_agents)
    
    async def run_workflow(input_messages: List[BaseMessage], metadata: dict = None):
        stream_queue = asyncio.Queue()
        
        # 修剪对话历史，保留最近的轮数
        trimmed_messages = trim_message_history(input_messages)
        
        state: OverallState = {
            "messages": trimmed_messages,
            "current_step": "start",
            "tool_results": [],
            "reflection_count": 0,
            "max_reflection": 3,
            "execution_status": "idle",
            "final_output": "",
            "error_info": "",
            "stream_queue": stream_queue,
            "metadata": metadata or {}
        }
        
        state.update(await nodes.planning_node(state))
        
        plan = state.get("metadata", {}).get("plan", {})
        needs_tools = plan.get("needs_tools", False)
        
        if needs_tools:
            state.update(await nodes.execution_node(state))
            
            has_errors = any(r.get("status") == "error" for r in state.get("tool_results", []))
            while has_errors and state.get("reflection_count", 0) < state.get("max_reflection", 3):
                state.update(await nodes.reflection_node(state))
                
                reflection = state.get("metadata", {}).get("reflection", {})
                decision = reflection.get("decision", "continue")
                
                if decision == "retry":
                    state.update(await nodes.execution_node(state))
                    has_errors = any(r.get("status") == "error" for r in state.get("tool_results", []))
                else:
                    break
        
        state.update(await nodes.synthesis_node(state))
        
        return state
    
    return run_workflow