#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aligo 商旅助手 - CLI 交互界面
使用 Rich 库实现美观的终端交互
"""
import asyncio
import sys
import os
from typing import Optional

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
import json

# 导入系统组件
from agentscope.model import OpenAIChatModel
from config_agentscope import init_agentscope
from config import LLM_CONFIG, SYSTEM_CONFIG, RESILIENCE_CONFIG
from context.memory_manager import MemoryManager
from utils.circuit_breaker import CircuitBreaker, CircuitOpenError
from utils.llm_resilience import run_health_check as check_llm_health
from agents.intention_agent import IntentionAgent
from agents.orchestration_agent import OrchestrationAgent
from services.turn_executor import AgentTurnExecutor, InvalidIntentionResultError
# 移除其他智能体的导入，改用懒加载


class AligoCLI:
    """Aligo 商旅助手 CLI"""

    def __init__(self):
        """初始化 CLI"""
        self.console = Console()
        self.user_id = None
        self.session_id = None
        self.memory_manager = None
        self.orchestrator = None
        self.intention_agent = None
        self.model = None
        self._agent_cache = {}  # 智能体缓存
        self.circuit_breaker = None  # 在 initialize_system 中从 RESILIENCE_CONFIG 初始化
        self.turn_executor = None

    def _ensure_api_key(self):
        """确保启动时已提供 LLM API Key。

        优先使用环境变量中的配置；未配置时在终端隐藏输入，
        密钥只保存在当前进程内，不会写入配置文件。
        """
        api_key = str(LLM_CONFIG.get("api_key") or "").strip()
        if api_key:
            return

        api_key = Prompt.ask(
            "DeepSeek API Key（输入内容不会显示）",
            password=True,
        ).strip()
        if not api_key:
            raise ValueError("DeepSeek API Key 不能为空")

        LLM_CONFIG["api_key"] = api_key

    def print_banner(self):
        """打印欢迎横幅"""
        self.console.print("\n[bold cyan]🌏 Aligo 商旅助手[/bold cyan] - 让差旅更简单\n", style="bold")

    def print_help(self):
        """打印帮助信息"""
        table = Table(title="命令列表", show_header=True, header_style="bold magenta")
        table.add_column("命令", style="cyan", width=20)
        table.add_column("说明", style="white")

        table.add_row("help", "显示此帮助信息")
        table.add_row("status", "查看当前状态和记忆")
        table.add_row("health", "检查 LLM 服务是否可用")
        table.add_row("clear", "清空当前任务（保留长期记忆）")
        table.add_row("history", "查看历史行程")
        table.add_row("preferences", "查看用户偏好")
        table.add_row("exit", "退出程序")
        table.add_row("", "")
        table.add_row("[自然语言]", "直接输入您的需求，如：")
        table.add_row("", "  - 我要从上海去北京出差")
        table.add_row("", "  - 北京的住宿标准是多少")
        table.add_row("", "  - 查询明天的天气")

        self.console.print(table)

    async def initialize_system(self):
        """初始化系统 - 使用懒加载优化启动速度"""
        # 未通过环境变量配置密钥时，启动后自动隐藏询问。
        self._ensure_api_key()

        # 获取用户信息
        self.user_id = Prompt.ask(
            "用户ID",
            default="default_user"
        )

        # 生成会话ID
        import uuid
        self.session_id = str(uuid.uuid4())[:8]

        with self.console.status("初始化中...", spinner="dots"):
            # 初始化AgentScope
            init_agentscope()

            # 初始化模型
            timeout_sec = SYSTEM_CONFIG.get("timeout", 60)
            self.model = OpenAIChatModel(
                model_name=LLM_CONFIG["model_name"],
                api_key=LLM_CONFIG["api_key"],
                client_kwargs={
                    "base_url": LLM_CONFIG["base_url"],
                    "timeout": float(timeout_sec),
                },
                generate_kwargs={
                    "temperature": LLM_CONFIG.get("temperature", 0.7),
                    "max_tokens": LLM_CONFIG.get("max_tokens", 2000),
                },
            )

            # 初始化记忆管理器（传入LLM模型用于总结）
            self.memory_manager = MemoryManager(
                user_id=self.user_id,
                session_id=self.session_id,
                llm_model=self.model
            )

            # 初始化意图识别智能体（必须预加载）
            self.intention_agent = IntentionAgent(
                name="IntentionAgent",
                model=self.model
            )

            # 使用懒加载注册器（智能体在首次使用时才加载）
            from agents.lazy_agent_registry import LazyAgentRegistry
            self._agent_cache = {}
            lazy_registry = LazyAgentRegistry(
                model=self.model, 
                cache=self._agent_cache,
                memory_manager=self.memory_manager
            )

            # 预先加载关键智能体（可选，利用 preload）
            # lazy_registry.preload("memory_query", "preference")

            # 初始化协调器
            self.orchestrator = OrchestrationAgent(
                name="OrchestrationAgent",
                agent_registry=lazy_registry,
                memory_manager=self.memory_manager
            )

            # 熔断器（连接与可用性）
            rc = RESILIENCE_CONFIG
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=rc.get("circuit_failure_threshold", 5),
                recovery_timeout_sec=rc.get("circuit_recovery_timeout_sec", 60.0),
                half_open_successes=rc.get("circuit_half_open_successes", 2),
            )

            # 核心业务执行器不负责终端展示，可同时被 CLI 和评估器复用。
            self.turn_executor = AgentTurnExecutor(
                intention_agent=self.intention_agent,
                orchestrator=self.orchestrator,
                memory_manager=self.memory_manager,
                circuit_breaker=self.circuit_breaker,
                resilience_config=RESILIENCE_CONFIG,
            )

        self.console.print(f"✓ 就绪 (用户: {self.user_id}) - 输入 help 查看帮助\n", style="green")

    async def process_query(self, user_input: str):
        try:
            with self.console.status("思考中...", spinner="dots"):
                turn_result = await self.turn_executor.execute_turn(user_input)
        except CircuitOpenError:
            self.console.print(
                "\n[bold yellow]⚠ 服务暂时不可用，请稍后再试。[/bold yellow]\n",
                style="dim",
            )
            return None
        except InvalidIntentionResultError:
            self.console.print(
                "❌ 无法理解您的需求，请重新描述",
                style="bold red",
            )
            return None

        # CLI 只负责把业务层返回的结构化结果展示给用户。
        result_data = turn_result["orchestration"]
        self._display_agents_called(result_data)
        self.console.print()
        self._display_results(result_data)
        return turn_result

    def _display_agents_called(self, result_data: dict):
        """显示调用的智能体列表"""
        results = result_data.get("results", [])
        if not results:
            return

        # 收集所有调用的智能体
        agents_called = []
        for result in results:
            agent_name = result.get("agent_name", "")
            status = result.get("status", "")

            display_name = self._get_agent_display_name(agent_name)

            # 根据状态添加标记
            if status == "success":
                agents_called.append(f"{display_name} ✓")
            elif status == "error":
                agents_called.append(f"{display_name} ✗")
            else:
                agents_called.append(f"{display_name} ?")

        if agents_called:
            self.console.print()
            self.console.print(f"🤖 调用智能体: {', '.join(agents_called)}", style="dim")

    def _display_results(self, result_data: dict):
        """显示执行结果 - 确保永远有回复"""
        self.console.print()

        # 获取结果列表
        results = result_data.get("results", [])

        if not results:
            # 情况1: 没有任何智能体被调用
            status = result_data.get("status", "unknown")
            if status == "no_agents":
                self.console.print("✓ 好的，我已记录下来。", style="green")
                self.console.print("\n💡 您可以继续补充信息，或者尝试：", style="dim")
                self.console.print("  • 规划行程：「帮我规划去北京的行程」", style="dim")
                self.console.print("  • 查询信息：「北京的天气怎么样」", style="dim")
                self.console.print("  • 问问题：「差旅标准是多少」", style="dim")
            else:
                self.console.print("未能获取有效结果，请重新描述您的需求。", style="yellow")
        else:
            # 情况2: 有智能体被调用，生成人性化回复
            needs_clarification = result_data.get("status") == "needs_clarification"
            has_output = self._generate_human_response(
                results,
                show_missing_info=not needs_clarification,
            )

            # 行程信息不足时，优先展示调度器生成的中文追问，
            # 不直接暴露子 Agent 的英文字段名。
            if needs_clarification:
                message = result_data.get("message", "行程信息不完整，请补充必要信息。")
                self.console.print(f"\n💡 {message}", style="yellow")
                has_output = True

            # 情况3: 智能体执行了但没有显示内容（兜底）
            if not has_output:
                self.console.print("✓ 已处理您的请求。", style="green")

        self.console.print()

    def _generate_human_response(
        self,
        results: list,
        show_missing_info: bool = True,
    ) -> bool:
        """
        根据结果生成人性化的回复
        """
        has_output = False

        for result in results:
            agent_name = result.get("agent_name", "")
            status = result.get("status", "")
            data = result.get("data", {})
            current_agent_shown = False  # 标记当前Agent是否有内容展示

            # 处理失败的智能体
            if status == "error":
                error_msg = data.get("error", "未知错误")
                agent_display_name = self._get_agent_display_name(agent_name)
                self.console.print(f"❌ {agent_display_name}执行失败: {error_msg}", style="red")
                has_output = True
                continue

            # 只处理成功的智能体 (RAG 的 no_knowledge 视为一种特殊的成功/提示)
            if status != "success" and not (agent_name == "rag_knowledge" and status == "no_knowledge"):
                continue

            # --- 特定 Agent 处理 ---

            # 行程规划
            if agent_name == "itinerary_planning":
                itinerary = data.get("itinerary")
                booking_summary = data.get("booking_summary")
                # 增强：支持从 data.data.itinerary 获取
                if not itinerary and "data" in data and isinstance(data["data"], dict):
                    itinerary = data["data"].get("itinerary")
                    booking_summary = data["data"].get("booking_summary")
                
                if itinerary:
                    title = itinerary.get('title', '行程规划')
                    self.console.print(f"\n✈️  [bold cyan]{title}[/bold cyan]")
                    self.console.print(f"时长: {itinerary.get('duration', '未知')}\n")

                    # 预订事实由代码根据可信booking_context统一渲染，
                    # 不直接展示模型可能改写过的车次、时间或酒店描述。
                    if isinstance(booking_summary, dict) and booking_summary:
                        self.console.print("[bold]🎫 预订信息[/bold]")
                        for booking_ref in ("outbound", "return", "hotel"):
                            summary_item = booking_summary.get(booking_ref)
                            if isinstance(summary_item, dict) and summary_item.get("text"):
                                self.console.print(f"  • {summary_item['text']}")
                        self.console.print()

                    # 每日行程
                    for day_plan in itinerary.get("daily_plans", []):
                        day_num = day_plan.get("day", 1)
                        self.console.print(f"[bold yellow]第 {day_num} 天[/bold yellow]")

                        # 兼容 activities 和 time_slots
                        activities = day_plan.get("activities") or day_plan.get("time_slots") or []
                        for slot in activities:
                            time = slot.get("time", "")
                            # 兼容 activity 和 location
                            activity = slot.get("activity") or slot.get("location") or ""
                            description = slot.get("description", "")
                            transport = slot.get("transport", "")

                            self.console.print(f"  {time} - {activity}")
                            if description:
                                self.console.print(f"    {description}", style="dim")
                            if transport:
                                self.console.print(f"    🚇 {transport}", style="dim")

                        # 餐食建议
                        meals = day_plan.get("meals", {})
                        if meals:
                            self.console.print()
                            if meals.get("lunch"):
                                self.console.print(f"  🍜 {meals['lunch']}", style="dim")
                            if meals.get("dinner"):
                                self.console.print(f"  🍽️  {meals['dinner']}", style="dim")
                        self.console.print()

                    # 注意事项
                    notes = itinerary.get("notes", [])
                    if notes:
                        self.console.print("[bold]📌 注意事项[/bold]")
                        for note in notes:
                            self.console.print(f"  • {note}")
                    current_agent_shown = True

            # 偏好管理
            elif agent_name == "preference":
                raw_prefs = data.get("preferences")
                # 增强：支持从 data.data.preferences 获取
                if not raw_prefs and "data" in data and isinstance(data["data"], dict):
                    raw_prefs = data["data"].get("preferences")

                if isinstance(raw_prefs, dict):
                    prefs_list = raw_prefs.get("preferences", [])
                else:
                    prefs_list = raw_prefs if isinstance(raw_prefs, list) else []

                if prefs_list:
                    self.console.print("✓ [bold green]已更新您的偏好设置[/bold green]")
                    type_names = {
                        "home_location": "常驻地",
                        "transportation_preference": "交通偏好",
                        "hotel_brands": "酒店偏好",
                        "airlines": "航空公司偏好",
                        "seat_preference": "座位偏好",
                        "meal_preference": "餐食偏好",
                        "budget_level": "预算等级"
                    }
                    for pref in prefs_list:
                        pref_type = pref.get("type", "")
                        pref_value = pref.get("value", "")
                        action = pref.get("action", "replace")
                        display_type = type_names.get(pref_type, pref_type)
                        action_text = "追加" if action == "append" else "设置为"
                        self.console.print(f"  • {display_type} {action_text} [cyan]{pref_value}[/cyan]")
                    current_agent_shown = True
                    has_itinerary = any(r.get("agent_name") == "itinerary_planning" for r in results)
                    if not has_itinerary:
                        self.console.print("\n💡 下次规划行程时会参考这些偏好。", style="dim")
                else:
                    # 检查是否有错误信息
                    err = data.get("error", "")
                    if err:
                        self.console.print(f"偏好未保存: {err}", style="yellow")
                        current_agent_shown = True
                    # 如果只是没提取到，可能就是没偏好，不强求显示，交给兜底逻辑

            # 事项收集
            elif agent_name == "event_collection":
                # 增强：支持从 data.data 获取
                origin = data.get("origin") or data.get("data", {}).get("origin")
                destination = data.get("destination") or data.get("data", {}).get("destination")
                start_date = data.get("start_date") or data.get("data", {}).get("start_date")
                end_date = data.get("end_date") or data.get("data", {}).get("end_date")
                missing_info = data.get("missing_info") or data.get("data", {}).get("missing_info") or []

                has_itinerary = any(r.get("agent_name") == "itinerary_planning" for r in results)
                info_shown = False
                if not has_itinerary:
                    if destination or origin:
                        self.console.print("✓ [bold green]已收集行程信息[/bold green]")
                        if origin: self.console.print(f"  • 出发地: [cyan]{origin}[/cyan]")
                        if destination: self.console.print(f"  • 目的地: [cyan]{destination}[/cyan]")
                        if start_date: self.console.print(f"  • 出发日期: [cyan]{start_date}[/cyan]")
                        if end_date: self.console.print(f"  • 返程日期: [cyan]{end_date}[/cyan]")
                        info_shown = True

                if missing_info and show_missing_info:
                    self.console.print(f"\n💡 还需要补充: {', '.join(missing_info)}", style="yellow")
                    info_shown = True
                
                if info_shown:
                    current_agent_shown = True

            # 信息查询
            elif agent_name == "information_query":
                query_results = data.get("results")
                if not query_results and "data" in data and isinstance(data["data"], dict):
                    query_results = data["data"].get("results")
                if not query_results:
                    query_results = data # 兜底：data 本身就是 results

                if not isinstance(query_results, dict):
                    query_results = {}

                summary = query_results.get("summary", "")
                sources = query_results.get("sources", []) or []
                message = query_results.get("message", "")
                error = query_results.get("error", "")

                if summary:
                    self.console.print(f"\n{summary}")
                    current_agent_shown = True
                elif message:
                    self.console.print(f"\n{message}", style="dim")
                    current_agent_shown = True
                elif error:
                    self.console.print(f"\n{error}", style="yellow")
                    current_agent_shown = True

                if sources:
                    self.console.print("\n[bold]参考来源[/bold]")
                    for i, source in enumerate(sources[:3], 1):
                        url = source.get("url", "") if isinstance(source, dict) else str(source)
                        self.console.print(f"  {i}. {url}", style="dim")
                    current_agent_shown = True

            # RAG知识库查询
            elif agent_name == "rag_knowledge":
                answer = data.get("answer")
                if not answer and "data" in data and isinstance(data["data"], dict):
                    answer = data["data"].get("answer")
                
                # 增强：也查找 content
                if not answer:
                    answer = data.get("content") or data.get("data", {}).get("content")

                # 深度清洗
                if isinstance(answer, dict):
                    answer = answer.get("answer", str(answer))
                
                if isinstance(answer, str) and answer.strip().startswith("{") and answer.strip().endswith("}"):
                    try:
                        import json
                        json_obj = json.loads(answer)
                        if isinstance(json_obj, dict) and "answer" in json_obj:
                            answer = json_obj["answer"]
                    except:
                        pass

                if answer:
                    self.console.print(f"\n{answer}")
                    current_agent_shown = True

            # 记忆查询
            elif agent_name == "memory_query":
                query_result = data.get("answer") or data.get("result") or data.get("content")
                if not query_result and "data" in data and isinstance(data["data"], dict):
                    inner = data["data"]
                    query_result = inner.get("answer") or inner.get("result") or inner.get("content")

                if query_result:
                    self.console.print(f"\n{query_result}")
                    current_agent_shown = True

            # --- 通用兜底 (如果特定逻辑未生效) ---
            if not current_agent_shown:
                # 尝试查找通用字段
                common_keys = ["answer", "content", "result", "message", "summary", "text", "description"]
                fallback_content = ""
                
                # 扁平查找
                for k in common_keys:
                    if k in data and isinstance(data[k], str) and data[k].strip():
                        fallback_content = data[k]
                        break
                
                # 嵌套查找 data.data
                if not fallback_content and "data" in data and isinstance(data["data"], dict):
                    for k in common_keys:
                        if k in data["data"] and isinstance(data["data"][k], str) and data["data"][k].strip():
                            fallback_content = data["data"][k]
                            break

                if fallback_content:
                    self.console.print(f"\n{fallback_content}")
                    current_agent_shown = True
                else:
                    # 实在啥也没有，打印个成功标记，避免完全静默
                    agent_display_name = self._get_agent_display_name(agent_name)
                    self.console.print(f"✓ {agent_display_name}已完成", style="green")
                    current_agent_shown = True

            if current_agent_shown:
                has_output = True

        return has_output

    def _get_agent_display_name(self, agent_name: str) -> str:
        """获取智能体的显示名称"""
        # 与 README / LazyAgentRegistry 保持一致，仅保留已存在的 6 个子智能体
        agent_display_names = {
            "event_collection": "事项收集",
            "preference": "偏好管理",
            "itinerary_planning": "行程规划",
            "information_query": "信息查询",
            "rag_knowledge": "知识库查询",
            "memory_query": "记忆查询",
        }
        return agent_display_names.get(agent_name, agent_name)

    def show_status(self):
        """显示当前状态"""
        # 记忆统计
        full_context = self.memory_manager.get_full_context()
        short_term_stats = full_context["short_term"]["statistics"]
        long_term_stats = full_context["long_term"]["statistics"]

        memory_table = Table(title="记忆状态", show_header=True, header_style="bold magenta")
        memory_table.add_column("类型", style="cyan")
        memory_table.add_column("状态", style="white")

        memory_table.add_row(
            "短期记忆",
            f"{short_term_stats['total_messages']} 条消息"
        )
        memory_table.add_row(
            "长期记忆",
            f"{long_term_stats['total_trips']} 次行程"
        )
        memory_table.add_row(
            "已加载智能体",
            f"{len(self._agent_cache)} 个"
        )

        self.console.print(memory_table)
        self.console.print()

        # 历史对话
        recent_messages = self.memory_manager.short_term.get_recent_context(n_turns=5)
        if recent_messages:
            dialogue_table = Table(title="最近对话 (最多5轮)", show_header=True, header_style="bold cyan")
            dialogue_table.add_column("角色", style="cyan", width=8)
            dialogue_table.add_column("内容", style="white", width=60)
            dialogue_table.add_column("时间", style="dim", width=12)

            for msg in recent_messages:
                role_name = "👤 用户" if msg["role"] == "user" else "🤖 助手"
                content = msg["content"]

                # 截断过长的内容
                if len(content) > 100:
                    content = content[:100] + "..."

                # 格式化时间
                timestamp = msg.get("timestamp", "")
                if timestamp:
                    from datetime import datetime
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M:%S")
                    except:
                        time_str = ""
                else:
                    time_str = ""

                dialogue_table.add_row(role_name, content, time_str)

            self.console.print(dialogue_table)
            self.console.print()

    async def run_health_check(self):
        """在会话内执行健康检查并显示熔断器状态"""
        if self.circuit_breaker:
            status = self.circuit_breaker.get_status()
            self.console.print(f"[bold]熔断器[/bold]: {status['state']}", style="cyan")
        ok, msg = await check_llm_health(
            base_url=LLM_CONFIG["base_url"],
            api_key=LLM_CONFIG["api_key"],
            model_name=LLM_CONFIG["model_name"],
            timeout_sec=RESILIENCE_CONFIG.get("health_check_timeout_sec", 10.0),
        )
        if ok:
            self.console.print("LLM 服务: [green]正常[/green]", style="bold")
        else:
            self.console.print(f"LLM 服务: [red]不可用[/red] - {msg}", style="bold")
        self.console.print()

    def show_history(self):
        """显示历史行程"""
        history = self.memory_manager.long_term.get_trip_history(10)

        if not history:
            self.console.print("暂无历史行程", style="yellow")
            return

        table = Table(title="历史行程", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan")
        table.add_column("出发地", style="white")
        table.add_column("目的地", style="white")
        table.add_column("日期", style="white")
        table.add_column("目的", style="white")

        for trip in history:
            table.add_row(
                trip.get("trip_id", ""),
                trip.get("origin", ""),
                trip.get("destination", ""),
                trip.get("start_date", ""),
                trip.get("purpose", "")
            )

        self.console.print(table)

    def show_preferences(self):
        """显示用户偏好"""
        prefs = self.memory_manager.long_term.get_preference()

        table = Table(title="用户偏好", show_header=True, header_style="bold magenta")
        table.add_column("类型", style="cyan")
        table.add_column("值", style="white")

        for key, value in prefs.items():
            if value:
                table.add_row(key, str(value))

        self.console.print(table)

    async def run(self):
        """运行 CLI"""
        # 打印横幅
        self.print_banner()

        # 初始化系统
        await self.initialize_system()

        # 主循环
        while True:
            try:
                # 获取用户输入
                user_input = Prompt.ask("\n[cyan]>[/cyan]")

                if not user_input.strip():
                    continue

                # 处理命令
                command = user_input.strip().lower()

                if command == "exit":
                    # 仅在会话结束时生成一次摘要；后续会话
                    # 直接从 SQLite 读取，避免每轮重复调用 LLM。
                    await self.memory_manager.end_session()
                    self.console.print("再见！", style="cyan")
                    break
                elif command == "help":
                    self.print_help()
                elif command == "status":
                    self.show_status()
                elif command == "health":
                    await self.run_health_check()
                elif command == "clear":
                    self.memory_manager.clear_session_state()

                    self.console.print(
                        "✓ 已清空当前任务和短期记忆",
                        style="green",
                    )
                elif command == "history":
                    self.show_history()
                elif command == "preferences":
                    self.show_preferences()
                else:
                    # 处理自然语言查询
                    await self.process_query(user_input)

            except KeyboardInterrupt:
                self.console.print("\n使用 'exit' 退出", style="dim")
            except CircuitOpenError:
                self.console.print("\n[bold yellow]⚠ 服务暂时不可用，请稍后再试。[/bold yellow]", style="dim")
            except Exception as e:
                self.console.print(f"\n错误: {e}", style="red")


def run_health_check_standalone() -> int:
    """
    独立执行健康检查（用于 `python cli.py health`）。
    不进入交互式 CLI，只检测 LLM 是否可达。
    Returns:
        0 成功，1 失败（便于脚本/监控）
    """
    import asyncio
    init_agentscope()
    ok, msg = asyncio.run(check_llm_health(
        base_url=LLM_CONFIG["base_url"],
        api_key=LLM_CONFIG["api_key"],
        model_name=LLM_CONFIG["model_name"],
        timeout_sec=RESILIENCE_CONFIG.get("health_check_timeout_sec", 10.0),
    ))
    if ok:
        print("OK")
        return 0
    print(f"FAIL: {msg}")
    return 1


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1].strip().lower() == "health":
        exit(run_health_check_standalone())
    cli = AligoCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
