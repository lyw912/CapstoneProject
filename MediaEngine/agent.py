"""
Deep Search Agent主类
整合所有模块，实现完整的深度搜索流程（LangGraph 编排）
"""

import asyncio
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from loguru import logger
from .llms import LLMClient
from .nodes import (
    FirstSearchNode,
    ReflectionNode,
    FirstSummaryNode,
    ReflectionSummaryNode,
    ReportFormattingNode
)
from .state import State
from .tools import BochaMultimodalSearch, BochaResponse, AnspireAISearch, AnspireResponse
from .utils import settings, Settings


class DeepSearchAgent:
    """Deep Search Agent主类"""
    
    def __init__(self, config: Optional[Settings] = None):
        """
        初始化Deep Search Agent
        
        Args:
            config: 配置对象，如果不提供则自动加载
        """
        self.config = config or settings
        
        # 初始化LLM客户端
        self.llm_client = self._initialize_llm()
        
        # 初始化搜索工具集
        self.search_agency = BochaMultimodalSearch(api_key=(self.config.BOCHA_API_KEY or self.config.BOCHA_WEB_SEARCH_API_KEY))
        
        # 初始化节点
        self._initialize_nodes()
        
        # 状态
        self.state = State()
        
        # 确保输出目录存在
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        self._media_graph = None
        
        logger.info(f"Media Agent已初始化")
        logger.info(f"使用LLM: {self.llm_client.get_model_info()}")
        logger.info(f"搜索工具集: BochaMultimodalSearch (支持5种多模态搜索工具)")
    
    def _initialize_llm(self) -> LLMClient:
        """初始化LLM客户端"""
        return LLMClient(
            api_key=(self.config.MEDIA_ENGINE_API_KEY or self.config.MINDSPIDER_API_KEY),
            model_name=(self.config.MEDIA_ENGINE_MODEL_NAME or self.config.MINDSPIDER_MODEL_NAME),
            base_url=(self.config.MEDIA_ENGINE_BASE_URL or self.config.MINDSPIDER_BASE_URL),
        )
    
    def _initialize_nodes(self):
        """初始化处理节点"""
        self.first_search_node = FirstSearchNode(self.llm_client)
        self.reflection_node = ReflectionNode(self.llm_client)
        self.first_summary_node = FirstSummaryNode(self.llm_client)
        self.reflection_summary_node = ReflectionSummaryNode(self.llm_client)
        self.report_formatting_node = ReportFormattingNode(self.llm_client)
    
    def execute_search_tool(self, tool_name: str, query: str, **kwargs) -> BochaResponse:
        """
        执行指定的搜索工具
        
        Args:
            tool_name: 工具名称，可选值：
                - "comprehensive_search": 全面综合搜索（默认）
                - "web_search_only": 纯网页搜索
                - "search_for_structured_data": 结构化数据查询
                - "search_last_24_hours": 24小时内最新信息
                - "search_last_week": 本周信息
            query: 搜索查询
            **kwargs: 额外参数（如max_results）
            
        Returns:
            BochaResponse对象
        """
        logger.info(f"  → 执行搜索工具: {tool_name}")
        
        if tool_name == "comprehensive_search":
            max_results = kwargs.get("max_results", 10)
            return self.search_agency.comprehensive_search(query, max_results)
        elif tool_name == "web_search_only":
            max_results = kwargs.get("max_results", 15)
            return self.search_agency.web_search_only(query, max_results)
        elif tool_name == "search_for_structured_data":
            return self.search_agency.search_for_structured_data(query)
        elif tool_name == "search_last_24_hours":
            return self.search_agency.search_last_24_hours(query)
        elif tool_name == "search_last_week":
            return self.search_agency.search_last_week(query)
        else:
            logger.info(f"  ⚠️  未知的搜索工具: {tool_name}，使用默认综合搜索")
            return self.search_agency.comprehensive_search(query)
    
    @property
    def media_graph(self):
        """延迟初始化 LangGraph，与 QueryEngine.query_graph 一致。"""
        if self._media_graph is None:
            from .graph.builder import build_media_agent_graph
            self._media_graph = build_media_agent_graph(self)
            logger.info("[MediaAgent] LangGraph 已初始化")
        return self._media_graph

    def _run_research_graph(self, query: str, save_report: bool) -> str:
        """执行 LangGraph（同步节点使用 invoke）。"""
        logger.info(f"\n{'='*60}")
        logger.info(f"开始深度研究: {query}")
        logger.info(f"{'='*60}")

        initial_state = {
            "original_query": query,
            "paragraph_index": 0,
            "max_reflections": self.config.MAX_REFLECTIONS,
            "trace_log": [],
            "error_log": [],
        }
        final_state = self.media_graph.invoke(initial_state)
        self.state = final_state["pipeline_state"]
        final_report = final_state.get("final_report") or ""

        if save_report:
            self._save_report(final_report)

        logger.info(f"\n{'='*60}")
        logger.info("深度研究完成！")
        logger.info(f"{'='*60}")
        return final_report

    async def research_async(self, query: str, save_report: bool = True) -> str:
        """异步入口：在线程池中运行 LangGraph，避免阻塞事件循环。"""
        return await asyncio.to_thread(self._run_research_graph, query, save_report)

    def research(self, query: str, save_report: bool = True) -> str:
        """
        执行深度研究（LangGraph 编排）。

        Args:
            query: 研究查询
            save_report: 是否保存报告到文件

        Returns:
            最终报告内容
        """
        try:
            return self._run_research_graph(query, save_report)
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"研究过程中发生错误: {str(e)} \n错误堆栈: {error_traceback}")
            raise e
    
    def _save_report(self, report_content: str):
        """保存报告到文件"""
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(c for c in self.state.query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        query_safe = query_safe.replace(' ', '_')[:30]
        
        filename = f"deep_search_report_{query_safe}_{timestamp}.md"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        
        # 保存报告
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"报告已保存到: {filepath}")
        
        # 保存状态（如果配置允许）
        if self.config.SAVE_INTERMEDIATE_STATES:
            state_filename = f"state_{query_safe}_{timestamp}.json"
            state_filepath = os.path.join(self.config.OUTPUT_DIR, state_filename)
            self.state.save_to_file(state_filepath)
            logger.info(f"状态已保存到: {state_filepath}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """获取进度摘要"""
        return self.state.get_progress_summary()
    
    def load_state(self, filepath: str):
        """从文件加载状态"""
        self.state = State.load_from_file(filepath)
        logger.info(f"状态已从 {filepath} 加载")
    
    def save_state(self, filepath: str):
        """保存状态到文件"""
        self.state.save_to_file(filepath)
        logger.info(f"状态已保存到 {filepath}")

class AnspireSearchAgent(DeepSearchAgent):
    """调用Anspire搜索引擎的Deep Search Agent"""
    
    def __init__(self, config: Settings | None = None):
        self.config = config or settings
        
        # 初始化LLM客户端
        self.llm_client = self._initialize_llm()
        
        # 初始化搜索工具集
        self.search_agency = AnspireAISearch(api_key=self.config.ANSPIRE_API_KEY)

        # 初始化节点
        self._initialize_nodes()
        
        # 状态
        self.state = State()
        
        # 确保输出目录存在
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        self._media_graph = None
        
        logger.info(f"Media Agent已初始化")
        logger.info(f"使用LLM: {self.llm_client.get_model_info()}")
        logger.info(f"搜索工具集: AnspireSearch")

    def execute_search_tool(self, tool_name: str, query: str, **kwargs) -> AnspireResponse:
        # TODO: 使用Anspire搜索工具执行搜索
        """
        执行指定的搜索工具
        
        Args:
            tool_name: 工具名称，可选值：
                - "comprehensive_search": 全面综合搜索（默认）
                - "search_last_24_hours": 24小时内最新信息
                - "search_last_week": 本周信息
            query: 搜索查询
            **kwargs: 额外参数（如max_results）
            
        Returns:
            AnspireResponse对象
        """
        logger.info(f"  → 执行搜索工具: {tool_name}")
        
        if tool_name == "comprehensive_search":
            max_results = kwargs.get("max_results", 10)
            return self.search_agency.comprehensive_search(query, max_results)
        elif tool_name == "search_last_24_hours":
            return self.search_agency.search_last_24_hours(query)
        elif tool_name == "search_last_week":
            return self.search_agency.search_last_week(query)
        else:
            logger.info(f"  ⚠️  未知的搜索工具: {tool_name}，使用默认综合搜索")
            return self.search_agency.comprehensive_search(query)


def create_agent(config_file: Optional[str] = None) -> DeepSearchAgent:
    """
    创建Deep Search Agent实例的便捷函数
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        DeepSearchAgent实例
    """
    settings = Settings()
    if settings.SEARCH_TOOL_TYPE == "AnspireAPI":
        return AnspireSearchAgent(settings)
    return DeepSearchAgent(settings)
