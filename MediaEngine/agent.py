"""
Deep Search Agent main class
Integrates all modules to implement the complete deep search workflow (LangGraph orchestration)
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
    """Deep Search Agent main class"""
    
    def __init__(self, config: Optional[Settings] = None):
        """
        Initialize Deep Search Agent
        
        Args:
            config: Configuration object, auto-loaded if not provided
        """
        self.config = config or settings
        
        # Initialize LLM client
        self.llm_client = self._initialize_llm()
        
        # Initialize search tools
        self.search_agency = BochaMultimodalSearch(api_key=(self.config.BOCHA_API_KEY or self.config.BOCHA_WEB_SEARCH_API_KEY))
        
        # Initialize nodes
        self._initialize_nodes()
        
        # State
        self.state = State()
        
        # Ensure output directory exists
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        self._media_graph = None
        
        logger.info(f"Media Agent initialized")
        logger.info(f"Using LLM: {self.llm_client.get_model_info()}")
        logger.info(f"Search tools: BochaMultimodalSearch (supports 5 multimodal search tools)")
    
    def _initialize_llm(self) -> LLMClient:
        """Initialize LLM client"""
        return LLMClient(
            api_key=(self.config.MEDIA_ENGINE_API_KEY or self.config.MINDSPIDER_API_KEY),
            model_name=(self.config.MEDIA_ENGINE_MODEL_NAME or self.config.MINDSPIDER_MODEL_NAME),
            base_url=(self.config.MEDIA_ENGINE_BASE_URL or self.config.MINDSPIDER_BASE_URL),
        )
    
    def _initialize_nodes(self):
        """Initialize processing nodes"""
        self.first_search_node = FirstSearchNode(self.llm_client)
        self.reflection_node = ReflectionNode(self.llm_client)
        self.first_summary_node = FirstSummaryNode(self.llm_client)
        self.reflection_summary_node = ReflectionSummaryNode(self.llm_client)
        self.report_formatting_node = ReportFormattingNode(self.llm_client)
    
    def execute_search_tool(self, tool_name: str, query: str, **kwargs) -> BochaResponse:
        """
        Execute specified search tool
        
        Args:
            tool_name: Tool name, available options:
                - "comprehensive_search": Comprehensive search (default)
                - "web_search_only": Web search only
                - "search_for_structured_data": Structured data query
                - "search_last_24_hours": Latest information within 24 hours
                - "search_last_week": Information from this week
            query: Search query
            **kwargs: Additional parameters (e.g., max_results)
            
        Returns:
            BochaResponse object
        """
        logger.info(f"  → Executing search tool: {tool_name}")
        
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
            logger.info(f"  ⚠️  Unknown search tool: {tool_name}, using default comprehensive search")
            return self.search_agency.comprehensive_search(query)
    
    @property
    def media_graph(self):
        """Lazy initialization of LangGraph, consistent with QueryEngine.query_graph."""
        if self._media_graph is None:
            from .graph.builder import build_media_agent_graph
            self._media_graph = build_media_agent_graph(self)
            logger.info("[MediaAgent] LangGraph initialized")
        return self._media_graph

    def _run_research_graph(self, query: str, save_report: bool) -> str:
        """Execute LangGraph (synchronous nodes use invoke)."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting deep research: {query}")
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
        logger.info("Deep research completed!")
        logger.info(f"{'='*60}")
        return final_report

    async def research_async(self, query: str, save_report: bool = True) -> str:
        """Async entry point: run LangGraph in thread pool to avoid blocking event loop."""
        return await asyncio.to_thread(self._run_research_graph, query, save_report)

    def research(self, query: str, save_report: bool = True) -> str:
        """
        Execute deep research (LangGraph orchestration).

        Args:
            query: Research query
            save_report: Whether to save report to file

        Returns:
            Final report content
        """
        try:
            return self._run_research_graph(query, save_report)
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error occurred during research: {str(e)} \nError traceback: {error_traceback}")
            raise e
    
    def _save_report(self, report_content: str):
        """Save report to file"""
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(c for c in self.state.query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        query_safe = query_safe.replace(' ', '_')[:30]
        
        filename = f"deep_search_report_{query_safe}_{timestamp}.md"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        
        # Save report
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {filepath}")
        
        # Save state (if configuration allows)
        if self.config.SAVE_INTERMEDIATE_STATES:
            state_filename = f"state_{query_safe}_{timestamp}.json"
            state_filepath = os.path.join(self.config.OUTPUT_DIR, state_filename)
            self.state.save_to_file(state_filepath)
            logger.info(f"State saved to: {state_filepath}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get progress summary"""
        return self.state.get_progress_summary()
    
    def load_state(self, filepath: str):
        """Load state from file"""
        self.state = State.load_from_file(filepath)
        logger.info(f"State loaded from {filepath}")
    
    def save_state(self, filepath: str):
        """Save state to file"""
        self.state.save_to_file(filepath)
        logger.info(f"State saved to {filepath}")

class AnspireSearchAgent(DeepSearchAgent):
    """Deep Search Agent using Anspire search engine"""
    
    def __init__(self, config: Settings | None = None):
        self.config = config or settings
        
        # Initialize LLM client
        self.llm_client = self._initialize_llm()
        
        # Initialize search tools
        self.search_agency = AnspireAISearch(api_key=self.config.ANSPIRE_API_KEY)

        # Initialize nodes
        self._initialize_nodes()
        
        # State
        self.state = State()
        
        # Ensure output directory exists
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        self._media_graph = None
        
        logger.info(f"Media Agent initialized")
        logger.info(f"Using LLM: {self.llm_client.get_model_info()}")
        logger.info(f"Search tools: AnspireSearch")

    def execute_search_tool(self, tool_name: str, query: str, **kwargs) -> AnspireResponse:
        # TODO: Use Anspire search tool to execute search
        """
        Execute specified search tool
        
        Args:
            tool_name: Tool name, available options:
                - "comprehensive_search": Comprehensive search (default)
                - "search_last_24_hours": Latest information within 24 hours
                - "search_last_week": Information from this week
            query: Search query
            **kwargs: Additional parameters (e.g., max_results)
            
        Returns:
            AnspireResponse object
        """
        logger.info(f"  → Executing search tool: {tool_name}")
        
        if tool_name == "comprehensive_search":
            max_results = kwargs.get("max_results", 10)
            return self.search_agency.comprehensive_search(query, max_results)
        elif tool_name == "search_last_24_hours":
            return self.search_agency.search_last_24_hours(query)
        elif tool_name == "search_last_week":
            return self.search_agency.search_last_week(query)
        else:
            logger.info(f"  ⚠️  Unknown search tool: {tool_name}, using default comprehensive search")
            return self.search_agency.comprehensive_search(query)


def create_agent(config_file: Optional[str] = None) -> DeepSearchAgent:
    """
    Convenience function to create Deep Search Agent instance
    
    Args:
        config_file: Configuration file path
        
    Returns:
        DeepSearchAgent instance
    """
    settings = Settings()
    if settings.SEARCH_TOOL_TYPE == "AnspireAPI":
        return AnspireSearchAgent(settings)
    return DeepSearchAgent(settings)
