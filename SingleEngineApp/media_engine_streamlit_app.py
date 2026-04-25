"""
Streamlit Web Interface
Provides a user-friendly web interface for Multimodal Agent (underlying MediaEngine).
"""

import os
import sys
import streamlit as st
from datetime import datetime
import json
import locale
from loguru import logger

# Set UTF-8 encoding environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# Set system encoding
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        pass

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from MediaEngine import DeepSearchAgent, AnspireSearchAgent, Settings
from config import settings
from utils.github_issues import error_with_issue_link


def main():
    """Main function"""
    st.set_page_config(
        page_title="Multimodal Agent",
        page_icon="",
        layout="wide"
    )

    st.title("Multimodal Agent")
    st.markdown("Multimodal content understanding: cross-modal analysis of videos, images, and structured info cards")
    st.markdown("Breaks through traditional text communication limitations, extensively browses videos, images, and live streams on TikTok, Kuaishou, and Xiaohongshu")
    st.markdown("Enhanced capabilities using multimodal structured information from modern search engines such as calendar cards, weather cards, and stock cards")

    # Check URL parameters
    try:
        # Try using new version query_params
        query_params = st.query_params
        auto_query = query_params.get('query', '')
        auto_search = query_params.get('auto_search', 'false').lower() == 'true'
    except AttributeError:
        # Fallback for older versions
        query_params = st.experimental_get_query_params()
        auto_query = query_params.get('query', [''])[0]
        auto_search = query_params.get('auto_search', ['false'])[0].lower() == 'true'

    # ----- Configuration is hardcoded -----
    # Force use Gemini
    model_name = settings.MEDIA_ENGINE_MODEL_NAME or "gemini-2.5-pro"
    # Default advanced configuration
    max_reflections = 2
    max_content_length = 20000

    # Simplified research query display area

    # If there's an auto query, use it as default, otherwise show placeholder
    display_query = auto_query if auto_query else "Waiting to receive analysis content from main page..."

    # Read-only query display area
    st.text_area(
        "Current Query",
        value=display_query,
        height=100,
        disabled=True,
        help="Query content is controlled by the search box on the main page",
        label_visibility="hidden"
    )

    # Auto search logic
    start_research = False
    query = auto_query

    if auto_search and auto_query and 'auto_search_executed' not in st.session_state:
        st.session_state.auto_search_executed = True
        start_research = True
    elif auto_query and not auto_search:
        st.warning("Waiting for search start signal...")

    # Validate configuration
    if start_research:
        if not query.strip():
            st.error("Please enter research query")
            logger.error("Please enter research query")
            return

        # Since using Gemini, check related API keys
        if not settings.MEDIA_ENGINE_API_KEY:
            st.error("Please set MEDIA_ENGINE_API_KEY in your environment variables")
            logger.error("Please set MEDIA_ENGINE_API_KEY in your environment variables")
            return

        # Automatically use API keys from configuration file
        engine_key = settings.MEDIA_ENGINE_API_KEY
        bocha_key = settings.BOCHA_WEB_SEARCH_API_KEY
        ansire_key = settings.ANSPIRE_API_KEY

        # Build Settings (pydantic_settings style, prioritize uppercase environment variables)
        if settings.SEARCH_TOOL_TYPE == "BochaAPI":
            if not bocha_key:
                st.error("Please set BOCHA_WEB_SEARCH_API_KEY in your environment variables")
                logger.error("Please set BOCHA_WEB_SEARCH_API_KEY in your environment variables")
                return
            logger.info("Using Bocha search API key")
            config = Settings(
                MEDIA_ENGINE_API_KEY=engine_key,
                MEDIA_ENGINE_BASE_URL=settings.MEDIA_ENGINE_BASE_URL,
                MEDIA_ENGINE_MODEL_NAME=model_name,
                SEARCH_TOOL_TYPE="BochaAPI",
                BOCHA_WEB_SEARCH_API_KEY=bocha_key,
                MAX_REFLECTIONS=max_reflections,
                SEARCH_CONTENT_MAX_LENGTH=max_content_length,
                OUTPUT_DIR="media_engine_streamlit_reports",
            )
        elif settings.SEARCH_TOOL_TYPE == "AnspireAPI":
            if not ansire_key:
                st.error("Please set ANSPIRE_API_KEY in your environment variables")
                logger.error("Please set ANSPIRE_API_KEY in your environment variables")
                return
            logger.info("Using Anspire search API key")
            config = Settings(
                MEDIA_ENGINE_API_KEY=engine_key,
                MEDIA_ENGINE_BASE_URL=settings.MEDIA_ENGINE_BASE_URL,
                MEDIA_ENGINE_MODEL_NAME=model_name,
                SEARCH_TOOL_TYPE="AnspireAPI",
                ANSPIRE_API_KEY=ansire_key,
                MAX_REFLECTIONS=max_reflections,
                SEARCH_CONTENT_MAX_LENGTH=max_content_length,
                OUTPUT_DIR="media_engine_streamlit_reports",
            )
        else:
            st.error(f"Unknown search tool type: {settings.SEARCH_TOOL_TYPE}")
            logger.error(f"Unknown search tool type: {settings.SEARCH_TOOL_TYPE}")
            return

        # Execute research
        execute_research(query, config)


def execute_research(query: str, config: Settings):
    """Execute research (MediaEngine unified to LangGraph, aligned with DeepSearchAgent / AnspireSearchAgent research())."""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Initializing Agent...")
        if config.SEARCH_TOOL_TYPE == "BochaAPI":
            agent = DeepSearchAgent(config)
        elif config.SEARCH_TOOL_TYPE == "AnspireAPI":
            agent = AnspireSearchAgent(config)
        else:
            raise ValueError(f"Unknown search tool type: {config.SEARCH_TOOL_TYPE}")
        st.session_state.agent = agent

        progress_bar.progress(15)
        status_text.text("Executing deep research (search, reflection and report generation)...")
        logger.info("Starting LangGraph deep research")
        final_report = agent.research(query, save_report=True)
        progress_bar.progress(100)

        status_text.text("Research completed!")
        logger.info("Research completed!")
        display_results(agent, final_report)

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_display = error_with_issue_link(
            f"Error occurred during research: {str(e)}",
            error_traceback,
            app_name="Multimodal Agent Streamlit App"
        )
        st.error(error_display)
        logger.exception(f"Error occurred during research: {str(e)}")


def display_results(agent: DeepSearchAgent, final_report: str):
    """Display research results"""
    st.header("Research Results")

    # Result tabs (download options removed)
    tab1, tab2 = st.tabs(["Research Summary", "Citation Info"])

    with tab1:
        st.markdown(final_report)

    with tab2:
        # Paragraph details
        st.subheader("Paragraph Details")
        for i, paragraph in enumerate(agent.state.paragraphs):
            with st.expander(f"Paragraph {i + 1}: {paragraph.title}"):
                st.write("**Expected Content:**", paragraph.content)
                st.write("**Final Content:**", paragraph.research.latest_summary[:300] + "..."
                if len(paragraph.research.latest_summary) > 300
                else paragraph.research.latest_summary)
                st.write("**Search Count:**", paragraph.research.get_search_count())
                st.write("**Reflection Count:**", paragraph.research.reflection_iteration)

        # Search history
        st.subheader("Search History")
        all_searches = []
        for paragraph in agent.state.paragraphs:
            all_searches.extend(paragraph.research.search_history)

        if all_searches:
            for i, search in enumerate(all_searches):
                query_label = search.query if search.query else "Unrecorded query"
                with st.expander(f"Search {i + 1}: {query_label}"):
                    paragraph_title = getattr(search, "paragraph_title", "") or "Unlabeled paragraph"
                    search_tool = getattr(search, "search_tool", "") or "Unlabeled tool"
                    has_result = getattr(search, "has_result", True)
                    st.write("**Paragraph:**", paragraph_title)
                    st.write("**Tool Used:**", search_tool)
                    preview = search.content or ""
                    if not isinstance(preview, str):
                        preview = str(preview)
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    st.write("**URL:**", search.url or "None")
                    st.write("**Title:**", search.title or "None")
                    st.write("**Content Preview:**", preview if preview else "No content available")
                    if not has_result:
                        st.info("This search returned no results")
                    if search.score:
                        st.write("**Relevance Score:**", search.score)


if __name__ == "__main__":
    main()
