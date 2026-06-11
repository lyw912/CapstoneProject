# -*- coding: utf-8 -*-
"""
Public Opinion Analysis System Configuration File

This module uses pydantic-settings to manage global configuration, supporting automatic loading from environment variables and .env files.
Data model definitions:
- This file - Configuration model definitions
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import Optional, Literal
from loguru import logger


# Calculate .env priority: current working directory first, then project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent
CWD_ENV: Path = Path.cwd() / ".env"
ENV_FILE: str = str(CWD_ENV if CWD_ENV.exists() else (PROJECT_ROOT / ".env"))


class Settings(BaseSettings):
    """
    Global configuration; supports .env and environment variable auto-loading.
    Variable names match original config.py in uppercase for smooth transition.
    """
    # ================== Flask Server Configuration ====================
    HOST: str = Field("0.0.0.0", description="Server host address, e.g., 0.0.0.0 or 127.0.0.1")
    PORT: int = Field(5000, description="Flask server port, default 5000")
    OUTPUT_DIR: Path = Field(Path("output"), description="Output file directory")
    SAVE_INTERMEDIATE_STATES: bool = Field(
        False,
        description="Whether to write intermediate state JSON when saving research reports (MediaEngine / QueryEngine)",
    )

    # ================== Report Engine Paths and Logs ====================
    LOG_FILE: Path = Field(
        Path("logs/report.log"),
        description="Report Engine dedicated log file path (loguru sink)",
    )
    CHAPTER_OUTPUT_DIR: Path = Field(
        Path("output/chapters"),
        description="Chapter-level JSON output directory",
    )
    DOCUMENT_IR_OUTPUT_DIR: Path = Field(
        Path("output/document_ir"),
        description="Bound Document IR output directory",
    )
    TEMPLATE_DIR: Path = Field(
        Path("ReportEngine/report_template"),
        description="Report Markdown template directory",
    )
    REPORT_OUTPUT_LANGUAGE: Literal["en", "zh"] = Field(
        "en",
        description="Report output language: en = English-only prose/headings; zh = allow Chinese templates and labels",
    )
    REPORT_TRANSLATE_INPUT_TO_EN: bool = Field(
        True,
        description="When REPORT_OUTPUT_LANGUAGE=en, pre-translate Chinese upstream inputs to English when detected",
    )
    REPORT_INPUT_TRANSLATION_TIMEOUT_SECONDS: int = Field(
        45,
        description="Maximum total seconds allowed for pre-translation; on timeout skip remaining fields and continue generation",
    )
    JSON_ERROR_LOG_DIR: Path = Field(
        Path("output/json_error_logs"),
        description="Diagnostic log directory for chapter JSON parsing failures",
    )
    CHAPTER_JSON_MAX_ATTEMPTS: int = Field(
        5,
        description="Maximum attempts for single chapter JSON generation/repair",
    )

    # ====================== Database Configuration ======================
    DB_DIALECT: str = Field("postgresql", description="Database type, optional mysql or postgresql; configure with other connection info")
    DB_HOST: str = Field("your_db_host", description="Database host, e.g., localhost or 127.0.0.1")
    DB_PORT: int = Field(3306, description="Database port, default 3306")
    DB_USER: str = Field("your_db_user", description="Database username")
    DB_PASSWORD: str = Field("your_db_password", description="Database password")
    DB_NAME: str = Field("your_db_name", description="Database name")
    DB_CHARSET: str = Field("utf8mb4", description="Database charset, recommended utf8mb4, emoji compatible")
    
    # ======================= LLM Related =======================
    # Our LLM model API sponsor: https://aihubmix.com/?aff=8Ds9, provides comprehensive model APIs
    
    # Insight Agent (recommended Kimi, apply at: https://platform.moonshot.cn/)
    INSIGHT_ENGINE_API_KEY: Optional[str] = Field(None, description="Insight Agent (recommended kimi-k2, official apply: https://platform.moonshot.cn/) API key for main LLM. 🚩Please apply with recommended config first and get it running before adjusting KEY, BASE_URL, and MODEL_NAME.")
    INSIGHT_ENGINE_BASE_URL: Optional[str] = Field("https://api.moonshot.cn/v1", description="Insight Agent LLM BaseUrl, customizable by provider")
    INSIGHT_ENGINE_MODEL_NAME: str = Field("kimi-k2-0711-preview", description="Insight Agent LLM model name, e.g., kimi-k2-0711-preview")
    
    # Media Agent (recommended Gemini, proxy provider: https://aihubmix.com/?aff=8Ds9)
    MEDIA_ENGINE_API_KEY: Optional[str] = Field(None, description="Media Agent (recommended gemini-2.5-pro, proxy apply: https://aihubmix.com/?aff=8Ds9) API key")
    MEDIA_ENGINE_BASE_URL: Optional[str] = Field("https://aihubmix.com/v1", description="Media Agent LLM BaseUrl, adjustable by proxy service")
    MEDIA_ENGINE_MODEL_NAME: str = Field("gemini-2.5-pro", description="Media Agent LLM model name, e.g., gemini-2.5-pro")
    
    # Query Agent (recommended DeepSeek, apply at: https://www.deepseek.com/)
    QUERY_ENGINE_API_KEY: Optional[str] = Field(None, description="Query Agent (recommended deepseek, official apply: https://platform.deepseek.com/) API key")
    QUERY_ENGINE_BASE_URL: Optional[str] = Field("https://api.deepseek.com", description="Query Agent LLM BaseUrl")
    QUERY_ENGINE_MODEL_NAME: str = Field("deepseek-chat", description="Query Agent LLM model name, e.g., deepseek-reasoner")
    
    # Report Agent (recommended Gemini, proxy provider: https://aihubmix.com/?aff=8Ds9)
    REPORT_ENGINE_API_KEY: Optional[str] = Field(None, description="Report Agent (recommended gemini-2.5-pro, proxy apply: https://aihubmix.com/?aff=8Ds9) API key")
    REPORT_ENGINE_BASE_URL: Optional[str] = Field("https://aihubmix.com/v1", description="Report Agent LLM BaseUrl, adjustable by proxy service")
    REPORT_ENGINE_MODEL_NAME: str = Field("gemini-2.5-pro", description="Report Agent LLM model name, e.g., gemini-2.5-pro")

    # ================== Agent Observability ==================
    LANGSMITH_TRACING: bool = Field(False, description="Enable LangSmith/LangChain tracing for agent runs")
    LANGSMITH_API_KEY: Optional[str] = Field(None, description="LangSmith API key")
    LANGSMITH_ENDPOINT: Optional[str] = Field("https://api.smith.langchain.com", description="LangSmith API endpoint")
    LANGSMITH_PROJECT: str = Field("public-opinion-analysis", description="LangSmith project name")
    LANGCHAIN_TRACING_V2: Optional[bool] = Field(None, description="Backward-compatible LangChain tracing flag")
    LANGCHAIN_PROJECT: Optional[str] = Field(None, description="Backward-compatible LangChain project name")

    # MindSpider Agent (recommended DeepSeek, official apply: https://platform.deepseek.com/)
    MINDSPIDER_API_KEY: Optional[str] = Field(None, description="MindSpider Agent (recommended deepseek, official apply: https://platform.deepseek.com/) API key")
    MINDSPIDER_BASE_URL: Optional[str] = Field(None, description="MindSpider Agent BaseUrl, configurable by selected service")
    MINDSPIDER_MODEL_NAME: Optional[str] = Field(None, description="MindSpider Agent model name, e.g., deepseek-reasoner")
    
    # Forum Host (Qwen3 latest model, using SiliconFlow platform, apply at: https://cloud.siliconflow.cn/)
    FORUM_HOST_API_KEY: Optional[str] = Field(None, description="Forum Host (recommended qwen-plus, official apply: https://www.aliyun.com/product/bailian) API key")
    FORUM_HOST_BASE_URL: Optional[str] = Field(None, description="Forum Host LLM BaseUrl, configurable by selected service")
    FORUM_HOST_MODEL_NAME: Optional[str] = Field(None, description="Forum Host LLM model name, e.g., qwen-plus")
    
    # SQL keyword Optimizer (small parameter Qwen3 model, using SiliconFlow platform, apply at: https://cloud.siliconflow.cn/)
    KEYWORD_OPTIMIZER_API_KEY: Optional[str] = Field(None, description="SQL Keyword Optimizer (recommended qwen-plus, official apply: https://www.aliyun.com/product/bailian) API key")
    KEYWORD_OPTIMIZER_BASE_URL: Optional[str] = Field(None, description="Keyword Optimizer BaseUrl, configurable by selected service")
    KEYWORD_OPTIMIZER_MODEL_NAME: Optional[str] = Field(None, description="Keyword Optimizer LLM model name, e.g., qwen-plus")
    
    # ================== Network Tool Configuration ====================
    # Tavily API (apply at: https://www.tavily.com/)
    TAVILY_API_KEY: Optional[str] = Field(None, description="Tavily API (apply at: https://www.tavily.com/) API key for Tavily web search")

    SEARCH_TOOL_TYPE: Literal["AnspireAPI", "BochaAPI"] = Field("AnspireAPI", description="Web search tool type, supports BochaAPI or AnspireAPI, default AnspireAPI")
    # Bocha API (apply at: https://open.bochaai.com/)
    BOCHA_BASE_URL: Optional[str] = Field("https://api.bocha.cn/v1/ai-search", description="Bocha AI Search BaseUrl or Bocha web search BaseUrl")
    BOCHA_WEB_SEARCH_API_KEY: Optional[str] = Field(None, description="Bocha API (apply at: https://open.bochaai.com/) API key for Bocha search")

    # Anspire AI Search API (apply at: https://open.anspire.cn/?share_code=3E1FUOUH)
    ANSPIRE_BASE_URL: Optional[str] = Field("https://plugin.anspire.cn/api/ntsearch/search", description="Anspire AI Search BaseUrl")
    ANSPIRE_API_KEY: Optional[str] = Field(None, description="Anspire AI Search API (apply at: https://open.anspire.cn/?share_code=3E1FUOUH) API key for Anspire search")

    
    # ================== Insight Engine Search Configuration ====================
    DEFAULT_SEARCH_HOT_CONTENT_LIMIT: int = Field(100, description="Default maximum hot content count")
    DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE: int = Field(50, description="Global topic maximum per table")
    DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE: int = Field(100, description="Topic maximum by date per table")
    DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT: int = Field(500, description="Maximum comments per topic")
    DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT: int = Field(200, description="Platform search topic maximum")
    MAX_SEARCH_RESULTS_FOR_LLM: int = Field(0, description="Maximum search results for LLM")
    MAX_HIGH_CONFIDENCE_SENTIMENT_RESULTS: int = Field(0, description="Maximum high confidence sentiment analysis results")
    MAX_REFLECTIONS: int = Field(3, description="Maximum reflection iterations")
    MAX_PARAGRAPHS: int = Field(6, description="Maximum paragraphs")
    SEARCH_TIMEOUT: int = Field(240, description="Single search request timeout")
    MAX_CONTENT_LENGTH: int = Field(500000, description="Maximum search content length")
    
    model_config = ConfigDict(
        env_file=ENV_FILE,
        env_prefix="",
        case_sensitive=False,
        extra="allow"
    )


# Create global configuration instance
settings = Settings()


def reload_settings() -> Settings:
    """
    Reload configuration
    
    Reload configuration from .env file and environment variables, update global settings instance.
    Used for dynamic configuration updates at runtime.
    
    Returns:
        Settings: Newly created configuration instance
    """
    
    global settings
    settings = Settings()
    return settings
