# -*- coding: utf-8 -*-
"""
Micro-Sentiment Configuration File

This module uses pydantic-settings to manage global configuration, supporting automatic loading from environment variables and .env files.
Data model definition locations:
- This file - Configuration model definitions
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from typing import Optional, Literal
from loguru import logger


# Calculate .env priority: prefer current working directory, then project root directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent
CWD_ENV: Path = Path.cwd() / ".env"
ENV_FILE: str = str(CWD_ENV if CWD_ENV.exists() else (PROJECT_ROOT / ".env"))


class Settings(BaseSettings):
    """
    Global configuration; supports automatic loading from .env and environment variables.
    Variable names are uppercase, consistent with the original config.py for smooth transition.
    """
    # ================== Flask Server Configuration ====================
    HOST: str = Field("0.0.0.0", description="Flask server host address, e.g., 0.0.0.0 or 127.0.0.1")
    PORT: int = Field(5000, description="Flask server port number, default 5000")
    OUTPUT_DIR: Path = Field(Path("output"), description="Output file directory")
    SAVE_INTERMEDIATE_STATES: bool = Field(
        False,
        description="Whether to write intermediate state JSON when saving research reports (MediaEngine / QueryEngine)",
    )

    # ====================== Database Configuration ======================
    DB_DIALECT: str = Field("postgresql", description="Database type, optional mysql or postgresql; please configure with other connection information")
    DB_HOST: str = Field("your_db_host", description="Database host, e.g., localhost or 127.0.0.1")
    DB_PORT: int = Field(3306, description="Database port number, default 3306")
    DB_USER: str = Field("your_db_user", description="Database username")
    DB_PASSWORD: str = Field("your_db_password", description="Database password")
    DB_NAME: str = Field("your_db_name", description="Database name")
    DB_CHARSET: str = Field("utf8mb4", description="Database charset, recommended utf8mb4, compatible with emoji")
    
    # ======================= LLM Related =======================
    # Our LLM model API sponsor: https://aihubmix.com/?aff=8Ds9, provides a very comprehensive model API
    
    # Insight Agent (recommended Kimi, application address: https://platform.moonshot.cn/)
    INSIGHT_ENGINE_API_KEY: Optional[str] = Field(None, description="Insight Agent (recommended kimi-k2, official application address: https://platform.moonshot.cn/) API key, used for main LLM. 🚩Please apply according to recommended configuration and get it running first, then adjust KEY, BASE_URL, and MODEL_NAME as needed.")
    INSIGHT_ENGINE_BASE_URL: Optional[str] = Field("https://api.moonshot.cn/v1", description="Insight Agent LLM BaseUrl, can be customized according to vendor")
    INSIGHT_ENGINE_MODEL_NAME: str = Field("kimi-k2-0711-preview", description="Insight Agent LLM model name, e.g., kimi-k2-0711-preview")
    
    # Media Agent (recommended Gemini, recommended relay vendor: https://aihubmix.com/?aff=8Ds9)
    MEDIA_ENGINE_API_KEY: Optional[str] = Field(None, description="Media Agent (recommended gemini-2.5-pro, relay vendor application address: https://aihubmix.com/?aff=8Ds9) API key")
    MEDIA_ENGINE_BASE_URL: Optional[str] = Field("https://aihubmix.com/v1", description="Media Agent LLM BaseUrl, can be adjusted according to relay service")
    MEDIA_ENGINE_MODEL_NAME: str = Field("gemini-2.5-pro", description="Media Agent LLM model name, e.g., gemini-2.5-pro")
    
    # Query Agent (recommended DeepSeek, application address: https://www.deepseek.com/)
    QUERY_ENGINE_API_KEY: Optional[str] = Field(None, description="Query Agent (recommended deepseek, official application address: https://platform.deepseek.com/) API key")
    QUERY_ENGINE_BASE_URL: Optional[str] = Field("https://api.deepseek.com", description="Query Agent LLM BaseUrl")
    QUERY_ENGINE_MODEL_NAME: str = Field("deepseek-chat", description="Query Agent LLM model name, e.g., deepseek-reasoner")
    
    # Report Agent (recommended Gemini, recommended relay vendor: https://aihubmix.com/?aff=8Ds9)
    REPORT_ENGINE_API_KEY: Optional[str] = Field(None, description="Report Agent (recommended gemini-2.5-pro, relay vendor application address: https://aihubmix.com/?aff=8Ds9) API key")
    REPORT_ENGINE_BASE_URL: Optional[str] = Field("https://aihubmix.com/v1", description="Report Agent LLM BaseUrl, can be adjusted according to relay service")
    REPORT_ENGINE_MODEL_NAME: str = Field("gemini-2.5-pro", description="Report Agent LLM model name, e.g., gemini-2.5-pro")

    # MindSpider Agent (recommended Deepseek, official application address: https://platform.deepseek.com/)
    MINDSPIDER_API_KEY: Optional[str] = Field(None, description="MindSpider Agent (recommended deepseek, official application address: https://platform.deepseek.com/) API key")
    MINDSPIDER_BASE_URL: Optional[str] = Field(None, description="MindSpider Agent BaseUrl, can be configured according to selected service")
    MINDSPIDER_MODEL_NAME: Optional[str] = Field(None, description="MindSpider Agent model name, e.g., deepseek-reasoner")
    
    # Forum Host (Qwen3 latest model, using Silicon Flow platform here, application address: https://cloud.siliconflow.cn/)
    FORUM_HOST_API_KEY: Optional[str] = Field(None, description="Forum Host (recommended qwen-plus, official application address: https://www.aliyun.com/product/bailian) API key")
    FORUM_HOST_BASE_URL: Optional[str] = Field(None, description="Forum Host LLM BaseUrl, can be configured according to selected service")
    FORUM_HOST_MODEL_NAME: Optional[str] = Field(None, description="Forum Host LLM model name, e.g., qwen-plus")
    
    # SQL Keyword Optimizer (small parameter Qwen3 model, using Silicon Flow platform here, application address: https://cloud.siliconflow.cn/)
    KEYWORD_OPTIMIZER_API_KEY: Optional[str] = Field(None, description="SQL Keyword Optimizer (recommended qwen-plus, official application address: https://www.aliyun.com/product/bailian) API key")
    KEYWORD_OPTIMIZER_BASE_URL: Optional[str] = Field(None, description="Keyword Optimizer BaseUrl, can be configured according to selected service")
    KEYWORD_OPTIMIZER_MODEL_NAME: Optional[str] = Field(None, description="Keyword Optimizer LLM model name, e.g., qwen-plus")
    
    # ================== Network Tool Configuration ====================
    # Tavily API (application address: https://www.tavily.com/)
    TAVILY_API_KEY: Optional[str] = Field(None, description="Tavily API (application address: https://www.tavily.com/) API key, used for Tavily web search")

    SEARCH_TOOL_TYPE: Literal["AnspireAPI", "BochaAPI"] = Field("AnspireAPI", description="Network search tool type, supports BochaAPI or AnspireAPI, default is AnspireAPI")
    # Bocha API (application address: https://open.bochaai.com/)
    BOCHA_BASE_URL: Optional[str] = Field("https://api.bocha.cn/v1/ai-search", description="Bocha AI Search BaseUrl or Bocha Web Search BaseUrl")
    BOCHA_WEB_SEARCH_API_KEY: Optional[str] = Field(None, description="Bocha API (application address: https://open.bochaai.com/) API key, used for Bocha search")

    # Anspire AI Search API (application address: https://open.anspire.cn/?share_code=3E1FUOUH)
    ANSPIRE_BASE_URL: Optional[str] = Field("https://plugin.anspire.cn/api/ntsearch/search", description="Anspire AI Search BaseUrl")
    ANSPIRE_API_KEY: Optional[str] = Field(None, description="Anspire AI Search API (application address: https://open.anspire.cn/?share_code=3E1FUOUH) API key, used for Anspire search")

    
    # ================== Insight Engine Search Configuration ====================
    DEFAULT_SEARCH_HOT_CONTENT_LIMIT: int = Field(100, description="Default maximum number of hot list content")
    DEFAULT_SEARCH_TOPIC_GLOBALLY_LIMIT_PER_TABLE: int = Field(50, description="Maximum number of global topics per table")
    DEFAULT_SEARCH_TOPIC_BY_DATE_LIMIT_PER_TABLE: int = Field(100, description="Maximum number of topics by date per table")
    DEFAULT_GET_COMMENTS_FOR_TOPIC_LIMIT: int = Field(500, description="Maximum number of comments per topic")
    DEFAULT_SEARCH_TOPIC_ON_PLATFORM_LIMIT: int = Field(200, description="Maximum number of topics for platform search")
    MAX_SEARCH_RESULTS_FOR_LLM: int = Field(0, description="Maximum number of search results for LLM")
    MAX_HIGH_CONFIDENCE_SENTIMENT_RESULTS: int = Field(0, description="Maximum number of high confidence sentiment analysis results")
    MAX_REFLECTIONS: int = Field(3, description="Maximum number of reflection iterations")
    MAX_PARAGRAPHS: int = Field(6, description="Maximum number of paragraphs")
    SEARCH_TIMEOUT: int = Field(240, description="Single search request timeout")
    MAX_CONTENT_LENGTH: int = Field(500000, description="Maximum content length for search")
    
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
    
    Reload configuration from .env file and environment variables, updating the global settings instance.
    Used for dynamically updating configuration at runtime.
    
    Returns:
        Settings: Newly created configuration instance
    """
    
    global settings
    settings = Settings()
    return settings
