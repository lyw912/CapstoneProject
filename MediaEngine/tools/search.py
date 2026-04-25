"""
Multimodal search toolkit designed for AI Agents (Bocha)

Version: 1.1
Last Updated: 2025-08-22

This script decomposes complex Bocha AI Search functionality into a series of independent tools 
with clear goals and minimal parameters, designed for AI Agent invocation. Agents simply select 
the appropriate tool based on task intent (e.g., regular search, structured data lookup, or timely news) 
without needing to understand complex parameter combinations.

Key Features:
- Powerful multimodal capabilities: Returns web pages, images, AI summaries, follow-up suggestions, 
  and rich "modal card" structured data simultaneously.
- Modal card support: For specific queries like weather, stocks, exchange rates, encyclopedia, 
  medical info, etc., returns structured data cards directly for easy Agent parsing and usage.

Main Tools:
- comprehensive_search: Perform comprehensive search, returns web pages, images, AI summaries and possible modal cards.
- search_for_structured_data: Specifically for querying structured information like weather, stocks, exchange rates that can trigger "modal cards".
- web_search_only: Perform pure web search without requesting AI summary, faster speed.
- search_last_24_hours: Get latest information from past 24 hours.
- search_last_week: Get major reports from the past week.
"""

import os
import json
import sys
import datetime
from typing import List, Dict, Any, Optional, Literal

from loguru import logger
from config import settings

# Please ensure requests library is installed before running: pip install requests
try:
    import requests
except ImportError:
    raise ImportError("requests library not installed, please run `pip install requests` to install.")

# Add utils directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
utils_dir = os.path.join(root_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

from retry_helper import with_graceful_retry, SEARCH_API_RETRY_CONFIG

# --- 1. Core Client and Tool Set ---
from dataclasses import dataclass, field

@dataclass
class WebpageResult:
    """Webpage search result"""
    name: str
    url: str
    snippet: str
    display_url: Optional[str] = None
    date_last_crawled: Optional[str] = None

@dataclass
class ImageResult:
    """Image search result"""
    name: str
    content_url: str
    host_page_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None

@dataclass
class ModalCardResult:
    """
    Modal card structured data result
    This is a core feature of Bocha search, used to return specific types of structured information.
    """
    card_type: str  # e.g., weather_china, stock, baike_pro, medical_common
    content: Dict[str, Any]  # Parsed JSON content

@dataclass
class BochaResponse:
    """Encapsulates complete Bocha API return result for passing between tools"""
    query: str
    conversation_id: Optional[str] = None
    answer: Optional[str] = None  # AI-generated summary answer
    follow_ups: List[str] = field(default_factory=list) # AI-generated follow-up questions
    webpages: List[WebpageResult] = field(default_factory=list)
    images: List[ImageResult] = field(default_factory=list)
    modal_cards: List[ModalCardResult] = field(default_factory=list)

@dataclass
class AnspireResponse:
    """Encapsulates complete Anspire API return result for passing between tools"""
    query: str
    conversation_id: Optional[str] = None
    score: Optional[float] = None
    webpages: List[WebpageResult] = field(default_factory=list)


# --- 2. Core Client and Tool Set ---

class BochaMultimodalSearch:
    """
    A client containing various dedicated multimodal search tools.
    Each public method is designed as a tool for independent AI Agent calls.
    """

    BOCHA_BASE_URL = settings.BOCHA_BASE_URL or "https://api.bocha.cn/v1/ai-search"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize client.
        Args:
            api_key: Bocha API key, if not provided, reads from BOCHA_API_KEY environment variable.
        """
        if api_key is None:
            api_key = settings.BOCHA_WEB_SEARCH_API_KEY
            if not api_key:
                raise ValueError("Bocha API Key not found! Please set BOCHA_API_KEY environment variable or provide during initialization")

        self._headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': '*/*'
        }

    def _parse_search_response(self, response_dict: Dict[str, Any], query: str) -> BochaResponse:
        """Parse structured BochaResponse object from API raw dictionary response"""

        final_response = BochaResponse(query=query)
        final_response.conversation_id = response_dict.get('conversation_id')

        messages = response_dict.get('messages', [])
        for msg in messages:
            role = msg.get('role')
            if role != 'assistant':
                continue

            msg_type = msg.get('type')
            content_type = msg.get('content_type')
            content_str = msg.get('content', '{}')

            try:
                content_data = json.loads(content_str)
            except json.JSONDecodeError:
                # If content is not valid JSON string (e.g., plain text answer), use directly
                content_data = content_str

            if msg_type == 'answer' and content_type == 'text':
                final_response.answer = content_data

            elif msg_type == 'follow_up' and content_type == 'text':
                final_response.follow_ups.append(content_data)

            elif msg_type == 'source':
                if content_type == 'webpage':
                    web_results = content_data.get('value', [])
                    for item in web_results:
                        final_response.webpages.append(WebpageResult(
                            name=item.get('name'),
                            url=item.get('url'),
                            snippet=item.get('snippet'),
                            display_url=item.get('displayUrl'),
                            date_last_crawled=item.get('dateLastCrawled')
                        ))
                elif content_type == 'image':
                    final_response.images.append(ImageResult(
                        name=content_data.get('name'),
                        content_url=content_data.get('contentUrl'),
                        host_page_url=content_data.get('hostPageUrl'),
                        thumbnail_url=content_data.get('thumbnailUrl'),
                        width=content_data.get('width'),
                        height=content_data.get('height')
                    ))
                # All other content_types are treated as modal cards
                else:
                    final_response.modal_cards.append(ModalCardResult(
                        card_type=content_type,
                        content=content_data
                    ))

        return final_response


    @with_graceful_retry(SEARCH_API_RETRY_CONFIG, default_return=BochaResponse(query="Search failed"))
    def _search_internal(self, **kwargs) -> BochaResponse:
        """Internal generic search executor, all tools ultimately call this method"""
        query = kwargs.get("query", "Unknown Query")
        payload = {
            "stream": False,  # Agent tools usually use non-streaming to get complete results
        }
        payload.update(kwargs)

        try:

            response = requests.post(self.BOCHA_BASE_URL, headers=self._headers, json=payload, timeout=30)
            response.raise_for_status()  # Raise exception if HTTP status is 4xx or 5xx

            response_dict = response.json()
            if response_dict.get("code") != 200:
                logger.error(f"API returned error: {response_dict.get('msg', 'Unknown error')}")
                return BochaResponse(query=query)

            return self._parse_search_response(response_dict, query)

        except requests.exceptions.RequestException as e:
            logger.exception(f"Network error occurred during search: {str(e)}")
            raise e  # Let retry mechanism catch and handle
        except Exception as e:
            logger.exception(f"Unknown error occurred while processing response: {str(e)}")
            raise e  # Let retry mechanism catch and handle

    # --- Agent Available Tool Methods ---

    def comprehensive_search(self, query: str, max_results: int = 10) -> BochaResponse:
        """
        [TOOL] Comprehensive Search: Execute a standard comprehensive search covering all information types.
        Returns webpages, images, AI summary, follow-up suggestions, and possible modal cards. This is the most commonly used general search tool.
        Agent provides search query and optional max_results.
        """
        logger.info(f"--- TOOL: Comprehensive Search (query: {query}) ---")
        return self._search_internal(
            query=query,
            count=max_results,
            answer=True  # Enable AI summary
        )

    def web_search_only(self, query: str, max_results: int = 15) -> BochaResponse:
        """
        [TOOL] Web Search Only: Only get webpage links and snippets, without requesting AI-generated answers.
        Suitable for scenarios requiring quick access to raw webpage information without additional AI analysis. Faster speed, lower cost.
        """
        logger.info(f"--- TOOL: Web Search Only (query: {query}) ---")
        return self._search_internal(
            query=query,
            count=max_results,
            answer=False # Disable AI summary
        )

    def search_for_structured_data(self, query: str) -> BochaResponse:
        """
        【工具】结构化数据查询: 专门用于可能触发“模态卡”的查询。
        当Agent意图是查询天气、股票、汇率、百科定义、火车票、汽车参数等结构化信息时，应优先使用此工具。
        它会返回所有信息，但Agent应重点关注结果中的 `modal_cards` 部分。
        """
        logger.info(f"--- TOOL: Structured Data Query (query: {query}) ---")
        # Implementation is same as comprehensive_search, but naming and documentation guide Agent intent
        return self._search_internal(
            query=query,
            count=5, # Structured queries usually don't need too many webpage results
            answer=True
        )

    def search_last_24_hours(self, query: str) -> BochaResponse:
        """
        [TOOL] Search Last 24 Hours: Get latest updates on a topic.
        This tool specifically searches content published in the past 24 hours. Suitable for tracking breaking events or latest developments.
        """
        logger.info(f"--- TOOL: Search Last 24 Hours (query: {query}) ---")
        return self._search_internal(query=query, freshness='oneDay', answer=True)

    def search_last_week(self, query: str) -> BochaResponse:
        """
        [TOOL] Search Last Week: Get major reports on a topic from the past week.
        Suitable for weekly sentiment summaries or reviews.
        """
        logger.info(f"--- TOOL: Search Last Week (query: {query}) ---")
        return self._search_internal(query=query, freshness='oneWeek', answer=True)

class AnspireAISearch:
    """
    Anspire AI Search client
    """
    ANSPIRE_BASE_URL = settings.ANSPIRE_BASE_URL or "https://plugin.anspire.cn/api/ntsearch/search"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize client.
        Args:
            api_key: Anspire API key, if not provided, reads from ANSPIRE_API_KEY environment variable.
        """
        if api_key is None:
            api_key = settings.ANSPIRE_API_KEY
            if not api_key:
                raise ValueError("Anspire API Key not found! Please set ANSPIRE_API_KEY environment variable or provide during initialization")

        self._headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Connection': 'keep-alive',
            'Accept': '*/*'
        }

    def _parse_search_response(self, response_dict: Dict[str, Any], query: str) -> AnspireResponse:
        final_response = AnspireResponse(query=query)
        final_response.conversation_id = response_dict.get('Uuid')

        messages = response_dict.get("results", [])
        for msg in messages:
            final_response.score = msg.get("score")
            final_response.webpages.append(WebpageResult(
                name = msg.get("title", ""),
                url = msg.get("url", ""),
                snippet = msg.get("content", ""),
                date_last_crawled = msg.get("date", None)
            ))

        return final_response
    
    @with_graceful_retry(SEARCH_API_RETRY_CONFIG, default_return=AnspireResponse(query="Search failed"))
    def _search_internal(self, **kwargs) -> AnspireResponse:
        """Internal generic search executor, all tools ultimately call this method"""
        query = kwargs.get("query", "Unknown Query")
        payload = {
            "query": query,
            "top_k": kwargs.get("top_k", 10),
            "Insite": kwargs.get("Insite", ""),
            "FromTime": kwargs.get("FromTime", ""),
            "ToTime": kwargs.get("ToTime", "")
        }

        try:
            response = requests.get(self.ANSPIRE_BASE_URL, headers=self._headers, params=payload, timeout=30)
            response.raise_for_status()  # Raise exception if HTTP status is 4xx or 5xx

            response_dict = response.json()
            return self._parse_search_response(response_dict, query)
        except requests.exceptions.RequestException as e:
            logger.exception(f"Network error occurred during search: {str(e)}")
            raise e  # Let retry mechanism catch and handle
        except Exception as e:
            logger.exception(f"Unknown error occurred while processing response: {str(e)}")
            raise e  # Let retry mechanism catch and handle
    
    def comprehensive_search(self, query: str, max_results: int = 10) -> AnspireResponse:
        """
        [TOOL] Comprehensive Search: Get comprehensive information about a topic, including webpages.
        Suitable for scenarios requiring multiple information sources.
        """
        logger.info(f"--- TOOL: Comprehensive Search (query: {query}) ---")
        return self._search_internal(
            query=query,
            top_k=max_results
        )

    def search_last_24_hours(self, query: str, max_results: int = 10) -> AnspireResponse:
        """
        [TOOL] Search Last 24 Hours: Get latest updates on a topic.
        This tool specifically searches content published in the past 24 hours. 
        Suitable for tracking breaking events or latest developments.
        """
        logger.info(f"--- TOOL: Search Last 24 Hours (query: {query}) ---")
        to_time = datetime.datetime.now()
        from_time = to_time - datetime.timedelta(days=1)
        return self._search_internal(query=query,
                                     top_k=max_results,
                                     FromTime=from_time.strftime("%Y-%m-%d %H:%M:%S"), 
                                     ToTime=to_time.strftime("%Y-%m-%d %H:%M:%S"))

    def search_last_week(self, query: str, max_results: int = 10) -> AnspireResponse:
        """
        [TOOL] Search Last Week: Get major reports on a topic from the past week.
        Suitable for weekly sentiment summaries or reviews.
        """
        logger.info(f"--- TOOL: Search Last Week (query: {query}) ---")
        to_time = datetime.datetime.now()
        from_time = to_time - datetime.timedelta(weeks=1)
        return self._search_internal(query=query,
                                     top_k=max_results,
                                     FromTime=from_time.strftime("%Y-%m-%d %H:%M:%S"),
                                     ToTime=to_time.strftime("%Y-%m-%d %H:%M:%S"))


# --- 3. Test and Usage Examples ---
def load_agent_from_config():
    """Select and load search Agent based on configuration file"""
    if settings.BOCHA_WEB_SEARCH_API_KEY:
        logger.info("Loading BochaMultimodalSearch Agent")
        return BochaMultimodalSearch()
    elif settings.ANSPIRE_API_KEY:
        logger.info("Loading AnspireAISearch Agent")
        return AnspireAISearch()
    else:
        raise ValueError("No valid search Agent configured")

def print_response_summary(response):
    """Simplified print function for displaying test results"""
    if not response or not response.query:
        logger.error("Failed to get valid response.")
        return

    logger.info(f"\nQuery: '{response.query}' | Session ID: {response.conversation_id}")
    if hasattr(response, 'answer') and response.answer:
        logger.info(f"AI Summary: {response.answer[:150]}...")

    logger.info(f"Found {len(response.webpages)} webpages")
    if hasattr(response, 'images'):
        logger.info(f"Found {len(response.images)} images")
    if hasattr(response, 'modal_cards'):
        logger.info(f"Found {len(response.modal_cards)} modal cards")

    if hasattr(response, 'modal_cards') and response.modal_cards:
        first_card = response.modal_cards[0]
        logger.info(f"First modal card type: {first_card.card_type}")

    if response.webpages:
        first_result = response.webpages[0]
        logger.info(f"First webpage result: {first_result.name}")

    if hasattr(response, 'follow_ups') and response.follow_ups:
        logger.info(f"Suggested follow-ups: {response.follow_ups}")

    logger.info("-" * 60)


if __name__ == "__main__":
    # Before running, ensure you have set the BOCHA_API_KEY environment variable

    try:
        # Initialize multimodal search client, which contains all tools internally
        search_client = load_agent_from_config()

        # Scenario 1: Agent performs a regular comprehensive search requiring AI summary
        response1 = search_client.comprehensive_search(query="人工智能对未来教育的影响")
        print_response_summary(response1)

        # Scenario 2: Agent needs to query specific structured information - weather
        if isinstance(search_client, BochaMultimodalSearch):
            response2 = search_client.search_for_structured_data(query="上海明天天气怎么样")
            print_response_summary(response2)
            # Deep parse first modal card
            if response2.modal_cards and response2.modal_cards[0].card_type == 'weather_china':
                logger.info("Weather modal card details:", json.dumps(response2.modal_cards[0].content, indent=2, ensure_ascii=False))


        # Scenario 3: Agent needs to query specific structured information - stocks
        if isinstance(search_client, BochaMultimodalSearch):
            response3 = search_client.search_for_structured_data(query="East Money stock")
            print_response_summary(response3)

        # Scenario 4: Agent needs to track latest developments of an event
        response4 = search_client.search_last_24_hours(query="C929 aircraft latest news")
        print_response_summary(response4)

        # Scenario 5: Agent only needs to quickly get webpage information without AI summary
        if isinstance(search_client, BochaMultimodalSearch):
            response5 = search_client.web_search_only(query="Python dataclasses usage")
            print_response_summary(response5)

        # Scenario 6: Agent needs to review news about a technology from the past week
        response6 = search_client.search_last_week(query="Quantum computing commercialization")
        print_response_summary(response6)

        '''Below is the test program output:
        --- TOOL: Comprehensive Search (query: Impact of AI on future education) ---

Query: 'Impact of AI on future education' | Session ID: bf43bfe4c7bb4f7b8a3945515d8ab69e
AI Summary: AI has multiple impacts on future education.

From a positive perspective:
- In terms of teaching resources, AI helps balance resource allocation[cite:4]. For example, through AI cloud platforms, quality resources can be shared, which is significant for remote areas, allowing students there to access quality educational content and alleviating teacher shortages to some extent, because AI-driven intelligent teaching assistants or virtual...
Found 10 webpages, 1 image, 1 modal card.
First modal card type: video
First webpage result: How AI Affects Education Transformation
Suggested follow-ups: [['How will AI change future education models?', 'What challenges will AI bring to teachers in future education?', 'How can students use AI to improve learning outcomes in future education?']]
------------------------------------------------------------
--- TOOL: Structured Data Query (query: Shanghai weather tomorrow) ---

Query: 'Shanghai weather tomorrow' | Session ID: e412aa1548cd43a295430e47a62adda2
AI Summary: Based on the given information, unable to determine Shanghai's weather for tomorrow.

First, all provided information is about weather conditions on August 22, 2025, including temperature, precipitation, wind, humidity, and heat warnings[cite:1][cite:2][cite:3][cite:5]. However, this information does not involve predictions for tomorrow (August 23). Although it mentions that subtropical high pressure will maintain high temperatures through the end of August...
Found 5 webpages, 1 image, 2 modal cards.
First modal card type: video
First webpage result: Today hitting 38! Shanghai August high temperature days and consecutive summer high temperature days likely to break records_weather_low pressure_meteorological station
Suggested follow-ups: [['Can you tell me the temperature range for Shanghai tomorrow?', 'Will it rain in Shanghai tomorrow?', 'Will the weather in Shanghai tomorrow be sunny or cloudy?']]
------------------------------------------------------------
--- TOOL: Structured Data Query (query: East Money stock) ---

Query: 'East Money stock' | Session ID: 584d62ed97834473b967127852e1eaa0
AI Summary: Based solely on the provided context, unable to obtain specific information about East Money stock.

From the given data, there is no specific data directly related to East Money stock. For example, no rise/fall status, trading volume, market cap, or other specific data[cite:1][cite:3]. Nor is there information about East Money stock in research reports or ratings[cite:2]. Meanwhile, the context regarding stock prices, trading...
Found 5 webpages, 1 image, 2 modal cards.
First modal card type: video
First webpage result: Stock prices_Real-time trading_Market trends-East Money Network
Suggested follow-ups: [['What is the recent trend of East Money stock?', 'What are the main investment highlights of East Money stock?', 'What are the historical highest and lowest prices of East Money stock?']]
------------------------------------------------------------
--- TOOL: 搜索24小时内信息 (query: C929大飞机最新消息) ---

查询: 'C929大飞机最新消息' | 会话ID: 5904021dc29d497e938e04db18d7f2e2
AI摘要: 根据提供的上下文，没有关于C929大飞机的直接消息，无法确切给出C929大飞机的最新消息。

目前提供的上下文涵盖了众多航空领域相关事件，但多是围绕波音787、空客A380相关专家的人事变动、国产飞机“C909云端之旅”、科德数控的营收情况、俄制航空发动机供应相关以及其他非C929大飞机相关的内容。...
找到 10 个网页, 1 张图片, 1 个模态卡。
第一个模态卡类型: video
第一条网页结果: 放弃美国千万年薪,波音787顶尖专家回国,或可协助破解C929
建议追问: [['C929大飞机目前的研发进度如何？', '有没有关于C929大飞机预计首飞时间的消息？', 'C929大飞机在技术创新方面有哪些新进展？']]
------------------------------------------------------------
--- TOOL: Web Search Only (query: Python dataclasses usage) ---

Query: 'Python dataclasses usage' | Session ID: 74c742759d2e4b17b52d8b735ce24537
Found 15 webpages, 1 image, 1 modal card.
First modal card type: video
First webpage result: Must-know dataclasses Python tips_python dataclasses-CSDN Blog
------------------------------------------------------------
--- TOOL: 搜索本周信息 (query: 量子计算商业化) ---

AI摘要: 量子计算商业化正在逐步推进。

量子计算商业化有着多方面的体现和推动因素。从国际上看，美国能源部橡树岭国家实验室选择IQM Radiance作为其首台本地部署的量子计算机，计划于2025年第三季度交付并集成至高性能计算系统中[引用:4]；英国量子计算公司Oxford Ionics的全栈离子阱量子计算...
找到 10 个网页, 1 张图片, 1 个模态卡。
第一个模态卡类型: video
第一条网页结果: 量子计算商业潜力释放正酣,微美全息(WIMI.US)创新科技卡位“生态高地”
建议追问: [['量子计算商业化目前有哪些成功的案例？', '哪些公司在推动量子计算商业化进程？', '量子计算商业化面临的主要挑战是什么？']]
------------------------------------------------------------'''

    except ValueError as e:
        logger.exception(f"Initialization failed: {e}")
        logger.error("Please ensure BOCHA_API_KEY environment variable is properly set.")
    except Exception as e:
        logger.exception(f"Unknown error occurred during testing: {e}")