#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BroadTopicExtraction Module - Topic Extractor
Extract keywords and generate news summary based on DeepSeek
"""

import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from openai import OpenAI

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
    from config import settings
except ImportError:
    raise ImportError("Failed to import settings.py configuration file")

class TopicExtractor:
    """Topic extractor"""

    def __init__(self):
        """Initialize topic extractor"""
        self.client = OpenAI(
            api_key=settings.MINDSPIDER_API_KEY,
            base_url=settings.MINDSPIDER_BASE_URL
        )
        self.model = settings.MINDSPIDER_MODEL_NAME
    
    def extract_keywords_and_summary(self, news_list: List[Dict], max_keywords: int = 100) -> Tuple[List[str], str]:
        """
        Extract keywords and generate summary from news list
        
        Args:
            news_list: List of news items
            max_keywords: Maximum number of keywords
            
        Returns:
            (keywords list, news analysis summary)
        """
        if not news_list:
            return [], "No hot news today"
        
        # Build news summary text
        news_text = self._build_news_summary(news_list)
        
        # Build prompt
        prompt = self._build_analysis_prompt(news_text, max_keywords)
        
        try:
            # Call DeepSeek API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional news analyst, skilled at extracting keywords from hot news and writing analysis summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse returned result
            result_text = response.choices[0].message.content
            keywords, summary = self._parse_analysis_result(result_text)
            
            print(f"Successfully extracted {len(keywords)} keywords and generated news summary")
            return keywords[:max_keywords], summary
            
        except Exception as e:
            print(f"Topic extraction failed: {e}")
            # Return simple fallback result
            fallback_keywords = self._extract_simple_keywords(news_list)
            fallback_summary = f"Collected {len(news_list)} hot news items today, covering trending topics from multiple platforms."
            return fallback_keywords[:max_keywords], fallback_summary
    
    def _build_news_summary(self, news_list: List[Dict]) -> str:
        """Build news summary text"""
        news_items = []
        
        for i, news in enumerate(news_list, 1):
            title = news.get('title', 'Untitled')
            source = news.get('source_platform', news.get('source', 'Unknown'))
            
            # Clean special characters from title
            title = re.sub(r'[#@]', '', title).strip()
            
            news_items.append(f"{i}. [{source}] {title}")
        
        return "\n".join(news_items)
    
    def _build_analysis_prompt(self, news_text: str, max_keywords: int) -> str:
        """Build analysis prompt"""
        news_count = len(news_text.split('\n'))
        
        prompt = f"""
Please analyze the following {news_count} hot news items today and complete two tasks:

News List:
{news_text}

Task 1: Extract Keywords (max {max_keywords})
- Extract keywords that represent today's hot topics
- Keywords should be suitable for social media platform search
- Prioritize topics with high popularity and discussion volume
- Avoid overly broad or overly specific terms

Task 2: Write News Analysis Summary (150-300 words)
- Briefly summarize the main content of today's hot news
- Point out the key topic directions currently of concern to society
- Analyze the social phenomena or trends reflected by these hot topics
- Use concise, clear, and objective language

Please strictly output in the following JSON format:
```json
{{
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "summary": "Today's news analysis summary content..."
}}
```

Please output the result directly in JSON format, without any other text.
"""
        return prompt
    
    def _parse_analysis_result(self, result_text: str) -> Tuple[List[str], str]:
        """Parse analysis result"""
        try:
            # Try to extract JSON portion
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # If no code block, try direct parsing
                json_text = result_text.strip()
            
            # Parse JSON
            data = json.loads(json_text)
            
            keywords = data.get('keywords', [])
            summary = data.get('summary', '')
            
            # Validate and clean keywords
            clean_keywords = []
            for keyword in keywords:
                keyword = str(keyword).strip()
                if keyword and len(keyword) > 1 and keyword not in clean_keywords:
                    clean_keywords.append(keyword)
            
            # Validate summary
            if not summary or len(summary.strip()) < 10:
                summary = "Today's hot news covers multiple fields, reflecting the diversified concerns of current society."
            
            return clean_keywords, summary.strip()
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Raw response: {result_text}")
            
            # Try manual parsing
            return self._manual_parse_result(result_text)

        except Exception as e:
            print(f"Failed to process analysis result: {e}")
            return [], "Analysis result processing failed, please try again later."
    
    def _manual_parse_result(self, text: str) -> Tuple[List[str], str]:
        """Manually parse result (fallback when JSON parsing fails)"""
        print("Attempting manual parsing of result...")
        
        keywords = []
        summary = ""
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for keywords
            if '关键词' in line or 'keywords' in line.lower():
                # Extract keywords
                keyword_match = re.findall(r'[""](.*?)["""]', line)
                if keyword_match:
                    keywords.extend(keyword_match)
                else:
                    # Try other delimiters
                    parts = re.split(r'[,，、]', line)
                    for part in parts:
                        clean_part = re.sub(r'[关键词：:keywords\[\]"]', '', part).strip()
                        if clean_part and len(clean_part) > 1:
                            keywords.append(clean_part)
            
            # Look for summary
            elif '总结' in line or '分析' in line or 'summary' in line.lower():
                if '：' in line or ':' in line:
                    summary = line.split('：')[-1].split(':')[-1].strip()
            
            # If this line looks like summary content
            elif len(line) > 50 and ('今日' in line or '热点' in line or '新闻' in line):
                if not summary:
                    summary = line
        
        # Clean keywords
        clean_keywords = []
        for keyword in keywords:
            keyword = keyword.strip()
            if keyword and len(keyword) > 1 and keyword not in clean_keywords:
                clean_keywords.append(keyword)
        
        # If no summary found, generate a simple one
        if not summary:
            summary = "Today's hot news is rich in content, covering concerns from all aspects of society."
        
        return clean_keywords[:max_keywords], summary
    
    def _extract_simple_keywords(self, news_list: List[Dict]) -> List[str]:
        """Simple keyword extraction (fallback method)"""
        keywords = []
        
        for news in news_list:
            title = news.get('title', '')
            
            # Simple keyword extraction
            # Remove common meaningless words
            title_clean = re.sub(r'[#@【】\[\]()（）]', ' ', title)
            words = title_clean.split()
            
            for word in words:
                word = word.strip()
                if (len(word) > 1 and 
                    word not in ['的', '了', '在', '和', '与', '或', '但', '是', '有', '被', '将', '已', '正在'] and
                    word not in keywords):
                    keywords.append(word)
        
        return keywords[:10]
    
    def get_search_keywords(self, keywords: List[str], limit: int = 10) -> List[str]:
        """
        Get keywords for search
        
        Args:
            keywords: List of keywords
            limit: Limit count
            
        Returns:
            List of keywords suitable for search
        """
        # Filter and optimize keywords
        search_keywords = []
        
        for keyword in keywords:
            keyword = str(keyword).strip()
            
            # Filter conditions
            if (len(keyword) > 1 and 
                len(keyword) < 20 and  # Not too long
                keyword not in search_keywords and
                not keyword.isdigit() and  # Not pure digits
                not re.match(r'^[a-zA-Z]+$', keyword)):  # Not pure English (unless proper noun)
                
                search_keywords.append(keyword)
        
        return search_keywords[:limit]

if __name__ == "__main__":
    # Test topic extractor
    extractor = TopicExtractor()
    
    # Simulate news data
    test_news = [
        {"title": "AI technology develops rapidly", "source_platform": "Tech News"},
        {"title": "Stock market analysis", "source_platform": "Finance News"},
        {"title": "Celebrity latest updates", "source_platform": "Entertainment News"}
    ]
    
    keywords, summary = extractor.extract_keywords_and_summary(test_news)
    
    print(f"Extracted keywords: {keywords}")
    print(f"News summary: {summary}")
    
    search_keywords = extractor.get_search_keywords(keywords)
    print(f"Search keywords: {search_keywords}")
