#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSentimentCrawling Module - Main Workflow
Performs multi-platform keyword crawling based on topics extracted by BroadTopicExtraction
"""

import sys
import argparse
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from keyword_manager import KeywordManager
from platform_crawler import PlatformCrawler

class DeepSentimentCrawling:
    """Deep Sentiment Crawling Main Workflow"""

    def __init__(self):
        """Initialize deep sentiment crawling"""
        self.keyword_manager = KeywordManager()
        self.platform_crawler = PlatformCrawler()
        self.supported_platforms = ['xhs', 'dy', 'ks', 'bili', 'wb', 'tieba', 'zhihu']
    
    def run_daily_crawling(self, target_date: date = None, platforms: List[str] = None,
                          max_keywords_per_platform: int = 50,
                          max_notes_per_platform: int = 50,
                          login_type: str = "qrcode") -> Dict:
        """
        Execute daily crawling task

        Args:
            target_date: Target date, default is today
            platforms: List of platforms to crawl, default is all supported platforms
            max_keywords_per_platform: Maximum number of keywords per platform
            max_notes_per_platform: Maximum number of content items to crawl per platform
            login_type: Login type

        Returns:
            Crawling result statistics
        """
        if not target_date:
            target_date = date.today()
        
        if not platforms:
            platforms = self.supported_platforms
        
        print(f"🚀 Starting deep sentiment crawling task for {target_date}")
        print(f"Target platforms: {platforms}")

        # 1. Get keywords summary
        summary = self.keyword_manager.get_crawling_summary(target_date)
        print(f"📊 Keywords summary: {summary}")

        if not summary['has_data']:
            print("⚠️ No topic data found, cannot perform crawling")
            print("💡 Please run the following command first to get today's topic data:")
            print("   uv run main.py --broad-topic")
            return {"success": False, "error": "No topic data"}

        # 2. Get keywords (no distribution, all platforms use same keywords)
        print(f"\n📝 Getting keywords...")
        keywords = self.keyword_manager.get_latest_keywords(target_date, max_keywords_per_platform)

        if not keywords:
            print("⚠️ No keywords found, cannot perform crawling")
            return {"success": False, "error": "No keywords"}

        print(f"   Retrieved {len(keywords)} keywords")
        print(f"   Will crawl each keyword on {len(platforms)} platforms")
        print(f"   Total crawling tasks: {len(keywords)} × {len(platforms)} = {len(keywords) * len(platforms)}")

        # 3. Execute multi-platform keyword crawling
        print(f"\n🔄 Starting multi-platform keyword crawling...")
        crawl_results = self.platform_crawler.run_multi_platform_crawl_by_keywords(
            keywords, platforms, login_type, max_notes_per_platform
        )

        # 4. Generate final report
        final_report = {
            "date": target_date.isoformat(),
            "summary": summary,
            "crawl_results": crawl_results,
            "success": crawl_results["successful_tasks"] > 0
        }

        print(f"\n✅ Deep sentiment crawling task completed!")
        print(f"   Date: {target_date}")
        print(f"   Successful tasks: {crawl_results['successful_tasks']}/{crawl_results['total_tasks']}")
        print(f"   Total keywords: {crawl_results['total_keywords']}")
        print(f"   Total platforms: {crawl_results['total_platforms']}")
        print(f"   Total content: {crawl_results['total_notes']} items")

        return final_report
    
    def run_platform_crawling(self, platform: str, target_date: date = None,
                             max_keywords: int = 50, max_notes: int = 50,
                             login_type: str = "qrcode") -> Dict:
        """
        Execute crawling task for a single platform

        Args:
            platform: Platform name
            target_date: Target date
            max_keywords: Maximum number of keywords
            max_notes: Maximum number of content items to crawl
            login_type: Login type

        Returns:
            Crawling result
        """
        if platform not in self.supported_platforms:
            raise ValueError(f"Unsupported platform: {platform}")

        if not target_date:
            target_date = date.today()

        print(f"🎯 Starting crawling task for {platform} platform ({target_date})")

        # Get keywords
        keywords = self.keyword_manager.get_keywords_for_platform(
            platform, target_date, max_keywords
        )

        if not keywords:
            print(f"⚠️ No keywords found for {platform} platform")
            return {"success": False, "error": "No keywords"}

        print(f"📝 Preparing to crawl {len(keywords)} keywords")

        # Execute crawling
        result = self.platform_crawler.run_crawler(
            platform, keywords, login_type, max_notes
        )

        return result
    
    def list_available_topics(self, days: int = 7):
        """List available topics from recent days"""
        print(f"📋 Topic data from last {days} days:")

        recent_topics = self.keyword_manager.db_manager.get_recent_topics(days)

        if not recent_topics:
            print("   No topic data available")
            return

        for topic in recent_topics:
            extract_date = topic['extract_date']
            keywords_count = len(topic.get('keywords', []))
            summary_preview = topic.get('summary', '')[:100] + "..." if len(topic.get('summary', '')) > 100 else topic.get('summary', '')

            print(f"   📅 {extract_date}: {keywords_count} keywords")
            print(f"      Summary: {summary_preview}")
            print()
    
    def show_platform_guide(self):
        """Show platform usage guide"""
        print("🔧 Platform Crawling Guide:")
        print()

        platform_info = {
            'xhs': 'Xiaohongshu - Beauty, lifestyle, fashion content',
            'dy': 'Douyin - Short videos, entertainment, lifestyle content',
            'ks': 'Kuaishou - Lifestyle, entertainment, rural-themed content',
            'bili': 'Bilibili - Tech, learning, gaming, anime content',
            'wb': 'Weibo - Hot news, celebrities, social topics',
            'tieba': 'Baidu Tieba - Interest discussions, gaming, learning',
            'zhihu': 'Zhihu - Knowledge Q&A, in-depth discussions'
        }

        for platform, desc in platform_info.items():
            print(f"   {platform}: {desc}")

        print()
        print("💡 Usage Tips:")
        print("   1. First-time use requires QR code login for each platform")
        print("   2. Recommend testing single platform first to verify login works")
        print("   3. Keep crawling volume moderate to avoid being rate-limited")
        print("   4. Use --test mode for small-scale testing")
    
    def close(self):
        """Close resources"""
        if self.keyword_manager:
            self.keyword_manager.close()

def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description="DeepSentimentCrawling - Topic-based deep sentiment crawling")

    # Basic parameters
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD), default is today")
    parser.add_argument("--platform", type=str, choices=['xhs', 'dy', 'ks', 'bili', 'wb', 'tieba', 'zhihu'],
                       help="Specify single platform to crawl")
    parser.add_argument("--platforms", type=str, nargs='+',
                       choices=['xhs', 'dy', 'ks', 'bili', 'wb', 'tieba', 'zhihu'],
                       help="Specify multiple platforms to crawl")

    # Crawling parameters
    parser.add_argument("--max-keywords", type=int, default=50,
                       help="Maximum keywords per platform (default: 50)")
    parser.add_argument("--max-notes", type=int, default=50,
                       help="Maximum content items to crawl per platform (default: 50)")
    parser.add_argument("--login-type", type=str, choices=['qrcode', 'phone', 'cookie'],
                       default='qrcode', help="Login type (default: qrcode)")

    # Feature parameters
    parser.add_argument("--list-topics", action="store_true", help="List recent topic data")
    parser.add_argument("--days", type=int, default=7, help="View topics from last N days (default: 7)")
    parser.add_argument("--guide", action="store_true", help="Show platform usage guide")
    parser.add_argument("--test", action="store_true", help="Test mode (small data volume)")

    args = parser.parse_args()

    # Parse date
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print("❌ Invalid date format, please use YYYY-MM-DD format")
            return

    # Create crawler instance
    crawler = DeepSentimentCrawling()

    try:
        # Show guide
        if args.guide:
            crawler.show_platform_guide()
            return

        # List topics
        if args.list_topics:
            crawler.list_available_topics(args.days)
            return

        # Test mode - adjust parameters
        if args.test:
            args.max_keywords = min(args.max_keywords, 10)
            args.max_notes = min(args.max_notes, 10)
            print("Test mode: limiting keywords and content count")

        # Single platform crawling
        if args.platform:
            result = crawler.run_platform_crawling(
                args.platform, target_date, args.max_keywords,
                args.max_notes, args.login_type
            )

            if result['success']:
                print(f"\n{args.platform} crawling successful!")
            else:
                print(f"\n{args.platform} crawling failed: {result.get('error', 'Unknown error')}")

            return

        # Multi-platform crawling
        platforms = args.platforms if args.platforms else None
        result = crawler.run_daily_crawling(
            target_date, platforms, args.max_keywords,
            args.max_notes, args.login_type
        )

        if result['success']:
            print(f"\nMulti-platform crawling task completed!")
        else:
            print(f"\nMulti-platform crawling failed: {result.get('error', 'Unknown error')}")

    except KeyboardInterrupt:
        print("\nUser interrupted operation")
    except Exception as e:
        print(f"\nExecution error: {e}")
    finally:
        crawler.close()

if __name__ == "__main__":
    main()
