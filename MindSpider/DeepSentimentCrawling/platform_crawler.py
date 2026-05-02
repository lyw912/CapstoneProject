#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSentimentCrawling Module - Platform Crawler Manager
Responsible for configuring and invoking MediaCrawler for multi-platform crawling
"""

import os
import re
import sys
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import config
except ImportError:
    raise ImportError("Failed to import config.py configuration file")

class PlatformCrawler:
    """Platform Crawler Manager"""

    def __init__(self):
        """Initialize platform crawler manager"""
        self.mediacrawler_path = Path(__file__).parent / "MediaCrawler"
        self.supported_platforms = ['xhs', 'dy', 'ks', 'bili', 'wb', 'tieba', 'zhihu']
        self.crawl_stats = {}

        # Ensure MediaCrawler submodule is initialized
        db_config_path = self.mediacrawler_path / "config" / "db_config.py"
        if not self.mediacrawler_path.exists() or not db_config_path.exists():
            logger.error("MediaCrawler submodule not initialized or incomplete")
            logger.error("Please run the following command in project root to initialize submodule:")
            logger.error("   git submodule update --init --recursive")
            raise FileNotFoundError("MediaCrawler submodule not initialized, please run: git submodule update --init --recursive")

        logger.info(f"Initializing platform crawler manager, MediaCrawler path: {self.mediacrawler_path}")
    
    def configure_mediacrawler_db(self):
        """Configure MediaCrawler to use our database (MySQL or PostgreSQL)"""
        try:
            # Determine database type
            db_dialect = (config.settings.DB_DIALECT or "mysql").lower()
            is_postgresql = db_dialect in ("postgresql", "postgres")

            # Modify MediaCrawler database configuration
            db_config_path = self.mediacrawler_path / "config" / "db_config.py"
            
            # Read original configuration
            with open(db_config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # PostgreSQL configuration values: use MindSpider config if PostgreSQL, otherwise use defaults or environment variables
            pg_password = config.settings.DB_PASSWORD if is_postgresql else "default_password"
            pg_user = config.settings.DB_USER if is_postgresql else "default_user"
            pg_host = config.settings.DB_HOST if is_postgresql else "127.0.0.1"
            pg_port = config.settings.DB_PORT if is_postgresql else 5444
            pg_db_name = config.settings.DB_NAME if is_postgresql else "default_db"

            # Replace database configuration - use MindSpider database configuration
            new_config = f'''# Declaration: This code is for educational and research purposes only. Users should abide by the following principles:
# 1. Not for any commercial use.
# 2. Use should comply with target platform's terms of service and robots.txt rules.
# 3. No large-scale crawling or operational interference with platforms.
# 4. Request frequency should be controlled reasonably to avoid unnecessary burden on target platforms.
# 5. Not for any illegal or improper purposes.
#
# For detailed license terms, please refer to the LICENSE file in the project root directory.
# Using this code indicates your agreement to abide by the above principles and all terms in the LICENSE.


import os

# mysql config - Use MindSpider database configuration
MYSQL_DB_PWD = "{config.settings.DB_PASSWORD}"
MYSQL_DB_USER = "{config.settings.DB_USER}"
MYSQL_DB_HOST = "{config.settings.DB_HOST}"
MYSQL_DB_PORT = {config.settings.DB_PORT}
MYSQL_DB_NAME = "{config.settings.DB_NAME}"

mysql_db_config = {{
    "user": MYSQL_DB_USER,
    "password": MYSQL_DB_PWD,
    "host": MYSQL_DB_HOST,
    "port": MYSQL_DB_PORT,
    "db_name": MYSQL_DB_NAME,
}}


# redis config
REDIS_DB_HOST = "127.0.0.1"  # your redis host
REDIS_DB_PWD = os.getenv("REDIS_DB_PWD", "123456")  # your redis password
REDIS_DB_PORT = os.getenv("REDIS_DB_PORT", 6379)  # your redis port
REDIS_DB_NUM = os.getenv("REDIS_DB_NUM", 0)  # your redis database number

# cache type
CACHE_TYPE_REDIS = "redis"
CACHE_TYPE_MEMORY = "memory"

# sqlite config
SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "sqlite_tables.db")

sqlite_db_config = {{
    "db_path": SQLITE_DB_PATH
}}

# mongodb config
MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
MONGODB_PORT = os.getenv("MONGODB_PORT", 27017)
MONGODB_USER = os.getenv("MONGODB_USER", "")
MONGODB_PWD = os.getenv("MONGODB_PWD", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "media_crawler")

mongodb_config = {{
    "host": MONGODB_HOST,
    "port": int(MONGODB_PORT),
    "user": MONGODB_USER,
    "password": MONGODB_PWD,
    "db_name": MONGODB_DB_NAME,
}}

# postgres config - Use MindSpider database configuration (if DB_DIALECT is postgresql) or environment variables
POSTGRES_DB_PWD = os.getenv("POSTGRES_DB_PWD", "{pg_password}")
POSTGRES_DB_USER = os.getenv("POSTGRES_DB_USER", "{pg_user}")
POSTGRES_DB_HOST = os.getenv("POSTGRES_DB_HOST", "{pg_host}")
POSTGRES_DB_PORT = os.getenv("POSTGRES_DB_PORT", "{pg_port}")
POSTGRES_DB_NAME = os.getenv("POSTGRES_DB_NAME", "{pg_db_name}")

postgres_db_config = {{
    "user": POSTGRES_DB_USER,
    "password": POSTGRES_DB_PWD,
    "host": POSTGRES_DB_HOST,
    "port": POSTGRES_DB_PORT,
    "db_name": POSTGRES_DB_NAME,
}}

'''
            
            # Write new configuration
            with open(db_config_path, 'w', encoding='utf-8') as f:
                f.write(new_config)

            db_type = "PostgreSQL" if is_postgresql else "MySQL"
            logger.info(f"Configured MediaCrawler to use MindSpider {db_type} database")
            return True

        except Exception as e:
            logger.exception(f"Failed to configure MediaCrawler database: {e}")
            return False
    
    # ── cookie helpers ────────────────────────────────────────────────────────

    _COOKIE_FILE_PLATFORM_MAP = {
        "xiaohongshu": "xhs",
        "douyin": "dy",
        "bilibili": "bili",
        "weibo": "wb",
        "tieba": "tieba",
        "zhihu": "zhihu",
    }

    def _load_cookies_from_file(self) -> dict:
        """Load per-platform cookies from <project_root>/cookie.txt."""
        cookie_file = project_root.parent / "cookie.txt"
        if not cookie_file.exists():
            return {}
        cookies = {}
        for line in cookie_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            code = self._COOKIE_FILE_PLATFORM_MAP.get(name)
            if code:
                cookies[code] = value.strip()
        return cookies

    def create_base_config(self, platform: str, keywords: List[str],
                          crawler_type: str = "search", max_notes: int = 50) -> bool:
        """
        Create MediaCrawler base configuration

        Args:
            platform: Platform name
            keywords: List of keywords
            crawler_type: Crawler type
            max_notes: Maximum number of notes to crawl

        Returns:
            Whether configuration was successful
        """
        try:
            # Determine database type to set SAVE_DATA_OPTION
            db_dialect = (config.settings.DB_DIALECT or "mysql").lower()
            is_postgresql = db_dialect in ("postgresql", "postgres")
            save_data_option = "postgres" if is_postgresql else "db"

            base_config_path = self.mediacrawler_path / "config" / "base_config.py"

            # Load per-platform cookie from cookie.txt
            platform_cookies = self._load_cookies_from_file()
            platform_cookie = platform_cookies.get(platform, "")
            if platform_cookie:
                logger.info(f"Loaded cookie for {platform} from cookie.txt ({len(platform_cookie)} chars)")
            else:
                logger.warning(f"No cookie found for {platform} in cookie.txt — will use existing config value")

            # Convert keywords list to comma-separated string
            keywords_str = ",".join(keywords)

            # Read original configuration file
            with open(base_config_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Modify key configuration items
            lines = content.split('\n')
            new_lines = []
            skip_until_paren = False

            for line in lines:
                if skip_until_paren:
                    if line.strip() == ')':
                        skip_until_paren = False
                    continue

                replaced = None
                if line.startswith('PLATFORM = '):
                    replaced = f'PLATFORM = "{platform}"  # Platform: xhs | dy | ks | bili | wb | tieba | zhihu'
                elif line.startswith('KEYWORDS = '):
                    replaced = f'KEYWORDS = "{keywords_str}"  # Keyword search configuration, comma-separated'
                elif line.startswith('CRAWLER_TYPE = '):
                    replaced = f'CRAWLER_TYPE = "{crawler_type}"  # Crawler type: search (keyword search) | detail (post details) | creator (creator homepage data)'
                elif line.startswith('SAVE_DATA_OPTION = '):
                    replaced = f'SAVE_DATA_OPTION = "{save_data_option}"  # csv or db or json or sqlite or postgres'
                elif line.startswith('CRAWLER_MAX_NOTES_COUNT = '):
                    replaced = f'CRAWLER_MAX_NOTES_COUNT = {max_notes}'
                elif line.startswith('ENABLE_GET_COMMENTS = '):
                    replaced = 'ENABLE_GET_COMMENTS = True'
                elif line.startswith('CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = '):
                    replaced = 'CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = 20'
                elif line.startswith('HEADLESS = '):
                    replaced = 'HEADLESS = True'
                elif line.startswith('SAVE_LOGIN_STATE = '):
                    replaced = 'SAVE_LOGIN_STATE = False'
                elif line.startswith('LOGIN_TYPE = '):
                    replaced = 'LOGIN_TYPE = "cookie"  # qrcode or phone or cookie'
                elif line.startswith('COOKIES = ') and platform_cookie:
                    safe = platform_cookie.replace("\\", "\\\\").replace('"', '\\"')
                    replaced = f'COOKIES = "{safe}"'
                elif line.startswith('ENABLE_CDP_MODE = '):
                    replaced = 'ENABLE_CDP_MODE = False'

                if replaced is not None:
                    new_lines.append(replaced)
                    if line.rstrip().endswith('('):
                        skip_until_paren = True
                else:
                    new_lines.append(line)

            # Write new configuration
            with open(base_config_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))

            logger.info(f"Configured {platform} platform, crawler type: {crawler_type}, keywords count: {len(keywords)}, max notes: {max_notes}, save data option: {save_data_option}")
            return True

        except Exception as e:
            logger.exception(f"Failed to create base configuration: {e}")
            return False
    
    def run_crawler(self, platform: str, keywords: List[str],
                   login_type: str = "qrcode", max_notes: int = 50) -> Dict:
        """
        Run crawler

        Args:
            platform: Platform name
            keywords: List of keywords
            login_type: Login type
            max_notes: Maximum number of notes to crawl

        Returns:
            Crawling result statistics
        """
        if platform not in self.supported_platforms:
            raise ValueError(f"Unsupported platform: {platform}")

        if not keywords:
            raise ValueError("Keywords list cannot be empty")

        start_message = f"\nStarting crawling for platform: {platform}"
        start_message += f"\nKeywords: {keywords[:5]}{'...' if len(keywords) > 5 else ''} (total {len(keywords)})"
        logger.info(start_message)

        start_time = datetime.now()

        # Bilibili uses dedicated API path to bypass Playwright 412 rate limiting
        if platform == "bili":
            return self._run_bili_api_crawler(keywords, max_notes, start_time)

        # Zhihu: refresh z_c0 cookie before crawling (expires within hours)
        if platform == "zhihu":
            try:
                from DeepSentimentCrawling.zhihu_cookie_refresher import refresh_sync
                logger.info("[Zhihu] Attempting to refresh z_c0 cookie...")
                ok = refresh_sync(timeout=30)
                if ok:
                    logger.info("[Zhihu] z_c0 cookie refreshed successfully")
                else:
                    logger.warning("[Zhihu] z_c0 refresh failed, continuing with existing cookie (may cause login failure)")
            except Exception as e:
                logger.warning(f"[Zhihu] z_c0 refresh error: {e}, continuing with existing cookie")

        try:
            # Configure database
            if not self.configure_mediacrawler_db():
                return {"success": False, "error": "Database configuration failed"}

            # Create base configuration
            if not self.create_base_config(platform, keywords, "search", max_notes):
                return {"success": False, "error": "Base configuration creation failed"}

            # Determine database type for save_data_option
            db_dialect = (config.settings.DB_DIALECT or "mysql").lower()
            is_postgresql = db_dialect in ("postgresql", "postgres")
            save_data_option = "postgres" if is_postgresql else "db"

            # Build command
            cmd = [
                sys.executable, "main.py",
                "--platform", platform,
                "--lt", "cookie",
                "--type", "search",
                "--save_data_option", save_data_option,
                "--headless", "true"
            ]

            logger.info(f"Executing command: {' '.join(cmd)}")

            # Switch to MediaCrawler directory and execute
            result = subprocess.run(
                cmd,
                cwd=self.mediacrawler_path,
                timeout=3600  # 60 minute timeout
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Create statistics
            crawl_stats = {
                "platform": platform,
                "keywords_count": len(keywords),
                "duration_seconds": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "notes_count": 0,
                "comments_count": 0,
                "errors_count": 0
            }

            # Save statistics
            self.crawl_stats[platform] = crawl_stats

            if result.returncode == 0:
                logger.info(f"✅ {platform} crawling completed, time taken: {duration:.1f}s")
            else:
                logger.error(f"❌ {platform} crawling failed, return code: {result.returncode}")

            return crawl_stats

        except subprocess.TimeoutExpired:
            logger.exception(f"❌ {platform} crawling timeout")
            return {"success": False, "error": "Crawling timeout", "platform": platform}
        except Exception as e:
            logger.exception(f"❌ {platform} crawling error: {e}")
            return {"success": False, "error": str(e), "platform": platform}
    
    def _parse_crawl_output(self, output_lines: List[str], error_lines: List[str]) -> Dict:
        """Parse crawling output and extract statistics"""
        stats = {
            "notes_count": 0,
            "comments_count": 0,
            "errors_count": 0,
            "login_required": False
        }

        # Parse output lines
        for line in output_lines:
            if "notes" in line or "content items" in line:
                try:
                    # Extract numbers
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        stats["notes_count"] = int(numbers[0])
                except:
                    pass
            elif "comments" in line:
                try:
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        stats["comments_count"] = int(numbers[0])
                except:
                    pass
            elif "login" in line.lower() or "QR code" in line:
                stats["login_required"] = True

        # Parse error lines
        for line in error_lines:
            if "error" in line.lower() or "exception" in line.lower():
                stats["errors_count"] += 1

        return stats
    
    def run_multi_platform_crawl_by_keywords(self, keywords: List[str], platforms: List[str],
                                            login_type: str = "qrcode", max_notes_per_keyword: int = 50) -> Dict:
        """
        Multi-platform crawling based on keywords - each keyword is crawled on all platforms

        Args:
            keywords: List of keywords
            platforms: List of platforms
            login_type: Login type
            max_notes_per_keyword: Maximum number of notes to crawl per keyword per platform

        Returns:
            Overall crawling statistics
        """
        
        start_message = f"\n🚀 Starting multi-platform keyword crawling"
        start_message += f"\n   Keywords count: {len(keywords)}"
        start_message += f"\n   Platforms count: {len(platforms)}"
        start_message += f"\n   Login type: {login_type}"
        start_message += f"\n   Max notes per keyword per platform: {max_notes_per_keyword}"
        start_message += f"\n   Total crawling tasks: {len(keywords)} × {len(platforms)} = {len(keywords) * len(platforms)}"
        logger.info(start_message)
        
        total_stats = {
            "total_keywords": len(keywords),
            "total_platforms": len(platforms),
            "total_tasks": len(keywords) * len(platforms),
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_notes": 0,
            "total_comments": 0,
            "keyword_results": {},
            "platform_summary": {}
        }
        
        # Initialize platform statistics
        for platform in platforms:
            total_stats["platform_summary"][platform] = {
                "successful_keywords": 0,
                "failed_keywords": 0,
                "total_notes": 0,
                "total_comments": 0
            }

        # Crawl all keywords for each platform in one batch
        for platform in platforms:
            logger.info(f"\n📝 Crawling all keywords on {platform} platform")
            logger.info(f"   Keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}")
            
            try:
                # Pass all keywords to platform in one batch
                result = self.run_crawler(platform, keywords, login_type, max_notes_per_keyword)

                if result.get("success"):
                    total_stats["successful_tasks"] += len(keywords)
                    total_stats["platform_summary"][platform]["successful_keywords"] = len(keywords)

                    notes_count = result.get("notes_count", 0)
                    comments_count = result.get("comments_count", 0)

                    total_stats["total_notes"] += notes_count
                    total_stats["total_comments"] += comments_count
                    total_stats["platform_summary"][platform]["total_notes"] = notes_count
                    total_stats["platform_summary"][platform]["total_comments"] = comments_count

                    # Record results for each keyword
                    for keyword in keywords:
                        if keyword not in total_stats["keyword_results"]:
                            total_stats["keyword_results"][keyword] = {}
                        total_stats["keyword_results"][keyword][platform] = result

                    logger.info(f"   ✅ Crawling successful")
                else:
                    total_stats["failed_tasks"] += len(keywords)
                    total_stats["platform_summary"][platform]["failed_keywords"] = len(keywords)

                    # Record failure results for each keyword
                    for keyword in keywords:
                        if keyword not in total_stats["keyword_results"]:
                            total_stats["keyword_results"][keyword] = {}
                        total_stats["keyword_results"][keyword][platform] = result

                    logger.error(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                total_stats["failed_tasks"] += len(keywords)
                total_stats["platform_summary"][platform]["failed_keywords"] = len(keywords)
                error_result = {"success": False, "error": str(e)}

                # Record exception results for each keyword
                for keyword in keywords:
                    if keyword not in total_stats["keyword_results"]:
                        total_stats["keyword_results"][keyword] = {}
                    total_stats["keyword_results"][keyword][platform] = error_result

                logger.error(f"   ❌ Exception: {e}")
        
        # Print detailed statistics
        finish_message = f"\n📊 Multi-platform keyword crawling completed!"
        finish_message += f"\n   Total tasks: {total_stats['total_tasks']}"
        finish_message += f"\n   Successful: {total_stats['successful_tasks']}"
        finish_message += f"\n   Failed: {total_stats['failed_tasks']}"
        finish_message += f"\n   Success rate: {total_stats['successful_tasks']/total_stats['total_tasks']*100:.1f}%"
        logger.info(finish_message)

        platform_summary_message = f"\n📈 Platform Statistics:"
        for platform, stats in total_stats["platform_summary"].items():
            success_rate = stats["successful_keywords"] / len(keywords) * 100 if keywords else 0
            platform_summary_message += f"\n   {platform}: {stats['successful_keywords']}/{len(keywords)} keywords successful ({success_rate:.1f}%)"
        logger.info(platform_summary_message)
        
        return total_stats
    
    def _run_bili_api_crawler(self, keywords: List[str], max_notes: int, start_time) -> Dict:
        """B站专用：用 bilibili-api-python 替代 Playwright，绕开 412 风控"""
        try:
            import asyncio
            from DeepSentimentCrawling.bilibili_api_crawler import search_and_crawl
        except ImportError:
            logger.error("bilibili_api_crawler 导入失败，请确认已安装 bilibili-api-python")
            return {"success": False, "error": "bilibili_api_crawler import failed", "platform": "bili"}

        try:
            result = asyncio.run(search_and_crawl(keywords, max_videos=max_notes, max_comments=20))
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            stats = {
                "platform": "bili",
                "keywords_count": len(keywords),
                "duration_seconds": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "return_code": 0 if result["success"] else 1,
                "success": result["success"],
                "notes_count": result["videos"],
                "comments_count": result["comments"],
                "errors_count": 0,
            }
            self.crawl_stats["bili"] = stats
            if result["success"]:
                logger.info(f"✅ bili crawling completed via API, videos={result['videos']}, comments={result['comments']}, time={duration:.1f}s")
            else:
                logger.error("❌ bili API crawling returned no data")
            return stats
        except Exception as e:
            logger.exception(f"❌ bili API crawling error: {e}")
            return {"success": False, "error": str(e), "platform": "bili"}

    def get_crawl_statistics(self) -> Dict:
        """Get crawling statistics"""
        return {
            "platforms_crawled": list(self.crawl_stats.keys()),
            "total_platforms": len(self.crawl_stats),
            "detailed_stats": self.crawl_stats
        }

    def save_crawl_log(self, log_path: str = None):
        """Save crawling log"""
        if not log_path:
            log_path = f"crawl_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(self.crawl_stats, f, ensure_ascii=False, indent=2)
            logger.info(f"Crawling log saved to: {log_path}")
        except Exception as e:
            logger.exception(f"Failed to save crawling log: {e}")

if __name__ == "__main__":
    # Test platform crawler manager
    crawler = PlatformCrawler()

    # Test configuration
    test_keywords = ["technology", "AI", "programming"]
    result = crawler.run_crawler("xhs", test_keywords, max_notes=5)

    logger.info(f"Test result: {result}")
    logger.info("Platform crawler manager test completed!")
