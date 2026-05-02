"""
MindSpiderDB — QueryEngine search tool for MindSpider social media database.

Connects to the capstone MySQL database and provides keyword-based search
across all crawled platform tables (xhs, douyin, kuaishou, bilibili, weibo, tieba, zhihu).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote_plus

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# DB connection helpers
# ---------------------------------------------------------------------------

def _build_mindspider_url() -> str:
    # Use QueryEngine settings (which loads .env) for DB credentials
    try:
        from ..utils.config import settings as _settings
        host = _settings.DB_HOST
        port = str(_settings.DB_PORT)
        user = _settings.DB_USER
        password = quote_plus(_settings.DB_PASSWORD)
        charset = _settings.DB_CHARSET
    except Exception:
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "3306")
        user = os.getenv("DB_USER", "root")
        password = quote_plus(os.getenv("DB_PASSWORD", ""))
        charset = os.getenv("DB_CHARSET", "utf8mb4")
    # Always use the project's unified capstone database
    db_name = os.getenv("DB_NAME", "capstone")
    try:
        from ..utils.config import settings as _s
        db_name = getattr(_s, "DB_NAME", "capstone")
    except Exception:
        pass
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}?charset={charset}"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MindSpiderResult:
    platform: str
    source_table: str
    title_or_content: str
    url: Optional[str]
    publish_time: Optional[datetime]
    hotness_score: Optional[float]
    source_keyword: Optional[str]


@dataclass
class MindSpiderComment:
    platform: str
    content: str
    like_count: int
    publish_time: Optional[datetime]


@dataclass
class MindSpiderResponse:
    results: List[MindSpiderResult] = field(default_factory=list)
    total: int = 0


# ---------------------------------------------------------------------------
# Per-platform query specs
# ---------------------------------------------------------------------------

_PLATFORM_QUERIES = [
    {
        "platform": "xhs",
        "table": "xhs_note",
        "content_col": "COALESCE(title, `desc`)",
        "url_col": "note_url",
        "time_col": "time",
        "time_is_ts": True,
        "search_cols": ["title", "`desc`", "source_keyword"],
    },
    {
        "platform": "douyin",
        "table": "douyin_aweme",
        "content_col": "COALESCE(title, `desc`)",
        "url_col": "aweme_url",
        "time_col": "create_time",
        "time_is_ts": True,
        "search_cols": ["title", "`desc`", "source_keyword"],
    },
    {
        "platform": "kuaishou",
        "table": "kuaishou_video",
        "content_col": "COALESCE(title, `desc`)",
        "url_col": "video_url",
        "time_col": "create_time",
        "time_is_ts": True,
        "search_cols": ["title", "`desc`", "source_keyword"],
    },
    {
        "platform": "bilibili",
        "table": "bilibili_video",
        "content_col": "COALESCE(title, `desc`)",
        "url_col": "video_url",
        "time_col": "create_time",
        "time_is_ts": True,
        "search_cols": ["title", "`desc`", "source_keyword"],
    },
    {
        "platform": "weibo",
        "table": "weibo_note",
        "content_col": "content",
        "url_col": "note_url",
        "time_col": "create_time",
        "time_is_ts": True,
        "search_cols": ["content", "source_keyword"],
    },
    {
        "platform": "tieba",
        "table": "tieba_note",
        "content_col": "COALESCE(title, `desc`)",
        "url_col": "note_url",
        "time_col": "publish_time",
        "time_is_ts": False,
        "search_cols": ["title", "`desc`", "source_keyword"],
    },
    {
        "platform": "zhihu",
        "table": "zhihu_content",
        "content_col": "COALESCE(title, content_text)",
        "url_col": "content_url",
        "time_col": "created_time",
        "time_is_ts": False,
        "search_cols": ["title", "content_text", "source_keyword"],
    },
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MindSpiderDB:
    """Search MindSpider social media database for QueryEngine integration."""

    def __init__(self):
        self._engine: Optional[Engine] = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            url = _build_mindspider_url()
            self._engine = create_engine(url, pool_pre_ping=True, pool_recycle=1800)
        return self._engine

    def search_topic_globally(
        self,
        keyword: str,
        limit_per_table: int = 20,
    ) -> MindSpiderResponse:
        """
        Search all platform tables for keyword, return unified results.
        Falls back gracefully if a table doesn't exist yet.
        """
        all_results: List[MindSpiderResult] = []

        for spec in _PLATFORM_QUERIES:
            try:
                rows = self._search_table(keyword, spec, limit_per_table)
                all_results.extend(rows)
            except Exception as exc:
                logger.debug(f"[MindSpiderDB] {spec['table']} search skipped: {exc}")

        all_results.sort(key=lambda r: r.hotness_score or 0.0, reverse=True)
        return MindSpiderResponse(results=all_results, total=len(all_results))

    def _search_table(
        self,
        keyword: str,
        spec: dict,
        limit: int,
    ) -> List[MindSpiderResult]:
        table = spec["table"]
        content_col = spec["content_col"]
        url_col = spec["url_col"]
        time_col = spec["time_col"]
        time_is_ts = spec["time_is_ts"]
        search_cols = spec["search_cols"]

        like_clauses = " OR ".join(f"{c} LIKE :kw" for c in search_cols)

        if time_is_ts:
            time_expr = f"FROM_UNIXTIME({time_col})"
        else:
            time_expr = f"STR_TO_DATE({time_col}, '%Y-%m-%d %H:%i:%s')"

        sql = text(f"""
            SELECT
                {content_col}  AS content,
                {url_col}      AS url,
                {time_expr}    AS publish_time,
                source_keyword
            FROM {table}
            WHERE {like_clauses}
            ORDER BY {time_col} DESC
            LIMIT :lim
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"kw": f"%{keyword}%", "lim": limit}).fetchall()

        results = []
        for row in rows:
            content = row[0] or ""
            url = row[1]
            publish_time = row[2]
            source_keyword = row[3]

            results.append(MindSpiderResult(
                platform=spec["platform"],
                source_table=table,
                title_or_content=content,
                url=url,
                publish_time=publish_time if isinstance(publish_time, datetime) else None,
                hotness_score=None,
                source_keyword=source_keyword,
            ))

        return results

    # ------------------------------------------------------------------
    # Probe: lightweight data availability check
    # ------------------------------------------------------------------

    _COMMENT_TABLES = [
        {"table": "weibo_note_comment", "content_col": "content"},
        {"table": "zhihu_comment", "content_col": "content"},
        {"table": "bilibili_video_comment", "content_col": "content"},
        {"table": "douyin_aweme_comment", "content_col": "content"},
        {"table": "kuaishou_video_comment", "content_col": "content"},
        {"table": "xhs_note_comment", "content_col": "content"},
        {"table": "tieba_comment", "content_col": "content"},
    ]

    def probe(self, keyword: str) -> dict:
        """
        Quick COUNT query across all platform tables.

        Returns:
            {
                "total_posts": int,
                "platforms": {"weibo": 5, "zhihu": 3, ...},
                "newest_timestamp": datetime | None,
                "freshness_hours": float | None,
            }
        """
        platforms: dict[str, int] = {}
        newest_ts: Optional[datetime] = None

        for spec in _PLATFORM_QUERIES:
            try:
                count, max_time = self._probe_table(keyword, spec)
                if count > 0:
                    platforms[spec["platform"]] = count
                    if max_time and (newest_ts is None or max_time > newest_ts):
                        newest_ts = max_time
            except Exception:
                pass

        total = sum(platforms.values())
        freshness = None
        if newest_ts:
            delta = datetime.now() - newest_ts
            freshness = delta.total_seconds() / 3600.0

        return {
            "total_posts": total,
            "platforms": platforms,
            "newest_timestamp": newest_ts,
            "freshness_hours": freshness,
        }

    def _probe_table(self, keyword: str, spec: dict) -> tuple:
        """Return (count, max_publish_time) for a single table."""
        table = spec["table"]
        time_col = spec["time_col"]
        time_is_ts = spec["time_is_ts"]
        search_cols = spec["search_cols"]

        like_clauses = " OR ".join(f"{c} LIKE :kw" for c in search_cols)

        if time_is_ts:
            time_expr = f"FROM_UNIXTIME(MAX({time_col}))"
        else:
            time_expr = f"MAX(STR_TO_DATE({time_col}, '%Y-%m-%d %H:%i:%s'))"

        sql = text(f"""
            SELECT COUNT(*), {time_expr}
            FROM {table}
            WHERE {like_clauses}
        """)

        with self.engine.connect() as conn:
            row = conn.execute(sql, {"kw": f"%{keyword}%"}).fetchone()

        count = row[0] if row else 0
        max_time = row[1] if row and row[1] else None
        if max_time and not isinstance(max_time, datetime):
            max_time = None
        return count, max_time

    def count_comments(self, keyword: str) -> int:
        """Count comments across all comment tables matching keyword."""
        total = 0
        for spec in self._COMMENT_TABLES:
            try:
                sql = text(f"""
                    SELECT COUNT(*)
                    FROM {spec['table']}
                    WHERE {spec['content_col']} LIKE :kw
                """)
                with self.engine.connect() as conn:
                    row = conn.execute(sql, {"kw": f"%{keyword}%"}).fetchone()
                    total += row[0] if row else 0
            except Exception:
                pass
        return total

    # ------------------------------------------------------------------
    # Comment search (Ext 1)
    # ------------------------------------------------------------------

    _COMMENT_QUERY_SPECS = [
        {
            "platform": "weibo", "table": "weibo_note_comment",
            "content_col": "content", "time_col": "create_time",
            "time_is_ts": True, "like_col": "comment_like_count",
        },
        {
            "platform": "zhihu", "table": "zhihu_comment",
            "content_col": "content", "time_col": "publish_time",
            "time_is_ts": False, "like_col": "like_count",
        },
        {
            "platform": "bilibili", "table": "bilibili_video_comment",
            "content_col": "content", "time_col": "create_time",
            "time_is_ts": True, "like_col": "like_count",
        },
        {
            "platform": "douyin", "table": "douyin_aweme_comment",
            "content_col": "content", "time_col": "create_time",
            "time_is_ts": True, "like_col": "like_count",
        },
        {
            "platform": "kuaishou", "table": "kuaishou_video_comment",
            "content_col": "content", "time_col": "create_time",
            "time_is_ts": True, "like_col": None,
        },
        {
            "platform": "xhs", "table": "xhs_note_comment",
            "content_col": "content", "time_col": "create_time",
            "time_is_ts": True, "like_col": "like_count",
        },
        {
            "platform": "tieba", "table": "tieba_comment",
            "content_col": "content", "time_col": "publish_time",
            "time_is_ts": False, "like_col": None,
        },
    ]

    def search_comments(
        self, keyword: str, limit_per_table: int = 10,
    ) -> List[MindSpiderComment]:
        """Search comment tables for keyword, return actual comment content."""
        all_comments: List[MindSpiderComment] = []

        for spec in self._COMMENT_QUERY_SPECS:
            try:
                comments = self._search_comment_table(keyword, spec, limit_per_table)
                all_comments.extend(comments)
            except Exception as exc:
                logger.debug(f"[MindSpiderDB] {spec['table']} comment search skipped: {exc}")

        all_comments.sort(key=lambda c: c.like_count, reverse=True)
        return all_comments

    def _search_comment_table(
        self, keyword: str, spec: dict, limit: int,
    ) -> List[MindSpiderComment]:
        table = spec["table"]
        content_col = spec["content_col"]
        time_col = spec["time_col"]
        time_is_ts = spec["time_is_ts"]
        like_col = spec.get("like_col")

        if time_is_ts:
            time_expr = f"FROM_UNIXTIME({time_col})"
        else:
            time_expr = f"STR_TO_DATE({time_col}, '%Y-%m-%d %H:%i:%s')"

        like_expr = like_col if like_col else "0"

        sql = text(f"""
            SELECT {content_col} AS content,
                   {time_expr} AS publish_time,
                   {like_expr} AS likes
            FROM {table}
            WHERE {content_col} LIKE :kw
            ORDER BY {time_col} DESC
            LIMIT :lim
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"kw": f"%{keyword}%", "lim": limit}).fetchall()

        results = []
        for row in rows:
            content = row[0] or ""
            pub_time = row[1] if isinstance(row[1], datetime) else None
            try:
                likes = int(row[2]) if row[2] else 0
            except (TypeError, ValueError):
                likes = 0
            results.append(MindSpiderComment(
                platform=spec["platform"],
                content=content,
                like_count=likes,
                publish_time=pub_time,
            ))
        return results

    # ------------------------------------------------------------------
    # Temporal search (Ext 2)
    # ------------------------------------------------------------------

    def search_with_time_buckets(
        self, keyword: str, days_back: int = 7,
    ) -> Dict[str, List[MindSpiderResult]]:
        """
        Search all platform tables with a time window, group results by date.
        Returns dict mapping "YYYY-MM-DD" -> list of MindSpiderResult.
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days_back)
        buckets: Dict[str, List[MindSpiderResult]] = {}

        for spec in _PLATFORM_QUERIES:
            try:
                rows = self._search_table_with_time(keyword, spec, cutoff, limit=50)
                for r in rows:
                    if r.publish_time:
                        date_key = r.publish_time.strftime("%Y-%m-%d")
                    else:
                        continue
                    buckets.setdefault(date_key, []).append(r)
            except Exception as exc:
                logger.debug(f"[MindSpiderDB] {spec['table']} temporal search skipped: {exc}")

        return buckets

    def _search_table_with_time(
        self, keyword: str, spec: dict, cutoff: datetime, limit: int,
    ) -> List[MindSpiderResult]:
        table = spec["table"]
        content_col = spec["content_col"]
        url_col = spec["url_col"]
        time_col = spec["time_col"]
        time_is_ts = spec["time_is_ts"]
        search_cols = spec["search_cols"]

        like_clauses = " OR ".join(f"{c} LIKE :kw" for c in search_cols)

        if time_is_ts:
            time_expr = f"FROM_UNIXTIME({time_col})"
            cutoff_clause = f"{time_col} >= :cutoff_ts"
            cutoff_val = int(cutoff.timestamp())
        else:
            time_expr = f"STR_TO_DATE({time_col}, '%Y-%m-%d %H:%i:%s')"
            cutoff_clause = f"STR_TO_DATE({time_col}, '%Y-%m-%d %H:%i:%s') >= :cutoff_ts"
            cutoff_val = cutoff.strftime("%Y-%m-%d %H:%M:%S")

        sql = text(f"""
            SELECT {content_col} AS content, {url_col} AS url,
                   {time_expr} AS publish_time, source_keyword
            FROM {table}
            WHERE ({like_clauses}) AND {cutoff_clause}
            ORDER BY {time_col} DESC
            LIMIT :lim
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(
                sql, {"kw": f"%{keyword}%", "cutoff_ts": cutoff_val, "lim": limit},
            ).fetchall()

        results = []
        for row in rows:
            pub_time = row[2] if isinstance(row[2], datetime) else None
            results.append(MindSpiderResult(
                platform=spec["platform"],
                source_table=table,
                title_or_content=row[0] or "",
                url=row[1],
                publish_time=pub_time,
                hotness_score=None,
                source_keyword=row[3],
            ))
        return results

    # ------------------------------------------------------------------
    # BTE check (Ext 3)
    # ------------------------------------------------------------------

    def has_extraction_today(self) -> bool:
        """Check if BroadTopicExtraction has run today."""
        try:
            sql = text("SELECT COUNT(*) FROM daily_topics WHERE extract_date = CURDATE()")
            with self.engine.connect() as conn:
                row = conn.execute(sql).fetchone()
                return (row[0] or 0) > 0
        except Exception:
            return False
