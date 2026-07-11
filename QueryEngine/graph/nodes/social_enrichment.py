"""
SocialEnrichment Node -- Probe-then-decide MindSpider integration.

Phase 3 node. Positioned: stance_classify -> social_enrichment -> coverage_check.

Extensions:
  - Ext 1: Comment sentiment aggregation (comment-level stance analysis)
  - Ext 2: Temporal sentiment tracking (per-day trend detection)
  - Ext 3: Active BroadTopicExtraction trigger (fire-and-forget when data stale)
  - Ext 4: LLM batch stance classifier (replaces rule-based for social posts)

References:
  - Resource Selection: Callan et al., CORI, SIGIR 1995
  - Corrective RAG: Yan et al., 2024
  - Stance Detection: Mohammad et al., SemEval 2016
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from loguru import logger

from ...classifiers.stance_classifier import HybridStanceClassifier
from ...llms import LLMClient
from ...utils.config import settings
from ..state import QueryAgentState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FRESHNESS_THRESHOLD_HOURS = 72.0
_MIN_POSTS_FOR_ENRICHMENT = 3

_STOPWORDS = frozenset([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些", "什么", "怎么",
    "如何", "为什么", "各方", "舆论", "评价", "分析", "讨论", "观点", "看法",
    "最新", "最近", "目前", "关于", "对于", "以及", "还是", "或者", "但是",
])

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity between two stance distribution vectors."""
    all_keys = set(vec_a) | set(vec_b)
    if not all_keys:
        return 1.0
    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in all_keys)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _extract_probe_keywords(query: str) -> List[str]:
    """Extract meaningful keywords from query for MindSpider probe."""
    normalized = query.replace("'", " ").replace("'", " ").replace('"', " ")
    tokens = re.findall(r"[a-zA-Z0-9]+[-.]?[a-zA-Z0-9]*|[\u4e00-\u9fff]+", normalized)
    keywords = []
    for token in tokens:
        token = token.strip()
        if not token or token.lower() in _STOPWORDS or len(token) <= 1:
            continue
        keywords.append(token)

    keywords.sort(key=len, reverse=True)
    deduped: List[str] = []
    seen = set()
    for kw in keywords:
        key = kw.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(kw)
    return deduped if deduped else ([query.strip()] if query.strip() else [query])


def _stance_distribution(items: list, key: str = "stance") -> Dict[str, float]:
    """Compute normalized stance distribution from a list of classified items."""
    counts = Counter(item[key] for item in items)
    total = max(len(items), 1)
    return {s: round(c / total, 3) for s, c in counts.items()}


def _disabled_result(crawl_triggered: bool = False) -> dict:
    sentiment = None
    if crawl_triggered:
        sentiment = {
            "mode": "disabled",
            "crawl_triggered": True,
            "platforms_queried": [],
            "total_posts": 0,
            "total_comments": 0,
            "sentiment_distribution": {},
            "divergence_score": 0.0,
            "divergence_summary": "",
            "freshness_hours": 0,
            "top_social_voices": [],
            "comment_sentiment": None,
            "sentiment_trend": None,
        }
    return {
        "mindspider_mode": "disabled",
        "social_sentiment": sentiment,
        "trace_log": ["[SocialEnrichment] MindSpider disabled (no data or probe failed)"],
    }


def _get_llm_client() -> LLMClient:
    return LLMClient(
        api_key=settings.QUERY_ENGINE_API_KEY,
        model_name=settings.QUERY_ENGINE_MODEL_NAME,
        base_url=settings.QUERY_ENGINE_BASE_URL,
    )


def _llm_chinese_probe_keywords(query: str) -> List[str]:
    """Derive Chinese probe keywords for English-topic queries."""
    if any("\u4e00" <= ch <= "\u9fff" for ch in query):
        return []
    try:
        import json

        llm = _get_llm_client()
        response = llm.invoke(
            system_prompt=(
                "You extract Chinese social-media search keywords. "
                "Output only a JSON array of 2-4 strings, no other text."
            ),
            user_prompt=(
                f"Topic: {query}\n"
                "Return concise Chinese keywords suitable for searching "
                "Weibo/Douyin/Bilibili/Zhihu discussions."
            ),
        )
        text = re.sub(r"```(?:json)?", "", response or "").strip()
        match = re.search(r"\[.*\]", text, re.DOTALL)
        payload = match.group() if match else text
        data = json.loads(payload)
        if not isinstance(data, list):
            return []
        return [str(item).strip() for item in data if str(item).strip()]
    except Exception as exc:
        logger.debug(f"[SocialEnrichment] Chinese keyword expansion skipped: {exc}")
        return []


def _resolve_python_bin(project_root: str) -> str:
    """Pick a Python executable that works on Windows and Unix."""
    import sys

    candidates = [
        sys.executable,
        os.path.join(project_root, ".venv", "Scripts", "python.exe"),
        os.path.join(project_root, ".venv", "bin", "python"),
        os.path.join(project_root, ".venv", "Scripts", "python"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return sys.executable


# ---------------------------------------------------------------------------
# Ext 3: BroadTopicExtraction trigger (fire-and-forget)
# ---------------------------------------------------------------------------

def _maybe_trigger_extraction(db) -> bool:
    """
    Check if BroadTopicExtraction has run today. If not, trigger it
    as a detached subprocess (non-blocking, ~30s, no Playwright).
    Returns True if triggered.
    """
    try:
        if db.has_extraction_today():
            return False
    except Exception:
        return False

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )))
    mindspider_dir = os.path.join(project_root, "MindSpider")
    python_bin = _resolve_python_bin(project_root)
    script = os.path.join(mindspider_dir, "BroadTopicExtraction", "main.py")

    if not os.path.exists(script):
        logger.debug(f"[SocialEnrichment] BTE script not found: {script}")
        return False

    popen_kwargs = {
        "cwd": mindspider_dir,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True

    try:
        subprocess.Popen(
            [python_bin, script, "--keywords", "30", "--quiet"],
            **popen_kwargs,
        )
        logger.info("[SocialEnrichment] Triggered BroadTopicExtraction in background")
        return True
    except Exception as exc:
        logger.warning(f"[SocialEnrichment] Failed to trigger BTE: {exc}")
        return False


# ---------------------------------------------------------------------------
# Ext 2: Temporal trend detection
# ---------------------------------------------------------------------------

def _detect_trend(buckets: List[dict]) -> str:
    """
    Compare support ratio in first half vs second half of time buckets.
    Returns "rising", "falling", or "stable".
    """
    if len(buckets) < 2:
        return "stable"

    mid = len(buckets) // 2
    first_half = buckets[:mid]
    second_half = buckets[mid:]

    def avg_support(bkts):
        vals = [b["distribution"].get("support", 0) for b in bkts]
        return sum(vals) / max(len(vals), 1)

    delta = avg_support(second_half) - avg_support(first_half)
    if delta > 0.10:
        return "rising"
    elif delta < -0.10:
        return "falling"
    return "stable"


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

DIVERGENCE_PROMPT = """Topic: {query}

Web search results stance distribution: {web_dist}
Social media discussion stance distribution: {social_dist}
Cross-Source Sentiment Difference Score: {nsds:.2f} (0=identical, 1=very different)

In 1-2 sentences, describe how the sentiment differs between web search results and social media discussions on this topic. Focus on factual differences in stance proportions. Do NOT frame this as "official vs public" or imply any political narrative. Simply compare the two information channels objectively."""


TREND_SUMMARY_PROMPT = """Topic: {query}

Sentiment trend over time (daily buckets):
{buckets_text}

Trend direction: {direction}

In 1-2 sentences, summarize how public sentiment on this topic has changed over time. If stable, note the dominant sentiment."""


async def _generate_divergence_summary(
    query: str, web_dist: Dict, social_dist: Dict, nsds: float,
) -> str:
    """Generate LLM summary of web vs social divergence."""
    try:
        llm = _get_llm_client()
        prompt = DIVERGENCE_PROMPT.format(
            query=query, web_dist=web_dist, social_dist=social_dist, nsds=nsds,
        )
        return llm.invoke(
            system_prompt=(
                "You are a public opinion analyst. Write in English only. Output only the summary."
            ),
            user_prompt=prompt,
        )
    except Exception as exc:
        logger.warning(f"[SocialEnrichment] Divergence summary LLM failed: {exc}")
        if nsds > 0.5:
            return f"Notable difference (score={nsds:.2f}) between web search and social media sentiment distributions."
        return f"Similar sentiment (score={nsds:.2f}) across web search and social media channels."


async def _generate_trend_summary(
    query: str, buckets: List[dict], direction: str,
) -> str:
    """Generate LLM summary of sentiment trend."""
    try:
        llm = _get_llm_client()
        buckets_text = "\n".join(
            f"  {b['date']}: {b['post_count']} posts, distribution={b['distribution']}"
            for b in buckets
        )
        prompt = TREND_SUMMARY_PROMPT.format(
            query=query, buckets_text=buckets_text, direction=direction,
        )
        return llm.invoke(
            system_prompt=(
                "You are a public opinion analyst. Write in English only. Output only the summary."
            ),
            user_prompt=prompt,
        )
    except Exception as exc:
        logger.warning(f"[SocialEnrichment] Trend summary LLM failed: {exc}")
        return f"Sentiment trend is {direction} over the observed period."


# ---------------------------------------------------------------------------
# Node Function
# ---------------------------------------------------------------------------


def _crawl_trigger_enabled(settings) -> bool:
    return bool(getattr(settings, "COORDINATOR_ALLOW_MINDSPIDER_CRAWL_TRIGGER", False))


async def social_enrichment_node(state: QueryAgentState) -> dict:
    """
    LangGraph Node: MindSpider social media enrichment with NSDS.

    Integrates: LLM stance classification (Ext 4), comment sentiment (Ext 1),
    temporal tracking (Ext 2), and BTE trigger (Ext 3).
    """
    query = state.get("original_query", "")

    from ...utils.config import settings

    if not bool(getattr(settings, "COORDINATOR_ENABLE_MINDSPIDER_DB", False)):
        return _disabled_result()

    # Skip if already computed in a previous iteration (coverage loop)
    if state.get("social_sentiment") is not None:
        return {
            "mindspider_mode": state.get("mindspider_mode", "disabled"),
            "social_sentiment": state.get("social_sentiment"),
            "trace_log": ["[SocialEnrichment] Skipped (already computed in previous iteration)"],
        }

    # -- Step 1: Probe MindSpider with extracted keywords --
    try:
        from ...tools.mindspider_search import MindSpiderDB
        db = MindSpiderDB()
    except Exception as exc:
        logger.warning(f"[SocialEnrichment] MindSpider import failed: {exc}")
        return _disabled_result()

    probe_keywords = _extract_probe_keywords(query)
    probe_result = None
    best_keyword = query

    for kw in probe_keywords:
        try:
            result = db.probe(kw)
            if result.get("total_posts", 0) >= _MIN_POSTS_FOR_ENRICHMENT:
                probe_result = result
                best_keyword = kw
                break
        except Exception:
            continue

    if probe_result is None or probe_result.get("total_posts", 0) < _MIN_POSTS_FOR_ENRICHMENT:
        for kw in _llm_chinese_probe_keywords(query):
            try:
                result = db.probe(kw)
                if result.get("total_posts", 0) >= _MIN_POSTS_FOR_ENRICHMENT:
                    probe_result = result
                    best_keyword = kw
                    break
            except Exception:
                continue

    if probe_result is None:
        try:
            probe_result = db.probe(query)
        except Exception as exc:
            logger.warning(f"[SocialEnrichment] MindSpider probe failed: {exc}")
            return _disabled_result()

    total_posts = probe_result.get("total_posts", 0)
    freshness_hours = probe_result.get("freshness_hours")

    if total_posts < _MIN_POSTS_FOR_ENRICHMENT:
        logger.info(f"[SocialEnrichment] Insufficient data ({total_posts} posts)")
        # Ext 3: trigger BTE if no data
        triggered = _maybe_trigger_extraction(db) if _crawl_trigger_enabled(settings) else False
        return _disabled_result(crawl_triggered=triggered)

    # -- Step 2: Determine mode --
    if freshness_hours is not None and freshness_hours < _FRESHNESS_THRESHOLD_HOURS:
        mode = "available"
    else:
        mode = "stale"

    # Ext 3: trigger BTE if stale
    crawl_triggered = False
    if mode == "stale":
        crawl_triggered = _maybe_trigger_extraction(db) if _crawl_trigger_enabled(settings) else False

    # -- Step 3: Full query --
    try:
        response = db.search_topic_globally(best_keyword, limit_per_table=20)
    except Exception as exc:
        logger.warning(f"[SocialEnrichment] MindSpider query failed: {exc}")
        return _disabled_result()

    if not response.results:
        return _disabled_result()

    # -- Step 4: Classify post stances (Ext 4: LLM batch) --
    classifier = HybridStanceClassifier()
    llm = _get_llm_client()

    source_dicts = [
        {
            "url": r.url or "",
            "title": (r.title_or_content or "")[:100],
            "snippet": (r.title_or_content or "")[:500],
            "_target_stance": "",
        }
        for r in response.results
    ]

    llm_results = classifier.classify_batch_llm(source_dicts, query, llm)

    classified_posts = []
    for i, r in enumerate(response.results):
        if llm_results and i < len(llm_results):
            stance, confidence = llm_results[i]
        else:
            stance, confidence = classifier.classify(source_dicts[i], query)

        classified_posts.append({
            "platform": r.platform,
            "content": (r.title_or_content or "")[:300],
            "url": r.url or "",
            "publish_time": r.publish_time.isoformat() if r.publish_time else None,
            "stance": stance,
            "confidence": confidence,
            "source_keyword": r.source_keyword,
        })

    # -- Step 5: Compute social stance distribution --
    social_stance_dist = _stance_distribution(classified_posts)

    # -- Per-platform breakdown (Fix 3) --
    platform_breakdown = {}
    for p in classified_posts:
        plat = p["platform"]
        platform_breakdown.setdefault(plat, []).append(p)
    per_platform = {
        plat: {
            "count": len(posts),
            "distribution": _stance_distribution(posts),
        }
        for plat, posts in platform_breakdown.items()
    }

    # -- Content dedup for bot/astroturfing detection (Fix 1) --
    content_texts = [p["content"] for p in classified_posts if p["content"]]
    unique_count = len(set(content_texts))
    total_count = max(len(content_texts), 1)
    content_diversity = round(unique_count / total_count, 3)
    low_diversity_warning = None
    if content_diversity < 0.7 and total_count >= 5:
        low_diversity_warning = (
            f"Low content diversity ({content_diversity:.0%}): "
            f"{total_count - unique_count} near-duplicate posts detected. "
            f"Results may be influenced by coordinated posting."
        )

    # -- Step 6: Compute NSDS --
    web_sources = state.get("classified_sources") or []
    web_stance_counts = Counter(
        s.get("stance_label") or "neutral" for s in web_sources
        if s.get("stance_label") and s.get("stance_label") != "unclassified"
    )
    web_total = max(sum(web_stance_counts.values()), 1)
    web_stance_dist = {
        s: round(c / web_total, 3) for s, c in web_stance_counts.items()
    }

    cos_sim = _cosine_similarity(web_stance_dist, social_stance_dist)
    nsds = round(1.0 - cos_sim, 3)

    # -- Step 7: LLM divergence summary --
    divergence_summary = await _generate_divergence_summary(
        query, web_stance_dist, social_stance_dist, nsds,
    )

    # -- Step 8: Top social voices --
    sorted_posts = sorted(classified_posts, key=lambda p: p["confidence"], reverse=True)
    top_voices = [
        {k: v for k, v in p.items() if k != "source_keyword" and k != "confidence"}
        for p in sorted_posts[:10]
    ]

    # -- Ext 1: Comment sentiment aggregation --
    comment_sentiment = None
    try:
        comments = db.search_comments(best_keyword, limit_per_table=10)
        if comments:
            comment_dicts = [
                {"snippet": (c.content or "")[:300], "url": "", "title": ""}
                for c in comments
            ]
            comment_llm = classifier.classify_batch_llm(comment_dicts, query, llm)

            classified_comments = []
            for j, c in enumerate(comments):
                if comment_llm and j < len(comment_llm):
                    cstance, cconf = comment_llm[j]
                else:
                    cstance, cconf = classifier.classify(
                        {"snippet": (c.content or "")[:300], "url": "", "title": "",
                         "_target_stance": ""}, query,
                    )
                classified_comments.append({
                    "platform": c.platform,
                    "content": (c.content or "")[:200],
                    "like_count": c.like_count,
                    "publish_time": c.publish_time.isoformat() if c.publish_time else None,
                    "stance": cstance,
                })

            comment_sentiment = {
                "total": len(classified_comments),
                "distribution": _stance_distribution(classified_comments),
                "top_comments": sorted(
                    classified_comments, key=lambda x: x["like_count"], reverse=True,
                )[:10],
            }
    except Exception as exc:
        logger.warning(f"[SocialEnrichment] Comment analysis failed: {exc}")

    # -- Ext 2: Temporal sentiment tracking (reuse post classifications) --
    sentiment_trend = None
    try:
        time_buckets = db.search_with_time_buckets(best_keyword, days_back=7)
        if time_buckets:
            # Build a lookup from content -> stance using already-classified posts
            content_stance_map = {p["content"]: p["stance"] for p in classified_posts}

            trend_buckets = []
            for date_str in sorted(time_buckets.keys()):
                bucket_results = time_buckets[date_str]
                bucket_classified = []
                unmatched = []
                for r in bucket_results:
                    content_key = (r.title_or_content or "")[:300]
                    if content_key in content_stance_map:
                        bucket_classified.append({"stance": content_stance_map[content_key]})
                    else:
                        unmatched.append(
                            {"snippet": content_key, "url": "", "title": ""}
                        )

                # Only call LLM for posts not already classified
                if unmatched:
                    extra_llm = classifier.classify_batch_llm(unmatched, query, llm)
                    for k, ud in enumerate(unmatched):
                        if extra_llm and k < len(extra_llm):
                            bucket_classified.append({"stance": extra_llm[k][0]})
                        else:
                            s, _ = classifier.classify(
                                {**ud, "_target_stance": ""}, query,
                            )
                            bucket_classified.append({"stance": s})

                if bucket_classified:
                    trend_buckets.append({
                        "date": date_str,
                        "post_count": len(bucket_results),
                        "distribution": _stance_distribution(bucket_classified),
                    })

            direction = _detect_trend(trend_buckets)
            trend_summary = await _generate_trend_summary(query, trend_buckets, direction)

            sentiment_trend = {
                "buckets": trend_buckets,
                "trend_direction": direction,
                "trend_summary": trend_summary,
            }
    except Exception as exc:
        logger.warning(f"[SocialEnrichment] Temporal analysis failed: {exc}")

    # -- Assemble output --
    platforms_queried = list(probe_result.get("platforms", {}).keys())

    social_sentiment = {
        "mode": mode,
        "platforms_queried": platforms_queried,
        "total_posts": total_posts,
        "total_comments": sum(1 for _ in (comment_sentiment or {}).get("top_comments", [])),
        "sentiment_distribution": social_stance_dist,
        "per_platform": per_platform,
        "content_diversity": content_diversity,
        "low_diversity_warning": low_diversity_warning,
        "divergence_score": nsds,
        "divergence_summary": divergence_summary,
        "freshness_hours": round(freshness_hours or 0, 1),
        "top_social_voices": top_voices,
        # Full source-bound sample for EvidenceCore; top_social_voices remains
        # the compact presentation subset.
        "evidence_posts": classified_posts,
        "comment_sentiment": comment_sentiment,
        "sentiment_trend": sentiment_trend,
        "crawl_triggered": crawl_triggered,
    }

    trace = (
        f"[SocialEnrichment] mode={mode}, posts={total_posts}, "
        f"NSDS={nsds:.3f}, comments={comment_sentiment['total'] if comment_sentiment else 0}, "
        f"trend={sentiment_trend['trend_direction'] if sentiment_trend else 'N/A'}, "
        f"crawl_triggered={crawl_triggered}, platforms={platforms_queried}"
    )
    logger.info(trace)

    return {
        "mindspider_mode": mode,
        "social_sentiment": social_sentiment,
        "trace_log": [trace],
    }
