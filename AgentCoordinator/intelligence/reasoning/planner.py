"""Query understanding and claim-driven retrieval planning."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List

from ..contracts import Claim, RetrievalTask


def task_id(seed: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"rt_{digest}"


@dataclass
class QueryUnderstanding:
    query: str
    target_entity: str
    analysis_type: str
    key_terms: List[str]
    web_queries: List[str]
    platform_queries: List[str]
    official_queries: List[str]
    mode: str = "query"


class RetrievalPlanner:
    """Create retrieval tasks with explicit purpose, target source, and budget."""

    CROSS_LINGUAL_TOPICS = [
        {
            "english": ["rising price", "rising prices", "price increase", "inflation", "cost of living", "living cost", "consumer price"],
            "chinese": ["物价上涨", "涨价", "生活成本", "通胀"],
            "analysis_type": "policy",
        },
        {
            "english": ["flood", "flooding", "flood relief", "disaster response", "rainstorm", "heavy rain", "emergency response"],
            "chinese": ["洪水", "抗洪", "防汛", "暴雨", "救援"],
            "analysis_type": "event",
        },
        {
            "english": ["guangxi flood", "guangxi flooding", "guangxi disaster", "guangxi relief"],
            "chinese": ["广西洪水", "广西抗洪", "广西灾情", "广西救援"],
            "analysis_type": "event",
        },
        {
            "english": ["typhoon", "storm warning", "extreme weather"],
            "chinese": ["台风", "台风巴威", "极端天气"],
            "analysis_type": "event",
        },
        {
            "english": ["world cup", "football match", "soccer match", "national team"],
            "chinese": ["世界杯", "足球", "国家队"],
            "analysis_type": "event",
        },
        {
            "english": ["stock market", "a-share", "a shares", "market selloff", "securities regulator"],
            "chinese": ["A股", "股市", "证监会", "A股失守4000点"],
            "analysis_type": "policy",
        },
        {
            "english": ["tourism incentive", "tourism subsidy", "culture and tourism", "travel reward"],
            "chinese": ["文旅奖励", "西藏文旅奖励", "旅游补贴"],
            "analysis_type": "policy",
        },
        {
            "english": ["concert injury", "singer injury", "concert accident"],
            "chinese": ["演唱会受伤", "王力宏演唱会受伤"],
            "analysis_type": "event",
        },
        {
            "english": ["ai model", "large language model", "deepseek", "openai", "model release"],
            "chinese": ["AI模型", "大模型", "DeepSeek", "模型发布"],
            "analysis_type": "technology",
        },
    ]

    def understand(self, query: str) -> QueryUnderstanding:
        clean = " ".join(str(query or "").split())
        if not clean:
            clean = "public opinion topic"
        lowered = clean.lower()
        target = self._target_entity(clean)
        analysis_type = "general"
        if self._contains(lowered, ["brand", "company", "product", "service", "customer", "\u54c1\u724c", "\u516c\u53f8", "\u552e\u540e"]):
            analysis_type = "brand"
        elif self._contains(lowered, ["policy", "regulation", "government", "\u653f\u7b56", "\u76d1\u7ba1"]):
            analysis_type = "policy"
        elif self._contains(lowered, ["model", "ai", "deepseek", "openai", "technology", "\u6280\u672f", "\u6a21\u578b"]):
            analysis_type = "technology"
        elif self._contains(lowered, ["crisis", "incident", "scandal", "controversy", "\u7a81\u53d1", "\u4e8b\u6545", "\u4e89\u8bae"]):
            analysis_type = "event"
        topic_terms = self._topic_terms(clean)
        if analysis_type == "general" and topic_terms.get("analysis_type"):
            analysis_type = topic_terms["analysis_type"]
        platform_queries = self._platform_queries(clean, topic_terms["chinese"])
        web_queries = self._web_queries(clean, target, topic_terms["english"])
        official_queries = self._official_queries(clean, target, topic_terms["english"])
        return QueryUnderstanding(
            query=clean,
            target_entity=target,
            analysis_type=analysis_type,
            key_terms=self._key_terms(clean),
            web_queries=web_queries,
            platform_queries=platform_queries,
            official_queries=official_queries,
        )

    def initial_tasks(self, understanding: QueryUnderstanding) -> List[RetrievalTask]:
        query = understanding.query
        target = understanding.target_entity
        budget = {"max_api_calls": 1, "max_tokens": 0}
        return [
            RetrievalTask(
                task_id=task_id(f"{query}:observable_discourse"),
                parent_claim_id=None,
                query=query,
                query_variants=understanding.web_queries,
                target_source="web",
                purpose="support",
                priority=1,
                deadline_sec=20,
                max_results=8,
                budget={**budget, "max_api_calls": min(3, max(1, len(understanding.web_queries)))},
                created_by="retrieval_planner_v1",
            ),
            RetrievalTask(
                task_id=task_id(f"{query}:official_media"),
                parent_claim_id=None,
                query=understanding.official_queries[0],
                query_variants=understanding.official_queries,
                target_source="official",
                purpose="primary_source",
                priority=2,
                deadline_sec=20,
                max_results=5,
                budget={**budget, "max_api_calls": min(2, max(1, len(understanding.official_queries)))},
                created_by="retrieval_planner_v1",
            ),
        ]

    def social_platform_task(self, understanding: QueryUnderstanding) -> RetrievalTask:
        query = understanding.query
        target = understanding.target_entity
        return RetrievalTask(
            task_id=task_id(f"{query}:mindspider_platform_samples"),
            parent_claim_id=None,
            query=understanding.platform_queries[0] if understanding.platform_queries else query,
            query_variants=understanding.platform_queries or [query, f"{target} social discussion", f"{target} user reactions"],
            target_source="mindspider_db",
            purpose="social_context",
            priority=2,
            deadline_sec=20,
            max_results=8,
            budget={"max_api_calls": min(5, max(1, len(understanding.platform_queries))), "max_tokens": 0},
            created_by="retrieval_planner_v1",
        )

    def follow_up_tasks(self, claim: Claim, reason_codes: List[str]) -> List[RetrievalTask]:
        tasks: List[RetrievalTask] = []
        target = claim.target_entity
        official_followup_aspects = {"pricing", "customer_service", "reputation_risk", "policy", "safety"}
        if claim.aspect in official_followup_aspects and any(code in reason_codes for code in ["single_source", "ugc_only", "missing_official_response"]):
            readable_aspect = claim.aspect.replace("_", " ")
            if claim.aspect == "pricing":
                query = f"{target} API pricing official documentation"
                query_variants = [
                    query,
                    f"{target} pricing details API docs",
                    f"api-docs {target} pricing-details-usd",
                ]
            else:
                query = f"{target} official statement {readable_aspect}"
                query_variants = [query, f"{target} official source {readable_aspect}"]
            tasks.append(
                RetrievalTask(
                    task_id=task_id(f"{claim.claim_id}:official:{query}"),
                    parent_claim_id=claim.claim_id,
                    query=query,
                    query_variants=query_variants,
                    target_source="official",
                    purpose="primary_source",
                    priority=1,
                    deadline_sec=20,
                    max_results=5,
                    budget={"max_api_calls": 1, "max_tokens": 0},
                    created_by="research_router_v1",
                )
            )
        if any(code in reason_codes for code in ["one_sided", "high_copy_ratio", "disputed"]):
            query = f"{target} counter evidence independent sources {claim.aspect}"
            tasks.append(
                RetrievalTask(
                    task_id=task_id(f"{claim.claim_id}:counter:{query}"),
                    parent_claim_id=claim.claim_id,
                    query=query,
                    query_variants=[query, f"{target} dispute {claim.aspect}", f"{target} criticism evidence"],
                    target_source="web",
                    purpose="refute",
                    priority=2,
                    deadline_sec=20,
                    max_results=5,
                    budget={"max_api_calls": 1, "max_tokens": 0},
                    created_by="research_router_v1",
                )
            )
        return tasks

    @staticmethod
    def _contains(text: str, terms: List[str]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _target_entity(query: str) -> str:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.-]*|[\u4e00-\u9fff]{2,}", query)
        stop = {"public", "opinion", "discussion", "controversy", "analysis", "media", "coverage"}
        for token in tokens:
            if token.lower() not in stop:
                return token[:80]
        return query[:80]

    @staticmethod
    def _key_terms(query: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_.-]*|[\u4e00-\u9fff]{2,}", query)
        seen = []
        for token in tokens:
            normalized = token.lower()
            if normalized not in seen:
                seen.append(normalized)
        return seen[:12]

    @classmethod
    def _topic_terms(cls, query: str) -> dict:
        lowered = query.lower()
        if "deepseek" in lowered and (
            cls._contains(
                lowered,
                [
                    "api",
                    "price",
                    "pricing",
                    "cost",
                    "fee",
                    "fees",
                    "discount",
                    "surcharge",
                    "rate",
                ],
            )
            or any(term in query for term in ["价格", "定价", "涨价", "降价", "收费", "折扣"])
        ):
            return {
                "english": [
                    "DeepSeek peak hour API pricing June 2026",
                    "DeepSeek V4 API pricing adjustment June 2026",
                    "DeepSeek V4 peak off peak pricing",
                    "DeepSeek API pricing",
                    "DeepSeek price adjustment",
                    "DeepSeek V4 Pro price",
                ],
                "chinese": [
                    "DeepSeek API",
                    "DeepSeek V4",
                    "DeepSeek V4 Pro",
                    "DeepSeek API 价格",
                    "DeepSeek API 定价",
                    "DeepSeek 6月29日",
                    "DeepSeek 价格调整",
                    "DeepSeek 降价",
                    "DeepSeek 高峰期收费",
                ],
                "analysis_type": "technology",
            }
        english_terms: List[str] = []
        chinese_terms: List[str] = []
        analysis_type = ""
        for topic in cls.CROSS_LINGUAL_TOPICS:
            english_hits = [term for term in topic["english"] if term.lower() in lowered]
            chinese_hits = [term for term in topic["chinese"] if term in query]
            if not english_hits and not chinese_hits:
                continue
            english_terms.extend(english_hits or topic["english"][:2])
            chinese_terms.extend(chinese_hits or topic["chinese"][:4])
            analysis_type = analysis_type or str(topic.get("analysis_type") or "")
        return {
            "english": cls._dedupe(english_terms)[:6],
            "chinese": cls._dedupe(chinese_terms)[:8],
            "analysis_type": analysis_type,
        }

    @classmethod
    def _platform_queries(cls, query: str, topic_chinese_terms: List[str]) -> List[str]:
        chinese_phrases = re.findall(r"[\u4e00-\u9fff]{2,}", query)
        candidates = cls._dedupe(list(topic_chinese_terms) + chinese_phrases)
        if not candidates and any("\u4e00" <= char <= "\u9fff" for char in query):
            candidates = [query[:40]]
        return candidates[:6]

    @classmethod
    def _web_queries(cls, query: str, target: str, topic_english_terms: List[str]) -> List[str]:
        candidates = [query]
        for term in topic_english_terms[:3]:
            candidates.append(term)
        candidates.extend([
            f"{query} public reaction",
            f"{query} media coverage",
        ])
        for term in topic_english_terms[:3]:
            candidates.append(f"{term} public opinion China")
        if target and target.lower() not in {"public", "opinion", "discussion"}:
            candidates.append(f"{target} public discussion")
        return cls._dedupe(candidates)[:5]

    @classmethod
    def _official_queries(cls, query: str, target: str, topic_english_terms: List[str]) -> List[str]:
        lowered = query.lower()
        is_deepseek_pricing = "deepseek" in lowered and (
            cls._contains(lowered, ["api", "price", "pricing", "cost", "fee", "fees", "rate"])
            or any(term in query for term in ["价格", "定价", "收费", "折扣"])
        )
        if is_deepseek_pricing:
            return cls._dedupe([
                "DeepSeek API pricing details USD api-docs.deepseek.com",
                "DeepSeek API models pricing official docs",
                "api-docs.deepseek.com quick_start pricing",
                "platform.deepseek.com DeepSeek API pricing",
            ])[:4]

        candidates = [
            f"{query} official documentation",
            f"{query} official statement",
            f"{target} official source {query}",
        ]
        if cls._contains(lowered, ["controversy", "crisis", "scandal", "incident", "response"]):
            candidates.append(f"{target} official response")
        for term in topic_english_terms[:2]:
            candidates.append(f"{term} official documentation")
        return cls._dedupe(candidates)[:4]

    @staticmethod
    def _dedupe(values: List[str]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            text = " ".join(str(value or "").split())
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            result.append(text)
        return result
