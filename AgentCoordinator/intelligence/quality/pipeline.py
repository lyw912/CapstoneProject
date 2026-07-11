"""Deterministic quality pipeline and EvidenceGraph construction."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

from ..contracts import (
    CanonicalCluster,
    EvidenceGraph,
    EvidenceItem,
    FreshnessSummary,
    NormalizedItem,
    ProviderDiagnostic,
    QualityFeatures,
    SourceSpan,
)
from ..providers import SemanticQualityRouter


TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
DATE_RE = re.compile(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}:\d{2})\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
URL_RE = re.compile(r"https?://|www\.")

NEGATIVE_TERMS = {
    "complaint",
    "complaints",
    "risk",
    "negative",
    "too expensive",
    "overpriced",
    "broken",
    "bad",
    "poor",
    "worse",
    "terrible",
    "problematic",
    "failed",
    "failure",
    "delay",
    "delayed",
    "unusable",
    "not worth",
    "ripoff",
    "crazy pricing",
    "pricing is crazy",
    "争议",
    "投诉",
    "负面",
    "太贵",
    "真贵",
    "贵啊",
    "很贵",
    "割韭菜",
    "破防",
    "垃圾",
    "不值",
    "很差",
    "太差",
    "差劲",
    "很慢",
    "太慢",
    "卡顿",
    "崩了",
    "失败",
}
SUPPORT_TERMS = {
    "support",
    "defend",
    "normal",
    "resolved",
    "clearer",
    "positive",
    "satisfied",
    "cheaper",
    "faster",
    "efficient",
    "improved",
    "improvement",
    "advantage",
    "confidence",
    "支持",
    "正常",
    "等待",
    "认可",
    "有助",
    "提升",
    "信心",
    "优势",
    "效率",
    "优化",
    "理性",
    "省钱",
    "降本",
    "加速",
}
HELP_REQUEST_TERMS = {
    "help",
    "how to",
    "tutorial",
    "question",
    "anyone know",
    "求解",
    "请问",
    "请教",
    "怎么",
    "如何",
    "有没有",
    "大佬",
    "小白",
    "教程",
    "保姆级",
    "在哪",
    "怎么用",
    "怎么接",
    "缓存命中率",
    "命中率",
    "楼主你好",
}
MARKET_POSITIVE_TERMS = {
    "benefit",
    "efficiency",
    "advantage",
    "confidence",
    "optimization",
    "optimized",
    "cost saving",
    "有助",
    "提升",
    "信心",
    "投资建议",
    "竞争优势",
    "效率",
    "优化",
    "理性",
    "省钱",
    "降本",
    "加速",
}
PRICING_FACTUAL_TERMS = {
    "pricing",
    "price",
    "api pricing",
    "tokens",
    "token",
    "peak",
    "off-peak",
    "定价",
    "价格",
    "涨价",
    "降价",
    "翻倍",
    "峰谷",
    "高峰",
    "低峰",
    "收费",
    "计费",
    "token",
    "api",
}
LOW_EFFORT_TERMS = {
    "+1",
    "agree",
    "same",
    "forwarded",
    "\u652f\u6301",
    "\u540c\u610f",
    "\u8f6c\u53d1",
    "\u9876",
}
BOILERPLATE_TERMS = {
    "click here",
    "limited time",
    "subscribe",
    "follow us",
    "\u5e7f\u544a",
    "\u4f18\u60e0",
    "\u8f6c\u53d1\u62bd\u5956",
}


@dataclass
class QualityPipelineResult:
    graph: EvidenceGraph
    quality_summary: Dict[str, object]
    freshness_summary: FreshnessSummary
    provider_diagnostics: List[ProviderDiagnostic]


class QualityPipeline:
    """Build canonical clusters, quality features, evidence items, and spans."""

    def __init__(self, settings: Optional[object] = None):
        self.semantic_router = SemanticQualityRouter(settings=settings)

    def run(self, items: Sequence[NormalizedItem], query: str, target_entity: str) -> QualityPipelineResult:
        self.semantic_router.reset()
        semantic_vectors = self.semantic_router.embed_for_clustering(items)
        clusters = self._cluster(items, semantic_vectors=semantic_vectors)
        cluster_by_item = {
            item_id: cluster
            for cluster in clusters
            for item_id in cluster.member_item_ids
        }
        item_by_id = {item.item_id: item for item in items}

        quality_features = [
            self._quality_for_item(item, cluster_by_item[item.item_id], query, target_entity)
            for item in items
            if item.item_id in cluster_by_item
        ]
        self._apply_semantic_rerank(query, items, quality_features)

        evidence_items = self._build_evidence_items(clusters, item_by_id)
        graph = EvidenceGraph(
            normalized_items=list(items),
            quality_features=quality_features,
            canonical_clusters=clusters,
            evidence_items=evidence_items,
        )
        freshness_summary = self._freshness_summary(items)
        quality_summary = self._quality_summary(graph)
        return QualityPipelineResult(
            graph=graph,
            quality_summary=quality_summary,
            freshness_summary=freshness_summary,
            provider_diagnostics=list(self.semantic_router.diagnostics),
        )

    def merge(self, graph: EvidenceGraph, new_items: Sequence[NormalizedItem], query: str, target_entity: str) -> QualityPipelineResult:
        by_key: Dict[str, NormalizedItem] = {}
        for item in list(graph.normalized_items) + list(new_items):
            key = item.canonical_url or self._signature(item.text)
            by_key[key] = item
        return self.run(list(by_key.values()), query=query, target_entity=target_entity)

    def _cluster(self, items: Sequence[NormalizedItem], semantic_vectors: Optional[Dict[str, List[float]]] = None) -> List[CanonicalCluster]:
        semantic_vectors = semantic_vectors or {}
        grouped: List[List[NormalizedItem]] = []
        for item in items:
            signature = self._signature(item.text or item.title)
            placed = False
            for group in grouped:
                if self._same_cluster(
                    signature,
                    self._signature(group[0].text or group[0].title),
                    semantic_vectors.get(item.item_id),
                    semantic_vectors.get(group[0].item_id),
                ):
                    group.append(item)
                    placed = True
                    break
            if not placed:
                grouped.append([item])

        clusters: List[CanonicalCluster] = []
        for index, group in enumerate(grouped, start=1):
            representative = max(group, key=self._representative_score)
            first_seen = min((item.observed_at for item in group if item.observed_at), default=representative.observed_at)
            last_seen = max((item.observed_at for item in group if item.observed_at), default=representative.observed_at)
            cluster_type = "original" if len(group) == 1 else "copy_cascade"
            if len(group) > 1 and len({self._signature(item.text) for item in group}) > 1:
                cluster_type = "semantic_paraphrase"
            clusters.append(
                CanonicalCluster(
                    canonical_item_id=f"can_{index:04d}",
                    representative_item_id=representative.item_id,
                    member_item_ids=[item.item_id for item in group],
                    cluster_type=cluster_type,
                    amplification_count=len(group),
                    unique_author_count=len({item.author_id_hash or item.source_item_id for item in group}),
                    platforms=sorted({item.platform for item in group}),
                    first_seen_at=first_seen,
                    last_seen_at=last_seen,
                    representative_reason="highest deterministic informativeness and authority score",
                )
            )
        return clusters

    def _same_cluster(
        self,
        left_sig: str,
        right_sig: str,
        left_vector: Optional[List[float]] = None,
        right_vector: Optional[List[float]] = None,
    ) -> bool:
        if left_sig == right_sig:
            return True
        left_tokens = set(left_sig.split())
        right_tokens = set(right_sig.split())
        if not left_tokens or not right_tokens:
            return self.semantic_router.semantic_duplicate(left_vector, right_vector)
        overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
        return overlap >= 0.82 or self.semantic_router.semantic_duplicate(left_vector, right_vector)

    @staticmethod
    def _signature(text: str) -> str:
        tokens = TOKEN_RE.findall(str(text or "").lower())
        return " ".join(tokens[:120])

    def _quality_for_item(self, item: NormalizedItem, cluster: CanonicalCluster, query: str, target_entity: str) -> QualityFeatures:
        text = f"{item.title} {item.text}".strip()
        tokens = TOKEN_RE.findall(text.lower())
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        length_score = min(1.0, math.log1p(len(text)) / math.log(1200))
        specificity = self._specificity(text)
        boilerplate = self._boilerplate_ratio(text)
        relevance = self._relevance(text, query, target_entity)
        source_authority = self._source_authority(item.source_type, item.url)
        copy_ratio = (cluster.amplification_count - 1) / max(1, cluster.amplification_count)
        freshness = self._freshness_score(item.published_at or item.observed_at)
        informativeness = max(0.0, min(1.0, 0.34 * length_score + 0.22 * unique_ratio + 0.26 * specificity + 0.18 * source_authority - 0.18 * boilerplate))
        originality = max(0.05, 1.0 - copy_ratio)
        stance, stance_conf = self._stance(text, item.source_type)
        sentiment, sent_conf = self._sentiment(text, stance)
        aspect = self._aspect(text)
        coordination = self._coordination_score(cluster)
        evidence_support = 0.45 + min(0.35, specificity * 0.35)
        reasoning_clarity = min(1.0, 0.35 + informativeness * 0.55)
        novelty = originality
        persuasiveness = (
            0.20 * relevance
            + 0.15 * specificity
            + 0.15 * evidence_support
            + 0.15 * reasoning_clarity
            + 0.10 * novelty
            + 0.10 * source_authority
            + 0.10 * stance_conf
            - 0.15 * coordination
        )
        reasons = []
        if relevance < 0.28:
            reasons.append("low_relevance")
        if informativeness < 0.32:
            reasons.append("low_information")
        if copy_ratio >= 0.5:
            reasons.append("high_copy_ratio")
        if boilerplate > 0.25:
            reasons.append("boilerplate")
        if any(term in text.lower() for term in LOW_EFFORT_TERMS) and len(tokens) < 16:
            reasons.append("low_effort_support")
        dup_type = "none"
        if cluster.amplification_count > 1:
            dup_type = "semantic" if cluster.cluster_type == "semantic_paraphrase" else "near"
        return QualityFeatures(
            item_id=item.item_id,
            canonical_item_id=cluster.canonical_item_id,
            dup_type=dup_type,
            dup_confidence=0.0 if cluster.amplification_count == 1 else 0.86,
            amplification_count=cluster.amplification_count,
            copy_ratio_in_cluster=round(copy_ratio, 4),
            relevance_score=round(relevance, 4),
            informativeness_score=round(informativeness, 4),
            originality_score=round(originality, 4),
            source_authority_score=round(source_authority, 4),
            freshness_score=round(freshness, 4),
            stance=stance,
            stance_confidence=round(stance_conf, 4),
            sentiment=sentiment,
            sentiment_confidence=round(sent_conf, 4),
            aspect=aspect,
            coordination_score=round(coordination, 4),
            persuasiveness_score=round(max(0.0, min(1.0, persuasiveness)), 4),
            low_quality_reasons=reasons,
            judge_route="rule",
        )

    def _apply_semantic_rerank(self, query: str, items: Sequence[NormalizedItem], features: List[QualityFeatures]) -> None:
        provider, scores = self.semantic_router.rerank(query, items)
        if not scores:
            return
        feature_by_item = {feature.item_id: feature for feature in features}
        item_by_id = {item.item_id: item for item in items}
        for item_id, score in scores.items():
            feature = feature_by_item.get(item_id)
            if not feature:
                continue
            item = item_by_id.get(item_id)
            original_score = feature.relevance_score
            final_score = score
            if item and self._strong_query_context_match(query, f"{item.title} {item.text}"):
                final_score = max(score, original_score)
                feature.judge_route = f"{provider}_rerank+lexical_floor"
            else:
                feature.judge_route = f"{provider}_rerank"
            feature.relevance_score = round(final_score, 4)
            feature.persuasiveness_score = round(max(0.0, min(1.0, feature.persuasiveness_score * 0.75 + final_score * 0.25)), 4)
            if final_score < 0.28 and "low_relevance" not in feature.low_quality_reasons:
                feature.low_quality_reasons.append("low_relevance")
            if final_score >= 0.28 and "low_relevance" in feature.low_quality_reasons:
                feature.low_quality_reasons.remove("low_relevance")

    @staticmethod
    def _strong_query_context_match(query: str, text: str) -> bool:
        query_lower = str(query or "").lower()
        text_lower = str(text or "").lower()
        if "deepseek" in query_lower and "deepseek" not in text_lower:
            return False
        wants_api = "api" in query_lower
        wants_pricing = any(term in query_lower for term in ["pricing", "price", "cost", "收费", "价格", "定价"])
        has_api = "api" in text_lower
        has_pricing = QualityPipeline._term_hits(text, PRICING_FACTUAL_TERMS) > 0
        if wants_api and not has_api:
            return False
        if wants_pricing and not has_pricing:
            return False
        return wants_api or wants_pricing

    @staticmethod
    def _representative_score(item: NormalizedItem) -> float:
        text = f"{item.title} {item.text}"
        source = 0.3 if item.source_type in {"official", "mainstream_media"} else 0.0
        return min(1.0, len(text) / 800.0) + source + (0.1 if NUMBER_RE.search(text) else 0.0)

    @staticmethod
    def _specificity(text: str) -> float:
        score = 0.0
        if NUMBER_RE.search(text):
            score += 0.26
        if DATE_RE.search(text):
            score += 0.24
        if URL_RE.search(text):
            score += 0.18
        if any(mark in text.lower() for mark in ["according to", "statement", "ticket", "source", "reported"]):
            score += 0.18
        if len(TOKEN_RE.findall(text)) >= 30:
            score += 0.14
        return min(1.0, score)

    @staticmethod
    def _boilerplate_ratio(text: str) -> float:
        lowered = text.lower()
        hits = sum(1 for term in BOILERPLATE_TERMS if term in lowered)
        repeated_punct = len(re.findall(r"([!?.,])\1{2,}", text))
        return min(1.0, (hits * 0.25) + (repeated_punct * 0.08))

    @staticmethod
    def _relevance(text: str, query: str, target_entity: str) -> float:
        text_tokens = set(TOKEN_RE.findall(text.lower()))
        query_tokens = set(TOKEN_RE.findall(query.lower()))
        overlap = len(text_tokens & query_tokens) / max(1, len(query_tokens))
        target_bonus = 0.25 if target_entity and target_entity.lower() in text.lower() else 0.0
        return max(0.08, min(1.0, 0.35 + overlap * 0.45 + target_bonus))

    @staticmethod
    def _source_authority(source_type: str, url: str) -> float:
        if source_type == "official":
            return 0.92
        if source_type == "mainstream_media":
            return 0.78
        if source_type == "search_result":
            return 0.55
        if source_type == "ugc":
            return 0.42
        if source_type == "comment":
            return 0.32
        if source_type == "replay_fixture":
            return 0.18
        return 0.45

    @staticmethod
    def _freshness_score(value: Optional[str]) -> float:
        parsed = _parse_time(value)
        if not parsed:
            return 0.45
        age_hours = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 3600.0)
        return max(0.08, math.exp(-age_hours / (24.0 * 7.0)))

    @staticmethod
    def _term_hits(text: str, terms: Sequence[str]) -> int:
        lowered = text.lower()
        return sum(1 for term in terms if str(term).lower() in lowered)

    @staticmethod
    def _stance(text: str, source_type: str) -> Tuple[str, float]:
        if source_type == "official":
            return "official", 0.86

        neg = QualityPipeline._term_hits(text, NEGATIVE_TERMS)
        pos = QualityPipeline._term_hits(text, SUPPORT_TERMS)
        help_hits = QualityPipeline._term_hits(text, HELP_REQUEST_TERMS)
        market_pos = QualityPipeline._term_hits(text, MARKET_POSITIVE_TERMS)
        pricing_hits = QualityPipeline._term_hits(text, PRICING_FACTUAL_TERMS)

        if help_hits and neg <= 1:
            return "neutral", min(0.72, 0.54 + help_hits * 0.06)
        if market_pos and (pos + market_pos) >= neg:
            return "support", min(0.78, 0.52 + (pos + market_pos) * 0.06)
        if neg > pos + market_pos:
            return "oppose", min(0.86, 0.50 + neg * 0.10)
        if pos + market_pos > neg:
            return "support", min(0.82, 0.48 + (pos + market_pos) * 0.08)
        if pricing_hits:
            return "neutral", 0.55
        return "neutral", 0.45

    @staticmethod
    def _sentiment(text: str, stance: str) -> Tuple[str, float]:
        neg = QualityPipeline._term_hits(text, NEGATIVE_TERMS)
        pos = QualityPipeline._term_hits(text, SUPPORT_TERMS)
        help_hits = QualityPipeline._term_hits(text, HELP_REQUEST_TERMS)
        market_pos = QualityPipeline._term_hits(text, MARKET_POSITIVE_TERMS)
        pricing_hits = QualityPipeline._term_hits(text, PRICING_FACTUAL_TERMS)

        if help_hits and neg <= 1:
            return "neutral", min(0.72, 0.54 + help_hits * 0.05)
        if market_pos and (pos + market_pos) >= neg:
            return "positive", min(0.80, 0.52 + (pos + market_pos) * 0.06)
        if neg > pos + market_pos:
            return "negative", min(0.86, 0.48 + neg * 0.11)
        if pos + market_pos > neg:
            return "positive", min(0.82, 0.46 + (pos + market_pos) * 0.08)
        if stance == "official":
            return "neutral", 0.65
        if pricing_hits:
            return "neutral", 0.55
        return "neutral", 0.42

    @staticmethod
    def _aspect(text: str) -> str:
        lowered = text.lower()
        if QualityPipeline._term_hits(text, PRICING_FACTUAL_TERMS):
            return "pricing"
        if QualityPipeline._term_hits(text, HELP_REQUEST_TERMS):
            return "usage_help"
        if any(token in lowered for token in ["support", "ticket", "service", "customer", "客服", "售后"]):
            return "customer_service"
        if any(token in lowered for token in ["risk", "reputation", "trust", "声誉", "信任"]):
            return "reputation_risk"
        if any(token in lowered for token in ["evidence", "duplicate", "repost", "source", "证据", "转发"]):
            return "evidence_quality"
        return "general_discourse"

    @staticmethod
    def _coordination_score(cluster: CanonicalCluster) -> float:
        if cluster.amplification_count <= 1:
            return 0.0
        copy_component = min(1.0, (cluster.amplification_count - 1) / max(1, cluster.amplification_count))
        platform_component = 0.18 if len(cluster.platforms) > 1 else 0.0
        return min(1.0, copy_component * 0.78 + platform_component)

    def _build_evidence_items(self, clusters: List[CanonicalCluster], item_by_id: Dict[str, NormalizedItem]) -> List[EvidenceItem]:
        evidence_items: List[EvidenceItem] = []
        for index, cluster in enumerate(clusters, start=1):
            item = item_by_id.get(cluster.representative_item_id)
            if not item:
                continue
            evidence_id = f"ev_{index:04d}"
            span_text, start, end, span_type = self._source_span(item.text)
            span = SourceSpan(
                span_id=f"sp_{index:04d}",
                evidence_id=evidence_id,
                text=span_text,
                start_char=start,
                end_char=end,
                span_type=span_type,
                extraction_route="rule",
                confidence=0.78 if span_text else 0.0,
            )
            evidence_items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    item_id=item.item_id,
                    canonical_item_id=cluster.canonical_item_id,
                    source_type=item.source_type,
                    platform=item.platform,
                    title=item.title,
                    text=item.text,
                    url=item.url,
                    source_name=item.source_name,
                    published_at=item.published_at,
                    quality_ref=f"qf:{item.item_id}",
                    spans=[span] if span_text else [],
                    acquisition_source=item.acquisition_source,
                )
            )
        return evidence_items

    @staticmethod
    def _source_span(text: str) -> Tuple[str, int, int, str]:
        clean = " ".join(str(text or "").split())
        if not clean:
            return "", 0, 0, "opinion"
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        selected = max(sentences, key=lambda item: (len(TOKEN_RE.findall(item)), 1 if NUMBER_RE.search(item) else 0))
        selected = selected[:360]
        start = clean.find(selected)
        start = max(0, start)
        span_type = "number" if NUMBER_RE.search(selected) else "timeline" if DATE_RE.search(selected) else "opinion"
        return selected, start, start + len(selected), span_type

    def _quality_summary(self, graph: EvidenceGraph) -> Dict[str, object]:
        features = graph.quality_features
        raw_count = len(graph.normalized_items)
        canonical_count = len(graph.canonical_clusters)
        duplicate_count = max(0, raw_count - canonical_count)
        near_duplicate_count = sum(1 for feature in features if feature.dup_type not in {"none", "semantic"})
        semantic_duplicate_count = sum(1 for feature in features if feature.dup_type == "semantic")
        low_information = sum(1 for feature in features if "low_information" in feature.low_quality_reasons)
        low_relevance = sum(1 for feature in features if "low_relevance" in feature.low_quality_reasons)
        low_stance = sum(1 for feature in features if feature.stance_confidence < 0.5)
        high_coordination = any(feature.coordination_score >= 0.55 for feature in features)
        warnings = [
            "Repeated posts are treated as coverage strength, not independent viewpoints.",
            "Observable discourse is query-time or replay-backed; it is not population opinion.",
        ]
        if high_coordination:
            warnings.append("Near-duplicate clusters show possible coordinated repeated coverage.")
        return {
            "raw_count": raw_count,
            "normalized_count": raw_count,
            "canonical_count": canonical_count,
            "exact_duplicate_count": duplicate_count,
            "near_duplicate_count": near_duplicate_count,
            "semantic_duplicate_count": semantic_duplicate_count,
            "amplification_ratio": round(duplicate_count / max(1, raw_count), 4),
            "low_information_ratio": round(low_information / max(1, raw_count), 4),
            "low_relevance_ratio": round(low_relevance / max(1, raw_count), 4),
            "stance_low_confidence_ratio": round(low_stance / max(1, raw_count), 4),
            "coordination_warning": high_coordination,
            "quality_warnings": warnings,
        }

    @staticmethod
    def _freshness_summary(items: Sequence[NormalizedItem]) -> FreshnessSummary:
        published = [_parse_time(item.published_at) for item in items if item.published_at]
        published = [item for item in published if item]
        if not published:
            return FreshnessSummary(
                newest_published_at=None,
                oldest_published_at=None,
                median_age_hours=None,
                retrieval_lag_p95_sec=None,
                stale_source_ratio=0.0,
            )
        now = datetime.now(timezone.utc)
        ages = sorted(max(0.0, (now - value).total_seconds() / 3600.0) for value in published)
        stale = sum(1 for age in ages if age > 24 * 14)
        newest = max(published).isoformat().replace("+00:00", "Z")
        oldest = min(published).isoformat().replace("+00:00", "Z")
        return FreshnessSummary(
            newest_published_at=newest,
            oldest_published_at=oldest,
            median_age_hours=round(float(median(ages)), 2),
            retrieval_lag_p95_sec=None,
            stale_source_ratio=round(stale / max(1, len(ages)), 4),
        )


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None
