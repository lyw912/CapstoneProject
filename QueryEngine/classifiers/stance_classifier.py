"""
HybridStanceClassifier — Hybrid Stance Classifier

Classification strategy (priority from high to low):
  1. Domain rules (confidence 0.90): official domains → "official"
  2. Title + snippet keyword matching (confidence 0.50–0.85)
  3. Sub-query weak labels (confidence 0.50): use target_stance from the sub-query that initiated this search
  4. Default → "neutral" (confidence 0.40)

Stance classification taxonomy:
  official    Government/official media/corporate official statements
  support     Support/positive evaluation
  oppose      Opposition/criticism/negative evaluation
  neutral     Neutral analysis/research institutions/objective assessment
  background  Background information/historical context/event timeline

Reference: Architecture Document v2.0 Part 2 § 8.5
"""

from __future__ import annotations

from typing import Tuple
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Official domain set
# ---------------------------------------------------------------------------

OFFICIAL_DOMAINS: frozenset[str] = frozenset({
    # Chinese Official
    "gov.cn", "xinhua.net", "people.com.cn", "cctv.com",
    "chinadaily.com.cn", "mofcom.gov.cn", "stats.gov.cn",
    "moe.gov.cn", "nhc.gov.cn", "miit.gov.cn", "pbc.gov.cn",
    "ndrc.gov.cn", "mps.gov.cn", "court.gov.cn",
    # International Official
    "gov.uk", "whitehouse.gov", "europa.eu", "un.org",
    "who.int", "imf.org", "worldbank.org", "state.gov",
    "senate.gov", "congress.gov",
})

# ---------------------------------------------------------------------------
# Keyword signals (Chinese and English mixed)
# ---------------------------------------------------------------------------

# ---- Support / Positive ----
_SUPPORT_CN = frozenset([
    "支持", "赞同", "赞赏", "好评", "认可", "肯定", "利好", "突破",
    "成功", "积极", "进展", "值得", "推荐", "点赞", "优秀", "领先",
    "创新", "振奋", "期待", "满意", "喜欢", "称赞", "表扬", "欢迎",
    "鼓励", "看好", "期待", "乐观", "利好", "红利", "福利", "获益",
])
_SUPPORT_EN = frozenset([
    "support", "praise", "positive", "excellent", "breakthrough",
    "success", "recommend", "impressive", "innovative", "promising",
    "welcome", "benefit", "favor", "endorse", "applaud",
])

# ---- Oppose / Negative ----
_OPPOSE_CN = frozenset([
    "反对", "质疑", "差评", "谴责", "投诉", "抗议", "担忧", "风险",
    "失败", "漏洞", "隐患", "批评", "指责", "争议", "负面", "不满",
    "抵制", "警告", "危险", "欺骗", "虚假", "违规", "处罚", "下架",
    "回收", "召回", "投诉", "举报", "封禁", "禁止", "限制", "整改",
])
_OPPOSE_EN = frozenset([
    "oppose", "criticize", "problem", "risk", "concern", "failure",
    "controversy", "negative", "protest", "warning", "danger",
    "ban", "fine", "penalty", "recall", "suspend", "reject",
    "complaint", "scandal", "fraud", "mislead",
])

# ---- Background Information ----
_BACKGROUND_CN = frozenset([
    "历史", "背景", "起因", "发展", "脉络", "回顾", "时间线",
    "经过", "始末", "由来", "演变", "梳理", "盘点", "前因",
    "溯源", "来龙去脉", "事件始末", "简史", "大事记",
])
_BACKGROUND_EN = frozenset([
    "history", "background", "origin", "timeline", "development",
    "evolution", "context", "overview", "chronicle", "retrospective",
])

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

STANCES: tuple[str, ...] = ("official", "support", "oppose", "neutral", "background")


def _extract_domain(url: str) -> str:
    """Extract primary domain from URL (removing www. prefix)."""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return ""


def _is_official_domain(domain: str) -> bool:
    """Support suffix matching (e.g., news.xinhua.net → xinhua.net)."""
    if domain in OFFICIAL_DOMAINS:
        return True
    parts = domain.split(".")
    for i in range(1, len(parts)):
        suffix = ".".join(parts[i:])
        if suffix in OFFICIAL_DOMAINS:
            return True
    return False


def _count_signals(text: str, signals_cn: frozenset, signals_en: frozenset) -> int:
    """Count occurrences of Chinese and English signal words in text."""
    text_lower = text.lower()
    cn_hits = sum(1 for s in signals_cn if s in text)
    en_hits = sum(1 for s in signals_en if s in text_lower)
    return cn_hits + en_hits


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class HybridStanceClassifier:
    """
    Hybrid stance classifier.

    Phase 2 uses rule-based version by default (no LLM calls).
    LLMStanceClassifier reserved for Phase 3 secondary verification.
    """

    def classify(self, source: dict, query: str = "") -> Tuple[str, float]:
        """
        Perform stance classification on a single source.

        Args:
            source: SourceItem dict, must contain url, title, snippet, _target_stance fields
            query:  Original query term (currently unused, reserved for Phase 3 LLM version)

        Returns:
            (stance_label, confidence)
            stance_label ∈ {"official","support","oppose","neutral","background"}
            confidence   ∈ [0.0, 1.0]
        """
        # ------------------------------------------------------------------
        # Layer 1: Official domain rules (highest confidence)
        # ------------------------------------------------------------------
        domain = _extract_domain(source.get("url", ""))
        if _is_official_domain(domain):
            return "official", 0.90

        # ------------------------------------------------------------------
        # Layer 2: Keyword matching
        # ------------------------------------------------------------------
        text = (
            (source.get("title") or "") + " "
            + (source.get("snippet") or "")
        )
        stance, confidence = self._keyword_classify(text)

        if confidence >= 0.65:
            return stance, confidence

        # ------------------------------------------------------------------
        # Layer 3: Sub-query weak labels (from _target_stance field)
        # ------------------------------------------------------------------
        target_stance = (
            source.get("_target_stance")
            or source.get("sub_query_stance")
            or ""
        )
        if target_stance and target_stance in STANCES:
            # If keywords show tendency but confidence is insufficient, strengthen with weak labels
            if stance == target_stance:
                return stance, min(confidence + 0.10, 0.64)
            return target_stance, 0.50

        # ------------------------------------------------------------------
        # Layer 4: Default neutral
        # ------------------------------------------------------------------
        return "neutral", 0.40

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_classify(text: str) -> Tuple[str, float]:
        """
        Stance judgment based on keyword matching.

        Returns:
            (stance, confidence)  confidence ∈ [0.40, 0.85]
        """
        sup = _count_signals(text, _SUPPORT_CN, _SUPPORT_EN)
        opp = _count_signals(text, _OPPOSE_CN,  _OPPOSE_EN)
        bg  = _count_signals(text, _BACKGROUND_CN, _BACKGROUND_EN)

        # Background information: at least 2 signal words, and background signals are dominant
        if bg >= 2 and bg >= sup and bg >= opp:
            return "background", min(0.50 + 0.08 * bg, 0.85)

        # Clear support: positive signals exceed negative by more than 2
        if sup > opp + 1:
            return "support", min(0.50 + 0.08 * (sup - opp), 0.85)

        # Clear opposition: negative signals exceed positive by more than 2
        if opp > sup + 1:
            return "oppose", min(0.50 + 0.08 * (opp - sup), 0.85)

        # No clear tendency
        return "neutral", 0.45
