"""
HybridStanceClassifier — 混合立场分类器

分类策略（优先级从高到低）：
  1. 域名规则（置信度 0.90）：官方域名 → "official"
  2. 标题 + snippet 关键词匹配（置信度 0.50–0.85）
  3. 子查询弱标签（置信度 0.50）：使用发起该搜索的子查询的 target_stance
  4. 默认 → "neutral"（置信度 0.40）

立场分类体系：
  official    政府/官方媒体/企业官方的表态
  support     支持/正面评价
  oppose      反对/批评/负面评价
  neutral     中立分析/研究机构/客观评估
  background  背景信息/历史脉络/事件始末

参考文献：架构文档 v2.0 Part 2 § 8.5
"""

from __future__ import annotations

from typing import Tuple
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# 官方域名集合
# ---------------------------------------------------------------------------

OFFICIAL_DOMAINS: frozenset[str] = frozenset({
    # 中国官方
    "gov.cn", "xinhua.net", "people.com.cn", "cctv.com",
    "chinadaily.com.cn", "mofcom.gov.cn", "stats.gov.cn",
    "moe.gov.cn", "nhc.gov.cn", "miit.gov.cn", "pbc.gov.cn",
    "ndrc.gov.cn", "mps.gov.cn", "court.gov.cn",
    # 国际官方
    "gov.uk", "whitehouse.gov", "europa.eu", "un.org",
    "who.int", "imf.org", "worldbank.org", "state.gov",
    "senate.gov", "congress.gov",
})

# ---------------------------------------------------------------------------
# 关键词信号（中英混合）
# ---------------------------------------------------------------------------

# ---- 支持 / 正面 ----
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

# ---- 反对 / 负面 ----
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

# ---- 背景信息 ----
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
# 辅助
# ---------------------------------------------------------------------------

STANCES: tuple[str, ...] = ("official", "support", "oppose", "neutral", "background")


def _extract_domain(url: str) -> str:
    """从 URL 提取主域名（去除 www. 前缀）。"""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return ""


def _is_official_domain(domain: str) -> bool:
    """支持后缀匹配（如 news.xinhua.net → xinhua.net）。"""
    if domain in OFFICIAL_DOMAINS:
        return True
    parts = domain.split(".")
    for i in range(1, len(parts)):
        suffix = ".".join(parts[i:])
        if suffix in OFFICIAL_DOMAINS:
            return True
    return False


def _count_signals(text: str, signals_cn: frozenset, signals_en: frozenset) -> int:
    """统计中英文信号词在 text 中出现的次数。"""
    text_lower = text.lower()
    cn_hits = sum(1 for s in signals_cn if s in text)
    en_hits = sum(1 for s in signals_en if s in text_lower)
    return cn_hits + en_hits


# ---------------------------------------------------------------------------
# 分类器
# ---------------------------------------------------------------------------

class HybridStanceClassifier:
    """
    混合立场分类器。

    Phase 2 默认使用规则版（无 LLM 调用）。
    LLMStanceClassifier 预留给 Phase 3 的二次确认。
    """

    def classify(self, source: dict, query: str = "") -> Tuple[str, float]:
        """
        对单条来源进行立场分类。

        Args:
            source: SourceItem 字典，需含 url、title、snippet、_target_stance 字段
            query:  原始查询词（目前未使用，为 Phase 3 LLM 版预留）

        Returns:
            (stance_label, confidence)
            stance_label ∈ {"official","support","oppose","neutral","background"}
            confidence   ∈ [0.0, 1.0]
        """
        # ------------------------------------------------------------------
        # 第 1 层：官方域名规则（最高置信度）
        # ------------------------------------------------------------------
        domain = _extract_domain(source.get("url", ""))
        if _is_official_domain(domain):
            return "official", 0.90

        # ------------------------------------------------------------------
        # 第 2 层：关键词匹配
        # ------------------------------------------------------------------
        text = (
            (source.get("title") or "") + " "
            + (source.get("snippet") or "")
        )
        stance, confidence = self._keyword_classify(text)

        if confidence >= 0.65:
            return stance, confidence

        # ------------------------------------------------------------------
        # 第 3 层：子查询弱标签（来自 _target_stance 字段）
        # ------------------------------------------------------------------
        target_stance = (
            source.get("_target_stance")
            or source.get("sub_query_stance")
            or ""
        )
        if target_stance and target_stance in STANCES:
            # 若关键词有倾向但置信度不足，结合弱标签强化
            if stance == target_stance:
                return stance, min(confidence + 0.10, 0.64)
            return target_stance, 0.50

        # ------------------------------------------------------------------
        # 第 4 层：默认中立
        # ------------------------------------------------------------------
        return "neutral", 0.40

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_classify(text: str) -> Tuple[str, float]:
        """
        基于关键词匹配的立场判断。

        Returns:
            (stance, confidence)  confidence ∈ [0.40, 0.85]
        """
        sup = _count_signals(text, _SUPPORT_CN, _SUPPORT_EN)
        opp = _count_signals(text, _OPPOSE_CN,  _OPPOSE_EN)
        bg  = _count_signals(text, _BACKGROUND_CN, _BACKGROUND_EN)

        # 背景信息：至少 2 个信号词，且背景信号最多
        if bg >= 2 and bg >= sup and bg >= opp:
            return "background", min(0.50 + 0.08 * bg, 0.85)

        # 明显支持：正向信号比负向多 2 个以上
        if sup > opp + 1:
            return "support", min(0.50 + 0.08 * (sup - opp), 0.85)

        # 明显反对：负向信号比正向多 2 个以上
        if opp > sup + 1:
            return "oppose", min(0.50 + 0.08 * (opp - sup), 0.85)

        # 无明显倾向
        return "neutral", 0.45
