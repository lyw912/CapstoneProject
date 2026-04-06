"""
TrustScore 计算器

多维可信度评分（0-1），权重分配：
  - 域名权威性  30%：来源平台的公信力
  - 时效性      25%：信息新鲜度（指数衰减，7天半衰期）
  - 内容质量    25%：snippet 丰富度 + 是否有全文
  - 搜索排名    20%：搜索引擎的相关性得分

参考文献：架构文档 v2.0 Part 2 § 8.4
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from loguru import logger

# ---------------------------------------------------------------------------
# 域名权威性字典（精简版，完整版可在 utils/domain_authority.py 扩展）
# ---------------------------------------------------------------------------

DOMAIN_AUTHORITY: dict[str, float] = {
    # 中国官方 (0.90–1.00)
    "gov.cn":              1.00,
    "xinhua.net":          0.95,
    "people.com.cn":       0.95,
    "cctv.com":            0.90,
    "chinadaily.com.cn":   0.85,
    "mofcom.gov.cn":       0.90,
    "stats.gov.cn":        0.90,
    "moe.gov.cn":          0.90,
    "nhc.gov.cn":          0.90,
    "miit.gov.cn":         0.90,
    "pbc.gov.cn":          0.90,
    "ndrc.gov.cn":         0.90,
    # 中国主流媒体 (0.60–0.82)
    "thepaper.cn":         0.82,
    "caixin.com":          0.82,
    "ifeng.com":           0.75,
    "163.com":             0.65,
    "sina.com.cn":         0.65,
    "sohu.com":            0.60,
    "tencent.com":         0.65,
    "qq.com":              0.60,
    "36kr.com":            0.72,
    "huxiu.com":           0.72,
    "guancha.cn":          0.70,
    "jiemian.com":         0.72,
    "yicai.com":           0.72,
    "cls.cn":              0.72,
    "eastmoney.com":       0.68,
    "stcn.com":            0.68,
    "jrj.com.cn":          0.65,
    "zaobao.com":          0.70,
    # 国际权威媒体 (0.80–0.95)
    "reuters.com":         0.95,
    "bbc.com":             0.90,
    "bbc.co.uk":           0.90,
    "nytimes.com":         0.90,
    "wsj.com":             0.90,
    "bloomberg.com":       0.88,
    "ft.com":              0.88,
    "economist.com":       0.88,
    "theguardian.com":     0.85,
    "apnews.com":          0.92,
    "afp.com":             0.90,
    "cnbc.com":            0.80,
    "forbes.com":          0.75,
    "businessinsider.com": 0.70,
    "techcrunch.com":      0.72,
    "theverge.com":        0.72,
    # 国际官方机构 (0.90–1.00)
    "un.org":              0.98,
    "who.int":             0.98,
    "imf.org":             0.95,
    "worldbank.org":       0.95,
    "whitehouse.gov":      0.98,
    "gov.uk":              0.95,
    "europa.eu":           0.95,
    "senate.gov":          0.95,
    "state.gov":           0.95,
    # 学术 (0.80–0.92)
    "nature.com":          0.92,
    "science.org":         0.92,
    "arxiv.org":           0.85,
    "scholar.google.com":  0.80,
    "ssrn.com":            0.80,
    # 社交 / UGC (0.30–0.55)
    "weibo.com":           0.40,
    "zhihu.com":           0.52,
    "douban.com":          0.48,
    "reddit.com":          0.42,
    "twitter.com":         0.38,
    "x.com":               0.38,
    "bilibili.com":        0.48,
    "douyin.com":          0.32,
    "xiaohongshu.com":     0.35,
    "tieba.baidu.com":     0.42,
    "v2ex.com":            0.50,
    "hupu.com":            0.40,
}

# λ = ln(2) / 7 ≈ 0.099（7 天半衰期）
_DECAY_LAMBDA: float = math.log(2) / 7


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """从 URL 提取主域名（去除 www. 前缀）。"""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return ""


def _get_domain_authority(domain: str) -> float:
    """
    查询域名权威性得分（0.30–1.00）。
    支持后缀匹配：如 news.xinhua.net → xinhua.net → 0.95。
    """
    # 精确匹配
    if domain in DOMAIN_AUTHORITY:
        return DOMAIN_AUTHORITY[domain]

    # 后缀匹配（从最长后缀开始）
    parts = domain.split(".")
    for i in range(1, len(parts)):
        suffix = ".".join(parts[i:])
        if suffix in DOMAIN_AUTHORITY:
            return DOMAIN_AUTHORITY[suffix]

    return 0.30  # 未知域名默认


def _parse_date(date_str: str) -> Optional[datetime]:
    """
    解析日期字符串为 naive datetime。
    依次尝试：dateutil → fromisoformat → 失败返回 None。
    """
    if not date_str:
        return None
    try:
        from dateutil import parser as _du
        dt = _du.parse(date_str)
        return dt.replace(tzinfo=None)  # 统一为 naive
    except ImportError:
        pass
    except Exception:
        pass

    # 降级：Python 内置 fromisoformat（Python 3.7+）
    try:
        # 去掉时区后缀 Z / +HH:MM
        cleaned = date_str.rstrip("Z").split("+")[0].split("-")[0:3]
        # 简单处理，只取日期部分
        dt_str = date_str[:10]  # YYYY-MM-DD
        return datetime.strptime(dt_str, "%Y-%m-%d")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def compute_trust_score(source: dict) -> float:
    """
    综合可信度评分（0.0–1.0）。

    权重分配：
      域名权威性 30%  +  时效性 25%  +  内容质量 25%  +  搜索排名 20%

    Args:
        source: SourceItem 字典

    Returns:
        float 归一化到 [0.0, 1.0]，保留 3 位小数
    """
    score: float = 0.0

    url = source.get("url", "")
    domain = _extract_domain(url)

    # ------------------------------------------------------------------
    # 1. 域名权威性 (30%)
    # ------------------------------------------------------------------
    authority = _get_domain_authority(domain)
    score += 0.30 * authority

    # ------------------------------------------------------------------
    # 2. 时效性 (25%)：指数衰减，7 天半衰期
    # ------------------------------------------------------------------
    published = source.get("published_at")
    if published:
        dt = _parse_date(published)
        if dt is not None:
            days_old = max((datetime.now() - dt).days, 0)
            timeliness = math.exp(-_DECAY_LAMBDA * days_old)
            score += 0.25 * timeliness
        else:
            score += 0.125  # 无法解析日期，给一半
    else:
        score += 0.125  # 无日期信息，给一半

    # ------------------------------------------------------------------
    # 3. 内容质量 (25%)：snippet 长度 70% + 是否有全文 30%
    # ------------------------------------------------------------------
    snippet_len = len(source.get("snippet", "") or "")
    has_full = 1.0 if source.get("full_content") else 0.0
    content_quality = min(snippet_len / 400.0, 1.0) * 0.70 + has_full * 0.30
    score += 0.25 * content_quality

    # ------------------------------------------------------------------
    # 4. 搜索排名分 (20%)：来自搜索 API 的相关性得分
    # ------------------------------------------------------------------
    rrf = source.get("rrf_score") or 0.0
    try:
        rrf = float(rrf)
    except (TypeError, ValueError):
        rrf = 0.0

    if rrf > 0:
        score += 0.20 * min(rrf, 1.0)
    else:
        score += 0.10  # 无排名信息给一半

    return round(min(score, 1.0), 3)
