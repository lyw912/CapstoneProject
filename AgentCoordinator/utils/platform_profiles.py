"""
Chinese social media platform profiles for Platform-Aware Deep Interpretation.

Each profile contains demographic data, content characteristics, bias tendencies,
and interpretation guidance for the deliberation engine.
"""

from typing import Dict, Any

PLATFORM_PROFILES: Dict[str, Dict[str, Any]] = {
    "weibo": {
        "display_name": "Weibo (微博)",
        "user_base": "General public, ages 20-40, urban Chinese",
        "content_style": "Short-form text, emotion-driven, trending-topic oriented",
        "bias_tendency": "Emotional amplification; high risk of astroturfing and marketing accounts",
        "weight_factor": 0.8,
        "demographic_note": "Represents immediate emotional reactions of the urban mass public",
        "interpretation_template": (
            "Weibo's {stance} stance likely reflects mass public's immediate emotional reaction. "
            "Consider: (1) potential astroturfing or viral marketing campaigns; "
            "(2) trending-topic algorithm amplification effects; "
            "(3) short-form format favors emotional expression over deliberation."
        ),
    },
    "zhihu": {
        "display_name": "Zhihu (知乎)",
        "user_base": "Highly educated (71.5% bachelor+), ages 25-45, knowledge workers",
        "content_style": "Long-form analysis, data citations, multi-angle argumentation",
        "bias_tendency": "Elite bias; perspectives may be disconnected from mass experience",
        "weight_factor": 1.0,
        "demographic_note": "Represents rational analysis of the educated elite; may miss grassroots sentiment",
        "interpretation_template": (
            "Zhihu's {stance} stance represents the educated elite's rational analysis. "
            "Consider: (1) knowledge-mass cognitive gap; "
            "(2) long-form discussions may over-theorize; "
            "(3) platform skews toward professionals and academics."
        ),
    },
    "bilibili": {
        "display_name": "Bilibili (B站)",
        "user_base": "Gen-Z (78.67% born 1990-2009), average age 22.8, urban youth",
        "content_style": "Danmaku (bullet comments), video responses, subculture communities",
        "bias_tendency": "Youth-centric bias; may neglect older generations' concerns",
        "weight_factor": 0.9,
        "demographic_note": "Represents Gen-Z attitudes; may signal future trends but skews very young",
        "interpretation_template": (
            "Bilibili's {stance} stance reflects Gen-Z attitudes and may signal future trends. "
            "Consider: (1) extreme age skew toward under-30; "
            "(2) subculture echo chambers within interest communities; "
            "(3) danmaku format encourages bandwagon/meme behavior."
        ),
    },
    "douyin": {
        "display_name": "Douyin (抖音)",
        "user_base": "Ages 18-35, urban, algorithm-driven content consumption",
        "content_style": "Short video, intuition-driven, heavily personalized feed",
        "bias_tendency": "Severe filter bubble from recommendation algorithm; highly polarized",
        "weight_factor": 0.75,
        "demographic_note": "Represents mass intuitive reactions but severely distorted by algorithm",
        "interpretation_template": (
            "Douyin's {stance} stance reflects mass intuitive reactions but is highly distorted by "
            "its recommendation algorithm. "
            "Consider: (1) individual feeds create extreme filter bubbles; "
            "(2) short video format strongly favors emotional/sensational content; "
            "(3) comment sections often unrepresentative of viewer distribution."
        ),
    },
    "xhs": {
        "display_name": "Xiaohongshu / Little Red Book (小红书)",
        "user_base": "70% female, ages 18-35, tier-1 and tier-2 city residents",
        "content_style": "UGC reviews, lifestyle content, consumer-oriented",
        "bias_tendency": "Female consumer bias; brand-sensitive; high commercial influence",
        "weight_factor": 0.85,
        "demographic_note": "Primarily represents female consumer perspectives in affluent urban areas",
        "interpretation_template": (
            "Xiaohongshu's {stance} stance predominantly represents female consumer perspectives. "
            "Consider: (1) strong gender and affluence skew (70% female, tier-1 cities); "
            "(2) commercial brand influence on user content; "
            "(3) consumer experience lens may not generalize to other demographics."
        ),
    },
    "tieba": {
        "display_name": "Baidu Tieba (贴吧)",
        "user_base": "Interest-driven communities, semi-anonymous, diverse ages",
        "content_style": "Niche interest discussion, semi-anonymous, long threads",
        "bias_tendency": "Niche community echo chambers; can be extreme/polarized within topics",
        "weight_factor": 0.8,
        "demographic_note": "Represents niche interest group deep attitudes; limited general representativeness",
        "interpretation_template": (
            "Tieba's {stance} stance reflects niche interest communities' deep attitudes. "
            "Consider: (1) semi-anonymity encourages more extreme expression; "
            "(2) forum communities are self-selected echo chambers; "
            "(3) interest-specific threads may have very limited general representativeness."
        ),
    },
}

# Source-level weight factors for divergence matrix computation
SOURCE_WEIGHTS = {
    "query_agent": 1.0,   # Web search + authoritative sources
    "media_agent": 0.9,   # Chinese media reporting
    "weibo": 0.8,
    "zhihu": 1.0,
    "bilibili": 0.9,
    "douyin": 0.75,
    "xhs": 0.85,
    "tieba": 0.8,
}
