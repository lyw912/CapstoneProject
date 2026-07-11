"""
Chinese social media platform profiles for Platform-Aware Deep Interpretation.

Each profile contains demographic data, content characteristics, bias tendencies,
and interpretation guidance for the deliberation engine.
"""

from typing import Dict, Any, Optional

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
    "kuaishou": {
        "display_name": "Kuaishou (快手)",
        "user_base": "Broad Chinese short-video audience with stronger lower-tier city coverage",
        "content_style": "Short video, live-commerce, everyday-life sharing",
        "bias_tendency": "Recommendation and creator-commerce incentives can amplify affective or promotional content",
        "weight_factor": 0.78,
        "demographic_note": "Can surface grassroots and lower-tier city reactions, but remains algorithmically filtered",
        "interpretation_template": (
            "Kuaishou's {stance} stance may reflect everyday-life and creator-community reactions. "
            "Consider: (1) algorithmic feed selection; "
            "(2) creator-commerce incentives; "
            "(3) stronger but still partial lower-tier city representation."
        ),
    },
    "reddit": {
        "display_name": "Reddit",
        "user_base": "Self-selected topic communities, heavily skewed by subreddit membership",
        "content_style": "Threaded discussion, anecdotes, technical and enthusiast communities",
        "bias_tendency": "Subreddit selection effects and voting dynamics can make visible comments unrepresentative",
        "weight_factor": 0.72,
        "demographic_note": "Represents English-language community discussion, not broad public opinion",
        "interpretation_template": (
            "Reddit's {stance} stance reflects self-selected community discussion. "
            "Consider: (1) subreddit-specific norms; "
            "(2) voting visibility effects; "
            "(3) English-language and enthusiast-community skew."
        ),
    },
    "twitter": {
        "display_name": "X / Twitter",
        "user_base": "Real-time public posters, journalists, creators, and highly active commentators",
        "content_style": "Short posts, quote-posts, real-time reaction chains",
        "bias_tendency": "High amplification, brigading, and elite-commentator effects; not population representative",
        "weight_factor": 0.70,
        "demographic_note": "Represents visible real-time discourse among active posters",
        "interpretation_template": (
            "X/Twitter's {stance} stance reflects visible real-time discourse. "
            "Consider: (1) amplification and quote-post cascades; "
            "(2) active-user and journalist/creator skew; "
            "(3) possible coordinated attention."
        ),
    },
    "youtube": {
        "display_name": "YouTube",
        "user_base": "Video viewers and creators, with strong topic-channel selection effects",
        "content_style": "Video titles/descriptions, comments, creator-led narratives",
        "bias_tendency": "Creator framing and recommendation effects can dominate visible reactions",
        "weight_factor": 0.68,
        "demographic_note": "Represents video-platform attention and creator/community framing",
        "interpretation_template": (
            "YouTube's {stance} stance reflects video-platform attention. "
            "Consider: (1) creator framing; "
            "(2) recommendation effects; "
            "(3) comments and viewership are not equivalent to population opinion."
        ),
    },
    "tiktok": {
        "display_name": "TikTok",
        "user_base": "Short-video audience with strong youth and algorithmic-feed effects",
        "content_style": "Short video, comments, trends, remix formats",
        "bias_tendency": "Highly algorithmic exposure; trend participation can overstate consensus",
        "weight_factor": 0.66,
        "demographic_note": "Represents algorithmically surfaced short-video reactions",
        "interpretation_template": (
            "TikTok's {stance} stance reflects algorithmically surfaced short-video reactions. "
            "Consider: (1) youth and creator skew; "
            "(2) trend mechanics; "
            "(3) personalized feed selection."
        ),
    },
    "instagram": {
        "display_name": "Instagram",
        "user_base": "Visual social network users, creators, brands, and lifestyle communities",
        "content_style": "Images, reels, stories, creator and brand posts",
        "bias_tendency": "Creator/brand incentives and visual presentation can distort issue salience",
        "weight_factor": 0.64,
        "demographic_note": "Represents visual-platform and creator-community reactions",
        "interpretation_template": (
            "Instagram's {stance} stance reflects visual-platform reactions. "
            "Consider: (1) creator/brand incentives; "
            "(2) lifestyle-community skew; "
            "(3) visibility is shaped by recommendation and follower networks."
        ),
    },
    "facebook": {
        "display_name": "Facebook",
        "user_base": "Broad but network/community-group mediated public and semi-public discussion",
        "content_style": "Posts, group discussion, pages, comments",
        "bias_tendency": "Group selection and page/community moderation affect visible sentiment",
        "weight_factor": 0.66,
        "demographic_note": "Represents visible page/group discussion rather than full population opinion",
        "interpretation_template": (
            "Facebook's {stance} stance reflects page and group-visible discussion. "
            "Consider: (1) group selection; "
            "(2) moderation and sharing dynamics; "
            "(3) private/networked discussion remains unobserved."
        ),
    },
}

SOCIAL_PLATFORM_ALIASES: Dict[str, str] = {
    "bilibili": "bilibili",
    "bilibili.com": "bilibili",
    "douyin": "douyin",
    "douyin.com": "douyin",
    "iesdouyin.com": "douyin",
    "kuaishou": "kuaishou",
    "kuaishou.com": "kuaishou",
    "xhs": "xhs",
    "xiaohongshu": "xhs",
    "xiaohongshu.com": "xhs",
    "xiaohongshu.cn": "xhs",
    "tieba": "tieba",
    "tieba.baidu.com": "tieba",
    "weibo": "weibo",
    "weibo.com": "weibo",
    "m.weibo.cn": "weibo",
    "zhihu": "zhihu",
    "zhihu.com": "zhihu",
    "reddit": "reddit",
    "reddit.com": "reddit",
    "old.reddit.com": "reddit",
    "x": "twitter",
    "x.com": "twitter",
    "twitter": "twitter",
    "twitter.com": "twitter",
    "t.co": "twitter",
    "youtube": "youtube",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "tiktok": "tiktok",
    "tiktok.com": "tiktok",
    "instagram": "instagram",
    "instagram.com": "instagram",
    "facebook": "facebook",
    "facebook.com": "facebook",
    "fb.com": "facebook",
}

SOCIAL_PLATFORM_KEYS = set(PLATFORM_PROFILES.keys())


def canonical_social_platform(value: str) -> Optional[str]:
    """Return a canonical platform key for known social domains/platform names."""
    key = str(value or "").strip().lower().replace("www.", "")
    if not key:
        return None
    if key in SOCIAL_PLATFORM_ALIASES:
        return SOCIAL_PLATFORM_ALIASES[key]
    parts = key.split(".")
    for index in range(len(parts)):
        candidate = ".".join(parts[index:])
        if candidate in SOCIAL_PLATFORM_ALIASES:
            return SOCIAL_PLATFORM_ALIASES[candidate]
    return None


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
    "kuaishou": 0.78,
    "reddit": 0.72,
    "twitter": 0.70,
    "youtube": 0.68,
    "tiktok": 0.66,
    "instagram": 0.64,
    "facebook": 0.66,
}
