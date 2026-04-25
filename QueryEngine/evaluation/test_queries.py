"""
Standard Test Query Set (20 queries)

Covers 4 topic categories: Brand Events, Policy Sentiment, Social Hot Topics, Tech Topics.
Reference: Architecture Doc v2.0 Part 3 § 12.3
"""

from typing import List

TEST_QUERIES: List[dict] = [
    # ================================================================
    # Brand Events (Q01–Q05)
    # ================================================================
    {
        "id":    "Q01",
        "query": "DeepSeek发布新模型 各方舆论反应",
        "category": "brand",
        "note":  "Enterprise official + media + user reviews, relatively easy to cover",
    },
    {
        "id":    "Q02",
        "query": "瑞幸咖啡财务造假事件 各方立场",
        "category": "brand",
        "note":  "Enterprise response + investors + regulators + consumers",
    },
    {
        "id":    "Q03",
        "query": "华为Mate系列新品发布 舆论口碑",
        "category": "brand",
        "note":  "Positive reviews + negative complaints + comparative analysis",
    },
    {
        "id":    "Q04",
        "query": "拼多多平台商家投诉 消费者维权",
        "category": "brand",
        "note":  "Merchants + consumers + platform official + regulators",
    },
    {
        "id":    "Q05",
        "query": "蔚来汽车换电模式 用户评价争议",
        "category": "brand",
        "note":  "Owner feedback + enterprise response + industry analysis",
    },
    # ================================================================
    # Policy Sentiment (Q06–Q10)
    # ================================================================
    {
        "id":    "Q06",
        "query": "人工智能监管政策 各方观点",
        "category": "policy",
        "note":  "Government + enterprises + scholars + public",
    },
    {
        "id":    "Q07",
        "query": "北京限行政策 市民反应",
        "category": "policy",
        "note":  "Government interpretation + supporters + opponents",
    },
    {
        "id":    "Q08",
        "query": "新高考改革方案 各方态度",
        "category": "policy",
        "note":  "Ministry of Education + parents + teachers + students",
    },
    {
        "id":    "Q09",
        "query": "房产税试点政策 舆论讨论",
        "category": "policy",
        "note":  "Policy + support + opposition + expert analysis",
    },
    {
        "id":    "Q10",
        "query": "个人信息保护法实施 企业和用户反应",
        "category": "policy",
        "note":  "Regulation + enterprise compliance + user experience",
    },
    # ================================================================
    # Social Hot Topics (Q11–Q15)
    # ================================================================
    {
        "id":    "Q11",
        "query": "职场35岁危机 各方讨论",
        "category": "social",
        "note":  "Job seekers + enterprise HR + scholars + government",
    },
    {
        "id":    "Q12",
        "query": "大学生就业难 社会各界观点",
        "category": "social",
        "note":  "Graduates + enterprises + education department + economists",
    },
    {
        "id":    "Q13",
        "query": "教育内卷现象 家长和教育者争议",
        "category": "social",
        "note":  "Parents + teachers + policy + international comparison",
    },
    {
        "id":    "Q14",
        "query": "网约车司机权益保障 各方立场",
        "category": "social",
        "note":  "Drivers + platform + regulators + consumers",
    },
    {
        "id":    "Q15",
        "query": "短视频对青少年影响 社会争议",
        "category": "social",
        "note":  "Parents + educators + platform + researchers",
    },
    # ================================================================
    # Tech Topics (Q16–Q20)
    # ================================================================
    {
        "id":    "Q16",
        "query": "AI大模型替代就业 各方观点",
        "category": "tech",
        "note":  "Technology optimism + technology pessimism + economic analysis",
    },
    {
        "id":    "Q17",
        "query": "大模型安全风险 各方担忧和回应",
        "category": "tech",
        "note":  "Enterprises + scholars + regulators + users",
    },
    {
        "id":    "Q18",
        "query": "自动驾驶事故责任认定 争议",
        "category": "tech",
        "note":  "Automakers + legal + insurance + users",
    },
    {
        "id":    "Q19",
        "query": "元宇宙概念降温 投资者和企业反应",
        "category": "tech",
        "note":  "Supporters + skeptics + investors + technical analysis",
    },
    {
        "id":    "Q20",
        "query": "量子计算商业化进展 各方评估",
        "category": "tech",
        "note":  "Academia + enterprises + investment + technical assessment",
    },
]


def get_queries_by_category(category: str) -> List[dict]:
    """Get test queries by category."""
    return [q for q in TEST_QUERIES if q["category"] == category]


def get_query_by_id(qid: str) -> dict:
    """Get a single query by ID."""
    for q in TEST_QUERIES:
        if q["id"] == qid:
            return q
    raise KeyError(f"Query ID '{qid}' not found")
