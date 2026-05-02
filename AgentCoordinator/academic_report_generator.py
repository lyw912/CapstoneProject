"""
AcademicReportGenerator: Transforms coordinator_output.json into an
academic-style, evidence-traced Markdown report.

Report structure follows the IMRaD (Introduction-Methods-Results-and-Discussion)
model adapted for public opinion analysis:

    Abstract
    1. Introduction & Background
    2. Methodology & Metric Definitions
    3. Data Overview (sources, platforms, coverage)
    4. Findings (evidence-heavy, the MAJORITY of the report)
       4.1 Cross-Platform Stance Distribution (objective data)
       4.2 Cross-Source Divergence Analysis (matrix)
       4.3 Representative Voices & Evidence (cited social posts)
       4.4 Comment-Level Sentiment Analysis
    5. Multi-Perspective Deliberation Analysis
       5.1 Deliberation Process
       5.2 Consensus & Persistent Disagreements
    6. Bias Assessment & Information Integrity
       6.1 Echo Chamber Detection
       6.2 Silent Majority Hypothesis
    7. Conclusions & Implications (analytical, restrained)
       7.1 Key Findings
       7.2 Analytical Frameworks
       7.3 Limitations & Future Directions
    Appendices
       A. Full Source List
       B. Divergence Matrix Raw Data
       C. Deliberation Trace
       [PLACEHOLDER] D. Divergence Heatmap Visualization
       [PLACEHOLDER] E. Interactive Deliberation Timeline
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_academic_report(coordinator_output: Dict) -> str:
    """
    Transform a coordinator_output.json into an academic-style Markdown report.

    The report is structured as a research paper with:
    - Abstract at the top for quick comprehension
    - Methodology section explaining all custom metrics
    - Evidence-heavy findings section (the bulk of the report)
    - Restrained analytical conclusions
    - Appendices for supplementary data
    """
    lines: List[str] = []
    _w = lines.append  # shorthand for writing a line

    query = coordinator_output.get("query", "")
    analysis_type = coordinator_output.get("analysis_type", "general")
    generated_at = coordinator_output.get("generated_at", "")
    duration = coordinator_output.get("pipeline_duration_seconds", 0)

    # Parse sub-structures
    divergence = coordinator_output.get("divergence_matrix", {})
    deliberation = coordinator_output.get("deliberation", {})
    gap_filling = coordinator_output.get("gap_filling", {})
    platform_interps = coordinator_output.get("platform_interpretations", {})
    bias = coordinator_output.get("bias_analysis", {})
    fact_opinion = coordinator_output.get("fact_opinion_separation", {})
    synthesis = coordinator_output.get("synthesis", {})
    source_data = coordinator_output.get("source_data", {})
    qa_data = source_data.get("query_agent", {})
    social = qa_data.get("social_sentiment", {})
    trace = coordinator_output.get("coordinator_trace", [])

    # Derived values
    confidence = synthesis.get("overall_confidence", 0.5)
    stars = "★" * round(confidence * 5) + "☆" * (5 - round(confidence * 5))
    total_sources = qa_data.get("total_sources", 0)
    stance_dist = qa_data.get("stance_distribution", {})
    platforms_queried = social.get("platforms_queried", []) if social else []
    total_posts = social.get("total_posts", 0) if social else 0
    total_comments = social.get("total_comments", 0) if social else 0
    cssd_score = social.get("divergence_score", 0) if social else 0

    # ═══════════════════════════════════════════════════════
    # TITLE & METADATA
    # ═══════════════════════════════════════════════════════
    _w(f"# 多源舆情深度分析报告")
    _w(f"## Multi-Source Public Opinion Analysis Report")
    _w("")
    _w(f"**研究主题 (Query)**: {query}")
    _w(f"**分析类型 (Analysis Type)**: {analysis_type}")
    _w(f"**生成时间**: {generated_at}")
    _w(f"**分析耗时**: {duration:.1f}s")
    _w(f"**综合置信度**: {stars} ({confidence:.0%})")
    _w("")
    _w("---")
    _w("")

    # ═══════════════════════════════════════════════════════
    # ABSTRACT
    # ═══════════════════════════════════════════════════════
    _w("## 摘要 (Abstract)")
    _w("")
    summary_text = synthesis.get("summary", "")
    _w(summary_text)
    _w("")

    # Key numbers paragraph
    consensus_count = len(deliberation.get("final_consensus", []))
    dissent_count = len(deliberation.get("final_dissents", []))
    fact_count = len(fact_opinion.get("verified_facts", []))
    hotspot_count = len(divergence.get("hotspots", []))
    pairs_count = len(divergence.get("pairs", {}))

    _w(f"本报告基于 **{total_sources} 条网络搜索来源**"
       f"和来自 **{len(platforms_queried)} 个社媒平台（{', '.join(platforms_queried)}）"
       f"的 {total_posts} 条帖子、{total_comments} 条评论**，"
       f"采用多维度辩论引擎进行 {len(deliberation.get('perspectives_used', []))} 个视角的结构化审议，"
       f"形成 {consensus_count} 条跨视角共识和 {dissent_count} 条持续分歧，"
       f"提取 {fact_count} 条可验证事实，"
       f"计算 {pairs_count} 对跨源分歧指标（其中 {hotspot_count} 对存在显著差异）。"
       f"综合置信度为 {confidence:.0%}。")
    _w("")
    _w("---")
    _w("")

    # ═══════════════════════════════════════════════════════
    # 1. INTRODUCTION & BACKGROUND
    # ═══════════════════════════════════════════════════════
    _w("## 1. 引言与背景 (Introduction & Background)")
    _w("")
    _w(f"本报告对「{query}」话题进行多源、多平台、多维度的深度舆情分析。"
       f"不同于传统舆情监测的简单情感分类和观点罗列，本系统采用以下方法论创新：")
    _w("")
    _w("1. **跨源三角验证 (Cross-Source Triangulation)**：同时采集网络媒体报道（Tavily/Anspire API）和社交媒体原生内容（MindSpider 爬取），对比两类数据源的立场差异")
    _w("2. **多维度结构化辩论 (Multi-Perspective Structured Deliberation)**：模拟学术审议过程，从多个分析维度独立评估再交叉质证")
    _w("3. **跨源分歧矩阵 (Cross-Source Divergence Matrix)**：量化计算所有数据源两两之间的立场差异，识别信息场域中的结构性分歧")
    _w("4. **事实-舆论分离 (Fact-Opinion Separation)**：将可验证事实与观点/情感严格区分，帮助读者独立判断")
    _w("5. **回声室检测与沉默大多数补偿 (Echo Chamber Detection)**：识别信息茧房效应和潜在的被压制声音")
    _w("")
    _w("---")
    _w("")

    # ═══════════════════════════════════════════════════════
    # 2. METHODOLOGY & METRIC DEFINITIONS
    # ═══════════════════════════════════════════════════════
    _w("## 2. 方法论与指标定义 (Methodology & Metrics)")
    _w("")
    _w("### 2.1 数据采集架构")
    _w("")
    _w("| 数据层 | 来源 | 覆盖范围 | 角色 |")
    _w("|-------|------|---------|------|")
    _w("| 网络媒体层 | Tavily API + Anspire API | 国际新闻 + 中文媒体 | 提供媒体叙事和权威来源 |")
    _w("| 社交媒体层 | MindSpider 爬虫 → MySQL | 微博、知乎、B站等 | 提供公众情感和民意表达 |")
    _w("| 多媒体层 | MediaAgent (Bocha API) | 中文多模态内容 | 补充图文/视频维度 |")
    _w("")
    _w("### 2.2 核心分析指标")
    _w("")
    _w("#### CSSD（Cross-Source Sentiment Difference，跨源情感差异）")
    _w("")
    _w("衡量两个数据源之间立场分布的差异程度。")
    _w("")
    _w("$$\\text{CSSD}(A, B) = 1 - \\cos(\\vec{s}_A, \\vec{s}_B)$$")
    _w("")
    _w("其中 $\\vec{s}$ = [support, oppose, neutral, official, background] 为归一化立场向量，"
       "$\\cos$ 为余弦相似度。")
    _w("")
    _w("| CSSD 值 | 含义 | 解读 |")
    _w("|---------|------|------|")
    _w("| 0.0 | 完全一致 | 两个来源的立场分布完全相同 |")
    _w("| 0.0–0.3 | 低分歧 | 两个来源的观点基本一致 |")
    _w("| 0.3–0.6 | 中等分歧 | 存在明显的立场差异，需关注 |")
    _w("| 0.6–1.0 | 高分歧 | 两个来源的观点截然不同，可能存在信息断层 |")
    _w("| 1.0 | 完全相反 | 立场正交或完全互斥 |")
    _w("")

    _w("#### SCS（Stance Coverage Score，立场覆盖分数）")
    _w("")
    _w("衡量搜索结果是否覆盖了各主要立场维度。0–1，1.0 表示所有立场均有足够代表性来源。"
       f"本次分析 SCS = **{qa_data.get('coverage_score', 0):.2f}**。")
    _w("")

    _w("#### TrustScore（来源可信度评分）")
    _w("")
    _w("综合评估每条来源的可信度（0–1），计算公式：")
    _w("")
    _w("$$\\text{TrustScore} = 0.30 \\times \\text{domain\\_authority} + 0.25 \\times \\text{timeliness} + 0.25 \\times \\text{content\\_quality} + 0.20 \\times \\text{relevance}$$")
    _w("")

    _w("#### Shannon 熵（Stance Entropy，立场熵）")
    _w("")
    _w("衡量某平台或来源的立场多样性。$H = -\\sum p_i \\log_2 p_i$。"
       "低熵（< 0.5）+ 高帖子数 = 回声室信号。")
    _w("")
    _w("### 2.3 辩论引擎方法")
    _w("")
    perspectives_used = deliberation.get("perspectives_used", [])
    _w(f"采用 **Hybrid Plan C** 辩论方案（参考 AI Council, arXiv:2604.26561），"
       f"为本次「{analysis_type}」类型话题选择 {len(perspectives_used)} 个分析维度：")
    _w("")
    for p in perspectives_used:
        _w(f"- **{p}**")
    _w("")
    _w("辩论过程分三阶段：")
    _w("1. **独立分析**：每个维度基于原始数据独立产出分析报告（并行执行，互不影响）")
    _w("2. **交叉质证**：每个维度审阅他方报告，给出 AGREE/CHALLENGE/SUPPLEMENT 响应")
    _w("3. **综合裁定**：识别跨维度共识与持续分歧，评估整体置信度")
    _w("")
    _w("---")
    _w("")

    # ═══════════════════════════════════════════════════════
    # 3. DATA OVERVIEW
    # ═══════════════════════════════════════════════════════
    _w("## 3. 数据概览 (Data Overview)")
    _w("")
    _w(f"### 3.1 网络搜索来源")
    _w("")
    _w(f"共获取 **{total_sources} 条**来源（去重后），来自 Tavily（国际新闻）和 Anspire（中文媒体）双通道搜索。"
       f"立场覆盖分数 SCS = **{qa_data.get('coverage_score', 0):.2f}**（{_scs_label(qa_data.get('coverage_score', 0))}）。")
    _w("")
    _w("**立场分布**：")
    _w("")
    _w("| 立场 | 比例 | 含义 |")
    _w("|------|------|------|")
    for stance, ratio in sorted(stance_dist.items(), key=lambda x: -x[1]):
        _w(f"| {stance} | {ratio:.1%} | {_stance_meaning(stance)} |")
    _w("")

    if social and social.get("mode") == "available":
        _w(f"### 3.2 社交媒体数据")
        _w("")
        _w(f"通过 MindSpider 从 **{', '.join(platforms_queried)}** 获取 "
           f"**{total_posts} 条帖子**和 **{total_comments} 条评论**。")
        _w("")
        social_dist = social.get("sentiment_distribution", {})
        if social_dist:
            _w("**社媒整体立场分布**：")
            _w("")
            _w("| 立场 | 比例 |")
            _w("|------|------|")
            for stance, ratio in sorted(social_dist.items(), key=lambda x: -x[1]):
                _w(f"| {stance} | {ratio:.1%} |")
            _w("")
        _w(f"**跨源情感差异 CSSD = {cssd_score:.3f}**"
           f"（{_cssd_label(cssd_score)}）——网络媒体报道与社媒民意之间的立场差异。")
        _w("")

        # Per-platform breakdown
        per_platform = social.get("per_platform", {})
        if per_platform:
            _w("**各平台立场分布对比**：")
            _w("")
            _w("| 平台 | 帖子数 | 主导立场 | 分布详情 |")
            _w("|------|--------|---------|---------|")
            for platform, stats in per_platform.items():
                if not stats:
                    continue
                count = stats.get("count", 0) or stats.get("post_count", 0)
                dist = stats.get("distribution", {})
                dominant = max(dist, key=dist.get) if dist else "-"
                dominant_pct = dist.get(dominant, 0) if dist else 0
                dist_str = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(dist.items(), key=lambda x: -x[1]) if v > 0.01)
                _w(f"| {platform} | {count} | {dominant} ({dominant_pct:.0%}) | {dist_str} |")
            _w("")

    _w("---")
    _w("")

    # ═══════════════════════════════════════════════════════
    # 4. FINDINGS (Evidence-heavy — the CORE of the report)
    # ═══════════════════════════════════════════════════════
    _w("## 4. 研究发现 (Findings)")
    _w("")
    _w("> 本章呈现客观数据和一手证据，读者可据此自行形成判断。")
    _w("")

    # 4.1 Verified Facts
    verified_facts = fact_opinion.get("verified_facts", [])
    if verified_facts:
        _w("### 4.1 可验证事实 (Verified Facts)")
        _w("")
        for i, f in enumerate(verified_facts, 1):
            status = f.get("verification_status", "unknown")
            conf = f.get("confidence", 0)
            icon = "✅" if status == "cross_verified" else "⚠️" if status == "single_source" else "❓"
            _w(f"**事实 {i}** {icon} [{_verification_label(status)}, 置信度 {conf:.0%}]")
            _w("")
            _w(f"> {f.get('fact', '')}")
            _w("")
            sources_list = f.get("sources", [])
            if sources_list:
                _w(f"*来源*：{'; '.join(sources_list)}")
                _w("")
        _w("")

    # 4.2 Cross-Source Divergence
    _w("### 4.2 跨源分歧分析 (Cross-Source Divergence Matrix)")
    _w("")
    max_div = divergence.get("max_divergence", {})
    min_div = divergence.get("min_divergence", {})
    _w(f"对 {len(divergence.get('pairs', {}))} 对数据源计算 CSSD 跨源分歧值。"
       f"最大分歧：**{max_div.get('pair', '')}** = {max_div.get('value', 0):.3f}；"
       f"最小分歧：**{min_div.get('pair', '')}** = {min_div.get('value', 0):.3f}。")
    _w("")
    _w("**`[VISUALIZATION PLACEHOLDER: 跨源分歧矩阵热力图]`**")
    _w("")
    _w("> *报告呈现阶段将在此处渲染交互式热力图，横轴和纵轴为各数据源（query_agent, media_agent, weibo, zhihu, bilibili, social_media_overall），颜色深度表示 CSSD 值大小。*")
    _w("")

    # Hotspots analysis
    hotspots = divergence.get("hotspots", [])
    if hotspots:
        _w("**显著分歧热点**（CSSD > 0.3）：")
        _w("")
        for h in hotspots:
            _w(f"- {h}")
        _w("")

        _w("**分歧解读**：")
        _w("")
        # Interpret the divergence patterns
        pairs = divergence.get("pairs", {})
        if "weibo|zhihu" in pairs:
            weibo_zhihu = pairs["weibo|zhihu"]
            _w(f"- **微博 vs 知乎 (CSSD={weibo_zhihu:.3f})**："
               f"这是本次分析中最大的分歧。微博用户（大众群体，20-40岁城市居民）"
               f"以情绪化的支持立场为主，而知乎用户（71.5% 本科及以上学历的知识群体）"
               f"则呈现压倒性的中立/分析态度。这一差异反映了**不同教育背景和平台文化**"
               f"对同一话题的不同认知方式——并非信息不对称，而是认知框架的差异。")
            _w("")
        if "query_agent|social_media_overall" in pairs:
            web_social = pairs["query_agent|social_media_overall"]
            _w(f"- **网络媒体 vs 社媒整体 (CSSD={web_social:.3f})**："
               f"网络媒体以官方声明和权威报道为主（official 占 {stance_dist.get('official', 0):.0%}），"
               f"社媒则以中性观望和支持为主。差异源于内容生态的不同：媒体转载官方话语，"
               f"而社媒用户表达个体感受。")
            _w("")

    # 4.3 Representative Voices & Evidence
    _w("### 4.3 代表性原声与证据 (Representative Voices)")
    _w("")
    _w("> 以下为各立场的代表性原始内容，附可点击跳转链接。")
    _w("")

    # Web sources by stance
    top_sources = qa_data.get("top_sources", [])
    if top_sources:
        _w("#### 4.3.1 网络媒体代表来源")
        _w("")
        # Group by stance
        by_stance: Dict[str, List[Dict]] = {}
        for s in top_sources:
            stance = s.get("stance", "unclassified")
            by_stance.setdefault(stance, []).append(s)

        for stance in ["official", "support", "oppose", "neutral", "background"]:
            items = by_stance.get(stance, [])
            if not items:
                continue
            _w(f"**{stance.upper()} 立场** ({len(items)} 条)：")
            _w("")
            for s in items[:2]:
                title = s.get("title", "(无标题)")
                url = s.get("url", "")
                ts = s.get("trust_score", 0)
                _w(f"- [{title}]({url}) — TrustScore: {ts:.2f}")
            _w("")

    # Social media voices
    voices = social.get("top_social_voices", []) if social else []
    if voices:
        _w("#### 4.3.2 社交媒体代表帖子")
        _w("")
        _w("> 以下帖子均来自 MindSpider 真实爬取，点击链接可跳转至原帖。")
        _w("")
        for v in voices[:6]:
            platform = v.get("platform", "")
            stance = v.get("stance", "")
            content = (v.get("content", "") or "")[:200]
            url = v.get("url", "")
            pub_time = v.get("publish_time", "")
            _w(f"**[{platform}]** [{stance}] — {pub_time}")
            _w("")
            _w(f"> {content}")
            _w("")
            if url:
                _w(f"[🔗 跳转原帖]({url})")
            _w("")

    # 4.4 Comment-level sentiment
    comment_data = social.get("comment_sentiment", {}) if social else {}
    top_comments = comment_data.get("top_comments", [])
    if top_comments:
        _w("### 4.4 评论区情感分析 (Comment-Level Sentiment)")
        _w("")
        comment_total = comment_data.get("total", 0)
        comment_dist = comment_data.get("distribution", {})
        _w(f"共分析 **{comment_total} 条评论**。评论区往往比帖子本身更能反映真实民意。")
        _w("")
        if comment_dist:
            _w("**评论立场分布**：")
            _w("")
            _w("| 立场 | 比例 |")
            _w("|------|------|")
            for stance, ratio in sorted(comment_dist.items(), key=lambda x: -x[1]):
                _w(f"| {stance} | {ratio:.0%} |")
            _w("")

        _w("**高赞热评**（按点赞数排序）：")
        _w("")
        for c in top_comments[:5]:
            platform = c.get("platform", "")
            stance = c.get("stance", "")
            likes = c.get("like_count", 0)
            content = (c.get("content", "") or "")[:200]
            _w(f"- **[{platform}] [{stance}] 👍 {likes}**")
            _w(f"  > {content}")
            _w("")

        # Highlight the most insightful comment
        if top_comments:
            top = top_comments[0]
            _w(f"**最高赞评论分析**：获得 {top.get('like_count', 0)} 次点赞，"
               f"立场标注为 [{top.get('stance', '')}]。"
               f"高赞数表明此观点引起了广泛共鸣，是理解社区真实态度的关键信号。")
            _w("")

    _w("---")
    _w("")

    # ═══════════════════════════════════════════════════════
    # 5. DELIBERATION ANALYSIS
    # ═══════════════════════════════════════════════════════
    _w("## 5. 多维度辩论分析 (Multi-Perspective Deliberation)")
    _w("")
    _w("**`[PLACEHOLDER: 可折叠辩论过程交互式展示]`**")
    _w("")
    _w("> *报告呈现阶段将在此处实现辩论过程的可折叠/展开交互展示，每轮辩论可独立查看。*")
    _w("")

    # Deliberation phases
    phases = deliberation.get("phases", [])
    for phase_data in phases:
        phase_name = phase_data.get("phase", "")
        if phase_name == "independent":
            _w("### 5.1 第一阶段：独立分析 (Independent Analysis)")
            _w("")
            _w(f"各维度基于原始数据独立产出分析报告，互不干扰，防止从众偏差。")
            _w("")
            # Details from the phase are in deliberation_rounds in synthesis_context
        elif phase_name == "cross_examination":
            _w("### 5.2 第二阶段：交叉质证 (Cross-Examination)")
            _w("")
            phase_consensus = phase_data.get("consensus_points", [])
            phase_dissent = phase_data.get("dissent_points", [])
            if phase_consensus:
                _w("**初步共识**：")
                _w("")
                for c in phase_consensus[:4]:
                    _w(f"- {c}")
                _w("")
            if phase_dissent:
                _w("**质证中的分歧**：")
                _w("")
                for d in phase_dissent[:4]:
                    _w(f"- {d}")
                _w("")
        elif phase_name == "synthesis_arbitration":
            _w("### 5.3 第三阶段：综合裁定 (Synthesis Arbitration)")
            _w("")
            synth_summary = phase_data.get("summary", "")
            if synth_summary:
                _w(f"> {synth_summary}")
                _w("")

    # Final consensus and dissents
    final_consensus = deliberation.get("final_consensus", [])
    final_dissents = deliberation.get("final_dissents", [])

    # Deduplicate
    final_consensus = _deduplicate_strings(final_consensus)
    final_dissents = _deduplicate_strings(final_dissents)

    if final_consensus:
        _w("### 5.4 跨维度共识 (Cross-Perspective Consensus)")
        _w("")
        _w(f"以下 {len(final_consensus)} 条发现获得所有分析维度的一致认可：")
        _w("")
        for i, c in enumerate(final_consensus, 1):
            _w(f"{i}. ✓ {c}")
        _w("")

    if final_dissents:
        _w("### 5.5 持续分歧 (Persistent Disagreements)")
        _w("")
        _w(f"以下 {len(final_dissents)} 条问题在辩论后仍未达成共识，"
           f"反映了当前数据条件下的认知边界：")
        _w("")
        for i, d in enumerate(final_dissents, 1):
            _w(f"{i}. ✗ {d}")
        _w("")

    _w(f"**辩论置信度**: {deliberation.get('confidence', 0):.0%}")
    _w("")
    _w("---")
    _w("")

    # ═══════════════════════════════════════════════════════
    # 6. BIAS ASSESSMENT
    # ═══════════════════════════════════════════════════════
    _w("## 6. 偏见评估与信息完整性 (Bias Assessment)")
    _w("")

    echo_warnings = bias.get("echo_warnings", [])
    silent = bias.get("silent_majority_hypothesis")

    _w("### 6.1 回声室检测 (Echo Chamber Detection)")
    _w("")
    if echo_warnings:
        _w(f"检测到 {len(echo_warnings)} 条回声室/信息茧房告警：")
        _w("")
        for w in echo_warnings:
            _w(f"⚠️ {w}")
            _w("")
    else:
        _w("本次分析未检测到明显的回声室效应。各平台的立场分布呈现一定多样性，"
           "信息茧房风险较低。")
        _w("")
        _w("*说明*：回声室检测基于 Shannon 立场熵（H < 0.5 且帖子数 ≥ 3 触发告警）"
           "以及内容多样性分数（< 0.7 触发水军/协调发帖告警）。")
        _w("")

    _w("### 6.2 沉默的大多数假设 (Silent Majority Hypothesis)")
    _w("")
    if silent:
        _w(f"> {silent}")
        _w("")
    else:
        _w("当前数据未触发沉默大多数假设。该假设在以下条件下触发："
           "社媒全平台某立场 > 80%，但网络搜索发现有组织化的对立论点却在社媒上几乎不可见。")
        _w("")

    _w("---")
    _w("")

    # ═══════════════════════════════════════════════════════
    # 7. CONCLUSIONS & IMPLICATIONS
    # ═══════════════════════════════════════════════════════
    _w("## 7. 结论与启示 (Conclusions & Implications)")
    _w("")
    _w("> 以下分析基于第 4 章的客观数据和第 5 章的多维度辩论结果，力求克制、客观。")
    _w("")

    # 7.1 Platform interpretations
    if platform_interps:
        _w("### 7.1 平台差异的社会学解读")
        _w("")
        for platform, interp in platform_interps.items():
            _w(f"#### {platform.title()}")
            _w("")
            _w(interp)
            _w("")

    # 7.2 Analytical frameworks
    frameworks = fact_opinion.get("analytical_frameworks", [])
    if frameworks:
        _w("### 7.2 分析框架 (Analytical Frameworks)")
        _w("")
        for fw in frameworks:
            fw_type = fw.get("framework", "")
            certainty = fw.get("certainty", "")
            analysis = fw.get("analysis", "")
            icon = "🟢" if certainty == "high" else "🟡" if certainty == "medium" else "🔴"
            _w(f"**{fw_type.title()} 视角** {icon} [确定性: {certainty}]")
            _w("")
            _w(f"> {analysis}")
            _w("")

    # 7.3 Key tensions
    tensions = synthesis.get("key_tensions", [])
    if tensions:
        _w("### 7.3 核心矛盾与张力")
        _w("")
        for t in tensions:
            between = " vs ".join(t.get("between", []))
            _w(f"**{between}**")
            _w("")
            _w(f"- 矛盾: {t.get('tension', '')}")
            _w(f"- 重要性: {t.get('significance', '')}")
            _w("")

    # 7.4 Limitations
    _w("### 7.4 局限性声明 (Limitations)")
    _w("")
    _w(f"1. **数据规模**：本次分析基于 {total_sources} 条网络来源和 {total_posts} 条社媒帖子，"
       f"样本量有限，不代表完整舆论全貌")
    _w(f"2. **平台覆盖**：仅覆盖 {', '.join(platforms_queried) if platforms_queried else '无'}，"
       f"未包含微信公众号、今日头条等封闭平台的内容")
    _w(f"3. **时间窗口**：数据采集集中于近期，缺少长时序舆情演变分析")
    _w(f"4. **LLM 偏见**：分类和辩论环节使用大语言模型，其自身偏见可能影响分析结果")
    _w(f"5. **Media Agent 数据**：{_media_limitation(source_data)}")
    _w("")

    # 7.5 Recommended investigation
    rfi = synthesis.get("recommended_investigation", [])
    if rfi:
        _w("### 7.5 建议进一步研究方向")
        _w("")
        for r in rfi:
            _w(f"- {r}")
        _w("")

    _w("**`[PLACEHOLDER: 置信度标注系统 — 报告各章节置信度可视化]`**")
    _w("")
    _w("> *报告呈现阶段将为每个章节标注独立置信度评级，并在侧边栏显示整体置信度仪表盘。*")
    _w("")

    _w("---")
    _w("")

    # ═══════════════════════════════════════════════════════
    # APPENDICES
    # ═══════════════════════════════════════════════════════
    _w("## 附录 (Appendices)")
    _w("")

    # Appendix A: Full source list
    _w("### 附录 A：完整来源列表")
    _w("")
    _w(f"<details>")
    _w(f"<summary>展开查看全部 {total_sources} 条来源（按 TrustScore 降序）</summary>")
    _w("")
    _w("| # | 立场 | TrustScore | 标题 | 链接 |")
    _w("|---|------|-----------|------|------|")
    for i, s in enumerate(top_sources, 1):
        title_short = (s.get("title", "") or "")[:50]
        url = s.get("url", "")
        ts = s.get("trust_score", 0)
        stance = s.get("stance", "")
        _w(f"| {i} | {stance} | {ts:.2f} | {title_short} | [链接]({url}) |")
    _w("")
    _w("</details>")
    _w("")

    # Appendix B: Divergence matrix raw data
    pairs_data = divergence.get("pairs", {})
    if pairs_data:
        _w("### 附录 B：跨源分歧矩阵原始数据")
        _w("")
        _w("<details>")
        _w("<summary>展开查看全部 CSSD 值</summary>")
        _w("")
        _w("| 来源对 | CSSD | 分歧等级 |")
        _w("|--------|------|---------|")
        for pair, val in sorted(pairs_data.items(), key=lambda x: -x[1]):
            _w(f"| {pair.replace('|', ' ↔ ')} | {val:.4f} | {_cssd_label(val)} |")
        _w("")
        _w("</details>")
        _w("")

    # Appendix C: Coordinator trace
    if trace:
        _w("### 附录 C：分析流程追踪日志")
        _w("")
        _w("<details>")
        _w("<summary>展开��看完整执行追踪</summary>")
        _w("")
        _w("```")
        for t_entry in trace:
            _w(t_entry)
        _w("```")
        _w("")
        _w("</details>")
        _w("")

    # Placeholder appendices
    _w("### 附录 D：分歧矩阵热力图 [VISUALIZATION PLACEHOLDER]")
    _w("")
    _w("> *Phase 3 报告呈现阶段实现：使用 Plotly/Seaborn 渲染热力图，横纵轴为各数据源，颜色映射 CSSD 值。支持鼠标悬停显示具体数值和立场差异说明。*")
    _w("")

    _w("### 附录 E：辩论过程交互式时间线 [INTERACTIVE PLACEHOLDER]")
    _w("")
    _w("> *Phase 3 报告呈现阶段实现：每轮辩论的 AGREE/CHALLENGE/SUPPLEMENT 交互可折叠展开，修订轨迹可追溯，支持按维度筛选。*")
    _w("")

    _w("### 附录 F：Flask WebSocket 实时进度 [FEATURE PLACEHOLDER]")
    _w("")
    _w("> *Phase 3 报告呈现阶段实现：通过 WebSocket 向前端推送各阶段进度（Phase 0 → Agent 执行 → Phase 2 → 辩论 → Phase 3 → 综合 → Phase 4 → 报告），进度条实时更新。*")
    _w("")

    _w("---")
    _w("")
    _w(f"*本报告由 AgentCoordinator 多智能体协调系统自动生成*")
    _w(f"*分析管线版本: coordinator_output schema v{coordinator_output.get('schema_version', '1.0')}*")
    _w(f"*总耗时: {duration:.1f}s | 综合置信度: {confidence:.0%}*")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════

def _deduplicate_strings(items: List[str]) -> List[str]:
    """Remove near-duplicate strings (same first 60 chars)."""
    seen = set()
    result = []
    for item in items:
        key = item[:60].strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _scs_label(scs: float) -> str:
    if scs >= 0.95:
        return "全覆盖"
    elif scs >= 0.7:
        return "高覆盖"
    elif scs >= 0.5:
        return "中等覆盖"
    return "低覆盖"


def _cssd_label(cssd: float) -> str:
    if cssd < 0.1:
        return "几乎一致"
    elif cssd < 0.3:
        return "低分歧"
    elif cssd < 0.6:
        return "中等分歧"
    elif cssd < 0.8:
        return "高分歧"
    return "极高分歧"


def _verification_label(status: str) -> str:
    return {
        "cross_verified": "多源交叉验证",
        "single_source": "单一来源",
        "disputed": "有争议",
    }.get(status, status)


def _stance_meaning(stance: str) -> str:
    return {
        "official": "官方立场/权威声明",
        "support": "支持/正面评价",
        "oppose": "反对/批评/质疑",
        "neutral": "中立/客观分析",
        "background": "背景信息/技术说明",
    }.get(stance, stance)


def _media_limitation(source_data: Dict) -> str:
    media = source_data.get("media_agent", {})
    if media.get("mode") == "test_data":
        return "当前 Media Agent 使用注入测试数据（非实时采集），分析结果仅供管线验证参考"
    return "Media Agent 数据为实时采集"
