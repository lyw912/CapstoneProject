"""
Query Agent Visualization Interface
Displays staged results of stance-aware search: Truthfulness, Comprehensiveness, Distributional Truthfulness, Source Traceability
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')

import streamlit as st
import asyncio
from datetime import datetime
import json

# Add project path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QueryEngine.agent import DeepSearchAgent
from QueryEngine.utils.config import settings

st.set_page_config(
    page_title="Query Agent - Stance-Aware Search",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 Query Agent - Stance-Aware Search Visualization")
    st.markdown("**Demonstrating the full process of multi-source search, stance classification, and coverage checking**")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        query = st.text_input("Query Content", value="DeepSeek releases new model", key="query_input")
        max_iterations = st.slider("Max Search Iterations", 1, 5, 3)

        if st.button("🚀 Start Search", type="primary", use_container_width=True):
            st.session_state.start_search = True

        st.divider()
        st.markdown("### 📊 Metrics Description")
        st.markdown("""
        - **SCS**: Stance Coverage Score (≥0.75)
        - **SDI**: Source Diversity Index (≥0.60)
        - **TSM**: Trust Score Mean (≥0.50)
        """)

    # Main interface
    if st.session_state.get('start_search'):
        execute_search(query, max_iterations)
    else:
        show_welcome()

def show_welcome():
    """Welcome Page"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### 🎯 Truthfulness\nEvaluate source credibility via TrustScore")
    with col2:
        st.success("### 📚 Comprehensiveness\nMulti-source search + Stance matrix coverage")
    with col3:
        st.warning("### ⚖️ Distributional Truthfulness\nStance balance check")

    st.markdown("---")
    st.markdown("### 💡 Instructions")
    st.markdown("""
    1. Enter the query content on the left
    2. Click the "Start Search" button
    3. View search results and metrics for each iteration in real-time
    4. Click on source cards to view detailed information
    """)

def execute_search(query: str, max_iterations: int):
    """Execute search and visualize"""
    if not query.strip():
        st.error("Please enter query content")
        return

    # Check API keys
    if not settings.QUERY_ENGINE_API_KEY or not settings.TAVILY_API_KEY:
        st.error("Please configure API keys: QUERY_ENGINE_API_KEY and TAVILY_API_KEY")
        return

    try:
        # Initialize Agent
        with st.spinner("Initializing Query Agent..."):
            agent = DeepSearchAgent()

        # Execute search
        st.success("✅ Agent initialization completed")

        # Create placeholders
        progress_placeholder = st.empty()
        metrics_placeholder = st.empty()
        results_placeholder = st.container()

        # Asynchronous execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            agent.research_structured(query)
        )

        # Display results
        display_final_results(result, metrics_placeholder, results_placeholder)

    except Exception as e:
        st.error(f"Error during search process: {str(e)}")
        import traceback
        with st.expander("View Error Details"):
            st.code(traceback.format_exc())

def display_final_results(output: dict, metrics_placeholder, results_placeholder):
    """Display Final Results"""

    # Top metrics cards
    with metrics_placeholder.container():
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            scs = output.get('coverage_score', 0)
            st.metric("Stance Coverage (SCS)", f"{scs:.2f}",
                     delta="Passed" if scs >= 0.75 else "Insufficient",
                     delta_color="normal" if scs >= 0.75 else "inverse")

        with col2:
            st.metric("Search Iterations", output.get('search_iterations', 0))

        with col3:
            st.metric("Total Sources Kept", output.get('total_sources_kept', 0))

        with col4:
            stance_dist = output.get('stance_distribution', {})
            st.metric("Stance Types", len([k for k in stance_dist.keys() if k != 'unclassified']))

        with col5:
            st.metric("Knowledge Gaps", len(output.get('knowledge_gaps', [])))

    st.divider()

    # Tabs display
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Stance Distribution", "📰 Source List", "💭 Opinion Clusters",
        "❓ Knowledge Gaps", "🌐 Social Sentiment"
    ])

    with tab1:
        display_stance_distribution(output)

    with tab2:
        display_sources(output)

    with tab3:
        display_opinion_clusters(output)

    with tab4:
        display_knowledge_gaps(output)

    with tab5:
        display_social_sentiment(output)

def display_stance_distribution(output: dict):
    """Display Stance Distribution"""
    st.subheader("Stance Distribution Analysis")

    stance_dist = output.get('stance_distribution', {})
    if not stance_dist:
        st.warning("No stance distribution data available")
        return

    # Display using columns
    cols = st.columns(len(stance_dist))

    stance_labels = {
        'support': '✅ Support',
        'oppose': '❌ Oppose',
        'official': '🏛️ Official',
        'neutral': '⚖️ Neutral',
        'background': '📚 Background',
        'unclassified': '❔ Unclassified'
    }

    for i, (stance, ratio) in enumerate(stance_dist.items()):
        with cols[i]:
            label = stance_labels.get(stance, stance)
            st.metric(label, f"{ratio*100:.1f}%")
            st.progress(ratio)

def display_sources(output: dict):
    """Display Source List"""
    st.subheader("Source Details (Sorted by Trust Score)")

    sources = output.get('sources', [])
    if not sources:
        st.warning("No source data available")
        return

    # Filter
    col1, col2 = st.columns([1, 3])
    with col1:
        filter_stance = st.selectbox(
            "Filter Stance",
            ['All'] + list(set(s.get('stance_label', 'unclassified') for s in sources))
        )

    # Display source cards
    filtered_sources = sources if filter_stance == 'All' else [
        s for s in sources if s.get('stance_label') == filter_stance
    ]

    st.caption(f"Total {len(filtered_sources)} sources")

    for i, source in enumerate(filtered_sources[:20]):  # Limit to 20 sources
        with st.expander(f"#{i+1} {source.get('title', 'No Title')[:60]}..."):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Source**: {source.get('platform', 'unknown')}")
                st.markdown(f"**URL**: [{source.get('url', '')}]({source.get('url', '')})")
                st.markdown(f"**Snippet**: {source.get('snippet', '')[:200]}...")

            with col2:
                st.metric("Trust Score", f"{source.get('trust_score', 0):.2f}")
                stance = source.get('stance_label', 'unclassified')
                st.metric("Stance", stance)
                st.caption(f"Source API: {source.get('source_api', 'unknown')}")

def display_opinion_clusters(output: dict):
    """Display Opinion Clusters"""
    st.subheader("Opinion Cluster Analysis")

    clusters = output.get('opinion_clusters', [])
    if not clusters:
        st.warning("No opinion cluster data available")
        return

    for cluster in clusters:
        stance = cluster.get('stance', 'unknown')
        with st.container():
            st.markdown(f"### {stance.upper()} Stance")
            st.markdown(f"**Core Argument**: {cluster.get('core_argument', '')}")
            st.markdown(f"**Representative Quote**: \"{cluster.get('representative_quote', '')}\"")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Source Count", cluster.get('source_count', 0))
            with col2:
                st.metric("Estimated Proportion", f"{cluster.get('estimated_proportion', 0)*100:.1f}%")

            st.divider()

def display_knowledge_gaps(output: dict):
    """Display Knowledge Gaps"""
    st.subheader("Knowledge Gap Identification")

    gaps = output.get('knowledge_gaps', [])
    if not gaps:
        st.info("✅ No obvious knowledge gaps found")
        return

    st.markdown("The following information dimensions are not yet fully covered in the current analysis:")

    for i, gap in enumerate(gaps, 1):
        st.markdown(f"{i}. {gap}")

def display_social_sentiment(output: dict):
    """Display Social Media Sentiment Analysis from MindSpider"""
    st.subheader("Social Media Sentiment Analysis (MindSpider)")

    social = output.get("social_sentiment")
    if not social or social.get("mode") == "disabled":
        st.info(
            "Social media data not available for this query. "
            "MindSpider may not have crawled data for this topic yet."
        )
        return

    mode = social.get("mode", "disabled")
    freshness = social.get("freshness_hours", 0)

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Data Status", mode.upper(),
            delta="Fresh" if mode == "available" else "Stale",
            delta_color="normal" if mode == "available" else "inverse",
        )
    with col2:
        st.metric("Total Posts", social.get("total_posts", 0))
    with col3:
        st.metric("Total Comments", social.get("total_comments", 0))
    with col4:
        st.metric("Data Freshness", f"{freshness:.1f}h ago")

    st.divider()

    # Platforms queried
    platforms = social.get("platforms_queried", [])
    if platforms:
        st.markdown(f"**Platforms Queried**: {', '.join(platforms)}")

    # Cross-source sentiment comparison
    nsds = social.get("divergence_score", 0)
    st.metric(
        "Cross-Source Sentiment Difference Score", f"{nsds:.3f}",
        help="0 = web search and social media sentiment identical, 1 = very different",
    )
    if nsds > 0.5:
        st.warning("Notable difference detected between web search and social media sentiment distributions.")

    # Comparison summary
    summary = social.get("divergence_summary", "")
    if summary:
        st.markdown(f"**Cross-Source Comparison**: {summary}")

    st.divider()

    # Social sentiment distribution
    st.markdown("### Social Media Sentiment Distribution (Aggregate)")
    sent_dist = social.get("sentiment_distribution", {})
    if sent_dist:
        cols = st.columns(len(sent_dist))
        for i, (stance, ratio) in enumerate(sent_dist.items()):
            with cols[i]:
                st.metric(stance.capitalize(), f"{ratio*100:.1f}%")
                st.progress(ratio)

    # Content diversity / bot detection warning
    diversity = social.get("content_diversity", 1.0)
    warning = social.get("low_diversity_warning")
    if warning:
        st.error(warning)
    else:
        st.caption(f"Content diversity: {diversity:.0%} (unique posts / total)")

    # Per-platform breakdown
    per_platform = social.get("per_platform", {})
    if per_platform:
        st.markdown("### Per-Platform Sentiment Breakdown")
        for plat, info in per_platform.items():
            with st.expander(f"{plat} ({info['count']} posts)"):
                pdist = info.get("distribution", {})
                if pdist:
                    cols = st.columns(len(pdist))
                    for j, (stance, ratio) in enumerate(pdist.items()):
                        with cols[j]:
                            st.metric(stance.capitalize(), f"{ratio*100:.1f}%")

    st.divider()

    # Top social voices with provenance
    st.markdown("### Top Social Media Voices")
    voices = social.get("top_social_voices", [])
    if not voices:
        st.info("No social media voices available")
        return

    for i, voice in enumerate(voices[:10]):
        platform = voice.get("platform", "unknown")
        stance = voice.get("stance", "neutral")
        content_preview = (voice.get("content") or "")[:60]
        with st.expander(f"#{i+1} [{platform}] {stance.upper()} - {content_preview}..."):
            st.markdown(f"**Platform**: {platform}")
            st.markdown(f"**Stance**: {stance}")
            st.markdown(f"**Content**: {voice.get('content', '')}")
            url = voice.get("url", "")
            if url:
                st.markdown(f"**Source URL**: [{url}]({url})")
            pub_time = voice.get("publish_time", "")
            if pub_time:
                st.markdown(f"**Published**: {pub_time}")

    # -- Ext 1: Comment Sentiment --
    st.divider()
    cs = social.get("comment_sentiment")
    if cs and cs.get("total", 0) > 0:
        st.markdown("### Comment Sentiment Analysis")
        st.caption(f"{cs['total']} comments analyzed")

        cdist = cs.get("distribution", {})
        if cdist:
            cols = st.columns(len(cdist))
            for i, (stance, ratio) in enumerate(cdist.items()):
                with cols[i]:
                    st.metric(stance.capitalize(), f"{ratio*100:.1f}%")
                    st.progress(ratio)

        top_comments = cs.get("top_comments", [])
        if top_comments:
            st.markdown("**Top Comments (by likes)**")
            for j, c in enumerate(top_comments[:5]):
                with st.expander(
                    f"#{j+1} [{c.get('platform','')}] {c.get('stance','').upper()} "
                    f"({c.get('like_count',0)} likes)"
                ):
                    st.markdown(c.get("content", ""))
                    if c.get("publish_time"):
                        st.caption(f"Published: {c['publish_time']}")

    # -- Ext 2: Temporal Sentiment Trend --
    st.divider()
    trend = social.get("sentiment_trend")
    if trend and trend.get("buckets"):
        st.markdown("### Sentiment Trend Over Time")

        direction = trend.get("trend_direction", "stable")
        icons = {"rising": "📈", "falling": "📉", "stable": "➡️"}
        st.metric("Trend Direction", f"{icons.get(direction, '')} {direction.upper()}")

        trend_summary = trend.get("trend_summary", "")
        if trend_summary:
            st.markdown(f"**Trend Analysis**: {trend_summary}")

        buckets = trend["buckets"]
        if len(buckets) > 1:
            import pandas as pd
            rows = []
            for b in buckets:
                for stance, ratio in b.get("distribution", {}).items():
                    rows.append({"Date": b["date"], "Stance": stance, "Ratio": ratio})
            if rows:
                df = pd.DataFrame(rows)
                pivot = df.pivot(index="Date", columns="Stance", values="Ratio").fillna(0)
                st.line_chart(pivot)
        else:
            st.info(f"Only 1 day of data ({buckets[0]['date']}). "
                    "Multi-day data needed for trend visualization.")

    # -- Ext 3: Crawl Trigger Notice --
    if social.get("crawl_triggered"):
        st.divider()
        st.info(
            "BroadTopicExtraction was triggered in the background to refresh topic data. "
            "Future queries on this topic will have fresher social media data."
        )


if __name__ == "__main__":
    main()
