"""
Query Agent Visualization Interface
Displays phased results of stance-aware search: authenticity, comprehensiveness, distributional authenticity, traceable sources
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
    st.markdown("**Displaying the complete process of multi-source search, stance classification, and coverage checking**")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        query = st.text_input("Query Content", value="DeepSeek releases new model", key="query_input")
        max_iterations = st.slider("Max Search Rounds", 1, 5, 3)

        if st.button("🚀 Start Search", type="primary", use_container_width=True):
            st.session_state.start_search = True

        st.divider()
        st.markdown("### 📊 Metrics Guide")
        st.markdown("""
        - **SCS**: Stance Coverage Score (≥0.75)
        - **SDI**: Source Diversity Index (≥0.60)
        - **TSM**: Average Trust Score (≥0.50)
        """)

    # Main interface
    if st.session_state.get('start_search'):
        execute_search(query, max_iterations)
    else:
        show_welcome()

def show_welcome():
    """Welcome page"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### 🎯 Authenticity\nEvaluates source credibility via TrustScore")
    with col2:
        st.success("### 📚 Comprehensiveness\nMulti-source search + Stance matrix coverage")
    with col3:
        st.warning("### ⚖️ Distributional Authenticity\nStance balance checking")

    st.markdown("---")
    st.markdown("### 💡 Usage Instructions")
    st.markdown("""
    1. Enter query content on the left
    2. Click "Start Search" button
    3. View results and metrics for each search round in real-time
    4. Click source cards to view detailed information
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
        st.success("✅ Agent initialization complete")

        # Create placeholders
        progress_placeholder = st.empty()
        metrics_placeholder = st.empty()
        results_placeholder = st.container()

        # Async execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            agent.research_structured(query)
        )

        # Display results
        display_final_results(result, metrics_placeholder, results_placeholder)

    except Exception as e:
        st.error(f"Search process error: {str(e)}")
        import traceback
        with st.expander("View Error Details"):
            st.code(traceback.format_exc())

def display_final_results(output: dict, metrics_placeholder, results_placeholder):
    """Display final results"""

    # Top metrics cards
    with metrics_placeholder.container():
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            scs = output.get('coverage_score', 0)
            st.metric("Stance Coverage (SCS)", f"{scs:.2f}",
                     delta="Pass" if scs >= 0.75 else "Insufficient",
                     delta_color="normal" if scs >= 0.75 else "inverse")

        with col2:
            st.metric("Search Rounds", output.get('search_iterations', 0))

        with col3:
            st.metric("Total Sources", output.get('total_sources_kept', 0))

        with col4:
            stance_dist = output.get('stance_distribution', {})
            st.metric("Stance Types", len([k for k in stance_dist.keys() if k != 'unclassified']))

        with col5:
            st.metric("Knowledge Gaps", len(output.get('knowledge_gaps', [])))

    st.divider()

    # Tabs display
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stance Distribution", "📰 Source List", "💭 Opinion Clusters", "❓ Knowledge Gaps"])

    with tab1:
        display_stance_distribution(output)

    with tab2:
        display_sources(output)

    with tab3:
        display_opinion_clusters(output)

    with tab4:
        display_knowledge_gaps(output)

def display_stance_distribution(output: dict):
    """Display stance distribution"""
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
    """Display source list"""
    st.subheader("Source Details (sorted by credibility)")

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

    for i, source in enumerate(filtered_sources[:20]):  # Limit to 20 entries
        with st.expander(f"#{i+1} {source.get('title', 'No Title')[:60]}..."):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Source**: {source.get('platform', 'unknown')}")
                st.markdown(f"**URL**: [{source.get('url', '')}]({source.get('url', '')})")
                st.markdown(f"**Snippet**: {source.get('snippet', '')[:200]}...")

            with col2:
                st.metric("Credibility", f"{source.get('trust_score', 0):.2f}")
                stance = source.get('stance_label', 'unclassified')
                st.metric("Stance", stance)
                st.caption(f"Source API: {source.get('source_api', 'unknown')}")

def display_opinion_clusters(output: dict):
    """Display opinion clusters"""
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
    """Display knowledge gaps"""
    st.subheader("Knowledge Gap Identification")

    gaps = output.get('knowledge_gaps', [])
    if not gaps:
        st.info("✅ No significant knowledge gaps found")
        return

    st.markdown("The following information dimensions are not yet fully covered in the current analysis:")

    for i, gap in enumerate(gaps, 1):
        st.markdown(f"{i}. {gap}")

if __name__ == "__main__":
    main()
