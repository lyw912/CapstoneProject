"""
Query Agent 可视化界面
展示立场感知搜索的阶段化结果：真实性、全面性、分布真实性、来源可追踪
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')

import streamlit as st
import asyncio
from datetime import datetime
import json

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QueryEngine.agent import DeepSearchAgent
from QueryEngine.utils.config import settings

st.set_page_config(
    page_title="Query Agent - 立场感知搜索",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 Query Agent - 立场感知搜索可视化")
    st.markdown("**展示多源搜索、立场分类、覆盖度检查的完整过程**")

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置")
        query = st.text_input("查询内容", value="DeepSeek发布新模型", key="query_input")
        max_iterations = st.slider("最大搜索轮次", 1, 5, 3)

        if st.button("🚀 开始搜索", type="primary", use_container_width=True):
            st.session_state.start_search = True

        st.divider()
        st.markdown("### 📊 指标说明")
        st.markdown("""
        - **SCS**: 立场覆盖度 (≥0.75)
        - **SDI**: 来源多样性 (≥0.60)
        - **TSM**: 平均可信度 (≥0.50)
        """)

    # 主界面
    if st.session_state.get('start_search'):
        execute_search(query, max_iterations)
    else:
        show_welcome()

def show_welcome():
    """欢迎页面"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### 🎯 真实性\n通过TrustScore评估来源可信度")
    with col2:
        st.success("### 📚 全面性\n多源搜索+立场矩阵覆盖")
    with col3:
        st.warning("### ⚖️ 分布真实性\n立场均衡度检查")

    st.markdown("---")
    st.markdown("### 💡 使用说明")
    st.markdown("""
    1. 在左侧输入查询内容
    2. 点击"开始搜索"按钮
    3. 实时查看每轮搜索的结果和指标
    4. 点击来源卡片查看详细信息
    """)

def execute_search(query: str, max_iterations: int):
    """执行搜索并可视化"""
    if not query.strip():
        st.error("请输入查询内容")
        return

    # 检查API密钥
    if not settings.QUERY_ENGINE_API_KEY or not settings.TAVILY_API_KEY:
        st.error("请配置API密钥：QUERY_ENGINE_API_KEY 和 TAVILY_API_KEY")
        return

    try:
        # 初始化Agent
        with st.spinner("正在初始化Query Agent..."):
            agent = DeepSearchAgent()

        # 执行搜索
        st.success("✅ Agent初始化完成")

        # 创建占位符
        progress_placeholder = st.empty()
        metrics_placeholder = st.empty()
        results_placeholder = st.container()

        # 异步执行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            agent.research_structured(query)
        )

        # 显示结果
        display_final_results(result, metrics_placeholder, results_placeholder)

    except Exception as e:
        st.error(f"搜索过程出错: {str(e)}")
        import traceback
        with st.expander("查看错误详情"):
            st.code(traceback.format_exc())

def display_final_results(output: dict, metrics_placeholder, results_placeholder):
    """显示最终结果"""

    # 顶部指标卡片
    with metrics_placeholder.container():
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            scs = output.get('coverage_score', 0)
            st.metric("立场覆盖度 (SCS)", f"{scs:.2f}",
                     delta="达标" if scs >= 0.75 else "不足",
                     delta_color="normal" if scs >= 0.75 else "inverse")

        with col2:
            st.metric("搜索轮次", output.get('search_iterations', 0))

        with col3:
            st.metric("来源总数", output.get('total_sources_kept', 0))

        with col4:
            stance_dist = output.get('stance_distribution', {})
            st.metric("立场类型", len([k for k in stance_dist.keys() if k != 'unclassified']))

        with col5:
            st.metric("知识缺口", len(output.get('knowledge_gaps', [])))

    st.divider()

    # 标签页展示
    tab1, tab2, tab3, tab4 = st.tabs(["📊 立场分布", "📰 来源列表", "💭 观点聚类", "❓ 知识缺口"])

    with tab1:
        display_stance_distribution(output)

    with tab2:
        display_sources(output)

    with tab3:
        display_opinion_clusters(output)

    with tab4:
        display_knowledge_gaps(output)

def display_stance_distribution(output: dict):
    """显示立场分布"""
    st.subheader("立场分布分析")

    stance_dist = output.get('stance_distribution', {})
    if not stance_dist:
        st.warning("暂无立场分布数据")
        return

    # 使用列展示
    cols = st.columns(len(stance_dist))

    stance_labels = {
        'support': '✅ 支持',
        'oppose': '❌ 反对',
        'official': '🏛️ 官方',
        'neutral': '⚖️ 中立',
        'background': '📚 背景',
        'unclassified': '❔ 未分类'
    }

    for i, (stance, ratio) in enumerate(stance_dist.items()):
        with cols[i]:
            label = stance_labels.get(stance, stance)
            st.metric(label, f"{ratio*100:.1f}%")
            st.progress(ratio)

def display_sources(output: dict):
    """显示来源列表"""
    st.subheader("来源详情（按可信度排序）")

    sources = output.get('sources', [])
    if not sources:
        st.warning("暂无来源数据")
        return

    # 筛选器
    col1, col2 = st.columns([1, 3])
    with col1:
        filter_stance = st.selectbox(
            "筛选立场",
            ['全部'] + list(set(s.get('stance_label', 'unclassified') for s in sources))
        )

    # 显示来源卡片
    filtered_sources = sources if filter_stance == '全部' else [
        s for s in sources if s.get('stance_label') == filter_stance
    ]

    st.caption(f"共 {len(filtered_sources)} 条来源")

    for i, source in enumerate(filtered_sources[:20]):  # 限制显示20条
        with st.expander(f"#{i+1} {source.get('title', '无标题')[:60]}..."):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**来源**: {source.get('platform', 'unknown')}")
                st.markdown(f"**URL**: [{source.get('url', '')}]({source.get('url', '')})")
                st.markdown(f"**摘要**: {source.get('snippet', '')[:200]}...")

            with col2:
                st.metric("可信度", f"{source.get('trust_score', 0):.2f}")
                stance = source.get('stance_label', 'unclassified')
                st.metric("立场", stance)
                st.caption(f"来源API: {source.get('source_api', 'unknown')}")

def display_opinion_clusters(output: dict):
    """显示观点聚类"""
    st.subheader("观点聚类分析")

    clusters = output.get('opinion_clusters', [])
    if not clusters:
        st.warning("暂无观点聚类数据")
        return

    for cluster in clusters:
        stance = cluster.get('stance', 'unknown')
        with st.container():
            st.markdown(f"### {stance.upper()} 立场")
            st.markdown(f"**核心论点**: {cluster.get('core_argument', '')}")
            st.markdown(f"**代表性引用**: \"{cluster.get('representative_quote', '')}\"")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("来源数量", cluster.get('source_count', 0))
            with col2:
                st.metric("估计占比", f"{cluster.get('estimated_proportion', 0)*100:.1f}%")

            st.divider()

def display_knowledge_gaps(output: dict):
    """显示知识缺口"""
    st.subheader("知识缺口识别")

    gaps = output.get('knowledge_gaps', [])
    if not gaps:
        st.info("✅ 未发现明显知识缺口")
        return

    st.markdown("以下是当前分析中尚未充分覆盖的信息维度：")

    for i, gap in enumerate(gaps, 1):
        st.markdown(f"{i}. {gap}")

if __name__ == "__main__":
    main()
