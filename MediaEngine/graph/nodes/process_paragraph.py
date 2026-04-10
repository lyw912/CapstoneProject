"""
单段落处理 — 首轮搜索+总结 + 反思循环（与原 _initial_search_and_summary / _reflection_loop 一致）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ...utils import build_search_results_from_response, format_search_results_for_prompt
from ..state import MediaAgentState

if TYPE_CHECKING:
    from ...agent import DeepSearchAgent


def process_paragraph_node(agent: DeepSearchAgent, state: MediaAgentState) -> dict:
    ps = state["pipeline_state"]
    paragraph_index = state["paragraph_index"]
    max_reflections = state.get("max_reflections", agent.config.MAX_REFLECTIONS)

    paragraph = ps.paragraphs[paragraph_index]
    logger.info(
        f"\n[LangGraph:process_paragraph] 段落 {paragraph_index + 1}/{len(ps.paragraphs)}: {paragraph.title}"
    )
    logger.info("-" * 50)

    search_input = {"title": paragraph.title, "content": paragraph.content}
    logger.info("  - 生成搜索查询...")
    search_output = agent.first_search_node.run(search_input)
    search_query = search_output["search_query"]
    search_tool = search_output.get("search_tool", "comprehensive_search")
    reasoning = search_output["reasoning"]

    logger.info(f"  - 搜索查询: {search_query}")
    logger.info(f"  - 选择的工具: {search_tool}")
    logger.info(f"  - 推理: {reasoning}")

    logger.info("  - 执行网络搜索...")
    search_kwargs = {}
    if search_tool in ["comprehensive_search", "web_search_only"]:
        search_kwargs["max_results"] = 10

    search_response = agent.execute_search_tool(search_tool, search_query, **search_kwargs)

    search_results = build_search_results_from_response(search_response, max_webpages=10, max_images=10)

    if search_results:
        n_web = sum(1 for r in search_results if r.get("result_type") == "webpage")
        n_img = sum(1 for r in search_results if r.get("result_type") == "image")
        _message = f"  - 找到 {len(search_results)} 条素材（网页 {n_web}，图片 {n_img}）"
        for j, result in enumerate(search_results, 1):
            date_info = (
                f" (发布于: {result.get('published_date', 'N/A')})"
                if result.get("published_date")
                else ""
            )
            _message += f"\n    {j}. {result['title'][:50]}...{date_info}"
        logger.info(_message)
    else:
        logger.info("  - 未找到搜索结果")

    paragraph.research.add_search_results(
        search_query,
        search_results,
        search_tool=search_tool,
        paragraph_title=paragraph.title,
    )

    logger.info("  - 生成初始总结...")
    summary_input = {
        "title": paragraph.title,
        "content": paragraph.content,
        "search_query": search_query,
        "search_results": format_search_results_for_prompt(
            search_results, agent.config.SEARCH_CONTENT_MAX_LENGTH
        ),
    }
    ps = agent.first_summary_node.mutate_state(summary_input, ps, paragraph_index)
    logger.info("  - 初始总结完成")

    paragraph = ps.paragraphs[paragraph_index]
    for reflection_i in range(max_reflections):
        logger.info(f"  - 反思 {reflection_i + 1}/{max_reflections}...")

        reflection_input = {
            "title": paragraph.title,
            "content": paragraph.content,
            "paragraph_latest_state": paragraph.research.latest_summary,
        }
        reflection_output = agent.reflection_node.run(reflection_input)
        rq = reflection_output["search_query"]
        rt = reflection_output.get("search_tool", "comprehensive_search")
        rr = reflection_output["reasoning"]

        logger.info(f"    反思查询: {rq}")
        logger.info(f"    选择的工具: {rt}")
        logger.info(f"    反思推理: {rr}")

        r_kwargs = {}
        if rt in ["comprehensive_search", "web_search_only"]:
            r_kwargs["max_results"] = 10

        r_response = agent.execute_search_tool(rt, rq, **r_kwargs)
        r_results = build_search_results_from_response(r_response, max_webpages=10, max_images=10)

        if r_results:
            n_rw = sum(1 for r in r_results if r.get("result_type") == "webpage")
            n_ri = sum(1 for r in r_results if r.get("result_type") == "image")
            _message = f"    找到 {len(r_results)} 条反思素材（网页 {n_rw}，图片 {n_ri}）"
            for j, result in enumerate(r_results, 1):
                date_info = (
                    f" (发布于: {result.get('published_date', 'N/A')})"
                    if result.get("published_date")
                    else ""
                )
                _message += f"\n      {j}. {result['title'][:50]}...{date_info}"
            logger.info(_message)
        else:
            logger.info("    未找到反思搜索结果")

        paragraph.research.add_search_results(
            rq,
            r_results,
            search_tool=rt,
            paragraph_title=paragraph.title,
        )

        reflection_summary_input = {
            "title": paragraph.title,
            "content": paragraph.content,
            "search_query": rq,
            "search_results": format_search_results_for_prompt(
                r_results, agent.config.SEARCH_CONTENT_MAX_LENGTH
            ),
            "paragraph_latest_state": paragraph.research.latest_summary,
        }
        ps = agent.reflection_summary_node.mutate_state(
            reflection_summary_input, ps, paragraph_index
        )
        paragraph = ps.paragraphs[paragraph_index]
        logger.info(f"    反思 {reflection_i + 1} 完成")

    paragraph.research.mark_completed()
    progress = (paragraph_index + 1) / len(ps.paragraphs) * 100
    logger.info(f"段落处理完成 ({progress:.1f}%)")

    trace = (
        f"[ProcessParagraph] 已完成段落 {paragraph_index + 1}/{len(ps.paragraphs)}: {paragraph.title}"
    )
    return {
        "pipeline_state": ps,
        "paragraph_index": paragraph_index + 1,
        "trace_log": [trace],
    }
