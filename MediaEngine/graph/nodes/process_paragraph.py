"""
Single paragraph processing — first round search+summary + reflection loop.

When MEDIA_PARAGRAPH_WORKERS > 1, process_all_paragraphs_node runs paragraphs in parallel.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from loguru import logger

from ...utils import build_search_results_from_response, format_search_results_for_prompt
from ..state import MediaAgentState

if TYPE_CHECKING:
    from ...agent import DeepSearchAgent


def _snippet_max_length(agent: "DeepSearchAgent") -> int:
    return int(getattr(agent.config, "SEARCH_CONTENT_MAX_LENGTH", 50000) or 50000)


def _reflection_state_max_chars(agent: "DeepSearchAgent") -> int:
    return int(getattr(agent.config, "MEDIA_REFLECTION_STATE_MAX_CHARS", 50000) or 50000)


def truncate_paragraph_state(text: str, max_chars: int) -> str:
    """Cap growing paragraph state before reflection-summary prompts."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated for prompt efficiency]"


def _paragraph_workers(agent: "DeepSearchAgent") -> int:
    workers = int(getattr(agent.config, "MEDIA_PARAGRAPH_WORKERS", 1) or 1)
    return max(1, workers)


def run_single_paragraph(
    agent: "DeepSearchAgent",
    ps,
    paragraph_index: int,
    max_reflections: int,
) -> None:
    """Process one paragraph in-place on ps.paragraphs[paragraph_index]."""
    snippet_max = _snippet_max_length(agent)
    state_max = _reflection_state_max_chars(agent)

    paragraph = ps.paragraphs[paragraph_index]
    logger.info(
        f"\n[LangGraph:process_paragraph] Paragraph {paragraph_index + 1}/{len(ps.paragraphs)}: {paragraph.title}"
    )
    logger.info("-" * 50)

    search_input = {"title": paragraph.title, "content": paragraph.content}
    logger.info("  - Generating search query...")
    search_output = agent.first_search_node.run(search_input)
    search_query = search_output["search_query"]
    search_tool = search_output.get("search_tool", "comprehensive_search")
    reasoning = search_output["reasoning"]

    logger.info(f"  - Search query: {search_query}")
    logger.info(f"  - Selected tool: {search_tool}")
    logger.info(f"  - Reasoning: {reasoning}")

    logger.info("  - Executing web search...")
    search_kwargs = {}
    if search_tool in ["comprehensive_search", "web_search_only"]:
        search_kwargs["max_results"] = 10

    search_response = agent.execute_search_tool(search_tool, search_query, **search_kwargs)
    search_results = build_search_results_from_response(search_response, max_webpages=10, max_images=10)

    if search_results:
        n_web = sum(1 for r in search_results if r.get("result_type") == "webpage")
        n_img = sum(1 for r in search_results if r.get("result_type") == "image")
        _message = f"  - Found {len(search_results)} materials (webpages {n_web}, images {n_img})"
        for j, result in enumerate(search_results, 1):
            date_info = (
                f" (Published: {result.get('published_date', 'N/A')})"
                if result.get("published_date")
                else ""
            )
            _message += f"\n    {j}. {result['title'][:50]}...{date_info}"
        logger.info(_message)
    else:
        logger.info("  - No search results found")

    paragraph.research.add_search_results(
        search_query,
        search_results,
        search_tool=search_tool,
        paragraph_title=paragraph.title,
    )

    logger.info("  - Generating initial summary...")
    summary_input = {
        "title": paragraph.title,
        "content": paragraph.content,
        "search_query": search_query,
        "search_results": format_search_results_for_prompt(search_results, snippet_max),
    }
    ps = agent.first_summary_node.mutate_state(summary_input, ps, paragraph_index)
    logger.info("  - Initial summary completed")

    paragraph = ps.paragraphs[paragraph_index]
    for reflection_i in range(max_reflections):
        logger.info(f"  - Reflection {reflection_i + 1}/{max_reflections}...")

        latest_state = truncate_paragraph_state(
            paragraph.research.latest_summary or "",
            state_max,
        )
        reflection_input = {
            "title": paragraph.title,
            "content": paragraph.content,
            "paragraph_latest_state": latest_state,
        }
        reflection_output = agent.reflection_node.run(reflection_input)
        rq = reflection_output["search_query"]
        rt = reflection_output.get("search_tool", "comprehensive_search")
        rr = reflection_output["reasoning"]

        logger.info(f"    Reflection query: {rq}")
        logger.info(f"    Selected tool: {rt}")
        logger.info(f"    Reflection reasoning: {rr}")

        r_kwargs = {}
        if rt in ["comprehensive_search", "web_search_only"]:
            r_kwargs["max_results"] = 10

        r_response = agent.execute_search_tool(rt, rq, **r_kwargs)
        r_results = build_search_results_from_response(r_response, max_webpages=10, max_images=10)

        if r_results:
            n_rw = sum(1 for r in r_results if r.get("result_type") == "webpage")
            n_ri = sum(1 for r in r_results if r.get("result_type") == "image")
            _message = f"    Found {len(r_results)} reflection materials (webpages {n_rw}, images {n_ri})"
            for j, result in enumerate(r_results, 1):
                date_info = (
                    f" (Published: {result.get('published_date', 'N/A')})"
                    if result.get("published_date")
                    else ""
                )
                _message += f"\n      {j}. {result['title'][:50]}...{date_info}"
            logger.info(_message)
        else:
            logger.info("    No reflection search results found")

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
            "search_results": format_search_results_for_prompt(r_results, snippet_max),
            "paragraph_latest_state": latest_state,
        }
        ps = agent.reflection_summary_node.mutate_state(
            reflection_summary_input, ps, paragraph_index
        )
        paragraph = ps.paragraphs[paragraph_index]
        logger.info(f"    Reflection {reflection_i + 1} completed")

    paragraph.research.mark_completed()
    progress = (paragraph_index + 1) / len(ps.paragraphs) * 100
    logger.info(f"Paragraph processing completed ({progress:.1f}%)")


def _paragraph_retry_passes(agent: "DeepSearchAgent") -> int:
    return max(0, int(getattr(agent.config, "MEDIA_PARAGRAPH_RETRY_PASSES", 1) or 0))


def _paragraph_has_summary(paragraph) -> bool:
    return bool((paragraph.research.latest_summary or "").strip())


def _run_paragraph_isolated(
    agent: "DeepSearchAgent",
    ps,
    paragraph_index: int,
    max_reflections: int,
):
    """Run one paragraph on a deep-copied state; return index + completed paragraph."""
    local_ps = copy.deepcopy(ps)
    try:
        run_single_paragraph(agent, local_ps, paragraph_index, max_reflections)
    except Exception as exc:
        partial = local_ps.paragraphs[paragraph_index]
        if _paragraph_has_summary(partial):
            logger.warning(
                f"[ProcessParagraph] Paragraph {paragraph_index + 1} failed mid-run; "
                f"keeping partial summary ({len(partial.research.latest_summary)} chars): {exc}"
            )
            return paragraph_index, copy.deepcopy(partial)
        raise
    return paragraph_index, copy.deepcopy(local_ps.paragraphs[paragraph_index])


def _merge_paragraph_result(
    agent: "DeepSearchAgent",
    ps,
    paragraph_index: int,
    max_reflections: int,
    *,
    isolated: bool,
) -> None:
    """Run one paragraph and merge into ps; salvage partial summary on failure."""
    if isolated:
        _, completed_paragraph = _run_paragraph_isolated(
            agent, ps, paragraph_index, max_reflections
        )
        ps.paragraphs[paragraph_index] = completed_paragraph
        return

    try:
        run_single_paragraph(agent, ps, paragraph_index, max_reflections)
    except Exception as exc:
        partial = ps.paragraphs[paragraph_index]
        if _paragraph_has_summary(partial):
            logger.warning(
                f"[ProcessParagraph] Paragraph {paragraph_index + 1} failed mid-run; "
                f"keeping partial summary ({len(partial.research.latest_summary)} chars): {exc}"
            )
            return
        raise


def _retry_failed_paragraphs(
    agent: "DeepSearchAgent",
    ps,
    failed: list[int],
    max_reflections: int,
    total: int,
    error_traces: list[str],
    succeeded: list[int],
) -> list[int]:
    """Sequentially retry failed paragraph indices; return those still failing."""
    retry_passes = _paragraph_retry_passes(agent)
    still_failed = list(dict.fromkeys(failed))

    for pass_num in range(1, retry_passes + 1):
        if not still_failed:
            break

        logger.info(
            f"[ProcessAllParagraphs] Retry pass {pass_num}/{retry_passes}: "
            f"{len(still_failed)} paragraph(s), sequential"
        )
        next_failed: list[int] = []
        for idx in still_failed:
            try:
                _merge_paragraph_result(
                    agent, ps, idx, max_reflections, isolated=True
                )
                if idx not in succeeded:
                    succeeded.append(idx)
                logger.info(
                    f"[ProcessAllParagraphs] Retry pass {pass_num} recovered paragraph {idx + 1}/{total}"
                )
            except Exception as exc:
                next_failed.append(idx)
                msg = (
                    f"[ProcessAllParagraphs] Retry pass {pass_num} paragraph "
                    f"{idx + 1}/{total} failed: {exc}"
                )
                logger.error(msg)
                error_traces.append(msg)

        still_failed = next_failed

    return still_failed


def process_paragraph_node(agent: "DeepSearchAgent", state: MediaAgentState) -> dict:
    """Sequential single-paragraph step (used when MEDIA_PARAGRAPH_WORKERS == 1)."""
    ps = state["pipeline_state"]
    paragraph_index = state["paragraph_index"]
    max_reflections = state.get("max_reflections", agent.config.MAX_REFLECTIONS)

    run_single_paragraph(agent, ps, paragraph_index, max_reflections)

    paragraph = ps.paragraphs[paragraph_index]
    trace = (
        f"[ProcessParagraph] Completed paragraph {paragraph_index + 1}/{len(ps.paragraphs)}: {paragraph.title}"
    )
    return {
        "pipeline_state": ps,
        "paragraph_index": paragraph_index + 1,
        "trace_log": [trace],
    }


def process_all_paragraphs_node(agent: "DeepSearchAgent", state: MediaAgentState) -> dict:
    """Process every paragraph with a thread pool when workers > 1."""
    ps = state["pipeline_state"]
    max_reflections = state.get("max_reflections", agent.config.MAX_REFLECTIONS)
    workers = min(_paragraph_workers(agent), len(ps.paragraphs))
    total = len(ps.paragraphs)
    succeeded: list[int] = []
    failed: list[int] = []
    error_traces: list[str] = []

    logger.info(
        f"[LangGraph:process_all_paragraphs] {total} paragraphs, "
        f"{max_reflections} reflections each, workers={workers}"
    )

    if workers <= 1:
        for idx in range(total):
            try:
                _merge_paragraph_result(
                    agent, ps, idx, max_reflections, isolated=False
                )
                succeeded.append(idx)
            except Exception as exc:
                failed.append(idx)
                msg = f"[ProcessAllParagraphs] Paragraph {idx + 1}/{total} failed (skipped): {exc}"
                logger.error(msg)
                error_traces.append(msg)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_paragraph_isolated, agent, ps, idx, max_reflections): idx
                for idx in range(total)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    paragraph_index, completed_paragraph = future.result()
                    ps.paragraphs[paragraph_index] = completed_paragraph
                    succeeded.append(paragraph_index)
                except Exception as exc:
                    failed.append(idx)
                    msg = f"[ProcessAllParagraphs] Paragraph {idx + 1}/{total} failed (skipped): {exc}"
                    logger.error(msg)
                    error_traces.append(msg)

    failed = _retry_failed_paragraphs(
        agent, ps, failed, max_reflections, total, error_traces, succeeded
    )
    succeeded = sorted(set(succeeded))

    if not succeeded:
        raise RuntimeError(
            f"[ProcessAllParagraphs] All {total} paragraphs failed; no partial report can be built"
        )

    if failed:
        failed_labels = ", ".join(str(i + 1) for i in sorted(failed))
        logger.warning(
            f"[ProcessAllParagraphs] Partial success: {len(succeeded)}/{total} paragraphs; "
            f"failed indices: {failed_labels}"
        )

    trace = (
        f"[ProcessAllParagraphs] Completed {len(succeeded)}/{total} paragraphs "
        f"(workers={workers}, failed={len(failed)})"
    )
    logger.info(trace)
    return {
        "pipeline_state": ps,
        "paragraph_index": total,
        "trace_log": [trace],
        "error_log": error_traces,
    }
