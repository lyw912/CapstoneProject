"""
Chapter word budget planning node.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from loguru import logger

from ..core import TemplateSection
from ..prompts import (
    SYSTEM_PROMPT_WORD_BUDGET,
    build_word_budget_prompt,
)
from ..utils.json_parser import RobustJSONParser, JSONParseError
from .base_node import BaseNode


class WordBudgetNode(BaseNode):
    """
    Plan word count and focus for each chapter.

    Outputs total word count, global writing guidelines, and per-chapter/section target/min/max word constraints.
    """

    def __init__(self, llm_client):
        """Only record LLM client reference for initiating requests during run phase"""
        super().__init__(llm_client, "WordBudgetNode")
        # Initialize robust JSON parser with all repair strategies enabled
        self.json_parser = RobustJSONParser(
            enable_json_repair=True,
            enable_llm_repair=False,  # LLM repair can be enabled if needed
            max_repair_attempts=3,
        )

    def run(
        self,
        sections: List[TemplateSection],
        design: Dict[str, Any],
        reports: Dict[str, str],
        forum_logs: str,
        query: str,
        template_overview: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Plan chapter word counts based on design draft and all materials, giving LLM clear length targets when writing.

        Args:
            sections: Template chapter list.
            design: Design draft returned by layout node (title/toc/hero, etc.).
            reports: Three-engine report mapping.
            forum_logs: Original forum logs.
            query: User query keyword.
            template_overview: Optional template overview containing chapter metadata.

        Returns:
            dict: Chapter word budget planning result containing `totalWords`, `globalGuidelines` and per-chapter `chapters`.
        """
        # Input includes chapter skeleton and layout node output, for visual hierarchy reference when constraining length
        payload = {
            "query": query,
            "design": design,
            "sections": [section.to_dict() for section in sections],
            "templateOverview": template_overview
            or {
                "title": sections[0].title if sections else "",
                "chapters": [section.to_dict() for section in sections],
            },
            "reports": reports,
            "forumLogs": forum_logs,
        }
        user = build_word_budget_prompt(payload)
        response = self.llm_client.stream_invoke_to_string(
            SYSTEM_PROMPT_WORD_BUDGET,
            user,
            temperature=0.25,
            top_p=0.85,
        )
        plan = self._parse_response(response)
        logger.info("Chapter word budget planning generated")
        return plan

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        Convert LLM output JSON text to dictionary, raise planning exception on failure.

        Use robust JSON parser for multiple repair attempts:
        1. Clean markdown tags and thinking content
        2. Local syntax repair (bracket balance, comma completion, control char escaping, etc.)
        3. Use json_repair library for advanced repair
        4. Optional LLM-assisted repair

        Args:
            raw: LLM return value, may contain ``` wrapping, thinking content, etc.

        Returns:
            dict: Valid word budget planning JSON.

        Raises:
            ValueError: When response is empty or JSON parsing fails.
        """
        try:
            result = self.json_parser.parse(
                raw,
                context_name="Word Budget Planning",
                expected_keys=["totalWords", "globalGuidelines", "chapters"],
            )
            # Validate key field types
            if not isinstance(result.get("totalWords"), (int, float)):
                logger.warning("Word budget planning missing totalWords field or type error, using default")
                result.setdefault("totalWords", 10000)
            if not isinstance(result.get("globalGuidelines"), list):
                logger.warning("Word budget planning missing globalGuidelines field or type error, using empty list")
                result.setdefault("globalGuidelines", [])
            if not isinstance(result.get("chapters"), (list, dict)):
                logger.warning("Word budget planning missing chapters field or type error, using empty list")
                result.setdefault("chapters", [])
            return result
        except JSONParseError as exc:
            # Convert to original exception type for backward compatibility
            raise ValueError(f"Word budget planning JSON parsing failed: {exc}") from exc


__all__ = ["WordBudgetNode"]
