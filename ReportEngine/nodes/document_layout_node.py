"""
Generate title/table of contents/theme design for the entire report based on template catalog and multi-source reports.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from loguru import logger

from ..core import TemplateSection
from ..prompts import (
    SYSTEM_PROMPT_DOCUMENT_LAYOUT,
    build_document_layout_prompt,
)
from ..utils.json_parser import RobustJSONParser, JSONParseError
from .base_node import BaseNode


class DocumentLayoutNode(BaseNode):
    """
    Responsible for generating global title, table of contents and Hero design.

    Combines template slices, report summaries and forum discussions to guide the visual and structural tone of the entire book.
    """

    def __init__(self, llm_client):
        """Record LLM client and set node name for BaseNode logging"""
        super().__init__(llm_client, "DocumentLayoutNode")
        # Initialize robust JSON parser with all repair strategies enabled
        self.json_parser = RobustJSONParser(
            enable_json_repair=True,
            enable_llm_repair=False,  # LLM repair can be enabled if needed
            max_repair_attempts=3,
        )

    def run(
        self,
        sections: List[TemplateSection],
        template_markdown: str,
        reports: Dict[str, str],
        forum_logs: str,
        query: str,
        template_overview: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Synthesize template + multi-source content to generate title, TOC structure and theme palette for the entire book.

        Args:
            sections: List of chapters after template slicing.
            template_markdown: Original template text for LLM context understanding.
            reports: Three-engine content mapping.
            forum_logs: Forum discussion summary.
            query: User query keyword.
            template_overview: Pre-generated template overview, reusable to reduce prompt length.

        Returns:
            dict: Dictionary containing design info like title/subtitle/toc/hero/themeTokens.
        """
        # Feed template text, slice structure and multi-source reports to LLM to help understand hierarchy and materials
        payload = {
            "query": query,
            "template": {
                "raw": template_markdown,
                "sections": [section.to_dict() for section in sections],
            },
            "templateOverview": template_overview
            or {
                "title": sections[0].title if sections else "",
                "chapters": [section.to_dict() for section in sections],
            },
            "reports": reports,
            "forumLogs": forum_logs,
        }

        user_message = build_document_layout_prompt(payload)
        response = self.llm_client.stream_invoke_to_string(
            SYSTEM_PROMPT_DOCUMENT_LAYOUT,
            user_message,
            temperature=0.3,
            top_p=0.9,
        )
        design = self._parse_response(response)
        logger.info("Document title/TOC design generated")
        return design

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """
        Parse JSON text returned by LLM, throw user-friendly error if failed.

        Use robust JSON parser for multiple repair attempts:
        1. Clean markdown tags and thinking content
        2. Local syntax repair (bracket balance, comma completion, control char escaping, etc.)
        3. Use json_repair library for advanced repair
        4. Optional LLM-assisted repair

        Args:
            raw: LLM raw response string, allowing ``` wrapping, thinking content, etc.

        Returns:
            dict: Structured design draft.

        Raises:
            ValueError: When response is empty or JSON parsing fails.
        """
        try:
            result = self.json_parser.parse(
                raw,
                context_name="Document Design",
                # TOC field renamed to tocPlan, following latest Schema validation
                expected_keys=["title", "tocPlan", "hero"],
            )
            # Validate key field types
            if not isinstance(result.get("title"), str):
                logger.warning("Document design missing title field or type error, using default")
                result.setdefault("title", "Untitled Report")

            # Process tocPlan field
            toc_plan = result.get("tocPlan", [])
            if not isinstance(toc_plan, list):
                logger.warning("Document design missing tocPlan field or type error, using empty list")
                result["tocPlan"] = []
            else:
                # Clean description fields in tocPlan
                result["tocPlan"] = self._clean_toc_plan_descriptions(toc_plan)

            if not isinstance(result.get("hero"), dict):
                logger.warning("Document design missing hero field or type error, using empty object")
                result.setdefault("hero", {})

            return result
        except JSONParseError as exc:
            # Convert to original exception type for backward compatibility
            raise ValueError(f"Document design JSON parsing failed: {exc}") from exc

    def _clean_toc_plan_descriptions(self, toc_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean description fields in each tocPlan entry, remove possible JSON fragments.

        Args:
            toc_plan: Original table of contents plan list.

        Returns:
            List[Dict[str, Any]]: Cleaned table of contents plan list.
        """
        import re

        def clean_text(text: Any) -> str:
            """Clean JSON fragments from text"""
            if not text or not isinstance(text, str):
                return ""

            cleaned = text

            # Remove incomplete JSON objects starting with comma+whitespace+{
            cleaned = re.sub(r',\s*\{[^}]*$', '', cleaned)

            # Remove incomplete JSON arrays starting with comma+whitespace+[
            cleaned = re.sub(r',\s*\[[^\]]*$', '', cleaned)

            # Remove isolated { with subsequent content (if no matching })
            open_brace_pos = cleaned.rfind('{')
            if open_brace_pos != -1:
                close_brace_pos = cleaned.rfind('}')
                if close_brace_pos < open_brace_pos:
                    cleaned = cleaned[:open_brace_pos].rstrip(',，、 \t\n')

            # Remove isolated [ with subsequent content (if no matching ])
            open_bracket_pos = cleaned.rfind('[')
            if open_bracket_pos != -1:
                close_bracket_pos = cleaned.rfind(']')
                if close_bracket_pos < open_bracket_pos:
                    cleaned = cleaned[:open_bracket_pos].rstrip(',，、 \t\n')

            # Remove fragments that look like JSON key-value pairs
            cleaned = re.sub(r',?\s*"[^"]+"\s*:\s*"[^"]*$', '', cleaned)
            cleaned = re.sub(r',?\s*"[^"]+"\s*:\s*[^,}\]]*$', '', cleaned)

            # Clean trailing commas and whitespace
            cleaned = cleaned.rstrip(',，、 \t\n')

            return cleaned.strip()

        cleaned_plan = []
        for entry in toc_plan:
            if not isinstance(entry, dict):
                continue

            # Clean description field
            if "description" in entry:
                original_desc = entry["description"]
                cleaned_desc = clean_text(original_desc)

                if cleaned_desc != original_desc:
                    logger.warning(
                        f"Cleaned JSON fragments from description field of TOC entry '{entry.get('display', 'unknown')}':\n"
                        f"  Original: {original_desc[:100]}...\n"
                        f"  Cleaned: {cleaned_desc[:100]}..."
                    )
                    entry["description"] = cleaned_desc

            cleaned_plan.append(entry)

        return cleaned_plan


__all__ = ["DocumentLayoutNode"]
