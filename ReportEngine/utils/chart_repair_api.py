"""
Chart API Repair Module.

Provides functionality to invoke LLM APIs from multiple Engines (ReportEngine, ForumEngine, MediaEngine, etc.)
to repair chart data.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from loguru import logger

from ReportEngine.prompts.prompts import ENGLISH_REPORT_LANGUAGE_RULE
from ReportEngine.utils.config import settings


# Chart repair system prompt
CHART_REPAIR_SYSTEM_PROMPT = f"""You are a professional Chart.js data repair assistant. Fix format errors in chart widget blocks so charts render correctly.

Language rule: {ENGLISH_REPORT_LANGUAGE_RULE}
All human-readable strings you add or rewrite in titles, labels, legends, and dataset names must be English.

**Chart.js standard data format:**

1. Standard charts (line, bar, pie, doughnut, radar, polarArea):
```json
{{
  "type": "widget",
  "widgetType": "chart.js/bar",
  "widgetId": "chart-001",
  "props": {{
    "type": "bar",
    "title": "Chart Title",
    "options": {{
      "responsive": true,
      "plugins": {{
        "legend": {{
          "display": true
        }}
      }}
    }}
  }},
  "data": {{
    "labels": ["A", "B", "C"],
    "datasets": [
      {{
        "label": "Series 1",
        "data": [10, 20, 30]
      }}
    ]
  }}
}}
```

2. Special charts (scatter, bubble):
```json
{{
  "data": {{
    "datasets": [
      {{
        "label": "Series 1",
        "data": [
          {{"x": 10, "y": 20}},
          {{"x": 15, "y": 25}}
        ]
      }}
    ]
  }}
}}
```

**Repair principles:**
1. **When unsure, do not guess** — keep original data if the fix is ambiguous
2. **Minimal changes** — fix only clear errors
3. **Preserve data** — do not drop original values
4. **Validate** — output must conform to Chart.js expectations

**Common errors and fixes:**
1. Missing `labels` → synthesize sensible default labels
2. `datasets` is not an array → wrap as an array
3. Length mismatch → truncate or pad with null
4. Non-numeric values → coerce or use null
5. Missing required fields → add safe defaults

Return the repaired full widget block as JSON only.
"""


# Table repair system prompt
TABLE_REPAIR_SYSTEM_PROMPT = f"""You are a professional IR table repair assistant. Fix format errors in table blocks so tables render correctly.

Language rule: {ENGLISH_REPORT_LANGUAGE_RULE}
All cell text you add or rewrite must be English.

**Standard table block format:**

```json
{{
  "type": "table",
  "rows": [
    {{
      "cells": [
        {{
          "header": true,
          "blocks": [
            {{
              "type": "paragraph",
              "inlines": [{{"text": "Column A", "marks": []}}]
            }}
          ]
        }},
        {{
          "header": true,
          "blocks": [
            {{
              "type": "paragraph",
              "inlines": [{{"text": "Column B", "marks": []}}]
            }}
          ]
        }}
      ]
    }},
    {{
      "cells": [
        {{
          "blocks": [
            {{
              "type": "paragraph",
              "inlines": [{{"text": "Value 1", "marks": []}}]
            }}
          ]
        }},
        {{
          "blocks": [
            {{
              "type": "paragraph",
              "inlines": [{{"text": "Value 2", "marks": []}}]
            }}
          ]
        }}
      ]
    }}
  ]
}}
```

**Common error: nested `cells`**

LLMs often nest sibling cells incorrectly:

**Wrong:**
```json
{{
  "cells": [
    {{ "blocks": [...], "colspan": 1 }},
    {{ "cells": [
        {{ "blocks": [...] }},
        {{ "cells": [...] }}
      ]
    }}
  ]
}}
```

**Correct:**
```json
{{
  "cells": [
    {{ "blocks": [...], "colspan": 1 }},
    {{ "blocks": [...] }},
    {{ "blocks": [...] }}
  ]
}}
```

**Repair principles:**
1. Flatten nested `cells` into a single sibling array
2. Every cell must have a `blocks` array
3. Put text in `paragraph` blocks inside `blocks`
4. Preserve original content where possible

**Fixes:**
1. Nested `cells` → flatten to siblings
2. Missing `blocks` → add a paragraph block
3. Empty `cells` → add a default empty cell
4. Invalid cell shape → convert to standard format

Return the repaired full table block as JSON only.
"""


# Word cloud repair system prompt
WORDCLOUD_REPAIR_SYSTEM_PROMPT = f"""You are a professional word-cloud widget repair assistant. Fix format errors so word clouds render correctly.

Language rule: {ENGLISH_REPORT_LANGUAGE_RULE}
Translate any Chinese token text into English when you must rewrite labels; preserve meaning.

**Standard word-cloud format:**

```json
{{
  "type": "widget",
  "widgetType": "wordcloud",
  "widgetId": "wordcloud-001",
  "title": "Word Cloud Title",
  "data": {{
    "words": [
      {{"text": "keyword_one", "weight": 10}},
      {{"text": "keyword_two", "weight": 8}},
      {{"text": "keyword_three", "weight": 6}}
    ]
  }}
}}
```

**Data path priority:**
1. `data.words` (preferred)
2. `data.items`
3. `props.words`
4. `props.items`
5. `props.data`

**Word item shape:**
- `text` or `word` or `label`: token text (required)
- `weight` or `value`: frequency (required)
- `category`: optional

**Repair principles:**
1. Normalize to `data.words` when possible
2. Every item needs text and weight
3. Convert legacy shapes to the standard object format
4. Preserve original tokens when valid

**Common fixes:**
1. Wrong path → move to `data.words`
2. Missing `weight` → assign a descending default weight
3. `word` field only → normalize to `text`
4. Bare strings in array → wrap as objects

Return the repaired full widget block as JSON only.
"""


def build_table_repair_prompt(
    table_block: Dict[str, Any],
    validation_errors: List[str]
) -> str:
    """
    Build table repair prompt.

    Args:
        table_block: Original table block
        validation_errors: List of validation errors

    Returns:
        str: The prompt
    """
    block_json = json.dumps(table_block, ensure_ascii=False, indent=2)
    errors_text = "\n".join(f"- {error}" for error in validation_errors)

    prompt = f"""Repair the table block below.

Language rule: {ENGLISH_REPORT_LANGUAGE_RULE}

**Original data:**
```json
{block_json}
```

**Validation errors:**
{errors_text}

**Requirements:**
1. Return the full repaired `table` block as JSON
2. Flatten any nested `cells` structures
3. Ensure every cell has a `blocks` array
4. If unsure, keep the original data

**Output format:**
1. Return a single JSON object only — no prose
2. Do not wrap output in ```json``` fences
3. Valid JSON syntax; double-quoted strings
"""
    return prompt


def build_wordcloud_repair_prompt(
    widget_block: Dict[str, Any],
    validation_errors: List[str]
) -> str:
    """
    Build word cloud repair prompt.

    Args:
        widget_block: Original word cloud widget block
        validation_errors: List of validation errors

    Returns:
        str: The prompt
    """
    block_json = json.dumps(widget_block, ensure_ascii=False, indent=2)
    errors_text = "\n".join(f"- {error}" for error in validation_errors)

    prompt = f"""Repair the word-cloud widget block below.

Language rule: {ENGLISH_REPORT_LANGUAGE_RULE}

**Original data:**
```json
{block_json}
```

**Validation errors:**
{errors_text}

**Requirements:**
1. Return the full repaired widget block as JSON
2. Prefer `data.words` for word items
3. Each item must have `text` and `weight`
4. If unsure, keep the original data

**Output format:**
1. Return a single JSON object only — no prose
2. Do not wrap output in ```json``` fences
3. Valid JSON syntax; double-quoted strings
"""
    return prompt


def build_chart_repair_prompt(
    widget_block: Dict[str, Any],
    validation_errors: List[str]
) -> str:
    """
    Build chart repair prompt.

    Args:
        widget_block: Original widget block
        validation_errors: List of validation errors

    Returns:
        str: The prompt
    """
    block_json = json.dumps(widget_block, ensure_ascii=False, indent=2)
    errors_text = "\n".join(f"- {error}" for error in validation_errors)

    prompt = f"""Repair the chart widget block below.

Language rule: {ENGLISH_REPORT_LANGUAGE_RULE}

**Original data:**
```json
{block_json}
```

**Validation errors:**
{errors_text}

**Requirements:**
1. Return the full repaired widget block as JSON
2. Fix only clear errors; leave other fields unchanged
3. Output must satisfy Chart.js data expectations
4. If unsure, keep the original data

**Output format:**
1. Return a single JSON object only — no prose
2. Do not wrap output in ```json``` fences
3. Valid JSON syntax; double-quoted strings
"""
    return prompt


def create_llm_repair_functions() -> List:
    """
    Create LLM repair function list.

    Returns repair functions for multiple Engines:
    1. ReportEngine
    2. ForumEngine (via ForumHost)
    3. MediaEngine

    Returns:
        List[Callable]: List of repair functions
    """
    repair_functions = []

    # 1. ReportEngine repair function
    if settings.REPORT_ENGINE_API_KEY and settings.REPORT_ENGINE_BASE_URL:
        def repair_with_report_engine(widget_block: Dict[str, Any], errors: List[str]) -> Optional[Dict[str, Any]]:
            """Use ReportEngine LLM to repair chart"""
            try:
                from ReportEngine.llms import LLMClient

                client = LLMClient(
                    api_key=settings.REPORT_ENGINE_API_KEY,
                    base_url=settings.REPORT_ENGINE_BASE_URL,
                    model_name=settings.REPORT_ENGINE_MODEL_NAME or "gpt-4",
                )

                prompt = build_chart_repair_prompt(widget_block, errors)
                response = client.invoke(
                    CHART_REPAIR_SYSTEM_PROMPT,
                    prompt,
                    temperature=0.0,
                    top_p=0.05
                )

                if not response:
                    return None

                # Parse response
                repaired = json.loads(response)
                return repaired

            except Exception as e:
                logger.exception(f"ReportEngine chart repair failed: {e}")
                return None

        repair_functions.append(repair_with_report_engine)
        logger.debug("Added ReportEngine chart repair function")

    # 2. ForumEngine repair function
    if settings.FORUM_HOST_API_KEY and settings.FORUM_HOST_BASE_URL:
        def repair_with_forum_engine(widget_block: Dict[str, Any], errors: List[str]) -> Optional[Dict[str, Any]]:
            """Use ForumEngine LLM to repair chart"""
            try:
                from ReportEngine.llms import LLMClient

                client = LLMClient(
                    api_key=settings.FORUM_HOST_API_KEY,
                    base_url=settings.FORUM_HOST_BASE_URL,
                    model_name=settings.FORUM_HOST_MODEL_NAME or "gpt-4",
                )

                prompt = build_chart_repair_prompt(widget_block, errors)
                response = client.invoke(
                    CHART_REPAIR_SYSTEM_PROMPT,
                    prompt,
                    temperature=0.0,
                    top_p=0.05
                )

                if not response:
                    return None

                repaired = json.loads(response)
                return repaired

            except Exception as e:
                logger.exception(f"ForumEngine chart repair failed: {e}")
                return None

        repair_functions.append(repair_with_forum_engine)
        logger.debug("Added ForumEngine chart repair function")

    # 3. MediaEngine repair function
    if settings.MEDIA_ENGINE_API_KEY and settings.MEDIA_ENGINE_BASE_URL:
        def repair_with_media_engine(widget_block: Dict[str, Any], errors: List[str]) -> Optional[Dict[str, Any]]:
            """Use MediaEngine LLM to repair chart"""
            try:
                from ReportEngine.llms import LLMClient

                client = LLMClient(
                    api_key=settings.MEDIA_ENGINE_API_KEY,
                    base_url=settings.MEDIA_ENGINE_BASE_URL,
                    model_name=settings.MEDIA_ENGINE_MODEL_NAME or "gpt-4",
                )

                prompt = build_chart_repair_prompt(widget_block, errors)
                response = client.invoke(
                    CHART_REPAIR_SYSTEM_PROMPT,
                    prompt,
                    temperature=0.0,
                    top_p=0.05
                )

                if not response:
                    return None

                repaired = json.loads(response)
                return repaired

            except Exception as e:
                logger.exception(f"MediaEngine chart repair failed: {e}")
                return None

        repair_functions.append(repair_with_media_engine)
        logger.debug("Added MediaEngine chart repair function")

    if not repair_functions:
        logger.warning("No Engine API configured, chart API repair feature will be unavailable")
    else:
        logger.info(f"Chart API repair feature enabled, {len(repair_functions)} Engine(s) available")

    return repair_functions


def create_table_repair_functions() -> List:
    """
    Create table LLM repair function list.

    Uses the same Engine configuration as chart repair.

    Returns:
        List[Callable]: List of repair functions
    """
    repair_functions = []

    # Use ReportEngine to repair tables
    if settings.REPORT_ENGINE_API_KEY and settings.REPORT_ENGINE_BASE_URL:
        def repair_table_with_report_engine(table_block: Dict[str, Any], errors: List[str]) -> Optional[Dict[str, Any]]:
            """Use ReportEngine LLM to repair table"""
            try:
                from ReportEngine.llms import LLMClient

                client = LLMClient(
                    api_key=settings.REPORT_ENGINE_API_KEY,
                    base_url=settings.REPORT_ENGINE_BASE_URL,
                    model_name=settings.REPORT_ENGINE_MODEL_NAME or "gpt-4",
                )

                prompt = build_table_repair_prompt(table_block, errors)
                response = client.invoke(
                    TABLE_REPAIR_SYSTEM_PROMPT,
                    prompt,
                    temperature=0.0,
                    top_p=0.05
                )

                if not response:
                    return None

                # Parse response
                repaired = json.loads(response)
                return repaired

            except Exception as e:
                logger.exception(f"ReportEngine table repair failed: {e}")
                return None

        repair_functions.append(repair_table_with_report_engine)
        logger.debug("Added ReportEngine table repair function")

    if not repair_functions:
        logger.warning("No Engine API configured, table API repair feature will be unavailable")
    else:
        logger.info(f"Table API repair feature enabled, {len(repair_functions)} Engine(s) available")

    return repair_functions


def create_wordcloud_repair_functions() -> List:
    """
    Create word cloud LLM repair function list.

    Uses the same Engine configuration as chart repair.

    Returns:
        List[Callable]: List of repair functions
    """
    repair_functions = []

    # Use ReportEngine to repair word clouds
    if settings.REPORT_ENGINE_API_KEY and settings.REPORT_ENGINE_BASE_URL:
        def repair_wordcloud_with_report_engine(widget_block: Dict[str, Any], errors: List[str]) -> Optional[Dict[str, Any]]:
            """Use ReportEngine LLM to repair word cloud"""
            try:
                from ReportEngine.llms import LLMClient

                client = LLMClient(
                    api_key=settings.REPORT_ENGINE_API_KEY,
                    base_url=settings.REPORT_ENGINE_BASE_URL,
                    model_name=settings.REPORT_ENGINE_MODEL_NAME or "gpt-4",
                )

                prompt = build_wordcloud_repair_prompt(widget_block, errors)
                response = client.invoke(
                    WORDCLOUD_REPAIR_SYSTEM_PROMPT,
                    prompt,
                    temperature=0.0,
                    top_p=0.05
                )

                if not response:
                    return None

                # Parse response
                repaired = json.loads(response)
                return repaired

            except Exception as e:
                logger.exception(f"ReportEngine word cloud repair failed: {e}")
                return None

        repair_functions.append(repair_wordcloud_with_report_engine)
        logger.debug("Added ReportEngine word cloud repair function")

    if not repair_functions:
        logger.warning("No Engine API configured, word cloud API repair feature will be unavailable")
    else:
        logger.info(f"Word cloud API repair feature enabled, {len(repair_functions)} Engine(s) available")

    return repair_functions
