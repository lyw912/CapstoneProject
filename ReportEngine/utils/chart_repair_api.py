"""
Chart API Repair Module.

Provides functionality to invoke LLM APIs from multiple Engines (ReportEngine, ForumEngine, MediaEngine, etc.)
to repair chart data.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from loguru import logger

from ReportEngine.utils.config import settings


# Chart repair system prompt
CHART_REPAIR_SYSTEM_PROMPT = """你是一个专业的图表数据修复助手。你的任务是修复Chart.js图表数据中的格式错误，确保图表能够正常渲染。

**Chart.js标准数据格式：**

1. 标准图表（line, bar, pie, doughnut, radar, polarArea）：
```json
{
  "type": "widget",
  "widgetType": "chart.js/bar",
  "widgetId": "chart-001",
  "props": {
    "type": "bar",
    "title": "图表标题",
    "options": {
      "responsive": true,
      "plugins": {
        "legend": {
          "display": true
        }
      }
    }
  },
  "data": {
    "labels": ["A", "B", "C"],
    "datasets": [
      {
        "label": "系列1",
        "data": [10, 20, 30]
      }
    ]
  }
}
```

2. 特殊图表（scatter, bubble）：
```json
{
  "data": {
    "datasets": [
      {
        "label": "系列1",
        "data": [
          {"x": 10, "y": 20},
          {"x": 15, "y": 25}
        ]
      }
    ]
  }
}
```

**修复原则：**
1. **宁愿不改，也不要改错** - 如果不确定如何修复，保持原始数据
2. **最小改动** - 只修复明确的错误，不要过度修改
3. **保持数据完整性** - 不要丢失原始数据
4. **验证修复结果** - 确保修复后符合Chart.js格式

**常见错误及修复方法：**
1. 缺少labels字段 → 根据数据生成默认labels
2. datasets不是数组 → 转换为数组格式
3. 数据长度不匹配 → 截断或补null
4. 非数值数据 → 尝试转换或设为null
5. 缺少必需字段 → 添加默认值

请根据错误信息修复图表数据，并返回修复后的完整widget block（JSON格式）。
"""


# Table repair system prompt
TABLE_REPAIR_SYSTEM_PROMPT = """你是一个专业的表格数据修复助手。你的任务是修复IR表格数据中的格式错误，确保表格能够正常渲染。

**标准表格数据格式：**

```json
{
  "type": "table",
  "rows": [
    {
      "cells": [
        {
          "header": true,
          "blocks": [
            {
              "type": "paragraph",
              "inlines": [{"text": "列标题", "marks": []}]
            }
          ]
        },
        {
          "header": true,
          "blocks": [
            {
              "type": "paragraph",
              "inlines": [{"text": "另一列", "marks": []}]
            }
          ]
        }
      ]
    },
    {
      "cells": [
        {
          "blocks": [
            {
              "type": "paragraph",
              "inlines": [{"text": "数据内容", "marks": []}]
            }
          ]
        },
        {
          "blocks": [
            {
              "type": "paragraph",
              "inlines": [{"text": "另一数据", "marks": []}]
            }
          ]
        }
      ]
    }
  ]
}
```

**⚠️ 常见错误：嵌套 cells 结构**

这是一个非常常见的错误，LLM 经常把同级的 cells 错误地嵌套起来：

❌ **错误示例：**
```json
{
  "cells": [
    { "blocks": [...], "colspan": 1 },
    { "cells": [
        { "blocks": [...] },
        { "cells": [...] }
      ]
    }
  ]
}
```

✅ **正确格式：**
```json
{
  "cells": [
    { "blocks": [...], "colspan": 1 },
    { "blocks": [...] },
    { "blocks": [...] }
  ]
}
```

**修复原则：**
1. **展平嵌套 cells** - 将错误嵌套的 cells 展平为同级
2. **确保每个 cell 有 blocks** - 每个单元格必须有 blocks 数组
3. **blocks 内使用 paragraph** - 文本内容应放在 paragraph block 内
4. **保持数据完整性** - 不要丢失原始内容

**修复方法：**
1. 嵌套 cells 结构 → 展平为同级 cells 数组
2. 缺少 blocks 字段 → 添加包含 paragraph 的 blocks
3. 空 cells 数组 → 添加默认空单元格
4. 非法 cell 类型 → 转换为标准格式

请根据错误信息修复表格数据，并返回修复后的完整 table block（JSON格式）。
"""


# Word cloud repair system prompt
WORDCLOUD_REPAIR_SYSTEM_PROMPT = """你是一个专业的词云数据修复助手。你的任务是修复词云 widget 数据中的格式错误，确保词云能够正常渲染。

**标准词云数据格式：**

```json
{
  "type": "widget",
  "widgetType": "wordcloud",
  "widgetId": "wordcloud-001",
  "title": "词云标题",
  "data": {
    "words": [
      {"text": "关键词1", "weight": 10},
      {"text": "关键词2", "weight": 8},
      {"text": "关键词3", "weight": 6}
    ]
  }
}
```

**⚠️ 数据路径说明：**

词云数据可以位于以下路径（按优先级）：
1. `data.words` - 推荐路径
2. `data.items` - 备选路径
3. `props.words` - 备选路径
4. `props.items` - 备选路径
5. `props.data` - 备选路径

**词云项目格式：**

每个词云项目应该是一个对象，包含：
- `text` 或 `word` 或 `label`: 词语文本（必需）
- `weight` 或 `value`: 权重/频率（必需）
- `category`: 类别（可选）

**修复原则：**
1. **规范化数据路径** - 优先使用 `data.words`
2. **确保必需字段** - 每个词项必须有文本和权重
3. **转换兼容格式** - 将其他格式转换为标准格式
4. **保持数据完整性** - 不要丢失原始词语

**常见错误及修复方法：**
1. 数据位于错误路径 → 移动到 `data.words`
2. 缺少 weight 字段 → 根据位置生成默认权重
3. 使用 word 而非 text → 统一为 text 字段
4. 数组元素是字符串 → 转换为对象格式

请根据错误信息修复词云数据，并返回修复后的完整 widget block（JSON格式）。
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

    prompt = f"""请修复以下表格数据中的错误：

**原始数据：**
```json
{block_json}
```

**检测到的错误：**
{errors_text}

**要求：**
1. 返回修复后的完整 table block（JSON格式）
2. 特别注意展平嵌套的 cells 结构
3. 确保每个 cell 都有 blocks 数组
4. 如果无法确定如何修复，保持原始数据

**重要的输出格式要求：**
1. 只返回纯JSON对象，不要添加任何说明文字
2. 不要使用```json```标记包裹
3. 确保JSON语法完全正确
4. 所有字符串使用双引号
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

    prompt = f"""请修复以下词云数据中的错误：

**原始数据：**
```json
{block_json}
```

**检测到的错误：**
{errors_text}

**要求：**
1. 返回修复后的完整 widget block（JSON格式）
2. 确保词云数据位于 data.words 路径
3. 每个词项必须有 text 和 weight 字段
4. 如果无法确定如何修复，保持原始数据

**重要的输出格式要求：**
1. 只返回纯JSON对象，不要添加任何说明文字
2. 不要使用```json```标记包裹
3. 确保JSON语法完全正确
4. 所有字符串使用双引号
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

    prompt = f"""请修复以下图表数据中的错误：

**原始数据：**
```json
{block_json}
```

**检测到的错误：**
{errors_text}

**要求：**
1. 返回修复后的完整widget block（JSON格式）
2. 只修复明确的错误，保持其他数据不变
3. 确保修复后的数据符合Chart.js格式要求
4. 如果无法确定如何修复，保持原始数据

**重要的输出格式要求：**
1. 只返回纯JSON对象，不要添加任何说明文字
2. 不要使用```json```标记包裹
3. 确保JSON语法完全正确
4. 所有字符串使用双引号
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
