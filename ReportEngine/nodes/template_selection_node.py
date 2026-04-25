"""
Template selection node.

Synthesize user query, three-engine reports, forum logs and local template library,
call LLM to pick the most suitable report skeleton.
"""

import os
import json
from typing import Dict, Any, List, Optional
from loguru import logger

from .base_node import BaseNode
from ..prompts import SYSTEM_PROMPT_TEMPLATE_SELECTION
from ..utils.json_parser import RobustJSONParser, JSONParseError


class TemplateSelectionNode(BaseNode):
    """
    Template selection processing node.

    Responsible for preparing template candidate list, building prompts, parsing LLM results,
    and falling back to built-in template on failure.
    """
    
    def __init__(self, llm_client, template_dir: str = "ReportEngine/report_template"):
        """
        Initialize template selection node.

        Args:
            llm_client: LLM client instance
            template_dir: Template directory path
        """
        super().__init__(llm_client, "TemplateSelectionNode")
        self.template_dir = template_dir
        # Initialize robust JSON parser with all repair strategies enabled
        self.json_parser = RobustJSONParser(
            enable_json_repair=True,
            enable_llm_repair=False,
            max_repair_attempts=3,
        )
        
    def run(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Execute template selection.
        
        Args:
            input_data: Dictionary containing query and report content
                - query: Original query
                - reports: List of three sub-agent reports
                - forum_logs: Forum log content
                
        Returns:
            Selected template info containing name, content and selection reason
        """
        logger.info("Starting template selection...")
        
        query = input_data.get('query', '')
        reports = input_data.get('reports', [])
        forum_logs = input_data.get('forum_logs', '')
        
        # Get available templates
        available_templates = self._get_available_templates()
        
        if not available_templates:
            logger.info("No preset templates found, using built-in default template")
            return self._get_fallback_template()
        
        # Use LLM for template selection
        try:
            llm_result = self._llm_template_selection(query, reports, forum_logs, available_templates)
            if llm_result:
                return llm_result
        except Exception as e:
            logger.exception(f"LLM template selection failed: {str(e)}")
        
        # If LLM selection fails, use fallback
        return self._get_fallback_template()
    

    
    def _llm_template_selection(self, query: str, reports: List[Any], forum_logs: str, 
                              available_templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Use LLM for template selection.

        Construct template list and report summary → Call LLM → Parse JSON →
        Verify template exists and return standard structure.

        Args:
            query: User input topic keyword.
            reports: Multiple analysis engine report contents.
            forum_logs: Forum logs, may be empty.
            available_templates: Local available template inventory.

        Returns:
            dict | None: Contains template info if LLM returns valid result, None otherwise.
        """
        logger.info("Trying to use LLM for template selection...")
        
        # Build template list
        template_list = "\n".join([f"- {t['name']}: {t['description']}" for t in available_templates])
        
        # Build report content summary
        reports_summary = ""
        if reports:
            reports_summary = "\n\n=== Analysis Engine Report Content ===\n"
            for i, report in enumerate(reports, 1):
                # Get report content, support different data formats
                if isinstance(report, dict):
                    content = report.get('content', str(report))
                elif hasattr(report, 'content'):
                    content = report.content
                else:
                    content = str(report)
                
                # Truncate overly long content, keep first 1000 characters
                if len(content) > 1000:
                    content = content[:1000] + "...(content truncated)"
                
                reports_summary += f"\nReport {i} Content:\n{content}\n"
        
        # Build forum logs summary
        forum_summary = ""
        if forum_logs and forum_logs.strip():
            forum_summary = "\n\n=== Three Engine Discussion Content ===\n"
            # Truncate overly long log content, keep first 800 characters
            if len(forum_logs) > 800:
                forum_content = forum_logs[:800] + "...(discussion content truncated)"
            else:
                forum_content = forum_logs
            forum_summary += forum_content
        
        user_message = f"""Query Content: {query}

Number of Reports: {len(reports)} analysis engine reports
Forum Logs: {'Yes' if forum_logs else 'No'}
{reports_summary}{forum_summary}

Available Templates:
{template_list}

Please select the most suitable template based on the query content, reports and forum logs."""
        
        # Call LLM
        response = self.llm_client.stream_invoke_to_string(SYSTEM_PROMPT_TEMPLATE_SELECTION, user_message)

        # Check if response is empty
        if not response or not response.strip():
            logger.error("LLM returned empty response")
            return None

        logger.info(f"LLM raw response: {response}")

        # Try to parse JSON response using robust parser
        try:
            result = self.json_parser.parse(
                response,
                context_name="Template Selection",
                expected_keys=["template_name", "selection_reason"],
            )

            # Verify selected template exists
            selected_template_name = result.get('template_name', '')
            for template in available_templates:
                if template['name'] == selected_template_name or selected_template_name in template['name']:
                    logger.info(f"LLM selected template: {selected_template_name}")
                    return {
                        'template_name': template['name'],
                        'template_content': template['content'],
                        'selection_reason': result.get('selection_reason', 'LLM Intelligent Selection')
                    }

            logger.error(f"LLM selected non-existent template: {selected_template_name}")
            return None

        except JSONParseError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            # Try to extract template info from text response
            return self._extract_template_from_text(response, available_templates)
    

    def _extract_template_from_text(self, response: str, available_templates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Extract template info from text response.

        When LLM doesn't output valid JSON, try to match template name keywords for fallback.

        Args:
            response: Unstructured LLM text.
            available_templates: Available template list.

        Returns:
            dict | None: Template details if match successful, None otherwise.
        """
        logger.info("Trying to extract template info from text response")
        
        # Check if response contains template name
        for template in available_templates:
            template_name_variants = [
                template['name'],
                template['name'].replace('.md', ''),
                template['name'].replace('Template', ''),
            ]
            
            for variant in template_name_variants:
                if variant in response:
                    logger.info(f"Found template in response: {template['name']}")
                    return {
                        'template_name': template['name'],
                        'template_content': template['content'],
                        'selection_reason': 'Extracted from text response'
                    }
        
        return None
    
    def _get_available_templates(self) -> List[Dict[str, Any]]:
        """
        Get list of available templates.

        Enumerate `.md` files in template directory and read content and description fields.

        Returns:
            list[dict]: Each item contains name/path/content/description.
        """
        templates = []
        
        if not os.path.exists(self.template_dir):
            logger.error(f"Template directory does not exist: {self.template_dir}")
            return templates
        
        # Find all markdown template files
        for filename in os.listdir(self.template_dir):
            if filename.endswith('.md'):
                template_path = os.path.join(self.template_dir, filename)
                try:
                    with open(template_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    template_name = filename.replace('.md', '')
                    description = self._extract_template_description(template_name)
                    
                    templates.append({
                        'name': template_name,
                        'path': template_path,
                        'content': content,
                        'description': description
                    })
                except Exception as e:
                    logger.exception(f"Failed to read template file {filename}: {str(e)}")
        
        return templates
    
    def _extract_template_description(self, template_name: str) -> str:
        """Generate description based on template name for LLM to understand template positioning."""
        if 'CorporateBrand' in template_name or 'EnterpriseBrand' in template_name:
            return "Suitable for corporate brand reputation and image analysis"
        elif 'MarketCompetition' in template_name or 'Competitive' in template_name:
            return "Suitable for market competition landscape and competitor analysis"
        elif 'Daily' in template_name or 'Periodic' in template_name or 'Routine' in template_name:
            return "Suitable for daily monitoring and periodic reporting"
        elif 'Policy' in template_name or 'Industry' in template_name:
            return "Suitable for policy impact and industry trend analysis"
        elif 'HotTopic' in template_name or 'Social' in template_name or 'Trending' in template_name:
            return "Suitable for social hot topics and public event analysis"
        elif 'Emergency' in template_name or 'Crisis' in template_name:
            return "Suitable for emergency incidents and crisis PR"
        
        return "General purpose report template"
    

    
    def _get_fallback_template(self) -> Dict[str, Any]:
        """
        Get fallback default template (empty template, let LLM improvise).

        Returns:
            dict: Structure fields consistent with LLM return for easy replacement.
        """
        logger.info("No suitable template found, using empty template for LLM improvisation")
        
        return {
            'template_name': 'Free Form Template',
            'template_content': '',
            'selection_reason': 'No suitable preset template found, letting LLM design report structure based on content'
        }
